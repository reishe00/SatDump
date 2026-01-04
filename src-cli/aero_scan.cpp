#include "aero_scan.h"
#include "logger.h"
#include "init.h"
#include "common/cli_utils.h"
#include "common/dsp_source_sink/dsp_sample_source.h"
#include "common/dsp/path/splitter_vfo.h"
#include "common/dsp/fft/fft_pan.h"
#include "core/live_pipeline.h"
#include <filesystem>
#include <atomic>
#include <mutex>
#include <map>
#include <set>
#include <optional>
#include <chrono>
#include <csignal>
#include <cmath>
#include <vector>
#include "libs/ctpl/ctpl_stl.h"

namespace
{
    struct Candidate
    {
        double offset_hz;
        int hits = 0;
        std::chrono::steady_clock::time_point last_seen;
    };

    struct ActiveVFO
    {
        std::string id;
        double offset_hz;
        std::shared_ptr<satdump::LivePipeline> pipeline;
        std::chrono::steady_clock::time_point last_seen;
    };

    bool aero_scan_should_exit = false;
    void sig_handler_aero_scan(int signo)
    {
        if (signo == SIGINT || signo == SIGTERM)
            aero_scan_should_exit = true;
    }
}

int main_aero_scan(int argc, char *argv[])
{
    if (argc < 4)
    {
        logger->error("Usage : " + std::string(argv[0]) + " aero_scan [output_directory] --frequency <center_hz> --samplerate <rate> --source <id> [options]");
        logger->error("Options:");
        logger->error(" --snr_margin_db <dB> (default 3)");
        logger->error(" --corr_threshold <value> (default 50)");
        logger->error(" --max_vfos <n> (default 8)");
        logger->error(" --confirm_hits <n> (default 2)");
        logger->error(" --drop_miss_ms <ms> (default 2000)");
        logger->error(" --min_spacing_hz <Hz> (default 10000)");
        logger->error("Standard source flags apply (gain, bias, etc.).");
        return 1;
    }

    std::string output_dir = argv[2];

    nlohmann::json parameters = parse_common_flags(argc - 3, &argv[3], {{"source_id", typeid(std::string)}});

    uint64_t samplerate;
    uint64_t frequency;
    std::string handler_id;
    std::string hdl_dev_id;

    try
    {
        samplerate = parameters["samplerate"].get<uint64_t>();
        frequency = parameters["frequency"].get<uint64_t>();
        handler_id = parameters["source"].get<std::string>();
        if (parameters.contains("source_id"))
            hdl_dev_id = parameters["source_id"].get<std::string>();
    }
    catch (std::exception &e)
    {
        logger->error("Error parsing arguments! %s", e.what());
        return 1;
    }

    float snr_margin_db = parameters.value("snr_margin_db", 3.0f);
    float corr_threshold = parameters.value("corr_threshold", 50.0f);
    int max_vfos = parameters.value("max_vfos", 8);
    int confirm_hits = parameters.value("confirm_hits", 2);
    int drop_miss_ms = parameters.value("drop_miss_ms", 2000);
    int min_spacing_hz = parameters.value("min_spacing_hz", 10000);
    (void)corr_threshold;

    satdump::tle_file_override = parameters.contains("tle_override") ? parameters["tle_override"].get<std::string>() : "";
    satdump::initSatdump();
    completeLoggerInit();

    parameters["baseband_format"] = "cf32";
    parameters["buffer_size"] = dsp::STREAM_BUFFER_SIZE;
    parameters["start_timestamp"] = (double)time(0);

    dsp::registerAllSources();
    std::vector<dsp::SourceDescriptor> source_tr = dsp::getAllAvailableSources();
    dsp::SourceDescriptor selected_src;

    bool src_found = false;
    for (dsp::SourceDescriptor src : source_tr)
    {
        if (handler_id == src.source_type)
        {
            if (parameters.contains("source_id"))
            {
                if (hdl_dev_id == src.unique_id)
                {
                    selected_src = src;
                    src_found = true;
                }
            }
            else
            {
                selected_src = src;
                src_found = true;
            }
        }
    }

    if (!src_found)
    {
        logger->error("Could not find a handler for source type : %s!", handler_id.c_str());
        return 1;
    }

    std::shared_ptr<dsp::DSPSampleSource> source_ptr = dsp::getSourceFromDescriptor(selected_src);
    source_ptr->open();
    source_ptr->set_frequency(frequency);
    source_ptr->set_samplerate(samplerate);
    source_ptr->set_settings(parameters);

    std::optional<satdump::Pipeline> pipeline_opt = satdump::getPipelineFromName("inmarsat_aero_84");
    if (!pipeline_opt.has_value())
    {
        logger->error("Pipeline inmarsat_aero_84 not found.");
        return 1;
    }

    if (!std::filesystem::exists(output_dir))
        std::filesystem::create_directories(output_dir);

    std::mutex state_mutex;
    std::map<int, Candidate> candidates; // keyed by rounded offset bin
    std::map<std::string, ActiveVFO> active_vfos;
    std::atomic<int> vfo_counter{0};

    std::unique_ptr<dsp::VFOSplitterBlock> splitter_vfo;
    std::unique_ptr<dsp::FFTPanBlock> fft;
    ctpl::thread_pool live_thread_pool(64);

    try
    {
        source_ptr->start();
        splitter_vfo = std::make_unique<dsp::VFOSplitterBlock>(source_ptr->output_stream);
        splitter_vfo->set_main_enabled(false);
        splitter_vfo->start();

        fft = std::make_unique<dsp::FFTPanBlock>(source_ptr->output_stream);
        int fft_size = parameters.value("fft_size", 2048);
        int fft_rate = parameters.value("fft_rate", 50);
        fft->set_fft_settings(fft_size, samplerate, fft_rate);
        fft->avg_num = parameters.value("fft_avgn", 3.0f);

        fft->on_fft = [&, fft_size, samplerate](float *fft_vals)
        {
            std::vector<float> values(fft_size);
            for (int i = 0; i < fft_size; i++)
                values[i] = fft_vals[i];

            std::vector<float> sorted = values;
            std::nth_element(sorted.begin(), sorted.begin() + fft_size / 2, sorted.end());
            float noise_floor = sorted[fft_size / 2];
            if (noise_floor <= 0)
                return;

            float threshold = noise_floor * powf(10.0f, snr_margin_db / 10.0f);
            double bin_hz = samplerate / fft_size;
            auto now = std::chrono::steady_clock::now();

            std::lock_guard<std::mutex> lk(state_mutex);

            std::set<int> seen_bins;

            for (int i = 0; i < fft_size; i++)
            {
                if (values[i] < threshold)
                    continue;

                double offset = (double(i) - (fft_size / 2)) * bin_hz;
                if (std::abs(offset) > (samplerate / 2))
                    continue;
                int rounded_bin = int(std::round(offset / 100.0) * 100.0);

                if (seen_bins.count(rounded_bin))
                    continue;
                seen_bins.insert(rounded_bin);

                bool near_active = false;
                for (auto &kv : active_vfos)
                {
                    if (fabs(kv.second.offset_hz - offset) < min_spacing_hz)
                    {
                        kv.second.last_seen = now;
                        near_active = true;
                        break;
                    }
                }
                if (near_active)
                    continue;

                auto &cand = candidates[rounded_bin];
                cand.offset_hz = offset;
                cand.hits += 1;
                cand.last_seen = now;
            }

            // expire stale candidates
            for (auto it = candidates.begin(); it != candidates.end();)
            {
                if (std::chrono::duration_cast<std::chrono::milliseconds>(now - it->second.last_seen).count() > drop_miss_ms)
                    it = candidates.erase(it);
                else
                    ++it;
            }
        };

        fft->start();
    }
    catch (std::exception &e)
    {
        logger->error("Fatal error starting scanner: %s", e.what());
        return 1;
    }

    signal(SIGINT, sig_handler_aero_scan);
    signal(SIGTERM, sig_handler_aero_scan);

    auto last_stats_update = std::chrono::steady_clock::now();
    while (!aero_scan_should_exit)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        auto now = std::chrono::steady_clock::now();

        std::vector<std::pair<std::string, double>> spawn_reqs;
        std::vector<std::string> stop_reqs;

        {
            std::unique_lock<std::mutex> lk(state_mutex);

            for (auto &kv : candidates)
            {
                if (kv.second.hits < confirm_hits)
                    continue;
                if ((int)active_vfos.size() + (int)spawn_reqs.size() >= max_vfos)
                    continue;

                double offset = kv.second.offset_hz;
                if (std::abs(offset) > (samplerate / 2))
                    continue;

                std::string vfo_id = "aero_" + std::to_string(vfo_counter.fetch_add(1));
                spawn_reqs.emplace_back(vfo_id, offset);
            }

            for (auto &req : spawn_reqs)
            {
                int rounded = int(std::round(req.second / 100.0) * 100.0);
                candidates.erase(rounded);
            }

            for (auto it = active_vfos.begin(); it != active_vfos.end(); ++it)
            {
                if (std::chrono::duration_cast<std::chrono::milliseconds>(now - it->second.last_seen).count() > drop_miss_ms)
                    stop_reqs.push_back(it->first);
            }
        }

        for (auto &req : spawn_reqs)
        {
            double offset = req.second;
            std::string vfo_id = req.first;
            double final_shift = -offset;

            splitter_vfo->add_vfo(vfo_id, samplerate, final_shift);

            nlohmann::json vparams;
            vparams["baseband_format"] = "cf32";
            vparams["buffer_size"] = dsp::STREAM_BUFFER_SIZE;
            vparams["start_timestamp"] = (double)time(0);
            vparams["samplerate"] = samplerate;
            vparams["inmarsat_downlink"] = frequency + offset;
            vparams["is_scan_vfo"] = true;

            std::string path = output_dir + "/" + vfo_id;
            if (!std::filesystem::exists(path))
                std::filesystem::create_directories(path);

            auto live_pipeline = std::make_shared<satdump::LivePipeline>(pipeline_opt.value(), vparams, path);
            bool server_mode = false;
            live_pipeline->start(splitter_vfo->get_vfo_output(vfo_id), live_thread_pool, server_mode);
            splitter_vfo->set_vfo_enabled(vfo_id, true);

            std::unique_lock<std::mutex> lk(state_mutex);
            ActiveVFO info{vfo_id, offset, live_pipeline, now};
            active_vfos.emplace(vfo_id, info);
            logger->info("Spawned VFO %s at offset %.1f Hz", vfo_id.c_str(), offset);
        }

        for (auto &id : stop_reqs)
        {
            splitter_vfo->set_vfo_enabled(id, false);
            splitter_vfo->del_vfo(id);
            {
                std::unique_lock<std::mutex> lk(state_mutex);
                auto it = active_vfos.find(id);
                if (it != active_vfos.end())
                {
                    it->second.pipeline->stop();
                    active_vfos.erase(it);
                }
            }
            logger->info("Stopped VFO %s (no activity)", id.c_str());
        }

        // Periodic stats update for logging/debug
        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_stats_update).count() >= 5)
        {
            logger->info("Active VFOs: %zu, Pending candidates: %zu", active_vfos.size(), candidates.size());
            last_stats_update = now;
        }
    }

    logger->warn("Stopping scanner...");
    fft->stop();
    splitter_vfo->stop();
    source_ptr->stop();

    for (auto &kv : active_vfos)
    {
        kv.second.pipeline->stop();
        splitter_vfo->del_vfo(kv.first);
        logger->info("Stopped VFO %s", kv.first.c_str());
    }

    return 0;
}
