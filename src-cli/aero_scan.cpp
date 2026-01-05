#include "aero_scan.h"
#include "logger.h"
#include "init.h"
#include "common/cli_utils.h"
#include "common/dsp_source_sink/dsp_sample_source.h"
#include "common/dsp/path/splitter_vfo.h"
#include "common/dsp/path/splitter.h"
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
#include <thread>
#include <deque>
#include <fftw3.h>
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
    int cluster_band_hz = parameters.value("cluster_band_hz", 8000); // bandwidth window to cluster bins
    int max_promotions_per_min = parameters.value("max_promotions_per_min", 10);
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
    std::map<long long, Candidate> candidates; // keyed by clustered offset
    std::map<std::string, ActiveVFO> active_vfos;
    std::atomic<int> vfo_counter{0};

    std::unique_ptr<dsp::VFOSplitterBlock> splitter_vfo;
    std::unique_ptr<dsp::SplitterBlock> splitter;
    std::thread psd_thread;
    std::atomic<bool> psd_running{true};
    std::atomic<long long> last_psd_ms{std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count()};
    std::deque<std::chrono::steady_clock::time_point> promotion_times;
    bool psd_thread_started = false;
    ctpl::thread_pool live_thread_pool(64);

    try
    {
        source_ptr->start();

        splitter = std::make_unique<dsp::SplitterBlock>(source_ptr->output_stream);
        splitter->add_output("fft");
        splitter->add_output("vfo");
        splitter->set_enabled("fft", true);
        splitter->set_enabled("vfo", true);

        splitter_vfo = std::make_unique<dsp::VFOSplitterBlock>(splitter->get_output("vfo"));
        splitter_vfo->set_main_enabled(false);

        int fft_size = parameters.value("fft_size", 2048);
        double bin_hz = samplerate / fft_size;
        auto fft_stream = splitter->get_output("fft");

        // Start PSD worker thread using FFTW directly
        splitter->start();
        splitter_vfo->start();

        auto start_psd = [&]()
        {
            psd_running.store(true);
            psd_thread = std::thread([&, fft_size, bin_hz]()
                                 {
                                     std::vector<complex_t> buf(fft_size);
                                     fftwf_complex *fftw_in = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * fft_size);
                                     fftwf_complex *fftw_out = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * fft_size);
                                     fftwf_plan plan = fftwf_plan_dft_1d(fft_size, fftw_in, fftw_out, FFTW_FORWARD, FFTW_ESTIMATE);

                                     std::vector<float> mags(fft_size);
                                     int heartbeat = 0;
                                     logger->info("PSD worker started");

                                     while (psd_running.load())
                                     {
                                         try
                                         {
                                             int got = fft_stream->read();
                                             if (got <= 0)
                                             {
                                                 fft_stream->flush();
                                                 std::this_thread::sleep_for(std::chrono::milliseconds(5));
                                                 heartbeat++;
                                                 if (heartbeat % 200 == 0)
                                                     logger->info("PSD heartbeat: candidates=%zu (no samples)", candidates.size());
                                                 last_psd_ms.store(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
                                                 continue;
                                             }

                                             int copy_n = std::min(got, fft_size);
                                             memcpy(buf.data(), fft_stream->readBuf, copy_n * sizeof(complex_t));
                                             fft_stream->flush();

                                             if (copy_n < fft_size)
                                                 continue;

                                             for (int i = 0; i < fft_size; i++)
                                             {
                                                fftw_in[i][0] = buf[i].real;
                                                fftw_in[i][1] = buf[i].imag;
                                             }

                                             fftwf_execute(plan);

                                             for (int i = 0; i < fft_size; i++)
                                                 mags[i] = fftw_out[i][0] * fftw_out[i][0] + fftw_out[i][1] * fftw_out[i][1];

                                             std::vector<float> sorted = mags;
                                             std::nth_element(sorted.begin(), sorted.begin() + fft_size / 2, sorted.end());
                                             float noise_floor = sorted[fft_size / 2];
                                             float safe_noise = noise_floor > 1e-9f ? noise_floor : 1e-9f;
                                             float threshold = safe_noise * powf(10.0f, snr_margin_db / 10.0f);
                                             auto now = std::chrono::steady_clock::now();

                                             std::lock_guard<std::mutex> lk(state_mutex);
                                             std::set<long long> seen_clusters;

                                             for (int i = 0; i < fft_size; i++)
                                             {
                                                 if (mags[i] < threshold)
                                                     continue;

                                                 double offset = (double(i) - (fft_size / 2)) * bin_hz;
                                                 if (std::abs(offset) > (samplerate / 2))
                                                     continue;

                                                 long long cluster_key = (long long)std::llround(offset / (double)cluster_band_hz) * (long long)cluster_band_hz;

                                                 if (seen_clusters.count(cluster_key))
                                                     continue;
                                                 seen_clusters.insert(cluster_key);

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

                                                 auto &cand = candidates[cluster_key];
                                                 cand.offset_hz = offset;
                                                 cand.hits += 1;
                                                 cand.last_seen = now;
                                             }

                                             for (auto it = candidates.begin(); it != candidates.end();)
                                             {
                                                 if (std::chrono::duration_cast<std::chrono::milliseconds>(now - it->second.last_seen).count() > drop_miss_ms)
                                                     it = candidates.erase(it);
                                                 else
                                                     ++it;
                                             }

                                             heartbeat++;
                                             if (heartbeat % 200 == 0)
                                                 logger->info("PSD heartbeat: candidates=%zu threshold=%.3e noise=%.3e", candidates.size(), threshold, noise_floor);
                                             last_psd_ms.store(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
                                         }
                                         catch (std::exception &e)
                                         {
                                             logger->error("PSD worker error: %s", e.what());
                                         }
                                     }

                                     logger->info("PSD worker stopping");

                                     fftwf_destroy_plan(plan);
                                     fftwf_free(fftw_in);
                                     fftwf_free(fftw_out);
                                 });
            psd_thread_started = true;
        };

        splitter->start();
        splitter_vfo->start();

        start_psd();
    }
    catch (std::exception &e)
    {
        logger->error("Fatal error starting scanner: %s", e.what());
        psd_running.store(false);
        if (psd_thread_started && psd_thread.joinable())
            psd_thread.join();
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

            // Age out stale candidates even if PSD stalled
            for (auto it = candidates.begin(); it != candidates.end();)
            {
                if (std::chrono::duration_cast<std::chrono::milliseconds>(now - it->second.last_seen).count() > drop_miss_ms)
                    it = candidates.erase(it);
                else
                    ++it;
            }

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
                long long cluster_key = (long long)std::llround(req.second / (double)cluster_band_hz) * (long long)cluster_band_hz;
                candidates.erase(cluster_key);
            }

            for (auto it = active_vfos.begin(); it != active_vfos.end(); ++it)
            {
                if (std::chrono::duration_cast<std::chrono::milliseconds>(now - it->second.last_seen).count() > drop_miss_ms)
                    stop_reqs.push_back(it->first);
            }
        }

        // Warn if PSD worker appears stale
        auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
        if (now_ms - last_psd_ms.load() > 5000)
            logger->warn("PSD worker appears stale (no heartbeat >5s)");

        for (auto &req : spawn_reqs)
        {
            double offset = req.second;
            std::string vfo_id = req.first;
            double final_shift = -offset;

            auto now = std::chrono::steady_clock::now();

            // Rate limit promotions per minute
            {
                std::unique_lock<std::mutex> lk(state_mutex);
                while (!promotion_times.empty() && std::chrono::duration_cast<std::chrono::seconds>(now - promotion_times.front()).count() > 60)
                    promotion_times.pop_front();
                if ((int)promotion_times.size() >= max_promotions_per_min)
                {
                    logger->warn("Promotion rate limit reached, skipping VFO %s", vfo_id.c_str());
                    continue;
                }
                promotion_times.push_back(now);
            }

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
    splitter_vfo->stop();
    splitter->stop();
    source_ptr->stop();
    psd_running.store(false);
    if (psd_thread_started && psd_thread.joinable())
        psd_thread.join();

    for (auto &kv : active_vfos)
    {
        kv.second.pipeline->stop();
        splitter_vfo->del_vfo(kv.first);
        logger->info("Stopped VFO %s", kv.first.c_str());
    }

    return 0;
}
