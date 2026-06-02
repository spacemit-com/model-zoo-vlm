/**
 * @file server_vlm_model.cpp
 * @brief Implementation of the server-compatible VLM backend.
 *
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "server_vlm_model.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <future>
#include <sstream>
#include <signal.h>
#include <string>
#include <thread>

#include <sys/types.h>
#include <sys/wait.h>
#include <sys/socket.h>
#include <unistd.h>

#include "vlm_utils.h"

namespace vlm {

namespace {

void TerminateProcess(pid_t pid) {
    if (pid <= 0) return;
    kill(pid, SIGTERM);
    for (int i = 0; i < 30; ++i) {
        int status = 0;
        pid_t done = waitpid(pid, &status, WNOHANG);
        if (done == pid || (done < 0 && errno == ECHILD)) {
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    kill(pid, SIGKILL);
    int status = 0;
    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
}

}  // namespace

ServerVlmModel::ServerVlmModel() = default;

ServerVlmModel::~ServerVlmModel() {
    Shutdown();
}

bool ServerVlmModel::Initialize(const VlmModelConfig& config,
                                std::string* error) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (config.model_dir.empty()) {
        SetError(error, "model_dir is required for server backend");
        return false;
    }
    config_ = config;
    ready_ = false;
    return true;
}

void ServerVlmModel::Shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    ready_ = false;
    if (server_pid_ > 0) {
        TerminateProcess(server_pid_);
        server_pid_ = 0;
    }
}

bool ServerVlmModel::IsReady() const {
    return ready_.load();
}

bool ServerVlmModel::Generate(const VlmInput& input,
                               VlmResult* result,
                               std::string* error) {
    if (!StartServerIfNeeded(error)) {
        return false;
    }

    std::string image_b64;
    if (!input.image_path.empty()) {
        image_b64 = ImageToBase64(input.image_path, error);
        if (image_b64.empty() && !input.image_path.empty()) {
            return false;
        }
    } else if (!input.image_bytes.empty()) {
        std::vector<unsigned char> bytes(input.image_bytes.begin(),
                                         input.image_bytes.end());
        image_b64 = Base64Encode(bytes);
    }

    auto messages = std::vector<VlmChatMessage>{
        VlmChatMessage::UserWithImage(input.prompt, input.image_path)
    };
    std::string payload = BuildChatCompletionPayload(config_, messages,
                                                     image_b64, false);
    std::string response;
    auto start = std::chrono::high_resolution_clock::now();
    if (!SyncRequest(payload, &response, error)) {
        return false;
    }
    auto end = std::chrono::high_resolution_clock::now();
    double latency_ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::string thinking;
    std::string content = StripThinkingBlocks(
        ExtractJsonString(response, "content"), &thinking);

    if (result) {
        result->text = content;
        result->reasoning_content = thinking;
        result->usage.prompt_tokens = ExtractJsonInt(response, "prompt_tokens");
        result->usage.completion_tokens = ExtractJsonInt(response, "completion_tokens");
        result->usage.total_tokens = ExtractJsonInt(response, "total_tokens");
        result->latency_ms = latency_ms;
        result->ttft_ms = 0.0;
        if (result->usage.completion_tokens > 0 && latency_ms > 0) {
            result->tokens_per_second = result->usage.completion_tokens / (latency_ms / 1000.0);
        }
    }

    UpdateMetricsForRequest(&metrics_, latency_ms, result ? result->ttft_ms : 0.0,
                            result ? result->usage.completion_tokens : -1,
                            result ? result->tokens_per_second : 0.0);
    return true;
}

bool ServerVlmModel::GenerateAsync(const VlmInput& input,
                                    VlmAsyncCallback callback,
                                    std::string* error) {
    VlmResult result;
    std::string err;
    bool ok = Generate(input, &result, &err);
    if (callback) {
        callback(result, err);
    }
    if (!ok) {
        SetError(error, err);
    }
    return ok;
}

bool ServerVlmModel::GenerateStream(const VlmInput& input,
                                     VlmStreamCallback callback,
                                     VlmResult* result,
                                     std::string* error) {
    if (!StartServerIfNeeded(error)) {
        return false;
    }

    std::string image_b64;
    if (!input.image_path.empty()) {
        image_b64 = ImageToBase64(input.image_path, error);
        if (image_b64.empty() && !input.image_path.empty()) {
            return false;
        }
    } else if (!input.image_bytes.empty()) {
        std::vector<unsigned char> bytes(input.image_bytes.begin(),
                                         input.image_bytes.end());
        image_b64 = Base64Encode(bytes);
    }

    auto messages = std::vector<VlmChatMessage>{
        VlmChatMessage::UserWithImage(input.prompt, input.image_path)
    };
    std::string payload = BuildChatCompletionPayload(config_, messages,
                                                     image_b64, true);
    std::string full_text;
    double ttft_ms = 0.0;
    int64_t output_chunks = 0;
    auto start = std::chrono::high_resolution_clock::now();
    if (!StreamRequest(payload, callback, &full_text, &ttft_ms,
                       &output_chunks, error)) {
        return false;
    }
    auto end = std::chrono::high_resolution_clock::now();
    double latency_ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::string thinking;
    std::string content = StripThinkingBlocks(full_text, &thinking);

    if (result) {
        result->text = content;
        result->reasoning_content = thinking;
        result->latency_ms = latency_ms;
        result->ttft_ms = ttft_ms;
        if (output_chunks > 0 && latency_ms > 0) {
            result->tokens_per_second = output_chunks / (latency_ms / 1000.0);
        }
    }

    UpdateMetricsForRequest(&metrics_, latency_ms, ttft_ms, output_chunks,
                            (output_chunks > 0 && latency_ms > 0)
                                ? output_chunks / (latency_ms / 1000.0)
                                : 0.0);
    return true;
}

bool ServerVlmModel::Chat(const std::vector<VlmChatMessage>& messages,
                           VlmChatResult* result,
                           std::string* error) {
    if (!StartServerIfNeeded(error)) {
        return false;
    }

    std::string image_b64;
    if (!ExtractImageBase64(messages, &image_b64, error)) {
        return false;
    }

    std::string payload = BuildChatCompletionPayload(config_, messages,
                                                     image_b64, false);
    std::string response;
    auto start = std::chrono::high_resolution_clock::now();
    if (!SyncRequest(payload, &response, error)) {
        return false;
    }
    auto end = std::chrono::high_resolution_clock::now();
    double latency_ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::string thinking;
    std::string content = StripThinkingBlocks(
        ExtractJsonString(response, "content"), &thinking);

    if (result) {
        result->content = content;
        result->reasoning_content = thinking;
        result->usage.prompt_tokens = ExtractJsonInt(response, "prompt_tokens");
        result->usage.completion_tokens = ExtractJsonInt(response, "completion_tokens");
        result->usage.total_tokens = ExtractJsonInt(response, "total_tokens");
        result->latency_ms = latency_ms;
        if (result->usage.completion_tokens > 0 && latency_ms > 0) {
            result->tokens_per_second = result->usage.completion_tokens / (latency_ms / 1000.0);
        }
    }

    UpdateMetricsForRequest(&metrics_, latency_ms, result ? result->ttft_ms : 0.0,
                            result ? result->usage.completion_tokens : -1,
                            result ? result->tokens_per_second : 0.0);
    return true;
}

bool ServerVlmModel::ChatStream(const std::vector<VlmChatMessage>& messages,
                                 VlmStreamCallback callback,
                                 VlmChatResult* result,
                                 std::string* error) {
    if (!StartServerIfNeeded(error)) {
        return false;
    }

    std::string image_b64;
    if (!ExtractImageBase64(messages, &image_b64, error)) {
        return false;
    }

    std::string payload = BuildChatCompletionPayload(config_, messages,
                                                     image_b64, true);
    std::string full_text;
    double ttft_ms = 0.0;
    int64_t output_chunks = 0;
    auto start = std::chrono::high_resolution_clock::now();
    if (!StreamRequest(payload, callback, &full_text, &ttft_ms,
                       &output_chunks, error)) {
        return false;
    }
    auto end = std::chrono::high_resolution_clock::now();
    double latency_ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::string thinking;
    std::string content = StripThinkingBlocks(full_text, &thinking);

    if (result) {
        result->content = content;
        result->reasoning_content = thinking;
        result->latency_ms = latency_ms;
        result->ttft_ms = ttft_ms;
        if (output_chunks > 0 && latency_ms > 0) {
            result->tokens_per_second = output_chunks / (latency_ms / 1000.0);
        }
    }

    UpdateMetricsForRequest(&metrics_, latency_ms, ttft_ms, output_chunks,
                            (output_chunks > 0 && latency_ms > 0)
                                ? output_chunks / (latency_ms / 1000.0)
                                : 0.0);
    return true;
}

bool ServerVlmModel::Reset(std::string* error) {
    VlmModelConfig saved = config_;
    Shutdown();
    return Initialize(saved, error);
}

bool ServerVlmModel::UpdateGenerationConfig(const VlmGenerationConfig& config,
                                             std::string* error) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_.generation = config;
    return true;
}

VlmGenerationConfig ServerVlmModel::GetGenerationConfig() const {
    return config_.generation;
}

VlmMetrics ServerVlmModel::GetMetrics() const {
    return metrics_;
}

std::string ServerVlmModel::BuildLlamaServerCommand(
    const VlmModelConfig& config) {
    std::ostringstream cmd;

    if (!config.cpu_affinity.empty()) {
        cmd << "taskset -c " << ShellQuote(config.cpu_affinity) << " ";
    }

    cmd << "llama-server";
    cmd << " -m " << ShellQuote(config.text_model_path);
    cmd << " --port " << config.port;
    cmd << " --ctx-size " << config.generation.context_size;
    cmd << " -t " << config.generation.threads;
    cmd << " -tb " << config.generation.batch_threads;
    cmd << " --host " << ShellQuote(config.host);
    cmd << " -ngl " << config.n_gpu_layers;

    bool has_vision = !config.vision_model_paths.empty();
    for (const auto& vp : config.vision_model_paths) {
        if (!vp.empty()) {
            has_vision = true;
            break;
        }
    }

    if (has_vision) {
        switch (config.media_backend) {
            case VlmMediaBackend::kSmt:
                cmd << " --vision-backend smt";
                if (!config.smt_config_dir.empty()) {
                    cmd << " --smt-config-dir " << ShellQuote(config.smt_config_dir);
                } else {
                    cmd << " --smt-config-dir " << ShellQuote(config.model_dir);
                }
                break;
            case VlmMediaBackend::kMtmd:
                cmd << " --vision-backend mtmd";
                for (const auto& vp : config.vision_model_paths) {
                    if (!vp.empty()) {
                        cmd << " --mmproj " << ShellQuote(vp);
                    }
                }
                break;
            case VlmMediaBackend::kAuto:
                cmd << " --vision-backend auto";
                if (!config.smt_config_dir.empty()) {
                    cmd << " --smt-config-dir " << ShellQuote(config.smt_config_dir);
                }
                for (const auto& vp : config.vision_model_paths) {
                    if (!vp.empty()) {
                        cmd << " --mmproj " << ShellQuote(vp);
                    }
                }
                break;
        }

        if (!config.media_path.empty()) {
            cmd << " --media-path " << ShellQuote(config.media_path);
        }
    }

    if (config.no_warmup) {
        cmd << " --no-warmup";
    }

    if (!config.generation.enable_thinking) {
        cmd << " --reasoning off";
    } else {
        cmd << " --reasoning on";
        if (config.generation.reasoning_budget > 0) {
            cmd << " --reasoning-budget " << config.generation.reasoning_budget;
        }
    }

    return cmd.str();
}

std::string ServerVlmModel::BuildChatCompletionPayload(
    const VlmModelConfig& config,
    const std::vector<VlmChatMessage>& messages,
    const std::string& image_base64,
    bool stream) {
    std::ostringstream js;
    js << "{\"messages\":[";

    for (size_t i = 0; i < messages.size(); ++i) {
        if (i > 0) js << ",";
        const auto& m = messages[i];
        std::string role;
        switch (m.role) {
            case VlmChatMessage::Role::SYSTEM:    role = "system"; break;
            case VlmChatMessage::Role::USER:      role = "user"; break;
            case VlmChatMessage::Role::ASSISTANT: role = "assistant"; break;
        }
        js << "{\"role\":\"" << role << "\"";

        if (i == messages.size() - 1 && !image_base64.empty() &&
            m.role == VlmChatMessage::Role::USER) {
            js << ",\"content\":[";
            js << "{\"type\":\"text\",\"text\":\"" << JsonEscape(m.content) << "\"},";
            js << "{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/jpeg;base64,"
               << image_base64 << "\"}}";
            js << "]";
        } else {
            js << ",\"content\":\"" << JsonEscape(m.content) << "\"";
        }
        js << "}";
    }

    js << "],\"max_tokens\":" << config.generation.max_tokens;
    js << ",\"temperature\":" << config.generation.temperature;
    js << ",\"top_p\":" << config.generation.top_p;

    if (config.generation.enable_thinking) {
        js << ",\"enable_thinking\":true";
        if (config.generation.reasoning_budget > 0) {
            js << ",\"reasoning_budget\":" << config.generation.reasoning_budget;
        }
    }

    switch (config.generation.history_policy) {
        case VlmHistoryPolicy::kKeepAll:  js << ",\"vision_history\":\"keep_all\""; break;
        case VlmHistoryPolicy::kLastOnly: js << ",\"vision_history\":\"last\""; break;
        case VlmHistoryPolicy::kNone:     js << ",\"vision_history\":\"none\""; break;
    }

    if (!config.generation.stop_sequences.empty()) {
        js << ",\"stop\":[";
        for (size_t i = 0; i < config.generation.stop_sequences.size(); ++i) {
            if (i > 0) js << ",";
            js << "\"" << JsonEscape(config.generation.stop_sequences[i]) << "\"";
        }
        js << "]";
    }

    js << ",\"stream\":" << (stream ? "true" : "false");
    js << "}";
    return js.str();
}

void ServerVlmModel::UpdateMetricsForRequest(VlmMetrics* metrics,
                                             double latency_ms,
                                             double ttft_ms,
                                             int64_t output_tokens,
                                             double tokens_per_second) {
    if (!metrics) {
        return;
    }
    const int previous_requests = metrics->total_requests;
    metrics->total_requests = previous_requests + 1;
    metrics->last_latency_ms = latency_ms;
    metrics->avg_latency_ms =
        ((metrics->avg_latency_ms * previous_requests) + latency_ms) /
        metrics->total_requests;
    metrics->last_ttft_ms = ttft_ms;
    metrics->last_output_tokens = output_tokens;
    metrics->last_tokens_per_second = tokens_per_second;
}

bool ServerVlmModel::StartServerIfNeeded(std::string* error) {
    if (ready_.load() && ServerResponds()) {
        return true;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    if (server_pid_ > 0) {
        if (ServerResponds()) {
            ready_ = true;
            return true;
        }
        TerminateProcess(server_pid_);
        server_pid_ = 0;
    }

    if (config_.media_backend == VlmMediaBackend::kSmt) {
        (void)std::system("spacemit-tcm-smi -c > /dev/null 2>&1");
    }

    std::string cmd = BuildLlamaServerCommand(config_);
    const std::string threads = std::to_string(config_.generation.threads);
    pid_t pid = fork();
    if (pid < 0) {
        SetError(error, std::string("fork() failed: ") + strerror(errno));
        return false;
    }
    if (pid == 0) {
        setenv("OMP_NUM_THREADS", threads.c_str(), 1);
        execl("/bin/sh", "sh", "-c", ("exec " + cmd).c_str(), nullptr);
        _exit(127);
    }
    server_pid_ = pid;

    for (int i = 0; i < 120; ++i) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        if (ServerResponds()) {
            ready_ = true;
            return true;
        }
    }

    SetError(error, "llama-server failed to start within 120 seconds");
    TerminateProcess(server_pid_);
    server_pid_ = 0;
    ready_ = false;
    return false;
}

bool ServerVlmModel::ServerResponds() const {
    std::string url = "http://" + config_.host + ":" + std::to_string(config_.port) + "/v1/models";
    std::string curl_cmd = "curl -sf " + ShellQuote(url) + " --connect-timeout 2 --max-time 5";
    std::string cmd = curl_cmd + " 2>/dev/null";
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return false;
    char buf[4096];
    std::string response;
    while (fgets(buf, sizeof(buf), pipe) != nullptr) {
        response += buf;
    }
    int status = pclose(pipe);
    if (status != 0) return false;
    return response.find("\"id\"") != std::string::npos;
}

std::string ServerVlmModel::ImageToBase64(const std::string& image_path,
                                           std::string* error) {
    std::string data = ReadBinaryFile(image_path, error);
    if (data.empty()) {
        SetError(error, "Failed to read image file: " + image_path);
        return {};
    }
    std::vector<unsigned char> bytes(data.begin(), data.end());
    return Base64Encode(bytes);
}

bool ServerVlmModel::SyncRequest(const std::string& payload,
                                  std::string* response,
                                  std::string* error) {
    std::string url = "http://" + config_.host + ":" + std::to_string(config_.port) + "/v1/chat/completions";

    char tmpname[] = "/tmp/vlm_payload_XXXXXX";
    int tmpfd = mkstemp(tmpname);
    if (tmpfd < 0) {
        SetError(error, "mkstemp() failed");
        return false;
    }
    write(tmpfd, payload.data(), payload.size());
    close(tmpfd);

    std::string curl_cmd =
        "curl -sf -X POST " + ShellQuote(url) +
        " -H 'Content-Type: application/json'"
        " -d @" + std::string(tmpname) +
        " --max-time 180";

    FILE* pipe = popen(curl_cmd.c_str(), "r");
    if (!pipe) {
        unlink(tmpname);
        SetError(error, "popen() failed for curl request");
        return false;
    }

    char buf[4096];
    std::string result;
    while (fgets(buf, sizeof(buf), pipe) != nullptr) {
        result += buf;
    }
    int status = pclose(pipe);
    unlink(tmpname);

    if (status != 0) {
        SetError(error, "curl request failed with status " + std::to_string(status));
        return false;
    }

    if (response) {
        *response = result;
    }
    return true;
}

bool ServerVlmModel::StreamRequest(const std::string& payload,
                                    VlmStreamCallback callback,
                                    std::string* full_text,
                                    double* ttft_ms,
                                    int64_t* output_chunks,
                                    std::string* error) {
    std::string url = "http://" + config_.host + ":" + std::to_string(config_.port) + "/v1/chat/completions";

    char tmpname[] = "/tmp/vlm_stream_payload_XXXXXX";
    int tmpfd = mkstemp(tmpname);
    if (tmpfd < 0) {
        SetError(error, std::string("mkstemp() failed for streaming payload: ") + strerror(errno));
        return false;
    }
    ssize_t written = write(tmpfd, payload.data(), payload.size());
    if (written < 0 || static_cast<size_t>(written) != payload.size()) {
        close(tmpfd);
        unlink(tmpname);
        SetError(error, std::string("write() failed for streaming payload: ") + strerror(errno));
        return false;
    }
    close(tmpfd);

    std::string curl_cmd =
        "curl -N --no-buffer -X POST " + ShellQuote(url) +
        " -H 'Content-Type: application/json'" +
        " -d @" + std::string(tmpname) +
        " --max-time 180";

    FILE* pipe = popen(curl_cmd.c_str(), "r");
    if (!pipe) {
        unlink(tmpname);
        SetError(error, "popen() failed for streaming curl request");
        return false;
    }

    char buf[4096];
    std::string accumulated;
    bool first_token = true;
    auto start = std::chrono::high_resolution_clock::now();
    if (ttft_ms) {
        *ttft_ms = 0.0;
    }
    if (output_chunks) {
        *output_chunks = 0;
    }

    while (fgets(buf, sizeof(buf), pipe) != nullptr) {
        std::string line(buf);
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
            line.pop_back();
        }

        if (line.find("data: [DONE]") != std::string::npos) {
            break;
        }
        if (line.find("data: ") != 0) {
            continue;
        }

        std::string json_str = line.substr(6);
        std::string delta = ExtractJsonString(json_str, "content");
        if (delta.empty()) {
            continue;
        }

        if (first_token) {
            first_token = false;
            if (ttft_ms) {
                auto first = std::chrono::high_resolution_clock::now();
                *ttft_ms = std::chrono::duration<double, std::milli>(
                    first - start).count();
            }
        }

        accumulated += delta;
        if (output_chunks) {
            ++(*output_chunks);
        }
        if (full_text) {
            *full_text = accumulated;
        }

        if (callback) {
            bool cont = callback(delta, false, "");
            if (!cont) {
                break;
            }
        }
    }

    int status = pclose(pipe);
    unlink(tmpname);

    if (status != 0) {
        SetError(error, "streaming curl request failed with status " + std::to_string(status));
        return false;
    }

    if (callback) {
        callback("", true, "");
    }

    return true;
}

bool ServerVlmModel::ExtractImageBase64(
    const std::vector<VlmChatMessage>& messages,
    std::string* image_b64,
    std::string* error) {
    if (!image_b64) {
        return true;
    }
    for (const auto& m : messages) {
        if (m.role == VlmChatMessage::Role::USER) {
            if (!m.image_path.empty()) {
                *image_b64 = ImageToBase64(m.image_path, error);
                if (image_b64->empty()) {
                    return false;
                }
                return true;
            }
            if (!m.image_bytes.empty()) {
                std::vector<unsigned char> bytes(m.image_bytes.begin(),
                                                 m.image_bytes.end());
                *image_b64 = Base64Encode(bytes);
                return true;
            }
        }
    }
    return true;
}

}  // namespace vlm
