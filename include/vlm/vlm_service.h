/**
 * @file vlm_service.h
 * @brief Public API for the VLM (Vision Language Model) inference service.
 *
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef VLM_VLM_SERVICE_H_
#define VLM_VLM_SERVICE_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace vlm {

enum class BackendType {
    kDirect,
    kServerCompat,
};

enum class VlmMediaBackend {
    kAuto,
    kMtmd,
    kSmt,
};

enum class VlmHistoryPolicy {
    kKeepAll,
    kLastOnly,
    kNone,
};

struct VlmGenerationConfig {
    // Default generation parameters loaded from YAML and used by sync/stream APIs.
    int max_tokens = 150;
    int context_size = 4096;
    int top_k = 40;
    float top_p = 0.95F;
    float temperature = 0.2F;
    float repeat_penalty = 1.2F;
    int threads = 8;
    int batch_threads = 8;
    bool enable_thinking = false;
    int reasoning_budget = -1;
    VlmHistoryPolicy history_policy = VlmHistoryPolicy::kLastOnly;
    std::vector<std::string> stop_sequences;
};

struct VlmModelConfig {
    // Model loading and runtime options. Server mode starts/owns a llama-server process.
    std::string model_dir;
    std::string model_name;
    std::string text_model_path;
    std::vector<std::string> vision_model_paths;
    std::string architecture;
    BackendType backend_type = BackendType::kDirect;
    VlmMediaBackend media_backend = VlmMediaBackend::kSmt;
    std::string smt_config_dir;
    std::string media_path;
    std::string cpu_affinity;
    int n_gpu_layers = 0;
    bool no_warmup = true;
    std::string host = "127.0.0.1";
    int port = 8086;
    VlmGenerationConfig generation;
};

struct VlmInput {
    // Single-turn request. Provide either image_path, image_bytes, or neither for text-only.
    std::string prompt;
    std::string image_path;
    std::vector<std::uint8_t> image_bytes;
};

struct VlmUsage {
    int prompt_tokens = 0;
    int completion_tokens = 0;
    int total_tokens = 0;
};

struct VlmResult {
    std::string text;
    std::string reasoning_content;
    VlmUsage usage;
    double latency_ms = 0.0;
    double ttft_ms = 0.0;
    double tokens_per_second = 0.0;
};

struct VlmChatMessage {
    enum class Role { SYSTEM, USER, ASSISTANT };
    Role role = Role::USER;
    std::string content;
    std::string image_path;
    std::vector<std::uint8_t> image_bytes;

    static VlmChatMessage System(const std::string& content) {
        return {Role::SYSTEM, content, "", {}};
    }

    static VlmChatMessage User(const std::string& content) {
        return {Role::USER, content, "", {}};
    }

    static VlmChatMessage UserWithImage(const std::string& content,
                                        const std::string& path) {
        return {Role::USER, content, path, {}};
    }

    static VlmChatMessage UserWithImageBytes(
        const std::string& content,
        const std::vector<std::uint8_t>& bytes) {
        return {Role::USER, content, "", bytes};
    }

    static VlmChatMessage Assistant(const std::string& content) {
        return {Role::ASSISTANT, content, "", {}};
    }
};

struct VlmChatResult {
    std::string content;
    std::string reasoning_content;
    VlmUsage usage;
    double latency_ms = 0.0;
    double ttft_ms = 0.0;
    double tokens_per_second = 0.0;
    std::string error;
};

using VlmStreamCallback = std::function<bool(const std::string& chunk,
                                             bool is_done,
                                             const std::string& error)>;

using VlmAsyncCallback = std::function<void(const VlmResult& result,
                                            const std::string& error)>;

struct VlmMetrics {
    int total_requests = 0;
    bool is_processing = false;
    double last_latency_ms = 0.0;
    double avg_latency_ms = 0.0;
    double last_ttft_ms = 0.0;
    int64_t last_output_tokens = -1;
    double last_tokens_per_second = 0.0;
};

class VlmService {
public:
    virtual ~VlmService() = default;

    virtual bool Initialize(const VlmModelConfig& config,
                            std::string* error) = 0;

    // Releases model resources and stops owned server processes. Safe before destruction.
    virtual void Shutdown() = 0;

    virtual bool IsReady() const = 0;

    // Synchronous single-turn text or image+text generation.
    virtual bool Generate(const VlmInput& input,
                          VlmResult* result,
                          std::string* error) = 0;

    // Asynchronous completion callback API. Implementations may dispatch internally.
    virtual bool GenerateAsync(const VlmInput& input,
                               VlmAsyncCallback callback,
                               std::string* error) = 0;

    // Streaming single-turn generation. Callback returns false to stop receiving chunks.
    virtual bool GenerateStream(const VlmInput& input,
                                VlmStreamCallback callback,
                                VlmResult* result,
                                std::string* error) = 0;

    // Multi-turn chat. User messages may include image_path or image_bytes.
    virtual bool Chat(const std::vector<VlmChatMessage>& messages,
                      VlmChatResult* result,
                      std::string* error) = 0;

    // Streaming multi-turn chat. Final metrics are written to result when available.
    virtual bool ChatStream(const std::vector<VlmChatMessage>& messages,
                            VlmStreamCallback callback,
                            VlmChatResult* result,
                            std::string* error) = 0;

    // Clears implementation-owned conversation/runtime state where supported.
    virtual bool Reset(std::string* error) = 0;

    // Updates service-wide generation defaults used by subsequent requests.
    virtual bool UpdateGenerationConfig(const VlmGenerationConfig& config,
                                        std::string* error) = 0;

    virtual VlmGenerationConfig GetGenerationConfig() const = 0;

    virtual VlmMetrics GetMetrics() const = 0;
};

std::unique_ptr<VlmService> CreateVlmService(const VlmModelConfig& config,
                                             std::string* error);

std::unique_ptr<VlmService> CreateVlmServiceFromConfig(
    const std::string& config_path,
    std::string* error);

}  // namespace vlm

#endif  // VLM_VLM_SERVICE_H_
