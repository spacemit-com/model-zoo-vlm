/**
 * @file server_vlm_model.h
 * @brief Server-compatible backend for VLM service via llama-server.
 *
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef VLM_SRC_CORE_CPP_SERVER_VLM_MODEL_H_
#define VLM_SRC_CORE_CPP_SERVER_VLM_MODEL_H_

#include <atomic>
#include <mutex>
#include <string>
#include <vector>

#include "vlm/vlm_service.h"

namespace vlm {

class ServerVlmModel : public VlmService {
public:
    ServerVlmModel();
    ~ServerVlmModel() override;

    bool Initialize(const VlmModelConfig& config,
                    std::string* error) override;
    void Shutdown() override;
    bool IsReady() const override;
    bool Generate(const VlmInput& input,
                  VlmResult* result,
                  std::string* error) override;
    bool GenerateAsync(const VlmInput& input,
                       VlmAsyncCallback callback,
                       std::string* error) override;
    bool GenerateStream(const VlmInput& input,
                        VlmStreamCallback callback,
                        VlmResult* result,
                        std::string* error) override;
    bool Chat(const std::vector<VlmChatMessage>& messages,
              VlmChatResult* result,
              std::string* error) override;
    bool ChatStream(const std::vector<VlmChatMessage>& messages,
                    VlmStreamCallback callback,
                    VlmChatResult* result,
                    std::string* error) override;
    bool Reset(std::string* error) override;
    bool UpdateGenerationConfig(const VlmGenerationConfig& config,
                                std::string* error) override;
    VlmGenerationConfig GetGenerationConfig() const override;
    VlmMetrics GetMetrics() const override;

    static std::string BuildLlamaServerCommand(const VlmModelConfig& config);
    static std::string BuildChatCompletionPayload(
        const VlmModelConfig& config,
        const std::vector<VlmChatMessage>& messages,
        const std::string& image_base64,
        bool stream);
    static void UpdateMetricsForRequest(VlmMetrics* metrics,
                                        double latency_ms,
                                        double ttft_ms,
                                        int64_t output_tokens,
                                        double tokens_per_second);

private:
    bool StartServerIfNeeded(std::string* error);
    bool ServerResponds() const;
    static std::string ImageToBase64(const std::string& image_path,
                                     std::string* error);
    bool SyncRequest(const std::string& payload,
                     std::string* response,
                     std::string* error);
    bool StreamRequest(const std::string& payload,
                       VlmStreamCallback callback,
                       std::string* full_text,
                       double* ttft_ms,
                       int64_t* output_chunks,
                       std::string* error);
    bool ExtractImageBase64(const std::vector<VlmChatMessage>& messages,
                            std::string* image_b64,
                            std::string* error);

    VlmModelConfig config_;
    std::mutex mutex_;
    std::atomic<bool> ready_{false};
    VlmMetrics metrics_;
    pid_t server_pid_ = 0;
};

}  // namespace vlm

#endif  // VLM_SRC_CORE_CPP_SERVER_VLM_MODEL_H_
