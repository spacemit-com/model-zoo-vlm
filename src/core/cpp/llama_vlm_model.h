/**
 * @file llama_vlm_model.h
 * @brief Direct llama.cpp/mtmd backend for VLM service.
 *
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef VLM_SRC_CORE_CPP_LLAMA_VLM_MODEL_H_
#define VLM_SRC_CORE_CPP_LLAMA_VLM_MODEL_H_

#include <atomic>
#include <mutex>
#include <string>
#include <vector>

#include "sampler.h"
#include "vlm_service.h"

struct llama_model;
struct llama_context;
struct mtmd_context;

namespace vlm {

class LlamaVlmModel : public VlmService {
public:
    LlamaVlmModel();
    ~LlamaVlmModel() override;

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

private:
    class LlamaBackendGuard {
    public:
        static LlamaBackendGuard& Instance();
        void EnsureInit();

    private:
        LlamaBackendGuard() = default;
        ~LlamaBackendGuard();
        bool initialized_ = false;
    };

    bool DoGenerate(const VlmInput& input,
                    VlmStreamCallback callback,
                    std::string* output,
                    VlmUsage* usage,
                    std::string* error);

    VlmModelConfig config_;
    llama_model* model_ = nullptr;
    llama_context* context_ = nullptr;
    mtmd_context* mtmd_ctx_ = nullptr;
    LlamaSampler sampler_;
    std::mutex mutex_;
    std::atomic<bool> ready_{false};
    VlmMetrics metrics_;
};

}  // namespace vlm

#endif  // VLM_SRC_CORE_CPP_LLAMA_VLM_MODEL_H_
