/**
 * @file sampler.h
 * @brief Llama sampler wrapper for VLM token generation.
 *
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef SAMPLER_H
#define SAMPLER_H

#include <functional>
#include <string>

#include "vlm_service.h"

struct llama_context;
struct llama_model;
struct llama_sampler;

namespace vlm {

class LlamaSampler {
public:
    LlamaSampler();
    ~LlamaSampler();

    bool Initialize(const VlmGenerationConfig& config, std::string* error);
    bool Sample(llama_context* ctx, const llama_model* model,
                int max_tokens, std::string* output,
                VlmUsage* usage, std::string* error);
    bool SampleStream(llama_context* ctx, const llama_model* model,
                        int max_tokens, VlmStreamCallback callback,
                        VlmUsage* usage, std::string* error);

private:
    llama_sampler* sampler_ = nullptr;
    VlmGenerationConfig config_;
};

}  // namespace vlm

#endif  // SAMPLER_H
