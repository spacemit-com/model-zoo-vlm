/**
 * @file sampler.cpp
 * @brief Implementation of the Llama sampler wrapper.
 *
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sampler.h"

#include <llama.h>

#include "vlm_utils.h"

namespace vlm {

LlamaSampler::LlamaSampler() = default;

LlamaSampler::~LlamaSampler() {
    if (sampler_) {
        llama_sampler_free(sampler_);
        sampler_ = nullptr;
    }
}

bool LlamaSampler::Initialize(const VlmGenerationConfig& config,
                              std::string* error) {
    if (sampler_) {
        llama_sampler_free(sampler_);
    }
    config_ = config;

    auto* chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(chain, llama_sampler_init_temp(config.temperature));
    llama_sampler_chain_add(chain, llama_sampler_init_top_k(config.top_k));
    llama_sampler_chain_add(chain, llama_sampler_init_top_p(config.top_p, 1));
    llama_sampler_chain_add(chain,
        llama_sampler_init_penalties(
            static_cast<int32_t>(config.repeat_penalty > 1.0F ? 64 : 0),
            config.repeat_penalty, 0.0F, 0.0F));
    llama_sampler_chain_add(chain, llama_sampler_init_dist(42));

    sampler_ = chain;
    return true;
}

bool LlamaSampler::Sample(llama_context* ctx, const llama_model* model,
                           int max_tokens, std::string* output,
                           VlmUsage* usage, std::string* error) {
    if (!sampler_ || !ctx || !model) {
        SetError(error, "Sampler not initialized");
        return false;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);
    int n_ctx = llama_n_ctx(ctx);
    int n_cur = static_cast<int>(llama_n_ctx_seq(ctx));
    int n_gen = 0;

    if (output) {
        output->clear();
    }

    while (n_gen < max_tokens) {
        if (n_cur + 1 >= n_ctx) {
            break;
        }

        llama_token new_token = llama_sampler_sample(sampler_, ctx, -1);
        if (llama_vocab_is_eog(vocab, new_token)) {
            break;
        }

        char buf[256];
        int n = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, true);
        if (n > 0) {
            if (output) {
                output->append(buf, n);
            }
        }

        llama_sampler_accept(sampler_, new_token);

        if (llama_decode(ctx, llama_batch_get_one(&new_token, 1))) {
            SetError(error, "llama_decode failed during sampling");
            return false;
        }

        ++n_cur;
        ++n_gen;
    }

    if (usage) {
        usage->completion_tokens = n_gen;
    }

    return true;
}

bool LlamaSampler::SampleStream(llama_context* ctx, const llama_model* model,
                                 int max_tokens, VlmStreamCallback callback,
                                 VlmUsage* usage, std::string* error) {
    if (!sampler_ || !ctx || !model) {
        SetError(error, "Sampler not initialized");
        return false;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);
    int n_ctx = llama_n_ctx(ctx);
    int n_cur = static_cast<int>(llama_n_ctx_seq(ctx));
    int n_gen = 0;

    while (n_gen < max_tokens) {
        if (n_cur + 1 >= n_ctx) {
            break;
        }

        llama_token new_token = llama_sampler_sample(sampler_, ctx, -1);
        if (llama_vocab_is_eog(vocab, new_token)) {
            break;
        }

        char buf[256];
        int n = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, true);
        if (n > 0) {
            std::string piece(buf, n);
            if (callback) {
                bool cont = callback(piece, false, "");
                if (!cont) {
                    break;
                }
            }
        }

        llama_sampler_accept(sampler_, new_token);

        if (llama_decode(ctx, llama_batch_get_one(&new_token, 1))) {
            SetError(error, "llama_decode failed during stream sampling");
            return false;
        }

        ++n_cur;
        ++n_gen;
    }

    if (callback) {
        callback("", true, "");
    }

    if (usage) {
        usage->completion_tokens = n_gen;
    }

    return true;
}

}  // namespace vlm
