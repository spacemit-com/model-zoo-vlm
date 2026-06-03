/**
 * @file llama_vlm_model.cpp
 * @brief Implementation of the direct llama.cpp/mtmd VLM backend.
 *
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "llama_vlm_model.h"

#include <chrono>
#include <cstring>
#include <future>
#include <thread>

#include <llama.h>
#include <mtmd.h>
#include <mtmd-helper.h>

#include "mtmd_image.h"
#include "vlm_utils.h"

namespace vlm {

LlamaVlmModel::LlamaBackendGuard& LlamaVlmModel::LlamaBackendGuard::Instance() {
    static LlamaBackendGuard guard;
    return guard;
}

LlamaVlmModel::LlamaBackendGuard::~LlamaBackendGuard() {
    if (initialized_) {
        llama_backend_free();
    }
}

void LlamaVlmModel::LlamaBackendGuard::EnsureInit() {
    if (!initialized_) {
        llama_backend_init();
        initialized_ = true;
    }
}

LlamaVlmModel::LlamaVlmModel() = default;

LlamaVlmModel::~LlamaVlmModel() {
    Shutdown();
}

bool LlamaVlmModel::Initialize(const VlmModelConfig& config,
                                std::string* error) {
    std::lock_guard<std::mutex> lock(mutex_);
    LlamaBackendGuard::Instance().EnsureInit();

    if (config.text_model_path.empty()) {
        SetError(error, "text_model_path is required for direct backend");
        return false;
    }

    for (const auto& vp : config.vision_model_paths) {
        if (!vp.empty() && vp.size() >= 5 && vp.substr(vp.size() - 5) == ".onnx") {
            SetError(error, "ONNX vision models not supported by direct backend: " + vp);
            return false;
        }
    }

    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    model_ = llama_model_load_from_file(config.text_model_path.c_str(),
                                        model_params);
    if (!model_) {
        SetError(error, "Failed to load model: " + config.text_model_path);
        return false;
    }

    auto ctx_params = llama_context_default_params();
    ctx_params.n_ctx = config.generation.context_size;
    ctx_params.n_batch = 512;
    ctx_params.n_ubatch = 512;
    ctx_params.n_threads = config.generation.threads;
    ctx_params.n_threads_batch = config.generation.batch_threads;
    context_ = llama_init_from_model(model_, ctx_params);
    if (!context_) {
        SetError(error, "Failed to create llama context");
        llama_model_free(model_);
        model_ = nullptr;
        return false;
    }

    if (!config.vision_model_paths.empty() && !config.vision_model_paths[0].empty()) {
        auto mtmd_params = mtmd_context_params_default();
        mtmd_params.use_gpu = false;
        mtmd_ctx_ = mtmd_init_from_file(config.vision_model_paths[0].c_str(),
                                        model_, mtmd_params);
        if (!mtmd_ctx_) {
            SetError(error, "Failed to initialize mtmd from: " + config.vision_model_paths[0]);
            llama_free(context_);
            context_ = nullptr;
            llama_model_free(model_);
            model_ = nullptr;
            return false;
        }
    }

    if (!sampler_.Initialize(config.generation, error)) {
        Shutdown();
        return false;
    }

    config_ = config;
    ready_ = true;
    return true;
}

void LlamaVlmModel::Shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    ready_ = false;
    if (mtmd_ctx_) {
        mtmd_free(mtmd_ctx_);
        mtmd_ctx_ = nullptr;
    }
    if (context_) {
        llama_free(context_);
        context_ = nullptr;
    }
    if (model_) {
        llama_model_free(model_);
        model_ = nullptr;
    }
}

bool LlamaVlmModel::IsReady() const {
    return ready_.load();
}

bool LlamaVlmModel::DoGenerate(const VlmInput& input,
                                VlmStreamCallback callback,
                                std::string* output,
                                VlmUsage* usage,
                                std::string* error) {
    if (!ready_.load() || !model_ || !context_) {
        SetError(error, "Model not initialized");
        return false;
    }

    MtmdBitmapPtr bitmap_ptr = LoadBitmapFromInput(mtmd_ctx_, input.image_path,
                                                    input.image_bytes, error);
    mtmd_bitmap* bitmap = bitmap_ptr.release();

    mtmd_input_text mm_text;
    mm_text.text = input.prompt.c_str();
    mm_text.add_special = true;
    mm_text.parse_special = true;

    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    if (!chunks) {
        SetError(error, "mtmd_input_chunks_init failed");
        if (bitmap) mtmd_bitmap_free(bitmap);
        return false;
    }

    int32_t chunk_result = 0;
    if (!mtmd_ctx_) {
        // No vision context: cannot use mtmd tokenization path
        SetError(error, "No vision context initialized; direct backend requires mtmd for tokenization");
        mtmd_input_chunks_free(chunks);
        if (bitmap) mtmd_bitmap_free(bitmap);
        return false;
    }
    if (bitmap) {
        const mtmd_bitmap* bitmaps[] = {bitmap};
        chunk_result = mtmd_tokenize(mtmd_ctx_, chunks, &mm_text, bitmaps, 1);
    } else {
        chunk_result = mtmd_tokenize(mtmd_ctx_, chunks, &mm_text, nullptr, 0);
    }

    if (chunk_result != 0) {
        SetError(error, "mtmd_tokenize failed");
        mtmd_input_chunks_free(chunks);
        if (bitmap) mtmd_bitmap_free(bitmap);
        return false;
    }

    llama_pos n_past = 0;
    int32_t n_tokens = mtmd_helper_eval_chunks(mtmd_ctx_, context_, chunks,
                                                n_past, 0,
                                                config_.generation.batch_threads,
                                                true, &n_past);
    mtmd_input_chunks_free(chunks);

    if (n_tokens < 0) {
        SetError(error, "mtmd_helper_eval_chunks failed");
        if (bitmap) mtmd_bitmap_free(bitmap);
        return false;
    }

    std::string result_text;
    VlmUsage result_usage;
    bool ok = false;

    if (callback) {
        ok = sampler_.SampleStream(context_, model_,
                                   config_.generation.max_tokens,
                                   callback, &result_usage, error);
    } else {
        ok = sampler_.Sample(context_, model_,
                             config_.generation.max_tokens,
                             &result_text, &result_usage, error);
    }

    if (bitmap) {
        mtmd_bitmap_free(bitmap);
    }

    if (!ok) {
        return false;
    }

    if (output) {
        *output = result_text;
    }
    if (usage) {
        *usage = result_usage;
    }
    return true;
}

bool LlamaVlmModel::Generate(const VlmInput& input,
                              VlmResult* result,
                              std::string* error) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto start = std::chrono::high_resolution_clock::now();

    std::string output;
    VlmUsage usage;
    if (!DoGenerate(input, nullptr, &output, &usage, error)) {
        return false;
    }

    auto end = std::chrono::high_resolution_clock::now();
    double latency_ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::string thinking;
    std::string content = StripThinkingBlocks(output, &thinking);

    if (result) {
        result->text = content;
        result->reasoning_content = thinking;
        result->usage = usage;
        result->latency_ms = latency_ms;
        result->ttft_ms = 0.0;
        if (usage.completion_tokens > 0 && latency_ms > 0) {
            result->tokens_per_second = usage.completion_tokens / (latency_ms / 1000.0);
        }
    }

    metrics_.total_requests++;
    metrics_.last_latency_ms = latency_ms;
    return true;
}

bool LlamaVlmModel::GenerateAsync(const VlmInput& input,
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

bool LlamaVlmModel::GenerateStream(const VlmInput& input,
                                    VlmStreamCallback callback,
                                    VlmResult* result,
                                    std::string* error) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto start = std::chrono::high_resolution_clock::now();

    std::string output;
    VlmUsage usage;
    if (!DoGenerate(input, callback, &output, &usage, error)) {
        return false;
    }

    auto end = std::chrono::high_resolution_clock::now();
    double latency_ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::string thinking;
    std::string content = StripThinkingBlocks(output, &thinking);

    if (result) {
        result->text = content;
        result->reasoning_content = thinking;
        result->usage = usage;
        result->latency_ms = latency_ms;
    }

    metrics_.total_requests++;
    return true;
}

bool LlamaVlmModel::Chat(const std::vector<VlmChatMessage>& messages,
                          VlmChatResult* result,
                          std::string* error) {
    std::string last_user_text;
    std::string image_path;
    std::vector<uint8_t> image_bytes;

    for (const auto& m : messages) {
        if (m.role == VlmChatMessage::Role::USER) {
            last_user_text = m.content;
            if (!m.image_path.empty()) image_path = m.image_path;
            if (!m.image_bytes.empty()) image_bytes = m.image_bytes;
        }
    }

    VlmInput input;
    input.prompt = last_user_text;
    input.image_path = image_path;
    input.image_bytes = image_bytes;

    VlmResult gen_result;
    if (!Generate(input, &gen_result, error)) {
        if (result) {
            result->error = error ? *error : "Generate failed";
        }
        return false;
    }

    if (result) {
        result->content = gen_result.text;
        result->reasoning_content = gen_result.reasoning_content;
        result->usage = gen_result.usage;
        result->latency_ms = gen_result.latency_ms;
        result->ttft_ms = gen_result.ttft_ms;
        result->tokens_per_second = gen_result.tokens_per_second;
    }
    return true;
}

bool LlamaVlmModel::ChatStream(const std::vector<VlmChatMessage>& messages,
                                VlmStreamCallback callback,
                                VlmChatResult* result,
                                std::string* error) {
    std::string last_user_text;
    std::string image_path;
    std::vector<uint8_t> image_bytes;

    for (const auto& m : messages) {
        if (m.role == VlmChatMessage::Role::USER) {
            last_user_text = m.content;
            if (!m.image_path.empty()) image_path = m.image_path;
            if (!m.image_bytes.empty()) image_bytes = m.image_bytes;
        }
    }

    VlmInput input;
    input.prompt = last_user_text;
    input.image_path = image_path;
    input.image_bytes = image_bytes;

    VlmResult gen_result;
    if (!GenerateStream(input, callback, &gen_result, error)) {
        if (result) {
            result->error = error ? *error : "GenerateStream failed";
        }
        return false;
    }

    if (result) {
        result->content = gen_result.text;
        result->reasoning_content = gen_result.reasoning_content;
        result->usage = gen_result.usage;
        result->latency_ms = gen_result.latency_ms;
        result->tokens_per_second = gen_result.tokens_per_second;
    }
    return true;
}

bool LlamaVlmModel::Reset(std::string* error) {
    VlmModelConfig saved = config_;
    Shutdown();
    return Initialize(saved, error);
}

bool LlamaVlmModel::UpdateGenerationConfig(const VlmGenerationConfig& config,
                                            std::string* error) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_.generation = config;
    return sampler_.Initialize(config, error);
}

VlmGenerationConfig LlamaVlmModel::GetGenerationConfig() const {
    return config_.generation;
}

VlmMetrics LlamaVlmModel::GetMetrics() const {
    return metrics_;
}

}  // namespace vlm
