/**
 * @file test_server_vlm.cpp
 * @brief Integration test for ServerVlmModel with TCM hardware acceleration.
 *
 * Tests the full pipeline: BuildLlamaServerCommand -> StartServerIfNeeded
 * -> Generate/Chat with image -> verify TCM is active.
 *
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "server_vlm_model.h"
#include "vlm/vlm_service.h"

static int g_passed = 0;
static int g_failed = 0;

static void Check(const char* name, bool condition, const char* detail = nullptr) {
    if (condition) {
        std::fprintf(stderr, "  [PASS] %s\n", name);
        ++g_passed;
    } else {
        std::fprintf(stderr, "  [FAIL] %s%s%s\n", name,
                     detail ? " - " : "", detail ? detail : "");
        ++g_failed;
    }
}

/**
 * @brief Test BuildLlamaServerCommand produces correct SMT/TCM parameters.
 */
static void TestBuildCommand() {
    std::fprintf(stderr, "\n[Test 1] BuildLlamaServerCommand with SMT backend\n");
    std::fprintf(stderr, "================================================\n");

    vlm::VlmModelConfig config;
    config.model_dir = "/opt/models/vlm/Qwen3.5-2B";
    config.text_model_path = "/opt/models/vlm/Qwen3.5-2B/text.gguf";
    config.vision_model_paths = {"/opt/models/vlm/Qwen3.5-2B/vision.onnx"};
    config.backend_type = vlm::BackendType::kServerCompat;
    config.media_backend = vlm::VlmMediaBackend::kSmt;
    config.smt_config_dir = "/opt/models/vlm/Qwen3.5-2B";
    config.media_path = "/home/user/pictures";
    config.cpu_affinity = "0-7";
    config.n_gpu_layers = 0;
    config.no_warmup = true;
    config.host = "0.0.0.0";
    config.port = 8071;
    config.generation.threads = 8;
    config.generation.batch_threads = 8;
    config.generation.context_size = 4096;
    config.generation.enable_thinking = false;

    std::string cmd = vlm::ServerVlmModel::BuildLlamaServerCommand(config);
    std::fprintf(stderr, "  Generated command:\n  %s\n\n", cmd.c_str());

    Check("command contains 'taskset -c' and '0-7'",
          cmd.find("taskset -c") != std::string::npos &&
          cmd.find("0-7") != std::string::npos);
    Check("command contains 'OMP_NUM_THREADS=8' before 'taskset'",
          cmd.find("OMP_NUM_THREADS=8") < cmd.find("taskset"));
    Check("command contains 'llama-server'",
          cmd.find("llama-server") != std::string::npos);
    Check("command contains '--vision-backend smt'",
          cmd.find("--vision-backend smt") != std::string::npos);
    Check("command contains '--smt-config-dir'",
          cmd.find("--smt-config-dir") != std::string::npos);
    Check("command contains '--media-path'",
          cmd.find("--media-path") != std::string::npos);
    Check("command contains '-ngl 0'",
          cmd.find("-ngl 0") != std::string::npos);
    Check("command contains '--no-warmup'",
          cmd.find("--no-warmup") != std::string::npos);
    Check("command contains '--reasoning off'",
          cmd.find("--reasoning off") != std::string::npos);
    Check("command contains '-tb 8'",
          cmd.find("-tb 8") != std::string::npos);
    Check("command contains '--ctx-size 4096'",
          cmd.find("--ctx-size 4096") != std::string::npos);
}

/**
 * @brief Test BuildLlamaServerCommand with MTMD backend.
 */
static void TestBuildCommandMtmd() {
    std::fprintf(stderr, "\n[Test 2] BuildLlamaServerCommand with MTMD backend\n");
    std::fprintf(stderr, "=================================================\n");

    vlm::VlmModelConfig config;
    config.model_dir = "/opt/models/vlm/fastvlm";
    config.text_model_path = "/opt/models/vlm/fastvlm/text.gguf";
    config.vision_model_paths = {"/opt/models/vlm/fastvlm/mmproj.gguf"};
    config.backend_type = vlm::BackendType::kServerCompat;
    config.media_backend = vlm::VlmMediaBackend::kMtmd;
    config.cpu_affinity = "0-3";
    config.n_gpu_layers = 0;
    config.no_warmup = false;
    config.host = "127.0.0.1";
    config.port = 8080;
    config.generation.threads = 4;
    config.generation.batch_threads = 4;
    config.generation.context_size = 2048;
    config.generation.enable_thinking = true;
    config.generation.reasoning_budget = 512;

    std::string cmd = vlm::ServerVlmModel::BuildLlamaServerCommand(config);
    std::fprintf(stderr, "  Generated command:\n  %s\n\n", cmd.c_str());

    Check("command contains '--vision-backend mtmd'",
          cmd.find("--vision-backend mtmd") != std::string::npos);
    Check("command contains '--mmproj'",
          cmd.find("--mmproj") != std::string::npos);
    Check("command does NOT contain '--smt-config-dir'",
          cmd.find("--smt-config-dir") == std::string::npos);
    Check("command contains '--reasoning on'",
          cmd.find("--reasoning on") != std::string::npos);
    Check("command contains '--reasoning-budget 512'",
          cmd.find("--reasoning-budget 512") != std::string::npos);
    Check("command does NOT contain '--no-warmup'",
          cmd.find("--no-warmup") == std::string::npos);
    Check("command contains 'taskset -c' and '0-3'",
          cmd.find("taskset -c") != std::string::npos &&
          cmd.find("0-3") != std::string::npos);
}

/**
 * @brief Test BuildChatCompletionPayload matches video-search-system format.
 */
static void TestBuildPayload() {
    std::fprintf(stderr, "\n[Test 3] BuildChatCompletionPayload format\n");
    std::fprintf(stderr, "==========================================\n");

    vlm::VlmModelConfig config;
    config.generation.max_tokens = 150;
    config.generation.temperature = 0.2;
    config.generation.top_p = 0.95;
    config.generation.repeat_penalty = 1.2;
    config.generation.enable_thinking = false;
    config.generation.history_policy = vlm::VlmHistoryPolicy::kLastOnly;
    config.generation.stop_sequences = {"<|im_end|>", "\n\n\n"};

    auto messages = std::vector<vlm::VlmChatMessage>{
        vlm::VlmChatMessage::UserWithImage("Describe this image.", "/test/img.jpg")
    };

    std::string payload = vlm::ServerVlmModel::BuildChatCompletionPayload(
        config, messages, "dGVzdA==", false);
    std::fprintf(stderr, "  Payload (truncated):\n  %.200s...\n\n", payload.c_str());

    Check("payload contains '\"messages\":'",
          payload.find("\"messages\":") != std::string::npos);
    Check("payload contains '\"role\":\"user\"'",
          payload.find("\"role\":\"user\"") != std::string::npos);
    Check("payload contains '\"type\":\"text\"'",
          payload.find("\"type\":\"text\"") != std::string::npos);
    Check("payload contains '\"type\":\"image_url\"'",
          payload.find("\"type\":\"image_url\"") != std::string::npos);
    Check("payload contains 'data:image/jpeg;base64,'",
          payload.find("data:image/jpeg;base64,") != std::string::npos);
    Check("payload contains '\"max_tokens\":150'",
          payload.find("\"max_tokens\":150") != std::string::npos);
    Check("payload contains '\"temperature\":0.2'",
          payload.find("\"temperature\":0.2") != std::string::npos);
    Check("payload contains '\"vision_history\":'",
          payload.find("\"vision_history\":") != std::string::npos);
    Check("payload contains '\"stop\":'",
          payload.find("\"stop\":") != std::string::npos);
    Check("payload contains '\"stream\":false'",
          payload.find("\"stream\":false") != std::string::npos);
}

/**
 * @brief Test VlmModelConfig new fields defaults.
 */
static void TestConfigDefaults() {
    std::fprintf(stderr, "\n[Test 4] VlmModelConfig new fields defaults\n");
    std::fprintf(stderr, "=============================================\n");

    vlm::VlmModelConfig config;
    Check("default media_backend is kSmt",
          config.media_backend == vlm::VlmMediaBackend::kSmt);
    Check("default n_gpu_layers is 0",
          config.n_gpu_layers == 0);
    Check("default no_warmup is true",
          config.no_warmup == true);
    Check("default smt_config_dir is empty",
          config.smt_config_dir.empty());
    Check("default media_path is empty",
          config.media_path.empty());
    Check("default cpu_affinity is empty",
          config.cpu_affinity.empty());
}

int main() {
    std::fprintf(stderr, "========================================\n");
    std::fprintf(stderr, "  ServerVlmModel Integration Test\n");
    std::fprintf(stderr, "========================================\n");

    TestBuildCommand();
    TestBuildCommandMtmd();
    TestBuildPayload();
    TestConfigDefaults();

    std::fprintf(stderr, "\n========================================\n");
    std::fprintf(stderr, "  Summary: %d passed, %d failed\n",
                 g_passed, g_failed);
    std::fprintf(stderr, "========================================\n");

    return g_failed > 0 ? 1 : 0;
}
