/**
 * @file test_vlm_model_factory.cpp
 * @brief Unit tests for VLM model factory.
 *
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cassert>
#include <iostream>
#include <string>

#include "vlm_service.h"
#include "server_vlm_model.h"

using vlm::BackendType;
using vlm::CreateVlmService;
using vlm::CreateVlmServiceFromConfig;
using vlm::ServerVlmModel;
using vlm::VlmChatMessage;
using vlm::VlmGenerationConfig;
using vlm::VlmMetrics;
using vlm::VlmModelConfig;
using vlm::VlmHistoryPolicy;

static void TestCreateWithEmptyConfig() {
    std::string error;
    VlmModelConfig config;
    config.backend_type = BackendType::kServerCompat;
    config.model_dir = "";
    auto svc = CreateVlmService(config, &error);
    assert(svc == nullptr);
    std::cout << "PASS: CreateWithEmptyConfig" << std::endl;
}

static void TestCreateFromInvalidPath() {
    std::string error;
    auto svc = CreateVlmServiceFromConfig("/nonexistent/config.yaml", &error);
    assert(svc == nullptr);
    assert(!error.empty());
    std::cout << "PASS: CreateFromInvalidPath" << std::endl;
}

static void TestBackendTypeEnum() {
    VlmModelConfig config;
    config.backend_type = BackendType::kDirect;
    assert(config.backend_type == BackendType::kDirect);
    config.backend_type = BackendType::kServerCompat;
    assert(config.backend_type == BackendType::kServerCompat);
    std::cout << "PASS: BackendTypeEnum" << std::endl;
}

static void TestGenerationConfigDefaults() {
    VlmGenerationConfig config;
    assert(config.max_tokens == 150);
    assert(config.context_size == 4096);
    assert(config.enable_thinking == false);
    assert(config.history_policy == VlmHistoryPolicy::kLastOnly);
    assert(config.stop_sequences.empty());
    std::cout << "PASS: GenerationConfigDefaults" << std::endl;
}

static void TestChatMessageFactory() {
    auto sys = VlmChatMessage::System("system prompt");
    assert(sys.role == VlmChatMessage::Role::SYSTEM);
    assert(sys.content == "system prompt");

    auto user = VlmChatMessage::User("hello");
    assert(user.role == VlmChatMessage::Role::USER);
    assert(user.content == "hello");

    auto user_img = VlmChatMessage::UserWithImage("describe", "/path/img.jpg");
    assert(user_img.role == VlmChatMessage::Role::USER);
    assert(user_img.image_path == "/path/img.jpg");

    auto asst = VlmChatMessage::Assistant("response");
    assert(asst.role == VlmChatMessage::Role::ASSISTANT);
    std::cout << "PASS: ChatMessageFactory" << std::endl;
}

static void TestServerMetricsUpdate() {
    VlmMetrics metrics;
    ServerVlmModel::UpdateMetricsForRequest(&metrics, 120.0, 30.0, 12, 100.0);
    ServerVlmModel::UpdateMetricsForRequest(&metrics, 80.0, 0.0, 8, 100.0);

    assert(metrics.total_requests == 2);
    assert(metrics.last_latency_ms == 80.0);
    assert(metrics.avg_latency_ms == 100.0);
    assert(metrics.last_ttft_ms == 0.0);
    assert(metrics.last_output_tokens == 8);
    assert(metrics.last_tokens_per_second == 100.0);
    std::cout << "PASS: ServerMetricsUpdate" << std::endl;
}

int main() {
    TestCreateWithEmptyConfig();
    TestCreateFromInvalidPath();
    TestBackendTypeEnum();
    TestGenerationConfigDefaults();
    TestChatMessageFactory();
    TestServerMetricsUpdate();
    std::cout << "\nAll factory tests passed!" << std::endl;
    return 0;
}
