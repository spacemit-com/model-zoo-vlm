/**
 * @file fastvlm_demo.cpp
 * @brief Basic VLM generation demo.
 *
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>
#include <memory>
#include <string>

#include "vlm_service.h"

namespace {

std::string GetArg(int argc, char** argv, const std::string& name,
                   const std::string& fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return argv[i + 1];
        }
    }
    return fallback;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string config_path = GetArg(argc, argv, "--config",
        "../examples/fastvlm/config/fastvlm.yaml");
    const std::string image_path = GetArg(argc, argv, "--image", "");
    const std::string prompt = GetArg(argc, argv, "--prompt", "Describe this image");

    std::string error;
    auto service = vlm::CreateVlmServiceFromConfig(config_path, &error);
    if (!service) {
        std::cerr << "Failed to initialize VLM: " << error << std::endl;
        return 1;
    }

    vlm::VlmInput input;
    input.image_path = image_path;
    input.prompt = prompt;

    vlm::VlmResult result;
    if (!service->Generate(input, &result, &error)) {
        std::cerr << "Failed to generate: " << error << std::endl;
        return 1;
    }

    std::cout << "Description: " << result.text << std::endl;
    if (!result.reasoning_content.empty()) {
        std::cout << "Reasoning: " << result.reasoning_content << std::endl;
    }
    std::cout << "Latency: " << result.latency_ms << " ms" << std::endl;
    std::cout << "Tokens/s: " << result.tokens_per_second << std::endl;
    return 0;
}
