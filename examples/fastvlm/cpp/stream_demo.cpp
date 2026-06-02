/**
 * @file stream_demo.cpp
 * @brief Streaming generation demo for the VLM service.
 *
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdio>
#include <iostream>
#include <memory>
#include <string>

#include "vlm/vlm_service.h"

namespace {

std::string GetArg(int argc, char** argv, const std::string& name,
                   const std::string& fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) return argv[i + 1];
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

    std::printf("Streaming response:\n");

    vlm::VlmResult result;
    if (!service->GenerateStream(input,
            [](const std::string& chunk, bool is_done, const std::string& err) -> bool {
                if (!err.empty()) {
                    std::fprintf(stderr, "\nStream error: %s\n", err.c_str());
                    return false;
                }
                if (!chunk.empty()) {
                    std::printf("%s", chunk.c_str());
                    std::fflush(stdout);
                }
                return true;
            }, &result, &error)) {
        std::cerr << "\nStream failed: " << error << std::endl;
        return 1;
    }

    std::printf("\n\n--- Stats ---\n");
    std::printf("Latency: %.1f ms\n", result.latency_ms);
    std::printf("Tokens/s: %.1f\n", result.tokens_per_second);
    return 0;
}
