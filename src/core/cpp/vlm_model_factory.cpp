/**
 * @file vlm_model_factory.cpp
 * @brief Factory methods for creating VLM service instances.
 *
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "vlm_service.h"

#include <memory>
#include <string>

#include "llama_vlm_model.h"
#include "server_vlm_model.h"
#include "vlm_config.h"
#include "vlm_utils.h"

namespace vlm {

std::unique_ptr<VlmService> CreateVlmService(const VlmModelConfig& config,
                                             std::string* error) {
    VlmModelConfig validated = config;
    if (!ValidateVlmModelConfig(&validated, error)) {
        return nullptr;
    }
    if (validated.backend_type == BackendType::kServerCompat) {
        auto service = std::make_unique<ServerVlmModel>();
        if (!service->Initialize(validated, error)) {
            return nullptr;
        }
        return service;
    }
    auto service = std::make_unique<LlamaVlmModel>();
    if (!service->Initialize(validated, error)) {
        return nullptr;
    }
    return service;
}

std::unique_ptr<VlmService> CreateVlmServiceFromConfig(
    const std::string& config_path,
    std::string* error) {
    VlmModelConfig config;
    if (!LoadVlmConfigFromYaml(config_path, &config, error)) {
        return nullptr;
    }
    if (!config.model_dir.empty()) {
        LoadVlmManifest(config.model_dir, &config, error);
    }
    if (!ValidateVlmModelConfig(&config, error)) {
        return nullptr;
    }
    return CreateVlmService(config, error);
}

}  // namespace vlm
