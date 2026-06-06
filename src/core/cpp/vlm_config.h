/**
 * @file vlm_config.h
 * @brief Configuration parsing and validation for the VLM service.
 *
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef VLM_CONFIG_H
#define VLM_CONFIG_H

#include <string>

#include "vlm_service.h"

namespace vlm {

/**
 * @brief Load VLM model configuration from a YAML file.
 * @param config_path Path to the YAML configuration file.
 * @param config Receives the parsed model configuration.
 * @param error Receives error description on failure.
 * @return True on success.
 * @note Uses yaml-cpp for YAML parsing.
 */
bool LoadVlmConfigFromYaml(const std::string& config_path,
                            VlmModelConfig* config,
                            std::string* error);

/**
 * @brief Load VLM model configuration from a model directory manifest.
 * @param model_dir Path to the model directory containing config.json.
 * @param config Receives the parsed model configuration.
 * @param error Receives error description on failure.
 * @return True on success.
 */
bool LoadVlmManifest(const std::string& model_dir,
                    VlmModelConfig* config,
                    std::string* error);

/**
 * @brief Validate and apply defaults to a VLM model configuration.
 * @param config The configuration to validate (modified in place).
 * @param error Receives error description on failure.
 * @return True if the configuration is valid.
 */
bool ValidateVlmModelConfig(VlmModelConfig* config, std::string* error);

}  // namespace vlm

#endif  // VLM_CONFIG_H
