/**
 * @file vlm_utils.h
 * @brief Shared utility functions for the VLM service.
 *
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef VLM_SRC_CORE_CPP_VLM_UTILS_H_
#define VLM_SRC_CORE_CPP_VLM_UTILS_H_

#include <cstdint>
#include <string>
#include <vector>

namespace vlm {

void SetError(std::string* error, const std::string& message);

std::string ShellQuote(const std::string& value);

std::string JsonEscape(const std::string& value);

std::string Base64Encode(const std::vector<unsigned char>& data);

std::string ReadFileToString(const std::string& path, std::string* error);

std::string ReadBinaryFile(const std::string& path, std::string* error);

std::string ExtractJsonString(const std::string& json,
                              const std::string& key);

int ExtractJsonInt(const std::string& json, const std::string& key);

std::string StripThinkingBlocks(const std::string& text,
                                std::string* thinking_content);

}  // namespace vlm

#endif  // VLM_SRC_CORE_CPP_VLM_UTILS_H_
