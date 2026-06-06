/**
 * @file vlm_config.cpp
 * @brief Implementation of configuration parsing and validation for the VLM service.
 *
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "vlm_config.h"
#include "vlm_utils.h"

#include <algorithm>
#include <cstdlib>
#include <sstream>

#include <nlohmann/json.hpp>
#include <yaml-cpp/yaml.h>

namespace vlm {

namespace {

/**
 * @brief Trim leading and trailing whitespace from a string.
 * @param s The string to trim.
 * @return The trimmed string.
 */
std::string Trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return {};
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

/**
 * @brief Split a string by a delimiter, trimming each part.
 * @param s The string to split.
 * @param delim The delimiter character.
 * @return Vector of trimmed tokens.
 */
std::vector<std::string> SplitAndTrim(const std::string& s, char delim) {
    std::vector<std::string> tokens;
    std::istringstream iss(s);
    std::string token;
    while (std::getline(iss, token, delim)) {
        std::string trimmed = Trim(token);
        if (!trimmed.empty()) {
            tokens.push_back(trimmed);
        }
    }
    return tokens;
}

/**
 * @brief Map a history policy string to the enum value.
 * @param policy_str The policy string from config.
 * @return The corresponding VlmHistoryPolicy enum value.
 */
VlmHistoryPolicy ParseHistoryPolicy(const std::string& policy_str) {
    std::string lower = policy_str;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    if (lower == "keep_all") return VlmHistoryPolicy::kKeepAll;
    if (lower == "last") return VlmHistoryPolicy::kLastOnly;
    if (lower == "none") return VlmHistoryPolicy::kNone;
    return VlmHistoryPolicy::kKeepAll;
}

/**
 * @brief Map a backend type string to the enum value.
 * @param backend_str The backend string from config.
 * @return The corresponding BackendType enum value.
 */
BackendType ParseBackendType(const std::string& backend_str) {
    std::string lower = backend_str;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    if (lower == "server_compat" || lower == "servercompat" ||
        lower == "server") {
        return BackendType::kServerCompat;
    }
    return BackendType::kDirect;
}

VlmMediaBackend ParseMediaBackend(const std::string& backend_str) {
    std::string lower = backend_str;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    if (lower == "smt") return VlmMediaBackend::kSmt;
    if (lower == "mtmd") return VlmMediaBackend::kMtmd;
    return VlmMediaBackend::kAuto;
}

std::string ExpandUserPath(const std::string& path) {
    if (path == "~" || path.rfind("~/", 0) == 0) {
        const char* home = std::getenv("HOME");
        if (home && home[0] != '\0') {
            if (path == "~") {
                return std::string(home);
            }
            return std::string(home) + path.substr(1);
        }
    }
    return path;
}

std::string JoinModelPath(const std::string& model_dir,
                            const std::string& model_path) {
    if (model_path.empty() || model_path[0] == '/') {
        return ExpandUserPath(model_path);
    }

    std::string base = ExpandUserPath(model_dir);
    if (!base.empty() && base.back() != '/') {
        base += '/';
    }
    if (model_path.rfind("./", 0) == 0) {
        return base + model_path.substr(2);
    }
    return base + model_path;
}

template <typename T>
void AssignYamlScalar(const YAML::Node& root,
                        const char* key,
                        T* target) {
    if (!root[key] || !target) {
        return;
    }
    try {
        *target = root[key].as<T>();
    } catch (const YAML::Exception&) {
    }
}

void AssignYamlStringPath(const YAML::Node& root,
                            const char* key,
                            std::string* target) {
    if (!root[key] || !target) {
        return;
    }
    try {
        *target = ExpandUserPath(root[key].as<std::string>());
    } catch (const YAML::Exception&) {
    }
}

std::vector<std::string> ParseYamlStringList(const YAML::Node& node) {
    std::vector<std::string> values;
    if (!node) {
        return values;
    }
    try {
        if (node.IsSequence()) {
            for (const auto& item : node) {
                values.push_back(item.as<std::string>());
            }
        } else if (node.IsScalar()) {
            values = SplitAndTrim(node.as<std::string>(), ',');
        }
    } catch (const YAML::Exception&) {
        values.clear();
    }
    return values;
}

std::string JsonStringAt(const nlohmann::json& json, const char* key) {
    auto iter = json.find(key);
    if (iter != json.end() && iter->is_string()) {
        return iter->get<std::string>();
    }
    return {};
}

int JsonIntAt(const nlohmann::json& json, const char* key, int default_value) {
    auto iter = json.find(key);
    if (iter != json.end() && iter->is_number_integer()) {
        return iter->get<int>();
    }
    return default_value;
}

}  // namespace

bool LoadVlmConfigFromYaml(const std::string& config_path,
                            VlmModelConfig* config,
                            std::string* error) {
    if (!config) {
        SetError(error, "Config pointer is null");
        return false;
    }

    YAML::Node root;
    try {
        root = YAML::LoadFile(config_path);
    } catch (const YAML::Exception& exception) {
        SetError(error, std::string("Failed to parse YAML config: ") +
                            exception.what());
        return false;
    }

    AssignYamlStringPath(root, "model_dir", &config->model_dir);
    AssignYamlScalar(root, "model_name", &config->model_name);
    AssignYamlStringPath(root, "text_model_path", &config->text_model_path);
    AssignYamlScalar(root, "architecture", &config->architecture);
    AssignYamlStringPath(root, "smt_config_dir", &config->smt_config_dir);
    AssignYamlStringPath(root, "media_path", &config->media_path);
    AssignYamlScalar(root, "cpu_affinity", &config->cpu_affinity);
    AssignYamlScalar(root, "host", &config->host);
    AssignYamlScalar(root, "port", &config->port);
    AssignYamlScalar(root, "no_warmup", &config->no_warmup);

    if (root["n_gpu_layers"]) {
        AssignYamlScalar(root, "n_gpu_layers", &config->n_gpu_layers);
    } else {
        AssignYamlScalar(root, "ngl", &config->n_gpu_layers);
    }

    if (root["backend_type"] || root["backend"]) {
        const auto& node = root["backend_type"] ? root["backend_type"] : root["backend"];
        config->backend_type = ParseBackendType(node.as<std::string>());
    }
    if (root["media_backend"] || root["vision_backend"]) {
        const auto& node = root["media_backend"] ? root["media_backend"] :
                                            root["vision_backend"];
        config->media_backend = ParseMediaBackend(node.as<std::string>());
    }

    auto vision_paths = ParseYamlStringList(root["vision_model_paths"]);
    if (!vision_paths.empty()) {
        config->vision_model_paths = vision_paths;
    }

    const YAML::Node generation = root["generation"];
    if (generation) {
        AssignYamlScalar(generation, "max_tokens", &config->generation.max_tokens);
        AssignYamlScalar(generation, "context_size", &config->generation.context_size);
        AssignYamlScalar(generation, "top_k", &config->generation.top_k);
        AssignYamlScalar(generation, "top_p", &config->generation.top_p);
        AssignYamlScalar(generation, "temperature", &config->generation.temperature);
        AssignYamlScalar(generation, "repeat_penalty", &config->generation.repeat_penalty);
        AssignYamlScalar(generation, "threads", &config->generation.threads);
        AssignYamlScalar(generation, "batch_threads", &config->generation.batch_threads);
        AssignYamlScalar(generation, "enable_thinking", &config->generation.enable_thinking);
        AssignYamlScalar(generation, "reasoning_budget", &config->generation.reasoning_budget);

        if (generation["history_policy"]) {
            config->generation.history_policy = ParseHistoryPolicy(
                generation["history_policy"].as<std::string>());
        }

        auto stop_sequences = ParseYamlStringList(generation["stop_sequences"]);
        if (!stop_sequences.empty()) {
            config->generation.stop_sequences = stop_sequences;
        }
    }

    return true;
}

bool LoadVlmManifest(const std::string& model_dir,
                    VlmModelConfig* config,
                    std::string* error) {
    if (!config) {
        SetError(error, "Config pointer is null");
        return false;
    }

    std::string expanded_model_dir = ExpandUserPath(model_dir);
    std::string manifest_path = expanded_model_dir;
    if (!manifest_path.empty() && manifest_path.back() != '/') {
        manifest_path += '/';
    }
    manifest_path += "config.json";

    std::string content = ReadFileToString(manifest_path, error);
    if (content.empty()) {
        return false;
    }

    nlohmann::json manifest;
    try {
        manifest = nlohmann::json::parse(content);
    } catch (const nlohmann::json::exception& exception) {
        SetError(error, std::string("Failed to parse model manifest: ") +
                            exception.what());
        return false;
    }

    std::string name = JsonStringAt(manifest, "model_name");
    if (!name.empty()) {
        config->model_name = name;
    }

    std::string arch = JsonStringAt(manifest, "architecture");
    if (arch.empty() && manifest.contains("architectures")) {
        const auto& architectures = manifest["architectures"];
        if (architectures.is_array() && !architectures.empty() &&
            architectures[0].is_string()) {
            arch = architectures[0].get<std::string>();
        } else if (architectures.is_string()) {
            arch = architectures.get<std::string>();
        }
    }
    if (!arch.empty()) {
        config->architecture = arch;
    }

    std::string text_path = JsonStringAt(manifest, "text_model_path");
    if (text_path.empty() && manifest.contains("text_model") &&
        manifest["text_model"].is_object()) {
        text_path = JsonStringAt(manifest["text_model"], "model_path");
    }
    if (!text_path.empty()) {
        config->text_model_path = JoinModelPath(expanded_model_dir, text_path);
    }

    std::string vision_path = JsonStringAt(manifest, "vision_model_path");
    if (vision_path.empty() && manifest.contains("vision_model") &&
        manifest["vision_model"].is_object()) {
        vision_path = JsonStringAt(manifest["vision_model"], "model_path");
    }
    if (!vision_path.empty()) {
        if (config->vision_model_paths.empty()) {
            config->vision_model_paths.push_back(
                JoinModelPath(expanded_model_dir, vision_path));
        }
    }

    int port = JsonIntAt(manifest, "port", -1);
    if (port > 0) {
        config->port = port;
    }

    std::string host = JsonStringAt(manifest, "host");
    if (!host.empty()) {
        config->host = host;
    }

    std::string backend = JsonStringAt(manifest, "backend_type");
    if (!backend.empty()) {
        config->backend_type = ParseBackendType(backend);
    }

    const nlohmann::json& generation = manifest.contains("generation") &&
        manifest["generation"].is_object() ? manifest["generation"] : manifest;

    int max_tokens = JsonIntAt(generation, "max_tokens", -1);
    if (max_tokens > 0) {
        config->generation.max_tokens = max_tokens;
    }

    int context_size = JsonIntAt(generation, "context_size", -1);
    if (context_size > 0) {
        config->generation.context_size = context_size;
    }

    return true;
}

bool ValidateVlmModelConfig(VlmModelConfig* config, std::string* error) {
    if (!config) {
        SetError(error, "Config pointer is null");
        return false;
    }

    if (config->model_dir.empty()) {
        SetError(error, "model_dir is required");
        return false;
    }

    if (config->model_name.empty()) {
        if (!config->model_dir.empty()) {
            std::string dir = config->model_dir;
            if (!dir.empty() && dir.back() == '/') {
                dir.pop_back();
            }
            size_t pos = dir.find_last_of('/');
            config->model_name = (pos != std::string::npos) ? dir.substr(pos + 1) : dir;
        }
    }

    if (config->text_model_path.empty() && !config->model_dir.empty()) {
        std::string manifest_err;
        LoadVlmManifest(config->model_dir, config, &manifest_err);
    }

    if (config->generation.max_tokens <= 0) {
        config->generation.max_tokens = 512;
    }

    if (config->generation.context_size <= 0) {
        config->generation.context_size = 4096;
    }

    if (config->generation.top_k <= 0) {
        config->generation.top_k = 40;
    }

    if (config->generation.top_p <= 0.0 || config->generation.top_p > 1.0) {
        config->generation.top_p = 0.9;
    }

    if (config->generation.temperature < 0.0) {
        config->generation.temperature = 0.7;
    }

    if (config->generation.repeat_penalty < 1.0) {
        config->generation.repeat_penalty = 1.1;
    }

    if (config->generation.threads <= 0) {
        config->generation.threads = 4;
    }

    if (config->generation.batch_threads <= 0) {
        config->generation.batch_threads = config->generation.threads;
    }

    if (config->generation.reasoning_budget <= 0) {
        config->generation.reasoning_budget = 1024;
    }

    if (config->port <= 0 || config->port > 65535) {
        config->port = 8080;
    }

    if (config->host.empty()) {
        config->host = "127.0.0.1";
    }

    if (config->backend_type == BackendType::kServerCompat) {
        if (config->smt_config_dir.empty() && !config->model_dir.empty()) {
            config->smt_config_dir = config->model_dir;
        }

        if (config->cpu_affinity.empty()) {
            config->cpu_affinity = "0-7";
        }
    }

    return true;
}

}  // namespace vlm
