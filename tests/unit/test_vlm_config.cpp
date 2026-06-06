/**
 * @file test_vlm_config.cpp
 * @brief Unit tests for VLM configuration parsing and validation.
 *
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>

#include "vlm_service.h"
#include "vlm_config.h"

using vlm::BackendType;
using vlm::LoadVlmConfigFromYaml;
using vlm::ValidateVlmModelConfig;
using vlm::VlmGenerationConfig;
using vlm::VlmHistoryPolicy;
using vlm::VlmMediaBackend;
using vlm::VlmModelConfig;

static std::string WriteTempYaml(const std::string& content) {
    char path[] = "/tmp/vlm_test_XXXXXX";
    int fd = mkstemp(path);
    write(fd, content.c_str(), content.size());
    close(fd);
    return std::string(path);
}

static std::string MakeTempDir() {
    char path[] = "/tmp/vlm_manifest_test_XXXXXX";
    char* dir = mkdtemp(path);
    assert(dir != nullptr);
    return std::string(dir);
}

static void WriteFile(const std::string& path, const std::string& content) {
    std::ofstream out(path);
    out << content;
}

static void TestLoadBasicConfig() {
    std::string yaml = "backend: server\nmodel_dir: /tmp/test_model\nhost: 127.0.0.1\nport: 8071\n";
    std::string path = WriteTempYaml(yaml);
    VlmModelConfig config;
    std::string error;
    bool ok = LoadVlmConfigFromYaml(path, &config, &error);
    std::remove(path.c_str());
    assert(ok);
    assert(config.backend_type == BackendType::kServerCompat);
    assert(config.model_dir == "/tmp/test_model");
    assert(config.host == "127.0.0.1");
    assert(config.port == 8071);
    std::cout << "PASS: LoadBasicConfig" << std::endl;
}

static void TestLoadGenerationConfig() {
    std::string yaml =
        "backend: server\nmodel_dir: /tmp/test\n"
        "generation:\n  max_tokens: 200\n  temperature: 0.5\n  top_p: 0.9\n  enable_thinking: true\n  reasoning_budget: 1024\n  history_policy: keep_all\n  stop_sequences: \"<|im_end|>,<|end|>\"\n";
    std::string path = WriteTempYaml(yaml);
    VlmModelConfig config;
    std::string error;
    bool ok = LoadVlmConfigFromYaml(path, &config, &error);
    std::remove(path.c_str());
    assert(ok);
    assert(config.generation.max_tokens == 200);
    assert(config.generation.temperature == 0.5F);
    assert(config.generation.enable_thinking == true);
    assert(config.generation.reasoning_budget == 1024);
    assert(config.generation.history_policy == VlmHistoryPolicy::kKeepAll);
    assert(config.generation.stop_sequences.size() == 2);
    std::cout << "PASS: LoadGenerationConfig" << std::endl;
}

static void TestValidateRejectsEmptyModelDir() {
    VlmModelConfig config;
    config.model_dir = "";
    std::string error;
    bool ok = ValidateVlmModelConfig(&config, &error);
    assert(!ok);
    std::cout << "PASS: ValidateRejectsEmptyModelDir" << std::endl;
}

static void TestHistoryPolicyParsing() {
    std::string yaml_last =
        "backend: server\nmodel_dir: /tmp\n"
        "generation:\n  history_policy: last\n";
    std::string yaml_none =
        "backend: server\nmodel_dir: /tmp\n"
        "generation:\n  history_policy: none\n";
    std::string yaml_keep =
        "backend: server\nmodel_dir: /tmp\n"
        "generation:\n  history_policy: keep_all\n";

    struct TestCase { std::string yaml; VlmHistoryPolicy expected; };
    TestCase cases[] = {
        {yaml_last, VlmHistoryPolicy::kLastOnly},
        {yaml_none, VlmHistoryPolicy::kNone},
        {yaml_keep, VlmHistoryPolicy::kKeepAll},
    };

    for (const auto& tc : cases) {
        std::string path = WriteTempYaml(tc.yaml);
        VlmModelConfig config;
        std::string error;
        bool ok = LoadVlmConfigFromYaml(path, &config, &error);
        std::remove(path.c_str());
        assert(ok);
        assert(config.generation.history_policy == tc.expected);
    }
    std::cout << "PASS: HistoryPolicyParsing" << std::endl;
}

static void TestYamlCppConfigForms() {
    std::string yaml =
        "backend_type: server_compat\n"
        "model_dir: /tmp/test_model\n"
        "vision_backend: mtmd\n"
        "vision_model_paths:\n"
        "  - vision-a.onnx\n"
        "  - \"vision-b.onnx\"\n"
        "generation:\n"
        "  max_tokens: 256\n"
        "  top_k: 32\n"
        "  stop_sequences: [\"<|im_end|>\", \"<|end|>\"]\n";
    std::string path = WriteTempYaml(yaml);
    VlmModelConfig config;
    std::string error;
    bool ok = LoadVlmConfigFromYaml(path, &config, &error);
    std::remove(path.c_str());
    assert(ok);
    assert(config.backend_type == BackendType::kServerCompat);
    assert(config.media_backend == VlmMediaBackend::kMtmd);
    assert(config.vision_model_paths.size() == 2);
    assert(config.vision_model_paths[0] == "vision-a.onnx");
    assert(config.vision_model_paths[1] == "vision-b.onnx");
    assert(config.generation.max_tokens == 256);
    assert(config.generation.top_k == 32);
    assert(config.generation.stop_sequences.size() == 2);
    assert(config.generation.stop_sequences[0] == "<|im_end|>");
    assert(config.generation.stop_sequences[1] == "<|end|>");
    std::cout << "PASS: YamlCppConfigForms" << std::endl;
}

static void TestLoadManifestNestedJson() {
    std::string model_dir = MakeTempDir();
    WriteFile(model_dir + "/config.json",
        "{\n"
        "  \"model_name\": \"fastvlm-test\",\n"
        "  \"architectures\": [\"FastVLM\"],\n"
        "  \"text_model\": {\"model_path\": \"./text.gguf\"},\n"
        "  \"vision_model\": {\"model_path\": \"vision.onnx\"},\n"
        "  \"host\": \"0.0.0.0\",\n"
        "  \"port\": 8099,\n"
        "  \"backend_type\": \"server\",\n"
        "  \"generation\": {\"max_tokens\": 333, \"context_size\": 2048}\n"
        "}\n");

    VlmModelConfig config;
    std::string error;
    bool ok = LoadVlmManifest(model_dir, &config, &error);
    std::remove((model_dir + "/config.json").c_str());
    rmdir(model_dir.c_str());
    assert(ok);
    assert(config.model_name == "fastvlm-test");
    assert(config.architecture == "FastVLM");
    assert(config.text_model_path == model_dir + "/text.gguf");
    assert(config.vision_model_paths.size() == 1);
    assert(config.vision_model_paths[0] == model_dir + "/vision.onnx");
    assert(config.host == "0.0.0.0");
    assert(config.port == 8099);
    assert(config.backend_type == BackendType::kServerCompat);
    assert(config.generation.max_tokens == 333);
    assert(config.generation.context_size == 2048);
    std::cout << "PASS: LoadManifestNestedJson" << std::endl;
}

int main() {
    TestLoadBasicConfig();
    TestLoadGenerationConfig();
    TestValidateRejectsEmptyModelDir();
    TestHistoryPolicyParsing();
    TestYamlCppConfigForms();
    TestLoadManifestNestedJson();
    std::cout << "\nAll config tests passed!" << std::endl;
    return 0;
}
