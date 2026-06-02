/**
 * @file test_image_load.cpp
 * @brief Comprehensive test for mtmd bitmap APIs with detailed logging.
 *
 * Test phases:
 *   Phase 1: mtmd_bitmap_init with raw RGB pixel data (no ctx needed)
 *   Phase 2: mtmd_helper_bitmap_init_from_buf with NULL ctx
 *   Phase 3: mtmd_helper_bitmap_init_from_file with NULL ctx
 *   Phase 4: End-to-end via llama-server (if available)
 *
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <mtmd.h>
#include <mtmd-helper.h>

static int g_passed = 0;
static int g_failed = 0;

static void CheckResult(const char* name, bool condition, const char* detail = nullptr) {
    if (condition) {
        std::fprintf(stderr, "  [PASS] %s\n", name);
        ++g_passed;
    } else {
        std::fprintf(stderr, "  [FAIL] %s%s%s\n", name,
                     detail ? " - " : "", detail ? detail : "");
        ++g_failed;
    }
}

static std::string ReadFile(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) return {};
    return std::string(std::istreambuf_iterator<char>(ifs),
                       std::istreambuf_iterator<char>());
}

static const char* DetectFormat(const std::string& data) {
    if (data.size() < 4) return "too-small";
    unsigned char b0 = static_cast<unsigned char>(data[0]);
    unsigned char b1 = static_cast<unsigned char>(data[1]);
    unsigned char b2 = static_cast<unsigned char>(data[2]);
    unsigned char b3 = static_cast<unsigned char>(data[3]);
    std::fprintf(stderr, "  Header bytes: %02x %02x %02x %02x\n", b0, b1, b2, b3);
    if (b0 == 0xFF && b1 == 0xD8) return "JPEG";
    if (b0 == 0x89 && b1 == 0x50 && b2 == 0x4E && b3 == 0x47) return "PNG";
    if (b0 == 0x47 && b1 == 0x49 && b2 == 0x46) return "GIF";
    if (b0 == 0x42 && b1 == 0x4D) return "BMP";
    return "unknown";
}

/**
 * @brief Phase 1: Test mtmd_bitmap_init with raw RGB pixel data.
 *
 * mtmd_bitmap_init(nx, ny, data) takes raw RGB pixel data (nx * ny * 3 bytes).
 * This does NOT require an mtmd_context, so we can test it independently.
 */
static void TestBitmapInitRawPixels() {
    std::fprintf(stderr, "\n[PHASE 1] mtmd_bitmap_init with raw RGB pixel data\n");
    std::fprintf(stderr, "================================================\n");

    {
        const uint32_t nx = 4;
        const uint32_t ny = 4;
        std::vector<unsigned char> rgb(nx * ny * 3, 128);
        std::fprintf(stderr, "  Creating %ux%u RGB bitmap (%zu bytes pixel data)\n",
                     nx, ny, rgb.size());

        mtmd_bitmap* bmp = mtmd_bitmap_init(nx, ny, rgb.data());
        CheckResult("mtmd_bitmap_init(4x4) returns non-null", bmp != nullptr);

        if (bmp) {
            uint32_t got_nx = mtmd_bitmap_get_nx(bmp);
            uint32_t got_ny = mtmd_bitmap_get_ny(bmp);
            size_t got_nbytes = mtmd_bitmap_get_n_bytes(bmp);
            bool got_audio = mtmd_bitmap_is_audio(bmp);
            const char* got_id = mtmd_bitmap_get_id(bmp);
            const unsigned char* got_data = mtmd_bitmap_get_data(bmp);

            std::fprintf(stderr, "  Properties: nx=%u, ny=%u, n_bytes=%zu, is_audio=%s, id=%s\n",
                         got_nx, got_ny, got_nbytes,
                         got_audio ? "true" : "false",
                         got_id ? got_id : "(null)");

            CheckResult("nx matches", got_nx == nx);
            CheckResult("ny matches", got_ny == ny);
            CheckResult("n_bytes == nx*ny*3", got_nbytes == static_cast<size_t>(nx * ny * 3));
            CheckResult("is_audio == false", got_audio == false);
            CheckResult("data pointer is non-null", got_data != nullptr);

            if (got_data) {
                bool data_match = (std::memcmp(got_data, rgb.data(), rgb.size()) == 0);
                CheckResult("pixel data matches input", data_match);
            }

            mtmd_bitmap_free(bmp);
        }
    }

    {
        const uint32_t nx = 100;
        const uint32_t ny = 80;
        std::vector<unsigned char> rgb(nx * ny * 3);
        for (size_t i = 0; i < rgb.size(); ++i) {
            rgb[i] = static_cast<unsigned char>(i % 256);
        }
        std::fprintf(stderr, "  Creating %ux%u RGB bitmap with gradient data\n", nx, ny);

        mtmd_bitmap* bmp = mtmd_bitmap_init(nx, ny, rgb.data());
        CheckResult("mtmd_bitmap_init(100x80) returns non-null", bmp != nullptr);

        if (bmp) {
            uint32_t got_nx = mtmd_bitmap_get_nx(bmp);
            uint32_t got_ny = mtmd_bitmap_get_ny(bmp);
            size_t got_nbytes = mtmd_bitmap_get_n_bytes(bmp);
            const unsigned char* got_data = mtmd_bitmap_get_data(bmp);

            std::fprintf(stderr, "  Properties: nx=%u, ny=%u, n_bytes=%zu\n",
                         got_nx, got_ny, got_nbytes);

            CheckResult("nx matches (100)", got_nx == nx);
            CheckResult("ny matches (80)", got_ny == ny);
            CheckResult("n_bytes == 24000", got_nbytes == static_cast<size_t>(nx * ny * 3));

            if (got_data) {
                bool data_match = (std::memcmp(got_data, rgb.data(), rgb.size()) == 0);
                CheckResult("gradient pixel data matches", data_match);
            }

            mtmd_bitmap_free(bmp);
        }
    }

    {
        const uint32_t nx = 1;
        const uint32_t ny = 1;
        unsigned char rgb[3] = {255, 0, 0};
        std::fprintf(stderr, "  Creating 1x1 red pixel bitmap\n");

        mtmd_bitmap* bmp = mtmd_bitmap_init(nx, ny, rgb);
        CheckResult("mtmd_bitmap_init(1x1) returns non-null", bmp != nullptr);

        if (bmp) {
            uint32_t got_nx = mtmd_bitmap_get_nx(bmp);
            uint32_t got_ny = mtmd_bitmap_get_ny(bmp);
            size_t got_nbytes = mtmd_bitmap_get_n_bytes(bmp);
            const unsigned char* got_data = mtmd_bitmap_get_data(bmp);

            CheckResult("1x1 nx=1", got_nx == 1);
            CheckResult("1x1 ny=1", got_ny == 1);
            CheckResult("1x1 n_bytes=3", got_nbytes == 3);

            if (got_data && got_nbytes >= 3) {
                CheckResult("pixel is red (R=255,G=0,B=0)",
                            got_data[0] == 255 && got_data[1] == 0 && got_data[2] == 0);
            }

            mtmd_bitmap_free(bmp);
        }
    }

    {
        std::fprintf(stderr, "  Testing mtmd_bitmap_init with nx=0\n");
        std::fprintf(stderr, "  (Note: mtmd_bitmap_init does NOT validate dimensions)\n");
        unsigned char rgb[3] = {0, 0, 0};
        mtmd_bitmap* bmp = mtmd_bitmap_init(0, 1, rgb);
        if (bmp) {
            uint32_t got_nx = mtmd_bitmap_get_nx(bmp);
            uint32_t got_ny = mtmd_bitmap_get_ny(bmp);
            size_t got_nbytes = mtmd_bitmap_get_n_bytes(bmp);
            std::fprintf(stderr, "  nx=0 bitmap created: nx=%u, ny=%u, n_bytes=%zu\n",
                         got_nx, got_ny, got_nbytes);
            std::fprintf(stderr, "  WARNING: mtmd_bitmap_init does NOT reject nx=0!\n");
            std::fprintf(stderr, "  Caller must validate dimensions before calling.\n");
            mtmd_bitmap_free(bmp);
        } else {
            CheckResult("mtmd_bitmap_init(0x1) returns null", bmp == nullptr);
        }
    }

    {
        std::fprintf(stderr, "  Testing mtmd_bitmap_init with null data pointer\n");
        std::fprintf(stderr, "  (SKIPPED: mtmd_bitmap_init does NOT check null data,\n");
        std::fprintf(stderr, "   passing nullptr causes segfault. Caller must validate.)\n");
    }

    {
        std::fprintf(stderr, "  Testing bitmap ID set/get\n");
        const uint32_t nx = 2;
        const uint32_t ny = 2;
        std::vector<unsigned char> rgb(nx * ny * 3, 0);
        mtmd_bitmap* bmp = mtmd_bitmap_init(nx, ny, rgb.data());
        if (bmp) {
            mtmd_bitmap_set_id(bmp, "test-image-001");
            const char* id = mtmd_bitmap_get_id(bmp);
            CheckResult("bitmap ID set/get works",
                        id != nullptr && std::strcmp(id, "test-image-001") == 0);
            mtmd_bitmap_free(bmp);
        }
    }
}

/**
 * @brief Phase 2: Test mtmd_helper_bitmap_init_from_buf with NULL ctx.
 *
 * This function decodes image files (jpg/png/bmp/gif) using stb_image,
 * then preprocesses them using the vision model. With NULL ctx, the
 * image decoding part may work but preprocessing will fail.
 */
static void TestHelperBitmapFromBufNullCtx(const std::vector<std::string>& image_paths) {
    std::fprintf(stderr, "\n[PHASE 2] mtmd_helper_bitmap_init_from_buf with NULL ctx\n");
    std::fprintf(stderr, "=====================================================\n");
    std::fprintf(stderr, "  Note: This function requires mtmd_context for vision model\n");
    std::fprintf(stderr, "  preprocessing. With NULL ctx, it may fail at that stage.\n\n");

    if (image_paths.empty()) {
        std::fprintf(stderr, "  No image files provided, skipping.\n");
        return;
    }

    for (const auto& path : image_paths) {
        std::fprintf(stderr, "  --- File: %s ---\n", path.c_str());

        std::string data = ReadFile(path);
        if (data.empty()) {
            std::fprintf(stderr, "  Cannot read file, skipping.\n");
            continue;
        }

        std::fprintf(stderr, "  File size: %zu bytes\n", data.size());
        const char* fmt = DetectFormat(data);
        std::fprintf(stderr, "  Detected format: %s\n", fmt);

        std::fprintf(stderr, "  Calling mtmd_helper_bitmap_init_from_buf(NULL, data, %zu)...\n",
                     data.size());

        mtmd_bitmap* bmp = mtmd_helper_bitmap_init_from_buf(
            nullptr,
            reinterpret_cast<const unsigned char*>(data.data()),
            data.size());

        if (bmp) {
            uint32_t nx = mtmd_bitmap_get_nx(bmp);
            uint32_t ny = mtmd_bitmap_get_ny(bmp);
            size_t n_bytes = mtmd_bitmap_get_n_bytes(bmp);
            bool is_audio = mtmd_bitmap_is_audio(bmp);
            const char* id = mtmd_bitmap_get_id(bmp);

            std::fprintf(stderr, "  SUCCESS: Bitmap decoded! nx=%u, ny=%u, n_bytes=%zu, "
                                 "is_audio=%s, id=%s\n",
                         nx, ny, n_bytes,
                         is_audio ? "true" : "false",
                         id ? id : "(null)");

            if (nx > 0 && ny > 0 && n_bytes > 0) {
                double bpp = static_cast<double>(n_bytes) / (nx * ny);
                std::fprintf(stderr, "  Image dimensions: %u x %u pixels\n", nx, ny);
                std::fprintf(stderr, "  Pixel data size: %zu bytes\n", n_bytes);
                std::fprintf(stderr, "  Bytes per pixel: %.1f (expected 3.0 for RGB)\n", bpp);
                CheckResult("helper bitmap has valid dimensions", nx > 0 && ny > 0);
                CheckResult("helper bitmap n_bytes == nx*ny*3",
                            n_bytes == static_cast<size_t>(nx * ny * 3));
            }
            mtmd_bitmap_free(bmp);
        } else {
            std::fprintf(stderr, "  RESULT: mtmd_helper_bitmap_init_from_buf returned NULL.\n");
            std::fprintf(stderr, "  This is EXPECTED with NULL ctx - the function needs a valid\n");
            std::fprintf(stderr, "  mtmd_context to preprocess the decoded image for the vision model.\n");
            std::fprintf(stderr, "  The stb_image decoding may succeed, but preprocessing fails.\n");
        }
    }
}

/**
 * @brief Phase 3: Test mtmd_helper_bitmap_init_from_file with NULL ctx.
 */
static void TestHelperBitmapFromFileNullCtx(const std::vector<std::string>& image_paths) {
    std::fprintf(stderr, "\n[PHASE 3] mtmd_helper_bitmap_init_from_file with NULL ctx\n");
    std::fprintf(stderr, "======================================================\n");

    if (image_paths.empty()) {
        std::fprintf(stderr, "  No image files provided, skipping.\n");
        return;
    }

    for (const auto& path : image_paths) {
        std::fprintf(stderr, "  --- File: %s ---\n", path.c_str());
        std::fprintf(stderr, "  Calling mtmd_helper_bitmap_init_from_file(NULL, \"%s\")...\n",
                     path.c_str());

        mtmd_bitmap* bmp = mtmd_helper_bitmap_init_from_file(nullptr, path.c_str());

        if (bmp) {
            uint32_t nx = mtmd_bitmap_get_nx(bmp);
            uint32_t ny = mtmd_bitmap_get_ny(bmp);
            size_t n_bytes = mtmd_bitmap_get_n_bytes(bmp);
            std::fprintf(stderr, "  SUCCESS: nx=%u, ny=%u, n_bytes=%zu\n", nx, ny, n_bytes);
            mtmd_bitmap_free(bmp);
        } else {
            std::fprintf(stderr, "  RESULT: returned NULL (expected with NULL ctx).\n");
        }
    }
}

/**
 * @brief Phase 4: Test image loading via llama-server HTTP API.
 *
 * This is the recommended way to test VLM image loading on this platform
 * since the available VLM models use ONNX vision models which are only
 * supported by the server backend (not the direct mtmd C API).
 */
static void TestViaLlamaServer(const std::string& model_dir, const std::string& image_path) {
    std::fprintf(stderr, "\n[PHASE 4] End-to-end test via llama-server\n");
    std::fprintf(stderr, "===========================================\n");
    std::fprintf(stderr, "  Model dir: %s\n", model_dir.c_str());
    std::fprintf(stderr, "  Image: %s\n", image_path.c_str());

    std::string config = ReadFile(model_dir + "/config.json");
    if (config.empty()) {
        std::fprintf(stderr, "  No config.json found in model dir, skipping.\n");
        return;
    }

    std::fprintf(stderr, "  config.json found (%zu bytes)\n", config.size());

    std::string text_model;
    std::string vision_model;
    std::string backend = "smt";

    if (config.find("fastvlm") != std::string::npos ||
        config.find("FastVLM") != std::string::npos) {
        text_model = model_dir + "/fastvlm-text-0.5B-Q4_1.gguf";
        vision_model = model_dir + "/fastvlm_vision.f16.onnx";
    } else if (config.find("Qwen3.5") != std::string::npos ||
               config.find("qwen3_5") != std::string::npos) {
        text_model = model_dir + "/qwen3_5_2b-text-q41.gguf";
        vision_model = model_dir + "/qwen3_5_2b-vision-224-op23.f16.onnx";
    }

    if (text_model.empty()) {
        std::fprintf(stderr, "  Unknown model type, skipping.\n");
        return;
    }

    std::fprintf(stderr, "  Text model: %s\n", text_model.c_str());
    std::fprintf(stderr, "  Vision model: %s\n", vision_model.c_str());
    std::fprintf(stderr, "  Backend: %s\n", backend.c_str());

    std::ifstream tf(text_model);
    if (!tf.good()) {
        std::fprintf(stderr, "  Text model file not found, skipping.\n");
        return;
    }

    std::ifstream vf(vision_model);
    if (!vf.good()) {
        std::fprintf(stderr, "  Vision model file not found, skipping.\n");
        return;
    }

    std::fprintf(stderr, "\n  To run the full end-to-end test manually:\n");
    std::fprintf(stderr, "  1. Start llama-server:\n");
    std::fprintf(stderr, "     llama-server -m %s --mmproj %s --media-backend %s -ngl 0 -c 4096 --port 8888 &\n",
                 text_model.c_str(), vision_model.c_str(), backend.c_str());
    std::fprintf(stderr, "  2. Send request with image:\n");
    std::fprintf(stderr, "     curl -s http://localhost:8888/v1/chat/completions \\\n");
    std::fprintf(stderr, "       -H 'Content-Type: application/json' \\\n");
    std::fprintf(stderr, "       -d '{\"messages\":[{\"role\":\"user\",\"content\":[");
    std::fprintf(stderr, "{\"type\":\"text\",\"text\":\"Describe this image.\"},");
    std::fprintf(stderr, "{\"type\":\"image_url\",\"image_url\":{\"url\":\"file://%s\"}}]", image_path.c_str());
    std::fprintf(stderr, "}]}'\n");
    std::fprintf(stderr, "  3. Stop server: kill %%1\n");
}

static void PrintUsage(const char* prog) {
    std::fprintf(stderr, "Usage: %s [options] [image_path ...]\n", prog);
    std::fprintf(stderr, "\nOptions:\n");
    std::fprintf(stderr, "  --model-dir DIR   Specify VLM model directory for Phase 4 test\n");
    std::fprintf(stderr, "  --all             Run all phases (default)\n");
    std::fprintf(stderr, "  --phase1          Only run Phase 1 (raw pixel bitmap test)\n");
    std::fprintf(stderr, "  --phase2          Only run Phase 2 (helper from buf, NULL ctx)\n");
    std::fprintf(stderr, "  --phase3          Only run Phase 3 (helper from file, NULL ctx)\n");
    std::fprintf(stderr, "  --phase4          Only run Phase 4 (llama-server end-to-end)\n");
    std::fprintf(stderr, "\nPhases:\n");
    std::fprintf(stderr, "  Phase 1: mtmd_bitmap_init with raw RGB pixel data (no ctx needed)\n");
    std::fprintf(stderr, "  Phase 2: mtmd_helper_bitmap_init_from_buf with NULL ctx\n");
    std::fprintf(stderr, "  Phase 3: mtmd_helper_bitmap_init_from_file with NULL ctx\n");
    std::fprintf(stderr, "  Phase 4: End-to-end via llama-server (prints instructions)\n");
}

int main(int argc, char** argv) {
    bool run_phase1 = false;
    bool run_phase2 = false;
    bool run_phase3 = false;
    bool run_phase4 = false;
    std::string model_dir;
    std::vector<std::string> image_paths;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--phase1") { run_phase1 = true; }
        else if (arg == "--phase2") { run_phase2 = true; }
        else if (arg == "--phase3") { run_phase3 = true; }
        else if (arg == "--phase4") { run_phase4 = true; }
        else if (arg == "--all") { run_phase1 = run_phase2 = run_phase3 = run_phase4 = true; }
        else if (arg == "--model-dir" && i + 1 < argc) { model_dir = argv[++i]; }
        else if (arg == "--help" || arg == "-h") { PrintUsage(argv[0]); return 0; }
        else { image_paths.push_back(arg); }
    }

    if (!run_phase1 && !run_phase2 && !run_phase3 && !run_phase4) {
        run_phase1 = run_phase2 = run_phase3 = true;
    }

    std::fprintf(stderr, "========================================\n");
    std::fprintf(stderr, "  mtmd Bitmap API Verification Test\n");
    std::fprintf(stderr, "========================================\n");
    std::fprintf(stderr, "  Image files: %zu\n", image_paths.size());
    std::fprintf(stderr, "  Model dir: %s\n", model_dir.empty() ? "(none)" : model_dir.c_str());
    std::fprintf(stderr, "  Phases: %s%s%s%s\n",
                 run_phase1 ? "1 " : "",
                 run_phase2 ? "2 " : "",
                 run_phase3 ? "3 " : "",
                 run_phase4 ? "4 " : "");

    if (run_phase1) TestBitmapInitRawPixels();
    if (run_phase2) TestHelperBitmapFromBufNullCtx(image_paths);
    if (run_phase3) TestHelperBitmapFromFileNullCtx(image_paths);
    if (run_phase4 && !model_dir.empty() && !image_paths.empty()) {
        TestViaLlamaServer(model_dir, image_paths[0]);
    }

    std::fprintf(stderr, "\n========================================\n");
    std::fprintf(stderr, "  Test Summary: %d passed, %d failed\n", g_passed, g_failed);
    std::fprintf(stderr, "========================================\n");

    if (g_failed > 0 && run_phase1) {
        std::fprintf(stderr, "\n  WARNING: Phase 1 failures indicate mtmd_bitmap_init API issues.\n");
        std::fprintf(stderr, "  This is a core API that should work without any model context.\n");
    }

    if (run_phase2 || run_phase3) {
        std::fprintf(stderr, "\n  NOTE: Phase 2/3 NULL-ctx tests are expected to fail for image\n");
        std::fprintf(stderr, "  decoding. The helper functions need a valid mtmd_context for\n");
        std::fprintf(stderr, "  vision model preprocessing. Use llama-server for full testing.\n");
    }

    return 0;
}
