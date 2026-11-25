/**
 * @file test_mtmd_image.cpp
 * @brief Image input and ownership regressions for both mtmd helper APIs.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "mtmd-helper.h"
#include "mtmd_image.h"

namespace {

enum class DecodeResult { kImage, kFailure, kVideo, kVideoAndImage };

DecodeResult result_kind = DecodeResult::kImage;
int decode_count = 0;
int bitmap_free_count = 0;
int video_free_count = 0;
mtmd_context context;
const std::vector<uint8_t> image_data = {0x89, 0x50, 0x4e, 0x47};

void Check(bool condition, const char* message) {
    // Keep checks active in Release builds, where assert() is disabled.
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(EXIT_FAILURE);
    }
}

void Reset(DecodeResult result = DecodeResult::kImage) {
    result_kind = result;
    decode_count = 0;
    bitmap_free_count = 0;
    video_free_count = 0;
}

void CheckDecodeInput(mtmd_context* ctx, const unsigned char* data, size_t size) {
    Check(ctx == &context, "decoder context");
    Check(std::vector<uint8_t>(data, data + size) == image_data, "decoder input bytes");
    ++decode_count;
}

std::string WriteInput(const std::vector<uint8_t>& data) {
    char path[] = "/tmp/vlm_image_test_XXXXXX";
    int fd = mkstemp(path);
    Check(fd >= 0, "create input file");
    Check(write(fd, data.data(), data.size()) == static_cast<ssize_t>(data.size()),
            "write input file");
    Check(close(fd) == 0, "close input file");
    return path;
}

void TestMemoryInput() {
    Reset();
    std::string error;
    {
        auto bitmap = vlm::LoadBitmapFromInput(&context, "", image_data, &error);
        Check(bitmap != nullptr && error.empty(), "memory image decoded");
        Check(decode_count == 1 && bitmap_free_count == 0, "transfer bitmap ownership");
    }
    Check(bitmap_free_count == 1 && video_free_count == 0, "release bitmap exactly once");
}

void TestFileInput() {
    Reset();
    std::string error;
    const std::string path = WriteInput(image_data);
    {
        // A provided file keeps precedence over the memory input.
        auto bitmap = vlm::LoadBitmapFromInput(&context, path, {1, 2, 3, 4}, &error);
        Check(bitmap != nullptr && error.empty(), "file image decoded");
    }
    Check(std::remove(path.c_str()) == 0, "remove input file");
    Check(decode_count == 1 && bitmap_free_count == 1, "file ownership unchanged");
}

void TestInvalidInputs() {
    Reset();
    std::string error;
    Check(!vlm::LoadBitmapFromInput(nullptr, "", image_data, &error), "reject null context");
    Check(!error.empty(), "null context error");
    error.clear();
    Check(!vlm::LoadBitmapFromInput(&context, "", {1, 2, 3}, &error), "reject short buffer");
    Check(!error.empty(), "short buffer error");
    const std::string path = WriteInput({1, 2, 3});
    error.clear();
    Check(!vlm::LoadBitmapFromInput(&context, path, {}, &error), "reject short file");
    Check(!error.empty(), "short file error");
    Check(std::remove(path.c_str()) == 0, "remove short input");
    error.clear();
    Check(!vlm::LoadBitmapFromInput(&context, path, {}, &error), "reject missing file");
    Check(!error.empty(), "missing file error");
    error.clear();
    Check(!vlm::LoadBitmapFromInput(&context, "", {}, &error), "empty input returns null");
    Check(decode_count == 0 && bitmap_free_count == 0, "invalid input skips decoder");
}

void TestDecodeFailure() {
    Reset(DecodeResult::kFailure);
    std::string error;
    Check(!vlm::LoadBitmapFromInput(&context, "", image_data, &error), "decoder failure");
    Check(!error.empty(), "decoder failure error");
    Check(decode_count == 1 && bitmap_free_count == 0, "no bitmap on failure");
}

#ifndef VLM_TEST_LEGACY_MTMD
void TestVideoCleanup() {
    for (auto result : {DecodeResult::kVideo, DecodeResult::kVideoAndImage}) {
        Reset(result);
        std::string error;
        Check(!vlm::LoadBitmapFromInput(&context, "", image_data, &error), "reject video");
        Check(!error.empty(), "video error");
        Check(video_free_count == 1, "release video exactly once");
        Check(bitmap_free_count == (result == DecodeResult::kVideoAndImage ? 1 : 0),
                "release any accompanying bitmap");
    }
}
#endif

}  // namespace

void mtmd_bitmap_free(mtmd_bitmap* bitmap) {
    if (bitmap) {
        ++bitmap_free_count;
        delete bitmap;
    }
}

uint32_t mtmd_bitmap_get_nx(const mtmd_bitmap*) { return 1; }
uint32_t mtmd_bitmap_get_ny(const mtmd_bitmap*) { return 1; }
size_t mtmd_bitmap_get_n_bytes(const mtmd_bitmap*) { return 3; }
bool mtmd_bitmap_is_audio(const mtmd_bitmap*) { return false; }
const char* mtmd_bitmap_get_id(const mtmd_bitmap*) { return "image"; }

#ifdef VLM_TEST_LEGACY_MTMD
mtmd_bitmap* mtmd_helper_bitmap_init_from_buf(
    mtmd_context* ctx, const unsigned char* data, size_t size) {
    CheckDecodeInput(ctx, data, size);
    return result_kind == DecodeResult::kFailure ? nullptr : new mtmd_bitmap;
}
#else
mtmd_helper_bitmap_wrapper mtmd_helper_bitmap_init_from_buf(
    mtmd_context* ctx, const unsigned char* data, size_t size, bool placeholder) {
    CheckDecodeInput(ctx, data, size);
    Check(!placeholder, "decode pixels instead of returning a placeholder");
    switch (result_kind) {
        case DecodeResult::kImage: return {new mtmd_bitmap, nullptr};
        case DecodeResult::kFailure: return {nullptr, nullptr};
        case DecodeResult::kVideo: return {nullptr, new mtmd_helper_video};
        case DecodeResult::kVideoAndImage: return {new mtmd_bitmap, new mtmd_helper_video};
    }
    std::abort();
}

void mtmd_helper_video_free(mtmd_helper_video* video) {
    if (video) {
        if (result_kind == DecodeResult::kVideoAndImage) {
            Check(bitmap_free_count == 1, "release lazy bitmap before its video context");
        }
        ++video_free_count;
        delete video;
    }
}
#endif

int main() {
    TestMemoryInput();
    TestFileInput();
    TestInvalidInputs();
    TestDecodeFailure();
#ifndef VLM_TEST_LEGACY_MTMD
    TestVideoCleanup();
#endif
    std::puts("PASS: mtmd image API and ownership");
    return EXIT_SUCCESS;
}
