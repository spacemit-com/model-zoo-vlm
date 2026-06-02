/**
 * @file mtmd_image.cpp
 * @brief Implementation of image loading utilities for mtmd with detailed logging.
 *
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mtmd_image.h"

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <mtmd.h>
#include <mtmd-helper.h>

#include "vlm_utils.h"

namespace vlm {

namespace {

void LogInfo(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    std::fprintf(stderr, "[VLM-IMAGE] ");
    std::vfprintf(stderr, fmt, args);
    std::fprintf(stderr, "\n");
    va_end(args);
}

}  // namespace

MtmdBitmapPtr MakeNullBitmap() {
    return MtmdBitmapPtr(nullptr, mtmd_bitmap_free);
}

MtmdBitmapPtr LoadBitmapFromInput(mtmd_context* ctx,
                                  const std::string& image_path,
                                  const std::vector<uint8_t>& image_bytes,
                                  std::string* error) {
    if (!ctx) {
        LogInfo("WARNING: mtmd_context is null; image decoding may succeed but "
                "subsequent tokenization/evaluation requires a valid context");
        SetError(error, "mtmd_context is null, cannot process image");
        return MakeNullBitmap();
    }

    if (!image_path.empty()) {
        LogInfo("Loading image from file: %s", image_path.c_str());

        std::string file_data = ReadBinaryFile(image_path, error);
        if (file_data.empty()) {
            LogInfo("Failed to read image file: %s (error: %s)",
                    image_path.c_str(), error ? error->c_str() : "unknown");
            SetError(error, "Failed to read image file: " + image_path);
            return MakeNullBitmap();
        }
        LogInfo("Read %zu bytes from file: %s", file_data.size(), image_path.c_str());

        if (file_data.size() < 4) {
            LogInfo("File too small (%zu bytes), not a valid image", file_data.size());
            SetError(error, "File too small to be a valid image: " + image_path);
            return MakeNullBitmap();
        }

        unsigned char b0 = static_cast<unsigned char>(file_data[0]);
        unsigned char b1 = static_cast<unsigned char>(file_data[1]);
        unsigned char b2 = static_cast<unsigned char>(file_data[2]);
        unsigned char b3 = static_cast<unsigned char>(file_data[3]);
        LogInfo("File header bytes: %02x %02x %02x %02x", b0, b1, b2, b3);

        if (b0 == 0xFF && b1 == 0xD8) {
            LogInfo("Detected JPEG image format");
        } else if (b0 == 0x89 && b1 == 0x50 && b2 == 0x4E && b3 == 0x47) {
            LogInfo("Detected PNG image format");
        } else if (b0 == 0x47 && b1 == 0x49 && b2 == 0x46) {
            LogInfo("Detected GIF image format");
        } else if (b0 == 0x42 && b1 == 0x4D) {
            LogInfo("Detected BMP image format");
        } else {
            LogInfo("WARNING: Unknown image format (header: %02x %02x %02x %02x)",
                    b0, b1, b2, b3);
        }

        LogInfo("Calling mtmd_helper_bitmap_init_from_buf(ctx=%p, data, %zu)",
                static_cast<const void*>(ctx), file_data.size());
        mtmd_bitmap* bitmap = mtmd_helper_bitmap_init_from_buf(
            ctx,
            reinterpret_cast<const unsigned char*>(file_data.data()),
            file_data.size());

        if (!bitmap) {
            LogInfo("mtmd_helper_bitmap_init_from_buf returned NULL for file: %s",
                    image_path.c_str());
            SetError(error,
                     "mtmd_helper_bitmap_init_from_buf failed for: " + image_path);
            return MakeNullBitmap();
        }

        uint32_t nx = mtmd_bitmap_get_nx(bitmap);
        uint32_t ny = mtmd_bitmap_get_ny(bitmap);
        size_t n_bytes = mtmd_bitmap_get_n_bytes(bitmap);
        bool is_audio = mtmd_bitmap_is_audio(bitmap);
        const char* id = mtmd_bitmap_get_id(bitmap);
        double bpp = (nx > 0 && ny > 0) ? static_cast<double>(n_bytes) / (nx * ny) : 0.0;
        LogInfo("Bitmap created successfully: nx=%u, ny=%u, n_bytes=%zu, "
                "is_audio=%s, id=%s, bytes_per_pixel=%.1f",
                nx, ny, n_bytes, is_audio ? "true" : "false",
                id ? id : "(null)", bpp);

        if (n_bytes != static_cast<size_t>(nx) * ny * 3) {
            LogInfo("WARNING: n_bytes(%zu) != nx*ny*3(%zu), unexpected pixel format",
                    n_bytes, static_cast<size_t>(nx) * ny * 3);
        }

        return MtmdBitmapPtr(bitmap, mtmd_bitmap_free);
    }

    if (!image_bytes.empty()) {
        LogInfo("Loading image from memory buffer: %zu bytes", image_bytes.size());

        if (image_bytes.size() < 4) {
            LogInfo("Buffer too small (%zu bytes), not a valid image",
                    image_bytes.size());
            SetError(error, "Image buffer too small to be valid");
            return MakeNullBitmap();
        }

        unsigned char b0 = image_bytes[0];
        unsigned char b1 = image_bytes[1];
        unsigned char b2 = image_bytes[2];
        unsigned char b3 = image_bytes[3];
        LogInfo("Buffer header bytes: %02x %02x %02x %02x", b0, b1, b2, b3);

        if (b0 == 0xFF && b1 == 0xD8) {
            LogInfo("Detected JPEG image format from buffer");
        } else if (b0 == 0x89 && b1 == 0x50 && b2 == 0x4E && b3 == 0x47) {
            LogInfo("Detected PNG image format from buffer");
        } else if (b0 == 0x47 && b1 == 0x49 && b2 == 0x46) {
            LogInfo("Detected GIF image format from buffer");
        } else if (b0 == 0x42 && b1 == 0x4D) {
            LogInfo("Detected BMP image format from buffer");
        } else {
            LogInfo("WARNING: Unknown image format from buffer (header: %02x %02x %02x %02x)",
                    b0, b1, b2, b3);
        }

        LogInfo("Calling mtmd_helper_bitmap_init_from_buf(ctx=%p, data, %zu)",
                static_cast<const void*>(ctx), image_bytes.size());
        mtmd_bitmap* bitmap = mtmd_helper_bitmap_init_from_buf(
            ctx, image_bytes.data(), image_bytes.size());

        if (!bitmap) {
            LogInfo("mtmd_helper_bitmap_init_from_buf returned NULL for buffer");
            SetError(error, "mtmd_helper_bitmap_init_from_buf failed for image buffer");
            return MakeNullBitmap();
        }

        uint32_t nx = mtmd_bitmap_get_nx(bitmap);
        uint32_t ny = mtmd_bitmap_get_ny(bitmap);
        size_t n_bytes = mtmd_bitmap_get_n_bytes(bitmap);
        bool is_audio = mtmd_bitmap_is_audio(bitmap);
        const char* id = mtmd_bitmap_get_id(bitmap);
        double bpp = (nx > 0 && ny > 0) ? static_cast<double>(n_bytes) / (nx * ny) : 0.0;
        LogInfo("Bitmap created successfully: nx=%u, ny=%u, n_bytes=%zu, "
                "is_audio=%s, id=%s, bytes_per_pixel=%.1f",
                nx, ny, n_bytes, is_audio ? "true" : "false",
                id ? id : "(null)", bpp);

        if (n_bytes != static_cast<size_t>(nx) * ny * 3) {
            LogInfo("WARNING: n_bytes(%zu) != nx*ny*3(%zu), unexpected pixel format",
                    n_bytes, static_cast<size_t>(nx) * ny * 3);
        }

        return MtmdBitmapPtr(bitmap, mtmd_bitmap_free);
    }

    LogInfo("No image path or bytes provided, returning null bitmap");
    return MakeNullBitmap();
}

}  // namespace vlm
