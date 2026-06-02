/**
 * @file mtmd_image.h
 * @brief Image loading utilities for mtmd bitmap handling.
 *
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef VLM_SRC_CORE_CPP_MTMD_IMAGE_H_
#define VLM_SRC_CORE_CPP_MTMD_IMAGE_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct mtmd_bitmap;
struct mtmd_context;

namespace vlm {

using MtmdBitmapPtr = std::unique_ptr<mtmd_bitmap, void (*)(mtmd_bitmap*)>;

MtmdBitmapPtr MakeNullBitmap();

MtmdBitmapPtr LoadBitmapFromInput(mtmd_context* ctx,
                                  const std::string& image_path,
                                  const std::vector<uint8_t>& image_bytes,
                                  std::string* error);

}  // namespace vlm

#endif  // VLM_SRC_CORE_CPP_MTMD_IMAGE_H_
