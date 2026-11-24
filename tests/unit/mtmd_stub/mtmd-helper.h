/**
 * @file mtmd-helper.h
 * @brief Mutually exclusive legacy and wrapper-returning helper declarations.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef VLM_TEST_MTMD_HELPER_H_
#define VLM_TEST_MTMD_HELPER_H_

#include "mtmd.h"

#ifdef VLM_TEST_LEGACY_MTMD
mtmd_bitmap* mtmd_helper_bitmap_init_from_buf(
    mtmd_context* ctx, const unsigned char* data, size_t size);
#else
struct mtmd_helper_video {};
struct mtmd_helper_bitmap_wrapper {
    mtmd_bitmap* bitmap;
    mtmd_helper_video* video_ctx;
};

mtmd_helper_bitmap_wrapper mtmd_helper_bitmap_init_from_buf(
    mtmd_context* ctx, const unsigned char* data, size_t size, bool placeholder);
void mtmd_helper_video_free(mtmd_helper_video* video);
#endif

#endif  // VLM_TEST_MTMD_HELPER_H_
