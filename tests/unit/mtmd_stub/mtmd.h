/**
 * @file mtmd.h
 * @brief Minimal bitmap API used by the image loader contract tests.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef VLM_TEST_MTMD_H_
#define VLM_TEST_MTMD_H_

#include <cstddef>
#include <cstdint>

struct mtmd_context {};
struct mtmd_bitmap {};

void mtmd_bitmap_free(mtmd_bitmap* bitmap);
uint32_t mtmd_bitmap_get_nx(const mtmd_bitmap* bitmap);
uint32_t mtmd_bitmap_get_ny(const mtmd_bitmap* bitmap);
size_t mtmd_bitmap_get_n_bytes(const mtmd_bitmap* bitmap);
bool mtmd_bitmap_is_audio(const mtmd_bitmap* bitmap);
const char* mtmd_bitmap_get_id(const mtmd_bitmap* bitmap);

#endif  // VLM_TEST_MTMD_H_
