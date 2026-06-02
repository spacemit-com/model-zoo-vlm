#!/usr/bin/env bash
# @file cpplint.sh
# @brief Run cpplint on the VLM C++ source tree.
#
# Recursively checks all C/C++ files under the given target directory
# (defaults to the project root).
#
# Copyright (C) 2026 SpacemiT
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TARGET=${1:-"$ROOT_DIR"}

if ! command -v cpplint >/dev/null 2>&1; then
    echo "cpplint is not installed. Install it with: pip install cpplint" >&2
    exit 1
fi

cpplint --recursive \
    --filter=-build/c++11,-runtime/references \
    "$TARGET"
