#!/usr/bin/env bash
# @file download_model.sh
# @brief Model download script for SpacemiT VLM models.
#
# Downloads model archives from https://archive.spacemit.com/spacemit-ai/model_zoo/vlm/
# and extracts them into a local cache directory.
#
# Copyright (C) 2026 SpacemiT
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

CACHE_DIR="${VLM_CACHE_DIR:-$HOME/.cache/models/vlm}"

BASE_URL="https://archive.spacemit.com/spacemit-ai/model_zoo/vlm"

AVAILABLE_MODELS=(
    "fastvlm-mm-0.5b-q4_1"
    "Qwen3.5-0.8B"
    "Qwen3.5-2B"
    "Qwen3.5-4B"
    "qwen30ba3b-mm-q4_1"
)

MODEL_SIZES=(
    "766M"
    "932M"
    "2.6G"
    "3.9G"
    "17.6G"
)

usage() {
    echo "Usage: $0 [MODEL_NAME]"
    echo ""
    echo "Download VLM models from SpacemiT archive."
    echo ""
    echo "Available models:"
    for i in "${!AVAILABLE_MODELS[@]}"; do
        printf "  %-30s %s\n" "${AVAILABLE_MODELS[$i]}" "${MODEL_SIZES[$i]}"
    done
    echo ""
    echo "Environment variables:"
    echo "  VLM_CACHE_DIR  Model storage directory (default: ~/.cache/models/vlm)"
    echo ""
    echo "Examples:"
    echo "  $0 fastvlm-mm-0.5b-q4_1"
    echo "  $0 Qwen3.5-0.8B"
    echo "  VLM_CACHE_DIR=/data/models $0 fastvlm-mm-0.5b-q4_1"
}

download_model() {
    local model_name="$1"
    local dest="$CACHE_DIR/$model_name"

    if [[ -d "$dest" && -f "$dest/config.json" ]]; then
        echo "Model $model_name already exists at $dest, skipping download."
        return 0
    fi

    mkdir -p "$dest"

    case "$model_name" in
        Qwen3VL-4B)
            download_qwen3vl_4b "$dest"
            ;;
        SmolVLM-256M)
            download_smollm_256m "$dest"
            ;;
        *)
            download_tar_model "$model_name" "$dest"
            ;;
    esac
}

download_tar_model() {
    local model_name="$1"
    local dest="$2"

    echo "Downloading model: $model_name"
    echo "Destination: $dest"
    echo "URL: $BASE_URL/$model_name.tar.gz"

    local tmp_tar="$CACHE_DIR/${model_name}.tar.gz"

    local url
    case "$model_name" in
        fastvlm-mm-0.5b-q4_1)
            url="$BASE_URL/fastvlm-mm-0.5b-q4%5F1.tar.gz"
            ;;
        qwen30ba3b-mm-q4_1)
            url="$BASE_URL/qwen30ba3b-mm-q4%5F1.tar.gz"
            ;;
        *)
            url="$BASE_URL/$model_name.tar.gz"
            ;;
    esac

    if ! curl -fSL --progress-bar -o "$tmp_tar" "$url"; then
        echo "ERROR: Failed to download $model_name from $url" >&2
        rm -f "$tmp_tar"
        return 1
    fi

    echo "Extracting..."
    tar -xzf "$tmp_tar" -C "$CACHE_DIR"
    rm -f "$tmp_tar"

    if [[ -f "$dest/config.json" ]]; then
        echo "Successfully downloaded and extracted: $model_name"
        echo "Model directory: $dest"
    else
        echo "WARNING: config.json not found in $dest after extraction." >&2
        echo "The archive structure may differ. Check $CACHE_DIR for extracted files." >&2
    fi
}

download_qwen3vl_4b() {
    local dest="$1"

    echo "Downloading model: Qwen3VL-4B (separate GGUF files)"
    echo "Destination: $dest"

    local text_gguf="Qwen3VL-4B-Instruct-Q4_K_M.gguf"
    local vision_gguf="mmproj-Qwen3VL-4B-Instruct-F16.gguf"

    if ! curl -fSL --progress-bar -o "$dest/$text_gguf" "$BASE_URL/Qwen3VL/Qwen3VL-4B-Instruct-Q4%5FK%5FM.gguf"; then
        echo "ERROR: Failed to download text model" >&2
        return 1
    fi

    if ! curl -fSL --progress-bar -o "$dest/$vision_gguf" "$BASE_URL/Qwen3VL/mmproj-Qwen3VL-4B-Instruct-F16.gguf"; then
        echo "ERROR: Failed to download vision model" >&2
        return 1
    fi

    cat > "$dest/config.json" << "CFGEOF"
{
    "architectures": ["Qwen3VLForConditionalGeneration"],
    "text_model": {
        "model_path": "Qwen3VL-4B-Instruct-Q4_K_M.gguf"
    },
    "vision_model": {
        "model_path": "mmproj-Qwen3VL-4B-Instruct-F16.gguf"
    },
    "media_backend": "auto",
    "n_gpu_layers": 0
}
CFGEOF

    echo "Successfully downloaded: Qwen3VL-4B"
    echo "Model directory: $dest"
}

download_smollm_256m() {
    local dest="$1"

    echo "Downloading model: SmolVLM-256M (separate GGUF files)"
    echo "Destination: $dest"

    local text_gguf="SmolVLM-256M-Instruct-f16.gguf"
    local vision_gguf="mmproj-SmolVLM-256M-Instruct-Q8_0.gguf"

    if ! curl -fSL --progress-bar -o "$dest/$text_gguf" "$BASE_URL/SmolVLM-256M-Instruct-f16.gguf"; then
        echo "ERROR: Failed to download text model" >&2
        return 1
    fi

    if ! curl -fSL --progress-bar -o "$dest/$vision_gguf" "$BASE_URL/mmproj-SmolVLM-256M-Instruct-Q8%5F0.gguf"; then
        echo "ERROR: Failed to download vision model" >&2
        return 1
    fi

    cat > "$dest/config.json" << "CFGEOF"
{
    "architectures": ["SmolVLMForConditionalGeneration"],
    "text_model": {
        "model_path": "SmolVLM-256M-Instruct-f16.gguf"
    },
    "vision_model": {
        "model_path": "mmproj-SmolVLM-256M-Instruct-Q8_0.gguf"
    },
    "media_backend": "auto",
    "n_gpu_layers": 0
}
CFGEOF

    echo "Successfully downloaded: SmolVLM-256M"
    echo "Model directory: $dest"
}

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

case "${1:-}" in
    -h|--help)
        usage
        exit 0
        ;;
    *)
        download_model "$1"
        ;;
esac
