/**
 * @file vlm_utils.cpp
 * @brief Implementation of shared utility functions for the VLM service.
 *
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "vlm_utils.h"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <sstream>

namespace vlm {

void SetError(std::string* error, const std::string& message) {
    if (error) {
        *error = message;
    }
}

std::string ShellQuote(const std::string& value) {
    std::string result = "'";
    for (char c : value) {
        if (c == '\'') {
            result += "'\\''";
        } else {
            result += c;
        }
    }
    result += '\'';
    return result;
}

std::string JsonEscape(const std::string& value) {
    std::string result;
    result.reserve(value.size());
    for (char c : value) {
        switch (c) {
            case '"':  result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\n': result += "\\n";  break;
            case '\r': result += "\\r";  break;
            case '\t': result += "\\t";  break;
            default:   result += c;      break;
        }
    }
    return result;
}

std::string Base64Encode(const std::vector<unsigned char>& data) {
    static const char kTable[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string result;
    result.reserve(((data.size() + 2) / 3) * 4);

    size_t i = 0;
    for (; i + 2 < data.size(); i += 3) {
        uint32_t triple = (static_cast<uint32_t>(data[i]) << 16) |
                            (static_cast<uint32_t>(data[i + 1]) << 8) |
                            static_cast<uint32_t>(data[i + 2]);
        result += kTable[(triple >> 18) & 0x3F];
        result += kTable[(triple >> 12) & 0x3F];
        result += kTable[(triple >> 6) & 0x3F];
        result += kTable[triple & 0x3F];
    }

    if (i < data.size()) {
        uint32_t triple = static_cast<uint32_t>(data[i]) << 16;
        if (i + 1 < data.size()) {
            triple |= static_cast<uint32_t>(data[i + 1]) << 8;
        }
        result += kTable[(triple >> 18) & 0x3F];
        result += kTable[(triple >> 12) & 0x3F];
        if (i + 1 < data.size()) {
            result += kTable[(triple >> 6) & 0x3F];
        } else {
            result += '=';
        }
        result += '=';
    }

    return result;
}

std::string ReadFileToString(const std::string& path, std::string* error) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        SetError(error, "Failed to open file: " + path);
        return {};
    }
    std::ostringstream oss;
    oss << ifs.rdbuf();
    if (ifs.fail() && !ifs.eof()) {
        SetError(error, "Failed to read file: " + path);
        return {};
    }
    return oss.str();
}

std::string ReadBinaryFile(const std::string& path, std::string* error) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) {
        SetError(error, "Failed to open binary file: " + path);
        return {};
    }
    std::ostringstream oss;
    oss << ifs.rdbuf();
    if (ifs.fail() && !ifs.eof()) {
        SetError(error, "Failed to read binary file: " + path);
        return {};
    }
    return oss.str();
}

std::string ExtractJsonString(const std::string& json,
                                const std::string& key) {
    std::string needle = "\"" + key + "\"";
    size_t pos = json.find(needle);
    if (pos == std::string::npos) {
        return {};
    }
    pos += needle.size();
    pos = json.find(':', pos);
    if (pos == std::string::npos) {
        return {};
    }
    ++pos;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' ||
            json[pos] == '\n' || json[pos] == '\r')) {
        ++pos;
    }
    if (pos >= json.size() || json[pos] != '"') {
        return {};
    }
    ++pos;
    std::string result;
    while (pos < json.size() && json[pos] != '"') {
        if (json[pos] == '\\' && pos + 1 < json.size()) {
            ++pos;
            switch (json[pos]) {
                case '"':  result += '"';  break;
                case '\\': result += '\\'; break;
                case 'n':  result += '\n'; break;
                case 'r':  result += '\r'; break;
                case 't':  result += '\t'; break;
                default:   result += json[pos]; break;
            }
        } else {
            result += json[pos];
        }
        ++pos;
    }
    return result;
}

int ExtractJsonInt(const std::string& json, const std::string& key) {
    std::string needle = "\"" + key + "\"";
    size_t pos = json.find(needle);
    if (pos == std::string::npos) {
        return 0;
    }
    pos += needle.size();
    pos = json.find(':', pos);
    if (pos == std::string::npos) {
        return 0;
    }
    ++pos;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' ||
            json[pos] == '\n' || json[pos] == '\r')) {
        ++pos;
    }
    bool negative = false;
    if (pos < json.size() && json[pos] == '-') {
        negative = true;
        ++pos;
    }
    int value = 0;
    while (pos < json.size() && json[pos] >= '0' && json[pos] <= '9') {
        value = value * 10 + (json[pos] - '0');
        ++pos;
    }
    return negative ? -value : value;
}

std::string StripThinkingBlocks(const std::string& text,
                                std::string* thinking_content) {
    const std::string kOpen = "<think";
    const std::string kClose = "</think";
    std::string result;
    std::string thinking;
    size_t pos = 0;

    while (pos < text.size()) {
        size_t open_pos = text.find(kOpen, pos);
        if (open_pos == std::string::npos) {
            result.append(text, pos, std::string::npos);
            break;
        }
        result.append(text, pos, open_pos - pos);
        size_t tag_end = text.find('>', open_pos);
        if (tag_end == std::string::npos) {
            break;
        }
        size_t close_pos = text.find(kClose, tag_end + 1);
        if (close_pos == std::string::npos) {
            break;
        }
        if (thinking_content) {
            thinking.append(text, tag_end + 1, close_pos - tag_end - 1);
        }
        size_t resume = close_pos + kClose.size();
        if (resume < text.size() && text[resume] == '>') {
            ++resume;
        }
        pos = resume;
    }

    if (thinking_content) {
        *thinking_content = thinking;
    }

    return result;
}

}  // namespace vlm
