# CMakeLists.txt for model-zoo-vlm
# @brief Top-level CMake build configuration for the VLM inference service.

cmake_minimum_required(VERSION 3.10)
project(model_zoo_vlm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_VLM_EXAMPLES "Build VLM examples" ON)
option(BUILD_VLM_TESTS "Build VLM tests" ON)
option(BUILD_VLM_BENCHMARKS "Build VLM benchmarks" OFF)

find_path(LLAMA_INCLUDE_DIR llama.h PATHS /usr/include /usr/local/include)
find_library(LLAMA_LIBRARY NAMES llama PATHS /usr/lib /usr/local/lib)
find_library(LLAMA_COMMON_LIBRARY NAMES llama-common PATHS /usr/lib /usr/local/lib)
find_library(MTMD_LIBRARY NAMES mtmd PATHS /usr/lib /usr/local/lib)
find_library(GGML_LIBRARY NAMES ggml PATHS /usr/lib /usr/local/lib)
find_library(GGML_BASE_LIBRARY NAMES ggml-base PATHS /usr/lib /usr/local/lib)
find_library(GGML_CPU_LIBRARY NAMES ggml-cpu PATHS /usr/lib /usr/local/lib)
find_package(yaml-cpp REQUIRED)


if(NOT LLAMA_INCLUDE_DIR OR NOT LLAMA_LIBRARY OR NOT MTMD_LIBRARY OR NOT GGML_LIBRARY)
    message(FATAL_ERROR "VLM requires llama.cpp-tools-spacemit headers and libraries. Install: sudo apt install llama.cpp-tools-spacemit")
endif()

# Validate all required library variables are found (not NOTFOUND)
foreach(_lib_var LLAMA_LIBRARY LLAMA_COMMON_LIBRARY MTMD_LIBRARY GGML_LIBRARY GGML_BASE_LIBRARY GGML_CPU_LIBRARY)
    if(NOT ${_lib_var})
        message(FATAL_ERROR "Required library ${_lib_var} not found")
    endif()
endforeach()

add_library(vlm_core
    src/core/cpp/llama_vlm_model.cpp
    src/core/cpp/mtmd_image.cpp
    src/core/cpp/sampler.cpp
    src/core/cpp/server_vlm_model.cpp
    src/core/cpp/vlm_config.cpp
    src/core/cpp/vlm_model_factory.cpp
    src/core/cpp/vlm_utils.cpp
)

target_include_directories(vlm_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/core/cpp
        ${LLAMA_INCLUDE_DIR}
)

target_link_libraries(vlm_core
    PUBLIC
        stdc++fs
    PRIVATE
        ${LLAMA_LIBRARY}
        ${LLAMA_COMMON_LIBRARY}
        ${MTMD_LIBRARY}
        ${GGML_LIBRARY}
        ${GGML_BASE_LIBRARY}
        ${GGML_CPU_LIBRARY}
        yaml-cpp
)

install(TARGETS vlm_core DESTINATION lib)
install(FILES include/vlm_service.h DESTINATION include)



if(BUILD_VLM_EXAMPLES)
    add_executable(fastvlm_demo examples/fastvlm/cpp/fastvlm_demo.cpp)
    target_link_libraries(fastvlm_demo PRIVATE vlm_core)

    add_executable(chat_demo examples/fastvlm/cpp/chat_demo.cpp)
    target_link_libraries(chat_demo PRIVATE vlm_core)

    add_executable(stream_demo examples/fastvlm/cpp/stream_demo.cpp)
    target_link_libraries(stream_demo PRIVATE vlm_core)

    install(TARGETS fastvlm_demo chat_demo stream_demo DESTINATION bin)
    install(DIRECTORY examples/fastvlm/config/ DESTINATION share/model-zoo-vlm)
endif()

if(BUILD_VLM_TESTS)
    enable_testing()
    add_executable(test_vlm_config tests/unit/test_vlm_config.cpp)
    target_include_directories(test_vlm_config PRIVATE src/core/cpp)
    target_link_libraries(test_vlm_config PRIVATE vlm_core)
    add_test(NAME test_vlm_config COMMAND test_vlm_config)

    add_executable(test_vlm_model_factory tests/unit/test_vlm_model_factory.cpp)
    target_include_directories(test_vlm_model_factory PRIVATE src/core/cpp)
    target_link_libraries(test_vlm_model_factory PRIVATE vlm_core)
    add_test(NAME test_vlm_model_factory COMMAND test_vlm_model_factory)
endif()

if(BUILD_VLM_BENCHMARKS)
    add_executable(fastvlm_benchmark tests/performance/fastvlm_benchmark.cpp)
    target_link_libraries(fastvlm_benchmark PRIVATE vlm_core)
    install(TARGETS fastvlm_benchmark DESTINATION bin)
endif()
