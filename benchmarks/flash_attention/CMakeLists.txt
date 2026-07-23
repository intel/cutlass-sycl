# Copyright (c) 2024 - 2025 Codeplay Software Ltd. All rights reserved.
# Copyright (c) 2026 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

set(CUTLASS_APPLICATIONS_DIR ${CUTLASS_DIR}/applications)
set(CONFIG_FILE_CRI_DECODE_FP8 --config_file=${CUTLASS_DIR}/benchmarks/device/cri/input_files/input_flash_attention_decode_fp8.in)
set(CONFIG_FILE_CRI_DECODE_FP8KV_FP16MMA --config_file=${CUTLASS_DIR}/benchmarks/device/cri/input_files/input_flash_attention_decode_fp8kv_fp16mma.in)
set(CONFIG_FILE_CRI_PREFILL_FP8 --config_file=${CUTLASS_DIR}/benchmarks/device/cri/input_files/input_flash_attention_prefill_fp8.in)
set(CONFIG_FILE_CRI_PREFILL_FP8KV_FP16MMA --config_file=${CUTLASS_DIR}/benchmarks/device/cri/input_files/input_flash_attention_prefill_fp8kv_fp16mma.in)
set(CONFIG_FILE_CRI_PREFILL_MXFP4 --config_file=${CUTLASS_DIR}/benchmarks/device/cri/input_files/input_flash_attention_prefill_mxfp4.in)
set(CONFIG_FILE_CRI_PREFILL_MXFP8 --config_file=${CUTLASS_DIR}/benchmarks/device/cri/input_files/input_flash_attention_prefill_mxfp8.in)
set(CONFIG_FILE_BMG_DECODE_BF16 --config_file=${CUTLASS_DIR}/benchmarks/device/bmg/input_files/input_flash_attention_decode_bf16.in)
set(CONFIG_FILE_BMG_PREFILL_BF16 --config_file=${CUTLASS_DIR}/benchmarks/device/bmg/input_files/input_flash_attention_prefill_bf16.in)

# Create a parent flash_attention suite
cutlass_benchmark_add_suite(cutlass_benchmarks_flash_attention)

# Pass these configuration files for the CI
if(CUTLASS_TEST_FOR_CRI AND SYCL_TARGET_INTEL_GPU_CRI)
  set(CONFIG_FILE_CRI_PREFILL_BF16 --config_file=${CUTLASS_DIR}/benchmarks/device/cri/input_files/input_flash_attention_prefill_bf16.in)

  # LLM Op Shapes - Real-scenario LLM operator shapes for testing
  set(CONFIG_FILE_LLM_OP_SHAPES_DECODE_FP8 --config_file=${CUTLASS_DIR}/benchmarks/device/cri/input_files/llm_op_shapes/input_flash_attention_decode_fp8.in)
  set(CONFIG_FILE_LLM_OP_SHAPES_DECODE_FP8KV_FP16MMA --config_file=${CUTLASS_DIR}/benchmarks/device/cri/input_files/llm_op_shapes/input_flash_attention_decode_fp8kv_fp16mma.in)
  set(CONFIG_FILE_LLM_OP_SHAPES_PREFILL_FP8 --config_file=${CUTLASS_DIR}/benchmarks/device/cri/input_files/llm_op_shapes/input_flash_attention_prefill_fp8.in)
  set(CONFIG_FILE_LLM_OP_SHAPES_PREFILL_FP8KV_FP16MMA --config_file=${CUTLASS_DIR}/benchmarks/device/cri/input_files/llm_op_shapes/input_flash_attention_prefill_fp8kv_fp16mma.in)
  set(CONFIG_FILE_LLM_OP_SHAPES_PREFILL_MXFP8 --config_file=${CUTLASS_DIR}/benchmarks/device/cri/input_files/llm_op_shapes/input_flash_attention_prefill_mxfp8.in)
  set(CONFIG_FILE_LLM_OP_SHAPES_PREFILL_MXFP4 --config_file=${CUTLASS_DIR}/benchmarks/device/cri/input_files/llm_op_shapes/input_flash_attention_prefill_mxfp4.in)

  set(DECODE_TEST_COMMAND_OPTIONS
      CONFIG_FILE_CRI_DECODE_FP8
      CONFIG_FILE_CRI_DECODE_FP8KV_FP16MMA
      CONFIG_FILE_LLM_OP_SHAPES_DECODE_FP8
      CONFIG_FILE_LLM_OP_SHAPES_DECODE_FP8KV_FP16MMA)
  set(PREFILL_TEST_COMMAND_OPTIONS
      CONFIG_FILE_CRI_PREFILL_BF16
      CONFIG_FILE_CRI_PREFILL_FP8
      CONFIG_FILE_CRI_PREFILL_FP8KV_FP16MMA
      CONFIG_FILE_CRI_PREFILL_MXFP8
      CONFIG_FILE_CRI_PREFILL_MXFP4
      CONFIG_FILE_LLM_OP_SHAPES_PREFILL_FP8
      CONFIG_FILE_LLM_OP_SHAPES_PREFILL_FP8KV_FP16MMA
      CONFIG_FILE_LLM_OP_SHAPES_PREFILL_MXFP8
      CONFIG_FILE_LLM_OP_SHAPES_PREFILL_MXFP4)
else()
  set(DECODE_TEST_COMMAND_OPTIONS CONFIG_FILE_BMG_DECODE_BF16 CONFIG_FILE_CRI_DECODE_FP8 CONFIG_FILE_CRI_DECODE_FP8KV_FP16MMA)
  set(PREFILL_TEST_COMMAND_OPTIONS CONFIG_FILE_BMG_PREFILL_BF16 CONFIG_FILE_CRI_PREFILL_FP8 CONFIG_FILE_CRI_PREFILL_FP8KV_FP16MMA)
endif()

# Create the decode subsuite under flash_attention
cutlass_benchmark_add_suite(cutlass_benchmarks_flash_attention_decode
                            SUPERSUITE cutlass_benchmarks_flash_attention)

set(DECODE_SOURCES
    benchmarks_decode_fp8.cpp
    benchmarks_decode_fp8kv_fp16mma.cpp
)
if(NOT (CUTLASS_TEST_FOR_CRI AND SYCL_TARGET_INTEL_GPU_CRI))
  list(APPEND DECODE_SOURCES benchmarks_decode_bf16.cpp)
endif()

add_library(decode_lib SHARED ${DECODE_SOURCES})

# Create the prefill subsuite under flash_attention
cutlass_benchmark_add_suite(cutlass_benchmarks_flash_attention_prefill
                            SUPERSUITE cutlass_benchmarks_flash_attention)

set(PREFILL_SOURCES
    benchmarks_prefill_fp8.cpp
    benchmarks_prefill_fp8kv_fp16mma.cpp
)
if(NOT (CUTLASS_TEST_FOR_CRI AND SYCL_TARGET_INTEL_GPU_CRI))
  list(APPEND PREFILL_SOURCES benchmarks_prefill_bf16.cpp)
endif()

if(SYCL_TARGET_INTEL_GPU_CRI)
  list(APPEND PREFILL_SOURCES
       benchmarks_prefill_mxfp8.cpp
       benchmarks_prefill_mxfp4.cpp
  )
endif()

add_library(prefill_lib SHARED ${PREFILL_SOURCES})

set(LIB_LIST decode_lib prefill_lib)

foreach(name IN LISTS LIB_LIST)
  target_include_directories(${name} PRIVATE ${CUTLASS_APPLICATIONS_DIR})
  target_link_libraries(${name} PRIVATE CUTLASS cutlass_tools_util_includes benchmark::benchmark)
  add_onemkl_to_target(TARGET ${name})
  # Add only SYCL include directories, not the full SYCL flags (to avoid duplication)
  add_sycl_include_directories_to_target(${name})
  if(SYCL_TARGET_INTEL_GPU_CRI)
    target_compile_definitions(${name} PRIVATE __SYCL_TARGET_INTEL_GPU_CRI__)
  endif()
endforeach()

cutlass_benchmark_add_executable(
    cutlass_benchmarks_flash_attention_decode_xe
    main.cpp
    TEST_COMMAND_OPTIONS ${DECODE_TEST_COMMAND_OPTIONS}
    LIBRARIES decode_lib
    INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}
    SUITE cutlass_benchmarks_flash_attention_decode
)

cutlass_benchmark_add_executable(
    cutlass_benchmarks_flash_attention_prefill_xe
    main.cpp
    TEST_COMMAND_OPTIONS ${PREFILL_TEST_COMMAND_OPTIONS}
    LIBRARIES prefill_lib
    INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}
    SUITE cutlass_benchmarks_flash_attention_prefill
)

# Include legacy flash_attention benchmarks
add_subdirectory(legacy)
