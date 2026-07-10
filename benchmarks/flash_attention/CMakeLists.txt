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

# Create a parent flash_attention suite
cutlass_benchmark_add_suite(cutlass_benchmarks_flash_attention)

# Pass these configuration files for the CI
set(CONFIG_FILE_DECODE_BF16 --config_file=${CUTLASS_DIR}/benchmarks/device/bmg/input_files/input_flash_attention_decode_bf16.in)
set(CONFIG_FILE_DECODE_BF16_HND ${CONFIG_FILE_DECODE_BF16} --layout=HND)
set(CONFIG_FILE_DECODE_BF16_NHD ${CONFIG_FILE_DECODE_BF16} --layout=NHD)

set(CONFIG_FILE_PREFILL_BF16 --config_file=${CUTLASS_DIR}/benchmarks/device/bmg/input_files/input_flash_attention_prefill_bf16.in)
set(CONFIG_FILE_PREFILL_BF16_HND ${CONFIG_FILE_PREFILL_BF16} --layout=HND)
set(CONFIG_FILE_PREFILL_BF16_NHD ${CONFIG_FILE_PREFILL_BF16} --layout=NHD)

# Create the decode subsuite under flash_attention
cutlass_benchmark_add_suite(cutlass_benchmarks_flash_attention_decode
                            SUPERSUITE cutlass_benchmarks_flash_attention)

add_library(decode_lib SHARED
            benchmarks_decode_bf16.cpp
)

# Create the prefill subsuite under flash_attention
cutlass_benchmark_add_suite(cutlass_benchmarks_flash_attention_prefill
                            SUPERSUITE cutlass_benchmarks_flash_attention)

set(PREFILL_SOURCES
    benchmarks_prefill_bf16.cpp
)

add_library(prefill_lib SHARED ${PREFILL_SOURCES})

set(LIB_LIST decode_lib prefill_lib)

foreach(name IN LISTS LIB_LIST)
  target_include_directories(${name} PRIVATE ${CUTLASS_APPLICATIONS_DIR})
  target_link_libraries(${name} PRIVATE CUTLASS cutlass_tools_util_includes benchmark::benchmark)
  add_onemkl_to_target(TARGET ${name})
  # Add only SYCL include directories, not the full SYCL flags (to avoid duplication)
  add_sycl_include_directories_to_target(${name})
endforeach()

cutlass_benchmark_add_executable(
    cutlass_benchmarks_flash_attention_decode_xe
    main.cpp
    TEST_COMMAND_OPTIONS CONFIG_FILE_DECODE_BF16_HND CONFIG_FILE_DECODE_BF16_NHD
    LIBRARIES decode_lib
    INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}
    SUITE cutlass_benchmarks_flash_attention_decode
)

cutlass_benchmark_add_executable(
    cutlass_benchmarks_flash_attention_prefill_xe
    main.cpp
    TEST_COMMAND_OPTIONS CONFIG_FILE_PREFILL_BF16_HND CONFIG_FILE_PREFILL_BF16_NHD
    LIBRARIES prefill_lib
    INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}
    SUITE cutlass_benchmarks_flash_attention_prefill
)

# Include legacy flash_attention benchmarks
add_subdirectory(legacy)
