# Copyright (c) 2024 - 2025 Codeplay Software Ltd. All rights reserved.
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
set(CONFIG_FILE_CRI_DECODE_NOKV --config_file=${CUTLASS_DIR}/benchmarks/device/cri/input_files/legacy/input_sglang_flash_attention_decode_nokvcache.in)
set(CONFIG_FILE_BMG_DECODE_NOKV --config_file=${CUTLASS_DIR}/benchmarks/device/bmg/input_files/legacy/input_sglang_flash_attention_decode_nokvcache.in)

# Pass these configuration files for the CI
if(CUTLASS_TEST_FOR_CRI AND SYCL_TARGET_INTEL_GPU_CRI)
  set(DECODE_LEGACY_CONFIG_ARGS CONFIG_FILE_CRI_DECODE_NOKV)
else()
  set(DECODE_LEGACY_CONFIG_ARGS CONFIG_FILE_BMG_DECODE_NOKV)
endif()

cutlass_benchmark_add_suite(cutlass_benchmarks_flash_attention_decode_legacy
                            SUPERSUITE cutlass_benchmarks_flash_attention_legacy)

add_library(decode_h64_legacy SHARED
            benchmarks_h64_512_nonpaged.cpp
            benchmarks_h64_1024_nonpaged.cpp
)

add_library(decode_h96_legacy SHARED
            benchmarks_h96_512_nonpaged.cpp
            benchmarks_h96_1024_nonpaged.cpp
)

add_library(decode_h128_legacy SHARED
            benchmarks_h128_512_nonpaged.cpp
            benchmarks_h128_1024_nonpaged.cpp
)

add_library(decode_h192_legacy SHARED
            benchmarks_h192_512_nonpaged.cpp
            benchmarks_h192_1024_nonpaged.cpp
)

set(LIB_LIST decode_h64_legacy decode_h96_legacy decode_h128_legacy decode_h192_legacy)

foreach(name IN LISTS LIB_LIST)
  target_include_directories(${name} PRIVATE ${CUTLASS_APPLICATIONS_DIR})
  target_link_libraries(${name} PRIVATE CUTLASS cutlass_tools_util_includes benchmark::benchmark)
  add_onemkl_to_target(TARGET ${name})
  # Add only SYCL include directories, not the full SYCL flags (to avoid duplication)
  add_sycl_include_directories_to_target(${name})
endforeach()

cutlass_benchmark_add_executable(
    cutlass_benchmarks_flash_attention_decode_legacy_xe
    main.cpp
    TEST_COMMAND_OPTIONS ${DECODE_LEGACY_CONFIG_ARGS}
    LIBRARIES decode_h64_legacy decode_h96_legacy decode_h128_legacy decode_h192_legacy
    INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}
    SUITE cutlass_benchmarks_flash_attention_decode_legacy
)
