# Copyright (C) 2026 Intel Corporation, All rights reserved.
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

# ---- GDN chunkwise attention benchmark ----
#
# Produces one executable:
#   cutlass_benchmarks_gdn_xe
#
# The GDN kernels and launcher are header-only under
# applications/gdn_attention/, shared with the example driver and the unit
# tests. The BF16 activation variant is compiled into the binary and selected
# at runtime via the benchmark config .in file.
#


set(CUTLASS_APPLICATIONS_DIR ${CUTLASS_DIR}/applications)

# ---- config file paths ----
# This subdirectory is added on CRI and BMG builds (see benchmarks/applications/CMakeLists.txt).
set(CONFIG_FILE_CRI_GDN_BF16
    --config_file=${CUTLASS_DIR}/benchmarks/config_files/03_gdn/cri/bf16.in)
set(CONFIG_FILE_BMG_GDN_BF16
    --config_file=${CUTLASS_DIR}/benchmarks/config_files/03_gdn/bmg/bf16.in)

if(CUTLASS_TEST_FOR_CRI AND SYCL_TARGET_INTEL_GPU_CRI)
  set(GDN_TEST_COMMAND_OPTIONS CONFIG_FILE_CRI_GDN_BF16)
else()
  set(GDN_TEST_COMMAND_OPTIONS CONFIG_FILE_BMG_GDN_BF16)
endif()

cutlass_benchmark_add_suite(cutlass_benchmarks_gdn)

# Build the BF16 variant TU (which instantiates the header-only GDN launcher
# template, emitting the AOT kernel device code) as a STATIC library.
#
# Compilation model: main.cpp includes benchmarks.hpp, which only forward-
# declares register_gdn_benchmarks_bf16(); the definition (and the AOT device
# blobs) live in benchmarks_bf16.cpp, compiled separately into gdn_lib. A STATIC
# archive pulls those symbols and device blobs directly into the executable at
# link time, leaving no runtime .so dependency. A SHARED lib would instead leave
# a DT_NEEDED entry that CI's ld.so fails to resolve. This matches what the
# example driver in examples/14_xe35_gdn_attention/ already does.
set(GDN_SOURCES
    benchmarks_bf16.cpp
)

add_library(gdn_lib STATIC ${GDN_SOURCES})

target_include_directories(gdn_lib
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUTLASS_APPLICATIONS_DIR}
)

target_link_libraries(gdn_lib
  PRIVATE
    CUTLASS
    cutlass_tools_util_includes
    benchmark::benchmark
)

add_onemkl_to_target(TARGET gdn_lib)
# Full SYCL flags (not just include dirs) so benchmarks_bf16.cpp emits AOT
# device code into the archive — these blobs are then pulled into the
# executable by add_sycl_to_target() inside
# cutlass_benchmark_add_executable().
add_sycl_to_target(TARGET gdn_lib)

if(SYCL_TARGET_INTEL_GPU_CRI)
  target_compile_definitions(gdn_lib PRIVATE __SYCL_TARGET_INTEL_GPU_CRI__)
endif()

cutlass_benchmark_add_executable(
    cutlass_benchmarks_gdn_xe
    main.cpp
    TEST_COMMAND_OPTIONS ${GDN_TEST_COMMAND_OPTIONS}
    LIBRARIES gdn_lib
    INCLUDES ${CMAKE_CURRENT_SOURCE_DIR} ${CUTLASS_APPLICATIONS_DIR}
    SUITE cutlass_benchmarks_gdn
)
