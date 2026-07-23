# Copyright (C) 2025 Intel Corporation, All rights reserved.
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

if (DEFINED SYCL_INTEL_TARGET AND SYCL_TARGET_INTEL_GPU_CRI)
  set(TEST_BASE "--m=${CRI_GEMM_DEFAULT_SIZE} --n=${CRI_GEMM_DEFAULT_SIZE} --k=${CRI_GEMM_DEFAULT_SIZE} --iterations=${CRI_DEFAULT_ITERATIONS} --verify=${CRI_DEFAULT_VERIFY}")
  set(TEST_SCALE "--m=128 --n=128 --k=128 --groups=2 --iterations=1 --verify=1")

cutlass_example_add_executable(
  51_xe35_block_scaled_grouped_gemm_e5m2
  51_xe35_block_scaled_grouped_gemm_e5m2.cpp
  TEST_COMMAND_OPTIONS
  TEST_BASE
)

cutlass_example_add_executable(
  51_xe35_block_scaled_grouped_gemm_e4m3
  51_xe35_block_scaled_grouped_gemm_e4m3.cpp
  TEST_COMMAND_OPTIONS
  TEST_BASE
)

cutlass_example_add_executable(
  51_xe35_block_scaled_grouped_gemm_e2m1
  51_xe35_block_scaled_grouped_gemm_e2m1.cpp
  TEST_COMMAND_OPTIONS
  TEST_BASE
)

cutlass_example_add_executable(
  51_xe35_block_scaled_grouped_gemm_fp8_e4m3
  51_xe35_block_scaled_grouped_gemm_fp8_e4m3.cpp
  TEST_COMMAND_OPTIONS
  TEST_BASE
  TEST_SCALE
)

endif()
