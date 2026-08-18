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

if (CUTLASS_TEST_FOR_CRI)
  set(TEST_CRI_ARGS "--m=${CRI_GEMM_DEFAULT_SIZE} --n=${CRI_GEMM_DEFAULT_SIZE} --k=${CRI_GEMM_DEFAULT_SIZE} --iterations=${CRI_DEFAULT_ITERATIONS} --verify=${CRI_DEFAULT_VERIFY}")

  set(GEMM_ARGS TEST_CRI_ARGS)
  set(GEMM_PADDED_ARGS TEST_CRI_ARGS)
  set(GEMM_QUEUE_ARGS TEST_CRI_ARGS)
else()
  set(TEST_BATCHES --l=2)
  set(TEST_LARGE "--l=513 --m=8 --n=16384 --k=512") # B matrix capacity > uint32_max
  set(TEST_SMALL_SHAPE --m=4 --n=8 --k=8 --l=2)
  set(TEST_SMALL_SHAPE_PADDABLE --m=1 --n=1 --k=2 --l=2)

  set(GEMM_ARGS TEST_BATCHES TEST_LARGE TEST_SMALL_SHAPE)
  set(GEMM_PADDED_ARGS TEST_BATCHES TEST_SMALL_SHAPE_PADDABLE)
  set(GEMM_QUEUE_ARGS TEST_BATCHES)
endif()

cutlass_example_add_executable(
  00_bmg_gemm
  00_bmg_gemm.cpp
  TEST_COMMAND_OPTIONS
  ${GEMM_ARGS}
)

if (CUTLASS_TEST_FOR_CRI)
  set(TEST_INTEGER_GEMM "--m=${CRI_GEMM_DEFAULT_SIZE} --n=${CRI_GEMM_DEFAULT_SIZE} --k=${CRI_GEMM_DEFAULT_SIZE} --l=1 --iterations=${CRI_DEFAULT_ITERATIONS} --verify=${CRI_DEFAULT_VERIFY}")
else()
  set(TEST_INTEGER_GEMM "--m=512 --n=512 --k=512 --l=2 --iterations=20 --verify=1")
endif()

cutlass_example_add_executable(
  00_bmg_gemm_int8
  00_bmg_gemm.cpp
  TEST_COMMAND_OPTIONS
  TEST_INTEGER_GEMM
)
target_compile_definitions(00_bmg_gemm_int8 PRIVATE CUTLASS_BMG_GEMM_INT8=1)

cutlass_example_add_executable(
  00_bmg_gemm_int4
  00_bmg_gemm.cpp
  TEST_COMMAND_OPTIONS
  TEST_INTEGER_GEMM
)
target_compile_definitions(00_bmg_gemm_int4 PRIVATE CUTLASS_BMG_GEMM_INT4=1)

cutlass_example_add_executable(
  00_bmg_gemm_int4_packed_b
  00_bmg_gemm.cpp
  TEST_COMMAND_OPTIONS
  TEST_INTEGER_GEMM
)
target_compile_definitions(00_bmg_gemm_int4_packed_b PRIVATE
  CUTLASS_BMG_GEMM_INT4=1
  CUTLASS_BMG_GEMM_INT4_PACKED_B=1
)

set(TEST_INT8_DEQUANT "--m=512 --n=512 --k=512 --l=2 --iterations=20 --verify=1")

cutlass_example_add_executable(
  00_bmg_gemm_int8_dequant_fp32
  00_bmg_gemm_int8_dequant.cpp
  TEST_COMMAND_OPTIONS
  TEST_INT8_DEQUANT
)

cutlass_example_add_executable(
  00_bmg_gemm_int8_dequant_bf16
  00_bmg_gemm_int8_dequant.cpp
  TEST_COMMAND_OPTIONS
  TEST_INT8_DEQUANT
)
target_compile_definitions(00_bmg_gemm_int8_dequant_bf16 PRIVATE
  CUTLASS_BMG_GEMM_INT8_DEQUANT_BF16_C=1
)

cutlass_example_add_executable(
  00_bmg_gemm_int8_dequant_per_row_a_fp32
  00_bmg_gemm_int8_dequant.cpp
  TEST_COMMAND_OPTIONS
  TEST_INT8_DEQUANT
)
target_compile_definitions(00_bmg_gemm_int8_dequant_per_row_a_fp32 PRIVATE
  CUTLASS_BMG_GEMM_INT8_DEQUANT_PER_ROW_A=1
)

cutlass_example_add_executable(
  00_bmg_gemm_int8_dequant_per_row_a_bf16
  00_bmg_gemm_int8_dequant.cpp
  TEST_COMMAND_OPTIONS
  TEST_INT8_DEQUANT
)
target_compile_definitions(00_bmg_gemm_int8_dequant_per_row_a_bf16 PRIVATE
  CUTLASS_BMG_GEMM_INT8_DEQUANT_BF16_C=1
  CUTLASS_BMG_GEMM_INT8_DEQUANT_PER_ROW_A=1
)

set(TEST_INT4_DEQUANT "--m=512 --n=512 --k=512 --l=2 --iterations=20 --verify=1")

cutlass_example_add_executable(
  00_bmg_gemm_int4_dequant_fp32
  00_bmg_gemm_int4_dequant.cpp
  TEST_COMMAND_OPTIONS
  TEST_INT4_DEQUANT
)

cutlass_example_add_executable(
  00_bmg_gemm_int4_dequant_bf16
  00_bmg_gemm_int4_dequant.cpp
  TEST_COMMAND_OPTIONS
  TEST_INT4_DEQUANT
)
target_compile_definitions(00_bmg_gemm_int4_dequant_bf16 PRIVATE
  CUTLASS_BMG_GEMM_INT4_DEQUANT_BF16_C=1
)

cutlass_example_add_executable(
  00_bmg_gemm_int4_dequant_per_row_a_fp32
  00_bmg_gemm_int4_dequant.cpp
  TEST_COMMAND_OPTIONS
  TEST_INT4_DEQUANT
)
target_compile_definitions(00_bmg_gemm_int4_dequant_per_row_a_fp32 PRIVATE
  CUTLASS_BMG_GEMM_INT4_DEQUANT_PER_ROW_A=1
)

cutlass_example_add_executable(
  00_bmg_gemm_int4_dequant_per_row_a_bf16
  00_bmg_gemm_int4_dequant.cpp
  TEST_COMMAND_OPTIONS
  TEST_INT4_DEQUANT
)
target_compile_definitions(00_bmg_gemm_int4_dequant_per_row_a_bf16 PRIVATE
  CUTLASS_BMG_GEMM_INT4_DEQUANT_BF16_C=1
  CUTLASS_BMG_GEMM_INT4_DEQUANT_PER_ROW_A=1
)

cutlass_example_add_executable(
  00_bmg_gemm_int4_dequant_packed_b_fp32
  00_bmg_gemm_int4_dequant.cpp
  TEST_COMMAND_OPTIONS
  TEST_INT4_DEQUANT
)
target_compile_definitions(00_bmg_gemm_int4_dequant_packed_b_fp32 PRIVATE
  CUTLASS_BMG_GEMM_INT4_DEQUANT_PACKED_B=1
)

cutlass_example_add_executable(
  00_bmg_gemm_int4_dequant_packed_b_bf16
  00_bmg_gemm_int4_dequant.cpp
  TEST_COMMAND_OPTIONS
  TEST_INT4_DEQUANT
)
target_compile_definitions(00_bmg_gemm_int4_dequant_packed_b_bf16 PRIVATE
  CUTLASS_BMG_GEMM_INT4_DEQUANT_BF16_C=1
  CUTLASS_BMG_GEMM_INT4_DEQUANT_PACKED_B=1
)

cutlass_example_add_executable(
  00_bmg_gemm_int4_dequant_packed_b_per_row_a_fp32
  00_bmg_gemm_int4_dequant.cpp
  TEST_COMMAND_OPTIONS
  TEST_INT4_DEQUANT
)
target_compile_definitions(00_bmg_gemm_int4_dequant_packed_b_per_row_a_fp32 PRIVATE
  CUTLASS_BMG_GEMM_INT4_DEQUANT_PACKED_B=1
  CUTLASS_BMG_GEMM_INT4_DEQUANT_PER_ROW_A=1
)

cutlass_example_add_executable(
  00_bmg_gemm_int4_dequant_packed_b_per_row_a_bf16
  00_bmg_gemm_int4_dequant.cpp
  TEST_COMMAND_OPTIONS
  TEST_INT4_DEQUANT
)
target_compile_definitions(00_bmg_gemm_int4_dequant_packed_b_per_row_a_bf16 PRIVATE
  CUTLASS_BMG_GEMM_INT4_DEQUANT_BF16_C=1
  CUTLASS_BMG_GEMM_INT4_DEQUANT_PACKED_B=1
  CUTLASS_BMG_GEMM_INT4_DEQUANT_PER_ROW_A=1
)

set(TEST_INT4_BLOCK_SCALED "--m=512 --n=512 --k=512 --l=2 --iterations=20 --verify=1")

cutlass_example_add_executable(
  00_bmg_gemm_int4_block_scaled_128_packed_b_fp32
  00_bmg_gemm_int4_block_scaled.cpp
  TEST_COMMAND_OPTIONS
  TEST_INT4_BLOCK_SCALED
)

cutlass_example_add_executable(
  00_bmg_gemm_int4_block_scaled_128_packed_b_bf16
  00_bmg_gemm_int4_block_scaled.cpp
  TEST_COMMAND_OPTIONS
  TEST_INT4_BLOCK_SCALED
)
target_compile_definitions(00_bmg_gemm_int4_block_scaled_128_packed_b_bf16 PRIVATE
  CUTLASS_BMG_GEMM_INT4_BLOCK_SCALED_BF16=1
)

cutlass_example_add_executable(
  00_bmg_gemm_int4_block_scaled_256_packed_b_fp32
  00_bmg_gemm_int4_block_scaled.cpp
  TEST_COMMAND_OPTIONS
  TEST_INT4_BLOCK_SCALED
)
target_compile_definitions(00_bmg_gemm_int4_block_scaled_256_packed_b_fp32 PRIVATE
  CUTLASS_BMG_GEMM_INT4_BLOCK_SCALED_256=1
)

cutlass_example_add_executable(
  00_bmg_gemm_int4_block_scaled_256_packed_b_bf16
  00_bmg_gemm_int4_block_scaled.cpp
  TEST_COMMAND_OPTIONS
  TEST_INT4_BLOCK_SCALED
)
target_compile_definitions(00_bmg_gemm_int4_block_scaled_256_packed_b_bf16 PRIVATE
  CUTLASS_BMG_GEMM_INT4_BLOCK_SCALED_256=1
  CUTLASS_BMG_GEMM_INT4_BLOCK_SCALED_BF16=1
)

cutlass_example_add_executable(
  00_bmg_gemm_int8_block_scaled_128_packed_b_fp32
  00_bmg_gemm_int4_block_scaled.cpp
  TEST_COMMAND_OPTIONS
  TEST_INT4_BLOCK_SCALED
)
target_compile_definitions(00_bmg_gemm_int8_block_scaled_128_packed_b_fp32 PRIVATE
  CUTLASS_BMG_GEMM_INT8_BLOCK_SCALED=1
)

cutlass_example_add_executable(
  00_bmg_gemm_int8_block_scaled_128_packed_b_bf16
  00_bmg_gemm_int4_block_scaled.cpp
  TEST_COMMAND_OPTIONS
  TEST_INT4_BLOCK_SCALED
)
target_compile_definitions(00_bmg_gemm_int8_block_scaled_128_packed_b_bf16 PRIVATE
  CUTLASS_BMG_GEMM_INT8_BLOCK_SCALED=1
  CUTLASS_BMG_GEMM_BLOCK_SCALED_BF16=1
)

cutlass_example_add_executable(
  00_bmg_gemm_int8_block_scaled_256_packed_b_fp32
  00_bmg_gemm_int4_block_scaled.cpp
  TEST_COMMAND_OPTIONS
  TEST_INT4_BLOCK_SCALED
)
target_compile_definitions(00_bmg_gemm_int8_block_scaled_256_packed_b_fp32 PRIVATE
  CUTLASS_BMG_GEMM_INT8_BLOCK_SCALED=1
  CUTLASS_BMG_GEMM_BLOCK_SCALED_256=1
)

cutlass_example_add_executable(
  00_bmg_gemm_int8_block_scaled_256_packed_b_bf16
  00_bmg_gemm_int4_block_scaled.cpp
  TEST_COMMAND_OPTIONS
  TEST_INT4_BLOCK_SCALED
)
target_compile_definitions(00_bmg_gemm_int8_block_scaled_256_packed_b_bf16 PRIVATE
  CUTLASS_BMG_GEMM_INT8_BLOCK_SCALED=1
  CUTLASS_BMG_GEMM_BLOCK_SCALED_256=1
  CUTLASS_BMG_GEMM_BLOCK_SCALED_BF16=1
)

cutlass_example_add_executable(
  00_bmg_gemm_padded
  00_bmg_gemm_padded.cpp
  TEST_COMMAND_OPTIONS
  ${GEMM_PADDED_ARGS}
)

cutlass_example_add_executable(
  00_bmg_gemm_with_sycl_queue
  00_bmg_gemm_with_sycl_queue.cpp
  TEST_COMMAND_OPTIONS
  ${GEMM_QUEUE_ARGS}
)
