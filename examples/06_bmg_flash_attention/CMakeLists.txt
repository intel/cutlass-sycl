# Copyright (c) 2024 - 2025 Codeplay Software Ltd. All rights reserved.
# Copyright (C) 2025 - 2026 Intel Corporation, All rights reserved.
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
# TODO For Multitarget, these input types may be wrong for non-cri hardware.
# Skip non-supported tests at runtime, keep this flag as well to not increase compile
# time for single target build.

if(DEFINED SYCL_INTEL_TARGET AND SYCL_TARGET_INTEL_GPU_CRI)
  set(INPUT_TYPES bfloat16_t float_e5m2_t float_e4m3_t float_e2m1_t mx_float_e5m2_t mx_float_e4m3_t mx_float_e2m1_t)
else()
  set(INPUT_TYPES bfloat16_t float_e5m2_t float_e4m3_t)
endif()

foreach(HEAD_DIM 64 96 128 192)
  foreach(INPUT_TYPE ${INPUT_TYPES})
    # FP4 (e2m1) DPAS atom has K=64, so head dim must be a multiple of 64.
    # hdim96 is not divisible by 64 and is therefore unsupported for FP4 inputs.
    if (HEAD_DIM EQUAL 96 AND (INPUT_TYPE STREQUAL "float_e2m1_t" OR INPUT_TYPE STREQUAL "mx_float_e2m1_t"))
      continue()
    endif()

    if (CUTLASS_TEST_FOR_CRI)
      set(TEST_BASE "--batch=1 --num_heads_q=4 --num_heads_kv=4 --seq_len_kv=256 --iterations=${CRI_DEFAULT_ITERATIONS} --verify=${CRI_DEFAULT_VERIFY}")
      set(PAGED_ARGS "--seq_len_kv_cache=64 --page_size=64 --use_paged_kv")
      set(TEST_PAGED_PREFILL ${TEST_BASE} ${PAGED_ARGS} "--seq_len_qo=256")
      set(TEST_PAGED_DECODE ${TEST_BASE} ${PAGED_ARGS} "--seq_len_qo=1")
      set(TEST_NO_PAGED_PREFILL ${TEST_BASE} "--seq_len_qo=256")
      set(TEST_NO_PAGED_DECODE ${TEST_BASE} "--seq_len_qo=1")
      set(TEST_CACHED_KV_PREFILL ${TEST_BASE} "--seq_len_kv_cache=64" "--seq_len_qo=256")
      set(TEST_CACHED_KV_DECODE ${TEST_BASE} "--seq_len_kv_cache=64" "--seq_len_qo=1")
    else()
      set(TEST_PAGED_PREFILL "--use_paged_kv")
      set(TEST_PAGED_DECODE "--use_paged_kv")
      set(TEST_NO_PAGED_PREFILL "")
      set(TEST_NO_PAGED_DECODE "")
      set(TEST_CACHED_KV_PREFILL "--seq_len_kv_cache=64")
      set(TEST_CACHED_KV_DECODE "--seq_len_kv_cache=64")
    endif()
 
    if(INPUT_TYPE STREQUAL "bfloat16_t")
      set(INPUT_MACRO "IS_BFLOAT16")
    elseif(INPUT_TYPE STREQUAL "float_e5m2_t")
      set(INPUT_MACRO "IS_FLOAT_E5M2")
    elseif(INPUT_TYPE STREQUAL "float_e4m3_t")
      set(INPUT_MACRO "IS_FLOAT_E4M3")
    elseif(INPUT_TYPE STREQUAL "mx_float_e5m2_t")
      set(INPUT_MACRO "IS_MX_FLOAT_E5M2")
    elseif(INPUT_TYPE STREQUAL "mx_float_e4m3_t")
      set(INPUT_MACRO "IS_MX_FLOAT_E4M3")
    elseif(INPUT_TYPE STREQUAL "mx_float_e2m1_t")
      set(INPUT_MACRO "IS_MX_FLOAT_E2M1")
    elseif(INPUT_TYPE STREQUAL "float_e2m1_t")
      set(INPUT_MACRO "IS_FLOAT_E2M1")
    endif()

    if (INPUT_TYPE STREQUAL "mx_float_e5m2_t" OR  INPUT_TYPE STREQUAL "mx_float_e4m3_t" OR INPUT_TYPE STREQUAL "mx_float_e2m1_t")
      cutlass_example_add_executable(
        06_xe_fmha_fwd_prefill_${INPUT_TYPE}_hdim${HEAD_DIM}
        06_xe_fmha_fwd_mxfp.cpp
        TEST_COMMAND_OPTIONS
        TEST_NO_PAGED_PREFILL
      )

      cutlass_example_add_executable(
        06_xe_fmha_fwd_decode_${INPUT_TYPE}_hdim${HEAD_DIM}
        06_xe_fmha_fwd_mxfp.cpp
        TEST_COMMAND_OPTIONS
        TEST_NO_PAGED_DECODE
      )
    else()
      cutlass_example_add_executable(
        06_xe_fmha_fwd_prefill_${INPUT_TYPE}_hdim${HEAD_DIM}
        06_xe_fmha_fwd.cpp
        TEST_COMMAND_OPTIONS
        TEST_NO_PAGED_PREFILL
      )

      cutlass_example_add_executable(
        06_xe_fmha_fwd_decode_${INPUT_TYPE}_hdim${HEAD_DIM}
        06_xe_fmha_fwd.cpp
        TEST_COMMAND_OPTIONS
        TEST_NO_PAGED_DECODE
      )

      if (NOT HEAD_DIM STREQUAL 192)
        # specific test for persistent kernel
        cutlass_example_add_executable(
          06_xe_fmha_fwd_decode_persistent_${INPUT_TYPE}_hdim${HEAD_DIM}
          06_xe_fmha_fwd.cpp
          TEST_COMMAND_OPTIONS
          TEST_NO_PAGED_DECODE
        )
        target_compile_definitions(06_xe_fmha_fwd_decode_persistent_${INPUT_TYPE}_hdim${HEAD_DIM} PRIVATE HEAD_DIM=${HEAD_DIM} DECODE PERSISTENT SHOW_DIFF=1 INPUT_TYPE=${INPUT_TYPE} ${INPUT_MACRO})
      endif()
    endif()

    target_compile_definitions(06_xe_fmha_fwd_prefill_${INPUT_TYPE}_hdim${HEAD_DIM} PRIVATE HEAD_DIM=${HEAD_DIM} PREFILL SHOW_DIFF=1 INPUT_TYPE=${INPUT_TYPE} ${INPUT_MACRO})
    target_compile_definitions(06_xe_fmha_fwd_decode_${INPUT_TYPE}_hdim${HEAD_DIM} PRIVATE HEAD_DIM=${HEAD_DIM} DECODE SHOW_DIFF=1 INPUT_TYPE=${INPUT_TYPE} ${INPUT_MACRO})

    # --- Cached KV targets (CachedKV=true, PagedKV=false, split into separate binary) ---
    if (NOT INPUT_TYPE STREQUAL "mx_float_e5m2_t" AND NOT INPUT_TYPE STREQUAL "mx_float_e4m3_t" AND NOT INPUT_TYPE STREQUAL "mx_float_e2m1_t")
      cutlass_example_add_executable(
        06_xe_fmha_fwd_prefill_cached_kv_${INPUT_TYPE}_hdim${HEAD_DIM}
        06_xe_fmha_fwd_cached_kv.cpp
        TEST_COMMAND_OPTIONS
        TEST_CACHED_KV_PREFILL
      )
      target_compile_definitions(06_xe_fmha_fwd_prefill_cached_kv_${INPUT_TYPE}_hdim${HEAD_DIM} PRIVATE HEAD_DIM=${HEAD_DIM} PREFILL SHOW_DIFF=1 INPUT_TYPE=${INPUT_TYPE} ${INPUT_MACRO})

      cutlass_example_add_executable(
        06_xe_fmha_fwd_decode_cached_kv_${INPUT_TYPE}_hdim${HEAD_DIM}
        06_xe_fmha_fwd_cached_kv.cpp
        TEST_COMMAND_OPTIONS
        TEST_CACHED_KV_DECODE
      )
      target_compile_definitions(06_xe_fmha_fwd_decode_cached_kv_${INPUT_TYPE}_hdim${HEAD_DIM} PRIVATE HEAD_DIM=${HEAD_DIM} DECODE SHOW_DIFF=1 INPUT_TYPE=${INPUT_TYPE} ${INPUT_MACRO})
    endif()
  endforeach()

  cutlass_example_add_executable(
    06_xe_fmha_fwd_prefill_fp8kvfp16mma_hdim${HEAD_DIM}
    06_xe_fmha_fwd_fp8kvfp16mma.cpp
    TEST_COMMAND_OPTIONS
    TEST_NO_PAGED_PREFILL
  )
  target_compile_definitions(06_xe_fmha_fwd_prefill_fp8kvfp16mma_hdim${HEAD_DIM} PRIVATE HEAD_DIM=${HEAD_DIM} PREFILL SHOW_DIFF=1)
endforeach()
