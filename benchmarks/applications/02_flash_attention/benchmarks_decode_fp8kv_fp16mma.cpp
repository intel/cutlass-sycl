/***************************************************************************************************
 * Copyright (c) 2026 Intel Corporation. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#include "benchmark_runner.hpp"
#include "fmha_configuration.hpp"

using namespace cutlass::flash_attention;

/* ---------------------------------------- HeadDim = 64 ------------------------------------------ */
using CriFMHADecode_FP16_E4M3_E4M3_BF16_RCR_h64_NonCausal_FixedLen = FMHAConfigGen</*Mode*/FMHAMode::Decode,
  /*ElementQ*/ cutlass::half_t, /*ElementK*/ cutlass::float_e4m3_t, /*ElementV*/ cutlass::float_e4m3_t, /*ElementO*/ cutlass::bfloat16_t,
  /*LayoutQ*/ cutlass::layout::RowMajor, /*LayoutK*/ cutlass::layout::ColumnMajor, /*LayoutV*/ cutlass::layout::RowMajor, /*LayoutO*/ cutlass::layout::RowMajor,
  /*ElementScale*/ float, /*Causal*/ false, /*VarLen*/ false, /*CachedKV*/ false, /*PagedKV*/ false, /*Persistent*/ false, /*UseScale*/ false, /*HeadDim*/ 64
>::type;

CUTLASS_CREATE_FMHA_BENCHMARK(CriFMHADecode_FP16_E4M3_E4M3_BF16_RCR_h64_NonCausal_FixedLen);

/* ---------------------------------------- HeadDim = 64 ------------------------------------------ */

static void register_flash_attention_decode_benchmarks_fp8kv_fp16mma() {
  CUTLASS_FMHA_BENCHMARK(CriFMHADecode_FP16_E4M3_E4M3_BF16_RCR_h64_NonCausal_FixedLen);
}