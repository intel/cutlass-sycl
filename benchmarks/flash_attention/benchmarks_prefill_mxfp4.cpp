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

using mxfp4 = cutlass::mx_float4_t<float_e2m1_t>;

/* ---------------------------------------- HeadDim = 64 ------------------------------------------ */

using CriFMHAPrefill_MXFP4_MXFP4_BF16_BF16_RCR_h64_Causal_VarLen = FMHAConfigGen</*Mode*/FMHAMode::Prefill,
  /*ElementQ*/ mxfp4::DataType, /*ElementK*/ mxfp4::DataType, /*ElementV*/ cutlass::bfloat16_t, /*ElementO*/ cutlass::bfloat16_t,
  /*LayoutQ*/ cutlass::layout::RowMajor, /*LayoutK*/ cutlass::layout::ColumnMajor, /*LayoutV*/ cutlass::layout::RowMajor, /*LayoutO*/ cutlass::layout::RowMajor,
  /*ElementScale*/ mxfp4::ScaleFactorType, /*Causal*/ true, /*VarLen*/ true, /*CachedKV*/ false, /*PagedKV*/ false, /*Persistent*/ false, /*UseScale*/ true, /*HeadDim*/ 64
>::type;

CUTLASS_CREATE_FMHA_BENCHMARK(CriFMHAPrefill_MXFP4_MXFP4_BF16_BF16_RCR_h64_Causal_VarLen);
/* ---------------------------------------- HeadDim = 64 ------------------------------------------ */

static void register_flash_attention_prefill_benchmarks_mxfp4() {
  CUTLASS_FMHA_BENCHMARK(CriFMHAPrefill_MXFP4_MXFP4_BF16_BF16_RCR_h64_Causal_VarLen);
}