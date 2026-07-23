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
using BmgFMHAPrefill_BF16_BF16_BF16_BF16_RCR_h64_NonCausal_FixedLen_CachedKV_PagedKV = FMHAConfigGen</*Mode*/FMHAMode::Prefill,
  /*ElementQ*/ cutlass::bfloat16_t, /*ElementK*/ cutlass::bfloat16_t, /*ElementV*/ cutlass::bfloat16_t, /*ElementO*/ cutlass::bfloat16_t,
  /*LayoutQ*/ cutlass::layout::RowMajor, /*LayoutK*/ cutlass::layout::ColumnMajor, /*LayoutV*/ cutlass::layout::RowMajor, /*LayoutO*/ cutlass::layout::RowMajor,
  /*ElementScale*/ float, /*Causal*/ false, /*VarLen*/ false, /*CachedKV*/ true, /*PagedKV*/ true, /*Persistent*/ false, /*UseScale*/ false, /*HeadDim*/ 64
>::type;

CUTLASS_CREATE_FMHA_BENCHMARK(BmgFMHAPrefill_BF16_BF16_BF16_BF16_RCR_h64_NonCausal_FixedLen_CachedKV_PagedKV);

/* ---------------------------------------- Custom Tiles ------------------------------------------ */
using CriFMHAPrefill_BF16_BF16_BF16_BF16_RCR_WgQ512K64V64_SgQ16K64_HDimQK64V128_NonCausal_FixedLen = FMHAConfigGenWithTileShape</*Mode*/FMHAMode::Prefill,
  /*ElementQ*/ cutlass::bfloat16_t, /*ElementK*/ cutlass::bfloat16_t, /*ElementV*/ cutlass::bfloat16_t, /*ElementO*/ cutlass::bfloat16_t,
  /*LayoutQ*/ cutlass::layout::RowMajor, /*LayoutK*/ cutlass::layout::ColumnMajor, /*LayoutV*/ cutlass::layout::RowMajor, /*LayoutO*/ cutlass::layout::RowMajor,
  /*ElementScale*/ float, /*Causal*/ false, /*VarLen*/ false, /*CachedKV*/ false, /*PagedKV*/ false, /*Persistent*/ false, /*UseScale*/ false, /*WgTileQ*/ 512, /*WgTileK*/ 64, /*WgTileV*/ 64,
  /*SgTileQ*/ 16, /*SgTileK*/ 64, /*HeadDimQK*/ 64, /*HeadDimV*/ 128
>::type;

using CriFMHAPrefill_BF16_BF16_BF16_BF16_RCR_WgQ512K64V64_SgQ16K64_HDimQK64V128_Causal_VarLen = FMHAConfigGenWithTileShape</*Mode*/FMHAMode::Prefill,
  /*ElementQ*/ cutlass::bfloat16_t, /*ElementK*/ cutlass::bfloat16_t, /*ElementV*/ cutlass::bfloat16_t, /*ElementO*/ cutlass::bfloat16_t,
  /*LayoutQ*/ cutlass::layout::RowMajor, /*LayoutK*/ cutlass::layout::ColumnMajor, /*LayoutV*/ cutlass::layout::RowMajor, /*LayoutO*/ cutlass::layout::RowMajor,
  /*ElementScale*/ float, /*Causal*/ true, /*VarLen*/ true, /*CachedKV*/ false, /*PagedKV*/ false, /*Persistent*/ false, /*UseScale*/ false, /*WgTileQ*/ 512, /*WgTileK*/ 64, /*WgTileV*/ 64,
  /*SgTileQ*/ 16, /*SgTileK*/ 64, /*HeadDimQK*/ 64, /*HeadDimV*/ 128
>::type;

using BmgFMHAPrefill_BF16_BF16_BF16_BF16_RCR_WgQ128K64V32_SgQ8K64_HDimQK32V64_NonCausal_FixedLen = FMHAConfigGenWithTileShape</*Mode*/FMHAMode::Prefill,
  /*ElementQ*/ cutlass::bfloat16_t, /*ElementK*/ cutlass::bfloat16_t, /*ElementV*/ cutlass::bfloat16_t, /*ElementO*/ cutlass::bfloat16_t,
  /*LayoutQ*/ cutlass::layout::RowMajor, /*LayoutK*/ cutlass::layout::ColumnMajor, /*LayoutV*/ cutlass::layout::RowMajor, /*LayoutO*/ cutlass::layout::RowMajor,
  /*ElementScale*/ float, /*Causal*/ false, /*VarLen*/ false, /*CachedKV*/ false, /*PagedKV*/ false, /*Persistent*/ false, /*UseScale*/ false, /*WgTileQ*/ 128, /*WgTileK*/ 64, /*WgTileV*/ 32,
  /*SgTileQ*/ 8, /*SgTileK*/ 64, /*HeadDimQK*/ 32, /*HeadDimV*/ 64
>::type;

using BmgFMHAPrefill_BF16_BF16_BF16_BF16_RCR_WgQ128K64V32_SgQ8K64_HDimQK32V64_Causal_VarLen = FMHAConfigGenWithTileShape</*Mode*/FMHAMode::Prefill,
  /*ElementQ*/ cutlass::bfloat16_t, /*ElementK*/ cutlass::bfloat16_t, /*ElementV*/ cutlass::bfloat16_t, /*ElementO*/ cutlass::bfloat16_t,
  /*LayoutQ*/ cutlass::layout::RowMajor, /*LayoutK*/ cutlass::layout::ColumnMajor, /*LayoutV*/ cutlass::layout::RowMajor, /*LayoutO*/ cutlass::layout::RowMajor,
  /*ElementScale*/ float, /*Causal*/ true, /*VarLen*/ true, /*CachedKV*/ false, /*PagedKV*/ false, /*Persistent*/ false, /*UseScale*/ false, /*WgTileQ*/ 128, /*WgTileK*/ 64, /*WgTileV*/ 32,
  /*SgTileQ*/ 8, /*SgTileK*/ 64, /*HeadDimQK*/ 32, /*HeadDimV*/ 64
>::type;

CUTLASS_CREATE_FMHA_BENCHMARK(CriFMHAPrefill_BF16_BF16_BF16_BF16_RCR_WgQ512K64V64_SgQ16K64_HDimQK64V128_NonCausal_FixedLen);
CUTLASS_CREATE_FMHA_BENCHMARK(CriFMHAPrefill_BF16_BF16_BF16_BF16_RCR_WgQ512K64V64_SgQ16K64_HDimQK64V128_Causal_VarLen);
CUTLASS_CREATE_FMHA_BENCHMARK(BmgFMHAPrefill_BF16_BF16_BF16_BF16_RCR_WgQ128K64V32_SgQ8K64_HDimQK32V64_NonCausal_FixedLen);
CUTLASS_CREATE_FMHA_BENCHMARK(BmgFMHAPrefill_BF16_BF16_BF16_BF16_RCR_WgQ128K64V32_SgQ8K64_HDimQK32V64_Causal_VarLen);


static void register_flash_attention_prefill_benchmarks_bf16() {
  CUTLASS_FMHA_BENCHMARK(BmgFMHAPrefill_BF16_BF16_BF16_BF16_RCR_h64_NonCausal_FixedLen_CachedKV_PagedKV);

  CUTLASS_FMHA_BENCHMARK(CriFMHAPrefill_BF16_BF16_BF16_BF16_RCR_WgQ512K64V64_SgQ16K64_HDimQK64V128_NonCausal_FixedLen);
  CUTLASS_FMHA_BENCHMARK(CriFMHAPrefill_BF16_BF16_BF16_BF16_RCR_WgQ512K64V64_SgQ16K64_HDimQK64V128_Causal_VarLen);

  CUTLASS_FMHA_BENCHMARK(BmgFMHAPrefill_BF16_BF16_BF16_BF16_RCR_WgQ128K64V32_SgQ8K64_HDimQK32V64_NonCausal_FixedLen);
  CUTLASS_FMHA_BENCHMARK(BmgFMHAPrefill_BF16_BF16_BF16_BF16_RCR_WgQ128K64V32_SgQ8K64_HDimQK32V64_Causal_VarLen);
}
