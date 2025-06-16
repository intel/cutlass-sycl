/***************************************************************************************************
* Copyright (c) 2025 - 2025 Codeplay Software Ltd. All rights reserved.
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
#include "fmha_decode_configuration.hpp"

using namespace cutlass::flash_attention;

using FMHADecodeBF16BF16FP32FP32_RCR_NonPaged_KVTile1024_h128_Causal_FixedLen = FMHADecodeConfigGen<cutlass::bfloat16_t, float, float, true, false, Shape_h128<1024, 16>, false>::type;
using FMHADecodeBF16BF16FP32FP32_RCR_NonPaged_KVTile1024_h128_Causal_VarLen = FMHADecodeConfigGen<cutlass::bfloat16_t, float, float, true, true, Shape_h128<1024, 16>, false>::type;
using FMHADecodeBF16BF16FP32FP32_RCR_NonPaged_KVTile1024_h128_NonCausal_FixedLen = FMHADecodeConfigGen<cutlass::bfloat16_t, float, float, false, false, Shape_h128<1024, 16>, false>::type;
using FMHADecodeBF16BF16FP32FP32_RCR_NonPaged_KVTile1024_h128_NonCausal_VarLen = FMHADecodeConfigGen<cutlass::bfloat16_t, float, float, false, true, Shape_h128<1024, 16>, false>::type;

using FMHADecodeFP16FP16FP32_RCR_NonPaged_KVTile1024_h128_Causal_FixedLen = FMHADecodeConfigGen<cutlass::half_t, float, float, true, false, Shape_h128<1024, 16>, false>::type;
using FMHADecodeFP16FP16FP32_RCR_NonPaged_KVTile1024_h128_Causal_VarLen = FMHADecodeConfigGen<cutlass::half_t, float, float, true, true, Shape_h128<1024, 16>, false>::type;
using FMHADecodeFP16FP16FP32_RCR_NonPaged_KVTile1024_h128_NonCausal_FixedLen = FMHADecodeConfigGen<cutlass::half_t, float, float, false, false, Shape_h128<1024, 16>, false>::type;
using FMHADecodeFP16FP16FP32_RCR_NonPaged_KVTile1024_h128_NonCausal_VarLen = FMHADecodeConfigGen<cutlass::half_t, float, float, false, true, Shape_h128<1024, 16>, false>::type;


CUTLASS_CREATE_FMHA_DECODE_BENCHMARK(FMHADecodeBF16BF16FP32FP32_RCR_NonPaged_KVTile1024_h128_Causal_FixedLen);
CUTLASS_CREATE_FMHA_DECODE_BENCHMARK(FMHADecodeBF16BF16FP32FP32_RCR_NonPaged_KVTile1024_h128_NonCausal_FixedLen);
CUTLASS_CREATE_FMHA_DECODE_BENCHMARK(FMHADecodeFP16FP16FP32_RCR_NonPaged_KVTile1024_h128_Causal_FixedLen);
CUTLASS_CREATE_FMHA_DECODE_BENCHMARK(FMHADecodeFP16FP16FP32_RCR_NonPaged_KVTile1024_h128_NonCausal_FixedLen);

CUTLASS_CREATE_FMHA_DECODE_BENCHMARK(FMHADecodeBF16BF16FP32FP32_RCR_NonPaged_KVTile1024_h128_Causal_VarLen);
CUTLASS_CREATE_FMHA_DECODE_BENCHMARK(FMHADecodeBF16BF16FP32FP32_RCR_NonPaged_KVTile1024_h128_NonCausal_VarLen);
CUTLASS_CREATE_FMHA_DECODE_BENCHMARK(FMHADecodeFP16FP16FP32_RCR_NonPaged_KVTile1024_h128_Causal_VarLen);
CUTLASS_CREATE_FMHA_DECODE_BENCHMARK(FMHADecodeFP16FP16FP32_RCR_NonPaged_KVTile1024_h128_NonCausal_VarLen);

static void register_flash_attention_decode_benchmarks_nonpaged_h128_fp32_fp32_1024() {
  CUTLASS_FMHA_DECODE_BENCHMARK(FMHADecodeBF16BF16FP32FP32_RCR_NonPaged_KVTile1024_h128_Causal_FixedLen);
  CUTLASS_FMHA_DECODE_BENCHMARK(FMHADecodeBF16BF16FP32FP32_RCR_NonPaged_KVTile1024_h128_NonCausal_FixedLen);
  CUTLASS_FMHA_DECODE_BENCHMARK(FMHADecodeFP16FP16FP32_RCR_NonPaged_KVTile1024_h128_Causal_FixedLen);
  CUTLASS_FMHA_DECODE_BENCHMARK(FMHADecodeFP16FP16FP32_RCR_NonPaged_KVTile1024_h128_NonCausal_FixedLen);

  CUTLASS_FMHA_DECODE_BENCHMARK(FMHADecodeBF16BF16FP32FP32_RCR_NonPaged_KVTile1024_h128_Causal_VarLen);
  CUTLASS_FMHA_DECODE_BENCHMARK(FMHADecodeBF16BF16FP32FP32_RCR_NonPaged_KVTile1024_h128_NonCausal_VarLen);
  CUTLASS_FMHA_DECODE_BENCHMARK(FMHADecodeFP16FP16FP32_RCR_NonPaged_KVTile1024_h128_Causal_VarLen);
  CUTLASS_FMHA_DECODE_BENCHMARK(FMHADecodeFP16FP16FP32_RCR_NonPaged_KVTile1024_h128_NonCausal_VarLen);
}
