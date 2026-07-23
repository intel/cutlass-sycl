/***************************************************************************************************
 * Copyright (c) 2024 - 2024 Codeplay Software Ltd. All rights reserved.
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
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

 #pragma once

#include "gemm_configuration_sycl.hpp"
#include "dual_gemm_benchmark_runner.hpp"
#include "cutlass/epilogue/thread/activation.h"

using Scheduler = cutlass::gemm::device::Scheduler;

template <
  typename TileShape,
  typename Tiler,
  typename GmemTiledCopyA,
  typename GmemTiledCopyB>
using Gemm_Bench_BF16FP32_RRR = cutlass::gemm::device::GemmConfiguration<
    cutlass::arch::IntelXe,
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    TileShape, Scheduler::Gemm, Tiler,
    GmemTiledCopyA, GmemTiledCopyB>;

using BmgGemm_BF16FP32_TileShape_512_256_32 = Shape<_512, _256, _32>;
using BmgGemm_BF16FP32_Tile_512_256_32 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cute::bfloat16_t>>, Layout<BmgGemm_BF16FP32_TileShape_512_256_32>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using BmgGemmBF16BF16FP32_RRR_TileShape_512_256_32 = Gemm_Bench_BF16FP32_RRR<BmgGemm_BF16FP32_TileShape_512_256_32, BmgGemm_BF16FP32_Tile_512_256_32, void, void>;
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmBF16BF16FP32_RRR_TileShape_512_256_32);

template <
  typename TileShape,
  typename Tiler,
  typename GmemTiledCopyA,
  typename GmemTiledCopyB>
using Gemm_Bench_BF16BF16BF16_RRR = cutlass::gemm::device::GemmConfiguration<
    cutlass::arch::IntelXe,
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    cutlass::bfloat16_t,
    TileShape, Scheduler::Gemm, Tiler,
    GmemTiledCopyA, GmemTiledCopyB,
    cutlass::epilogue::fusion::LinearCombination<cutlass::bfloat16_t, cutlass::bfloat16_t>>;

using BmgGemm_BF16BF16BF16_TileShape_512_256_64 = Shape<_512, _256, _64>;
using BmgGemm_BF16BF16BF16_Tile_512_256_64 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, cutlass::bfloat16_t, cute::bfloat16_t>>, Layout<BmgGemm_BF16BF16BF16_TileShape_512_256_64>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using BmgGemmBF16BF16BF16_RRR_TileShape_512_256_64 = Gemm_Bench_BF16BF16BF16_RRR<BmgGemm_BF16BF16BF16_TileShape_512_256_64, BmgGemm_BF16BF16BF16_Tile_512_256_64, void, void>;
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmBF16BF16BF16_RRR_TileShape_512_256_64);

// StreamK variant matching example 03_bmg_gemm_streamk (TileShape 256x256x32,
// KernelXeCooperative + StreamKScheduler).
template <
  typename TileShape,
  typename Tiler,
  typename GmemTiledCopyA,
  typename GmemTiledCopyB>
using Gemm_Bench_BF16FP32_RRR_StreamK = cutlass::gemm::device::GemmConfiguration<
    cutlass::arch::IntelXe,
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    TileShape, Scheduler::GemmStreamK, Tiler,
    GmemTiledCopyA, GmemTiledCopyB>;

using BmgGemm_BF16FP32_TileShape_256_256_32 = Shape<_256, _256, _32>;
using BmgGemm_BF16FP32_Tile_256_256_32 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cute::bfloat16_t>>, Layout<BmgGemm_BF16FP32_TileShape_256_256_32>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using BmgGemmBF16BF16FP32_StreamK_TileShape_256_256_32 = Gemm_Bench_BF16FP32_RRR_StreamK<BmgGemm_BF16FP32_TileShape_256_256_32, BmgGemm_BF16FP32_Tile_256_256_32, void, void>;
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmBF16BF16FP32_StreamK_TileShape_256_256_32);

using BmgGemm_BF16FP32_TileShape_8_128_32 = Shape<_8, _128, _32>;
using BmgGemm_BF16FP32_Tile_8_128_32 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cute::bfloat16_t>>, Layout<BmgGemm_BF16FP32_TileShape_8_128_32>, Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using BmgGemmBF16BF16FP32_RRR_TileShape_8_128_32 = Gemm_Bench_BF16FP32_RRR<BmgGemm_BF16FP32_TileShape_8_128_32, BmgGemm_BF16FP32_Tile_8_128_32, void, void>;
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmBF16BF16FP32_RRR_TileShape_8_128_32);

using BmgGemm_BF16FP32_TileShape_16_64_32 = Shape<_16, _64, _32>;
using BmgGemm_BF16FP32_Tile_16_64_32 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cute::bfloat16_t>>, Layout<BmgGemm_BF16FP32_TileShape_16_64_32>, Layout<Shape<_2, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using BmgGemmBF16BF16FP32_RRR_TileShape_16_64_32 = Gemm_Bench_BF16FP32_RRR<BmgGemm_BF16FP32_TileShape_16_64_32, BmgGemm_BF16FP32_Tile_16_64_32, void, void>;
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmBF16BF16FP32_RRR_TileShape_16_64_32);

template <
  typename TileShape,
  typename Tiler,
  typename GmemTiledCopyA,
  typename GmemTiledCopyB>
using Gemm_Bench_FP16FP16FP32_RRR = cutlass::gemm::device::GemmConfiguration<
    cutlass::arch::IntelXe,
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    TileShape, Scheduler::Gemm, Tiler,
    GmemTiledCopyA, GmemTiledCopyB>;

using CriGemm_FP16FP16FP32_TileShape_512_256_32 = Shape<_512, _256, _32>;
using CriGemm_FP16FP16FP32_Tile_512_256_32 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cutlass::half_t>>, Layout<CriGemm_FP16FP16FP32_TileShape_512_256_32>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using CriGemmFP16FP16FP32_RRR_TileShape_512_256_32 = Gemm_Bench_FP16FP16FP32_RRR<CriGemm_FP16FP16FP32_TileShape_512_256_32, CriGemm_FP16FP16FP32_Tile_512_256_32, void, void>;
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmFP16FP16FP32_RRR_TileShape_512_256_32);

template <
  typename TileShape,
  typename Tiler,
  typename GmemTiledCopyA,
  typename GmemTiledCopyB>
using Gemm_Bench_FP32FP32FP32_RRR = cutlass::gemm::device::GemmConfiguration<
    cutlass::arch::IntelXe,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    TileShape, Scheduler::Gemm, Tiler,
    GmemTiledCopyA, GmemTiledCopyB>;

using CriGemm_FP32FP32FP32_TileShape_512_256_16 = Shape<_512, _256, _16>;
using CriGemm_FP32FP32FP32_Tile_512_256_16 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cutlass::tfloat32_t>>, Layout<CriGemm_FP32FP32FP32_TileShape_512_256_16>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using CriGemmFP32FP32FP32_RRR_TileShape_512_256_16 = Gemm_Bench_FP32FP32FP32_RRR<CriGemm_FP32FP32FP32_TileShape_512_256_16, CriGemm_FP32FP32FP32_Tile_512_256_16, void, void>;
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmFP32FP32FP32_RRR_TileShape_512_256_16);

template <
  typename TileShape,
  typename Tiler,
  typename GmemTiledCopyA,
  typename GmemTiledCopyB>
using Gemm_Bench_TF32TF32FP32_RRR = cutlass::gemm::device::GemmConfiguration<
    cutlass::arch::IntelXe,
    cutlass::tfloat32_t, cutlass::layout::RowMajor,
    cutlass::tfloat32_t, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    TileShape, Scheduler::Gemm, Tiler,
    GmemTiledCopyA, GmemTiledCopyB>;

using CriGemm_TF32TF32FP32_TileShape_512_256_16 = Shape<_512, _256, _16>;
using CriGemm_TF32TF32FP32_Tile_512_256_16 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cutlass::tfloat32_t>>, Layout<CriGemm_TF32TF32FP32_TileShape_512_256_16>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using CriGemmTF32TF32FP32_RRR_TileShape_512_256_16 = Gemm_Bench_TF32TF32FP32_RRR<CriGemm_TF32TF32FP32_TileShape_512_256_16, CriGemm_TF32TF32FP32_Tile_512_256_16, void, void>;
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmTF32TF32FP32_RRR_TileShape_512_256_16);

// Dual GEMM (sync from example 07_bmg_dual_gemm): one shared A matrix multiplied by two B
// matrices, fused through a SiLU activation epilogue. Uses MainloopIntelXeXMX16<2> + two
// linear-combination epilogues. TileShape <_128,_128,_64>, MMA XE_8x16x16_F32BF16BF16F32_TT.
using BmgDualGemm_BF16FP32_TileShape_128_128_64 = Shape<_128, _128, _64>;
using BmgDualGemm_BF16FP32_Tile_128_128_64 = typename TiledMMAHelper<
    MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>,
    Layout<BmgDualGemm_BF16FP32_TileShape_128_128_64>,
    Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using BmgDualGemmBF16BF16FP32_RRR_TileShape_128_128_64 = cutlass::gemm::device::DualGemmConfiguration<
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    float,               cutlass::layout::RowMajor,
    BmgDualGemm_BF16FP32_TileShape_128_128_64,
    BmgDualGemm_BF16FP32_Tile_128_128_64,
    XE_2D_U16x16x32_LD_N, XE_2D_U16x32x32_LD_V, 2>;
CUTLASS_CREATE_DUAL_GEMM_BENCHMARK(BmgDualGemmBF16BF16FP32_RRR_TileShape_128_128_64);

// ---------------------------------------------------------------------------
// Activation-fused epilogue variants (sync from example 05_bmg_gemm_with_epilogues:
// 05_bmg_gemm_with_epilogue_{relu,silu,gelu}). Same BF16 GEMM as the baseline but
// with D = Act(alpha * A*B + beta * C). Uses MainloopXeL1Staged + IntelXeGeneric
// epilogue, TileShape <_256,_256,_32>, MMA XE_DPAS_TT<8,float,bf16>.
//
// NOTE: the generic benchmark verify() compares against a plain GEMM reference and
// does NOT apply the activation, so it only matches for the baseline. Verification
// is disabled under CUTLASS_TEST_FOR_CRI (the simulator path), so these cases are
// intended to be run on the CRI simulator (see input_files/cri/input_epilogue_gemm.in).
// ---------------------------------------------------------------------------
template <
  template <class> class ActivationFn,
  typename TileShape,
  typename Tiler,
  typename GmemTiledCopyA,
  typename GmemTiledCopyB>
using Gemm_Bench_BF16FP32_RRR_EltAct = cutlass::gemm::device::GemmConfiguration<
    cutlass::arch::IntelXe,
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    TileShape, Scheduler::Gemm, Tiler,
    GmemTiledCopyA, GmemTiledCopyB,
    cutlass::epilogue::fusion::LinCombEltAct<ActivationFn, float, float, float, float,
        cutlass::FloatRoundStyle::round_to_nearest>>;

using BmgGemm_EltAct_BF16FP32_TileShape_256_256_32 = Shape<_256, _256, _32>;
using BmgGemm_EltAct_BF16FP32_Tile_256_256_32 = typename TiledMMAHelper<
    MMA_Atom<XE_DPAS_TT<8, float, cute::bfloat16_t>>,
    Layout<BmgGemm_EltAct_BF16FP32_TileShape_256_256_32>,
    Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;

using BmgGemmReLUBF16BF16FP32_RRR_TileShape_256_256_32 = Gemm_Bench_BF16FP32_RRR_EltAct<
    cutlass::epilogue::thread::ReLu, BmgGemm_EltAct_BF16FP32_TileShape_256_256_32,
    BmgGemm_EltAct_BF16FP32_Tile_256_256_32, void, void>;
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmReLUBF16BF16FP32_RRR_TileShape_256_256_32);

using BmgGemmSiLUBF16BF16FP32_RRR_TileShape_256_256_32 = Gemm_Bench_BF16FP32_RRR_EltAct<
    cutlass::epilogue::thread::SiLu, BmgGemm_EltAct_BF16FP32_TileShape_256_256_32,
    BmgGemm_EltAct_BF16FP32_Tile_256_256_32, void, void>;
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmSiLUBF16BF16FP32_RRR_TileShape_256_256_32);

using BmgGemmGELUBF16BF16FP32_RRR_TileShape_256_256_32 = Gemm_Bench_BF16FP32_RRR_EltAct<
    cutlass::epilogue::thread::GELU, BmgGemm_EltAct_BF16FP32_TileShape_256_256_32,
    BmgGemm_EltAct_BF16FP32_Tile_256_256_32, void, void>;
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmGELUBF16BF16FP32_RRR_TileShape_256_256_32);

#if defined(SYCL_INTEL_TARGET) && (SYCL_INTEL_TARGET == 35)

using E4M3ElementType = cutlass::mx_float8_t<float_e4m3_t>;
using E4M3ElementInputA = typename E4M3ElementType::DataType;
using E4M3ElementInputB = typename E4M3ElementType::DataType;
using E4M3ElementScale = typename E4M3ElementType::ScaleFactorType;
template <
  typename TileShape,
  typename Tiler,
  typename GmemTiledCopyA,
  typename GmemTiledCopyB>
using BLockScalingGemm_Bench_E4M3E4M3FP32_RRR = cutlass::gemm::device::BlockScalingGemmConfiguration<
    cutlass::arch::IntelXe,
    E4M3ElementInputA, cutlass::layout::RowMajor,
    E4M3ElementInputB, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    E4M3ElementScale,
    cute::Stride<_1, int64_t, int64_t>,
    float,
    TileShape, Scheduler::Gemm, Tiler,
    GmemTiledCopyA, GmemTiledCopyB, void, void>;

using CriBLockScalingGemm_E4M3E4M3FP32_TileShape_256_256_32 = Shape<_256, _256, _32>;
using CriBLockScalingGemm_E4M3E4M3FP32_Tile_256_256_32 = typename TiledMMAHelper<MMA_Atom<XE_BDPAS_TT<8, float, E4M3ElementInputA>>, 
        Layout<CriBLockScalingGemm_E4M3E4M3FP32_TileShape_256_256_32>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using CriBLockScalingGemm_E4M3E4M3FP32_RRR_TileShape_256_256_32 = BLockScalingGemm_Bench_E4M3E4M3FP32_RRR<CriBLockScalingGemm_E4M3E4M3FP32_TileShape_256_256_32, CriBLockScalingGemm_E4M3E4M3FP32_Tile_256_256_32, void, void>;

CUTLASS_CREATE_GEMM_BENCHMARK(CriBLockScalingGemm_E4M3E4M3FP32_RRR_TileShape_256_256_32);

using CriBLockScalingGemm_E4M3E4M3FP32_TileShape_512_256_64 = Shape<_512, _256, _64>;
using CriBLockScalingGemm_E4M3E4M3FP32_Tile_512_256_64 = typename TiledMMAHelper<MMA_Atom<XE_BDPAS_TT<8, float, E4M3ElementInputA>>,
        Layout<CriBLockScalingGemm_E4M3E4M3FP32_TileShape_512_256_64>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using CriBLockScalingGemm_E4M3E4M3FP32_RRR_TileShape_512_256_64 = BLockScalingGemm_Bench_E4M3E4M3FP32_RRR<CriBLockScalingGemm_E4M3E4M3FP32_TileShape_512_256_64, CriBLockScalingGemm_E4M3E4M3FP32_Tile_512_256_64, void, void>;

CUTLASS_CREATE_GEMM_BENCHMARK(CriBLockScalingGemm_E4M3E4M3FP32_RRR_TileShape_512_256_64);

template <
  typename TileShape,
  typename Tiler,
  typename GmemTiledCopyA,
  typename GmemTiledCopyB>
using BLockScalingGemm_Bench_E4M3E4M3BF16_RRR = cutlass::gemm::device::BlockScalingGemmConfiguration<
    cutlass::arch::IntelXe,
    E4M3ElementInputA, cutlass::layout::RowMajor,
    E4M3ElementInputB, cutlass::layout::RowMajor,
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    E4M3ElementScale,
    cute::Stride<_1, int64_t, int64_t>,
    cutlass::bfloat16_t,
    TileShape, Scheduler::Gemm, Tiler,
    GmemTiledCopyA, GmemTiledCopyB, void, void,
    cutlass::epilogue::fusion::LinearCombination<cutlass::bfloat16_t, cutlass::bfloat16_t>>;

using CriBLockScalingGemm_E4M3E4M3BF16_TileShape_512_256_128 = Shape<_512, _256, _128>;
using CriBLockScalingGemm_E4M3E4M3BF16_Tile_512_256_128 = typename TiledMMAHelper<MMA_Atom<XE_BDPAS_TT<8, cutlass::bfloat16_t, E4M3ElementInputA>>,
  Layout<CriBLockScalingGemm_E4M3E4M3BF16_TileShape_512_256_128>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using CriBLockScalingGemm_E4M3E4M3BF16_RRR_TileShape_512_256_128 = BLockScalingGemm_Bench_E4M3E4M3BF16_RRR<CriBLockScalingGemm_E4M3E4M3BF16_TileShape_512_256_128, CriBLockScalingGemm_E4M3E4M3BF16_Tile_512_256_128, void, void>;

CUTLASS_CREATE_GEMM_BENCHMARK(CriBLockScalingGemm_E4M3E4M3BF16_RRR_TileShape_512_256_128);

using E5M2ElementType = cutlass::mx_float8_t<float_e5m2_t>;
using E5M2ElementInputA = typename E5M2ElementType::DataType;
using E5M2ElementInputB = typename E5M2ElementType::DataType;
using E5M2ElementScale = typename E5M2ElementType::ScaleFactorType;
template <
  typename TileShape,
  typename Tiler,
  typename GmemTiledCopyA,
  typename GmemTiledCopyB>
using BLockScalingGemm_Bench_E5M2E5M2FP32_RRR = cutlass::gemm::device::BlockScalingGemmConfiguration<
    cutlass::arch::IntelXe,
    E5M2ElementInputA, cutlass::layout::RowMajor,
    E5M2ElementInputB, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    E5M2ElementScale,
    cute::Stride<_1, int64_t, int64_t>,
    float,
    TileShape, Scheduler::Gemm, Tiler,
    GmemTiledCopyA, GmemTiledCopyB, void, void>;

using CriBLockScalingGemm_E5M2E5M2FP32_TileShape_256_256_32 = Shape<_256, _256, _32>;
using CriBLockScalingGemm_E5M2E5M2FP32_Tile_256_256_32 = typename TiledMMAHelper<MMA_Atom<XE_BDPAS_TT<8, float, E5M2ElementInputA>>, 
        Layout<CriBLockScalingGemm_E5M2E5M2FP32_TileShape_256_256_32>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using CriBLockScalingGemm_E5M2E5M2FP32_RRR_TileShape_256_256_32 = BLockScalingGemm_Bench_E5M2E5M2FP32_RRR<CriBLockScalingGemm_E5M2E5M2FP32_TileShape_256_256_32, CriBLockScalingGemm_E5M2E5M2FP32_Tile_256_256_32, void, void>;

CUTLASS_CREATE_GEMM_BENCHMARK(CriBLockScalingGemm_E5M2E5M2FP32_RRR_TileShape_256_256_32);

using CriBLockScalingGemm_E5M2E5M2FP32_TileShape_512_256_64 = Shape<_512, _256, _64>;
using CriBLockScalingGemm_E5M2E5M2FP32_Tile_512_256_64 = typename TiledMMAHelper<MMA_Atom<XE_BDPAS_TT<8, float, E5M2ElementInputA>>,
        Layout<CriBLockScalingGemm_E5M2E5M2FP32_TileShape_512_256_64>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using CriBLockScalingGemm_E5M2E5M2FP32_RRR_TileShape_512_256_64 = BLockScalingGemm_Bench_E5M2E5M2FP32_RRR<CriBLockScalingGemm_E5M2E5M2FP32_TileShape_512_256_64, CriBLockScalingGemm_E5M2E5M2FP32_Tile_512_256_64, void, void>;

CUTLASS_CREATE_GEMM_BENCHMARK(CriBLockScalingGemm_E5M2E5M2FP32_RRR_TileShape_512_256_64);

template <
  typename TileShape,
  typename Tiler,
  typename GmemTiledCopyA,
  typename GmemTiledCopyB>
using BLockScalingGemm_Bench_E5M2E5M2BF16_RRR = cutlass::gemm::device::BlockScalingGemmConfiguration<
    cutlass::arch::IntelXe,
    E5M2ElementInputA, cutlass::layout::RowMajor,
    E5M2ElementInputB, cutlass::layout::RowMajor,
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    E5M2ElementScale,
    cute::Stride<_1, int64_t, int64_t>,
    cutlass::bfloat16_t,
    TileShape, Scheduler::Gemm, Tiler,
    GmemTiledCopyA, GmemTiledCopyB, void, void,
    cutlass::epilogue::fusion::LinearCombination<cutlass::bfloat16_t, cutlass::bfloat16_t>>;

using CriBLockScalingGemm_E5M2E5M2BF16_TileShape_512_256_128 = Shape<_512, _256, _128>;
using CriBLockScalingGemm_E5M2E5M2BF16_Tile_512_256_128 = typename TiledMMAHelper<MMA_Atom<XE_BDPAS_TT<8, cutlass::bfloat16_t, E5M2ElementInputA>>,
  Layout<CriBLockScalingGemm_E5M2E5M2BF16_TileShape_512_256_128>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using CriBLockScalingGemm_E5M2E5M2BF16_RRR_TileShape_512_256_128 = BLockScalingGemm_Bench_E5M2E5M2BF16_RRR<CriBLockScalingGemm_E5M2E5M2BF16_TileShape_512_256_128, CriBLockScalingGemm_E5M2E5M2BF16_Tile_512_256_128, void, void>;

CUTLASS_CREATE_GEMM_BENCHMARK(CriBLockScalingGemm_E5M2E5M2BF16_RRR_TileShape_512_256_128);

using E2M1ElementType = cutlass::mx_float4_t<float_e2m1_t>;
using E2M1ElementInputA = typename E2M1ElementType::DataType;
using E2M1ElementInputB = typename E2M1ElementType::DataType;
using E2M1ElementScale = typename E2M1ElementType::ScaleFactorType;
template <
  typename TileShape,
  typename Tiler,
  typename GmemTiledCopyA,
  typename GmemTiledCopyB>
using BLockScalingGemm_Bench_E2M1E2M1FP32_RCR = cutlass::gemm::device::BlockScalingGemmConfiguration<
    cutlass::arch::IntelXe,
    E2M1ElementInputA, cutlass::layout::RowMajor,
    E2M1ElementInputB, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor,
    E2M1ElementScale,
    cute::Stride<_1, int64_t, int64_t>,
    float,
    TileShape, Scheduler::Gemm, Tiler,
    GmemTiledCopyA, GmemTiledCopyB, void, void>;

using CriBLockScalingGemm_E2M1E2M1FP32_TileShape_256_256_64 = Shape<_256, _256, _64>;
using CriBLockScalingGemm_E2M1E2M1FP32_Tile_256_256_64 = typename TiledMMAHelper<MMA_Atom<XE_BDPAS_TT<8, float, E2M1ElementInputA>>, 
        Layout<CriBLockScalingGemm_E2M1E2M1FP32_TileShape_256_256_64>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using CriBLockScalingGemm_E2M1E2M1FP32_RCR_TileShape_256_256_64 = BLockScalingGemm_Bench_E2M1E2M1FP32_RCR<CriBLockScalingGemm_E2M1E2M1FP32_TileShape_256_256_64, CriBLockScalingGemm_E2M1E2M1FP32_Tile_256_256_64, void, void>;

CUTLASS_CREATE_GEMM_BENCHMARK(CriBLockScalingGemm_E2M1E2M1FP32_RCR_TileShape_256_256_64);

using CriBLockScalingGemm_E2M1E2M1FP32_TileShape_512_256_128 = Shape<_512, _256, _128>;
using CriBLockScalingGemm_E2M1E2M1FP32_Tile_512_256_128 = typename TiledMMAHelper<MMA_Atom<XE_BDPAS_TT<8, float, E2M1ElementInputA>>,
        Layout<CriBLockScalingGemm_E2M1E2M1FP32_TileShape_512_256_128>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using CriBLockScalingGemm_E2M1E2M1FP32_RCR_TileShape_512_256_128 = BLockScalingGemm_Bench_E2M1E2M1FP32_RCR<CriBLockScalingGemm_E2M1E2M1FP32_TileShape_512_256_128, CriBLockScalingGemm_E2M1E2M1FP32_Tile_512_256_128, void, void>;

CUTLASS_CREATE_GEMM_BENCHMARK(CriBLockScalingGemm_E2M1E2M1FP32_RCR_TileShape_512_256_128);

template <
  typename TileShape,
  typename Tiler,
  typename GmemTiledCopyA,
  typename GmemTiledCopyB>
using BLockScalingGemm_Bench_E2M1E2M1BF16_RCR = cutlass::gemm::device::BlockScalingGemmConfiguration<
    cutlass::arch::IntelXe,
    E2M1ElementInputA, cutlass::layout::RowMajor,
    E2M1ElementInputB, cutlass::layout::ColumnMajor,
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    E2M1ElementScale,
    cute::Stride<_1, int64_t, int64_t>,
    cutlass::bfloat16_t,
    TileShape, Scheduler::Gemm, Tiler,
    GmemTiledCopyA, GmemTiledCopyB, void, void,
    cutlass::epilogue::fusion::LinearCombination<cutlass::bfloat16_t, cutlass::bfloat16_t>>;

using CriBLockScalingGemm_E2M1E2M1BF16_TileShape_512_256_256 = Shape<_512, _256, _256>;
using CriBLockScalingGemm_E2M1E2M1BF16_Tile_512_256_256 = typename TiledMMAHelper<MMA_Atom<XE_BDPAS_TT<8, cutlass::bfloat16_t, E2M1ElementInputA>>,
  Layout<CriBLockScalingGemm_E2M1E2M1BF16_TileShape_512_256_256>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using CriBLockScalingGemm_E2M1E2M1BF16_RCR_TileShape_512_256_256 = BLockScalingGemm_Bench_E2M1E2M1BF16_RCR<CriBLockScalingGemm_E2M1E2M1BF16_TileShape_512_256_256, CriBLockScalingGemm_E2M1E2M1BF16_Tile_512_256_256, void, void>;

CUTLASS_CREATE_GEMM_BENCHMARK(CriBLockScalingGemm_E2M1E2M1BF16_RCR_TileShape_512_256_256);

template <
  typename TileShape,
  typename Tiler,
  typename GmemTiledCopyA,
  typename GmemTiledCopyB>
using Gemm_Bench_E5M2E5M2FP32_RRR = cutlass::gemm::device::GemmConfiguration<
    cutlass::arch::IntelXe,
    cutlass::float_e5m2_t, cutlass::layout::RowMajor,
    cutlass::float_e5m2_t, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    TileShape, Scheduler::Gemm, Tiler,
    GmemTiledCopyA, GmemTiledCopyB>;

using CriGemm_E5M2E5M2FP32_TileShape_256_256_32 = Shape<_256, _256, _32>;
using CriGemm_E5M2E5M2FP32_Tile_256_256_32 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cutlass::float_e5m2_t>>, Layout<CriGemm_E5M2E5M2FP32_TileShape_256_256_32>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using CriGemmE5M2E5M2FP32_RRR_TileShape_256_256_32 = Gemm_Bench_E5M2E5M2FP32_RRR<CriGemm_E5M2E5M2FP32_TileShape_256_256_32, CriGemm_E5M2E5M2FP32_Tile_256_256_32, void, void>;

CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmE5M2E5M2FP32_RRR_TileShape_256_256_32);

using CriGemm_E5M2E5M2FP32_TileShape_512_256_64 = Shape<_512, _256, _64>;
using CriGemm_E5M2E5M2FP32_Tile_512_256_64 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cutlass::float_e5m2_t>>, Layout<CriGemm_E5M2E5M2FP32_TileShape_512_256_64>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using CriGemmE5M2E5M2FP32_RRR_TileShape_512_256_64 = Gemm_Bench_E5M2E5M2FP32_RRR<CriGemm_E5M2E5M2FP32_TileShape_512_256_64, CriGemm_E5M2E5M2FP32_Tile_512_256_64, void, void>;

CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmE5M2E5M2FP32_RRR_TileShape_512_256_64);

template <
  typename TileShape,
  typename Tiler,
  typename GmemTiledCopyA,
  typename GmemTiledCopyB>
using Gemm_Bench_E5M2E5M2BF16_RRR = cutlass::gemm::device::GemmConfiguration<
    cutlass::arch::IntelXe,
    cutlass::float_e5m2_t, cutlass::layout::RowMajor,
    cutlass::float_e5m2_t, cutlass::layout::RowMajor,
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    cutlass::bfloat16_t,
    TileShape, Scheduler::Gemm, Tiler,
    GmemTiledCopyA, GmemTiledCopyB,
    cutlass::epilogue::fusion::LinearCombination<cutlass::bfloat16_t, cutlass::bfloat16_t>>;

using CriGemm_E5M2E5M2BF16_TileShape_512_256_128 = Shape<_512, _256, _128>;
using CriGemm_E5M2E5M2BF16_Tile_512_256_128 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, cutlass::bfloat16_t, cutlass::float_e5m2_t>>, Layout<CriGemm_E5M2E5M2BF16_TileShape_512_256_128>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using CriGemmE5M2E5M2BF16_RRR_TileShape_512_256_128 = Gemm_Bench_E5M2E5M2BF16_RRR<CriGemm_E5M2E5M2BF16_TileShape_512_256_128, CriGemm_E5M2E5M2BF16_Tile_512_256_128, void, void>;

CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmE5M2E5M2BF16_RRR_TileShape_512_256_128);

template <
  typename TileShape,
  typename Tiler,
  typename GmemTiledCopyA,
  typename GmemTiledCopyB>
using Gemm_Bench_E4M3E4M3FP32_RRR = cutlass::gemm::device::GemmConfiguration<
    cutlass::arch::IntelXe,
    cutlass::float_e4m3_t, cutlass::layout::RowMajor,
    cutlass::float_e4m3_t, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    TileShape, Scheduler::Gemm, Tiler,
    GmemTiledCopyA, GmemTiledCopyB>;

using CriGemm_E4M3E4M3FP32_TileShape_256_256_32 = Shape<_256, _256, _32>;
using CriGemm_E4M3E4M3FP32_Tile_256_256_32 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cutlass::float_e4m3_t>>, Layout<CriGemm_E4M3E4M3FP32_TileShape_256_256_32>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using CriGemmE4M3E4M3FP32_RRR_TileShape_256_256_32 = Gemm_Bench_E4M3E4M3FP32_RRR<CriGemm_E4M3E4M3FP32_TileShape_256_256_32, CriGemm_E4M3E4M3FP32_Tile_256_256_32, void, void>;

CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmE4M3E4M3FP32_RRR_TileShape_256_256_32);

using CriGemm_E4M3E4M3FP32_TileShape_512_256_64 = Shape<_512, _256, _64>;
using CriGemm_E4M3E4M3FP32_Tile_512_256_64 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cutlass::float_e4m3_t>>, Layout<CriGemm_E4M3E4M3FP32_TileShape_512_256_64>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using CriGemmE4M3E4M3FP32_RRR_TileShape_512_256_64 = Gemm_Bench_E4M3E4M3FP32_RRR<CriGemm_E4M3E4M3FP32_TileShape_512_256_64, CriGemm_E4M3E4M3FP32_Tile_512_256_64, void, void>;

CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmE4M3E4M3FP32_RRR_TileShape_512_256_64);

/////////////////////////////////////////////////////////////////////////
// W8A8 FP8 -> FP16-MMA fast path benchmark cases (sync from
// examples/08_bmg_gemm_f8). FP8 inputs are upcast to FP16, MMA via
// XE_8x16x16_F32F16F16F32_TT, MainloopIntelW8A8 dispatch policy.
// Pair-compare these against the native-FP8 DPAS cases above on the
// same shape to measure the two pipelines.
/////////////////////////////////////////////////////////////////////////
template <
  typename TileShape,
  typename Tiler,
  typename GmemTiledCopyA,
  typename GmemTiledCopyB>
using Gemm_Bench_W8A8_E4M3FP16MMA_RRR = cutlass::gemm::device::W8A8GemmConfiguration<
    cutlass::float_e4m3_t, cutlass::layout::RowMajor,
    cutlass::float_e4m3_t, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    TileShape,
    Tiler,
    GmemTiledCopyA, GmemTiledCopyB>;

template <
  typename TileShape,
  typename Tiler,
  typename GmemTiledCopyA,
  typename GmemTiledCopyB>
using Gemm_Bench_W8A8_E5M2FP16MMA_RRR = cutlass::gemm::device::W8A8GemmConfiguration<
    cutlass::float_e5m2_t, cutlass::layout::RowMajor,
    cutlass::float_e5m2_t, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    TileShape,
    Tiler,
    GmemTiledCopyA, GmemTiledCopyB>;

// TileShape matches example 08_bmg_gemm_f8 (<_256, _256, _32>).
using CriGemm_W8A8_E4M3FP16MMA_TileShape_256_256_32 = Shape<_256, _256, _32>;
using CriGemm_W8A8_E4M3FP16MMA_Tile_256_256_32 = typename TiledMMAHelper<MMA_Atom<XE_8x16x16_F32F16F16F32_TT>, Layout<CriGemm_W8A8_E4M3FP16MMA_TileShape_256_256_32>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using CriGemm_W8A8_E4M3E4M3FP16MMA_RRR_TileShape_256_256_32 = Gemm_Bench_W8A8_E4M3FP16MMA_RRR<
    CriGemm_W8A8_E4M3FP16MMA_TileShape_256_256_32, CriGemm_W8A8_E4M3FP16MMA_Tile_256_256_32,
    XE_2D_U8x32x32_LD_N, XE_2D_U8x32x32_LD_V>;
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemm_W8A8_E4M3E4M3FP16MMA_RRR_TileShape_256_256_32);

using CriGemm_W8A8_E5M2FP16MMA_TileShape_256_256_32 = Shape<_256, _256, _32>;
using CriGemm_W8A8_E5M2FP16MMA_Tile_256_256_32 = typename TiledMMAHelper<MMA_Atom<XE_8x16x16_F32F16F16F32_TT>, Layout<CriGemm_W8A8_E5M2FP16MMA_TileShape_256_256_32>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using CriGemm_W8A8_E5M2E5M2FP16MMA_RRR_TileShape_256_256_32 = Gemm_Bench_W8A8_E5M2FP16MMA_RRR<
    CriGemm_W8A8_E5M2FP16MMA_TileShape_256_256_32, CriGemm_W8A8_E5M2FP16MMA_Tile_256_256_32,
    XE_2D_U8x32x32_LD_N, XE_2D_U8x32x32_LD_V>;
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemm_W8A8_E5M2E5M2FP16MMA_RRR_TileShape_256_256_32);

template <
  typename TileShape,
  typename Tiler,
  typename GmemTiledCopyA,
  typename GmemTiledCopyB>
using Gemm_Bench_E4M3E4M3BF16_RRR = cutlass::gemm::device::GemmConfiguration<
    cutlass::arch::IntelXe,
    cutlass::float_e4m3_t, cutlass::layout::RowMajor,
    cutlass::float_e4m3_t, cutlass::layout::RowMajor,
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    cutlass::bfloat16_t,
    TileShape, Scheduler::Gemm, Tiler,
    GmemTiledCopyA, GmemTiledCopyB,
    cutlass::epilogue::fusion::LinearCombination<cutlass::bfloat16_t, cutlass::bfloat16_t>>;

using CriGemm_E4M3E4M3BF16_TileShape_512_256_128 = Shape<_512, _256, _128>;
using CriGemm_E4M3E4M3BF16_Tile_512_256_128 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, cutlass::bfloat16_t, cutlass::float_e4m3_t>>, Layout<CriGemm_E4M3E4M3BF16_TileShape_512_256_128>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using CriGemmE4M3E4M3BF16_RRR_TileShape_512_256_128 = Gemm_Bench_E4M3E4M3BF16_RRR<CriGemm_E4M3E4M3BF16_TileShape_512_256_128, CriGemm_E4M3E4M3BF16_Tile_512_256_128, void, void>;

CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmE4M3E4M3BF16_RRR_TileShape_512_256_128);
template <
  typename TileShape,
  typename Tiler,
  typename GmemTiledCopyA,
  typename GmemTiledCopyB>
using Gemm_Bench_E2M1E2M1FP32_RCR = cutlass::gemm::device::GemmConfiguration<
    cutlass::arch::IntelXe,
    cutlass::float_e2m1_t, cutlass::layout::RowMajor,
    cutlass::float_e2m1_t, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor,
    float,
    TileShape, Scheduler::Gemm, Tiler,
    GmemTiledCopyA, GmemTiledCopyB>;

using CriGemm_E2M1E2M1FP32_TileShape_256_256_64 = Shape<_256, _256, _64>;
using CriGemm_E2M1E2M1FP32_Tile_256_256_64 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cutlass::float_e2m1_t>>, Layout<CriGemm_E2M1E2M1FP32_TileShape_256_256_64>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using CriGemmE2M1E2M1FP32_RCR_TileShape_256_256_64 = Gemm_Bench_E2M1E2M1FP32_RCR<CriGemm_E2M1E2M1FP32_TileShape_256_256_64, CriGemm_E2M1E2M1FP32_Tile_256_256_64, void, void>;
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmE2M1E2M1FP32_RCR_TileShape_256_256_64);

using CriGemm_E2M1E2M1FP32_TileShape_512_256_128 = Shape<_512, _256, _128>;
using CriGemm_E2M1E2M1FP32_Tile_512_256_128 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cutlass::float_e2m1_t>>, Layout<CriGemm_E2M1E2M1FP32_TileShape_512_256_128>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using CriGemmE2M1E2M1FP32_RCR_TileShape_512_256_128 = Gemm_Bench_E2M1E2M1FP32_RCR<CriGemm_E2M1E2M1FP32_TileShape_512_256_128, CriGemm_E2M1E2M1FP32_Tile_512_256_128, void, void>;
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmE2M1E2M1FP32_RCR_TileShape_512_256_128);

template <
  typename TileShape,
  typename Tiler,
  typename GmemTiledCopyA,
  typename GmemTiledCopyB>
using Gemm_Bench_E2M1E2M1BF16_RCR = cutlass::gemm::device::GemmConfiguration<
    cutlass::arch::IntelXe,
    cutlass::float_e2m1_t, cutlass::layout::RowMajor,
    cutlass::float_e2m1_t, cutlass::layout::ColumnMajor,
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    cutlass::bfloat16_t,
    TileShape, Scheduler::Gemm, Tiler,
    GmemTiledCopyA, GmemTiledCopyB,
    cutlass::epilogue::fusion::LinearCombination<cutlass::bfloat16_t, cutlass::bfloat16_t>>;

using CriGemm_E2M1E2M1BF16_TileShape_512_256_256 = Shape<_512, _256, _256>;
using CriGemm_E2M1E2M1BF16_Tile_512_256_256 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, cutlass::bfloat16_t, cutlass::float_e2m1_t>>, Layout<CriGemm_E2M1E2M1BF16_TileShape_512_256_256>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using CriGemmE2M1E2M1BF16_RCR_TileShape_512_256_256 = Gemm_Bench_E2M1E2M1BF16_RCR<CriGemm_E2M1E2M1BF16_TileShape_512_256_256, CriGemm_E2M1E2M1BF16_Tile_512_256_256, void, void>;

CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmE2M1E2M1BF16_RCR_TileShape_512_256_256);

#endif

static void register_gemm_benchmarks() {
  // TODO: support sglang cases
  CUTLASS_BENCHMARK(BmgGemmBF16BF16FP32_RRR_TileShape_512_256_32);
  CUTLASS_BENCHMARK(BmgGemmBF16BF16BF16_RRR_TileShape_512_256_64);
  CUTLASS_BENCHMARK(BmgGemmBF16BF16FP32_StreamK_TileShape_256_256_32);
  CUTLASS_BENCHMARK(BmgGemmBF16BF16FP32_RRR_TileShape_8_128_32);
  CUTLASS_BENCHMARK(BmgGemmBF16BF16FP32_RRR_TileShape_16_64_32);
  CUTLASS_BENCHMARK(CriGemmFP16FP16FP32_RRR_TileShape_512_256_32);
  CUTLASS_BENCHMARK(CriGemmFP32FP32FP32_RRR_TileShape_512_256_16);
  CUTLASS_BENCHMARK(CriGemmTF32TF32FP32_RRR_TileShape_512_256_16);
  // Dual GEMM (sync from example 07_bmg_dual_gemm)
  CUTLASS_BENCHMARK(BmgDualGemmBF16BF16FP32_RRR_TileShape_128_128_64);
  // Activation-fused epilogues (sync from example 05_bmg_gemm_with_epilogues)
  CUTLASS_BENCHMARK(BmgGemmReLUBF16BF16FP32_RRR_TileShape_256_256_32);
  CUTLASS_BENCHMARK(BmgGemmSiLUBF16BF16FP32_RRR_TileShape_256_256_32);
  CUTLASS_BENCHMARK(BmgGemmGELUBF16BF16FP32_RRR_TileShape_256_256_32);
#if defined(SYCL_INTEL_TARGET) && (SYCL_INTEL_TARGET == 35)
  CUTLASS_BENCHMARK(CriGemmE5M2E5M2FP32_RRR_TileShape_256_256_32);
  CUTLASS_BENCHMARK(CriGemmE4M3E4M3FP32_RRR_TileShape_256_256_32);
  CUTLASS_BENCHMARK(CriGemmE2M1E2M1FP32_RCR_TileShape_256_256_64);
  CUTLASS_BENCHMARK(CriBLockScalingGemm_E4M3E4M3FP32_RRR_TileShape_256_256_32);
  CUTLASS_BENCHMARK(CriBLockScalingGemm_E5M2E5M2FP32_RRR_TileShape_256_256_32);
  CUTLASS_BENCHMARK(CriBLockScalingGemm_E2M1E2M1FP32_RCR_TileShape_256_256_64);
  // Tile shapes aligned with sycl example 00_bmg_gemm_fp4_fp8
  CUTLASS_BENCHMARK(CriGemmE4M3E4M3FP32_RRR_TileShape_512_256_64);
  CUTLASS_BENCHMARK(CriGemmE5M2E5M2FP32_RRR_TileShape_512_256_64);
  CUTLASS_BENCHMARK(CriGemmE2M1E2M1FP32_RCR_TileShape_512_256_128);
  CUTLASS_BENCHMARK(CriBLockScalingGemm_E4M3E4M3FP32_RRR_TileShape_512_256_64);
  CUTLASS_BENCHMARK(CriBLockScalingGemm_E5M2E5M2FP32_RRR_TileShape_512_256_64);
  CUTLASS_BENCHMARK(CriBLockScalingGemm_E2M1E2M1FP32_RCR_TileShape_512_256_128);
  // W8A8 FP8 -> FP16-MMA fast path (sync from example 08_bmg_gemm_f8)
  CUTLASS_BENCHMARK(CriGemm_W8A8_E4M3E4M3FP16MMA_RRR_TileShape_256_256_32);
  CUTLASS_BENCHMARK(CriGemm_W8A8_E5M2E5M2FP16MMA_RRR_TileShape_256_256_32);
  CUTLASS_BENCHMARK(CriGemmE4M3E4M3BF16_RRR_TileShape_512_256_128);
  CUTLASS_BENCHMARK(CriGemmE5M2E5M2BF16_RRR_TileShape_512_256_128);
  CUTLASS_BENCHMARK(CriGemmE2M1E2M1BF16_RCR_TileShape_512_256_256);
  CUTLASS_BENCHMARK(CriBLockScalingGemm_E4M3E4M3BF16_RRR_TileShape_512_256_128);
  CUTLASS_BENCHMARK(CriBLockScalingGemm_E5M2E5M2BF16_RRR_TileShape_512_256_128);
  CUTLASS_BENCHMARK(CriBLockScalingGemm_E2M1E2M1BF16_RCR_TileShape_512_256_256);
#endif
}
