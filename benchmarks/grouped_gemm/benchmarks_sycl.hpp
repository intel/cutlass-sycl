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

#include "grouped_gemm_configuration_sycl.hpp"

template <
  typename TileShape,
  typename Tiler,
  typename GmemTiledCopyA,
  typename GmemTiledCopyB>
using GroupedGemm_Bench_BF16FP32_RRR = cutlass::gemm::device::GroupedGemmConfiguration<
    cutlass::arch::IntelXe,
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    TileShape, Tiler,
    GmemTiledCopyA, GmemTiledCopyB>;

using BmgGroupedGemm_BF16FP32_TileShape_512_256_32 = Shape<_512, _256, _32>;
using BmgGroupedGemm_BF16FP32_Tile_512_256_32 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cute::bfloat16_t>>, Layout<BmgGroupedGemm_BF16FP32_TileShape_512_256_32>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using BmgGroupedGemmBF16BF16FP32_RRR_TileShape_512_256_32 = GroupedGemm_Bench_BF16FP32_RRR<BmgGroupedGemm_BF16FP32_TileShape_512_256_32, BmgGroupedGemm_BF16FP32_Tile_512_256_32, void, void>;
CUTLASS_CREATE_GROUPED_GEMM_BENCHMARK(BmgGroupedGemmBF16BF16FP32_RRR_TileShape_512_256_32);

// TileShape aligned with example 04_bmg_grouped_gemm (Shape<_256, _256, _32>).
using BmgGroupedGemm_BF16FP32_TileShape_256_256_32 = Shape<_256, _256, _32>;
using BmgGroupedGemm_BF16FP32_Tile_256_256_32 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cute::bfloat16_t>>, Layout<BmgGroupedGemm_BF16FP32_TileShape_256_256_32>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using BmgGroupedGemmBF16BF16FP32_RRR_TileShape_256_256_32 = GroupedGemm_Bench_BF16FP32_RRR<BmgGroupedGemm_BF16FP32_TileShape_256_256_32, BmgGroupedGemm_BF16FP32_Tile_256_256_32, void, void>;
CUTLASS_CREATE_GROUPED_GEMM_BENCHMARK(BmgGroupedGemmBF16BF16FP32_RRR_TileShape_256_256_32);

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
using BLockScalingGroupedGemm_Bench_E4M3E4M3FP32_RRR = cutlass::gemm::device::BlockScalingGroupedGemmConfiguration<
    cutlass::arch::IntelXe,
    E4M3ElementInputA, cutlass::layout::RowMajor,
    E4M3ElementInputB, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    E4M3ElementScale,
    cute::Stride<_1, int64_t, int64_t>,
    float,
    TileShape, Tiler,
    GmemTiledCopyA, GmemTiledCopyB, void, void>;

using CriBLockScalingGroupedGemm_E4M3E4M3FP32_TileShape_256_256_32 = Shape<_256, _256, _32>;
using CriBLockScalingGroupedGemm_E4M3E4M3FP32_Tile_256_256_32 = typename TiledMMAHelper<MMA_Atom<XE_BDPAS_TT<8, float, E4M3ElementInputA>>, 
        Layout<CriBLockScalingGroupedGemm_E4M3E4M3FP32_TileShape_256_256_32>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using CriBLockScalingGroupedGemm_E4M3E4M3FP32_RRR_TileShape_256_256_32 = BLockScalingGroupedGemm_Bench_E4M3E4M3FP32_RRR<CriBLockScalingGroupedGemm_E4M3E4M3FP32_TileShape_256_256_32, CriBLockScalingGroupedGemm_E4M3E4M3FP32_Tile_256_256_32, void, void>;

CUTLASS_CREATE_GROUPED_GEMM_BENCHMARK(CriBLockScalingGroupedGemm_E4M3E4M3FP32_RRR_TileShape_256_256_32);

// TileShape matching example 51_xe35_block_scaled_grouped_gemm_e4m3 (GroupSize=32 run).
using CriBLockScalingGroupedGemm_E4M3E4M3FP32_TileShape_512_256_64 = Shape<_512, _256, _64>;
using CriBLockScalingGroupedGemm_E4M3E4M3FP32_Tile_512_256_64 = typename TiledMMAHelper<MMA_Atom<XE_BDPAS_TT<8, float, E4M3ElementInputA>>,
        Layout<CriBLockScalingGroupedGemm_E4M3E4M3FP32_TileShape_512_256_64>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using CriBLockScalingGroupedGemm_E4M3E4M3FP32_RRR_TileShape_512_256_64 = BLockScalingGroupedGemm_Bench_E4M3E4M3FP32_RRR<CriBLockScalingGroupedGemm_E4M3E4M3FP32_TileShape_512_256_64, CriBLockScalingGroupedGemm_E4M3E4M3FP32_Tile_512_256_64, void, void>;

CUTLASS_CREATE_GROUPED_GEMM_BENCHMARK(CriBLockScalingGroupedGemm_E4M3E4M3FP32_RRR_TileShape_512_256_64);

using E5M2ElementType = cutlass::mx_float8_t<float_e5m2_t>;
using E5M2ElementInputA = typename E5M2ElementType::DataType;
using E5M2ElementInputB = typename E5M2ElementType::DataType;
using E5M2ElementScale = typename E5M2ElementType::ScaleFactorType;
template <
  typename TileShape,
  typename Tiler,
  typename GmemTiledCopyA,
  typename GmemTiledCopyB>
using BLockScalingGroupedGemm_Bench_E5M2E5M2FP32_RRR = cutlass::gemm::device::BlockScalingGroupedGemmConfiguration<
    cutlass::arch::IntelXe,
    E5M2ElementInputA, cutlass::layout::RowMajor,
    E5M2ElementInputB, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    E5M2ElementScale,
    cute::Stride<_1, int64_t, int64_t>,
    float,
    TileShape, Tiler,
    GmemTiledCopyA, GmemTiledCopyB, void, void>;

using CriBLockScalingGroupedGemm_E5M2E5M2FP32_TileShape_256_256_32 = Shape<_256, _256, _32>;
using CriBLockScalingGroupedGemm_E5M2E5M2FP32_Tile_256_256_32 = typename TiledMMAHelper<MMA_Atom<XE_BDPAS_TT<8, float, E5M2ElementInputA>>, 
        Layout<CriBLockScalingGroupedGemm_E5M2E5M2FP32_TileShape_256_256_32>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using CriBLockScalingGroupedGemm_E5M2E5M2FP32_RRR_TileShape_256_256_32 = BLockScalingGroupedGemm_Bench_E5M2E5M2FP32_RRR<CriBLockScalingGroupedGemm_E5M2E5M2FP32_TileShape_256_256_32, CriBLockScalingGroupedGemm_E5M2E5M2FP32_Tile_256_256_32, void, void>;

CUTLASS_CREATE_GROUPED_GEMM_BENCHMARK(CriBLockScalingGroupedGemm_E5M2E5M2FP32_RRR_TileShape_256_256_32);
using E2M1ElementType = cutlass::mx_float4_t<float_e2m1_t>;
using E2M1ElementInputA = typename E2M1ElementType::DataType;
using E2M1ElementInputB = typename E2M1ElementType::DataType;
using E2M1ElementScale = typename E2M1ElementType::ScaleFactorType;
template <
  typename TileShape,
  typename Tiler,
  typename GmemTiledCopyA,
  typename GmemTiledCopyB>
using BLockScalingGroupedGemm_Bench_E2M1E2M1FP32_RCR = cutlass::gemm::device::BlockScalingGroupedGemmConfiguration<
    cutlass::arch::IntelXe,
    E2M1ElementInputA, cutlass::layout::RowMajor,
    E2M1ElementInputB, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor,
    E2M1ElementScale,
    cute::Stride<_1, int64_t, int64_t>,
    float,
    TileShape, Tiler,
    GmemTiledCopyA, GmemTiledCopyB, void, void>;

using CriBLockScalingGroupedGemm_E2M1E2M1FP32_TileShape_256_256_64 = Shape<_256, _256, _64>;
using CriBLockScalingGroupedGemm_E2M1E2M1FP32_Tile_256_256_64 = typename TiledMMAHelper<MMA_Atom<XE_BDPAS_TT<8, float, E2M1ElementInputA>>, 
        Layout<CriBLockScalingGroupedGemm_E2M1E2M1FP32_TileShape_256_256_64>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using CriBLockScalingGroupedGemm_E2M1E2M1FP32_RCR_TileShape_256_256_64 = BLockScalingGroupedGemm_Bench_E2M1E2M1FP32_RCR<CriBLockScalingGroupedGemm_E2M1E2M1FP32_TileShape_256_256_64, CriBLockScalingGroupedGemm_E2M1E2M1FP32_Tile_256_256_64, void, void>;

CUTLASS_CREATE_GROUPED_GEMM_BENCHMARK(CriBLockScalingGroupedGemm_E2M1E2M1FP32_RCR_TileShape_256_256_64);

template <
  typename TileShape,
  typename Tiler,
  typename GmemTiledCopyA,
  typename GmemTiledCopyB>
using GroupedGemm_Bench_E5M2E5M2FP32_RRR = cutlass::gemm::device::GroupedGemmConfiguration<
    cutlass::arch::IntelXe,
    cutlass::float_e5m2_t, cutlass::layout::RowMajor,
    cutlass::float_e5m2_t, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    TileShape, Tiler,
    GmemTiledCopyA, GmemTiledCopyB>;

using CriGroupedGemm_E5M2E5M2FP32_TileShape_256_256_32 = Shape<_256, _256, _32>;
using CriGroupedGemm_E5M2E5M2FP32_Tile_256_256_32 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cutlass::float_e5m2_t>>, Layout<CriGroupedGemm_E5M2E5M2FP32_TileShape_256_256_32>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using CriGroupedGemm_E5M2E5M2FP32_RRR_TileShape_256_256_32 = GroupedGemm_Bench_E5M2E5M2FP32_RRR<CriGroupedGemm_E5M2E5M2FP32_TileShape_256_256_32, CriGroupedGemm_E5M2E5M2FP32_Tile_256_256_32, void, void>;

CUTLASS_CREATE_GROUPED_GEMM_BENCHMARK(CriGroupedGemm_E5M2E5M2FP32_RRR_TileShape_256_256_32);
template <
  typename TileShape,
  typename Tiler,
  typename GmemTiledCopyA,
  typename GmemTiledCopyB>
using GroupedGemm_Bench_E4M3E4M3FP32_RRR = cutlass::gemm::device::GroupedGemmConfiguration<
    cutlass::arch::IntelXe,
    cutlass::float_e4m3_t, cutlass::layout::RowMajor,
    cutlass::float_e4m3_t, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    TileShape, Tiler,
    GmemTiledCopyA, GmemTiledCopyB>;

using CriGroupedGemm_E4M3E4M3FP32_TileShape_256_256_32 = Shape<_256, _256, _32>;
using CriGroupedGemm_E4M3E4M3FP32_Tile_256_256_32 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cutlass::float_e4m3_t>>, Layout<CriGroupedGemm_E4M3E4M3FP32_TileShape_256_256_32>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using CriGroupedGemm_E4M3E4M3FP32_RRR_TileShape_256_256_32 = GroupedGemm_Bench_E4M3E4M3FP32_RRR<CriGroupedGemm_E4M3E4M3FP32_TileShape_256_256_32, CriGroupedGemm_E4M3E4M3FP32_Tile_256_256_32, void, void>;

CUTLASS_CREATE_GROUPED_GEMM_BENCHMARK(CriGroupedGemm_E4M3E4M3FP32_RRR_TileShape_256_256_32);
template <
  typename TileShape,
  typename Tiler,
  typename GmemTiledCopyA,
  typename GmemTiledCopyB>
using GroupedGemm_Bench_E2M1E2M1FP32_RCR = cutlass::gemm::device::GroupedGemmConfiguration<
    cutlass::arch::IntelXe,
    cutlass::float_e2m1_t, cutlass::layout::RowMajor,
    cutlass::float_e2m1_t, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor,
    float,
    TileShape, Tiler,
    GmemTiledCopyA, GmemTiledCopyB>;

using CriGroupedGemm_E2M1E2M1FP32_TileShape_256_256_64 = Shape<_256, _256, _64>;
using CriGroupedGemm_E2M1E2M1FP32_Tile_256_256_64 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cutlass::float_e2m1_t>>, Layout<CriGroupedGemm_E2M1E2M1FP32_TileShape_256_256_64>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using CriGroupedGemm_E2M1E2M1FP32_RCR_TileShape_256_256_64 = GroupedGemm_Bench_E2M1E2M1FP32_RCR<CriGroupedGemm_E2M1E2M1FP32_TileShape_256_256_64, CriGroupedGemm_E2M1E2M1FP32_Tile_256_256_64, void, void>;

CUTLASS_CREATE_GROUPED_GEMM_BENCHMARK(CriGroupedGemm_E2M1E2M1FP32_RCR_TileShape_256_256_64);

#endif

static void register_grouped_gemm_benchmarks() {
  CUTLASS_BENCHMARK(BmgGroupedGemmBF16BF16FP32_RRR_TileShape_512_256_32);
  CUTLASS_BENCHMARK(BmgGroupedGemmBF16BF16FP32_RRR_TileShape_256_256_32);
#if defined(SYCL_INTEL_TARGET) && (SYCL_INTEL_TARGET == 35)
  CUTLASS_BENCHMARK(CriGroupedGemm_E5M2E5M2FP32_RRR_TileShape_256_256_32);
  CUTLASS_BENCHMARK(CriGroupedGemm_E4M3E4M3FP32_RRR_TileShape_256_256_32);
  CUTLASS_BENCHMARK(CriGroupedGemm_E2M1E2M1FP32_RCR_TileShape_256_256_64);
  CUTLASS_BENCHMARK(CriBLockScalingGroupedGemm_E4M3E4M3FP32_RRR_TileShape_256_256_32);
  CUTLASS_BENCHMARK(CriBLockScalingGroupedGemm_E4M3E4M3FP32_RRR_TileShape_512_256_64);
  CUTLASS_BENCHMARK(CriBLockScalingGroupedGemm_E5M2E5M2FP32_RRR_TileShape_256_256_32);
  CUTLASS_BENCHMARK(CriBLockScalingGroupedGemm_E2M1E2M1FP32_RCR_TileShape_256_256_64);
#endif
}