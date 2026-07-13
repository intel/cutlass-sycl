// AUTO-GENERATED tile list. Tile-sweep set per dtype — every entry maps to a config
// typedef already defined in applications/moe_grouped_gemm/runner/moe_gemm_runner.hpp.
// The X-macro drives BOTH benchmark registration (moe_benchmark_runner.hpp) and the
// runtime name->config dispatch table (moe_kernel_launch.cpp).
//
// Each dtype's tiles live in a named list macro (MOE_TILE_LIST_<TAG>). The active
// MOE_TILE_X_LIST is the concatenation of whichever MOE_DTYPE_<TAG>s this binary was
// built with:
//   - Single-dtype binary (one -DMOE_DTYPE_<TAG>, the CMakeLists MOE_BENCH_VARIANTS
//     default): X-list = that one dtype's tiles (unchanged behavior).
//   - Merged binary (-DMOE_BENCH_ALL defines all five MOE_DTYPE_<TAG>s): X-list =
//     all dtypes' tiles registered into the one runtime name->config table. The
//     registry keys on the (dtype-prefixed) tile NAME, so there is no collision.
//
// WIDE-N TILES (N in {320, 640, 896}) — now supported.
//   These tiles (N not a multiple of 256) previously failed a CuTe "Shape/Stride
//   Divisibility Condition" static_assert at compile time. ROOT CAUSE:
//   moe_launch_timed hardcoded the D output STORE atom (XE_STORE_2D<16,8,32>) and
//   built the store copy over the full WG (M,N) tile, so the layout composition in
//   make_block_2d_copy_X required N to be a multiple of 256. FIX: the D store atom is
//   now void -> auto-selected by block_2d_selector, which gcd-sizes the 2D-block atom
//   to divide any N (the same way the production Xe epilogue does). The hot-loop A/B
//   load atoms are unchanged.

#pragma once

// -DMOE_BENCH_ALL builds one binary with every dtype's device kernels (merged
// image). It simply enables all per-dtype tags; each block below then contributes
// its tiles to the combined X-list.
#ifdef MOE_BENCH_ALL
#define MOE_DTYPE_FP8_TENSOR_E4M3
#define MOE_DTYPE_MXFP8_E4M3
#define MOE_DTYPE_MXFP8_E5M2
#define MOE_DTYPE_MXFP4_E2M1
#define MOE_DTYPE_BF16
#endif

// ---- Per-dtype named lists ----

#ifdef MOE_DTYPE_FP8_TENSOR_E4M3
// PO_sanity CI trim: only the tile referenced by
// config_files/01_grouped_gemm/cri/PO_sanity/fp8.in (TileShape_256_256_64) is
// built. All other FP8 tiles are commented out to cut CI build time; restore
// the lines below to re-enable the full sweep.
#define MOE_TILE_LIST_FP8_TENSOR_E4M3 \
  X(CriGroupedGemm_E4M3E4M3BF16_RRR_TileShape_256_256_64, cutlass::moe::Fp8Tensor_256_256_64)
//  X(CriGroupedGemm_E4M3E4M3BF16_RRR_TileShape_256_256_32, cutlass::moe::Fp8Tensor_256_256_64)
//  X(CriGroupedGemm_E4M3E4M3BF16_RRR_TileShape_352_256_64, cutlass::moe::Fp8Tensor_352_256_64)
//  X(CriGroupedGemm_E4M3E4M3BF16_RRR_TileShape_320_512_64, cutlass::moe::Fp8Tensor_320_512_64)
//  X(CriGroupedGemm_E4M3E4M3BF16_RRR_TileShape_64_1792_64, cutlass::moe::Fp8Tensor_64_1792_64)
//  X(CriGroupedGemm_E4M3E4M3BF16_RRR_TileShape_448_256_64, cutlass::moe::Fp8Tensor_448_256_64)
//  X(CriGroupedGemm_E4M3E4M3BF16_RRR_TileShape_608_128_64, cutlass::moe::Fp8Tensor_608_128_64)
//  X(CriGroupedGemm_E4M3E4M3BF16_RRR_TileShape_64_1280_64, cutlass::moe::Fp8Tensor_64_1280_64)
//  X(CriGroupedGemm_E4M3E4M3BF16_RRR_TileShape_128_896_64, cutlass::moe::Fp8Tensor_128_896_64)
//  X(CriGroupedGemm_E4M3E4M3BF16_RRR_TileShape_96_896_64, cutlass::moe::Fp8Tensor_96_896_64)
//  X(CriGroupedGemm_E4M3E4M3BF16_RRR_TileShape_64_896_64, cutlass::moe::Fp8Tensor_64_896_64)
//  X(CriGroupedGemm_E4M3E4M3BF16_RRR_TileShape_192_640_64, cutlass::moe::Fp8Tensor_192_640_64)
//  X(CriGroupedGemm_E4M3E4M3BF16_RRR_TileShape_448_320_64, cutlass::moe::Fp8Tensor_448_320_64)
//  X(CriGroupedGemm_E4M3E4M3BF16_RRR_TileShape_256_448_64, cutlass::moe::Fp8Tensor_256_448_64)
//  X(CriGroupedGemm_E4M3E4M3BF16_RRR_TileShape_256_512_64, cutlass::moe::Fp8Tensor_256_512_64)
//  X(CriGroupedGemm_E4M3E4M3BF16_RRR_TileShape_320_320_64, cutlass::moe::Fp8Tensor_320_320_64)
//  X(CriGroupedGemm_E4M3E4M3BF16_RRR_TileShape_192_512_64, cutlass::moe::Fp8Tensor_192_512_64)
#else
#define MOE_TILE_LIST_FP8_TENSOR_E4M3
#endif

#ifdef MOE_DTYPE_MXFP8_E4M3
// PO_sanity CI trim: only the tile referenced by
// config_files/01_grouped_gemm/cri/PO_sanity/mxfp8.in (TileShape_352_256_64) is
// built. All other MXFP8 E4M3 tiles are commented out to cut CI build time;
// restore the lines below to re-enable the full sweep.
#define MOE_TILE_LIST_MXFP8_E4M3 \
  X(CriBLockScalingGroupedGemm_E4M3E4M3BF16_RRR_TileShape_352_256_64, cutlass::moe::MxFp8_352_256_64)
//  X(CriBLockScalingGroupedGemm_E4M3E4M3BF16_RRR_TileShape_256_256_32, cutlass::moe::MxFp8_256_256_64)
//  X(CriBLockScalingGroupedGemm_E4M3E4M3BF16_RRR_TileShape_256_256_64, cutlass::moe::MxFp8_256_256_64)
//  X(CriBLockScalingGroupedGemm_E4M3E4M3BF16_RRR_TileShape_320_512_64, cutlass::moe::MxFp8_320_512_64)
//  X(CriBLockScalingGroupedGemm_E4M3E4M3BF16_RRR_TileShape_64_1792_64, cutlass::moe::MxFp8_64_1792_64)
//  X(CriBLockScalingGroupedGemm_E4M3E4M3BF16_RRR_TileShape_448_256_64, cutlass::moe::MxFp8_448_256_64)
//  X(CriBLockScalingGroupedGemm_E4M3E4M3BF16_RRR_TileShape_608_128_64, cutlass::moe::MxFp8_608_128_64)
//  X(CriBLockScalingGroupedGemm_E4M3E4M3BF16_RRR_TileShape_64_1280_64, cutlass::moe::MxFp8_64_1280_64)
//  X(CriBLockScalingGroupedGemm_E4M3E4M3BF16_RRR_TileShape_128_896_64, cutlass::moe::MxFp8_128_896_64)
//  X(CriBLockScalingGroupedGemm_E4M3E4M3BF16_RRR_TileShape_96_896_64, cutlass::moe::MxFp8_96_896_64)
//  X(CriBLockScalingGroupedGemm_E4M3E4M3BF16_RRR_TileShape_192_640_64, cutlass::moe::MxFp8_192_640_64)
//  X(CriBLockScalingGroupedGemm_E4M3E4M3BF16_RRR_TileShape_448_320_64, cutlass::moe::MxFp8_448_320_64)
//  X(CriBLockScalingGroupedGemm_E4M3E4M3BF16_RRR_TileShape_256_448_64, cutlass::moe::MxFp8_256_448_64)
//  X(CriBLockScalingGroupedGemm_E4M3E4M3BF16_RRR_TileShape_256_512_64, cutlass::moe::MxFp8_256_512_64)
//  X(CriBLockScalingGroupedGemm_E4M3E4M3BF16_RRR_TileShape_320_320_64, cutlass::moe::MxFp8_320_320_64)
//  X(CriBLockScalingGroupedGemm_E4M3E4M3BF16_RRR_TileShape_192_512_64, cutlass::moe::MxFp8_192_512_64)
#else
#define MOE_TILE_LIST_MXFP8_E4M3
#endif

#ifdef MOE_DTYPE_MXFP4_E2M1
// PO_sanity CI trim: only the tile referenced by
// config_files/01_grouped_gemm/cri/PO_sanity/mxfp4.in (TileShape_256_512_128)
// is built. All other MXFP4 tiles are commented out to cut CI build time;
// restore the lines below to re-enable the full sweep.
#define MOE_TILE_LIST_MXFP4_E2M1 \
  X(CriBLockScalingGroupedGemm_E2M1E2M1BF16_RCR_TileShape_256_512_128, cutlass::moe::MxFp4_256_512_128)
//  X(CriGroupedGemm_E2M1E2M1BF16_RCR_TileShape_256_256_64, cutlass::moe::MxFp4_256_256_128)
//  X(CriBLockScalingGroupedGemm_E2M1E2M1BF16_RCR_TileShape_256_256_64, cutlass::moe::MxFp4_256_256_128)
//  X(CriBLockScalingGroupedGemm_E2M1E2M1BF16_RCR_TileShape_256_256_128, cutlass::moe::MxFp4_256_256_128)
//  X(CriBLockScalingGroupedGemm_E2M1E2M1BF16_RCR_TileShape_352_256_128, cutlass::moe::MxFp4_352_256_128)
//  X(CriBLockScalingGroupedGemm_E2M1E2M1BF16_RCR_TileShape_320_512_128, cutlass::moe::MxFp4_320_512_128)
//  X(CriBLockScalingGroupedGemm_E2M1E2M1BF16_RCR_TileShape_192_512_128, cutlass::moe::MxFp4_192_512_128)
//  X(CriBLockScalingGroupedGemm_E2M1E2M1BF16_RCR_TileShape_448_256_128, cutlass::moe::MxFp4_448_256_128)
//  X(CriBLockScalingGroupedGemm_E2M1E2M1BF16_RCR_TileShape_608_128_128, cutlass::moe::MxFp4_608_128_128)
//  X(CriBLockScalingGroupedGemm_E2M1E2M1BF16_RCR_TileShape_96_1280_128, cutlass::moe::MxFp4_96_1280_128)
//  X(CriBLockScalingGroupedGemm_E2M1E2M1BF16_RCR_TileShape_128_896_128, cutlass::moe::MxFp4_128_896_128)
//  X(CriBLockScalingGroupedGemm_E2M1E2M1BF16_RCR_TileShape_96_896_128, cutlass::moe::MxFp4_96_896_128)
//  X(CriBLockScalingGroupedGemm_E2M1E2M1BF16_RCR_TileShape_96_640_128, cutlass::moe::MxFp4_96_640_128)
//  X(CriBLockScalingGroupedGemm_E2M1E2M1BF16_RCR_TileShape_192_640_128, cutlass::moe::MxFp4_192_640_128)
//  X(CriBLockScalingGroupedGemm_E2M1E2M1BF16_RCR_TileShape_448_320_128, cutlass::moe::MxFp4_448_320_128)
//  X(CriBLockScalingGroupedGemm_E2M1E2M1BF16_RCR_TileShape_256_448_128, cutlass::moe::MxFp4_256_448_128)
//  X(CriBLockScalingGroupedGemm_E2M1E2M1BF16_RCR_TileShape_320_320_128, cutlass::moe::MxFp4_320_320_128)
#else
#define MOE_TILE_LIST_MXFP4_E2M1
#endif

#ifdef MOE_DTYPE_BF16
// PO_sanity CI trim: only the tile referenced by
// config_files/01_grouped_gemm/cri/PO_sanity/bf16.in (TileShape_64_1536_32) is
// built. All other BF16 tiles are commented out to cut CI build time; restore
// the lines below to re-enable the full sweep.
#define MOE_TILE_LIST_BF16 \
  X(CriGroupedGemmBF16BF16BF16_RRR_TileShape_64_1536_32, cutlass::moe::Bf16_64_1536_32)
//  X(CriMoEGemmBF16BF16BF16_RRR_TileShape_256_128_32, cutlass::moe::Bf16_256_128_32)
//  X(CriGroupedGemmBF16BF16BF16_RRR_TileShape_256_256_32, cutlass::moe::Bf16_256_256_32)
//  X(CriGroupedGemmBF16BF16BF16_RRR_TileShape_512_256_32, cutlass::moe::Bf16_512_256_32)
#else
#define MOE_TILE_LIST_BF16
#endif

#ifdef MOE_DTYPE_MXFP8_E5M2
// PO_sanity CI trim: no MXFP8 E5M2 tile is referenced by any
// config_files/01_grouped_gemm/cri/PO_sanity config, so the entire E5M2 tile
// list is disabled to cut CI build time; restore the lines below to re-enable.
#define MOE_TILE_LIST_MXFP8_E5M2
//  X(CriGroupedGemm_E5M2E5M2BF16_RRR_TileShape_256_256_32, cutlass::moe::MxFp8E5m2_256_256_64)
//  X(CriBLockScalingGroupedGemm_E5M2E5M2BF16_RRR_TileShape_256_256_32, cutlass::moe::MxFp8E5m2_256_256_64)
//  X(CriBLockScalingGroupedGemm_E5M2E5M2BF16_RRR_TileShape_256_256_64, cutlass::moe::MxFp8E5m2_256_256_64)
//  X(CriBLockScalingGroupedGemm_E5M2E5M2BF16_RRR_TileShape_352_256_64, cutlass::moe::MxFp8E5m2_352_256_64)
//  X(CriBLockScalingGroupedGemm_E5M2E5M2BF16_RRR_TileShape_320_512_64, cutlass::moe::MxFp8E5m2_320_512_64)
//  X(CriBLockScalingGroupedGemm_E5M2E5M2BF16_RRR_TileShape_64_1792_64, cutlass::moe::MxFp8E5m2_64_1792_64)
//  X(CriBLockScalingGroupedGemm_E5M2E5M2BF16_RRR_TileShape_448_256_64, cutlass::moe::MxFp8E5m2_448_256_64)
//  X(CriBLockScalingGroupedGemm_E5M2E5M2BF16_RRR_TileShape_608_128_64, cutlass::moe::MxFp8E5m2_608_128_64)
//  X(CriBLockScalingGroupedGemm_E5M2E5M2BF16_RRR_TileShape_64_1280_64, cutlass::moe::MxFp8E5m2_64_1280_64)
//  X(CriBLockScalingGroupedGemm_E5M2E5M2BF16_RRR_TileShape_128_896_64, cutlass::moe::MxFp8E5m2_128_896_64)
//  X(CriBLockScalingGroupedGemm_E5M2E5M2BF16_RRR_TileShape_96_896_64, cutlass::moe::MxFp8E5m2_96_896_64)
//  X(CriBLockScalingGroupedGemm_E5M2E5M2BF16_RRR_TileShape_192_640_64, cutlass::moe::MxFp8E5m2_192_640_64)
//  X(CriBLockScalingGroupedGemm_E5M2E5M2BF16_RRR_TileShape_448_320_64, cutlass::moe::MxFp8E5m2_448_320_64)
//  X(CriBLockScalingGroupedGemm_E5M2E5M2BF16_RRR_TileShape_256_448_64, cutlass::moe::MxFp8E5m2_256_448_64)
//  X(CriBLockScalingGroupedGemm_E5M2E5M2BF16_RRR_TileShape_256_512_64, cutlass::moe::MxFp8E5m2_256_512_64)
//  X(CriBLockScalingGroupedGemm_E5M2E5M2BF16_RRR_TileShape_320_320_64, cutlass::moe::MxFp8E5m2_320_320_64)
//  X(CriBLockScalingGroupedGemm_E5M2E5M2BF16_RRR_TileShape_192_512_64, cutlass::moe::MxFp8E5m2_192_512_64)
#else
#define MOE_TILE_LIST_MXFP8_E5M2
#endif

// Combined X-list = concatenation of the enabled dtype lists. For a single-dtype
// binary exactly one is non-empty (identical to the prior behavior); for the
// merged (MOE_BENCH_ALL) binary all five are concatenated.
#if defined(MOE_DTYPE_FP8_TENSOR_E4M3) || defined(MOE_DTYPE_MXFP8_E4M3) || \
    defined(MOE_DTYPE_MXFP8_E5M2) || defined(MOE_DTYPE_MXFP4_E2M1) || \
    defined(MOE_DTYPE_BF16)
#define MOE_TILE_X_LIST \
  MOE_TILE_LIST_FP8_TENSOR_E4M3 \
  MOE_TILE_LIST_MXFP8_E4M3 \
  MOE_TILE_LIST_MXFP8_E5M2 \
  MOE_TILE_LIST_MXFP4_E2M1 \
  MOE_TILE_LIST_BF16
#endif
