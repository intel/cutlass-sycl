/* Copyright (C) 2026 Intel Corporation, All rights reserved.
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
/*! \file
    \brief Unit tests for block-wise quantization via cute::quantize API.
*/

#include "cutlass/detail/layout.hpp"

#include <cmath>
#include <vector>

#include <cute/tensor.hpp>
#include <cute/tensor_sg.hpp>
#include <cute/quantization/quantize.hpp>
#include <cute/algorithm/reorder.hpp>
#include <sycl/sycl.hpp>
#include <cute/util/compat.hpp>

#include "cutlass_unit_test.h"

#include <cutlass/bfloat16.h>
#include <cutlass/float8.h>
#include <cutlass/numeric_types.h>

using namespace cute;
using namespace cutlass;
using namespace compat::experimental;

// ============================================================================
// Test Kernels
// ============================================================================

template<class...> class BlockWiseQuantizeKernelName;

// Kernel: loads src from global (round-robin), quantizes, stores dst & scale.
//
// Global memory layout (round-robin):
//   global[tid + v * sg_size]  ↔  register slot v of thread tid
//
// The TV layouts map (thread, value) → logical (M, N) coordinates.
template <int BlockSize,
          class SrcType, class DstType, class ScaleType,
          class SrcTVLayout, class DstTVLayout, class ScaleTVLayout>
void block_quantize_kernel(SrcType*  src_global,
                           DstType*  dst_global,
                           ScaleType* scale_global)
{
  const int tid = ThreadIdxX();

  // --- Source: global → register (round-robin) ---
  constexpr int src_total   = size(SrcTVLayout{});
  constexpr int src_per_thr = src_total / intel::sg_size;
  static_assert(src_total % intel::sg_size == 0);

  SrcType src_local[src_per_thr];
  for (int i = 0; i < src_per_thr; ++i)
    src_local[i] = src_global[tid + i * intel::sg_size];

  auto src_frag = make_tensor(make_rmem_ptr(src_local),
                               make_layout(Shape<Int<src_per_thr>>{}));
  auto src_sg   = make_subgroup_tensor(src_frag, SrcTVLayout{});

  // --- Destination: owning register fragment ---
  // Uses make_tensor<DstType> (owning) so that subbyte types (e.g., float_e2m1_t)
  // are backed by array_subbyte with correct packed storage, matching the
  // hardware byte-granularity packing assumed by subbyte_sg_tv_swizzle in reorder.
  constexpr int dst_total   = size(DstTVLayout{});
  constexpr int dst_per_thr = dst_total / intel::sg_size;
  static_assert(dst_total % intel::sg_size == 0);

  auto dst_frag = make_tensor<DstType>(make_layout(Shape<Int<dst_per_thr>>{}));
  auto dst_sg   = make_subgroup_tensor(dst_frag, DstTVLayout{});

  // --- Scale: zero-initialized register fragment ---
  constexpr int scale_total   = size(ScaleTVLayout{});
  constexpr int scale_per_thr = scale_total / intel::sg_size;
  static_assert(scale_total % intel::sg_size == 0);

  ScaleType scale_local[scale_per_thr]{};
  auto scale_frag = make_tensor(make_rmem_ptr(scale_local),
                                 make_layout(Shape<Int<scale_per_thr>>{}));
  auto scale_sg   = make_subgroup_tensor(scale_frag, ScaleTVLayout{});

  // --- Quantize ---
  quantize<BlockSize>(src_sg, dst_sg, scale_sg);

  // --- Store dst back to global (round-robin) ---
  // Read through tensor accessor to handle subbyte packed storage correctly.
  for (int i = 0; i < dst_per_thr; ++i)
    dst_global[tid + i * intel::sg_size] = static_cast<DstType>(dst_frag(i));

  // --- Store scale back to global (round-robin) ---
  for (int i = 0; i < scale_per_thr; ++i)
    scale_global[tid + i * intel::sg_size] = scale_local[i];
}

// Launch helper: allocates device memory, launches kernel, copies results back.
template <int BlockSize,
          class SrcType, class DstType, class ScaleType,
          class SrcTVLayout, class DstTVLayout, class ScaleTVLayout, int TestID>
void run_block_quantize_test(
    cutlass::host_vector<SrcType>&   host_src,
    cutlass::host_vector<DstType>&   host_dst,
    cutlass::host_vector<ScaleType>& host_scale)
{
  cutlass::device_vector<SrcType>   device_src   = host_src;
  cutlass::device_vector<DstType>   device_dst(size(DstTVLayout{}));
  cutlass::device_vector<ScaleType> device_scale(size(ScaleTVLayout{}));

  launch<block_quantize_kernel<BlockSize, SrcType, DstType, ScaleType,
                               SrcTVLayout, DstTVLayout, ScaleTVLayout>,
         BlockWiseQuantizeKernelName<SrcType, DstType, ScaleType, Int<TestID>>>(
      launch_policy{compat::dim3(1), compat::dim3(intel::sg_size),
                    kernel_properties{sycl_exp::sub_group_size<intel::sg_size>}},
      device_src.data(), device_dst.data(), device_scale.data());

  compat::wait_and_throw();
  host_dst   = device_dst;
  host_scale = device_scale;
}

// ============================================================================
// CPU Reference Implementation for Block-Wise Quantization
// ============================================================================
//
// Operates on row-major 2D logical data: element (m, n) at index m*N + n.
//
//   Blocks along N, scale shape (M, N/BlockSize), stored row-major
//
// Algorithm (matches the GPU implementation):
//   Phase 1: Compute per-block absolute max
//   Phase 2: scale = round_toward_zero(target_max / block_amax)
//   Phase 3: dst[i] = round_to_nearest(src[i] * scale)
//
template <int BlockSize, class SrcType, class DstType, class ScaleType>
void reference_block_wise_quantize(
    const std::vector<SrcType>& src,     // M*N elements, row-major
    std::vector<DstType>&       dst,     // M*N elements, row-major (output)
    std::vector<ScaleType>&     scales,  // scale factors (output)
    int M, int N)
{
  using FloatToScale  = cutlass::NumericConverter<ScaleType, float,
                         cutlass::FloatRoundStyle::round_toward_zero>;
  using FloatToDst   = cutlass::NumericConverter<DstType, float,
                         cutlass::FloatRoundStyle::round_to_nearest>;

  const float target_max = static_cast<float>(
      cutlass::platform::numeric_limits<DstType>::max());

  dst.resize(static_cast<size_t>(M * N));

  const int NumBlocks = N / BlockSize;
  scales.resize(static_cast<size_t>(M * NumBlocks));

  for (int m = 0; m < M; ++m) {
    for (int b = 0; b < NumBlocks; ++b) {
      // Phase 1: block_amax for row m, block b
      float block_amax = 0.0f;
      for (int k = 0; k < BlockSize; ++k) {
        int n = b * BlockSize + k;
        float val = static_cast<float>(src[static_cast<size_t>(m * N + n)]);
        block_amax = std::max(block_amax, std::abs(val));
      }

      // Phase 2: compute & store scale
      float s = (block_amax > 0.0f) ? (target_max / block_amax) : 0.0f;
      scales[static_cast<size_t>(m * NumBlocks + b)] = FloatToScale{}(s);

      // Phase 3: quantize each element in the block
      for (int k = 0; k < BlockSize; ++k) {
        int n = b * BlockSize + k;
        float val = static_cast<float>(src[static_cast<size_t>(m * N + n)]);
        dst[static_cast<size_t>(m * N + n)] = FloatToDst{}(val * s);
      }
    }
  }
}

// ============================================================================
// Conversion between row-major logical data and round-robin global memory
// ============================================================================
//
// GPU global memory uses round-robin layout:
//   global[tid + v * sg_size]  ↔  register slot v of thread tid
//
// A TV layout maps (tid, v) → (m, n).
// Logical row-major data indexes as: logical[m * N + n].
//
// CONSTRAINT: TV layouts must be rank 2 for tv(tid, v) to work.
//
// logical_to_roundrobin: fills a pre-allocated host_vector from row-major src
// roundrobin_to_logical: fills a row-major std::vector from round-robin host_vector
//
template <class TVLayout, class T>
void logical_to_roundrobin(const std::vector<T>& logical,  // M*N row-major
                           cutlass::host_vector<T>& global, // pre-allocated, size(TVLayout)
                           int N)
{
  constexpr auto tv = TVLayout{};
  constexpr int total = size(tv);
  constexpr int vals = total / intel::sg_size;

  for (int tid = 0; tid < intel::sg_size; ++tid) {
    for (int v = 0; v < vals; ++v) {
      auto coord = tv(tid, v);
      int m = get<0>(coord);
      int n = get<1>(coord);
      global[tid + v * intel::sg_size] = logical[static_cast<size_t>(m * N + n)];
    }
  }
}

template <class TVLayout, class T>
void roundrobin_to_logical(const cutlass::host_vector<T>& global, // size(TVLayout) round-robin
                           std::vector<T>& logical,               // M*N row-major (output)
                           int M, int N)
{
  constexpr auto tv = TVLayout{};
  constexpr int total = size(tv);
  constexpr int vals = total / intel::sg_size;
  logical.assign(static_cast<size_t>(M * N), T{});

  for (int tid = 0; tid < intel::sg_size; ++tid) {
    for (int v = 0; v < vals; ++v) {
      auto coord = tv(tid, v);
      int m = get<0>(coord);
      int n = get<1>(coord);
      logical[static_cast<size_t>(m * N + n)] = global[tid + v * intel::sg_size];
    }
  }
}

// ============================================================================
// Initialize source data (row-major logical)
// ============================================================================
template <int BlockSize, class SrcType>
void initialize_source(std::vector<SrcType>& src, int M, int N) {
  src.resize(static_cast<size_t>(M * N));
  for (size_t i = 0; i < src.size(); ++i) {
    int col = static_cast<int>(i) % N;
    int block_idx = col / BlockSize;
    // Alternating sign pattern to exercise negative values
    float sign = (i % 3 == 0) ? -1.0f : 1.0f;
    // Scale by (block_idx+1) so different blocks have distinct amax values
    float val  = sign * static_cast<float>(i % 17) * 0.2f + static_cast<float>(block_idx + 1);
    if constexpr (std::is_same_v<SrcType, cutlass::bfloat16_t>) {
      src[i] = cutlass::bfloat16_t(val);
    } else if constexpr (std::is_same_v<SrcType, cutlass::half_t>) {
      src[i] = cutlass::half_t(val);
    } else if constexpr (std::is_same_v<SrcType, float>) {
      src[i] = val;
    } else {
      CUTE_INVALID_CONTROL_PATH("Not Implemented");
    }
  }
}

// ============================================================================
// TV Layouts
// ============================================================================
//
// Design for src TV layouts:
//   - The row index m must be THREAD-INDEPENDENT (depends only on value v).
//   - Thread must map ONLY to dimension 1 (n).
//
// Design for scale TV layouts:
//   Scales are subgroup-uniform (block_amax computed via cross-lane reduce).
//   Thread stride is Int<0> (degenerate): all threads redundantly store
//   the full set of scale values.
//

// ---- Src/Dst TV Layout: M=16, N=32 ----
// Shape:  (Int<16>, (Int<16>, Int<2>))
// Stride: (ScaledBasis<Int<1>,1>, (ScaledBasis<Int<1>,0>, ScaledBasis<Int<16>,1>))
//
// Mapping: (t, (v0, v1)) → m = v0,  n = t + v1*16
//   - Thread t maps to n = t  (dim 1 only, thread-independent m) ✓
//   - v0 ∈ [0,16) → m,  v1 ∈ [0,2) → n offset {0, 16}
//   - coshape: (16, 32),  total: 512,  per_thr: 32
//
using DataTVLayout_16x32 = decltype(make_layout(
    make_shape(Int<16>{}, make_shape(Int<16>{}, Int<2>{})),
    make_stride(ScaledBasis<Int<1>,1>{},
                make_stride(ScaledBasis<Int<1>,0>{}, ScaledBasis<Int<16>,1>{}))
));

// ---- Src/Dst TV Layout: M=16, N=64 ----
// Shape:  (Int<16>, (Int<16>, Int<4>))
// Stride: (ScaledBasis<Int<1>,1>, (ScaledBasis<Int<1>,0>, ScaledBasis<Int<16>,1>))
//
// Mapping: (t, (v0, v1)) → m = v0,  n = t + v1*16
//   - coshape: (16, 64),  total: 1024,  per_thr: 64
//
using DataTVLayout_16x64 = decltype(make_layout(
    make_shape(Int<16>{}, make_shape(Int<16>{}, Int<4>{})),
    make_stride(ScaledBasis<Int<1>,1>{},
                make_stride(ScaledBasis<Int<1>,0>{}, ScaledBasis<Int<16>,1>{}))
));

// ---- MMA-like Src/Dst TV Layout: M=32, N=64 (atom=8, 4 M-iters, 4 N-iters) ----
// Shape:  (Int<16>, (Int<8>, (Int<4>, Int<4>)))
// Stride: (ScaledBasis<Int<1>,1>, (ScaledBasis<Int<1>,0>, (ScaledBasis<Int<8>,0>, ScaledBasis<Int<16>,1>)))
// These match the real MMA C fragment layout from partition_sg_fragment_C:
//   Value mode = (atom_M, (M_iters, N_iters))
//   - atom_M=8: rows within one DPAS atom (stride 1 in M)
//   - M_iters:  number of atom repetitions in M (stride 8 in M)
//   - N_iters:  number of atom repetitions in N (stride 16 in N)
//
// Mapping: (t, (v0, (v1, v2))) → m = v0 + v1*8,  n = t + v2*16
//   - coshape: (32, 64),  total: 2048,  per_thr: 128
//
using DataTVLayout_32x64 = decltype(make_layout(
    make_shape(Int<16>{}, make_shape(Int<8>{}, make_shape(Int<4>{}, Int<4>{}))),
    make_stride(ScaledBasis<Int<1>,1>{},
                make_stride(ScaledBasis<Int<1>,0>{},
                            make_stride(ScaledBasis<Int<8>,0>{}, ScaledBasis<Int<16>,1>{})))
));

// ---- Scale TV Layouts ----
//
// Scale shape: (M, NumBlocks) — one scale per (row, block) pair.
// Requires M % 16 = 0 (distributed): Each thread owns its assigned rows' scales.
//   Thread t → row(s) via ScaledBasis<1,0>; value mode holds NumBlocks.
//   per_thr = (M/16) * NumBlocks,  total = M * NumBlocks
//

// M=16, NB=1: thread t → (m=t, nb=0), per_thr=1, total=16
using ScaleTVLayout_16x1 = decltype(make_layout(
    make_shape(Int<16>{}, make_shape(Int<1>{}, Int<1>{})),
    make_stride(ScaledBasis<Int<1>,0>{},
                make_stride(Int<0>{}, ScaledBasis<Int<1>,1>{}))
));

// M=16, NB=2: thread t → (m=t, nb=v), per_thr=2, total=32
using ScaleTVLayout_16x2 = decltype(make_layout(
    make_shape(Int<16>{}, make_shape(Int<1>{}, Int<2>{})),
    make_stride(ScaledBasis<Int<1>,0>{},
                make_stride(Int<0>{}, ScaledBasis<Int<1>,1>{}))
));

// M=32, NB=2: thread t → rows {t, t+16}, per_thr=4, total=64
using ScaleTVLayout_32x2 = decltype(make_layout(
    make_shape(Int<16>{}, make_shape(Int<2>{}, Int<2>{})),
    make_stride(ScaledBasis<Int<1>,0>{},
                make_stride(ScaledBasis<Int<16>,0>{}, ScaledBasis<Int<1>,1>{}))
));

// M=16, NB=4: thread t → (m=t, nb=v), per_thr=4, total=64
using ScaleTVLayout_16x4 = decltype(make_layout(
    make_shape(Int<16>{}, make_shape(Int<1>{}, Int<4>{})),
    make_stride(ScaledBasis<Int<1>,0>{},
                make_stride(Int<0>{}, ScaledBasis<Int<1>,1>{}))
));


// ============================================================================
// Test struct: XeBlockQuantizeTest
//
// Template parameters:
//   BlockSize  — quantization block size
//   SrcType, DstType, ScaleType — element types
//   SrcTVLayout, DstTVLayout, ScaleTVLayout — TV layouts for the 3 tensors
//   TestID — unique ID for kernel name dedup
// ============================================================================
template <int BlockSize,
          class SrcType, class DstType, class ScaleType,
          class SrcTVLayout, class DstTVLayout, class ScaleTVLayout,
          int TestID>
struct XeBlockQuantizeTest {
  static void run() {
    // ---- Logical dimensions from src TV layout ----
    constexpr auto logical_shape = atuple_coshape(SrcTVLayout{});
    constexpr int M = get<0>(logical_shape);
    constexpr int N = get<1>(logical_shape);
    constexpr int NumBlocks = N / BlockSize;

    // ---- 1. Initialize row-major logical source data ----
    std::vector<SrcType> src_logical;
    initialize_source<BlockSize>(src_logical, M, N);

    // ---- 2. CPU reference (per-row, per-block quantization) ----
    std::vector<DstType>   ref_dst;
    std::vector<ScaleType> ref_scales;
    reference_block_wise_quantize<BlockSize>(src_logical, ref_dst, ref_scales, M, N);

    // ---- 3. Convert logical src → round-robin for GPU ----
    constexpr int src_total = size(SrcTVLayout{});
    cutlass::host_vector<SrcType> rr_src(src_total);
    logical_to_roundrobin<SrcTVLayout>(src_logical, rr_src, N);

    // ---- 4. Run GPU kernel ----
    constexpr int dst_total   = size(DstTVLayout{});
    constexpr int scale_total = size(ScaleTVLayout{});
    cutlass::host_vector<DstType>   rr_dst(dst_total);
    cutlass::host_vector<ScaleType> rr_scale(scale_total);
    run_block_quantize_test<BlockSize, SrcType, DstType, ScaleType,
                            SrcTVLayout, DstTVLayout, ScaleTVLayout, TestID>(
        rr_src, rr_dst, rr_scale);

    // ---- 5. Convert round-robin GPU results → logical ----
    std::vector<DstType> gpu_dst_logical;
    roundrobin_to_logical<DstTVLayout>(rr_dst, gpu_dst_logical, M, N);

    // Scale: reconstruct logical (M, NumBlocks) from round-robin global memory.
    // Works for distributed layouts (each thread stores its own subset).
    constexpr int scale_per_thr = scale_total / intel::sg_size;
    constexpr auto scale_tv = ScaleTVLayout{};
    std::vector<ScaleType> gpu_scales(static_cast<size_t>(M * NumBlocks), ScaleType{});
    for (int tid = 0; tid < intel::sg_size; ++tid) {
      for (int v = 0; v < scale_per_thr; ++v) {
        auto coord = scale_tv(tid, v);
        int sm  = int(get<0>(coord));
        int snb = int(get<1>(coord));
        gpu_scales[static_cast<size_t>(sm * NumBlocks + snb)] =
            rr_scale[static_cast<size_t>(tid + v * intel::sg_size)];
      }
    }

    // ---- 6. Compare dst element-by-element ----
    const float dst_max_f = static_cast<float>(
        cutlass::platform::numeric_limits<DstType>::max());
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        size_t idx = static_cast<size_t>(m * N + n);
        EXPECT_NEAR(static_cast<float>(gpu_dst_logical[idx]),
                    static_cast<float>(ref_dst[idx]), 1e-4f)
            << "dst mismatch at (" << m << ", " << n << ")";
      }
    }

    // ---- 7. Compare scales element-by-element ----
    // Both ref_scales and gpu_scales are in row-major (M, NumBlocks) order:
    //   index = m * NumBlocks + b
    for (int m = 0; m < M; ++m) {
      for (int b = 0; b < NumBlocks; ++b) {
        int idx = m * NumBlocks + b;
        float gpu_val = static_cast<float>(gpu_scales[static_cast<size_t>(idx)]);
        float ref_val = static_cast<float>(ref_scales[static_cast<size_t>(idx)]);
        float tol = 1e-6f * std::max(std::abs(gpu_val), std::abs(ref_val));
        EXPECT_NEAR(gpu_val, ref_val, std::max(tol, 1e-6f))
            << "scale mismatch at (" << m << ", block " << b << ")";
      }
    }
  }
};

// ============================================================================
// Test Cases — M=16, N=32, BlockSize=32 (single block per row, NumBlocks=1)
// ============================================================================
// Src/Dst: DataTVLayout_16x32 — 32 values per thread, uses optimized path.

TEST(CuTe_Xe_BlockQuantize, bf16_to_e4m3_16x32_bs32) {
  XeBlockQuantizeTest<32,
      cutlass::bfloat16_t, cutlass::float_e4m3_t, float,
      DataTVLayout_16x32, DataTVLayout_16x32, ScaleTVLayout_16x1, 0>::run();
}

TEST(CuTe_Xe_BlockQuantize, bf16_to_e5m2_16x32_bs32) {
  XeBlockQuantizeTest<32,
      cutlass::bfloat16_t, cutlass::float_e5m2_t, float,
      DataTVLayout_16x32, DataTVLayout_16x32, ScaleTVLayout_16x1, 1>::run();
}

TEST(CuTe_Xe_BlockQuantize, half_to_e4m3_16x32_bs32) {
  XeBlockQuantizeTest<32,
      cutlass::half_t, cutlass::float_e4m3_t, float,
      DataTVLayout_16x32, DataTVLayout_16x32, ScaleTVLayout_16x1, 2>::run();
}

TEST(CuTe_Xe_BlockQuantize, half_to_e5m2_16x32_bs32) {
  XeBlockQuantizeTest<32,
      cutlass::half_t, cutlass::float_e5m2_t, float,
      DataTVLayout_16x32, DataTVLayout_16x32, ScaleTVLayout_16x1, 3>::run();
}

// --- M=16, N=32, BlockSize=16 (two blocks per row, NumBlocks=2) ---
// Scale shape: (16, 2)

TEST(CuTe_Xe_BlockQuantize, bf16_to_e4m3_16x32_bs16) {
  XeBlockQuantizeTest<16,
      cutlass::bfloat16_t, cutlass::float_e4m3_t, float,
      DataTVLayout_16x32, DataTVLayout_16x32, ScaleTVLayout_16x2, 4>::run();
}

TEST(CuTe_Xe_BlockQuantize, bf16_to_e5m2_16x32_bs16) {
  XeBlockQuantizeTest<16,
      cutlass::bfloat16_t, cutlass::float_e5m2_t, float,
      DataTVLayout_16x32, DataTVLayout_16x32, ScaleTVLayout_16x2, 5>::run();
}

TEST(CuTe_Xe_BlockQuantize, half_to_e4m3_16x32_bs16) {
  XeBlockQuantizeTest<16,
      cutlass::half_t, cutlass::float_e4m3_t, float,
      DataTVLayout_16x32, DataTVLayout_16x32, ScaleTVLayout_16x2, 6>::run();
}

TEST(CuTe_Xe_BlockQuantize, half_to_e5m2_16x32_bs16) {
  XeBlockQuantizeTest<16,
      cutlass::half_t, cutlass::float_e5m2_t, float_ue8m0_t,
      DataTVLayout_16x32, DataTVLayout_16x32, ScaleTVLayout_16x2, 7>::run();
}

// ============================================================================
// Test Cases — M=16, N=64, BlockSize=32 (two blocks per row, NumBlocks=2)
// ============================================================================
// Src/Dst: DataTVLayout_16x64 — 64 values per thread, uses optimized path.

TEST(CuTe_Xe_BlockQuantize, half_to_e4m3_16x64_bs32) {
  XeBlockQuantizeTest<32,
      cutlass::half_t, cutlass::float_e4m3_t, float,
      DataTVLayout_16x64, DataTVLayout_16x64, ScaleTVLayout_16x2, 11>::run();
}

TEST(CuTe_Xe_BlockQuantize, bf16_to_e5m2_16x64_bs32) {
  XeBlockQuantizeTest<32,
      cutlass::bfloat16_t, cutlass::float_e5m2_t, float,
      DataTVLayout_16x64, DataTVLayout_16x64, ScaleTVLayout_16x2, 12>::run();
}

TEST(CuTe_Xe_BlockQuantize, half_to_e5m2_16x64_bs32) {
  XeBlockQuantizeTest<32,
      cutlass::half_t, cutlass::float_e5m2_t, float,
      DataTVLayout_16x64, DataTVLayout_16x64, ScaleTVLayout_16x2, 13>::run();
}

TEST(CuTe_Xe_BlockQuantize, bf16_to_e4m3_16x64_bs32) {
  XeBlockQuantizeTest<32,
      cutlass::bfloat16_t, cutlass::float_e4m3_t, float,
      DataTVLayout_16x64, DataTVLayout_16x64, ScaleTVLayout_16x2, 14>::run();
}

// --- M=16, N=64, BlockSize=16 (four blocks per row, NumBlocks=4) ---
// Scale shape: (16, 4) via ScaleTVLayout_16x4

TEST(CuTe_Xe_BlockQuantize, bf16_to_e4m3_16x64_bs16) {
  XeBlockQuantizeTest<16,
      cutlass::bfloat16_t, cutlass::float_e4m3_t, float,
      DataTVLayout_16x64, DataTVLayout_16x64, ScaleTVLayout_16x4, 15>::run();
}

TEST(CuTe_Xe_BlockQuantize, half_to_e5m2_16x64_bs16) {
  XeBlockQuantizeTest<16,
      cutlass::half_t, cutlass::float_e5m2_t, float,
      DataTVLayout_16x64, DataTVLayout_16x64, ScaleTVLayout_16x4, 16>::run();
}

// ============================================================================
// Test Cases — M=32, N=64, BlockSize=32 (two blocks per row, NumBlocks=2)
// ============================================================================
// Src/Dst: DataTVLayout_32x64 — 128 values per thread, uses optimized path.

TEST(CuTe_Xe_BlockQuantize, half_to_e4m3_32x64_bs32) {
  XeBlockQuantizeTest<32,
      cutlass::half_t, cutlass::float_e4m3_t, float,
      DataTVLayout_32x64, DataTVLayout_32x64, ScaleTVLayout_32x2, 17>::run();
}

TEST(CuTe_Xe_BlockQuantize, bf16_to_e5m2_32x64_bs32) {
  XeBlockQuantizeTest<32,
      cutlass::bfloat16_t, cutlass::float_e5m2_t, float,
      DataTVLayout_32x64, DataTVLayout_32x64, ScaleTVLayout_32x2, 18>::run();
}

TEST(CuTe_Xe_BlockQuantize, half_to_e5m2_32x64_bs32) {
  XeBlockQuantizeTest<32,
      cutlass::half_t, cutlass::float_e5m2_t, float,
      DataTVLayout_32x64, DataTVLayout_32x64, ScaleTVLayout_32x2, 19>::run();
}

TEST(CuTe_Xe_BlockQuantize, bf16_to_e4m3_32x64_bs32) {
  XeBlockQuantizeTest<32,
      cutlass::bfloat16_t, cutlass::float_e4m3_t, float,
      DataTVLayout_32x64, DataTVLayout_32x64, ScaleTVLayout_32x2, 20>::run();
}

// ============================================================================
// Test Cases — Src dtype = float32
// ============================================================================

// --- M=16, N=32, BlockSize=32 (single block per row, NumBlocks=1) ---
TEST(CuTe_Xe_BlockQuantize, f32_to_e4m3_16x32_bs32) {
  XeBlockQuantizeTest<32,
      float, cutlass::float_e4m3_t, float,
      DataTVLayout_16x32, DataTVLayout_16x32, ScaleTVLayout_16x1, 21>::run();
}

TEST(CuTe_Xe_BlockQuantize, f32_to_e5m2_16x32_bs32) {
  XeBlockQuantizeTest<32,
      float, cutlass::float_e5m2_t, float,
      DataTVLayout_16x32, DataTVLayout_16x32, ScaleTVLayout_16x1, 22>::run();
}

// --- M=16, N=64, BlockSize=32 (two blocks per row, NumBlocks=2) ---
TEST(CuTe_Xe_BlockQuantize, f32_to_e4m3_16x64_bs32) {
  XeBlockQuantizeTest<32,
      float, cutlass::float_e4m3_t, float,
      DataTVLayout_16x64, DataTVLayout_16x64, ScaleTVLayout_16x2, 23>::run();
}

TEST(CuTe_Xe_BlockQuantize, f32_to_e5m2_16x64_bs32) {
  XeBlockQuantizeTest<32,
      float, cutlass::float_e5m2_t, float,
      DataTVLayout_16x64, DataTVLayout_16x64, ScaleTVLayout_16x2, 24>::run();
}

// --- M=32, N=64, BlockSize=32 (two blocks per row, NumBlocks=2) ---
TEST(CuTe_Xe_BlockQuantize, f32_to_e4m3_32x64_bs32) {
  XeBlockQuantizeTest<32,
      float, cutlass::float_e4m3_t, float,
      DataTVLayout_32x64, DataTVLayout_32x64, ScaleTVLayout_32x2, 25>::run();
}

TEST(CuTe_Xe_BlockQuantize, f32_to_e5m2_32x64_bs32) {
  XeBlockQuantizeTest<32,
      float, cutlass::float_e5m2_t, float,
      DataTVLayout_32x64, DataTVLayout_32x64, ScaleTVLayout_32x2, 26>::run();
}

// ============================================================================
// Test Cases — Dst dtype = float_e2m1_t
// ============================================================================
// No optimized ASM path for E2M1; all tests use the fallback C++ path.

// --- M=16, N=32, BlockSize=32 (single block per row, NumBlocks=1) ---
TEST(CuTe_Xe_BlockQuantize, bf16_to_e2m1_16x32_bs32) {
  XeBlockQuantizeTest<32,
      cutlass::bfloat16_t, cutlass::float_e2m1_t, float,
      DataTVLayout_16x32, DataTVLayout_16x32, ScaleTVLayout_16x1, 27>::run();
}

TEST(CuTe_Xe_BlockQuantize, half_to_e2m1_16x32_bs32) {
  XeBlockQuantizeTest<32,
      cutlass::half_t, cutlass::float_e2m1_t, float,
      DataTVLayout_16x32, DataTVLayout_16x32, ScaleTVLayout_16x1, 28>::run();
}

TEST(CuTe_Xe_BlockQuantize, f32_to_e2m1_16x32_bs32) {
  XeBlockQuantizeTest<32,
      float, cutlass::float_e2m1_t, float,
      DataTVLayout_16x32, DataTVLayout_16x32, ScaleTVLayout_16x1, 29>::run();
}

// --- M=16, N=64, BlockSize=32 (two blocks per row, NumBlocks=2) ---
TEST(CuTe_Xe_BlockQuantize, bf16_to_e2m1_16x64_bs32) {
  XeBlockQuantizeTest<32,
      cutlass::bfloat16_t, cutlass::float_e2m1_t, float,
      DataTVLayout_16x64, DataTVLayout_16x64, ScaleTVLayout_16x2, 30>::run();
}

TEST(CuTe_Xe_BlockQuantize, half_to_e2m1_16x64_bs32) {
  XeBlockQuantizeTest<32,
      cutlass::half_t, cutlass::float_e2m1_t, float,
      DataTVLayout_16x64, DataTVLayout_16x64, ScaleTVLayout_16x2, 31>::run();
}

// --- M=32, N=64, BlockSize=32 (two blocks per row, NumBlocks=2) ---
TEST(CuTe_Xe_BlockQuantize, bf16_to_e2m1_32x64_bs32) {
  XeBlockQuantizeTest<32,
      cutlass::bfloat16_t, cutlass::float_e2m1_t, float,
      DataTVLayout_32x64, DataTVLayout_32x64, ScaleTVLayout_32x2, 32>::run();
}

TEST(CuTe_Xe_BlockQuantize, f32_to_e2m1_32x64_bs32) {
  XeBlockQuantizeTest<32,
      float, cutlass::float_e2m1_t, float,
      DataTVLayout_32x64, DataTVLayout_32x64, ScaleTVLayout_32x2, 33>::run();
}


