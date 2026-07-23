/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
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
    \brief Block-wise quantization for SubgroupTensor on Intel Xe GPUs.

    Provides in-register quantization from high-precision types (half_t, BF16, FP32) to
    low-precision types (E4M3, E5M2, E2M1) with block-wise scale factors.

    The optimized path fuses abs-max reduction, scale computation, type conversion
    and reorder into a single inline vISA ASM block.

    This API is designed to enable fusion with GEMM prologue/epilogue operations,
    avoiding expensive register-to-memory conversions.
*/

#pragma once

#include <cute/config.hpp>
#include <cute/tensor.hpp>
#include <cute/tensor_sg.hpp>
#include <cute/algorithm/reorder.hpp>
#include <cute/atom/quantize_atom.hpp>

#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include <cutlass/float8.h>

#include <sycl/sycl.hpp>

namespace cute {

//////////////////////////////////////////////////////////////////////////////
/// compute_block_scales — Shared Steps 1-3 for both optimized and fallback paths
///
/// Step 1: Per-thread abs-max per (row, block) pair
/// Step 2: Cross-lane reduction + compute per-(row, block) scale in F32
/// Step 3: Write scales to output (converted to ScaleType)
///
/// Returns the per-(row, block) scale factors in block_scale_f32[M * NumBlocks],
/// indexed colexicographically as block_scale_f32[m + block_id * M].
//////////////////////////////////////////////////////////////////////////////
template <typename SrcType, typename DstType, int NumBlocks,
          int BlockSize, int NumValues, int ScaleNumValues,
          class SrcTVLayout, class ScaleTVLayout,
          class SrcEngine, class SrcLayoutWI,
          class ScaleEngine, class ScaleLayoutWI>
CUTE_HOST_DEVICE
void compute_block_scales(
  SubgroupTensor<SrcEngine, SrcLayoutWI, SrcTVLayout> const& src,
  SubgroupTensor<ScaleEngine, ScaleLayoutWI, ScaleTVLayout>& scale,
  float* block_scale_f32)
{
  using ScaleType = typename ScaleEngine::element_type;
  constexpr auto tv_layout = SrcTVLayout{};
  constexpr auto logical_shape = atuple_coshape(tv_layout);
  constexpr int M = get<0>(logical_shape);
  constexpr int NumScales = M * NumBlocks;

  // Step 1: Per-thread abs-max per (row, block) pair
  float local_max[NumScales];
  CUTE_UNROLL
  for (int i = 0; i < NumScales; i++) {
    local_max[i] = 0.0f;
  }
  CUTE_UNROLL
  for (int v = 0; v < NumValues; v++) {
    auto coord    = tv_layout(0, C<0>{} + v);
    int  m        = get<0>(coord);
    int  block_id = get<1>(coord) / BlockSize;
    int  idx      = m + block_id * M;
    float abs_val = sycl::fabs(static_cast<float>(src.tensor()(v)));
    local_max[idx] = sycl::fmax(local_max[idx], abs_val);
  }

  // Step 2: Cross-lane reduction + compute per-(row, block) scale in F32
  const float target_max = static_cast<float>(cutlass::platform::numeric_limits<DstType>::max());
  using FloatToScale = cutlass::NumericConverter<ScaleType, float,
                                                cutlass::FloatRoundStyle::round_toward_zero>;
  auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
  CUTE_UNROLL
  for (int i = 0; i < NumScales; i++) {
    local_max[i] = reduce_over_group(sg, local_max[i],
                                        sycl::maximum<float>{});
    float amax = local_max[i];
    block_scale_f32[i] = (amax > 0.0f) ? (target_max / amax) : 0.0f;
  }

  // Step 3: Write scales to output (converted to ScaleType)
  // Use lane ID so distributed scale layouts write the
  // correct per-thread subset.
  int lane = sg.get_local_id();
  CUTE_UNROLL
  for (int sv = 0; sv < ScaleNumValues; sv++) {
    auto sc = ScaleTVLayout{}(lane, C<0>{} + sv);
    int sm = int(get<0>(sc));
    int sb = int(get<1>(sc));
    scale.tensor()(sv) = FloatToScale{}(block_scale_f32[sm + sb * M]);
  }
}

//////////////////////////////////////////////////////////////////////////////
/// quantize_impl — Optimized path (C++ abs-max/reduce/scale + ASM atoms)
///
/// Steps 1-3 are handled by compute_block_scales.
/// Step 4 uses Xe_Quantize_Optimized ASM atoms for the multiply + type-convert
/// step (mul + mov F32→HF + fcvt HF→FP8).
///
/// Supported type combinations:
///   half_t     → float_e5m2_t, float_e4m3_t
///   bfloat16_t → float_e5m2_t, float_e4m3_t
///   float      → float_e5m2_t, float_e4m3_t
///
/// Requires: Xe target >= 35, sg_size == 16, NumValues divisible by 4.
/// For unsupported combinations, the dispatcher falls back to the pure C++ path.
//////////////////////////////////////////////////////////////////////////////
template <typename SrcType, typename DstType, int NumBlocks,
          int BlockSize, int NumValues, int ScaleNumValues,
          class SrcLayout, class SrcTVLayout,
          class SrcEngine, class SrcLayoutWI,
          class DstFrag,
          class ScaleEngine, class ScaleLayoutWI, class ScaleTVLayout>
CUTE_HOST_DEVICE
void quantize_impl(
  QuantizeDispatchOptimized const&,
  SubgroupTensor<SrcEngine, SrcLayoutWI, SrcTVLayout> const& src,
  DstFrag& tmp_dst_frag,
  SubgroupTensor<ScaleEngine, ScaleLayoutWI, ScaleTVLayout>& scale)
{
  constexpr auto tv_layout = SrcTVLayout{};
  constexpr auto logical_shape = atuple_coshape(tv_layout);
  constexpr int M = get<0>(logical_shape);
  constexpr int NumScales = M * NumBlocks;

  // Steps 1-3: Compute per-block scales and write to scale tensor
  float block_scale_f32[NumScales];
  compute_block_scales<SrcType, DstType, NumBlocks,
                       BlockSize, NumValues, ScaleNumValues,
                       SrcTVLayout, ScaleTVLayout>(
    src, scale, block_scale_f32);

  // Step 4: Quantize using ASM atoms (mul + fcvt, 4 values per call)
  using Impl = Xe_Quantize_Optimized<SrcType, DstType>;
  static_assert(NumValues % 4 == 0,
    "Quantize requires NumValues to be a multiple of 4");
  CUTE_UNROLL
  for (int v = 0; v < NumValues; v += 4) {
    typename Impl::ScaleRegister scales;
    CUTE_UNROLL
    for (int dv = 0; dv < 4; ++dv) {
      auto coord    = tv_layout(0, C<0>{} + v + dv);
      int  m        = int(get<0>(coord));
      int  block_id = int(get<1>(coord)) / BlockSize;
      scales[dv]    = block_scale_f32[m + block_id * M];
    }
    Impl::quantize(
      recast_ptr<typename Impl::SrcRegister>(&src.tensor()(v)),
      recast_ptr<typename Impl::DstRegister>(&tmp_dst_frag(v)),
      scales);
  }
}

//////////////////////////////////////////////////////////////////////////////
/// quantize_impl — Fallback path (pure C++, no inline ASM)
///
/// Steps 1-3 are handled by compute_block_scales.
/// Step 4 uses standard C++ multiply + NumericConverter for type conversion.
//////////////////////////////////////////////////////////////////////////////
template <typename SrcType, typename DstType,
          int NumBlocks, int BlockSize, int NumValues, int ScaleNumValues,
          class SrcLayout, class SrcTVLayout,
          class SrcEngine, class SrcLayoutWI,
          class DstFrag,
          class ScaleEngine, class ScaleLayoutWI, class ScaleTVLayout>
CUTE_HOST_DEVICE
void quantize_impl(
  QuantizeDispatchFallback const&,
  SubgroupTensor<SrcEngine, SrcLayoutWI, SrcTVLayout> const& src,
  DstFrag& tmp_dst_frag,
  SubgroupTensor<ScaleEngine, ScaleLayoutWI, ScaleTVLayout>& scale)
{
  constexpr auto tv_layout = SrcTVLayout{};
  constexpr auto logical_shape = atuple_coshape(tv_layout);
  constexpr int M = get<0>(logical_shape);
  constexpr int NumScales = M * NumBlocks;

  // Steps 1-3: Compute per-block scales and write to scale tensor
  float block_scale_f32[NumScales];
  compute_block_scales<SrcType, DstType, NumBlocks,
                       BlockSize, NumValues, ScaleNumValues,
                       SrcTVLayout, ScaleTVLayout>(
    src, scale, block_scale_f32);

  // Step 4: Quantize using pure C++ (multiply + convert via NumericConverter)
  using FloatToDst = cutlass::NumericConverter<DstType, float,
                                              cutlass::FloatRoundStyle::round_to_nearest>;
  CUTE_UNROLL
  for (int v = 0; v < NumValues; v++) {
    auto coord    = tv_layout(0, C<0>{} + v);
    int  m        = int(get<0>(coord));
    int  block_id = int(get<1>(coord)) / BlockSize;
    float s       = block_scale_f32[m + block_id * M];
    float scaled  = static_cast<float>(src.tensor()(v)) * s;
    tmp_dst_frag(v) = FloatToDst{}(scaled);
  }
}

//////////////////////////////////////////////////////////////////////////////
/// quantize_block_wise
///
/// Quantizes a high-precision source tensor to a low-precision destination
/// tensor with per-block scale factors following the MX format algorithm:
///   Step 0: Validate preconditions and extract compile-time constants
///   Step 1: Dispatch to quantize_impl (optimized or fallback)
///   Step 2: Reorder elements to destination layout
///
/// Preconditions:
///   - BlockSize >= sg_size (16).  Each quantization block must span at
///     least one full subgroup width for compile-time block index derivation.
///   - The source TV layout's thread stride must map ONLY to dimension 1 (N).
///     The row index (M) must be thread-independent so that tv_layout(0, v)
///     yields the correct row for all threads.
///
/// @tparam BlockSize  Number of elements per quantization block
//////////////////////////////////////////////////////////////////////////////
template <int BlockSize,
          class SrcEngine, class SrcLayout, class SrcTVLayout,
          class DstEngine, class DstLayout, class DstTVLayout,
          class ScaleEngine, class ScaleLayout, class ScaleTVLayout>
CUTE_HOST_DEVICE
void quantize_block_wise(
  SubgroupTensor<SrcEngine, SrcLayout, SrcTVLayout> const& src,
  SubgroupTensor<DstEngine, DstLayout, DstTVLayout>& dst,
  SubgroupTensor<ScaleEngine, ScaleLayout, ScaleTVLayout>& scale)
{
  using SrcType   = typename SrcEngine::element_type;
  using DstType   = typename DstEngine::element_type;
  using ScaleType = typename ScaleEngine::element_type;

  // ---------- Step 0: Validate preconditions ----------
  constexpr auto src_tv_layout     = SrcTVLayout{};
  constexpr auto logical_shape = atuple_coshape(src_tv_layout);
  constexpr int  M              = get<0>(logical_shape);
  constexpr int  N              = get<1>(logical_shape);
  constexpr int  NumBlocks      = N / BlockSize;
  constexpr int  NumValues      = cosize_v<SrcLayout>;
  constexpr int  ScaleNumValues = cosize_v<ScaleLayout>;

  static_assert(N % BlockSize == 0,
    "BlockSize must evenly divide the N dimension");
  static_assert(M % intel::sg_size == 0,
    "M must be a multiple of subgroup size for distributed scale layouts");
  static_assert(BlockSize >= intel::sg_size,
    "BlockSize must be >= subgroup size for compile-time block index");
  static_assert(BlockSize % intel::sg_size == 0,
    "BlockSize must be a multiple of the subgroup size for lane-independent block indices");

  // Verify dst logical shape matches src
  constexpr auto dst_shape = atuple_coshape(DstTVLayout{});
  static_assert(get<0>(dst_shape) == M && get<1>(dst_shape) == N,
    "Source and destination must have the same logical shape");

  // Verify scale shape is (M, NumBlocks)
  // When NumBlocks == 1, atuple_coshape may collapse to a 1D scalar
  constexpr auto scale_shape = atuple_coshape(ScaleTVLayout{});
  if constexpr (NumBlocks > 1) {
    static_assert(get<0>(scale_shape) == M,
      "Scale M dimension must match source M");
    static_assert(get<1>(scale_shape) == NumBlocks,
      "Scale N dimension must equal N/BlockSize");
  } else {
    // For the single-block case, allow rank-collapsed layouts but enforce:
    //   - Leading dimension equals M
    //   - Total number of elements equals M * NumBlocks
    static_assert(get<0>(scale_shape) == M,
      "Scale M dimension must match source M");
    constexpr int scale_num_elems = cosize_v<ScaleTVLayout>;
    static_assert(scale_num_elems == M * NumBlocks,
      "Scale tensor must have M * (N/BlockSize) elements");
  }

  // ---------- Step 1: Choose implementation and dispatch ----------
  auto impl = choose_xe_quantize_impl<SrcType, DstType, NumValues>();

  auto dst_frag = make_fragment_like<DstType>(src.tensor());
  quantize_impl<SrcType, DstType, NumBlocks,
                BlockSize, NumValues, ScaleNumValues,
                SrcLayout, SrcTVLayout>(
    impl, src, dst_frag, scale);

  // ---------- Step 2: Reorder to destination layout ----------
  auto dst_sgt = make_subgroup_tensor(dst_frag, src_tv_layout);
  reorder(dst_sgt, dst);
}

//////////////////////////////////////////////////////////////////////////////
/// quantize — Public API dispatcher
///
/// Delegates to quantize_block_wise.  See that function for the full
/// algorithm description and preconditions.
///
/// @tparam BlockSize  Number of elements per quantization block
/// @param  src        High-precision input  (half_t, BF16, FP32)
/// @param  dst        Low-precision output  (E4M3, E5M2, E2M1)
/// @param  scale      Block-wise scale factors output
//////////////////////////////////////////////////////////////////////////////
template <int BlockSize,
          class SrcEngine, class SrcLayout, class SrcTVLayout,
          class DstEngine, class DstLayout, class DstTVLayout,
          class ScaleEngine, class ScaleLayout, class ScaleTVLayout>
CUTE_HOST_DEVICE
void quantize(
  SubgroupTensor<SrcEngine, SrcLayout, SrcTVLayout> const& src,
  SubgroupTensor<DstEngine, DstLayout, DstTVLayout>& dst,
  SubgroupTensor<ScaleEngine, ScaleLayout, ScaleTVLayout>& scale)
{
  quantize_block_wise<BlockSize>(src, dst, scale);
}

// tensor-wise quantize: single global scale factor for the entire tensor (no blocking)
template <class SrcEngine, class SrcLayout, class SrcTVLayout,
          class DstEngine, class DstLayout, class DstTVLayout,
          class ScaleT>
CUTE_HOST_DEVICE
void quantize(
  SubgroupTensor<SrcEngine, SrcLayout, SrcTVLayout> const& src,
  SubgroupTensor<DstEngine, DstLayout, DstTVLayout>& dst,
  ScaleT& scale)
{
  static_assert(cute::dependent_false<ScaleT>, "Not implemented: tensor-wise quantization with a single global scale factor.");
}

} // namespace cute
