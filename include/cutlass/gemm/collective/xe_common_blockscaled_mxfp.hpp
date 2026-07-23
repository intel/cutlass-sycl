/***************************************************************************************************
 * Copyright (C) 2025 - 2026 Intel Corporation, All rights reserved.
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

#include "cutlass/cutlass.h"
#include "cute/algorithm/functional.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/copy_traits_xe_2d.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/layout.hpp"
#include "cute/tensor.hpp"

namespace cute
{

template <int Bits, int Height, int Width, int BlockWidth = Width>
struct MXFP_SCALE_LOAD_2D : XE_LOAD_2D<Bits, Height, Width, BlockWidth> {};

template <class XMode, class YMode, typename ValType, typename TiledStrides,
          int CopyBits, int Height, int Width, int BlockWidth>
struct Copy_Traits<MXFP_SCALE_LOAD_2D<CopyBits, Height, Width, BlockWidth>, XMode, YMode, ValType, TiledStrides>
    : Xe2DLoadTraitsBase<MXFP_SCALE_LOAD_2D<CopyBits, Height, Width, BlockWidth>, XMode, YMode, ValType, TiledStrides>
{
  using Op = MXFP_SCALE_LOAD_2D<CopyBits, Height, Width, BlockWidth>;
  using Super = Xe2DLoadTraitsBase<Op, XMode, YMode, ValType, TiledStrides>;
  using Super::Super;

  template <typename SEngine, typename SLayout>
  CUTE_DEVICE
  Copy_Traits(Tensor<SEngine, SLayout> const& src)
      : Super()
  {
    this->base_ptr = (uint64_t)&*src.data();
    this->tiled_strides = replace<XMode::value>(replace<YMode::value>(src.stride(), _0{}), _0{});

    constexpr auto SBits = sizeof_bits_v<typename SEngine::value_type>;
    uint32_t logical_width = (shape<XMode::value>(src) * SBits) >> 3;
    this->width = (logical_width + 3) & ~uint32_t(3);
    this->height = shape<YMode::value>(src);
    this->pitch = (stride<YMode::value>(src) * SBits) >> 3;

#ifdef CUTE_ENABLE_XE_BLOCK_2D_ASSERT
    assert((this->base_ptr % 64 == 0) && "CuTe runtime error: misaligned block 2D base pointer");
    assert((this->width % 4 == 0) && "CuTe runtime error: misaligned block 2D tensor width");
    assert((this->pitch % 4 == 0) && "CuTe runtime error: misaligned block 2D tensor pitch");
    assert((this->width <= 0xFFFFFF) && "CuTe runtime error: block 2D tensor width exceeds 2^24");
    assert((this->height <= 0xFFFFFF) && "CuTe runtime error: block 2D tensor height exceeds 2^24");
    assert((this->pitch <= 0xFFFFFF) && "CuTe runtime error: block 2D tensor pitch exceeds 2^24");
#endif
    this->device_init();
  }

  using DstLayout = XeInterleavedLayout<Layout<Shape<Int<BlockWidth>, Int<Height>, Int<Width/BlockWidth>>,
                                               Stride<_1, Int<Width>, Int<BlockWidth>>>,
                                        CopyBits,
                                        sizeof_bits_v<ValType>>;

  using RefLayout = DstLayout;
  using SrcLayout = decltype(replace<0>(RefLayout{}, Layout<Shape<intel::_SGSize>, Stride<_0>>{}));
};

} // namespace cute

namespace cutlass::gemm::collective
{

  using namespace cute;

  // -----------------------------------------------------------------------------
  // Traits for scale loading
  // -----------------------------------------------------------------------------

  template <class Dtype, size_t Height, size_t Width, class Stride = cute::Stride<_1, int64_t, int64_t>, class = void>
  struct ScaleCopyTraits
  {
    static_assert(cute::dependent_false<cute::tuple<Dtype, Int<Height>, Int<Width>, Stride>>, "ScaleCopyTraits not defined");
  };

  // 8 bits specialization for height <= 1 (covers 1 and 0 if applicable)
  template <class Dtype, size_t Height, size_t Width, class Stride>
  struct ScaleCopyTraits<Dtype, Height, Width, Stride, std::enable_if_t<sizeof_bits_v<Dtype> == 8>>
  {
    static_assert(Height > 0);
    // Width2D must between 32/64 due to limitation
    static constexpr auto Width2D = Width <= 32 ? 32 : 64;
    using Type = MXFP_SCALE_LOAD_2D<8, Height, Width2D, 32>;
  };

    template <class Dtype,class Stride>
  struct ScaleCopyTraits<Dtype, 1, 64, Stride, std::enable_if_t<sizeof_bits_v<Dtype> == 8>>
  {
    using Type = MXFP_SCALE_LOAD_2D<8, 1, 64>;
  };

  // -----------------------------------------------------------------------------
  // Helper Functions
  // -----------------------------------------------------------------------------

  // Helper to create scale copy iterator
  template <
      int TraitsSize,
      int TraitsNum,
      int SgK,
      int GroupK,
      class BlockShape>
  CUTLASS_DEVICE static auto
  make_scale_copy_iterator(int mn_coord, int l_coord, int k_count)
  {
    return make_tensor(make_inttuple_iter(make_coord(mn_coord, 0, l_coord)),
                       make_layout(make_shape(Int<TraitsSize>{}, Int<TraitsNum>{}, _1{}, k_count),
                                   make_stride(E<0>{} * _16{}, E<0>{} * size<1>(BlockShape{}),
                                               E<1>{} * size<0>(BlockShape{}), E<1>{} * cute::ceil_div(
                                                SgK, GroupK))));
  }

  // Helper to generate index data for GEMM offsets (M/N/Q dimension)
  template <int Iter, int Step, typename BlockShape>
  CUTLASS_DEVICE static auto
  make_scaled_offsets_mn()
  {
    auto offsets = make_tensor<uint16_t>(Layout<Shape<_1, Int<Iter>, _1>, Stride<_0, _1, _0>>{});
    constexpr auto height = size<0>(BlockShape{});
    constexpr auto width = size<1>(BlockShape{});
    constexpr auto mod = width / Step;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Iter; ++i)
    {
      offsets(i) = (i / mod) * decltype(size(BlockShape{}))::value + (i % mod) * Step;
    }
    return offsets;
  }

  template <int IterM, typename BlockShape>
  CUTLASS_DEVICE static auto
  make_scaled_offsets_m()
  {
    return make_scaled_offsets_mn<IterM, 8, BlockShape>();
  }

  template <int IterN, typename BlockShape>
  CUTLASS_DEVICE static auto
  make_scaled_offsets_n()
  {
    return make_scaled_offsets_mn<IterN, 16, BlockShape>();
  }

  // Helper to generate K offsets
  template <int IterK, int MmaK, int GroupK, typename BlockShape>
  CUTLASS_DEVICE static auto
  make_scaled_offsets_k()
  {
    auto offsets = make_tensor<uint16_t>(Layout<Shape<_1, _1, Int<IterK>>, Stride<_0, _0, _1>>{});
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < IterK; ++k)
    {
      offsets(k) = (MmaK / GroupK) * decltype(size<1>(BlockShape{}))::value * k;
    }
    return offsets;
  }

  template <typename ScaleCopy, typename Element, int SgMN, int SgK, int GroupK, typename Tensor>
  CUTLASS_DEVICE static auto
  make_scaled_copy(Tensor const &tensor, int mn_coord = 0, int l_coord = 0, int k_count = 0)
  {
    using Stride = cute::remove_cvref_t<decltype(tensor.stride())>;
    using NonVoidScaleTraits = 
        ScaleCopyTraits<Element, cute::ceil_div(SgK, GroupK), SgMN>;
    using NonVoidScaleCopy = typename NonVoidScaleTraits::Type;
    using SelectedCopy = cute::conditional_t<cute::is_void_v<ScaleCopy>, NonVoidScaleCopy, ScaleCopy>;

    // tile copy
    auto tiled_copy = make_block_2d_copy(SelectedCopy{}, tensor);

    using AtomShape = typename decltype(tiled_copy)::AtomShape;
    static_assert(size<0>(typename decltype(tiled_copy)::BlockShape{}) == 1
               || size<1>(typename decltype(tiled_copy)::BlockShape{}) == 32,
                "2D load width must be 32 to match MXFP4 BDPAS requirement for scale layout");

    // copy_iter
    constexpr auto SubgroupSize = 16;
    static constexpr auto scale_traits_size = decltype(size(AtomShape{}))::value / SubgroupSize;
    static constexpr auto scale_traits_num = cute::ceil_div(SgMN , size<1>(AtomShape{}));
    auto copy_iter = make_scale_copy_iterator<scale_traits_size, scale_traits_num, SgK, GroupK, AtomShape>(mn_coord, l_coord, k_count);

    // fragment
    auto fragment = make_tensor<Element>(Layout<Shape<Int<scale_traits_size>, Int<scale_traits_num>, _1>>{});

    return cute::make_tuple(tiled_copy, copy_iter, fragment);
  }

  template <typename ScaleCopy, int SgMN, int SgK, int GroupK>
  CUTLASS_DEVICE static auto
  make_scaled_prefetch(ScaleCopy const &tiled_copy, int mn_coord, int l_coord, int k_count)
  {
    // Create prefetch from tiled copy
    auto tiled_prefetch = make_block_2d_prefetch(tiled_copy);
    using PrefetchAtomShape = typename decltype(tiled_prefetch)::AtomShape;
    constexpr auto SubgroupSize = 16;
    static constexpr auto prefetch_traits_size = decltype(size(PrefetchAtomShape{}))::value / SubgroupSize;
    static constexpr auto prefetch_traits_num = cute::ceil_div(SgMN, int(size<1>(PrefetchAtomShape{})));

    auto prefetch_iter = make_scale_copy_iterator<prefetch_traits_size, prefetch_traits_num, SgK, GroupK, PrefetchAtomShape>(
        mn_coord, l_coord, k_count);

    return cute::make_tuple(tiled_prefetch, prefetch_iter);
  }

  template <int IterM, int IterN, int IterK, int MmaK, int GroupK, typename BlockShapeA, typename BlockShapeB>
  CUTLASS_DEVICE static auto
  make_scaled_offsets()
  {
    return cute::make_tuple(make_scaled_offsets_m<IterM, BlockShapeA>(),
                            make_scaled_offsets_n<IterN, BlockShapeB>(),
                            make_scaled_offsets_k<IterK, MmaK, GroupK, BlockShapeA>(),
                            make_scaled_offsets_k<IterK, MmaK, GroupK, BlockShapeB>());
  }

} // namespace cutlass::gemm::collective
