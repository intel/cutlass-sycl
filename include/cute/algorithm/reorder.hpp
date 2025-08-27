/***************************************************************************************************
 * Copyright (c) 2025 -----
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

#include <cute/config.hpp>            // CUTE_HOST_DEVICE
#include <cute/tensor_impl.hpp>       // cute::Tensor
#include <cute/atom/reorder_atom.hpp>

namespace cute
{

namespace detail {

// Modify subgroup TV layout for subbyte types.
//
// In general on Xe successive elements in registers are assigned to threads in
//   round-robin order (interleaved at element granularity). However, subbyte types are
//   only interleaved at byte granularity.
//
// This routine modifies the incoming layout to appear as though thread ownership for subbyte
//   types is also at element granularity, to uniformize later logic.
template <class T, class InLayout>
CUTE_HOST_DEVICE
constexpr decltype(auto)
subbyte_sg_tv_swizzle(const InLayout &layout)
{
  if constexpr (sizeof_bits_v<T> >= 8)
    return layout;
  else {
    static_assert(is_static_v<InLayout>, "Layout must be static");
    constexpr auto values = size(InLayout{}) / 16;
    constexpr auto per_byte = 8 / sizeof_bits_v<T>;
    static_assert(values % per_byte == 0, "Partially-occupied bytes in layout");
    return composition(layout, Layout<Shape<Shape<C<per_byte>, C<16/per_byte>>, Shape<C<per_byte>, C<values/per_byte>>>,
                                      Stride<Stride<_16, _1>, Stride<C<16/per_byte>, C<16*per_byte>>>>{});
  }
}

} /* namespace detail */

// Subgroup-cooperative reorder.
//          src, dst: WI-owned fragments
//  slayout, dlayout: subgroup TV-layouts for these fragments.
//
// The layout of src/dst can be arbitrary. The TV layouts
//   are used to map values in src to values in dst.
template <class SEngine, class SLayoutWI, class SLayout,
          class DEngine, class DLayoutWI, class DLayout>
CUTE_HOST_DEVICE
void
reorder(Tensor<SEngine,SLayoutWI> const& src,       // WI fragment
        Tensor<DEngine,DLayoutWI> &      dst,       // WI fragment
        SLayout                   const& slayout,   // (src thr, src val) -> coord
        DLayout                   const& dlayout)   // (dst thr, dst val) -> coord
{
  using SType = typename SEngine::element_type;
  using DType = typename DEngine::element_type;

  static_assert(is_static_v<SLayout>, "Reorder source layout must be static");
  static_assert(is_static_v<DLayout>, "Reorder destination layout must be static");

  auto sl0 = detail::subbyte_sg_tv_swizzle<SType>(project_strides(slayout));
  auto dl0 = detail::subbyte_sg_tv_swizzle<DType>(project_strides(dlayout));

  auto impl = choose_xe_reorder_impl<SType, DType>(sl0, dl0);   // -> atom or dispatch tag

  reorder_impl(impl, src, dst, sl0, dl0);
}

template <class SEngine, class SLayoutWI, class SLayout,
          class DEngine, class DLayoutWI, class DLayout>
CUTE_HOST_DEVICE
void
reorder(SubgroupTensor<SEngine,SLayoutWI,SLayout> const& src,
        SubgroupTensor<DEngine,DLayoutWI,DLayout> &      dst)
{
  reorder(src, dst, SLayout{}, DLayout{});
}

// Base case for reorders: loop over reorder atoms
template <class ReorderAtom,
          class SEngine, class SLayoutWI, class SLayout,
          class DEngine, class DLayoutWI, class DLayout>
CUTE_HOST_DEVICE
void
reorder_impl(ReorderAtom               const& atom,
             Tensor<SEngine,SLayoutWI> const& src,       // WI fragment
             Tensor<DEngine,DLayoutWI> &      dst,       // WI fragment
             SLayout                   const& slayout,   // (src thr, src val) -> coord
             DLayout                   const& dlayout)   // (dst thr, dst val) -> coord
{
  using SType = typename SEngine::element_type;
  using RegistersSrc = typename ReorderAtom::SRegisters;
  using RegistersDst = typename ReorderAtom::DRegisters;
  using RegTypeSrc   = typename remove_extent<RegistersSrc>::type;
  using RegTypeDst   = typename remove_extent<RegistersDst>::type;
  constexpr int RegNumSrc = extent<RegistersSrc>::value;
  constexpr int RegNumDst = extent<RegistersDst>::value;
  constexpr int values = size(SLayout{}) / size<0>(SLayout{});
  constexpr int vchunk = sizeof_bits_v<RegistersSrc> / sizeof_bits_v<SType>;

  // Calculate mapping from src val -> dst val on a chunk-by-chunk basis. Unlike a plain copy, there is no intrinsic
  //   correspondence of src/dst values for subgroup reorders.
  auto rlayout = coalesce(composition(right_inverse(dlayout), slayout));                 // src index -> dst index
  auto vrlayout = composition(composition(Layout<Shape<_16, Int<values>>, Stride<_0, _1>>{},
                                          rlayout),
                              Layout<Shape<_1, Int<values>>, Stride<_0, _16>>{});        // src val -> dst val

  CUTE_UNROLL
  for (int sv = 0; sv < values; sv += vchunk) {
    auto pS = recast_ptr<RegTypeSrc>(src.data() + sv);
    auto pD = recast_ptr<RegTypeDst>(dst.data() + vrlayout(sv));

    detail::explode(detail::CallReorder<ReorderAtom>{},
                    pS, make_int_sequence<RegNumSrc>{},
                    pD, make_int_sequence<RegNumDst>{});
  }
}

template <typename T>
using upcast_subbyte_t = conditional_t<is_subbyte_v<T>,
                                       conditional_t<cutlass::platform::numeric_limits<T>::is_integer,
                                                     conditional_t<cutlass::platform::numeric_limits<T>::is_signed,
                                                                   int8_t, uint8_t>,
                                                     half_t>,
                                       T>;

// Reorder strategy: type conversion, then layout change.
template <class SEngine, class SLayoutWI, class SLayout,
          class DEngine, class DLayoutWI, class DLayout>
CUTE_HOST_DEVICE
void
reorder_impl(ReorderDispatchConvertRelayout const&,
             Tensor<SEngine,SLayoutWI> const& src,       // WI fragment
             Tensor<DEngine,DLayoutWI> &      dst,       // WI fragment
             SLayout                   const& slayout,   // (src thr, src val) -> coord
             DLayout                   const& dlayout)   // (dst thr, dst val) -> coord
{
  using SrcType = typename SEngine::element_type;
  using DstType = typename DEngine::element_type;
  using NewSrcType = conditional_t<is_same_v<SrcType, DstType>, upcast_subbyte_t<SrcType>, DstType>;
  auto src_c = make_fragment_like<NewSrcType>(src);

  reorder(src, src_c, slayout, slayout);
  reorder(src_c, dst, slayout, dlayout);
}

// Reorder strategy: layout change, then type conversion
template <class SEngine, class SLayoutWI, class SLayout,
          class DEngine, class DLayoutWI, class DLayout>
CUTE_HOST_DEVICE
void
reorder_impl(ReorderDispatchRelayoutConvert const&,
             Tensor<SEngine,SLayoutWI> const& src,       // WI fragment
             Tensor<DEngine,DLayoutWI> &      dst,       // WI fragment
             SLayout                   const& slayout,   // (src thr, src val) -> coord
             DLayout                   const& dlayout)   // (dst thr, dst val) -> coord
{
  using SrcType = typename SEngine::element_type;
  using DstType = typename DEngine::element_type;
  using NewDstType = conditional_t<is_same_v<SrcType, DstType>, upcast_subbyte_t<DstType>, SrcType>;
  auto dst_c = make_fragment_like<NewDstType>(dst);

  reorder(src, dst_c, slayout, dlayout);
  reorder(dst_c, dst, dlayout, dlayout);
}

// Copy a strided vector to a strided vector in GRF.
//   src and dst must each fit within a single register.
template <int simd, int sstride, int dstride, int sidx, int didx,
          class SEngine, class SLayoutWI,
          class DEngine, class DLayoutWI>
CUTE_HOST_DEVICE
void
reorder_span(Tensor<SEngine,SLayoutWI> const& src,
             Tensor<DEngine,DLayoutWI> &      dst)
{
  using ValType = typename SEngine::element_type;
  using StorageType = intel::storage_vector_t<ValType, 32>;
  constexpr int grf_elems = 64 / sizeof(ValType);
  const auto& sv = *recast_ptr<StorageType>(src.data() + ((sidx / grf_elems) * (grf_elems / 16)));
  auto&       dv = *recast_ptr<StorageType>(dst.data() + ((didx / grf_elems) * (grf_elems / 16)));
  constexpr auto soff = sidx % grf_elems;
  constexpr auto doff = didx % grf_elems;
#ifdef __SYCL_DEVICE_ONLY__
  asm (
    "mov (M1_NM, %2) %0(0,%5)<%3> %1(0,%6)<%4;1,0>"
    : "+rw"(dv)
    : "rw"(sv), "P"(simd), "P"(dstride), "P"(sstride), "P"(doff), "P"(soff)
  );
#endif
}

// Generic Xe reorders, supporting arbitrary layout changes, but not type conversions.
template <class SEngine, class SLayoutWI, class SLayout,
          class DEngine, class DLayoutWI, class DLayout>
CUTE_HOST_DEVICE
void
reorder_impl(ReorderDispatchXeGeneric  const&,
             Tensor<SEngine,SLayoutWI> const& src,       // WI fragment
             Tensor<DEngine,DLayoutWI> &      dst,       // WI fragment
             SLayout                   const& slayout,   // (src thr, src val) -> coord
             DLayout                   const& dlayout)   // (dst thr, dst val) -> coord
{
  using SrcType = typename SEngine::element_type;
  using DstType = typename DEngine::element_type;
  static_assert(is_same_v<SrcType, DstType>, "No type conversions allowed on this path");

  auto rlayout = coalesce(composition(right_inverse(dlayout), slayout));          // src index -> dst index
  auto ilayout = coalesce(composition(right_inverse(slayout), dlayout));          // dst index -> src index

  // Decide whether to stride on src or dst, depending on which allows a longer vector length.
  static constexpr int elems_per_grf = 64 / sizeof(SrcType);
  static constexpr int ds_vl = cute::min(32, cute::min(shape<0>(rlayout), elems_per_grf / stride<0>(rlayout)));
  static constexpr int ss_vl = cute::min(32, cute::min(shape<0>(ilayout), elems_per_grf / stride<0>(ilayout)));

  // Make dst live, to prevent compiler from inserting its own initialization.
#ifdef __SYCL_DEVICE_ONLY__
  using StorageType = intel::storage_vector_t<DstType, 32>;

  CUTE_UNROLL
  for (int i = 0; i < dst.size(); i += 4 / sizeof(DstType)) {
    auto &dv = *recast_ptr<StorageType>(dst.data() + i);
    asm("" : "=rw"(dv));
  }
#endif

  if constexpr (ss_vl >= ds_vl) {
    // Stride on src. For simplicity, take 1 GRF at a time.
    for_each(make_seq<size(SLayout{}) / ss_vl>{}, [&](auto i) {
      constexpr auto didx = i * ss_vl;
      constexpr auto sidx = ilayout(didx);
      reorder_span<ss_vl, stride<0>(decltype(ilayout){}), 1, sidx, didx>(src, dst);
    });
  } else {
    // Stride on dst.
    for_each(make_seq<size(SLayout{}) / ds_vl>{}, [&](auto i) {
      constexpr auto sidx = i * ds_vl;
      constexpr auto didx = rlayout(sidx);
      reorder_span<ds_vl, 1, stride<0>(decltype(rlayout){}), sidx, didx>(src, dst);
    });
  }
}

} // end namespace cute
