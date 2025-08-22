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

#include <cute/atom/copy_atom.hpp>
#include <cute/atom/copy_traits.hpp>

#include <cute/algorithm/prefetch.hpp>
#include <cute/arch/copy_xe_2d.hpp>

// 2D block payload intrinsics
SYCL_EXTERNAL extern "C" int* __builtin_IB_subgroup_createBlock2DAddressPayload(long base, int width_minus_one, int height_minus_one, int pitch_minus_one,
                                                                                int blockX, int blockY, int blockWidth, int blockHeight, int numBlocks);
SYCL_EXTERNAL extern "C" int* __builtin_IB_subgroup_copyBlock2DAddressPayload(int* AP);

SYCL_EXTERNAL extern "C" void __builtin_IB_subgroup_addBlock2DAddressPayloadBlockX(int* addrPayload, int blockX);
SYCL_EXTERNAL extern "C" void __builtin_IB_subgroup_addBlock2DAddressPayloadBlockY(int* addrPayload, int blockY);
SYCL_EXTERNAL extern "C" void __builtin_IB_subgroup_setBlock2DAddressPayloadBlockX(int* addrPayload, int blockX);
SYCL_EXTERNAL extern "C" void __builtin_IB_subgroup_setBlock2DAddressPayloadBlockY(int* addrPayload, int blockY);
SYCL_EXTERNAL extern "C" void __builtin_IB_subgroup_setBlock2DAddressPayloadBase(int* addrPayload, long base);
SYCL_EXTERNAL extern "C" void __builtin_IB_subgroup_setBlock2DAddressPayloadWidth(int* addrPayload, int width_minus_one);
SYCL_EXTERNAL extern "C" void __builtin_IB_subgroup_setBlock2DAddressPayloadHeigth(int* addrPayload, int height_minus_one);
SYCL_EXTERNAL extern "C" void __builtin_IB_subgroup_setBlock2DAddressPayloadPitch(int* addrPayload, int pitch_minus_one);


namespace cute {

// Utility to check if a layout belongs to a coordinate tensor.
template <typename Layout>
static constexpr bool is_counting_layout_v = is_arithmetic_tuple_like<decltype(Layout{}(0))>::value;




// Base traits class for block 2D messages.
//
// XMode and YMode are mode indices into the tensor, identifying which modes map to the block 2D dimensions.
//   X: consecutive dimension
//   Y: strided dimension internal to the copy atom
// While individual atoms perform 2D copies, additional dimensions are supported by tiling.
//
// If the value type of the tensor has a different size from the underlying copy atoms,
//   it must be specified via the ValType template argument. Due to the SIMD-like layout of data
//   in registers, the generic CuTe code for handling type size changes (via Copy_Atom) does not
//   work properly in most cases.
template <class Op, class XMode, class YMode, typename ValType, typename TiledStrides = Stride<_1>>
struct Xe2DTraitsBase
{
  using Traits = Copy_Traits<Op, XMode, YMode, ValType, TiledStrides>;
  using ThrID = Layout<_16>;

  static constexpr int ValBits = is_void_v<ValType> ? Op::CopyBits
                                                    : int(sizeof_bits_v<ValType>);
  static_assert(Op::CopyBits % ValBits == 0, "Type is incompatible with this copy atom");

  // Payload for 2D block message:
  //   - base pointer
  //   - matrix width/height/pitch in global memory
  //   - x/y offsets (overwritten during each copy operation)
  //   - block width/height/count
  // Note the payload is mutable to allow x/y offsets to be dynamically updated for each use.
  mutable int *payload;

  // Copy of base pointer, to allow payload updates for >2D tensors.
  uint64_t base_ptr;

  // Strides not handled by block 2D operations (>2D tensors).
  TiledStrides tiled_strides;

  static constexpr bool nontrivial_tiled_strides = !is_static_v<TiledStrides>
      || !is_constant_v<0, decltype(cute::max(TiledStrides{}))>;

  // Uninitialized atom, available on host or device.
  CUTE_HOST_DEVICE
  Xe2DTraitsBase() {}

  // Initialized atom, device-only.
  template <typename SEngine, typename SLayout>
  CUTE_DEVICE
  Xe2DTraitsBase(Tensor<SEngine, SLayout> const& src)
      : base_ptr((uint64_t) &*src.data()),
        tiled_strides(replace<XMode::value>(replace<YMode::value>(src.stride(), _0{}), _0{}))
  {
    constexpr auto SBits = sizeof_bits_v<typename SEngine::value_type>;
    uint32_t width = (shape<XMode::value>(src) * SBits) >> 3;
    uint32_t height = shape<YMode::value>(src);
    uint32_t pitch = (stride<YMode::value>(src) * SBits) >> 3;
#ifdef CUTE_ENABLE_XE_BLOCK_2D_ASSERT
    assert((base_ptr % 64 == 0) && "CuTe runtime error: misaligned block 2D base pointer");
    assert((width % 4 == 0) && "CuTe runtime error: misaligned block 2D tensor width");
    assert((pitch % 4 == 0) && "CuTe runtime error: misaligned block 2D tensor pitch");
    assert((width <= 0xFFFFFF) && "CuTe runtime error: block 2D tensor width exceeds 2^24");
    assert((height <= 0xFFFFFF) && "CuTe runtime error: block 2D tensor height exceeds 2^24");
    assert((pitch <= 0xFFFFFF) && "CuTe runtime error: block 2D tensor pitch exceeds 2^24");
#endif
#ifdef __SYCL_DEVICE_ONLY__
    payload = __builtin_IB_subgroup_createBlock2DAddressPayload(
      base_ptr,
      width - 1,
      height - 1,
      pitch - 1,
      0,  /* x offset, configured per-copy */
      0,  /* y offset, configured per-copy */
      Op::AtomWidth / Op::BlockCount,
      Op::AtomHeight,
      Op::BlockCount
    );
#endif
  }

  template <int Bits, typename Coord>
  CUTE_DEVICE
  void update_payload(const Coord &coord) const
  {
    // Update x/y offsets in payload
    int32_t x = get<XMode::value>(coord) * Bits / Op::CopyBits;
    int32_t y = get<YMode::value>(coord);
    __builtin_IB_subgroup_setBlock2DAddressPayloadBlockX(payload, x);
    __builtin_IB_subgroup_setBlock2DAddressPayloadBlockY(payload, y);

#ifdef CUTE_ENABLE_XE_BLOCK_2D_ASSERT
    assert((x % 4 == 0) && "CuTe runtime error: misaligned block 2D x offset");
#endif

    // Perform stride calculation and update base pointer for > 2D tensors
    if constexpr (nontrivial_tiled_strides) {
      auto offset = inner_product(coord, tiled_strides);
      auto byte_offset = (offset * Bits) >> 3;
      __builtin_IB_subgroup_setBlock2DAddressPayloadBase(payload, base_ptr + byte_offset);

#ifdef CUTE_ENABLE_XE_BLOCK_2D_ASSERT
      assert((byte_offset % 64 == 0) && "CuTe runtime error: misaligned block 2D base pointer");
#endif
    }
  }

  static constexpr auto get_x_mode() { return XMode{}; }
  static constexpr auto get_y_mode() { return YMode{}; }
};

template <class Op, class XMode, class YMode, typename ValType, typename TiledStrides = Stride<_1>>
struct Xe2DLoadTraitsBase : Xe2DTraitsBase<Op, XMode, YMode, ValType, TiledStrides>
{
  using Super = Xe2DTraitsBase<Op, XMode, YMode, ValType, TiledStrides>;
  using Traits = typename Super::Traits;
  using ThrID = typename Super::ThrID;

  // Provide a global memory tensor to a previously-uninitialized atom.
  template <typename SEngine, typename SLayout>
  CUTE_DEVICE static auto
  with(Tensor<SEngine, SLayout> const& src) {
    return Traits(src);
  }

  // Execution.
  template <class SEngine, class SLayout,
            class DEngine, class DLayout>
  CUTE_DEVICE friend constexpr void
  copy_unpack(Traits const&                   traits,
              Tensor<SEngine, SLayout> const& src,
              Tensor<DEngine, DLayout> &      dst) {
    using SType = typename SEngine::value_type;
    using DType = typename DEngine::value_type;
    using SrcLayout = typename Traits::SrcLayout;
    using DstLayout = typename Traits::DstLayout;
    constexpr auto DBits = sizeof_bits_v<DType>;

    static_assert(is_counting_layout_v<SLayout>, "Source tensor must be a coordinate tensor.");
    static_assert(is_rmem_v<DEngine>, "Destination tensor must be in registers.");
    static_assert(size(SLayout{}) * DBits == size<1>(SrcLayout{}),
                  "Source tensor size does not match copy atom size.");
    static_assert(size(DLayout{}) * DBits == size<1>(DstLayout{}),
                  "Destination tensor size does not match copy atom size.");

    traits.template update_payload<DBits>(src.data().coord_);
    Op::copy(traits.payload, recast_ptr<int_byte_t<bits_to_bytes(Super::ValBits)>>(&*dst.data()));
  }
};


// Split a subgroup-level layout into a TV-layout.
template <typename InLayout, int CopyBits, int ValBits, int Threads>
struct XeInterleavedLayoutHelper {
  // Underlying SIMD vector type's element width:
  static constexpr int VecTypeBits = cute::max(ValBits, 8);

  // Expand from CopyBits to VecTypeBits in x dimension:
  using Expanded = decltype(logical_product(Layout<Shape<Int<CopyBits/VecTypeBits>>>{}, InLayout{}));  // V' -> (x', y)

  // Split elements between work-items, interleaving:
  using TVLayout = decltype(composition(Expanded{}, make_layout(make_shape(Int<Threads>{}, Int<size(Expanded{})/Threads>{}))));

  // Expand from elements to bits:
  using PreResult = decltype(blocked_product(Layout<Shape<_1, Int<VecTypeBits>>>{}, TVLayout{}));

  // Simplify for nicer-looking layouts:
  using Result = decltype(coalesce(PreResult{}, Step<_1, _1>{}));

  // Examples:

  // U16 32x16 nontranspose -> U4/U8
  //  In:  (_32, _16):(_1, _32)                                           V -> (x,y)
  // Exp:  (_2, _32, _16):(_1, _2, _64)                                Vbit -> (xbit,y)
  //   Compose with (_16, _64):(_1, _16)
  //  TV:  (_16, _64):(_1, _16)
  // Res:  (_16, (_8, _64)):(_8, (_1, _128))

  // U32 8x16 transpose -> U16 (16x16)  LD_T
  //  In:  (_16, _8):(_8, _1)                                             V -> (x,y)
  // Exp:  (_2, _16, _8):(_1, _16, _2)                                  V16 -> (x16,y)
  //    Compose with (_16, _16):(_1, _16)
  //  TV:  ((_2, _8), (_2, _8)):((_1, _16), (_128, _2))               (T,V) -> (x16,y)
  // Res:  ((_2, _8), (_16, _2, _8)):((_16, _256), (_1, _2048, _32))  (T,V) -> (xbit,y)
};

template <typename Layout, int CopyBits, int ValBits, int Threads = 16>
using XeInterleavedLayout = typename XeInterleavedLayoutHelper<Layout, CopyBits, ValBits, Threads>::Result;

// Block 2D load traits.
template <class XMode, class YMode, typename ValType, typename TiledStrides,
          int CopyBits, int Height, int Width, int BlockWidth>
struct Copy_Traits<XE_LOAD_2D<CopyBits, Height, Width, BlockWidth>, XMode, YMode, ValType, TiledStrides>
    : Xe2DLoadTraitsBase<XE_LOAD_2D<CopyBits, Height, Width, BlockWidth>, XMode, YMode, ValType, TiledStrides>
{
  // (dst-thr, dst-val) -> (x, y)
  using DstLayout = XeInterleavedLayout<Layout<Shape<Int<BlockWidth>, Int<Height>, Int<Width/BlockWidth>>,
                                               Stride<_1, Int<Width>, Int<BlockWidth>>>,
                                        CopyBits,
                                        sizeof_bits_v<ValType>>;

  using RefLayout = DstLayout;
  using SrcLayout = decltype(replace<0>(RefLayout{}, Layout<Shape<_16>, Stride<_0>>{}));
};

// Block 2D VNNI load traits.
template <class XMode, class YMode, typename ValType, typename TiledStrides,
          int CopyBits, int Height, int Width, int BlockWidth>
struct Copy_Traits<XE_LOAD_2D_VNNI<CopyBits, Height, Width, BlockWidth>, XMode, YMode, ValType, TiledStrides>
    : Xe2DLoadTraitsBase<XE_LOAD_2D_VNNI<CopyBits, Height, Width, BlockWidth>, XMode, YMode, ValType, TiledStrides>
{
  static constexpr int BV = 32 / CopyBits;

  // (dst-thr, dst-val) -> (x, y)
  using DstLayout = XeInterleavedLayout<Layout<Shape<Int<BV>, Int<BlockWidth>, Int<Height/BV>, Int<Width/BlockWidth>>,
                                               Stride<Int<Width>, _1, Int<Width*BV>, Int<BlockWidth>>>,
                                        CopyBits,
                                        sizeof_bits_v<ValType>>;

  using RefLayout = DstLayout;
  using SrcLayout = decltype(replace<0>(RefLayout{}, Layout<Shape<_16>, Stride<_0>>{}));
};

// Block 2D transposed load traits.
template <class XMode, class YMode, typename ValType, typename TiledStrides,
          int CopyBits, int Height, int Width>
struct Copy_Traits<XE_LOAD_2D_TRANSPOSE<CopyBits, Height, Width>, XMode, YMode, ValType, TiledStrides>
    : Xe2DLoadTraitsBase<XE_LOAD_2D_TRANSPOSE<CopyBits, Height, Width>, XMode, YMode, ValType, TiledStrides>
{
  // (dst-thr, dst-val) -> (x, y)
  using DstLayout = XeInterleavedLayout<Layout<Shape<Int<Height>, Int<Width>>,
                                               Stride<Int<Width>, _1>>,
                                        CopyBits,
                                        sizeof_bits_v<ValType>>;

  using RefLayout = DstLayout;
  using SrcLayout = decltype(replace<0>(RefLayout{}, Layout<Shape<_16>, Stride<_0>>{}));
};

// Block 2D store traits.
template <class XMode, class YMode, typename ValType, typename TiledStrides,
          int CopyBits, int Height, int Width>
struct Copy_Traits<XE_STORE_2D<CopyBits, Height, Width>, XMode, YMode, ValType, TiledStrides>
    : Xe2DTraitsBase<XE_STORE_2D<CopyBits, Height, Width>, XMode, YMode, ValType, TiledStrides>
{
  // (src-thr, src-val) -> (x, y)
  using SrcLayout = XeInterleavedLayout<Layout<Shape<Int<Width>, Int<Height>>>,
                                        CopyBits,
                                        sizeof_bits_v<ValType>>;

  using RefLayout = SrcLayout;
  using DstLayout = decltype(replace<0>(RefLayout{}, Layout<Shape<_16>, Stride<_0>>{}));

  using Op = XE_STORE_2D<CopyBits, Height, Width>;
  using Super = Xe2DTraitsBase<Op, XMode, YMode, ValType, TiledStrides>;
  using Traits = typename Super::Traits;  // a.k.a. this class
  using ThrID = typename Super::ThrID;

  // Provide a global memory tensor to a previously-uninitialized atom.
  template <typename DEngine, typename DLayout>
  CUTE_DEVICE static auto
  with(Tensor<DEngine, DLayout> const& dst) {
    return Traits(dst);
  }

  // Execution.
  template <class SEngine, class SLayout,
            class DEngine, class DLayout>
  CUTE_DEVICE friend constexpr void
  copy_unpack(Traits const&                   traits,
              Tensor<SEngine, SLayout> const& src,
              Tensor<DEngine, DLayout> &      dst) {
    using SType = typename SEngine::value_type;
    using DType = typename DEngine::value_type;
    using SrcLayout = typename Traits::SrcLayout;
    using DstLayout = typename Traits::DstLayout;
    constexpr auto SBits = sizeof_bits_v<SType>;

    static_assert(is_counting_layout_v<DLayout>, "Destination tensor must be a coordinate tensor.");
    static_assert(is_rmem_v<SEngine>, "Source tensor must be in registers.");
    static_assert(size(SLayout{}) * SBits == size<1>(SrcLayout{}),
                  "Source tensor size does not match copy atom size.");
    static_assert(size(DLayout{}) * SBits == size<1>(DstLayout{}),
                  "Destination tensor size does not match copy atom size.");

    traits.template update_payload<SBits>(dst.data().coord_);
    Op::copy(traits.payload, recast_ptr<int_byte_t<bits_to_bytes(Super::ValBits)>>(&*src.data()));
  }
};

// Block 2D prefetch traits.
//
// Note prefetch does not use/need block width; it is present for template arg compatibility
//  between loads and their prefetches.
template <class XMode, class YMode, typename ValType, typename TiledStrides,
          int CopyBits, int Height, int Width, int BlockWidth>
struct Copy_Traits<XE_PREFETCH_2D<CopyBits, Height, Width, BlockWidth>, XMode, YMode, ValType, TiledStrides>
    : Xe2DTraitsBase<XE_PREFETCH_2D<CopyBits, Height, Width, BlockWidth>, XMode, YMode, ValType, TiledStrides>
{
  // (dst-thr, dst-val) -> (x, y)
  using DstLayout = XeInterleavedLayout<Layout<Shape<Int<Width>, Int<Height>>>,
                                        CopyBits,
                                        sizeof_bits_v<ValType>>;

  using RefLayout = DstLayout;
  using SrcLayout = decltype(replace<0>(RefLayout{}, Layout<Shape<_16>, Stride<_0>>{}));

  using Op = XE_PREFETCH_2D<CopyBits, Height, Width, BlockWidth>;
  using Super = Xe2DTraitsBase<Op, XMode, YMode, ValType, TiledStrides>;
  using Traits = typename Super::Traits;  // a.k.a. this class
  using ThrID = typename Super::ThrID;

  using Super::Super;

  // Provide a global memory tensor to a previously-uninitialized atom.
  template <typename SEngine, typename SLayout>
  CUTE_DEVICE static auto
  with(Tensor<SEngine, SLayout> const& src) {
    return Traits(src);
  }

  // Execution.
  template <class SEngine, class SLayout,
            class DEngine, class DLayout>
  CUTE_DEVICE friend constexpr void
  copy_unpack(Traits const&                   traits,
              Tensor<SEngine, SLayout> const& src,
              Tensor<DEngine, DLayout> &      dst) {
    using SType = typename SEngine::value_type;
    using SrcLayout = typename Traits::SrcLayout;

    static_assert(is_counting_layout_v<SLayout>, "Source tensor must be a coordinate tensor.");
    static_assert(size(SLayout{}) * Super::ValBits == size<1>(SrcLayout{}),
                  "Source tensor size does not match copy atom size.");

    traits.template update_payload<Super::ValBits>(src.data().coord_);
    Op::copy(traits.payload);
  }
};

// Helpers for creating a tiling of block 2D copy atoms for a given global memory tensor.
//
// The x/y modes are deduced according to the rules:
//   x: innermost constant-stride-1 mode
//   y: innermost dynamic-stride mode, or innermost non-1 stride if there are no dynamic strides.
template <class CopyOp,
          class Engine, class Layout>
CUTE_HOST_DEVICE
auto
make_block_2d_copy(const CopyOp& op, const Tensor<Engine, Layout>& gmem) {
  return make_block_2d_copy<typename Engine::value_type>(op, gmem.stride()).with(gmem);
}

template <class OptionalValType = void, class CopyOp, class... Strides>
CUTE_HOST_DEVICE
auto
make_block_2d_copy(const CopyOp& op, const Stride<Strides...>&)
{
  // Configure traits for this atom, identifying x and y modes.
  using ValType = std::conditional_t<std::is_void_v<OptionalValType>,
                                     int_bit_t<CopyOp::CopyBits>,
                                     OptionalValType>;

  Stride<Strides...> strides{};
  return make_block_2d_copy<ValType>(op, find_x_mode(strides), find_y_mode(strides), strides);
}

template <class ValType, class XMode, class YMode, class CopyOp, class... Strides>
CUTE_HOST_DEVICE
auto
make_block_2d_copy(const CopyOp& op, const XMode&, const YMode&, const Stride<Strides...>&)
{
  static constexpr auto ValBits = sizeof_bits_v<ValType>;

  Stride<Strides...> strides{};
  XMode x_mode{};
  YMode y_mode{};

  using TiledStrides = decltype(replace<x_mode()>(replace<y_mode()>(strides, _0{}), _0{}));

  using Traits = Copy_Traits<CopyOp, XMode, YMode, ValType, TiledStrides>;
  using Atom = Copy_Atom<Traits, ValType>;

  // Create tiler for the TiledCopy.
  constexpr auto tile_1 = tuple_repeat<rank(strides)>(_1{});
  constexpr auto Width = CopyOp::AtomWidth * CopyOp::CopyBits / ValBits;
  constexpr auto Height = CopyOp::AtomHeight;
  using ShapeTiler_MN = decltype(replace<x_mode()>(replace<y_mode()>(tile_1, Int<Height>{}), Int<Width>{}));

  // Create proper TV-layout for the TiledCopy, using the copy atom's reference layout.
  //
  // ValLayoutRef for all block 2D atoms is (T,V)->(X,Y).
  // If the x/y ordering in ValLayoutRef matches the order of XMode/YMode in the given strides, then
  //    the TiledCopy's TV-layout is just ValLayoutRef. Otherwise, we need to transpose x/y in the RefLayout.
  constexpr bool transpose_tv = (y_mode < x_mode);
  using MaybeTranspose = Layout<Shape<Int<Width>, Int<Height>>,
                                Stride<Int<transpose_tv ? Height : 1>,
                                       Int<transpose_tv ? 1 : Width>>>;
  using LayoutCopy_TV = decltype(composition(MaybeTranspose{}, typename Atom::ValLayoutRef{}));

  return TiledCopy<Atom, LayoutCopy_TV, ShapeTiler_MN>{};
}

// Low-level routine for creating a block 2D TiledCopy for multiple subgroups.
// In addition to the usual parameters, it takes:
//   - atom_shape = "subgroup shape" (# of copy blocks in each dimension)
//   - sv_layout = "subgroup-value layout" (ordering for subgroups in the tiling)
template <class ValType, class XMode, class YMode,
          class CopyOp, class... Strides,
          class SGShape, class SVLayout>
CUTE_HOST_DEVICE
auto
make_block_2d_copy(const CopyOp& op,
                   const XMode& x_mode, const YMode& y_mode,
                   const Stride<Strides...>& strides,
                   const SGShape& atom_shape,             // (SG_M, SG_N, ...)
                   const SVLayout& sv_layout)           // (SG #, SG value) -> (SG_M, SG_N, ...)
{
  // Create TiledCopy for a single subgroup.
  using SGCopy = decltype(make_block_2d_copy<ValType>(op, x_mode, y_mode, strides));
  using Atom          = typename SGCopy::Atom;
  using ShapeTiler_MN = typename SGCopy::Tiler_MN;
  using LayoutCopy_TV = typename SGCopy::TiledLayout_TV;

  // Expand the shape.
  auto x_shape = elem_scale(ShapeTiler_MN{}, atom_shape);

  // Expand the single-SG TV layout to the full shape, then tile.
  auto x_tv_layout1 = composition(make_layout(ShapeTiler_MN{}, make_layout(x_shape).stride()), LayoutCopy_TV{});
  auto x_tv_layout = blocked_product(x_tv_layout1, sv_layout);

  return TiledCopy<Atom, decltype(x_tv_layout), decltype(x_shape)>{};
}


template <class... Strides>
CUTE_HOST_DEVICE
constexpr auto
find_x_mode(const Stride<Strides...> &) {
  Stride<Strides...> strides{};
  return find_if(strides, [](auto const &x) { return C<is_constant_v<1, decltype(x)>>{}; });
}

template <class... Strides>
CUTE_HOST_DEVICE
constexpr auto
find_y_mode(const Stride<Strides...>&) {
  Stride<Strides...> strides{};
  constexpr auto YModeDyn = find_if(strides, [](auto const &x) { return C<std::is_integral_v<decltype(x)>>{}; });
  if constexpr (YModeDyn < rank(strides))
    return YModeDyn;
  else
    return find_if(strides, [](auto const &x) { return C<!is_constant_v<1, decltype(x)>>{}; });
}

// Prefetch selection and creation.
namespace detail {
  template <class Op, class XMode, class YMode, typename ValType, typename TiledStrides>
  CUTE_HOST_DEVICE decltype(auto)
  as_block_2d_traits(Xe2DTraitsBase<Op, XMode, YMode, ValType, TiledStrides> const &o) {
    return o;
  }
};

template <int SGCount, class Copy_Atom, class LayoutCopy_TV, class ShapeTiler_MN>
CUTE_HOST_DEVICE
auto
make_block_2d_prefetch(TiledCopy<Copy_Atom, LayoutCopy_TV, ShapeTiler_MN> const& tiled_copy)
{
  auto &traits = detail::as_block_2d_traits(tiled_copy);
  return make_block_2d_prefetch<typename Copy_Atom::ValType, SGCount>(
    ShapeTiler_MN{}, traits.get_x_mode(), traits.get_y_mode(), traits.tiled_strides
  );
  // TODO: copy base_ptr/width/height/pitch to prefetch atom
}

template <int SGCount, class Shape, class Engine, class Layout>
CUTE_HOST_DEVICE
auto
make_block_2d_prefetch(const Shape& shape, Tensor<Engine, Layout> const& gmem)
{
  using ValType = typename Engine::value_type;
  return make_block_2d_prefetch<ValType, SGCount>(shape, gmem.stride()).with(gmem);
}

template <typename ValType, int SGCount, class Shape, class... Strides>
CUTE_HOST_DEVICE
auto
make_block_2d_prefetch(const Shape& shape, Stride<Strides...> const& stride)
{
  return make_block_2d_prefetch<ValType, SGCount>(shape, find_x_mode(stride), find_y_mode(stride), stride);
}

template <typename ValType, int SGCount, class XMode, class YMode, class Shape, class... Strides>
CUTE_HOST_DEVICE
auto
make_block_2d_prefetch(const Shape&, const XMode& x_mode, const YMode& y_mode, Stride<Strides...> const& stride)
{
  constexpr auto shape_x = get<XMode::value>(Shape{});
  constexpr auto shape_y = get<YMode::value>(Shape{});

  // Try to retrieve whole cache lines (contiguous dimension = x)
  constexpr auto width = cute::min(shape_x, 512 / sizeof_bits_v<ValType>);

  // Do a preliminary tiling to choose appropriate height.
  constexpr int n_sg_x = cute::gcd(SGCount, ceil_div(shape_x, width));
  constexpr int n_sg_y = SGCount / n_sg_x;

  constexpr auto max_height = 32;
  constexpr auto height = cute::min(max_height, ceil_div(shape_y, n_sg_y));

  // Select op.
  using CopyType = int_byte_t<sizeof(ValType)>;
  using CopyOp = XE_PREFETCH_2D<sizeof_bits_v<CopyType>,
                                height,
                                ceil_div(width * sizeof_bits_v<ValType>, sizeof_bits_v<CopyType>)>;

  return make_block_2d_prefetch<ValType, SGCount>(CopyOp{}, Shape{}, x_mode, y_mode, stride);
}

// Low-level prefetch creation utility.
template <typename ValType, int SGCount, class XMode, class YMode, class PrefetchOp, class Shape, class... Strides>
CUTE_HOST_DEVICE
auto
make_block_2d_prefetch(const PrefetchOp& op, const Shape& shape, const XMode& x_mode, const YMode& y_mode, Stride<Strides...> const& stride)
{
  constexpr auto all_1s = tuple_repeat<rank(Shape{})>(_1{});
  constexpr auto width = PrefetchOp::AtomWidth * PrefetchOp::CopyBits / sizeof_bits_v<ValType>;
  constexpr auto height = PrefetchOp::AtomHeight;

  auto op_tile = replace<XMode::value>(replace<YMode::value>(all_1s, Int<height>{}), Int<width>{});

  // Reduce shape to grid of atoms.
  auto atom_shape = shape_div(shape, op_tile);

  // Replicate op tile across subgroups, traversing the innermost dimension first.
  // Ensure the resulting collective tile goes evenly into the given shape (may not be a power of 2)
  constexpr int n_sg_x = cute::gcd(SGCount, get<XMode::value>(atom_shape));
  constexpr int n_sg_y = SGCount / n_sg_x;

  auto collective_op_tile = replace<XMode::value>(replace<YMode::value>(all_1s,
                                                                        Int<n_sg_y>{}),
                                                  Int<n_sg_x>{});

  // Tile atom grid across collective op tile.
  auto sv_layout = logical_divide(make_layout(collective_op_tile), atom_shape);

//  if (cute::thread0()) {
//     #define XSHOW(x)               if (cute::thread0()) { print(#x ": "); print(x); print("\n"); }
//
//     XSHOW(op_tile)
//     printf("n_sg=(%d,%d)\n", n_sg_x, n_sg_y);
//     XSHOW(collective_op_tile)
//     XSHOW(atom_shape)
//     XSHOW(sv_layout)
//  }

  // Create the TiledCopy object.
  return make_block_2d_copy<ValType>(op, x_mode, y_mode, stride, atom_shape, sv_layout);
}

} // end namespace cute
