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
#pragma once

#include <cute/util/sycl_vec.hpp>
#include "cute/config.hpp"
#include "cute/numeric/numeric_types.hpp"

namespace cute
{

template<int TSizeBits, int Height, int Width, int InstSizeBits = TSizeBits>
struct XE_2D_LD_N_PREFETCH {
  static_assert(TSizeBits == 4 || TSizeBits == 8 || TSizeBits == 16 || TSizeBits == 32 || TSizeBits == 64, 
      "Expected TSizeBits to be a power of 2, less then or equal 64");
  static_assert(Height == 1 || Height == 2 || Height == 4 || Height == 8 || Height == 16 || Height == 32, 
      "Expected Height to be a power of 2, less then or equal 32");

  static_assert(InstSizeBits % 8 == 0, "Expected InstSizeBits to be a multiple of 8.");
  static constexpr int InstSizeBytes = InstSizeBits / 8;
  static_assert(InstSizeBits % TSizeBits == 0, "Expected InstSizeBits to be a multiple of TSizeBits.");
  static constexpr int VecSize = InstSizeBits / TSizeBits;
  static constexpr int BlockWidth = 16 * VecSize;
  static_assert(Width % BlockWidth == 0, "Expected Width to be a multiple of 16 * InstSizeBits / TSizeBits.");
  static constexpr int NBlocks = Width / BlockWidth;

  // shape of the block in global memory 
  using BlockShape = Shape<Int<Height>, Int<Width>>;
  
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch,
                                    intel::coord_t coord) {
#if defined(CUTE_ARCH_COPY_XE_ENABLED)
  detail::XeSubgroup2DBlockPrefetch<InstSizeBytes, BlockWidth, Height, NBlocks>{}(baseoffset, width, height, pitch, coord);
#else
    CUTE_INVALID_CONTROL_PATH(
        "Trying to use block prefetch on non-Xe hardware");
#endif
  }
};

template<int TSizeBits, int Height, int Width, int InstSizeBits = TSizeBits>
struct XE_2D_LD_N {
  static_assert(TSizeBits == 4 || TSizeBits == 8 || TSizeBits == 16 || TSizeBits == 32 || TSizeBits == 64, 
      "Expected TSizeBits to be a power of 2, less then or equal 64");
  static_assert(Height == 1 || Height == 2 || Height == 4 || Height == 8 || Height == 16 || Height == 32, 
      "Expected Height to be a power of 2, less then or equal 32");

  static_assert(InstSizeBits % 8 == 0, "Expected InstSizeBits to be a multiple of 8.");
  static constexpr int InstSizeBytes = InstSizeBits / 8;
  static_assert(InstSizeBits % TSizeBits == 0, "Expected InstSizeBits to be a multiple of TSizeBits.");
  static constexpr int VecSize = InstSizeBits / TSizeBits;
  static constexpr int BlockWidth = 16 * VecSize;
  static_assert(Width % BlockWidth == 0, "Expected Width to be a multiple of 16 * InstSizeBits / TSizeBits.");
  static constexpr int NBlocks = Width / BlockWidth;

  // shape of the block in global memory 
  using BlockShape = Shape<Int<Height>, Int<Width>>;

  template<typename T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(CUTE_ARCH_COPY_XE_ENABLED)
    static_assert(sizeof_bits_v<T> == TSizeBits, "Expected T to have size equal to TSizeBits.");
    detail::XeSubgroup2DBlockLoad<InstSizeBytes, BlockWidth, Height, NBlocks>{}(baseoffset, width, height, pitch, coord, dst);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-Xe hardware");
#endif
  }

  using PREFETCH = XE_2D_LD_N_PREFETCH<TSizeBits, Height, Width, InstSizeBits>;

};

template<int Height, int Width>
struct XE_2D_LD_N<4, Height, Width, 4> {
  static_assert(Height == 1 || Height == 2 || Height == 4 || Height == 8 || Height == 16 || Height == 32, 
      "Expected Height to be a power of 2, less then or equal 32");
  static constexpr int VecSize = 1;
  static constexpr int BlockWidth = 16;
  static constexpr int InstWidth = Width / 2;
  //TODO(Codeplay): Do we need to add support for multiple blocks?
  static_assert(Width == 64, "Expected Width to be 64. Other cases are currently not supported.");
  // shape of the block in global memory 
  using BlockShape = Shape<Int<Height>, Int<Width>>;
  using inst_dtype = int8_t;

  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(CUTE_ARCH_COPY_XE_ENABLED)
    static_assert(sizeof(T) == 1, "Expected T to have size 1");
    detail::XeSubgroup2DBlockLoad<1, InstWidth, Height, 1>{}(baseoffset, width, height, pitch, coord, dst);

   // ================= shuffle begin =================
   // FIXME: the performance of shuffle algorithm here is too bad, we are working with
   // compiler/IGC team to optimize it.

    static constexpr auto subgroup_size = 16;
    static constexpr auto copy_W = Width / subgroup_size;
    static constexpr auto copy_H = Height;

    auto sg = syclcompat::get_nd_item<1>().get_sub_group();
    auto id = int(ThreadIdxX()) % subgroup_size;

    cute::subbyte_iterator<int4_t> dst_iter(dst);
    cute::array_subbyte<int4_t, copy_W * copy_H> dst_tmp{};

    #pragma unroll
    for (int cw = 0; cw < copy_W; cw++) {
      auto remote_id = (id + cw * subgroup_size) / copy_W;

      // TODO: select 'ushort32' will cause compiling error, use 'ushort16' instead, why?
      intel::ushort16 remote_dst[2];
      remote_dst[0] = sycl::select_from_group(sg, *(reinterpret_cast<intel::ushort16 *>(dst)), remote_id);
      remote_dst[1] = sycl::select_from_group(sg, *((reinterpret_cast<intel::ushort16 *>(dst)) + 1), remote_id);

      cute::subbyte_iterator<int4_t> remote_dst_iter(remote_dst);

      #pragma unroll
      for (int row = 0; row < copy_H; row++) {
        dst_tmp[row + cw * copy_H] = remote_dst_iter[row * copy_W + id % copy_W].get();
      }
    }

   *reinterpret_cast<intel::ushort32 *>(cute::raw_pointer_cast(dst_iter)) = *reinterpret_cast<intel::ushort32 *>(cute::raw_pointer_cast(dst_tmp.begin()));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-Xe hardware");
#endif
  }
};

/*struct XE_2D_U4x16x64_LD_N {
  using BlockShape = Shape<_16, _64>;
  using inst_dtype = int8_t;

  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(CUTE_ARCH_COPY_XE_ENABLED)
    static_assert(sizeof(T) == 1, "Expected T to have size 1");
    detail::XeSubgroup2DBlockLoad<1, 32, 16, 1>{}(baseoffset, width, height, pitch, coord, dst);

   // ================= shuffle begin =================
   // FIXME: the performance of shuffle algorithm here is too bad, we are working with
   // compiler/IGC team to optimize it.

    static constexpr auto subgroup_size = 16;
    static constexpr auto copy_W = decltype(size<1>(BlockShape{}))::value / subgroup_size;
    static constexpr auto copy_H = decltype(size<0>(BlockShape{}))::value;

    auto sg = syclcompat::get_nd_item<1>().get_sub_group();
    auto id = int(ThreadIdxX()) % subgroup_size;

    cute::subbyte_iterator<int4_t> dst_iter(dst);
    cute::array_subbyte<int4_t, copy_W * copy_H> dst_tmp{};

    #pragma unroll
    for (int cw = 0; cw < copy_W; cw++) {
      auto remote_id = (id + cw * subgroup_size) / copy_W;

      intel::ushort16 remote_dst;
      remote_dst = sycl::select_from_group(sg, *(reinterpret_cast<intel::ushort16 *>(dst)), remote_id);

      cute::subbyte_iterator<int4_t> remote_dst_iter(&remote_dst);


      #pragma unroll
      for (int row = 0; row < copy_H; row++) {
        dst_tmp[row + cw * copy_H] = remote_dst_iter[row * copy_W + id % copy_W].get();
      }
    }

   *reinterpret_cast<intel::ushort16 *>(cute::raw_pointer_cast(dst_iter)) = *reinterpret_cast<intel::ushort16 *>(cute::raw_pointer_cast(dst_tmp.begin()));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-Xe hardware");
#endif
  }
};*/

template<int TSizeBits, int Height, int Width, int InstSizeBits = TSizeBits>
CUTE_HOST_DEVICE void print(cute::XE_2D_LD_N<TSizeBits, Height, Width, InstSizeBits> const&){
  print("XE_2D_LD_N<"); print(TSizeBits); print(", "); print(Height); print(", "); print(Width); print(", "); print(InstSizeBits); print(">");
}

// deprecated aliases
using XE_2D_Packed_U8x1x32_LD_N = XE_2D_LD_N<8,1,32>;
using XE_2D_Packed_U8x2x32_LD_N = XE_2D_LD_N<8,2,32>;
using XE_2D_Packed_U8x4x32_LD_N = XE_2D_LD_N<8,4,32>;
using XE_2D_Packed_U8x8x32_LD_N = XE_2D_LD_N<8,8,32>;

using XE_2D_Packed_U8x1x64_LD_N = XE_2D_LD_N<8,1,64>;
using XE_2D_Packed_U8x2x64_LD_N = XE_2D_LD_N<8,2,64>;
using XE_2D_Packed_U8x4x64_LD_N = XE_2D_LD_N<8,4,64>;
using XE_2D_Packed_U8x8x64_LD_N = XE_2D_LD_N<8,8,64>;

using XE_2D_Packed_U8x16x32_LD_N = XE_2D_LD_N<8,16,32>;
using XE_2D_Packed_U8x32x32_LD_N = XE_2D_LD_N<8,32,32>;

using XE_2D_Packed_U8x16x64_LD_N = XE_2D_LD_N<8,16,64>;
using XE_2D_Packed_U8x32x64_LD_N = XE_2D_LD_N<8,32,64>;

using XE_2D_U16x1x16_LD_N = XE_2D_LD_N<16,1,16>;
using XE_2D_U16x2x16_LD_N = XE_2D_LD_N<16,2,16>;
using XE_2D_U16x4x16_LD_N = XE_2D_LD_N<16,4,16>;
using XE_2D_U16x8x16_LD_N = XE_2D_LD_N<16,8,16>;

using XE_2D_U16x1x32_LD_N = XE_2D_LD_N<16,1,32>;
using XE_2D_U16x2x32_LD_N = XE_2D_LD_N<16,2,32>;
using XE_2D_U16x4x32_LD_N = XE_2D_LD_N<16,4,32>;
using XE_2D_U16x8x32_LD_N = XE_2D_LD_N<16,8,32>;
using XE_2D_U16x16x32_LD_N = XE_2D_LD_N<16,16,32>;

//using XE_2D_TF32x1x8_LD_N = XE_2D_LD_N<32,1,8>; //TODO  < sg???
//using XE_2D_TF32x2x8_LD_N = XE_2D_LD_N<32,2,8>;
//using XE_2D_TF32x4x8_LD_N = XE_2D_LD_N<32,4,8>;
//using XE_2D_TF32x8x8_LD_N = XE_2D_LD_N<32,8,8>;

using XE_2D_U32x1x16_LD_N = XE_2D_LD_N<32,1,16>;
using XE_2D_U32x2x16_LD_N = XE_2D_LD_N<32,2,16>;
using XE_2D_U32x4x16_LD_N = XE_2D_LD_N<32,4,16>;
using XE_2D_U32x8x16_LD_N = XE_2D_LD_N<32,8,16>;

using XE_2D_U16x16x16_LD_N = XE_2D_LD_N<16,16,16>;
using XE_2D_U16x32x16_LD_N = XE_2D_LD_N<16,32,16>;
using XE_2D_U16x32x32_LD_N = XE_2D_LD_N<16,32,32>;

//using XE_2D_TF32x16x8_LD_N = XE_2D_LD_N<32,16,8>;
//using XE_2D_TF32x32x8_LD_N = XE_2D_LD_N<32,32,8>;

using XE_2D_TF32x1x16_LD_N = XE_2D_LD_N<32,1,16>;
using XE_2D_TF32x2x16_LD_N = XE_2D_LD_N<32,2,16>;
using XE_2D_TF32x4x16_LD_N = XE_2D_LD_N<32,4,16>;
using XE_2D_TF32x8x16_LD_N = XE_2D_LD_N<32,8,16>;

using XE_2D_U32x16x16_LD_N = XE_2D_LD_N<32,16,16>;
using XE_2D_U32x32x16_LD_N = XE_2D_LD_N<32,32,16>;

using XE_2D_TF32x16x16_LD_N = XE_2D_LD_N<32,16,16>;
using XE_2D_TF32x32x16_LD_N = XE_2D_LD_N<32,32,16>;

using XE_2D_U4x32x64_LD_N = XE_2D_LD_N<4,32,64>;
using XE_2D_U4x16x64_LD_N = XE_2D_LD_N<4,16,64>;

} // end namespace cute
