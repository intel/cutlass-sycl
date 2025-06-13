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
template<int TSizeBits, int Height, int Width>
struct XE_2D_LD_T {
  static_assert(TSizeBits == 4 || TSizeBits == 8 || TSizeBits == 16 || TSizeBits == 32 || TSizeBits == 64, 
      "Expected TSizeBits to be a power of 2, less then or equal 64");
  static_assert(Width == 1 || Width == 2 || Width == 4 || Width == 8 || Width == 16 || Width == 32, 
      "Expected Width to be a power of 2, less then or equal 32");

  //static_assert(InstSizeBits % 8 == 0, "Expected InstSizeBits to be a multiple of 8.");
  //static constexpr int InstSizeBytes = InstSizeBits / 8;
  //static_assert(InstSizeBits % TSizeBits == 0, "Expected InstSizeBits to be a multiple of TSizeBits.");

  //static constexpr int TSizeBytes = TSizeBits / 8;
  static constexpr int InstSizeBits = 32;
  static constexpr int InstSizeBytes = InstSizeBits / 8;
  static constexpr int VecSize = InstSizeBits / TSizeBits;
  static constexpr int BlockHeight = 16;
  static constexpr int InstBlockWidth = Width / VecSize;
  //static constexpr int BlockHeight = InstBlockHeight * VecSize;
  //static_assert(Height % BlockHeight == 0, "Expected Height to be a multiple of 16 * InstSizeBits / TSizeBits.");
  static constexpr int NBlocks = Height / BlockHeight;


  /*static constexpr int VecSize = 1;
  static constexpr int BlockHeight = 16 * VecSize; //TODO SG size?
  static_assert(Height % BlockHeight == 0, "Expected Height to be a multiple of 16.");
  static constexpr int NBlocks = Height / BlockHeight;
  // currently no XE_2D_LD_T builtin supports non-32 InstSizeBits
  static constexpr int InstSizeBits = 32;
  static constexpr int InstSizeBytes = InstSizeBits / 8;
  static constexpr int InstWidth = Width * TSizeBits / InstSizeBits;*/

  // shape of the block in global memory 
  using BlockShape = Shape<Int<Height>, Int<Width>>;
  using inst_dtype = uint32_t;
  static constexpr bool is_transpose = true;
  
  template<typename T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
    /*if(thread0()){
        print("XE_2D_LD_T<"); print(TSizeBits); print(", "); print(Height); print(", "); print(Width); print(">\n");
        print("Calling XeSubgroup2DBlockLoadTranspose<"); print(InstSizeBytes); print(", "); print(InstBlockWidth); print(", "); print(BlockHeight); print(", "); print(NBlocks); print(">("); 
        print(baseoffset); print(", "); print(width); print(", "); print(height); print(", "); print(pitch); print(", ("); print(coord[0]); print(", "); print(coord[1]); print("), "); print(dst); print("\n"); 
    }*/
#if defined(CUTE_ARCH_COPY_XE_ENABLED)
    static_assert(sizeof_bits_v<T> == TSizeBits, "Expected T to have size equal to TSizeBits.");
    //detail::XeSubgroup2DBlockLoadTranspose<InstSizeBytes, InstWidth, BlockHeight, NBlocks>{}(baseoffset, width, height, pitch, coord, dst);
    detail::XeSubgroup2DBlockLoadTranspose<InstSizeBytes, InstBlockWidth, BlockHeight, NBlocks>{}(baseoffset, width, height, pitch, coord, dst);
    //detail::XeSubgroup2DBlockLoadTranspose<4, 8, 16, 1>{}(baseoffset, width, height, pitch, coord, dst);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-Xe hardware");
#endif
  }
};

template<int TSizeBits, int Height, int Width>
CUTE_HOST_DEVICE void print(cute::XE_2D_LD_T<TSizeBits, Height, Width> const&){
  print("XE_2D_LD_T<"); print(TSizeBits); print(", "); print(Height); print(", "); print(Width); print(">\n");
  print("Call XeSubgroup2DBlockLoadTranspose<"); print(cute::XE_2D_LD_T<TSizeBits, Height, Width>::TSizeBytes); print(", "); 
                                                 print(Width); print(", "); 
                                                 print(cute::XE_2D_LD_T<TSizeBits, Height, Width>::BlockHeight); print(", "); 
                                                 print(cute::XE_2D_LD_T<TSizeBits, Height, Width>::NBlocks); print(">\n");
}

// deprecated aliases

//using XE_2D_U64x8x1_LD_T = XE_2D_LD_T<64,8,1>;
//using XE_2D_U64x8x2_LD_T = XE_2D_LD_T<64,8,2>;
//using XE_2D_U64x8x4_LD_T = XE_2D_LD_T<64,8,4>;

using XE_2D_U8x16x32_LD_T = XE_2D_LD_T<8,16,32>;
using XE_2D_U8x16x16_LD_T = XE_2D_LD_T<8,16,16>;
using XE_2D_U8x16x8_LD_T = XE_2D_LD_T<8,16,8>;

using XE_2D_U16x16x8_LD_T = XE_2D_LD_T<16,16,8>;

using XE_2D_U32x16x2_LD_T = XE_2D_LD_T<32,16,2>;
using XE_2D_U32x16x4_LD_T = XE_2D_LD_T<32,16,4>;
using XE_2D_U32x16x8_LD_T = XE_2D_LD_T<32,16,8>;

using XE_2D_U16x16x16_LD_T = XE_2D_LD_T<16,16,16>;

struct XE_2D_U16x16x16_LD_T_ {
  using BlockShape = Shape<_16, _16>;
  using inst_dtype = uint32_t;

  static constexpr bool is_transpose = true;

  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
    if(thread0()){
        print("XE_2D_U16x16x16_LD_T_\n");
        print("Calling XeSubgroup2DBlockLoadTranspose<4, 8, 16, 1>("); 
        print(baseoffset); print(", "); print(width); print(", "); print(height); print(", "); print(pitch); print(", ("); print(coord[0]); print(", "); print(coord[1]); print("), "); print(dst); print("\n"); 
    }
#if defined(CUTE_ARCH_COPY_XE_ENABLED)
    static_assert(sizeof(T) == 2, "Expected T to have size 2");
    detail::XeSubgroup2DBlockLoadTranspose<4, 8, 16, 1>{}(baseoffset, width, height, pitch, coord, dst);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-Xe hardware");
#endif
  }
};

CUTE_HOST_DEVICE void print(XE_2D_U16x16x16_LD_T_ const&){
  print("XE_2D_U16x16x16_LD_T_\n");
  print("Call XeSubgroup2DBlockLoadTranspose<4, 8, 16, 1>\n");
}

using XE_2D_U4x32x16_LD_T = XE_2D_LD_T<4,32,16>;
using XE_2D_U4x16x16_LD_T = XE_2D_LD_T<4,16,16>;

} // end namespace cute
