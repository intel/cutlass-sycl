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
// currently no XE_2D_LD_V builtin supports non-default InstSizeBits
template<int TSizeBits, int Height, int Width, int InstSizeBits = TSizeBits>
struct XE_2D_LD_V {
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
    detail::XeSubgroup2DBlockLoadTransform<InstSizeBytes, BlockWidth, Height, NBlocks>{}(baseoffset, width, height, pitch, coord, dst);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-Xe hardware");
#endif
  }
};

template<int TSizeBits, int Height, int Width, int InstSizeBits = TSizeBits>
CUTE_HOST_DEVICE void print(cute::XE_2D_LD_V<TSizeBits, Height, Width, InstSizeBits> const&){
  print("XE_2D_LD_V<"); print(TSizeBits); print(", "); print(Height); print(", "); print(Width); print(", "); print(InstSizeBits); print(">");
}

// deprecated aliases
using XE_2D_U8x32x16_LD_V = XE_2D_LD_V<8,32,16>;
using XE_2D_U8x32x32_LD_V = XE_2D_LD_V<8,32,32>;
using XE_2D_U8x32x64_LD_V = XE_2D_LD_V<8,32,64>;

using XE_2D_U16x16x16_LD_V = XE_2D_LD_V<16,16,16>;
using XE_2D_U16x32x16_LD_V = XE_2D_LD_V<16,32,16>;
using XE_2D_U16x32x32_LD_V = XE_2D_LD_V<16,32,32>;
using XE_2D_U16x16x32_LD_V = XE_2D_LD_V<16,16,32>;

// performance workaround
using XE_2D_U8x32x32_LD_N = XE_2D_LD_V<8,32,32>;

} // end namespace cute
