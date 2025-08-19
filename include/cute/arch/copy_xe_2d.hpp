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

#include "cute/numeric/int.hpp"

#if defined(__SYCL_DEVICE_ONLY__) && defined(SYCL_INTEL_TARGET)
#define CUTE_ARCH_COPY_XE_ENABLED
#endif

namespace cute {

// Xe 2D copy atoms.
//       Bits: bits per element in underlying memory operation.
//     Height: number of elements in the gmem-strided matrix dimension
//      Width: number of elements in the gmem-contiguous matrix dimension
// BlockWidth: blocking factor (in registers) for the width dimension.
//
// For a row-major-in-memory matrix:
//    height = #rows, width = #columns.
//
// For a column-major-in-memory matrix:
//    height = #columns, width = #rows.

/* Base class for 2D copy ops, to support common queries */
template <int Bits, int Height, int Width, int Count = 1, bool Transpose = false>
struct XE_Copy_Op_2D_Base
{
  static constexpr int CopyBits = Bits;
  static constexpr int AtomWidth = Width;
  static constexpr int AtomHeight = Height;
  static constexpr int BlockCount = Count;
  static constexpr bool Transposing = Transpose;
};


template <int Bits, int Height, int Width, int BlockWidth = Width>
struct XE_LOAD_2D : XE_Copy_Op_2D_Base<Bits, Height, Width, Width/BlockWidth>
{
  template <typename T>
  CUTE_HOST_DEVICE static void copy(const int *payload, T *dst) {
#ifdef CUTE_ARCH_COPY_XE_ENABLED
    auto &dv = *reinterpret_cast<intel::storage_vector_t<T, Width * Height * Bits / 16>*>(dst);
    asm (
      "lsc_load_block2d.ugm (M1, 1)  %0:d%2.%3x%4x%5nn flat[%1+(0,0)]"
        : "=rw"(dv)
        : "rw.u"(payload), "P"(Bits), "P"(Width/BlockWidth), "P"(BlockWidth), "P"(Height)
    );
#else
    CUTE_INVALID_CONTROL_PATH("Cannot use Xe block 2D copy atom on non-Xe hardware");
#endif
  }
};

template <int Bits, int Height, int Width, int BlockWidth = Width>
struct XE_LOAD_2D_VNNI : XE_Copy_Op_2D_Base<Bits, Height, Width, Width/BlockWidth>
{
  template <typename T>
  CUTE_HOST_DEVICE static void copy(const int *payload, T *dst) {
#ifdef CUTE_ARCH_COPY_XE_ENABLED
    auto &dv = *reinterpret_cast<intel::storage_vector_t<T, Width * Height * Bits / 16>*>(dst);
    asm (
      "lsc_load_block2d.ugm (M1, 1)  %0:d%2.%3x%4x%5nt flat[%1+(0,0)]"
        : "=rw"(dv)
        : "rw.u"(payload), "P"(Bits), "P"(Width/BlockWidth), "P"(BlockWidth), "P"(Height)
    );
#else
    CUTE_INVALID_CONTROL_PATH("Cannot use Xe block 2D copy atom on non-Xe hardware");
#endif
  }
};

template <int Bits, int Height, int Width>
struct XE_LOAD_2D_TRANSPOSE : XE_Copy_Op_2D_Base<Bits, Height, Width, 1, true>
{
  template <typename T>
  CUTE_HOST_DEVICE static void copy(const int *payload, T *dst) {
#ifdef CUTE_ARCH_COPY_XE_ENABLED
    auto &dv = *reinterpret_cast<intel::storage_vector_t<T, Width * Height * Bits / 16>*>(dst);
    asm (
      "lsc_load_block2d.ugm (M1, 1)  %0:d%2.%3x%4tn flat[%1+(0,0)]"
        : "=rw"(dv)
        : "rw.u"(payload), "P"(Bits), "P"(Width), "P"(Height)
    );
#else
    CUTE_INVALID_CONTROL_PATH("Cannot use Xe block 2D copy atom on non-Xe hardware");
#endif
  }
};

template <int Bits, int Height, int Width>
struct XE_PREFETCH_2D : XE_Copy_Op_2D_Base<Bits, Height, Width>
{
  template <typename T>
  CUTE_HOST_DEVICE static void copy(const int *payload, T *dst) {
#ifdef CUTE_ARCH_COPY_XE_ENABLED
    asm (
      "lsc_load_block2d.ugm (M1, 1)  %null:d%1.%2x%3nn flat[%0+(0,0)]"
        :: "rw.u"(payload), "P"(Bits), "P"(Width), "P"(Height)
    );
#else
    CUTE_INVALID_CONTROL_PATH("Cannot use Xe block 2D copy atom on non-Xe hardware");
#endif
  }
};

template <int Bits, int Height, int Width>
struct XE_STORE_2D : XE_Copy_Op_2D_Base<Bits, Height, Width>
{
  template <typename T>
  CUTE_HOST_DEVICE static void copy(const int *payload, const T *src) {
#ifdef CUTE_ARCH_COPY_XE_ENABLED
    auto &sv = *reinterpret_cast<const intel::storage_vector_t<T, Width * Height * Bits / 16>*>(src); \
    asm (
      "lsc_store_block2d.ugm (M1, 1) flat[%1+(0,0)] %0:d%2.%3x%4nn"
        :: "rw"(sv), "rw.u"(payload), "P"(Bits), "P"(Width), "P"(Height)
    );
#else
    CUTE_INVALID_CONTROL_PATH("Cannot use Xe block 2D copy atom on non-Xe hardware");
#endif
  }
};

} // end namespace cute
