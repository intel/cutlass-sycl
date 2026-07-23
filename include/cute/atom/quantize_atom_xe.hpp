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

#pragma once

#include <cute/util/sycl_vec.hpp>
#include <cute/util/type_traits.hpp>

namespace cute
{

// Check for the existence of a quantize atom (multiply + convert with precomputed scales).
template <typename SrcType, typename DstType>
constexpr bool has_xe_quantize_optimized_impl(char) { return true; }

template <typename SrcType, typename DstType,
          typename V = typename Xe_Quantize_Optimized<SrcType, DstType>::Unimplemented>
constexpr bool has_xe_quantize_optimized_impl(int) { return false; }

template <typename SrcType, typename DstType>
constexpr bool has_xe_quantize_optimized() {
  return has_xe_quantize_optimized_impl<SrcType, DstType>(0);
}

// Dispatch: choose optimized (C++ + ASM atoms) or pure-C++ fallback.
//
// Optimized path: C++ computes per-block abs-max, cross-lane reduction, and
// scale factors.  Then Xe_Quantize_Optimized ASM atoms handle the multiply +
// type-convert step (mul + mov F32→HF + fcvt HF→FP8).
//
// Fallback path: pure C++ — no inline ASM.  Uses NumericConverter for the
// F32→FP8 conversion.
//
// @tparam SrcType    Source element type (half_t, bfloat16_t, float)
// @tparam DstType    Destination element type (float_e4m3_t, float_e5m2_t)
// @tparam NumValues  Number of per-thread values in the source tensor
template <typename SrcType, typename DstType, int NumValues>
auto choose_xe_quantize_impl()
{
#if defined(SYCL_INTEL_TARGET) && (SYCL_INTEL_TARGET >= 35)
  // Optimized path: C++ abs-max/reduce/scale + ASM mul+fcvt atoms.
  // Requires sg_size == 16, NumValues divisible by 4.
  if constexpr (has_xe_quantize_optimized<SrcType, DstType>() &&
                NumValues % 4 == 0 &&
                intel::sg_size == 16)
    return QuantizeDispatchOptimized{};
  else
#endif
  // Fallback path: pure C++ (no inline ASM)
  return QuantizeDispatchFallback{};
}

} // end namespace cute
