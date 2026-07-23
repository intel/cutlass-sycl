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

#include <cute/config.hpp>            // CUTE_HOST_DEVICE
#include <cute/tensor_impl.hpp>

#if defined(__SYCL_DEVICE_ONLY__) && defined(SYCL_INTEL_TARGET)
#define CUTE_ARCH_DEQUANTIZE_XE_ENABLED
#endif

namespace cute
{

template <typename IntermediateType>
struct Xe_Dequantize_Fallback {};

template <class IntermediateType,
          class ElemEngine, class ElemLayoutWI,
          class ScaleType>
CUTE_HOST_DEVICE
void
dequantize_impl(Xe_Dequantize_Fallback<IntermediateType> const&,
                Tensor<ElemEngine,ElemLayoutWI>               & tensor,
                ScaleType                                       scale)
{
  using ElemType = typename ElemEngine::element_type;
  constexpr auto NumElemsPerThread = size(ElemLayoutWI{});

  CUTE_UNROLL
  for (int i = 0; i < NumElemsPerThread; i++) {
    tensor(i) = static_cast<ElemType>(static_cast<IntermediateType>(scale) * static_cast<IntermediateType>(tensor(i)));
  }
}

template <typename ElemType, typename IntermediateType>
struct Xe_Dequantize {
  using Unimplemented = void;
};

template <>
struct Xe_Dequantize<cutlass::half_t, float>
{
  using Register = intel::ushort4;

  template <typename ScaleType>
  CUTE_HOST_DEVICE static void
  dequantize(Register* p, ScaleType scale)
  {
#if defined(CUTE_ARCH_DEQUANTIZE_XE_ENABLED)
#if SYCL_INTEL_TARGET >= 35
    auto& data = *p;
    auto s = static_cast<float>(scale);
    asm (
      "{\n"
      ".decl IO_HF v_type=G type=HF num_elts=64 alias=<%0,0>\n"
      ".decl SCALE_F v_type=G type=F num_elts=1 alias=<%1,0>\n"
      ".decl TMP_F v_type=G type=F num_elts=64 align=32\n"
      "mov (M1_NM, 32) TMP_F(0,0)<1> IO_HF(0,0)<1;1,0>\n"
      "mov (M1_NM, 32) TMP_F(0,32)<1> IO_HF(0,32)<1;1,0>\n"
      "mul (M1_NM, 32) TMP_F(0,0)<1> TMP_F(0,0)<1;1,0> SCALE_F(0,0)<0;1,0>\n"
      "mul (M1_NM, 32) TMP_F(0,32)<1> TMP_F(0,32)<1;1,0> SCALE_F(0,0)<0;1,0>\n"
      "mov (M1_NM, 32) IO_HF(0,0)<1> TMP_F(0,0)<1;1,0>\n"
      "mov (M1_NM, 32) IO_HF(0,32)<1> TMP_F(0,32)<1;1,0>\n"
      "}\n"
      : "+rw"(data)
      : "rw.u"(s)
    );
#else
    CUTE_INVALID_CONTROL_PATH("Xe Version < 35 not implemented");
#endif
#else
    CUTE_INVALID_CONTROL_PATH("Not Xe");
#endif
  }
};

template <>
struct Xe_Dequantize<cutlass::bfloat16_t, float>
{
  using Register = intel::ushort4;

  template <typename ScaleType>
  CUTE_HOST_DEVICE static void
  dequantize(Register* p, ScaleType scale)
  {
#if defined(CUTE_ARCH_DEQUANTIZE_XE_ENABLED)
#if SYCL_INTEL_TARGET >= 35
    auto& data = *p;
    auto s = static_cast<float>(scale);
    asm (
      "{\n"
      ".decl IO_BF v_type=G type=BF num_elts=64 alias=<%0,0>\n"
      ".decl SCALE_F v_type=G type=F num_elts=1 alias=<%1,0>\n"
      ".decl TMP_F v_type=G type=F num_elts=64 align=32\n"
      "mov (M1_NM, 32) TMP_F(0,0)<1> IO_BF(0,0)<1;1,0>\n"
      "mov (M1_NM, 32) TMP_F(0,32)<1> IO_BF(0,32)<1;1,0>\n"
      "mul (M1_NM, 32) TMP_F(0,0)<1> TMP_F(0,0)<1;1,0> SCALE_F(0,0)<0;1,0>\n"
      "mul (M1_NM, 32) TMP_F(0,32)<1> TMP_F(0,32)<1;1,0> SCALE_F(0,0)<0;1,0>\n"
      "mov (M1_NM, 32) IO_BF(0,0)<1> TMP_F(0,0)<1;1,0>\n"
      "mov (M1_NM, 32) IO_BF(0,32)<1> TMP_F(0,32)<1;1,0>\n"
      "}\n"
      : "+rw"(data)
      : "rw.u"(s)
    );
#else
    CUTE_INVALID_CONTROL_PATH("Xe Version < 35 not implemented");
#endif
#else
    CUTE_INVALID_CONTROL_PATH("Not Xe");
#endif
  }
};

template <class Xe_Dequantize,
          class ElemEngine, class ElemLayoutWI,
          class ScaleType>
CUTE_HOST_DEVICE
void
dequantize_impl(Xe_Dequantize              const&,
                Tensor<ElemEngine,ElemLayoutWI> & tensor,
                ScaleType                         scale)
{
  constexpr auto NumElemsPerThread = size(ElemLayoutWI{});
  static_assert(NumElemsPerThread % 4 == 0, "This implementation requires the number of elements per thread to be a multiple of 4");

  CUTE_UNROLL
  for (int i = 0; i < NumElemsPerThread; i = i + 4) {
    Xe_Dequantize::dequantize(recast_ptr<typename Xe_Dequantize::Register>(&tensor(i)), scale);
  }
}

// Check for the existence of an optimized dequantize sequence.
template <typename ElemType, typename IntermediateType>
constexpr bool has_xe_optimized_dequantize_impl(char) { return true; }

template <typename ElemType, typename IntermediateType,
          typename V = typename Xe_Dequantize<ElemType, IntermediateType>::Unimplemented>
constexpr bool has_xe_optimized_dequantize_impl(int) { return false; }

template <typename ElemType, typename IntermediateType>
constexpr bool has_xe_optimized_dequantize() {
  return has_xe_optimized_dequantize_impl<ElemType, IntermediateType>(0);
}

template <typename IntermediateType = float,
          typename LayoutWI,
          typename ElemType>
auto choose_xe_dequantize_impl()
{
  constexpr auto NumElemsPerThread = size(LayoutWI{});
  // Use the optimized vectorized path when each thread handles a multiple of 4
  // elements (the current FP8 KV-cache scenario). Additional optimized paths can
  // be added for other layouts/workloads as needed.
  if constexpr (NumElemsPerThread % 4 != 0 || intel::sg_size != 16) {
    return Xe_Dequantize_Fallback<IntermediateType>{};
  }

#if defined(SYCL_INTEL_TARGET) && (SYCL_INTEL_TARGET >= 35)
  if constexpr (has_xe_optimized_dequantize<ElemType, IntermediateType>())
    return Xe_Dequantize<ElemType, IntermediateType>{};
  else
#endif
    return Xe_Dequantize_Fallback<IntermediateType>{};
}

template <class IntermediateType = float,
          class Engine, class Layout,
          class ScaleType>
CUTE_HOST_DEVICE
void
dequantize(Tensor<Engine, Layout>& tensor,
           ScaleType               scale)
{
    using ElemType = typename Engine::element_type;

    auto impl = choose_xe_dequantize_impl<IntermediateType, Layout, ElemType>();

    dequantize_impl(impl, tensor, scale);
}

} // end namespace cute
