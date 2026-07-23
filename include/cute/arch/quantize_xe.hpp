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

#include <cute/config.hpp>
#include <cute/util/sycl_vec.hpp>

#include <cutlass/numeric_types.h>
#include <cutlass/float8.h>

#if defined(__SYCL_DEVICE_ONLY__) && defined(SYCL_INTEL_TARGET)
#define CUTE_ARCH_QUANTIZE_XE_ENABLED
#endif

namespace cute {

//////////////////////////////////////////////////////////////////////////////
// Xe_Quantize_Optimized — per-chunk scale-multiply + type-convert atoms
//
// Each specialization fuses: scale multiply (in F32) + type conversion
// (F32→HF→FP8 via fcvt) into a single ASM block.
// Operates on 4 SrcType elements → 4 DstType elements per call.
// SrcRegister = intel::ushort4 (for 16-bit src: 4 × uint16 → 4 HF/BF values)
// DstRegister = intel::uchar4  (for 8-bit dst: 4 × uint8  → 4 FP8 values)
// ScaleRegister = sycl::float4 (4 per-value scales, one per element).
//
// Register layout (SIMD-16, 4 values per thread):
//   After HF/BF → F32 widening, 64 floats span 4 GRF rows (16 floats/row):
//     GRF row 0: lane 0..15  value 0
//     GRF row 1: lane 0..15  value 1
//     GRF row 2: lane 0..15  value 2
//     GRF row 3: lane 0..15  value 3
//   Each value is 1 GRF rows use the same scale via <0;1,0> broadcast.
//   4× SIMD-16 mul selects the correct per-value scale from SCALE_F.
//////////////////////////////////////////////////////////////////////////////

template <typename SrcType, typename DstType>
struct Xe_Quantize_Optimized {
  using Unimplemented = void;
};

// half_t → float_e5m2_t: mul(HF→F) + fcvt(HF→UB)
template <>
struct Xe_Quantize_Optimized<cutlass::half_t, cutlass::float_e5m2_t>
{
  using SrcRegister = intel::ushort4;
  using DstRegister = intel::uchar4;
  using ScaleRegister = intel::float4;

  CUTE_HOST_DEVICE static void
  quantize(SrcRegister const* src_p, DstRegister* dst_p, ScaleRegister scales)
  {
#if defined(CUTE_ARCH_QUANTIZE_XE_ENABLED)
#if SYCL_INTEL_TARGET >= 35
    auto const& src = *src_p;
    auto& dst = *dst_p;
    asm (
      "{\n"
      ".decl IN_HF    v_type=G type=HF num_elts=64 alias=<%1,0>\n"
      ".decl OUT_UB    v_type=G type=UB num_elts=64 alias=<%0,0>\n"
      ".decl SCALE_F  v_type=G type=F  num_elts=64 alias=<%2,0>\n"
      ".decl TMP_F     v_type=G type=F  num_elts=64 align=32\n"
      ".decl TMP_HF    v_type=G type=HF num_elts=64 align=32\n"
      // HF → F32 (widen for precision): 32 HF → 32 F32 each
      "mov (M1_NM, 32) TMP_F(0,0)<1>   IN_HF(0,0)<1;1,0>\n"
      "mov (M1_NM, 32) TMP_F(0,32)<1>  IN_HF(0,32)<1;1,0>\n"
      // Multiply by per-value scale in F32 (4× SIMD-16)
      // value 0: elements 0..15,  value 1: elements 16..31
      // value 2: elements 32..47, value 3: elements 48..63
      "mul (M1_NM, 16) TMP_F(0,0)<1>   TMP_F(0,0)<1;1,0>   SCALE_F(0,0)<0;1,0>\n"
      "mul (M1_NM, 16) TMP_F(0,16)<1>  TMP_F(0,16)<1;1,0>  SCALE_F(0,16)<0;1,0>\n"
      "mul (M1_NM, 16) TMP_F(0,32)<1>  TMP_F(0,32)<1;1,0>  SCALE_F(0,32)<0;1,0>\n"
      "mul (M1_NM, 16) TMP_F(0,48)<1>  TMP_F(0,48)<1;1,0>  SCALE_F(0,48)<0;1,0>\n"
      // F32 → HF (narrow back for fcvt input)
      "mov (M1_NM, 32) TMP_HF(0,0)<1>  TMP_F(0,0)<1;1,0>\n"
      "mov (M1_NM, 32) TMP_HF(0,32)<1> TMP_F(0,32)<1;1,0>\n"
      // HF → E5M2 via fcvt
      "fcvt (M1_NM, 32) OUT_UB(0,0)<1>  TMP_HF(0,0)<1;1,0>\n"
      "fcvt (M1_NM, 32) OUT_UB(0,32)<1> TMP_HF(0,32)<1;1,0>\n"
      "}\n"
      : "=rw"(dst)
      : "rw"(src), "rw"(scales)
    );
#else
    CUTE_INVALID_CONTROL_PATH("Xe Version < 35 not implemented");
#endif
#else
    CUTE_INVALID_CONTROL_PATH("Not Xe");
#endif
  }
};

// half_t → float_e4m3_t: mul(HF→F) + fcvt(HF→B)
template <>
struct Xe_Quantize_Optimized<cutlass::half_t, cutlass::float_e4m3_t>
{
  using SrcRegister = intel::ushort4;
  using DstRegister = intel::uchar4;
  using ScaleRegister = intel::float4;

  CUTE_HOST_DEVICE static void
  quantize(SrcRegister const* src_p, DstRegister* dst_p, ScaleRegister scales)
  {
#if defined(CUTE_ARCH_QUANTIZE_XE_ENABLED)
#if SYCL_INTEL_TARGET >= 35
    auto const& src = *src_p;
    auto& dst = *dst_p;
    asm (
      "{\n"
      ".decl IN_HF    v_type=G type=HF num_elts=64 alias=<%1,0>\n"
      ".decl OUT_B     v_type=G type=B  num_elts=64 alias=<%0,0>\n"
      ".decl SCALE_F  v_type=G type=F  num_elts=64 alias=<%2,0>\n"
      ".decl TMP_F     v_type=G type=F  num_elts=64 align=32\n"
      ".decl TMP_HF    v_type=G type=HF num_elts=64 align=32\n"
      // HF → F32
      "mov (M1_NM, 32) TMP_F(0,0)<1>   IN_HF(0,0)<1;1,0>\n"
      "mov (M1_NM, 32) TMP_F(0,32)<1>  IN_HF(0,32)<1;1,0>\n"
      // Multiply by per-value scale in F32 (4× SIMD-16)
      "mul (M1_NM, 16) TMP_F(0,0)<1>   TMP_F(0,0)<1;1,0>   SCALE_F(0,0)<0;1,0>\n"
      "mul (M1_NM, 16) TMP_F(0,16)<1>  TMP_F(0,16)<1;1,0>  SCALE_F(0,16)<0;1,0>\n"
      "mul (M1_NM, 16) TMP_F(0,32)<1>  TMP_F(0,32)<1;1,0>  SCALE_F(0,32)<0;1,0>\n"
      "mul (M1_NM, 16) TMP_F(0,48)<1>  TMP_F(0,48)<1;1,0>  SCALE_F(0,48)<0;1,0>\n"
      // F32 → HF
      "mov (M1_NM, 32) TMP_HF(0,0)<1>  TMP_F(0,0)<1;1,0>\n"
      "mov (M1_NM, 32) TMP_HF(0,32)<1> TMP_F(0,32)<1;1,0>\n"
      // HF → E4M3 via fcvt
      "fcvt (M1_NM, 32) OUT_B(0,0)<1>  TMP_HF(0,0)<1;1,0>\n"
      "fcvt (M1_NM, 32) OUT_B(0,32)<1> TMP_HF(0,32)<1;1,0>\n"
      "}\n"
      : "=rw"(dst)
      : "rw"(src), "rw"(scales)
    );
#else
    CUTE_INVALID_CONTROL_PATH("Xe Version < 35 not implemented");
#endif
#else
    CUTE_INVALID_CONTROL_PATH("Not Xe");
#endif
  }
};

// bfloat16_t → float_e5m2_t: mul(BF→F) + fcvt(F→HF→UB)
template <>
struct Xe_Quantize_Optimized<cutlass::bfloat16_t, cutlass::float_e5m2_t>
{
  using SrcRegister = intel::ushort4;
  using DstRegister = intel::uchar4;
  using ScaleRegister = intel::float4;

  CUTE_HOST_DEVICE static void
  quantize(SrcRegister const* src_p, DstRegister* dst_p, ScaleRegister scales)
  {
#if defined(CUTE_ARCH_QUANTIZE_XE_ENABLED)
#if SYCL_INTEL_TARGET >= 35
    auto const& src = *src_p;
    auto& dst = *dst_p;
    asm (
      "{\n"
      ".decl IN_BF    v_type=G type=BF num_elts=64 alias=<%1,0>\n"
      ".decl OUT_UB    v_type=G type=UB num_elts=64 alias=<%0,0>\n"
      ".decl SCALE_F  v_type=G type=F  num_elts=64 alias=<%2,0>\n"
      ".decl TMP_F     v_type=G type=F  num_elts=64 align=32\n"
      ".decl TMP_HF    v_type=G type=HF num_elts=64 align=32\n"
      // BF16 → F32
      "mov (M1_NM, 32) TMP_F(0,0)<1>   IN_BF(0,0)<1;1,0>\n"
      "mov (M1_NM, 32) TMP_F(0,32)<1>  IN_BF(0,32)<1;1,0>\n"
      // Multiply by per-value scale in F32 (4× SIMD-16)
      "mul (M1_NM, 16) TMP_F(0,0)<1>   TMP_F(0,0)<1;1,0>   SCALE_F(0,0)<0;1,0>\n"
      "mul (M1_NM, 16) TMP_F(0,16)<1>  TMP_F(0,16)<1;1,0>  SCALE_F(0,16)<0;1,0>\n"
      "mul (M1_NM, 16) TMP_F(0,32)<1>  TMP_F(0,32)<1;1,0>  SCALE_F(0,32)<0;1,0>\n"
      "mul (M1_NM, 16) TMP_F(0,48)<1>  TMP_F(0,48)<1;1,0>  SCALE_F(0,48)<0;1,0>\n"
      // F32 → HF
      "mov (M1_NM, 32) TMP_HF(0,0)<1>  TMP_F(0,0)<1;1,0>\n"
      "mov (M1_NM, 32) TMP_HF(0,32)<1> TMP_F(0,32)<1;1,0>\n"
      // HF → E5M2 via fcvt
      "fcvt (M1_NM, 32) OUT_UB(0,0)<1>  TMP_HF(0,0)<1;1,0>\n"
      "fcvt (M1_NM, 32) OUT_UB(0,32)<1> TMP_HF(0,32)<1;1,0>\n"
      "}\n"
      : "=rw"(dst)
      : "rw"(src), "rw"(scales)
    );
#else
    CUTE_INVALID_CONTROL_PATH("Xe Version < 35 not implemented");
#endif
#else
    CUTE_INVALID_CONTROL_PATH("Not Xe");
#endif
  }
};

// bfloat16_t → float_e4m3_t: mul(BF→F) + fcvt(F→HF→B)
template <>
struct Xe_Quantize_Optimized<cutlass::bfloat16_t, cutlass::float_e4m3_t>
{
  using SrcRegister = intel::ushort4;
  using DstRegister = intel::uchar4;
  using ScaleRegister = intel::float4;

  CUTE_HOST_DEVICE static void
  quantize(SrcRegister const* src_p, DstRegister* dst_p, ScaleRegister scales)
  {
#if defined(CUTE_ARCH_QUANTIZE_XE_ENABLED)
#if SYCL_INTEL_TARGET >= 35
    auto const& src = *src_p;
    auto& dst = *dst_p;
    asm (
      "{\n"
      ".decl IN_BF    v_type=G type=BF num_elts=64 alias=<%1,0>\n"
      ".decl OUT_B     v_type=G type=B  num_elts=64 alias=<%0,0>\n"
      ".decl SCALE_F  v_type=G type=F  num_elts=64 alias=<%2,0>\n"
      ".decl TMP_F     v_type=G type=F  num_elts=64 align=32\n"
      ".decl TMP_HF    v_type=G type=HF num_elts=64 align=32\n"
      // BF16 → F32
      "mov (M1_NM, 32) TMP_F(0,0)<1>   IN_BF(0,0)<1;1,0>\n"
      "mov (M1_NM, 32) TMP_F(0,32)<1>  IN_BF(0,32)<1;1,0>\n"
      // Multiply by per-value scale in F32 (4× SIMD-16)
      "mul (M1_NM, 16) TMP_F(0,0)<1>   TMP_F(0,0)<1;1,0>   SCALE_F(0,0)<0;1,0>\n"
      "mul (M1_NM, 16) TMP_F(0,16)<1>  TMP_F(0,16)<1;1,0>  SCALE_F(0,16)<0;1,0>\n"
      "mul (M1_NM, 16) TMP_F(0,32)<1>  TMP_F(0,32)<1;1,0>  SCALE_F(0,32)<0;1,0>\n"
      "mul (M1_NM, 16) TMP_F(0,48)<1>  TMP_F(0,48)<1;1,0>  SCALE_F(0,48)<0;1,0>\n"
      // F32 → HF
      "mov (M1_NM, 32) TMP_HF(0,0)<1>  TMP_F(0,0)<1;1,0>\n"
      "mov (M1_NM, 32) TMP_HF(0,32)<1> TMP_F(0,32)<1;1,0>\n"
      // HF → E4M3 via fcvt
      "fcvt (M1_NM, 32) OUT_B(0,0)<1>  TMP_HF(0,0)<1;1,0>\n"
      "fcvt (M1_NM, 32) OUT_B(0,32)<1> TMP_HF(0,32)<1;1,0>\n"
      "}\n"
      : "=rw"(dst)
      : "rw"(src), "rw"(scales)
    );
#else
    CUTE_INVALID_CONTROL_PATH("Xe Version < 35 not implemented");
#endif
#else
    CUTE_INVALID_CONTROL_PATH("Not Xe");
#endif
  }
};

// float → float_e5m2_t: mul(F32) + fcvt(F→HF→UB)
template <>
struct Xe_Quantize_Optimized<float, cutlass::float_e5m2_t>
{
  using SrcRegister = intel::float4;
  using DstRegister = intel::uchar4;
  using ScaleRegister = intel::float4;

  CUTE_HOST_DEVICE static void
  quantize(SrcRegister const* src_p, DstRegister* dst_p, ScaleRegister scales)
  {
#if defined(CUTE_ARCH_QUANTIZE_XE_ENABLED)
#if SYCL_INTEL_TARGET >= 35
    auto const& src = *src_p;
    auto& dst = *dst_p;
    asm (
      "{\n"
      ".decl IN_F     v_type=G type=F  num_elts=64 alias=<%1,0>\n"
      ".decl OUT_UB    v_type=G type=UB num_elts=64 alias=<%0,0>\n"
      ".decl SCALE_F  v_type=G type=F  num_elts=64 alias=<%2,0>\n"
      ".decl TMP_F     v_type=G type=F  num_elts=64 align=32\n"
      ".decl TMP_HF    v_type=G type=HF num_elts=64 align=32\n"
      // Multiply by per-value scale in F32 (4× SIMD-16)
      "mul (M1_NM, 16) TMP_F(0,0)<1>   IN_F(0,0)<1;1,0>   SCALE_F(0,0)<0;1,0>\n"
      "mul (M1_NM, 16) TMP_F(0,16)<1>  IN_F(0,16)<1;1,0>  SCALE_F(0,16)<0;1,0>\n"
      "mul (M1_NM, 16) TMP_F(0,32)<1>  IN_F(0,32)<1;1,0>  SCALE_F(0,32)<0;1,0>\n"
      "mul (M1_NM, 16) TMP_F(0,48)<1>  IN_F(0,48)<1;1,0>  SCALE_F(0,48)<0;1,0>\n"
      // F32 → HF (narrow for fcvt input)
      "mov (M1_NM, 32) TMP_HF(0,0)<1>  TMP_F(0,0)<1;1,0>\n"
      "mov (M1_NM, 32) TMP_HF(0,32)<1> TMP_F(0,32)<1;1,0>\n"
      // HF → E5M2 via fcvt
      "fcvt (M1_NM, 32) OUT_UB(0,0)<1>  TMP_HF(0,0)<1;1,0>\n"
      "fcvt (M1_NM, 32) OUT_UB(0,32)<1> TMP_HF(0,32)<1;1,0>\n"
      "}\n"
      : "=rw"(dst)
      : "rw"(src), "rw"(scales)
    );
#else
    CUTE_INVALID_CONTROL_PATH("Xe Version < 35 not implemented");
#endif
#else
    CUTE_INVALID_CONTROL_PATH("Not Xe");
#endif
  }
};

// float → float_e4m3_t: mul(F32) + fcvt(F→HF→B)
template <>
struct Xe_Quantize_Optimized<float, cutlass::float_e4m3_t>
{
  using SrcRegister = intel::float4;
  using DstRegister = intel::uchar4;
  using ScaleRegister = intel::float4;

  CUTE_HOST_DEVICE static void
  quantize(SrcRegister const* src_p, DstRegister* dst_p, ScaleRegister scales)
  {
#if defined(CUTE_ARCH_QUANTIZE_XE_ENABLED)
#if SYCL_INTEL_TARGET >= 35
    auto const& src = *src_p;
    auto& dst = *dst_p;
    asm (
      "{\n"
      ".decl IN_F     v_type=G type=F  num_elts=64 alias=<%1,0>\n"
      ".decl OUT_B     v_type=G type=B  num_elts=64 alias=<%0,0>\n"
      ".decl SCALE_F  v_type=G type=F  num_elts=64 alias=<%2,0>\n"
      ".decl TMP_F     v_type=G type=F  num_elts=64 align=32\n"
      ".decl TMP_HF    v_type=G type=HF num_elts=64 align=32\n"
      // Multiply by per-value scale in F32 (4× SIMD-16)
      "mul (M1_NM, 16) TMP_F(0,0)<1>   IN_F(0,0)<1;1,0>   SCALE_F(0,0)<0;1,0>\n"
      "mul (M1_NM, 16) TMP_F(0,16)<1>  IN_F(0,16)<1;1,0>  SCALE_F(0,16)<0;1,0>\n"
      "mul (M1_NM, 16) TMP_F(0,32)<1>  IN_F(0,32)<1;1,0>  SCALE_F(0,32)<0;1,0>\n"
      "mul (M1_NM, 16) TMP_F(0,48)<1>  IN_F(0,48)<1;1,0>  SCALE_F(0,48)<0;1,0>\n"
      // F32 → HF
      "mov (M1_NM, 32) TMP_HF(0,0)<1>  TMP_F(0,0)<1;1,0>\n"
      "mov (M1_NM, 32) TMP_HF(0,32)<1> TMP_F(0,32)<1;1,0>\n"
      // HF → E4M3 via fcvt
      "fcvt (M1_NM, 32) OUT_B(0,0)<1>  TMP_HF(0,0)<1;1,0>\n"
      "fcvt (M1_NM, 32) OUT_B(0,32)<1> TMP_HF(0,32)<1;1,0>\n"
      "}\n"
      : "=rw"(dst)
      : "rw"(src), "rw"(scales)
    );
#else
    CUTE_INVALID_CONTROL_PATH("Xe Version < 35 not implemented");
#endif
#else
    CUTE_INVALID_CONTROL_PATH("Not Xe");
#endif
  }
};

} // namespace cute

