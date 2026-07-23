/***************************************************************************************************
* Copyright (C) 2025 Intel Corporation, All rights reserved.
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

#include <cute/util/sycl_vec.hpp>           // native vector types
#include <cute/arch/reorder.hpp>            // Universal_Reorder_UU

#if defined(__SYCL_DEVICE_ONLY__) && defined(SYCL_INTEL_TARGET)
#define CUTE_ARCH_REORDER_XE_ENABLED
#endif

namespace cute {

#if defined(SYCL_INTEL_TARGET) && (SYCL_INTEL_TARGET == 35)
template <>
struct Xe_Reorder<ReorderKind::UU, bfloat16_t, float_e4m3_t>
{
  using SRegisters = intel::ushort4[1];
  using DRegisters = intel::uchar4[1];

  CUTE_HOST_DEVICE static void
  reorder(intel::ushort4 const& src0, intel::uchar4& dst0)
  {
#if defined(CUTE_ARCH_REORDER_XE_ENABLED)
    asm (
      "{\n"
      ".decl IN_BF v_type=G type=BF num_elts=64 alias=<%1,0>\n"
      ".decl OUT_B v_type=G type=B num_elts=64 alias=<%0,0>\n"
      ".decl TMP_F v_type=G type=F num_elts=64 align=32\n"
      ".decl TMP_HF v_type=G type=HF num_elts=64 align=32\n"
      "mov (M1_NM, 32) TMP_F(0,0)<1> IN_BF(0,0)<1;1,0>\n"
      "mov (M1_NM, 32) TMP_F(2,0)<1> IN_BF(1,0)<1;1,0>\n"
      "mov (M1_NM, 32) TMP_HF(0,0)<1> TMP_F(0,0)<1;1,0>\n"
      "mov (M1_NM, 32) TMP_HF(1,0)<1> TMP_F(2,0)<1;1,0>\n"
      "fcvt (M1_NM, 32) OUT_B(0,0)<1> TMP_HF(0,0)<1;1,0>\n"
      "fcvt (M1_NM, 32) OUT_B(0,32)<1> TMP_HF(1,0)<1;1,0>\n"
      "}\n"
      : "=rw"(dst0)
      : "rw"(src0)
    );
#else
  CUTE_INVALID_CONTROL_PATH("Not Xe");
#endif
  }
};

template <>
struct Xe_Reorder<ReorderKind::VV, bfloat16_t, float_e4m3_t>
{
  using SRegisters = intel::ushort4[1];
  using DRegisters = intel::uchar4[1];

  CUTE_HOST_DEVICE static void
  reorder(intel::ushort4 const& src0, intel::uchar4& dst0)
  {
#if defined(CUTE_ARCH_REORDER_XE_ENABLED)
    asm (
      "{\n"
      ".decl IN_BF v_type=G type=BF num_elts=64 alias=<%1,0>\n"
      ".decl OUT_UW v_type=G type=UW num_elts=32 alias=<%0,0>\n"
      ".decl TMP_F v_type=G type=F num_elts=64 align=32\n"
      ".decl TMP_HF v_type=G type=HF num_elts=64 align=32\n"
      ".decl TMP_B v_type=G type=B num_elts=64 align=32\n"
      ".decl TMP_UW v_type=G type=UW num_elts=32 alias=<TMP_B,0>\n"
      "mov (M1_NM, 32) TMP_F(0,0)<1> IN_BF(0,0)<1;1,0>\n"
      "mov (M1_NM, 32) TMP_F(2,0)<1> IN_BF(1,0)<1;1,0>\n"
      "mov (M1_NM, 32) TMP_HF(0,0)<1> TMP_F(0,0)<1;1,0>\n"
      "mov (M1_NM, 32) TMP_HF(1,0)<1> TMP_F(2,0)<1;1,0>\n"
      "fcvt (M1_NM, 32) TMP_B(0,0)<1> TMP_HF(0,0)<1;1,0>\n"
      "fcvt (M1_NM, 32) TMP_B(0,32)<1> TMP_HF(1,0)<1;1,0>\n"
      "mov (M1_NM, 16) OUT_UW(0,0)<2> TMP_UW(0,0)<1;1,0>\n"
      "mov (M1_NM, 16) OUT_UW(0,1)<2> TMP_UW(0,16)<1;1,0>\n"
      "}\n"
      : "=rw"(dst0)
      : "rw"(src0)
    );
#else
  CUTE_INVALID_CONTROL_PATH("Not Xe");
#endif
  }
};
template <>
struct Xe_Reorder<ReorderKind::UU, half_t, float_e4m3_t>
{
  using SRegisters = intel::ushort4[1];
  using DRegisters = intel::uchar4[1];

  CUTE_HOST_DEVICE static void
  reorder(intel::ushort4 const& src0, intel::uchar4& dst0)
  {
#if defined(CUTE_ARCH_REORDER_XE_ENABLED)
    asm (
      "{\n"
      ".decl IN_HF v_type=G type=HF num_elts=64 alias=<%1,0>\n"
      ".decl OUT_B v_type=G type=B num_elts=64 alias=<%0,0>\n"
      "fcvt (M1_NM, 32) OUT_B(0,0)<1> IN_HF(0,0)<1;1,0>\n"
      "fcvt (M1_NM, 32) OUT_B(0,32)<1> IN_HF(1,0)<1;1,0>\n"
      "}\n"
      : "=rw"(dst0)
      : "rw"(src0)
    );
#else
  CUTE_INVALID_CONTROL_PATH("Not Xe");
#endif
  }
};

template <>
struct Xe_Reorder<ReorderKind::VV, half_t, float_e4m3_t>
{
  using SRegisters = intel::ushort4[1];
  using DRegisters = intel::uchar4[1];

  CUTE_HOST_DEVICE static void
  reorder(intel::ushort4 const& src0, intel::uchar4& dst0)
  {
#if defined(CUTE_ARCH_REORDER_XE_ENABLED)
    asm (
      "{\n"
      ".decl IN_HF v_type=G type=HF num_elts=64 alias=<%1,0>\n"
      ".decl OUT_UW v_type=G type=UW num_elts=32 alias=<%0,0>\n"
      ".decl TMP_B v_type=G type=B num_elts=64 align=32\n"
      ".decl TMP_UW v_type=G type=UW num_elts=32 alias=<TMP_B,0>\n"
      "fcvt (M1_NM, 32) TMP_B(0,0)<1> IN_HF(0,0)<1;1,0>\n"
      "fcvt (M1_NM, 32) TMP_B(0,32)<1> IN_HF(1,0)<1;1,0>\n"
      "mov (M1_NM, 16) OUT_UW(0,0)<2> TMP_UW(0,0)<1;1,0>\n"
      "mov (M1_NM, 16) OUT_UW(0,1)<2> TMP_UW(0,16)<1;1,0>\n"
      "}\n"
      : "=rw"(dst0)
      : "rw"(src0)
    );
#else
  CUTE_INVALID_CONTROL_PATH("Not Xe");
#endif
  }
};

#define CUTE_XE_REORDER_DNSCL_SEQ_VNNI(CVT_TYPE) \
    ".decl IN_UD v_type=G type=UD num_elts=64 alias=<%1,0>\n"   \
    ".decl OUT_UD v_type=G type=UD num_elts=16 alias=<%0,0>\n"  \
    ".decl TMP0 v_type=G type=UD num_elts=16 align=wordx32\n"   \
    ".decl TMP1 v_type=G type=UD num_elts=16 align=wordx32\n"   \
    /* mode0: src0=GRF0(K0,K1), src1=GRF2(K4,K5) */             \
    /* [23:20]=K5, [19:16]=K4, [7:4]=K1, [3:0]=K0 */            \
    "dnscl." CVT_TYPE ".mode0.rne (M1, 16) TMP0.0 IN_UD.0 IN_UD.128 %%null.0\n" \
    /* mode2: src0=GRF1(K2,K3), src1=GRF3(K6,K7) */             \
    /* [31:28]=K7, [27:24]=K6, [15:12]=K3, [11:8]=K2 */         \
    "dnscl." CVT_TYPE ".mode2.rne (M1, 16) TMP1.0 IN_UD.64 IN_UD.192 %%null.0\n"\
    "or (M1, 16) OUT_UD(0,0)<1> TMP0(0,0)<1;1,0> TMP1(0,0)<1;1,0>\n"

#define CUTE_XE_REORDER_DNSCL_SEQ(CVT_TYPE) \
    ".decl IN_UD v_type=G type=UD num_elts=64 alias=<%1,0>\n"   \
    ".decl OUT_UW v_type=G type=UW num_elts=32 alias=<%0,0>\n"  \
    ".decl TMP_UD0 v_type=G type=UD num_elts=16 align=wordx32\n"              \
    ".decl TMP_UD1 v_type=G type=UD num_elts=16 align=wordx32\n"              \
    ".decl TMP_HI v_type=G type=UD num_elts=16\n"               \
    ".decl TMP_LO v_type=G type=UD num_elts=16\n"               \
    /* First 64 BF16 elements (GRF0-1, UD 0-31) */              \
    "dnscl." CVT_TYPE ".mode0.rne (M1, 16) TMP_UD0.0 IN_UD.0 IN_UD.64 %%null.0\n" \
    "shr (M1, 16) TMP_HI(0,0)<1> TMP_UD0(0,0)<1;1,0> 0x8:ud\n"  \
    "and (M1, 16) TMP_LO(0,0)<1> TMP_UD0(0,0)<1;1,0> 0xFF:ud\n" \
    "or (M1, 16) OUT_UW(0,0)<1> TMP_HI(0,0)<1;1,0> TMP_LO(0,0)<1;1,0>\n" \
    /* Second 64 BF16 elements (GRF2-3, UD 32-63) */            \
    "dnscl." CVT_TYPE ".mode0.rne (M1, 16) TMP_UD1.0 IN_UD.128 IN_UD.192 %%null.0\n" \
    "shr (M1, 16) TMP_HI(0,0)<1> TMP_UD1(0,0)<1;1,0> 0x8:ud\n"  \
    "and (M1, 16) TMP_LO(0,0)<1> TMP_UD1(0,0)<1;1,0> 0xFF:ud\n" \
    "or (M1, 16) OUT_UW(0,16)<1> TMP_HI(0,0)<1;1,0> TMP_LO(0,0)<1;1,0>\n"

template <>
struct Xe_Reorder<ReorderKind::UU, bfloat16_t, float_e2m1_t>
{
  using SRegisters = intel::ushort8[1]; // 256 bytes (4 GRFs 128 elements of BF16)
  using DRegisters = intel::uchar4[1];  // 64 bytes (128 elements of 4-bit, packed)

  CUTE_HOST_DEVICE static void
  reorder(intel::ushort8 const& src0, intel::uchar4& dst0)
  {
#if defined(CUTE_ARCH_REORDER_XE_ENABLED)
    asm (
      "{\n"
      CUTE_XE_REORDER_DNSCL_SEQ("bftoe2m1")
      "}\n"
      : "=rw"(dst0)
      : "rw"(src0)
    );
#else
  CUTE_INVALID_CONTROL_PATH("Not Xe");
#endif
  }
};

template <>
struct Xe_Reorder<ReorderKind::UU, bfloat16_t, int4_t>
{
  using SRegisters = intel::ushort8[1];
  using DRegisters = intel::uchar4[1];

  CUTE_HOST_DEVICE static void
  reorder(intel::ushort8 const& src0, intel::uchar4& dst0)
  {
#if defined(CUTE_ARCH_REORDER_XE_ENABLED)
    asm (
      "{\n"
      CUTE_XE_REORDER_DNSCL_SEQ("bftoint4")
      "}\n"
      : "=rw"(dst0)
      : "rw"(src0)
    );
#else
  CUTE_INVALID_CONTROL_PATH("Not Xe");
#endif
  }
};

template <>
struct Xe_Reorder<ReorderKind::UU, half_t, float_e2m1_t>
{
  using SRegisters = intel::ushort8[1];
  using DRegisters = intel::uchar4[1];

  CUTE_HOST_DEVICE static void
  reorder(intel::ushort8 const& src0, intel::uchar4& dst0)
  {
#if defined(CUTE_ARCH_REORDER_XE_ENABLED)
    asm (
      "{\n"
      CUTE_XE_REORDER_DNSCL_SEQ("hftoe2m1")
      "}\n"
      : "=rw"(dst0)
      : "rw"(src0)
    );
#else
  CUTE_INVALID_CONTROL_PATH("Not Xe");
#endif
  }
};

template <>
struct Xe_Reorder<ReorderKind::UU_Universal, float, float_e2m1_t>
{
  using SRegisters = intel::float2[1];
  using DRegisters = intel::vector_t<uint8_t, 1>[1];

  CUTE_HOST_DEVICE static void
  reorder(intel::float2 const& src0, intel::vector_t<uint8_t, 1>& dst0)
  {
#if defined(CUTE_ARCH_REORDER_XE_ENABLED)
    // Convert 32 float -> 32 half -> 32 e2m1 (16 bytes output)
    asm (
      "{\n"
      ".decl IN_F v_type=G type=F num_elts=32 alias=<%1,0>\n"
      ".decl TMP_HF v_type=G type=HF num_elts=16 align=wordx32\n"
      ".decl TMP_HF_1 v_type=G type=HF num_elts=16 align=wordx32\n"
      ".decl TMP_UD v_type=G type=UD num_elts=8 alias=<TMP_HF,0>\n"
      ".decl TMP_UD_1 v_type=G type=UD num_elts=8 alias=<TMP_HF_1,0>\n"
      ".decl OUT_UD v_type=G type=UD num_elts=8 align=wordx32\n"
      ".decl OUT_UD_UB v_type=G type=UB num_elts=32 alias=<OUT_UD,0>\n"
      ".decl OUT_UB v_type=G type=UB num_elts=16 alias=<%0,0>\n"
      "mov (M1, 16) TMP_HF(0,0)<1> IN_F(0,0)<1;1,0>\n"
      "mov (M1, 16) TMP_HF_1(0,0)<1> IN_F(0,16)<1;1,0>\n"
      "dnscl.hftoe2m1.mode0.rne (M1, 8) OUT_UD.0 TMP_UD.0 TMP_UD_1.0 %%null.0\n"
      "mov (M1, 8) OUT_UB(0,0)<1> OUT_UD_UB(0,0)<4;1,0>\n"
      "mov (M1, 8) OUT_UB(0,8)<1> OUT_UD_UB(0,2)<4;1,0>\n"
      "}\n"
      : "=rw"(dst0)
      : "rw"(src0)
    );
#else
  CUTE_INVALID_CONTROL_PATH("Not Xe");
#endif
  }
};

template <>
struct Xe_Reorder<ReorderKind::UU, half_t, int4_t>
{
  using SRegisters = intel::ushort8[1];
  using DRegisters = intel::uchar4[1];

  CUTE_HOST_DEVICE static void
  reorder(intel::ushort8 const& src0, intel::uchar4& dst0)
  {
#if defined(CUTE_ARCH_REORDER_XE_ENABLED)
    asm (
      "{\n"
      CUTE_XE_REORDER_DNSCL_SEQ("hftoint4")
      "}\n"
      : "=rw"(dst0)
      : "rw"(src0)
    );
#else
  CUTE_INVALID_CONTROL_PATH("Not Xe");
#endif
  }
};

template <>
struct Xe_Reorder<ReorderKind::VV, bfloat16_t, float_e2m1_t>
{
  using SRegisters = intel::ushort8[1];
  using DRegisters = intel::uchar4[1];

  CUTE_HOST_DEVICE static void
  reorder(intel::ushort8 const& src0, intel::uchar4& dst0)
  {
#if defined(CUTE_ARCH_REORDER_XE_ENABLED)
    asm (
      "{\n"
      CUTE_XE_REORDER_DNSCL_SEQ_VNNI("bftoe2m1")
      "}\n"
      : "=rw"(dst0)
      : "rw"(src0)
    );
#else
  CUTE_INVALID_CONTROL_PATH("Not Xe");
#endif
  }
};

template <>
struct Xe_Reorder<ReorderKind::VV, bfloat16_t, int4_t>
{
  using SRegisters = intel::ushort8[1];
  using DRegisters = intel::uchar4[1];

  CUTE_HOST_DEVICE static void
  reorder(intel::ushort8 const& src0, intel::uchar4& dst0)
  {
#if defined(CUTE_ARCH_REORDER_XE_ENABLED)
    asm (
      "{\n"
      CUTE_XE_REORDER_DNSCL_SEQ_VNNI("bftoint4")
      "}\n"
      : "=rw"(dst0)
      : "rw"(src0)
    );
#else
  CUTE_INVALID_CONTROL_PATH("Not Xe");
#endif
  }
};

template <>
struct Xe_Reorder<ReorderKind::VV, half_t, float_e2m1_t>
{
  using SRegisters = intel::ushort8[1];
  using DRegisters = intel::uchar4[1];

  CUTE_HOST_DEVICE static void
  reorder(intel::ushort8 const& src0, intel::uchar4& dst0)
  {
#if defined(CUTE_ARCH_REORDER_XE_ENABLED)
    asm (
      "{\n"
      CUTE_XE_REORDER_DNSCL_SEQ_VNNI("hftoe2m1")
      "}\n"
      : "=rw"(dst0)
      : "rw"(src0)
    );
#else
  CUTE_INVALID_CONTROL_PATH("Not Xe");
#endif
  }
};

template <>
struct Xe_Reorder<ReorderKind::VV, half_t, int4_t>
{
  using SRegisters = intel::ushort8[1];
  using DRegisters = intel::uchar4[1];

  CUTE_HOST_DEVICE static void
  reorder(intel::ushort8 const& src0, intel::uchar4& dst0)
  {
#if defined(CUTE_ARCH_REORDER_XE_ENABLED)
    asm (
      "{\n"
      CUTE_XE_REORDER_DNSCL_SEQ_VNNI("hftoint4")
      "}\n"
      : "=rw"(dst0)
      : "rw"(src0)
    );
#else
  CUTE_INVALID_CONTROL_PATH("Not Xe");
#endif
  }
};

#endif

} // end namespace cute
