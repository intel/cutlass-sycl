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

#include <cute/util/sycl_vec.hpp>           // native vector types
#include <cute/arch/reorder.hpp>            // Universal_Reorder_UU

#if defined(__SYCL_DEVICE_ONLY__) && defined(SYCL_INTEL_TARGET)
#define CUTE_ARCH_REORDER_XE_ENABLED
#endif

namespace cute {

template <typename SrcType, typename DstType>
struct Xe_Reorder<ReorderKind::UU, SrcType, DstType> : Universal_Reorder_UU<SrcType, DstType> {};

template <>
struct Xe_Reorder<ReorderKind::UU, uint8_t, bfloat16_t>
{
  using SRegisters = intel::uchar2[1];
  using DRegisters = intel::ushort2[1];

  CUTE_HOST_DEVICE static void
  reorder(intel::uchar2 const& src0, intel::ushort2& dst0)
  {
#if defined(CUTE_ARCH_COPY_XE_ENABLED)
    asm (     /* Latency: 4 cycles/output register */
      "{\n"
      ".decl IN_UB v_type=G type=UB num_elts=32 alias=<%1,0>\n"
      ".decl OUT_UW v_type=G type=UW num_elts=32 alias=<%0,0>\n"
      ".decl OUT_HF v_type=G type=HF num_elts=32 alias=<%0,0>\n"
      ".decl NZ_PRED0 v_type=P num_elts=32\n"
      "mov (M1_NM, 32) OUT_UW(0,0)<1> IN_UB(0,0)<1;1,0>\n"
      "mul (M1_NM, 32) OUT_HF(0,0)<1> OUT_HF(0,0)<1;1,0> 0x6C00:hf\n"
      "cmp.ne (M1_NM, 32) NZ_PRED0 OUT_HF(0,0)<1;1,0> 0x0:hf\n"       /* fused with preceding mul */
      "shr (M1_NM,32) OUT_UW(0,0)<1> OUT_UW(0,0)<1;1,0> 3:uw\n"
      "(NZ_PRED0) add (M1_NM,32) OUT_UW(0,0)<1> OUT_UW(0,0)<1;1,0> 0x3E00:uw\n"
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
struct Xe_Reorder<ReorderKind::VV, uint8_t, bfloat16_t>
{
  using SRegisters = intel::uchar4[1];
  using DRegisters = intel::ushort4[1];

  CUTE_HOST_DEVICE static void
  reorder(intel::uchar4 const& src0, intel::ushort4& dst0)
  {
#if defined(CUTE_ARCH_COPY_XE_ENABLED)
    asm (     /* Latency: 4 cycles/output register */
      "{\n"
      ".decl IN_UB v_type=G type=UB num_elts=64 alias=<%1,0>\n"
      ".decl OUT_UW v_type=G type=UW num_elts=64 alias=<%0,0>\n"
      ".decl OUT_HF v_type=G type=HF num_elts=64 alias=<%0,0>\n"
      ".decl NZ_PRED0 v_type=P num_elts=32\n"
      ".decl NZ_PRED1 v_type=P num_elts=32\n"
      "mov (M1_NM, 32) OUT_UW(0,0)<1> IN_UB(0,0)<4;2,1>\n"
      "mov (M1_NM, 32) OUT_UW(1,0)<1> IN_UB(0,2)<4;2,1>\n"
      "mul (M1_NM, 32) OUT_HF(0,0)<1> OUT_HF(0,0)<1;1,0> 0x6C00:hf\n"
      "mul (M1_NM, 32) OUT_HF(1,0)<1> OUT_HF(1,0)<1;1,0> 0x6C00:hf\n"
      "cmp.ne (M1_NM, 32) NZ_PRED0 OUT_HF(0,0)<1;1,0> 0x0:hf\n"       /* fused with preceding mul */
      "cmp.ne (M1_NM, 32) NZ_PRED1 OUT_HF(1,0)<1;1,0> 0x0:hf\n"
      "shr (M1_NM,32) OUT_UW(0,0)<1> OUT_UW(0,0)<1;1,0> 3:uw\n"
      "shr (M1_NM,32) OUT_UW(1,0)<1> OUT_UW(1,0)<1;1,0> 3:uw\n"
      "(NZ_PRED0) add (M1_NM,32) OUT_UW(0,0)<1> OUT_UW(0,0)<1;1,0> 0x3E00:uw\n"
      "(NZ_PRED1) add (M1_NM,32) OUT_UW(1,0)<1> OUT_UW(1,0)<1;1,0> 0x3E00:uw\n"
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
struct Xe_Reorder<ReorderKind::VV, uint8_t, half_t>
{
  using SRegisters = intel::uchar4[1];
  using DRegisters = intel::ushort4[1];

  CUTE_HOST_DEVICE static void
  reorder(intel::uchar4 const& src0, intel::ushort4& dst0)
  {
#if defined(CUTE_ARCH_COPY_XE_ENABLED)
    asm (     /* Latency: 3 cycles/output register */
      "{\n"
      ".decl IN_UB v_type=G type=UB num_elts=64 alias=<%1,0>\n"
      ".decl OUT_UW v_type=G type=UW num_elts=64 alias=<%0,0>\n"
      ".decl OUT_HF v_type=G type=HF num_elts=64 alias=<%0,0>\n"
      "mov (M1_NM, 32) OUT_UW(0,0)<1> IN_UB(0,0)<4;2,1>\n"
      "mov (M1_NM, 32) OUT_UW(1,0)<1> IN_UB(0,2)<4;2,1>\n"
      "mul (M1_NM, 32) OUT_HF(0,0)<1> OUT_HF(0,0)<1;1,0> 0x6c00:hf\n"
      "mul (M1_NM, 32) OUT_HF(1,0)<1> OUT_HF(1,0)<1;1,0> 0x6c00:hf\n"
      "mul (M1_NM, 32) OUT_HF(0,0)<1> OUT_HF(0,0)<1;1,0> 0x6c00:hf\n"
      "mul (M1_NM, 32) OUT_HF(1,0)<1> OUT_HF(1,0)<1;1,0> 0x6c00:hf\n"
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
struct Xe_Reorder<ReorderKind::UU, float_e5m2_t, half_t>
{
  using SRegisters = intel::uchar4[1];
  using DRegisters = intel::ushort4[1];

  CUTE_HOST_DEVICE static void
  reorder(intel::uchar4 const& src0, intel::ushort4& dst0)
  {
#if defined(CUTE_ARCH_COPY_XE_ENABLED)
    asm (     /* Latency: 1 cycle/output register */
      "{\n"
      ".decl IN_UB v_type=G type=UB num_elts=64 alias=<%1,0>\n"
      ".decl OUT_UW v_type=G type=UW num_elts=64 alias=<%0,0>\n"
      "shl (M1_NM, 32) OUT_UW(0,0)<1> IN_UB(0,0)<1;1,0> 8:uw\n"
      "shl (M1_NM, 32) OUT_UW(1,0)<1> IN_UB(0,32)<1;1,0> 8:uw\n"
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
struct Xe_Reorder<ReorderKind::VV, float_e5m2_t, half_t>
{
  using SRegisters = intel::uchar4[1];
  using DRegisters = intel::ushort4[1];

  CUTE_HOST_DEVICE static void
  reorder(intel::uchar4 const& src0, intel::ushort4& dst0)
  {
#if defined(CUTE_ARCH_COPY_XE_ENABLED)
    asm (     /* Latency: 1 cycle/output register */
      "{\n"
      ".decl IN_UB v_type=G type=UB num_elts=64 alias=<%1,0>\n"
      ".decl OUT_UW v_type=G type=UW num_elts=64 alias=<%0,0>\n"
      "shl (M1_NM, 32) OUT_UW(0,0)<1> IN_UB(0,0)<4;2,1> 8:uw\n"
      "shl (M1_NM, 32) OUT_UW(1,0)<1> IN_UB(0,2)<4;2,1> 8:uw\n"
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
struct Xe_Reorder<ReorderKind::UU, uint4_t, half_t>
{
  using SRegisters = intel::uchar4[1];
  using DRegisters = intel::ushort8[1];

  CUTE_HOST_DEVICE static void
  reorder(intel::uchar4 const& src0, intel::ushort8& dst0)
  {
#if defined(CUTE_ARCH_COPY_XE_ENABLED)
    asm (     /* Latency: 4 cycles/output register */
      "{\n"
      ".decl IN_UB v_type=G type=UB num_elts=64 alias=<%1,0>\n"
      ".decl OUT_UW v_type=G type=UW num_elts=128 alias=<%0,0>\n"
      ".decl OUT_HF v_type=G type=HF num_elts=128 alias=<%0,0>\n"
      "and (M1_NM, 32) OUT_UW(0,0)<2> IN_UB(0,0)<1;1,0> 0xf:uw\n"
      "shr (M1_NM, 32) OUT_UW(0,1)<2> IN_UB(0,0)<1;1,0> 0x4:uw\n"
      "and (M1_NM, 32) OUT_UW(2,0)<2> IN_UB(0,32)<1;1,0> 0xf:uw\n"
      "shr (M1_NM, 32) OUT_UW(2,1)<2> IN_UB(0,32)<1;1,0> 0x4:uw\n"
      "mul (M1_NM, 32) OUT_HF(0,0)<1> OUT_HF(0,0)<1;1,0> 0x6c00:hf\n"
      "mul (M1_NM, 32) OUT_HF(1,0)<1> OUT_HF(1,0)<1;1,0> 0x6c00:hf\n"
      "mul (M1_NM, 32) OUT_HF(2,0)<1> OUT_HF(2,0)<1;1,0> 0x6c00:hf\n"
      "mul (M1_NM, 32) OUT_HF(3,0)<1> OUT_HF(3,0)<1;1,0> 0x6c00:hf\n"
      "mul (M1_NM, 32) OUT_HF(0,0)<1> OUT_HF(0,0)<1;1,0> 0x6c00:hf\n"
      "mul (M1_NM, 32) OUT_HF(1,0)<1> OUT_HF(1,0)<1;1,0> 0x6c00:hf\n"
      "mul (M1_NM, 32) OUT_HF(2,0)<1> OUT_HF(2,0)<1;1,0> 0x6c00:hf\n"
      "mul (M1_NM, 32) OUT_HF(3,0)<1> OUT_HF(3,0)<1;1,0> 0x6c00:hf\n"
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
struct Xe_Reorder<ReorderKind::VV, uint4_t, half_t>
{
  using SRegisters = intel::uchar4[1];
  using DRegisters = intel::ushort8[1];

  CUTE_HOST_DEVICE static void
  reorder(intel::uchar4 const& src0, intel::ushort8& dst0)
  {
#if defined(CUTE_ARCH_COPY_XE_ENABLED)
    asm (     /* Latency: 4 cycles/output register */
      "{\n"
      ".decl IN_UB v_type=G type=UB num_elts=64 alias=<%1,0>\n"
      ".decl OUT_UW v_type=G type=UW num_elts=128 alias=<%0,0>\n"
      ".decl OUT_HF v_type=G type=HF num_elts=128 alias=<%0,0>\n"
      "and (M1_NM, 16) OUT_UW(0,0)<2> IN_UB(0,0)<4;1,0> 0xf:uw\n"
      "shr (M1_NM, 16) OUT_UW(0,1)<2> IN_UB(0,0)<4;1,0> 0x4:uw\n"
      "and (M1_NM, 16) OUT_UW(1,0)<2> IN_UB(0,1)<4;1,0> 0xf:uw\n"
      "shr (M1_NM, 16) OUT_UW(1,1)<2> IN_UB(0,1)<4;1,0> 0x4:uw\n"
      "and (M1_NM, 16) OUT_UW(2,0)<2> IN_UB(0,2)<4;1,0> 0xf:uw\n"
      "shr (M1_NM, 16) OUT_UW(2,1)<2> IN_UB(0,2)<4;1,0> 0x4:uw\n"
      "and (M1_NM, 16) OUT_UW(3,0)<2> IN_UB(0,3)<4;1,0> 0xf:uw\n"
      "shr (M1_NM, 16) OUT_UW(3,1)<2> IN_UB(0,3)<4;1,0> 0x4:uw\n"
      "mul (M1_NM, 32) OUT_HF(0,0)<1> OUT_HF(0,0)<1;1,0> 0x6c00:hf\n"
      "mul (M1_NM, 32) OUT_HF(1,0)<1> OUT_HF(1,0)<1;1,0> 0x6c00:hf\n"
      "mul (M1_NM, 32) OUT_HF(2,0)<1> OUT_HF(2,0)<1;1,0> 0x6c00:hf\n"
      "mul (M1_NM, 32) OUT_HF(3,0)<1> OUT_HF(3,0)<1;1,0> 0x6c00:hf\n"
      "mul (M1_NM, 32) OUT_HF(0,0)<1> OUT_HF(0,0)<1;1,0> 0x6c00:hf\n"
      "mul (M1_NM, 32) OUT_HF(1,0)<1> OUT_HF(1,0)<1;1,0> 0x6c00:hf\n"
      "mul (M1_NM, 32) OUT_HF(2,0)<1> OUT_HF(2,0)<1;1,0> 0x6c00:hf\n"
      "mul (M1_NM, 32) OUT_HF(3,0)<1> OUT_HF(3,0)<1;1,0> 0x6c00:hf\n"
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
struct Xe_Reorder<ReorderKind::VV, float_e2m1_t, half_t>
{
  using SRegisters = intel::uchar4[1];
  using DRegisters = intel::ushort8[1];

  CUTE_HOST_DEVICE static void
  reorder(intel::uchar4 const& src0, intel::ushort8& dst0)
  {
#if defined(CUTE_ARCH_COPY_XE_ENABLED)
    const uint32_t shifts = 0x0008000C;
    asm (   /* Latency: 4 cycles/output register */
      "{\n"
      ".decl IN_UB v_type=G type=UB num_elts=64 alias=<%1,0>\n"
      ".decl OUT_UW v_type=G type=UW num_elts=128 alias=<%0,0>\n"
      ".decl OUT_W v_type=G type=W num_elts=128 alias=<%0,0>\n"
      ".decl OUT_UD v_type=G type=UD num_elts=64 alias=<%0,0>\n"
      ".decl OUT_HF v_type=G type=HF num_elts=128 alias=<%0,0>\n"
      ".decl SHIFTS v_type=G type=UW num_elts=2 alias=<%2,0>\n"
      "shl (M1_NM, 32) OUT_UW(0,0)<1> IN_UB(0,0)<4;2,0> SHIFTS(0,0)<0;2,1>\n"
      "shl (M1_NM, 32) OUT_UW(1,0)<1> IN_UB(0,1)<4;2,0> SHIFTS(0,0)<0;2,1>\n"
      "shl (M1_NM, 32) OUT_UW(2,0)<1> IN_UB(0,2)<4;2,0> SHIFTS(0,0)<0;2,1>\n"
      "shl (M1_NM, 32) OUT_UW(3,0)<1> IN_UB(0,3)<4;2,0> SHIFTS(0,0)<0;2,1>\n"
      "asr (M1_NM, 32) OUT_W(0,0)<1> OUT_W(0,0)<1;1,0> 3:uw\n"
      "asr (M1_NM, 32) OUT_W(1,0)<1> OUT_W(1,0)<1;1,0> 3:uw\n"
      "asr (M1_NM, 32) OUT_W(2,0)<1> OUT_W(2,0)<1;1,0> 3:uw\n"
      "asr (M1_NM, 32) OUT_W(3,0)<1> OUT_W(3,0)<1;1,0> 3:uw\n"
      "and (M1_NM, 32) OUT_UD(0,0)<1> OUT_UD(0,0)<1;1,0> 0x8E008E00:ud\n"
      "and (M1_NM, 32) OUT_UD(2,0)<1> OUT_UD(2,0)<1;1,0> 0x8E008E00:ud\n"
      "mul (M1_NM, 32) OUT_HF(0,0)<1> OUT_HF(0,0)<1;1,0> 0x7400:hf\n"
      "mul (M1_NM, 32) OUT_HF(1,0)<1> OUT_HF(1,0)<1;1,0> 0x7400:hf\n"
      "mul (M1_NM, 32) OUT_HF(2,0)<1> OUT_HF(2,0)<1;1,0> 0x7400:hf\n"
      "mul (M1_NM, 32) OUT_HF(3,0)<1> OUT_HF(3,0)<1;1,0> 0x7400:hf\n"
      "}\n"
      : "=rw"(dst0)
      : "rw"(src0), "rw.u"(shifts)
    );
#else
  CUTE_INVALID_CONTROL_PATH("Not Xe");
#endif
  }
};

template <>
struct Xe_Reorder<ReorderKind::VV, float_e2m1_t, bfloat16_t>
{
  using SRegisters = intel::uchar4[1];
  using DRegisters = intel::ushort8[1];

  CUTE_HOST_DEVICE static void
  reorder(intel::uchar4 const& src0, intel::ushort8& dst0)
  {
#if defined(CUTE_ARCH_COPY_XE_ENABLED)
    const uint32_t shifts = 0x0008000C;
    asm (   /* Latency: 5 cycles/output register */
      "{\n"
      ".decl IN_UB v_type=G type=UB num_elts=64 alias=<%1,0>\n"
      ".decl OUT_UW v_type=G type=UW num_elts=128 alias=<%0,0>\n"
      ".decl OUT_W v_type=G type=W num_elts=128 alias=<%0,0>\n"
      ".decl OUT_UD v_type=G type=UD num_elts=64 alias=<%0,0>\n"
      ".decl OUT_BF v_type=G type=BF num_elts=128 alias=<%0,0>\n"
      ".decl SHIFTS v_type=G type=UW num_elts=2 alias=<%2,0>\n"
      "shl (M1_NM, 32) OUT_UW(0,0)<1> IN_UB(0,0)<4;2,0> SHIFTS(0,0)<0;2,1>\n"
      "shl (M1_NM, 32) OUT_UW(1,0)<1> IN_UB(0,1)<4;2,0> SHIFTS(0,0)<0;2,1>\n"
      "shl (M1_NM, 32) OUT_UW(2,0)<1> IN_UB(0,2)<4;2,0> SHIFTS(0,0)<0;2,1>\n"
      "shl (M1_NM, 32) OUT_UW(3,0)<1> IN_UB(0,3)<4;2,0> SHIFTS(0,0)<0;2,1>\n"
      "asr (M1_NM, 32) OUT_W(0,0)<1> OUT_W(0,0)<1;1,0> 6:uw\n"
      "asr (M1_NM, 32) OUT_W(1,0)<1> OUT_W(1,0)<1;1,0> 6:uw\n"
      "asr (M1_NM, 32) OUT_W(2,0)<1> OUT_W(2,0)<1;1,0> 6:uw\n"
      "asr (M1_NM, 32) OUT_W(3,0)<1> OUT_W(3,0)<1;1,0> 6:uw\n"
      "and (M1_NM, 32) OUT_UD(0,0)<1> OUT_UD(0,0)<1;1,0> 0x81C081C0:ud\n"
      "and (M1_NM, 32) OUT_UD(2,0)<1> OUT_UD(2,0)<1;1,0> 0x81C081C0:ud\n"
      "mul (M1_NM, 32) OUT_BF(0,0)<1> OUT_BF(0,0)<1;1,0> 0x7E800000:f\n"
      "mul (M1_NM, 32) OUT_BF(1,0)<1> OUT_BF(1,0)<1;1,0> 0x7E800000:f\n"
      "mul (M1_NM, 32) OUT_BF(2,0)<1> OUT_BF(2,0)<1;1,0> 0x7E800000:f\n"
      "mul (M1_NM, 32) OUT_BF(3,0)<1> OUT_BF(3,0)<1;1,0> 0x7E800000:f\n"
      "}\n"
      : "=rw"(dst0)
      : "rw"(src0), "rw.u"(shifts)
    );
#else
  CUTE_INVALID_CONTROL_PATH("Not Xe");
#endif
  }
};



} // end namespace cute
