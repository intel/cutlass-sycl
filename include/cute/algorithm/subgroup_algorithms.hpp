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

#include "cute/tensor.hpp"
#include "cute/util/sycl_vec.hpp"

namespace cute {

// Uniformize a value, in case the compiler cannot prove it is subgroup-uniform.
template <typename T>
CUTE_HOST_DEVICE
T
assert_uniform(T x) {
  auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
  return group_broadcast(sg, x, 0);
}

// Set a value in a single work-item -- x[i] = val.
// WARNING: i _must_ be a compile-time constant.
//   No diagnostics/error will be issued by the compiler if it is not.
template <typename T>
CUTE_HOST_DEVICE void
set_wi_value(T &x, int i, T val)
{
#if defined(__SYCL_DEVICE_ONLY__) && defined(SYCL_INTEL_TARGET)
  asm (
    "mov (M1_NM, 1) %0(0,%2)<1> %1(0,0)<1;1,0>"
    : "+rw"(x)
    : "rw.u"(val), "P"(i)
  );
#else
  int lane = sycl::ext::oneapi::this_work_item::get_sub_group().get_local_id()[0];
  if (lane == i)
    x = val;
#endif
}

// Set an element of a 1D SG-shared fragment x.
// WARNING: i _must_ be a compile-time constant.
//   No diagnostics/error will be issued by the compiler if it is not.
template <typename FragX>
CUTE_HOST_DEVICE void
set_single_value(FragX& x, int i, typename FragX::element_type val) {
  set_wi_value(x(i / intel::sg_size), i % intel::sg_size, val);
}

// Broadcast the element from a 1D SG-shared fragment x
//   corresponding to the Mode'th dimension of the logical coordinates of src(val).
template <int Mode, typename FragX, typename SGTensorSrc,
          __CUTE_REQUIRES(is_sg_tensor<SGTensorSrc>::value)>
CUTE_HOST_DEVICE
constexpr auto
broadcast(FragX const& x, SGTensorSrc const& src, int val)
{
  auto coord = src.tv_layout()(0, val);
  auto coord_i = get<Mode>(coord);

  constexpr auto TMode = rank(as_arithmetic_tuple(stride<0>(SGTensorSrc{}.tv_layout()))) - 1;
  if constexpr (TMode == Mode) {
    return x(coord_i / intel::sg_size);
  } else {
    auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
    return group_broadcast(sg, x(coord_i / intel::sg_size), coord_i % intel::sg_size);
  }
}
using float16v = intel::vector_t<float, 16>;
using float8v  = intel::vector_t<float, 8>;
#if defined(__SYCL_DEVICE_ONLY__) && defined(SYCL_INTEL_TARGET)
#define DEFINE_HREDUCE16_FLOAT(op) \
  CUTE_DEVICE \
  float \
  hreduce16_float_ ## op(float16v x) \
  { \
    float y; \
    asm ( \
      "{\n" \
      ".decl X16_S v_type=G type=F num_elts=512 align=wordx32\n" \
      ".decl X16_T v_type=G type=F num_elts=256 align=wordx32\n" \
      "mov (M1_NM,16) X16_S(0,0)<1>   %1(0,0)<16;8,1>\n"  \
      "mov (M1_NM,16) X16_S(1,0)<1>   %1(0,8)<16;8,1>\n"  \
      "mov (M1_NM,16) X16_S(2,0)<1>   %1(2,0)<16;8,1>\n"  \
      "mov (M1_NM,16) X16_S(3,0)<1>   %1(2,8)<16;8,1>\n"  \
      "mov (M1_NM,16) X16_S(4,0)<1>   %1(4,0)<16;8,1>\n"  \
      "mov (M1_NM,16) X16_S(5,0)<1>   %1(4,8)<16;8,1>\n"  \
      "mov (M1_NM,16) X16_S(6,0)<1>   %1(6,0)<16;8,1>\n"  \
      "mov (M1_NM,16) X16_S(7,0)<1>   %1(6,8)<16;8,1>\n"  \
      "mov (M1_NM,16) X16_S(8,0)<1>   %1(8,0)<16;8,1>\n"  \
      "mov (M1_NM,16) X16_S(9,0)<1>   %1(8,8)<16;8,1>\n"  \
      "mov (M1_NM,16) X16_S(10,0)<1>  %1(10,0)<16;8,1>\n" \
      "mov (M1_NM,16) X16_S(11,0)<1>  %1(10,8)<16;8,1>\n" \
      "mov (M1_NM,16) X16_S(12,0)<1>  %1(12,0)<16;8,1>\n" \
      "mov (M1_NM,16) X16_S(13,0)<1>  %1(12,8)<16;8,1>\n" \
      "mov (M1_NM,16) X16_S(14,0)<1>  %1(14,0)<16;8,1>\n" \
      "mov (M1_NM,16) X16_S(15,0)<1>  %1(14,8)<16;8,1>\n" \
      #op " (M1_NM,16) X16_T(0,0)<1>  X16_S(0,0)<1;1,0>   X16_S(1,0)<1;1,0>\n"  \
      #op " (M1_NM,16) X16_T(1,0)<1>  X16_S(2,0)<1;1,0>   X16_S(3,0)<1;1,0>\n"  \
      #op " (M1_NM,16) X16_T(2,0)<1>  X16_S(4,0)<1;1,0>   X16_S(5,0)<1;1,0>\n"  \
      #op " (M1_NM,16) X16_T(3,0)<1>  X16_S(6,0)<1;1,0>   X16_S(7,0)<1;1,0>\n"  \
      #op " (M1_NM,16) X16_T(4,0)<1>  X16_S(8,0)<1;1,0>   X16_S(9,0)<1;1,0>\n"  \
      #op " (M1_NM,16) X16_T(5,0)<1>  X16_S(10,0)<1;1,0>  X16_S(11,0)<1;1,0>\n" \
      #op " (M1_NM,16) X16_T(6,0)<1>  X16_S(12,0)<1;1,0>  X16_S(13,0)<1;1,0>\n" \
      #op " (M1_NM,16) X16_T(7,0)<1>  X16_S(14,0)<1;1,0>  X16_S(15,0)<1;1,0>\n" \
      "mov (M1_NM,16) X16_S(16,0)<1>  X16_T(0,0)<8;4,1>\n"    \
      "mov (M1_NM,16) X16_S(17,0)<1>  X16_T(0,4)<8;4,1>\n"    \
      "mov (M1_NM,16) X16_S(18,0)<1>  X16_T(2,0)<8;4,1>\n"    \
      "mov (M1_NM,16) X16_S(19,0)<1>  X16_T(2,4)<8;4,1>\n"    \
      "mov (M1_NM,16) X16_S(20,0)<1>  X16_T(4,0)<8;4,1>\n"    \
      "mov (M1_NM,16) X16_S(21,0)<1>  X16_T(4,4)<8;4,1>\n"    \
      "mov (M1_NM,16) X16_S(22,0)<1>  X16_T(6,0)<8;4,1>\n"    \
      "mov (M1_NM,16) X16_S(23,0)<1>  X16_T(6,4)<8;4,1>\n"    \
      #op " (M1_NM,16) X16_T(8,0)<1>  X16_S(16,0)<1;1,0>  X16_S(17,0)<1;1,0>\n"  \
      #op " (M1_NM,16) X16_T(9,0)<1>  X16_S(18,0)<1;1,0>  X16_S(19,0)<1;1,0>\n"  \
      #op " (M1_NM,16) X16_T(10,0)<1>  X16_S(20,0)<1;1,0>  X16_S(21,0)<1;1,0>\n" \
      #op " (M1_NM,16) X16_T(11,0)<1>  X16_S(22,0)<1;1,0>  X16_S(23,0)<1;1,0>\n" \
      "mov (M1_NM,16) X16_S(24,0)<1>  X16_T(8,0)<4;2,1>\n"    \
      "mov (M1_NM,16) X16_S(25,0)<1>  X16_T(8,2)<4;2,1>\n"    \
      "mov (M1_NM,16) X16_S(26,0)<1>  X16_T(10,0)<4;2,1>\n"   \
      "mov (M1_NM,16) X16_S(27,0)<1>  X16_T(10,2)<4;2,1>\n"   \
      #op " (M1_NM,16) X16_T(12,0)<1>  X16_S(24,0)<1;1,0>  X16_S(25,0)<1;1,0>\n" \
      #op " (M1_NM,16) X16_T(13,0)<1>  X16_S(26,0)<1;1,0>  X16_S(27,0)<1;1,0>\n" \
      "mov (M1_NM,16) X16_S(28,0)<1>  X16_T(12,0)<2;1,0>\n"   \
      "mov (M1_NM,16) X16_S(29,0)<1>  X16_T(12,1)<2;1,0>\n"   \
      #op " (M1_NM,16) %0(0,0)<1> X16_S(28,0)<1;1,0>  X16_S(29,0)<1;1,0>\n"  \
      "}\n" \
      : "=rw"(y) \
      : "rw"(x) \
    ); \
    return y; \
  }

#define DEFINE_HREDUCE8_FLOAT(op) \
  CUTE_DEVICE \
  float \
  hreduce8_float_ ## op(float8v x) \
  { \
    float y; \
    asm ( \
      "{\n" \
      ".decl X8_S v_type=G type=F num_elts=256 align=wordx32\n" \
      ".decl X8_T v_type=G type=F num_elts=128  align=wordx32\n" \
      "mov (M1_NM,16) X8_S(0,0)<1>  %1(0,0)<16;8,1>\n"  \
      "mov (M1_NM,16) X8_S(1,0)<1>  %1(0,8)<16;8,1>\n"  \
      "mov (M1_NM,16) X8_S(2,0)<1>  %1(2,0)<16;8,1>\n"  \
      "mov (M1_NM,16) X8_S(3,0)<1>  %1(2,8)<16;8,1>\n"  \
      "mov (M1_NM,16) X8_S(4,0)<1>  %1(4,0)<16;8,1>\n"  \
      "mov (M1_NM,16) X8_S(5,0)<1>  %1(4,8)<16;8,1>\n"  \
      "mov (M1_NM,16) X8_S(6,0)<1>  %1(6,0)<16;8,1>\n"  \
      "mov (M1_NM,16) X8_S(7,0)<1>  %1(6,8)<16;8,1>\n"  \
      #op " (M1_NM,16) X8_T(0,0)<1>  X8_S(0,0)<1;1,0>  X8_S(1,0)<1;1,0>\n"  \
      #op " (M1_NM,16) X8_T(1,0)<1>  X8_S(2,0)<1;1,0>  X8_S(3,0)<1;1,0>\n"  \
      #op " (M1_NM,16) X8_T(2,0)<1>  X8_S(4,0)<1;1,0>  X8_S(5,0)<1;1,0>\n"  \
      #op " (M1_NM,16) X8_T(3,0)<1>  X8_S(6,0)<1;1,0>  X8_S(7,0)<1;1,0>\n"  \
      "mov (M1_NM,16) X8_S(8,0)<1>  X8_T(0,0)<8;4,1>\n"    \
      "mov (M1_NM,16) X8_S(9,0)<1>  X8_T(0,4)<8;4,1>\n"    \
      "mov (M1_NM,16) X8_S(10,0)<1>  X8_T(2,0)<8;4,1>\n"   \
      "mov (M1_NM,16) X8_S(11,0)<1>  X8_T(2,4)<8;4,1>\n"   \
      #op " (M1_NM,16) X8_T(4,0)<1>  X8_S(8,0)<1;1,0>   X8_S(9,0)<1;1,0>\n"  \
      #op " (M1_NM,16) X8_T(5,0)<1>  X8_S(10,0)<1;1,0>  X8_S(11,0)<1;1,0>\n" \
      "mov (M1_NM,16) X8_S(12,0)<1>  X8_T(4,0)<4;2,1>\n"   \
      "mov (M1_NM,16) X8_S(13,0)<1>  X8_T(4,2)<4;2,1>\n"   \
      #op " (M1_NM,16) X8_T(6,0)<1>  X8_S(12,0)<1;1,0>  X8_S(13,0)<1;1,0>\n"  \
      "mov (M1_NM,8)  X8_S(14,0)<1>  X8_T(6,0)<2;1,0>\n"   \
      "mov (M1_NM,8)  X8_S(15,0)<1>  X8_T(6,1)<2;1,0>\n"   \
      #op " (M1_NM,8)  %0(0,0)<1> X8_S(14,0)<1;1,0>  X8_S(15,0)<1;1,0>\n"  \
      "}\n" \
      : "=rw"(y) \
      : "rw"(x) \
    ); \
    return y; \
  }

#define DEFINE_REDUCE64_FLOAT(op) \
  CUTE_DEVICE \
  float \
  reduce64_float_ ## op(float8v s0, float8v s1, float8v s2, float8v s3, \
                        float8v s4, float8v s5, float8v s6, float8v s7) \
  { \
    float y; \
    asm ( \
      "{\n" \
      ".decl S0 v_type=G type=F num_elts=128 alias=<%1, 0>\n" \
      ".decl S1 v_type=G type=F num_elts=128 alias=<%2, 0>\n" \
      ".decl S2 v_type=G type=F num_elts=128 alias=<%3, 0>\n" \
      ".decl S3 v_type=G type=F num_elts=128 alias=<%4, 0>\n" \
      ".decl S4 v_type=G type=F num_elts=128 alias=<%5, 0>\n" \
      ".decl S5 v_type=G type=F num_elts=128 alias=<%6, 0>\n" \
      ".decl S6 v_type=G type=F num_elts=128 alias=<%7, 0>\n" \
      ".decl S7 v_type=G type=F num_elts=128 alias=<%8, 0>\n" \
      ".decl ACC_F v_type=G type=F num_elts=256 align=wordx32\n" \
      ".decl T_A   v_type=G type=F num_elts=128 align=wordx32\n" \
      ".decl T_B   v_type=G type=F num_elts=64  align=wordx32\n" \
      ".decl T_C   v_type=G type=F num_elts=32  align=wordx32\n" \
      ".decl SM_A_LO v_type=G type=F num_elts=128 align=wordx32\n" \
      ".decl SM_A_HI v_type=G type=F num_elts=128 align=wordx32\n" \
      ".decl SM_B_LO v_type=G type=F num_elts=64  align=wordx32\n" \
      ".decl SM_B_HI v_type=G type=F num_elts=64  align=wordx32\n" \
      ".decl SM_C_LO v_type=G type=F num_elts=32  align=wordx32\n" \
      ".decl SM_C_HI v_type=G type=F num_elts=32  align=wordx32\n" \
      ".decl SM_D_LO v_type=G type=F num_elts=16  align=wordx32\n" \
      ".decl SM_D_HI v_type=G type=F num_elts=16  align=wordx32\n" \
      ".decl OUT_F v_type=G type=F num_elts=16  alias=<%0, 0>\n" \
      #op " (M1_NM,32) ACC_F(0, 0)<1>   S0(0,0)<1;1,0>       S0(2,0)<1;1,0>\n" \
      #op " (M1_NM,32) ACC_F(0, 0)<1>   ACC_F(0, 0)<1;1,0>   S0(4,0)<1;1,0>\n" \
      #op " (M1_NM,32) ACC_F(2, 0)<1>   S1(0,0)<1;1,0>       S1(2,0)<1;1,0>\n" \
      #op " (M1_NM,32) ACC_F(2, 0)<1>   ACC_F(2, 0)<1;1,0>   S1(4,0)<1;1,0>\n" \
      #op " (M1_NM,32) ACC_F(4, 0)<1>   S2(0,0)<1;1,0>       S2(2,0)<1;1,0>\n" \
      #op " (M1_NM,32) ACC_F(4, 0)<1>   ACC_F(4, 0)<1;1,0>   S2(4,0)<1;1,0>\n" \
      #op " (M1_NM,32) ACC_F(6, 0)<1>   S3(0,0)<1;1,0>       S3(2,0)<1;1,0>\n" \
      #op " (M1_NM,32) ACC_F(6, 0)<1>   ACC_F(6, 0)<1;1,0>   S3(4,0)<1;1,0>\n" \
      #op " (M1_NM,32) ACC_F(8, 0)<1>   S4(0,0)<1;1,0>       S4(2,0)<1;1,0>\n" \
      #op " (M1_NM,32) ACC_F(8, 0)<1>   ACC_F(8, 0)<1;1,0>   S4(4,0)<1;1,0>\n" \
      #op " (M1_NM,32) ACC_F(10,0)<1>   S5(0,0)<1;1,0>       S5(2,0)<1;1,0>\n" \
      #op " (M1_NM,32) ACC_F(10,0)<1>   ACC_F(10,0)<1;1,0>   S5(4,0)<1;1,0>\n" \
      #op " (M1_NM,32) ACC_F(12,0)<1>   S6(0,0)<1;1,0>       S6(2,0)<1;1,0>\n" \
      #op " (M1_NM,32) ACC_F(12,0)<1>   ACC_F(12,0)<1;1,0>   S6(4,0)<1;1,0>\n" \
      #op " (M1_NM,32) ACC_F(14,0)<1>   S7(0,0)<1;1,0>       S7(2,0)<1;1,0>\n" \
      #op " (M1_NM,32) ACC_F(14,0)<1>   ACC_F(14,0)<1;1,0>   S7(4,0)<1;1,0>\n" \
      #op " (M1_NM,32) ACC_F(0, 0)<1>   ACC_F(0, 0)<1;1,0>   S0(6,0)<1;1,0>\n" \
      #op " (M1_NM,32) ACC_F(2, 0)<1>   ACC_F(2, 0)<1;1,0>   S1(6,0)<1;1,0>\n" \
      #op " (M1_NM,32) ACC_F(4, 0)<1>   ACC_F(4, 0)<1;1,0>   S2(6,0)<1;1,0>\n" \
      #op " (M1_NM,32) ACC_F(6, 0)<1>   ACC_F(6, 0)<1;1,0>   S3(6,0)<1;1,0>\n" \
      #op " (M1_NM,32) ACC_F(8, 0)<1>   ACC_F(8, 0)<1;1,0>   S4(6,0)<1;1,0>\n" \
      #op " (M1_NM,32) ACC_F(10,0)<1>   ACC_F(10,0)<1;1,0>   S5(6,0)<1;1,0>\n" \
      #op " (M1_NM,32) ACC_F(12,0)<1>   ACC_F(12,0)<1;1,0>   S6(6,0)<1;1,0>\n" \
      #op " (M1_NM,32) ACC_F(14,0)<1>   ACC_F(14,0)<1;1,0>   S7(6,0)<1;1,0>\n" \
      "mov (M1_NM,16) SM_A_LO(0,0)<1>   ACC_F(0, 0)<16;8,1>\n" \
      "mov (M1_NM,16) SM_A_LO(1,0)<1>   ACC_F(2, 0)<16;8,1>\n" \
      "mov (M1_NM,16) SM_A_LO(2,0)<1>   ACC_F(4, 0)<16;8,1>\n" \
      "mov (M1_NM,16) SM_A_LO(3,0)<1>   ACC_F(6, 0)<16;8,1>\n" \
      "mov (M1_NM,16) SM_A_LO(4,0)<1>   ACC_F(8, 0)<16;8,1>\n" \
      "mov (M1_NM,16) SM_A_LO(5,0)<1>   ACC_F(10,0)<16;8,1>\n" \
      "mov (M1_NM,16) SM_A_LO(6,0)<1>   ACC_F(12,0)<16;8,1>\n" \
      "mov (M1_NM,16) SM_A_LO(7,0)<1>   ACC_F(14,0)<16;8,1>\n" \
      "mov (M1_NM,16) SM_A_HI(0,0)<1>   ACC_F(0, 8)<16;8,1>\n" \
      "mov (M1_NM,16) SM_A_HI(1,0)<1>   ACC_F(2, 8)<16;8,1>\n" \
      "mov (M1_NM,16) SM_A_HI(2,0)<1>   ACC_F(4, 8)<16;8,1>\n" \
      "mov (M1_NM,16) SM_A_HI(3,0)<1>   ACC_F(6, 8)<16;8,1>\n" \
      "mov (M1_NM,16) SM_A_HI(4,0)<1>   ACC_F(8, 8)<16;8,1>\n" \
      "mov (M1_NM,16) SM_A_HI(5,0)<1>   ACC_F(10,8)<16;8,1>\n" \
      "mov (M1_NM,16) SM_A_HI(6,0)<1>   ACC_F(12,8)<16;8,1>\n" \
      "mov (M1_NM,16) SM_A_HI(7,0)<1>   ACC_F(14,8)<16;8,1>\n" \
      "fence_sw\n" \
      #op " (M1_NM,16) T_A(0,0)<1>   SM_A_LO(0,0)<1;1,0>  SM_A_HI(0,0)<1;1,0>\n" \
      #op " (M1_NM,16) T_A(1,0)<1>   SM_A_LO(1,0)<1;1,0>  SM_A_HI(1,0)<1;1,0>\n" \
      #op " (M1_NM,16) T_A(2,0)<1>   SM_A_LO(2,0)<1;1,0>  SM_A_HI(2,0)<1;1,0>\n" \
      #op " (M1_NM,16) T_A(3,0)<1>   SM_A_LO(3,0)<1;1,0>  SM_A_HI(3,0)<1;1,0>\n" \
      #op " (M1_NM,16) T_A(4,0)<1>   SM_A_LO(4,0)<1;1,0>  SM_A_HI(4,0)<1;1,0>\n" \
      #op " (M1_NM,16) T_A(5,0)<1>   SM_A_LO(5,0)<1;1,0>  SM_A_HI(5,0)<1;1,0>\n" \
      #op " (M1_NM,16) T_A(6,0)<1>   SM_A_LO(6,0)<1;1,0>  SM_A_HI(6,0)<1;1,0>\n" \
      #op " (M1_NM,16) T_A(7,0)<1>   SM_A_LO(7,0)<1;1,0>  SM_A_HI(7,0)<1;1,0>\n" \
      "mov (M1_NM,16) SM_B_LO(0,0)<1>  T_A(0,0)<8;4,1>\n" \
      "mov (M1_NM,16) SM_B_LO(1,0)<1>  T_A(2,0)<8;4,1>\n" \
      "mov (M1_NM,16) SM_B_LO(2,0)<1>  T_A(4,0)<8;4,1>\n" \
      "mov (M1_NM,16) SM_B_LO(3,0)<1>  T_A(6,0)<8;4,1>\n" \
      "mov (M1_NM,16) SM_B_HI(0,0)<1>  T_A(0,4)<8;4,1>\n" \
      "mov (M1_NM,16) SM_B_HI(1,0)<1>  T_A(2,4)<8;4,1>\n" \
      "mov (M1_NM,16) SM_B_HI(2,0)<1>  T_A(4,4)<8;4,1>\n" \
      "mov (M1_NM,16) SM_B_HI(3,0)<1>  T_A(6,4)<8;4,1>\n" \
      "fence_sw\n" \
      #op " (M1_NM,16) T_B(0,0)<1>   SM_B_LO(0,0)<1;1,0>  SM_B_HI(0,0)<1;1,0>\n" \
      #op " (M1_NM,16) T_B(1,0)<1>   SM_B_LO(1,0)<1;1,0>  SM_B_HI(1,0)<1;1,0>\n" \
      #op " (M1_NM,16) T_B(2,0)<1>   SM_B_LO(2,0)<1;1,0>  SM_B_HI(2,0)<1;1,0>\n" \
      #op " (M1_NM,16) T_B(3,0)<1>   SM_B_LO(3,0)<1;1,0>  SM_B_HI(3,0)<1;1,0>\n" \
      "mov (M1_NM,16) SM_C_LO(0,0)<1>  T_B(0,0)<4;2,1>\n" \
      "mov (M1_NM,16) SM_C_LO(1,0)<1>  T_B(2,0)<4;2,1>\n" \
      "mov (M1_NM,16) SM_C_HI(0,0)<1>  T_B(0,2)<4;2,1>\n" \
      "mov (M1_NM,16) SM_C_HI(1,0)<1>  T_B(2,2)<4;2,1>\n" \
      "fence_sw\n" \
      #op " (M1_NM,16) T_C(0,0)<1>   SM_C_LO(0,0)<1;1,0>  SM_C_HI(0,0)<1;1,0>\n" \
      #op " (M1_NM,16) T_C(1,0)<1>   SM_C_LO(1,0)<1;1,0>  SM_C_HI(1,0)<1;1,0>\n" \
      "mov (M1_NM,16) SM_D_LO(0,0)<1>  T_C(0,0)<2;1,0>\n" \
      "mov (M1_NM,16) SM_D_HI(0,0)<1>  T_C(0,1)<2;1,0>\n" \
      #op " (M1_NM,16) OUT_F(0,0)<1>   SM_D_LO(0,0)<1;1,0>  SM_D_HI(0,0)<1;1,0>\n" \
      "}\n" \
      : "=rw"(y) \
      : "rw"(s0), "rw"(s1), "rw"(s2), "rw"(s3), \
        "rw"(s4), "rw"(s5), "rw"(s6), "rw"(s7) \
    ); \
    return y; \
  }
#else
#define DEFINE_HREDUCE16_FLOAT(op) \
  CUTE_DEVICE float hreduce16_float_ ## op(float16v x) { return 0.f; }
#define DEFINE_HREDUCE8_FLOAT(op) \
  CUTE_DEVICE float hreduce8_float_ ## op(float8v x) { return 0.f; }
#define DEFINE_REDUCE64_FLOAT(op) \
  CUTE_DEVICE float reduce64_float_ ## op(float8v, float8v, float8v, float8v, \
                                          float8v, float8v, float8v, float8v) \
  { return 0.f; }
#endif

DEFINE_HREDUCE8_FLOAT(add)
DEFINE_HREDUCE8_FLOAT(max)
DEFINE_HREDUCE16_FLOAT(add)
DEFINE_HREDUCE16_FLOAT(max)
DEFINE_REDUCE64_FLOAT(add)
DEFINE_REDUCE64_FLOAT(max)

enum class ReduceMode { Full, Vertical, Horizontal };
template <int Mode = 0, ReduceMode RMode = ReduceMode::Full, bool EnableFast64Rows = true, class BinaryOp,
          class Engine, class FragLayout, class SubgroupTVLayout>
CUTE_HOST_DEVICE
auto
reduce(SubgroupTensor<Engine, FragLayout, SubgroupTVLayout> const& src, BinaryOp op)
{
  using SrcTensor = SubgroupTensor<Engine, FragLayout, SubgroupTVLayout>;
  using T = typename SrcTensor::value_type;

  if constexpr (RMode == ReduceMode::Horizontal) {
    auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
    constexpr int N = decltype(size(SrcTensor{}.layout()))::value;
    constexpr int M = (N + intel::sg_size - 1) / intel::sg_size;

    auto out = make_subgroup_tensor(make_tensor<T>(Int<M>{}),
                                    make_identity_layout(Shape<Int<N>>{}));

    constexpr bool is_float = is_same_v<T, float>;
    constexpr bool is_add = is_same_v<BinaryOp, sycl::plus<void>>;
    constexpr bool is_max = is_same_v<BinaryOp, sycl::maximum<void>>;
    constexpr bool align16 = (N % 16 == 0);
    constexpr bool is_8 = (N == 8);

    if constexpr (is_float && (is_add || is_max) && (align16 || is_8)) {
      if constexpr (align16) {
        CUTE_UNROLL
        for (int j = 0; j < N; j += 16) {
          float16v v;
          CUTE_UNROLL
          for (int k = 0; k < 16; k++) v[k] = src(j + k);
          if constexpr (is_add)
            out(j/16) = hreduce16_float_add(v);
          else
            out(j/16) = hreduce16_float_max(v);
        }
      } else { // is_8
        float8v v;
        CUTE_UNROLL
        for (int k = 0; k < 8; k++) v[k] = src(k);
        if constexpr (is_add)
          out(0) = hreduce8_float_add(v);
        else
          out(0) = hreduce8_float_max(v);
      }
    } else {
      CUTE_UNROLL
      for (int r = 0; r < N; r++) {
        T val = reduce_over_group(sg, src(r), op);
        set_single_value(out, r, val);
      }
    }

    return out;

  } else {
    auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
    using TVToV = Layout<Shape<intel::_SGSize,int>, Stride<_0,_1>>;

    constexpr auto tv_layout = SrcTensor{}.tv_layout();
    constexpr auto shape = atuple_coshape(tv_layout);
    constexpr auto coord_to_tv = right_inverse(project_strides(tv_layout)).with_shape(shape);

    constexpr auto rcoord_to_tv = make_layout(select<Mode>(coord_to_tv), remove<Mode>(coord_to_tv));
    constexpr auto rcoord_to_v = filter(composition(TVToV{}, rcoord_to_tv), Step<_1,_1>{});

    Tensor src_r = make_tensor(src.data(), rcoord_to_v);

    if constexpr (RMode == ReduceMode::Vertical) {
      auto rshape = replace<Mode>(shape, _1{});
      auto out = make_subgroup_tensor(make_tensor<T>(Int<size<1>(rcoord_to_v)>{}),
                                      remove<Mode>(make_identity_layout(rshape)));

      CUTE_UNROLL
      for (int j = 0; j < size<1>(rcoord_to_v); j++) {
        out(j) = src_r(0, j);
      }
      CUTE_UNROLL
      for (int i = 1; i < size<0>(rcoord_to_v); i++) {
        CUTE_UNROLL
        for (int j = 0; j < size<1>(rcoord_to_v); j++) {
          if constexpr (is_same_v<BinaryOp, sycl::maximum<void>>)
            out(j) = sycl::max(out(j), src_r(i, j));
          else
            out(j) = op(out(j), src_r(i, j));
        }
      }
      return out;

    } else {
      auto rshape = replace<Mode>(shape, _1{});
      auto out = make_subgroup_tensor(make_tensor<T>(ceil_div(size(rshape), intel::_SGSize{})),
                                      remove<Mode>(make_identity_layout(rshape)));

      constexpr bool horizontal = (size<0>(rcoord_to_tv) == intel::_SGSize{} * size<0>(rcoord_to_v));
      constexpr bool vertical   = (size<1>(rcoord_to_tv) == intel::_SGSize{} * size<1>(rcoord_to_v));

      constexpr bool align16 = is_constant_v<0, decltype(size<1>(rcoord_to_v) % _16{})>;
      constexpr bool align8  = is_constant_v<8, decltype(size<1>(rcoord_to_v))>;

      constexpr bool hadd = (horizontal && is_same_v<T, float> && is_same_v<BinaryOp, sycl::plus<void>>);
      constexpr bool hmax = (horizontal && is_same_v<T, float> && is_same_v<BinaryOp, sycl::maximum<void>>);

      constexpr bool hadd16 = hadd && align16;
      constexpr bool hmax16 = hmax && align16;
      constexpr bool hadd8 = hadd && align8;
      constexpr bool hmax8 = hmax && align8;
      //TODO: comment the align64 optmization as it has acc issue on hdim64
      constexpr bool align64 = false;
        // is_constant_v<4,  decltype(size<0>(rcoord_to_v))> &&
        // is_constant_v<16, decltype(size<1>(rcoord_to_v))>;
      constexpr bool hadd64 = EnableFast64Rows && hadd && align64;
      constexpr bool hmax64 = EnableFast64Rows && hmax && align64;
      if constexpr (hadd64 || hmax64) {
        auto const* p = reinterpret_cast<float const*>(&*src.data());
        auto const& s0 = *reinterpret_cast<float8v const*>(p +  0);
        auto const& s1 = *reinterpret_cast<float8v const*>(p +  8);
        auto const& s2 = *reinterpret_cast<float8v const*>(p + 16);
        auto const& s3 = *reinterpret_cast<float8v const*>(p + 24);
        auto const& s4 = *reinterpret_cast<float8v const*>(p + 32);
        auto const& s5 = *reinterpret_cast<float8v const*>(p + 40);
        auto const& s6 = *reinterpret_cast<float8v const*>(p + 48);
        auto const& s7 = *reinterpret_cast<float8v const*>(p + 56);
        if constexpr (hmax64) out(0) = reduce64_float_max(s0, s1, s2, s3, s4, s5, s6, s7);
        else                  out(0) = reduce64_float_add(s0, s1, s2, s3, s4, s5, s6, s7);
        return out;
      } else {

      [[maybe_unused]] T temp[size<1>(rcoord_to_v)];

      CUTE_UNROLL
      for (int j = 0; j < size<1>(rcoord_to_v); j++) {
        temp[j] = src_r(0, j);
      }
      CUTE_UNROLL
      for (int i = 1; i < size<0>(rcoord_to_v); i++) {
        CUTE_UNROLL
        for (int j = 0; j < size<1>(rcoord_to_v); j++) {
          if constexpr (is_same_v<BinaryOp, sycl::maximum<void>>)
            temp[j] = sycl::max(temp[j], src_r(i, j));
          else
            temp[j] = op(temp[j], src_r(i, j));
        }
      }

      if constexpr (!(hadd16 || hmax16 || hadd8 || hmax8)) {
        CUTE_UNROLL
        for (int j = 0; j < size<1>(rcoord_to_v); j++) {
          if constexpr (horizontal)
            set_single_value(out, j, reduce_over_group(sg, temp[j], op));
          else if constexpr (vertical)
            out(j) = temp[j];
          else
            static_assert(dependent_false<BinaryOp>, "Unimplemented reduction type");
        }
      }
      if constexpr (hadd16 || hmax16) {
        CUTE_UNROLL
        for (int j = 0; j < size<1>(rcoord_to_v); j += 16) {
          float16v v;
          CUTE_UNROLL
          for (int k = 0; k < 16; k++) v[k] = temp[j + k];
          if constexpr (hadd16) out(j/16) = hreduce16_float_add(v);
          else                  out(j/16) = hreduce16_float_max(v);
        }
      } else if constexpr (hadd8 || hmax8) {
        float8v v;
        CUTE_UNROLL
        for (int k = 0; k < 8; k++) v[k] = temp[k];
        if constexpr (hadd8) out(0) = hreduce8_float_add(v);
        else                 out(0) = hreduce8_float_max(v);
      }

      return out;
      }
    }
  }
}

} // namespace cute
