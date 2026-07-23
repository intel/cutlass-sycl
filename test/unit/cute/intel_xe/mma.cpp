/***************************************************************************************************
 * Copyright (c) 2024 - 2024 Codeplay Software Ltd. All rights reserved.
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

#include "cutlass_unit_test.h"

#include <cute/tensor.hpp>
#include <cute/arch/mma_xe.hpp>
#include <cute/util/compat.hpp>
#include <sycl/sycl.hpp>

#include "../cooperative_gemm_common.hpp"

using namespace cute;
using namespace cutlass;
using namespace compat::experimental;

namespace {
  constexpr uint32_t thread_block_size = 128;
  constexpr uint32_t max_vec_bits = 128;
}

template<typename MMAAtom, typename TA, typename TB, typename TC,
         typename ShapeMNK, typename LayoutShape>
void run_mma_test(ShapeMNK shape_mnk, LayoutShape layout_shape) {
  auto tiled_mma = TiledMMA<MMA_Atom<MMAAtom>, Layout<LayoutShape>>{};
  test_cooperative_gemm_col_major_layout<thread_block_size, max_vec_bits, TA, TB, TC>(
    shape_mnk, tiled_mma);
}

TEST(PVC_CuTe_Xe, MMA_XE_8x16x32_S32S8S8S32_TT) {
  run_mma_test<XE_8x16x32_S32S8S8S32_TT, int8_t, int8_t, int32_t>(
    Shape<_128, _128, _16>{}, Shape<_2, _2, _1>{});
}

TEST(PVC_CuTe_Xe, MMA_XE_4x16x32_S32S8S8S32_TT) {
  run_mma_test<XE_4x16x32_S32S8S8S32_TT, int8_t, int8_t, int32_t>(
    Shape<_128, _128, _16>{}, Shape<_2, _2, _1>{});
}

TEST(PVC_CuTe_Xe, MMA_XE_2x16x32_S32S8S8S32_TT) {
  run_mma_test<XE_2x16x32_S32S8S8S32_TT, int8_t, int8_t, int32_t>(
    Shape<_128, _128, _16>{}, Shape<_4, _2, _1>{});
}

TEST(PVC_CuTe_Xe, MMA_XE_1x16x32_S32S8S8S32_TT) {
  run_mma_test<XE_1x16x32_S32S8S8S32_TT, int8_t, int8_t, int32_t>(
    Shape<_128, _128, _16>{}, Shape<_1, _1, _1>{});
}

TEST(PVC_CuTe_Xe, MMA_XE_8x16x32_S32U8U8S32_TT) {
  run_mma_test<XE_8x16x32_S32U8U8S32_TT, uint8_t, uint8_t, int32_t>(
    Shape<_128, _128, _16>{}, Shape<_2, _2, _1>{});
}

TEST(PVC_CuTe_Xe, MMA_XE_4x16x32_S32U8U8S32_TT) {
  run_mma_test<XE_4x16x32_S32U8U8S32_TT, uint8_t, uint8_t, int32_t>(
    Shape<_128, _128, _16>{}, Shape<_2, _2, _1>{});
}

TEST(PVC_CuTe_Xe, MMA_XE_2x16x32_S32U8U8S32_TT) {
  run_mma_test<XE_2x16x32_S32U8U8S32_TT, uint8_t, uint8_t, int32_t>(
    Shape<_128, _128, _16>{}, Shape<_4, _2, _1>{});
}

TEST(PVC_CuTe_Xe, MMA_XE_1x16x32_S32U8U8S32_TT) {
  run_mma_test<XE_1x16x32_S32U8U8S32_TT, uint8_t, uint8_t, int32_t>(
    Shape<_128, _128, _16>{}, Shape<_1, _1, _1>{});
}

// TODO: This case will fail when export IGC_ExtraOCLOptions="-cl-intel-512-GRF-per-thread" 
// on CRI, so we temporarily disable it here, it will be enabled again when the 
// issue is resolved.
// TEST(PVC_CuTe_Xe, MMA_XE_8x16x16_F32BF16BF16F32_TT) {
//   MMA_Test<XE_8x16x16_F32BF16BF16F32_TT, 256, 256, 32, 64, 32, bfloat16_t,
//            bfloat16_t, float>(512, 512, 256);
// }

TEST(PVC_CuTe_Xe, MMA_XE_8x16x16_F32BF16BF16F32_TT) {
  run_mma_test<XE_8x16x16_F32BF16BF16F32_TT, 
               cutlass::bfloat16_t, cutlass::bfloat16_t, float>(
    Shape<_128, _128, _16>{}, Shape<_2, _2, _1>{});
}

TEST(PVC_CuTe_Xe, MMA_XE_4x16x16_F32BF16BF16F32_TT) {
  run_mma_test<XE_4x16x16_F32BF16BF16F32_TT, 
               cutlass::bfloat16_t, cutlass::bfloat16_t, float>(
    Shape<_128, _128, _16>{}, Shape<_2, _2, _1>{});
}

TEST(PVC_CuTe_Xe, MMA_XE_2x16x16_F32BF16BF16F32_TT) {
  run_mma_test<XE_2x16x16_F32BF16BF16F32_TT, 
               cutlass::bfloat16_t, cutlass::bfloat16_t, float>(
    Shape<_128, _128, _16>{}, Shape<_2, _4, _1>{});
}

TEST(PVC_CuTe_Xe, MMA_XE_1x16x16_F32BF16BF16F32_TT) {
  run_mma_test<XE_1x16x16_F32BF16BF16F32_TT, 
               cutlass::bfloat16_t, cutlass::bfloat16_t, float>(
    Shape<_128, _128, _16>{}, Shape<_1, _1, _1>{});
}

TEST(PVC_CuTe_Xe, MMA_XE_8x16x16_F32F16F16F32_TT) {
  run_mma_test<XE_8x16x16_F32F16F16F32_TT, 
               cutlass::half_t, cutlass::half_t, float>(
    Shape<_128, _128, _16>{}, Shape<_2, _2, _1>{});
}

TEST(PVC_CuTe_Xe, MMA_XE_4x16x16_F32F16F16F32_TT) {
  run_mma_test<XE_4x16x16_F32F16F16F32_TT, 
               cutlass::half_t, cutlass::half_t, float>(
    Shape<_128, _128, _16>{}, Shape<_2, _2, _1>{});
}

TEST(PVC_CuTe_Xe, MMA_XE_2x16x16_F32F16F16F32_TT) {
  run_mma_test<XE_2x16x16_F32F16F16F32_TT, 
               cutlass::half_t, cutlass::half_t, float>(
    Shape<_128, _128, _16>{}, Shape<_4, _2, _1>{});
}

TEST(PVC_CuTe_Xe, MMA_XE_1x16x16_F32F16F16F32_TT) {
  run_mma_test<XE_1x16x16_F32F16F16F32_TT, 
               cutlass::half_t, cutlass::half_t, float>(
    Shape<_128, _128, _16>{}, Shape<_1, _1, _1>{});
}

TEST(PVC_CuTe_Xe, MMA_XE_8x16x8_F32TF32TF32F32_TT) {
  run_mma_test<XE_8x16x8_F32TF32TF32F32_TT, 
               cutlass::tfloat32_t, cutlass::tfloat32_t, float>(
    Shape<_128, _128, _16>{}, Shape<_2, _2, _1>{});
}

TEST(PVC_CuTe_Xe, MMA_XE_4x16x8_F32TF32TF32F32_TT) {
  run_mma_test<XE_4x16x8_F32TF32TF32F32_TT, 
               cutlass::tfloat32_t, cutlass::tfloat32_t, float>(
    Shape<_128, _128, _16>{}, Shape<_2, _2, _1>{});
}

TEST(PVC_CuTe_Xe, MMA_XE_2x16x8_F32TF32TF32F32_TT) {
  run_mma_test<XE_2x16x8_F32TF32TF32F32_TT, 
               cutlass::tfloat32_t, cutlass::tfloat32_t, float>(
    Shape<_128, _128, _16>{}, Shape<_4, _2, _1>{});
}

TEST(PVC_CuTe_Xe, MMA_XE_1x16x8_F32TF32TF32F32_TT) {
  run_mma_test<XE_1x16x8_F32TF32TF32F32_TT, 
               cutlass::tfloat32_t, cutlass::tfloat32_t, float>(
    Shape<_128, _128, _16>{}, Shape<_1, _1, _1>{});
}

TEST(PVC_CuTe_Xe, FMA_XE_UniversalFMA_F32F32F32F32) {
  run_mma_test<UniversalFMA<float, float, float, float>, float, float, float>(
    Shape<_64, _64, _16>{}, Shape<_1, _1, _1>{});
}

#if (IGC_VERSION_MAJOR > 2) || (IGC_VERSION_MAJOR == 2 && IGC_VERSION_MINOR >= 18)

TEST(PVC_CuTe_Xe, MMA_DPAS_S8_8x16) {
  run_mma_test<XE_DPAS_TT<8, int32_t, int8_t>, int8_t, int8_t, int32_t>(
    Shape<_128, _128, _16>{}, Shape<_2, _2, _1>{});
}

TEST(PVC_CuTe_Xe, MMA_DPAS_S8_4x16) {
  run_mma_test<XE_DPAS_TT<4, int32_t, int8_t>, int8_t, int8_t, int32_t>(
    Shape<_128, _128, _16>{}, Shape<_2, _2, _1>{});
}

TEST(PVC_CuTe_Xe, MMA_DPAS_S8_2x16) {
  run_mma_test<XE_DPAS_TT<2, int32_t, int8_t>, int8_t, int8_t, int32_t>(
    Shape<_128, _128, _16>{}, Shape<_4, _2, _1>{});
}

TEST(PVC_CuTe_Xe, MMA_DPAS_S8_1x16) {
  run_mma_test<XE_DPAS_TT<1, int32_t, int8_t>, int8_t, int8_t, int32_t>(
    Shape<_128, _128, _16>{}, Shape<_1, _1, _1>{});
}

TEST(PVC_CuTe_Xe, MMA_DPAS_U8_8x16) {
  run_mma_test<XE_DPAS_TT<8, int32_t, uint8_t>, uint8_t, uint8_t, int32_t>(
    Shape<_128, _128, _16>{}, Shape<_2, _2, _1>{});
}

TEST(PVC_CuTe_Xe, MMA_DPAS_U8_4x16) {
  run_mma_test<XE_DPAS_TT<4, int32_t, uint8_t>, uint8_t, uint8_t, int32_t>(
    Shape<_128, _128, _16>{}, Shape<_2, _2, _1>{});
}

TEST(PVC_CuTe_Xe, MMA_DPAS_U8_2x16) {
  run_mma_test<XE_DPAS_TT<2, int32_t, uint8_t>, uint8_t, uint8_t, int32_t>(
    Shape<_128, _128, _16>{}, Shape<_4, _2, _1>{});
}

TEST(PVC_CuTe_Xe, MMA_DPAS_U8_1x16) {
  run_mma_test<XE_DPAS_TT<1, int32_t, uint8_t>, uint8_t, uint8_t, int32_t>(
    Shape<_128, _128, _16>{}, Shape<_1, _1, _1>{});
}

TEST(PVC_CuTe_Xe, MMA_DPAS_BF16_8x16) {
  run_mma_test<XE_DPAS_TT<8, float, cutlass::bfloat16_t>, 
               cutlass::bfloat16_t, cutlass::bfloat16_t, float>(
    Shape<_128, _128, _16>{}, Shape<_2, _2, _1>{});
}

TEST(PVC_CuTe_Xe, MMA_DPAS_BF16_4x16) {
  run_mma_test<XE_DPAS_TT<4, float, cutlass::bfloat16_t>, 
               cutlass::bfloat16_t, cutlass::bfloat16_t, float>(
    Shape<_128, _128, _16>{}, Shape<_2, _2, _1>{});
}

TEST(PVC_CuTe_Xe, MMA_DPAS_BF16_2x16) {
  run_mma_test<XE_DPAS_TT<2, float, cutlass::bfloat16_t>, 
               cutlass::bfloat16_t, cutlass::bfloat16_t, float>(
    Shape<_128, _128, _16>{}, Shape<_2, _4, _1>{});
}

TEST(PVC_CuTe_Xe, MMA_DPAS_BF16_1x16) {
  run_mma_test<XE_DPAS_TT<1, float, cutlass::bfloat16_t>, 
               cutlass::bfloat16_t, cutlass::bfloat16_t, float>(
    Shape<_128, _128, _32>{}, Shape<_1, _1, _1>{});
}

TEST(PVC_CuTe_Xe, MMA_DPAS_F16_8x16) {
  run_mma_test<XE_DPAS_TT<8, float, cutlass::half_t>, 
               cutlass::half_t, cutlass::half_t, float>(
    Shape<_128, _128, _16>{}, Shape<_2, _2, _1>{});
}

TEST(PVC_CuTe_Xe, MMA_DPAS_F16_4x16) {
  run_mma_test<XE_DPAS_TT<4, float, cutlass::half_t>, 
               cutlass::half_t, cutlass::half_t, float>(
    Shape<_128, _128, _16>{}, Shape<_2, _2, _1>{});
}

TEST(PVC_CuTe_Xe, MMA_DPAS_F16_2x16) {
  run_mma_test<XE_DPAS_TT<2, float, cutlass::half_t>, 
               cutlass::half_t, cutlass::half_t, float>(
    Shape<_128, _128, _16>{}, Shape<_4, _2, _1>{});
}

TEST(PVC_CuTe_Xe, MMA_DPAS_F16_1x16) {
  run_mma_test<XE_DPAS_TT<1, float, cutlass::half_t>, 
               cutlass::half_t, cutlass::half_t, float>(
    Shape<_128, _128, _16>{}, Shape<_1, _1, _1>{});
}

TEST(PVC_CuTe_Xe, MMA_DPAS_TF32_8x16) {
  run_mma_test<XE_DPAS_TT<8, float, cutlass::tfloat32_t>, 
               cutlass::tfloat32_t, cutlass::tfloat32_t, float>(
    Shape<_128, _128, _16>{}, Shape<_2, _2, _1>{});
}

TEST(PVC_CuTe_Xe, MMA_DPAS_TF32_4x16) {
  run_mma_test<XE_DPAS_TT<4, float, cutlass::tfloat32_t>, 
               cutlass::tfloat32_t, cutlass::tfloat32_t, float>(
    Shape<_128, _128, _16>{}, Shape<_2, _2, _1>{});
}

TEST(PVC_CuTe_Xe, MMA_DPAS_TF32_2x16) {
  run_mma_test<XE_DPAS_TT<2, float, cutlass::tfloat32_t>, 
               cutlass::tfloat32_t, cutlass::tfloat32_t, float>(
    Shape<_128, _128, _16>{}, Shape<_4, _2, _1>{});
}

TEST(PVC_CuTe_Xe, MMA_DPAS_TF32_1x16) {
  run_mma_test<XE_DPAS_TT<1, float, cutlass::tfloat32_t>, 
               cutlass::tfloat32_t, cutlass::tfloat32_t, float>(
    Shape<_128, _128, _16>{}, Shape<_1, _1, _1>{});
}

#if defined(SYCL_INTEL_TARGET) && (SYCL_INTEL_TARGET == 35)
// TODO: add full FP8/FP4/MXFP8/MXFP4 test here
// missing FP4/MXFP8/MMXFP4 case here due to:
// 1. Examples under folder examples/50_xe35_block_scaled_gemm covered MXFP8/MXFP4 cases.
// 2. Examples/cute/tutorial/xe_gemm.cpp covered FP8/FP4 cases.
// 3. It is somewhat tedious and repetitive to da that here.
TEST(PVC_CuTe_Xe, MMA_DPAS_E5M2) {
  run_mma_test<XE_DPAS_TT<8, float, cutlass::float_e5m2_t>, 
               cutlass::float_e5m2_t, cutlass::float_e5m2_t, float>(
    Shape<_128, _128, _16>{}, Shape<_2, _2, _1>{});
}

TEST(PVC_CuTe_Xe, MMA_DPAS_E4M3) {
  run_mma_test<XE_DPAS_TT<8, float, cutlass::float_e4m3_t>, 
               cutlass::float_e4m3_t, cutlass::float_e4m3_t, float>(
    Shape<_128, _128, _16>{}, Shape<_2, _2, _1>{});
}

namespace {

template <class...> class BDpasNullSrc0KernelName;

template <class MMAOp>
void bdpas_null_src0_kernel(typename MMAOp::DVector* d_null_out,
                            typename MMAOp::DVector* d_ref_out) {
  using AVector = typename MMAOp::AVector;
  using BVector = typename MMAOp::BVector;
  using CVector = typename MMAOp::CVector;
  using DVector = typename MMAOp::DVector;

  AVector a{};
  BVector b{};
  CVector zero{};

  auto* a_bytes = reinterpret_cast<unsigned char*>(&a);
  auto* b_bytes = reinterpret_cast<unsigned char*>(&b);
  CUTE_UNROLL
  for (size_t i = 0; i < sizeof(AVector); ++i) {
    a_bytes[i] = static_cast<unsigned char>((i * 7u + 1u) & 0x7Fu);
  }
  CUTE_UNROLL
  for (size_t i = 0; i < sizeof(BVector); ++i) {
    b_bytes[i] = static_cast<unsigned char>((i * 11u + 3u) & 0x7Fu);
  }

  // 0x7F (=127) is 1.0 in e8m0, so the scale product is 1.0.
  using SFVec = intel::vector_t<uint8_t, 8>;
  SFVec sfa, sfb;
  CUTE_UNROLL
  for (int i = 0; i < 8; ++i) {
    sfa[i] = 0x7F;
    sfb[i] = 0x7F;
  }

  DVector d_null{};
  DVector d_ref{};

  MMAOp::template fma<true>(d_null, a, b, zero, sfa, sfb, 0, 0);
  MMAOp::template fma<false>(d_ref,  a, b, zero, sfa, sfb, 0, 0);

  const int tid = ThreadIdxX();
  d_null_out[tid] = d_null;
  d_ref_out[tid]  = d_ref;
}

template <class MMAOp>
void run_bdpas_null_src0_test() {
  using DVector = typename MMAOp::DVector;
  constexpr int sg_size = 16;

  cutlass::device_vector<DVector> d_null_dev(sg_size);
  cutlass::device_vector<DVector> d_ref_dev(sg_size);

  launch<bdpas_null_src0_kernel<MMAOp>, BDpasNullSrc0KernelName<MMAOp>>(
      launch_policy{compat::dim3(1), compat::dim3(sg_size),
                    kernel_properties{sycl_exp::sub_group_size<sg_size>}},
      d_null_dev.data(), d_ref_dev.data());
  compat::wait_and_throw();

  cutlass::host_vector<DVector> d_null_host = d_null_dev;
  cutlass::host_vector<DVector> d_ref_host  = d_ref_dev;

  using TD = typename MMAOp::DType;
  constexpr int M = sizeof(DVector) / sizeof(TD);
  bool any_nonzero = false;
  int  mismatches  = 0;
  for (int wi = 0; wi < sg_size; ++wi) {
    auto const* dn_p = reinterpret_cast<TD const*>(&d_null_host[wi]);
    auto const* dr_p = reinterpret_cast<TD const*>(&d_ref_host[wi]);
    for (int i = 0; i < M; ++i) {
      float vn = static_cast<float>(dn_p[i]);
      float vr = static_cast<float>(dr_p[i]);
      if (vr != 0.0f) any_nonzero = true;
      if (vn != vr) ++mismatches;
    }
  }
  EXPECT_EQ(mismatches, 0) << "null-src0 BDPAS differs from src0=0 BDPAS";
  EXPECT_TRUE(any_nonzero) << "Reference output is all zero; inputs likely degenerate";
}

} // namespace

TEST(PVC_CuTe_Xe, BDPAS_NullSrc0_BF8_F32) {
  run_bdpas_null_src0_test<XE_BDPAS_TT<8, float, cutlass::float_e5m2_t, cutlass::float_e5m2_t, float>>();
}

TEST(PVC_CuTe_Xe, BDPAS_NullSrc0_HF8_F32) {
  run_bdpas_null_src0_test<XE_BDPAS_TT<8, float, cutlass::float_e4m3_t, cutlass::float_e4m3_t, float>>();
}

TEST(PVC_CuTe_Xe, BDPAS_NullSrc0_BF16_F32) {
  run_bdpas_null_src0_test<XE_BDPAS_TT<8, float, cutlass::bfloat16_t, cutlass::bfloat16_t, float>>();
}
#endif

#else

// For the fallback case
#include "cutlass_unit_test.h"

TEST(PVC_CuTe_Xe, MMA_DPAS_TESTS) {
  GTEST_SKIP() << "MMA DPAS tests require IGC version 2.18 or higher. skipped";
}

#endif
