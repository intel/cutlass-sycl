/* Copyright (C) 2026 Intel Corporation, All rights reserved.
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

#include "cutlass/detail/layout.hpp"

#include <cute/tensor.hpp>
#include <cute/tensor_sg.hpp>
#include <sycl/sycl.hpp>
#include <cute/util/compat.hpp>

#include "cutlass_unit_test.h"

using namespace cute;
using namespace cutlass;
using namespace compat::experimental;

// ============================================================================
// Test Helpers
// ============================================================================

template<class...> class DequantizeKernelName;

// Generic dequantize test kernel for SubgroupTensor
template <class SrcType, class DstType, class SrcSGLayout, class DstSGLayout, bool reference>
void dequantize_kernel_subgroup_tensor(SrcType* src_global, DstType* dst_global, float scale)
{
  const int tid = ThreadIdxX();
  constexpr int total_size = size(SrcSGLayout{});
  static_assert(total_size % intel::sg_size == 0, "Total size must be divisible by subgroup size");
  constexpr int values_per_thread = total_size / intel::sg_size;

  // Each thread owns a slice of values (round-robin pattern)
  SrcType src_local[values_per_thread];
  DstType dst_local[values_per_thread];

  // Load from global memory (each thread loads its values)
  for (int i = 0; i < values_per_thread; ++i) {
    src_local[i] = src_global[tid + i * intel::sg_size];
  }

  // Create fragments
  auto src_tensor = make_tensor(make_rmem_ptr(src_local),
                                 make_layout(Shape<Int<values_per_thread>>{}));
  auto dst_tensor = make_tensor(make_rmem_ptr(dst_local),
                                 make_layout(Shape<Int<values_per_thread>>{}));

  // Create SubgroupTensors and perform dequantize
  auto src_sg = make_subgroup_tensor(src_tensor, SrcSGLayout{});
  auto dst_sg = make_subgroup_tensor(dst_tensor, DstSGLayout{});
  reorder(src_sg, dst_sg);
  if constexpr (reference) {
    CUTE_UNROLL
    for (int i = 0; i < dst_sg.size(); ++i) {
      dst_sg(i) = static_cast<DstType>(scale * static_cast<float>(dst_sg(i)));
    }
  } else {
    dequantize(dst_sg, scale);
  }

  // Store back to global memory
  for (int i = 0; i < values_per_thread; ++i) {
    dst_global[tid + i * intel::sg_size] = dst_local[i];
  }
}

// Helper function to run a dequantize test
template <class SrcType, class DstType, class SrcSGLayout, class DstSGLayout, bool reference, int TestID>
void run_dequantize_test(cutlass::host_vector<SrcType>& host_src,
                      cutlass::host_vector<DstType>& host_dst,
                      float scale)
{
  cutlass::device_vector<SrcType> device_src = host_src;
  cutlass::device_vector<DstType> device_dst(size(DstSGLayout{}));

  launch<dequantize_kernel_subgroup_tensor<SrcType, DstType, SrcSGLayout, DstSGLayout, reference>,
         DequantizeKernelName<SrcType, DstType, std::bool_constant<reference>, Int<TestID>>>(
      launch_policy{compat::dim3(1), compat::dim3(intel::sg_size),
                    kernel_properties{sycl_exp::sub_group_size<intel::sg_size>}},
      device_src.data(), device_dst.data(), scale);

  compat::wait_and_throw();
  host_dst = device_dst;
}

// Helper function to initialize source data
template <class SrcType>
void initialize_source(cutlass::host_vector<SrcType>& host_src) {
  for (size_t i = 0; i < host_src.size(); ++i) {
    if constexpr (std::is_same_v<SrcType, cutlass::float_e4m3_t> || std::is_same_v<SrcType, cutlass::float_e5m2_t>) {
      host_src[i] = static_cast<SrcType>(static_cast<float>(i % 17) * 0.5f);
    } else {
      CUTE_INVALID_CONTROL_PATH("Not Implemented");
    }
  }
}

// Generic template-based test helper for dequantize operations
template <class SrcType, class DstType, int TestID>
struct XeDequantizeVVTest {
  static void run() {
    constexpr auto src_sg_tv_layout = make_layout(
      make_shape(
        make_shape(_4{}, _4{}),
        make_shape(
          make_shape(_4{}, _8{}),
          _4{}
        ),
        make_shape(_1{}, _1{})
      ),
      make_stride(
        make_stride(ScaledBasis<Int<1>,1>{}, ScaledBasis<Int<1>,0>{}),
        make_stride(
          make_stride(ScaledBasis<Int<4>,0>{}, ScaledBasis<Int<4>,1>{}),
          ScaledBasis<Int<16>,0>{}
        ),
        make_stride(Int<0>{}, Int<0>{})
      )
    );

    constexpr auto dst_sg_tv_layout = make_layout(
      make_shape(
          make_shape(Int<2>{}, Int<8>{}),
          make_shape(
            make_shape(Int<2>{}, Int<8>{}),
            make_shape(Int<4>{}, Int<2>{})
          )
      ),
      make_stride(
          make_stride(ScaledBasis<Int<1>,1>{}, ScaledBasis<Int<1>,0>{}),
          make_stride(
            make_stride(ScaledBasis<Int<8>,0>{}, ScaledBasis<Int<2>,1>{}),
            make_stride(ScaledBasis<Int<16>,0>{}, ScaledBasis<Int<16>,1>{})
          )
      )
    );

    using SrcSGLayout = decltype(src_sg_tv_layout);
    using DstSGLayout = decltype(dst_sg_tv_layout);
    static_assert(size(SrcSGLayout{}) == size(DstSGLayout{}),
                  "Source and Destination TensorLayouts must have the same number of elements");

    constexpr auto total_elements = size(SrcSGLayout{});
    cutlass::host_vector<SrcType> host_src(total_elements);
    cutlass::host_vector<DstType> host_dst(total_elements);
    cutlass::host_vector<DstType> host_expected(total_elements);

    // Initialize source data
    initialize_source(host_src);

    float scale = 2.0f;

    // Run the dequantize kernel
    run_dequantize_test<SrcType, DstType, SrcSGLayout, DstSGLayout, true, TestID>(host_src, host_expected, scale);
    run_dequantize_test<SrcType, DstType, SrcSGLayout, DstSGLayout, false, TestID>(host_src, host_dst, scale);

    // Verify correctness (with tolerance for floating point)
    for (size_t i = 0; i < host_expected.size(); ++i) {
      EXPECT_NEAR(static_cast<float>(host_dst[i]),
                  static_cast<float>(host_expected[i]), 1e-4f);
    }
  }
};

// Test: dequantize VNNI fp8 to VNNI half/bf16
TEST(CuTe_Xe_Dequantize, VV_e5m2_to_half) {
  XeDequantizeVVTest<cutlass::float_e5m2_t, cutlass::half_t, 0>::run();
}

TEST(CuTe_Xe_Dequantize, VV_e4m3_to_half) {
  XeDequantizeVVTest<cutlass::float_e4m3_t, cutlass::half_t, 1>::run();
}

TEST(CuTe_Xe_Dequantize, VV_e5m2_to_bf16) {
  XeDequantizeVVTest<cutlass::float_e5m2_t, cutlass::bfloat16_t, 2>::run();
}

TEST(CuTe_Xe_Dequantize, VV_e4m3_to_bf16) {
  XeDequantizeVVTest<cutlass::float_e4m3_t, cutlass::bfloat16_t, 3>::run();
}
