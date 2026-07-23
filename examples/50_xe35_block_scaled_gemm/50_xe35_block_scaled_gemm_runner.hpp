/***************************************************************************************************
 * Copyright (c) 2025 - 2025 Codeplay Software Ltd. All rights reserved.
 * Copyright (c) 2025 - 2026 Intel Corporation, All rights reserved.
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
/*! \file
\brief CUTLASS Intel xe35 Block Scaled E5M2/E4M3/E2M1 Gemm Implementation.

  - Requirements:
      - Group scaled k size must be 32
      - scales must be MN-major
*/

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/collective/xe_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/util/GPU_Clock.hpp"

#include <cute/tensor.hpp>
#include <random>

#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "sycl_common.hpp"
#include "helper.h"
#include "cutlass/util/mixed_dtype_utils.hpp"

using namespace cute;

///////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help;
  bool error;

  int m, n, k, l, iterations, verify;
  int const_scale;
  float alpha, beta;

  Options():
    help(false),
    error(false),
    m(5120), n(4096), k(4096), l(1), iterations(20),
    alpha(1.f), beta(0.f),
    const_scale(0)
  { }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("m", m, 5120);
    cmd.get_cmd_line_argument("n", n, 4096);
    cmd.get_cmd_line_argument("k", k, 4096);
    cmd.get_cmd_line_argument("l", l, 1);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("iterations", iterations, 100);
    cmd.get_cmd_line_argument("verify", verify, 1);
    cmd.get_cmd_line_argument("const_scale", const_scale, 0);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "BMG GEMM Example\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --m=<int>                   Sets the M extent of the GEMM\n"
      << "  --n=<int>                   Sets the N extent of the GEMM\n"
      << "  --k=<int>                   Sets the K extent of the GEMM\n"
      << "  --l=<int>                   Sets the L extent (batch count) of the GEMM\n"
      << "  --alpha=<s32>               Epilogue scalar alpha\n"
      << "  --beta=<s32>                Epilogue scalar beta\n\n"
      << "  --iterations=<int>          Iterations\n\n"
      << "  --verify=<int>              Specify whether to verify.\n\n"
      << "  --const_scale=<int>         If not specified, a random scale value will be generated.\n\n";

    return out;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <
  class Gemm
>
struct ExampleRunner {

  using CollectiveMainloop = typename Gemm::CollectiveMainloop;
  using CollectiveEpilogue = typename Gemm::CollectiveEpilogue;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  using LayoutA = typename Gemm::LayoutA;
  using LayoutB = typename Gemm::LayoutB;
  using LayoutC = typename Gemm::LayoutC;
  using LayoutD = typename Gemm::LayoutD;

  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementAccumulator = typename Gemm::ElementAccumulator;
  using ElementMMA = typename CollectiveMainloop::ElementMMA;
  using ElementMMAVerify = float;

  using ElementScaleA = typename CollectiveMainloop::ElementScaleA;
  using ElementScaleB = typename CollectiveMainloop::ElementScaleB;

  using StrideScaleA = typename CollectiveMainloop::StrideScaleA;
  using StrideScaleB = typename CollectiveMainloop::StrideScaleB;

  using ElementC = typename Gemm::ElementC;
  using ElementOutput = typename CollectiveEpilogue::ElementOutput;
  using ElementCompute = typename CollectiveEpilogue::ElementCompute;

  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;

  //
  // Data members
  //

  /// Initialization
  StrideA stride_A;
  StrideB stride_B;
  StrideC stride_C;
  StrideD stride_D;
  StrideScaleA stride_SA;
  StrideScaleB stride_SB;

  uint64_t seed = 0;

  cutlass::DeviceAllocation<ElementA> block_A;
  cutlass::DeviceAllocation<ElementB> block_B;
  cutlass::DeviceAllocation<ElementMMAVerify> block_A_dq; // Dequantized copy of A for validation
  cutlass::DeviceAllocation<ElementMMAVerify> block_B_dq; // Dequantized copy of B for validation
  cutlass::DeviceAllocation<ElementC> block_C;
  cutlass::DeviceAllocation<ElementScaleA> block_scaleA;
  cutlass::DeviceAllocation<ElementScaleB> block_scaleB;
  cutlass::DeviceAllocation<ElementOutput> block_D;
  cutlass::DeviceAllocation<ElementOutput> block_ref_D;

  //
  // Methods
  //

  bool verify(const ProblemShapeType& problem_size, ElementCompute alpha, ElementCompute beta) {
     auto [M, N, K, L] = problem_size;

    cutlass::TensorRef ref_A(block_A_dq.get(), LayoutA::packed({M, K}));
    cutlass::TensorRef ref_B(block_B_dq.get(), LayoutB::packed({K, N}));
    cutlass::TensorRef ref_C(block_C.get(), LayoutC::packed({M, N}));
    cutlass::TensorRef ref_D(block_ref_D.get(), LayoutD::packed({M, N}));

    cutlass::reference::device::GemmComplex(
          {M, N, K},
          alpha,
          ref_A,
          cutlass::ComplexTransform::kNone,
          ref_B,
          cutlass::ComplexTransform::kNone,
          beta,
          ref_C,
          ref_D,
          ElementAccumulator(0),
          L,     // batch_count
          M * K, // batch_stride_A
          K * N, // batch_stride_B
          M * N, // batch_stride_C
          M * N  // batch_stride_D
        );

    // CUTLASS on SYCL uses the compatibility library compat for e.g. default in-order queue
    compat::wait();

    // Check if output from CUTLASS kernel and reference kernel are equal or not
    // compare_reference
    ElementOutput const epsilon(1e-2f);
    ElementOutput const non_zero_floor(1e-4f);
    bool passed = cutlass::reference::device::BlockCompareRelativelyEqual(
      block_ref_D.get(), block_D.get(), block_D.size(), epsilon, non_zero_floor);
    if (!passed) {  
      std::vector<ElementOutput> block_ref_D_host(block_ref_D.size());
      std::vector<ElementOutput> block_D_host(block_D.size());
      compat::memcpy(block_ref_D_host.data(), block_ref_D.get(), block_ref_D_host.size() * sizeof(ElementOutput));
      compat::memcpy(block_D_host.data(), block_D.get(), block_D_host.size() * sizeof(ElementOutput));
      for (int i = 0; i < block_D_host.size(); i++) {
        printf("i: %d , ref: %f, comp: %f\n", i, block_ref_D_host[i], block_D_host[i]);
      }
    }
    return passed;
  }

  template <class Element>
  bool initialize_scale(
    cutlass::DeviceAllocation<Element>& block,
    Options const& options) {
    const float elt_max_f = float(cutlass::platform::numeric_limits<Element>::max());
    // Need to fix max_dequant_val and min_dequant_val?
    const float max_dequant_val = elt_max_f * 0.25f;
    const float min_dequant_val = 0.5f;
    const float scale_max = options.const_scale ? 1.0 : max_dequant_val / elt_max_f;
    const float scale_min = options.const_scale ? 1.0 : min_dequant_val / elt_max_f;
#if defined(CUTLASS_TEST_FOR_CRI)
    cutlass::reference::device::BlockFillRandomUniformCopyFromHost(
        block.get(), block.size(), seed, Element(scale_max), Element(scale_min));
#else
    cutlass::reference::device::BlockFillRandomUniform(
        block.get(), block.size(), seed, Element(scale_max), Element(scale_min));
#endif
    return true;
  }

  template <
  class DstElement,
  class SrcElement,
  class Layout,
  class ElementScale,
  class ScaleLayout>
  static void apply_scale(DstElement* dq_buffer,
                       SrcElement const* q_buffer,
                       Layout const operand_layout,
                       ElementScale const* scale_buffer,
                       ScaleLayout const scale_layout,
                       Options const& options,
                       int group_size) {

    std::vector<uint8_t> dst(size(operand_layout) * sizeof_bits_v<DstElement> / 8, 0);
    cutlass::device_memory::copy_to_host(dst.data(), (uint8_t*)dq_buffer, dst.size());

    std::vector<uint8_t> src(size(operand_layout) * sizeof_bits_v<SrcElement> / 8, 0);
    cutlass::device_memory::copy_to_host(src.data(), (uint8_t*)q_buffer, src.size());

    std::vector<uint8_t> scale(size(scale_layout) * sizeof_bits_v<ElementScale> / 8, 0);
    cutlass::device_memory::copy_to_host(scale.data(), (uint8_t*)scale_buffer, scale.size());

    compat::wait();

    static_assert(sizeof_bits_v<DstElement> >= 8);

    auto dst_tensor = make_tensor(make_gmem_ptr(reinterpret_cast<DstElement*>(dst.data())), operand_layout);

    auto src_tensor = [&]() {
      if constexpr (sizeof_bits_v<SrcElement> < 8) {
        return make_tensor(cute::subbyte_iterator<const SrcElement>(src.data()), operand_layout);
      } else {
        return make_tensor(make_gmem_ptr(reinterpret_cast<SrcElement const *>(src.data())), operand_layout);
      }
    }();

    auto scale_tensor = make_tensor(make_gmem_ptr(reinterpret_cast<ElementScale const *>(scale.data())), scale_layout);

    auto MN = size<0>(src_tensor);
    auto K = size<1>(src_tensor);
    auto L = size<2>(src_tensor);

    using ret_type = float;

    for (int l = 0; l < L; l++) {
      for (int k= 0; k < K; k++) {
        for (int mn = 0; mn < MN; mn++) {
          auto src_data = [&]() {
            if constexpr (sizeof_bits_v<SrcElement> >= 8) {
              return  (ret_type)(src_tensor(mn, k, l));
            } else {
              return (ret_type)(src_tensor(mn, k, l).get());
            }
          }();

          auto scale_data = (ret_type)(scale_tensor(mn, k / group_size, l));

          dst_tensor(mn, k, l) = (src_data) * scale_data;
        }
      }
    }

    cutlass::device_memory::copy_to_device(dq_buffer, (DstElement*)(raw_pointer_cast(dst_tensor.data())), dst_tensor.size());
    compat::wait();
  }

  /// Initialize operands to be used in the GEMM and reference GEMM
  void initialize(Options const& options) {
    auto [M, N, K, L] = ProblemShapeType{options.m, options.n, options.k, options.l};

    constexpr int scaleGroupSize = CollectiveMainloop::GroupK;
    const int scale_k = cute::ceil_div(options.k, scaleGroupSize);
    auto shape_A = cute::make_shape(M, K, L);
    auto shape_B = cute::make_shape(N, K, L);
    auto shape_CD = cute::make_shape(M, N, L);

    // 2D block load requires surface width to be 4-byte aligned
    constexpr int scaleAlign = cute::ceil_div(4, (int)sizeof(ElementScaleA));
    int padded_M = cute::round_up(options.m, scaleAlign);
    int padded_N = cute::round_up(options.n, scaleAlign);
    auto shape_scale_A_padded = cute::make_shape(padded_M, scale_k, L);
    auto shape_scale_B_padded = cute::make_shape(padded_N, scale_k, L);

    stride_A = cutlass::make_cute_packed_stride(StrideA{}, shape_A);
    stride_B = cutlass::make_cute_packed_stride(StrideB{}, shape_B);
    stride_C = cutlass::make_cute_packed_stride(StrideC{}, shape_CD);
    stride_D = cutlass::make_cute_packed_stride(StrideD{}, shape_CD);
    stride_SA = cutlass::make_cute_packed_stride(StrideScaleA{}, shape_scale_A_padded);
    stride_SB = cutlass::make_cute_packed_stride(StrideScaleB{}, shape_scale_B_padded);

    block_A.reset(static_cast<std::size_t>(M) * K * L);
    block_A_dq.reset(static_cast<std::size_t>(M) * K * L);
    block_B.reset(static_cast<std::size_t>(K) * N * L);
    block_B_dq.reset(static_cast<std::size_t>(K) * N * L);
    block_C.reset(static_cast<std::size_t>(M) * N * L);
    block_D.reset(static_cast<std::size_t>(M) * N * L);
    block_ref_D.reset(static_cast<std::size_t>(M) * N * L);
    block_scaleA.reset(static_cast<std::size_t>(scale_k) * L * padded_M);
    block_scaleB.reset(static_cast<std::size_t>(scale_k) * L * padded_N);
    if constexpr (std::is_same_v<ElementA, half_t> || std::is_same_v<ElementA, float>) {
      initialize_block(block_A, seed + 2023, ElementA(0.f), ElementA(1.f));
    } else {
      initialize_block(block_A, seed + 2023);
    }
    initialize_block(block_B, seed + 2022);
    initialize_block(block_C, seed + 2021);

    convert_dtype<ElementA, ElementMMAVerify, ExampleRunner>(
        block_A,
        block_A_dq
    );
    convert_dtype<ElementB, ElementMMAVerify, ExampleRunner>(
        block_B,
        block_B_dq
    );

    initialize_scale(block_scaleA, options);
    initialize_scale(block_scaleB, options);

    auto layout_A = make_layout(shape_A, stride_A);
    auto layout_B = make_layout(shape_B, stride_B);
    auto layout_scale_A = make_layout(shape_scale_A_padded, stride_SA);
    auto layout_scale_B = make_layout(shape_scale_B_padded, stride_SB);

    apply_scale(block_A_dq.get(), block_A.get(), layout_A, block_scaleA.get(),  layout_scale_A, options, scaleGroupSize);
    apply_scale(block_B_dq.get(), block_B.get(), layout_B, block_scaleB.get(),  layout_scale_B, options, scaleGroupSize);
  }
  
  cutlass::Status run(const Options& options, const cutlass::KernelHardwareInfo& hw_info) {
    ProblemShapeType problem_size = ProblemShapeType{options.m, options.n, options.k, options.l};

    initialize(options);

    typename Gemm::GemmKernel::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size,
      {block_A.get(), stride_A, block_B.get(), stride_B,
      block_scaleA.get(), stride_SA, block_scaleB.get(), stride_SB},
      {{options.alpha, options.beta}, block_C.get(), stride_C, block_D.get(), stride_D},
      hw_info
    };

    Gemm gemm_op;

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    if (gemm_op.can_implement(arguments) != cutlass::Status::kSuccess){
      std::cout << "Invalid Problem Size: " << options.m << 'x' << options.n << 'x' << options.k << 'x' << options.l << std::endl;
      std::exit(1);
    }

    CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));

    // Run the GEMM
    CUTLASS_CHECK(gemm_op.run());

    compat::wait();

    if (options.verify != 0) {
    // Verify that the result is correct
    bool passed = verify(problem_size, options.alpha, options.beta);
    std::cout << "Disposition: " << (passed ? "Passed" : "Failed") << std::endl;

    if (!passed) return cutlass::Status::kErrorInternal;
    } else {
      std::cout << "Disposition is skipped." << std::endl;
    }

    if (options.iterations > 0) {
      GPU_Clock timer;
      timer.start();
      for (int i = 0; i < options.iterations; ++i) {
        gemm_op.run();
      }
      compat::wait();

      float cute_time = timer.seconds() / options.iterations;
      double tflops = (2.0 * options.m * options.n * options.k * options.l) * 1e-12;
      std::cout << "Problem Size: " << options.m << 'x' << options.n << 'x' << options.k << 'x' << options.l << std::endl;
      if constexpr (std::is_same_v<ElementA, float_e4m3_t>) {
        std::cout << "Datatype: float_e4m3_t"<< std::endl;
      } else if constexpr (std::is_same_v<ElementA, float_e5m2_t>) {
        std::cout << "Datatype: float_e5m2_t"<< std::endl;
      }
      printf("Cutlass GEMM Performance:     [%4.3f]TFlop/s  (%6.4f)ms\n", tflops / cute_time, cute_time*1000);
    }

    return cutlass::Status::kSuccess;
  }

};
