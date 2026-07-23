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
/*! \file
    \brief CUTLASS Intel BMG Gemm with Linear Scaling, Aux Store, and Element-wise Addition epilogue.

    This example constructs and executes a standard GEMM fused with an epilogue pipeline
    consisting of Linear Scaling, Auxiliary output, and Element-wise C addition.

    CUTLASS 3.x epilogues are implemented using the Epilogue Visitor Tree design pattern.
    This example performs:

    1. Linear Scaling:     Accum = alpha * (A*B)
    2. Aux Store:          Aux   = Accum (stored to global memory)
    3. Element-wise Add:   D     = Accum + C

    To build & run this example (from your build dir):

      $ ninja 05_bmg_gemm_with_epilogue_lincombauxstore
      $ ./examples/05_bmg_gemm_with_epilogues/05_bmg_gemm_with_epilogue_lincombauxstore

    Call with `--help` for information about available options
*/
#include "cutlass/cutlass.h"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/kernel/sm90_tile_scheduler.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "stddef.h"
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
#include "cutlass/util/reference/device/tensor_relu.h"
#include "cutlass/tensor_view.h"
#include "cutlass/coord.h"

#include "sycl_common.hpp"
#include "helper.h"

using namespace cute;

// Command line options parsing
struct Options {

  bool help;
  bool error;

  int m, n, k, l, iterations, verify;
  float alpha, beta;

  Options():
    help(false),
    error(false),
    m(768), n(768), k(128), l(3), iterations(100), verify(1),
    alpha(1.f), beta(0.f)
  { }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("m", m, 768);
    cmd.get_cmd_line_argument("n", n, 768);
    cmd.get_cmd_line_argument("k", k, 128);
    cmd.get_cmd_line_argument("l", l, 3);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("iterations", iterations, 100);
    cmd.get_cmd_line_argument("verify", verify, 1);
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
      << "  --beta=<s32>                Epilogue scalar beta\n"
      << "  --iterations=<int>          Iterations\n"
      << "  --verify=<int>              Specify whether to verify.\n\n";

    return out;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

using EpilogueDescriptor = cutlass::epilogue::collective::detail::EpilogueDescriptor<
  cute::Shape<_256, _256, _32>, cutlass::epilogue::collective::EpilogueTileAuto,
  cutlass::half_t, cutlass::half_t,
  cutlass::epilogue::collective::EpilogueScheduleAuto
>;

using ElementC = cutlass::half_t;
using StrideC = cute::Stride<int64_t, cute::Int<1>, int64_t>; 
using StrideD = cute::Stride<int64_t, cute::Int<1>, int64_t>; 
using TensorC = cutlass::epilogue::fusion::XeSrcFetch<cutlass::half_t>;

using ElementD = cutlass::half_t;

using Accum = cutlass::epilogue::fusion::XeAccFetch;

using Alpha = cutlass::epilogue::fusion::XeScalarBroadcast<
    float, cute::Stride<cute::Int<0>, cute::Int<0>, cute::Int<0>>, 1, cutlass::multiplies
>;

using Compute0 = cutlass::epilogue::fusion::XeCompute<
    cutlass::multiplies, float, float,
    cutlass::FloatRoundStyle::round_to_nearest
>;

using EVTCompute0 = cutlass::epilogue::fusion::XeEVT<
    Compute0,
    Alpha,
Accum>;

using F = cutlass::epilogue::fusion::XeAuxStore<
    cutlass::half_t,
    cute::Stride<int64_t, cute::Int<1>, int64_t>
>;

using EVTF = cutlass::epilogue::fusion::XeEVT<
        F,
        EVTCompute0>;

using Compute1 = cutlass::epilogue::fusion::XeCompute<
    cutlass::plus, cutlass::half_t, float,
    cutlass::FloatRoundStyle::round_to_nearest
>;

using EVTCompute1 = cutlass::epilogue::fusion::XeEVT<
        Compute1,
        EVTF,
    TensorC>;


using CollectiveEpilogue =
  typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Xe20, cutlass::arch::OpClassTensorOp,
    cute::Shape<cute::_256, cute::_256, cute::_32>,
    cute::Shape<cute::_1,cute::_1,cute::_1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float,
    ElementC, StrideC, 8,
    ElementD, StrideD, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto,
    EVTCompute1
  >::CollectiveOp;

using CollectiveMainloop =
  typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Xe20, cutlass::arch::OpClassTensorOp,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    float,
    cute::Shape<cute::_256, cute::_256, cute::_32>,
    cute::Shape<cute::_1,cute::_1,cute::_1>,
    cutlass::gemm::collective::StageCountAuto,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

// Gemm operator cutlass3x_Xe20_tensorop_s81616gemm_f16_f16_f32_f16_f16_256x256x32_1x1x1_0_ttt_align8
using cutlass3x_Xe20_tensorop_s81616gemm_f16_f16_f32_f16_f16_256x256x32_1x1x1_0_ttt_align8_base = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    cutlass::gemm::PersistentScheduler
>;

// Define named type
struct cutlass3x_Xe20_tensorop_s81616gemm_f16_f16_f32_f16_f16_256x256x32_1x1x1_0_ttt_align8_type :
  public cutlass3x_Xe20_tensorop_s81616gemm_f16_f16_f32_f16_f16_256x256x32_1x1x1_0_ttt_align8_base { };

using Operator = cutlass3x_Xe20_tensorop_s81616gemm_f16_f16_f32_f16_f16_256x256x32_1x1x1_0_ttt_align8_type;


template <
  class Gemm
>
struct ExampleRunner {

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

  using CollectiveEpilogue = typename Gemm::CollectiveEpilogue;

  using ElementC = cutlass::half_t;
  using ElementOutput = cutlass::half_t;
  using ElementCompute = float; // Epilogue (alpha/beta/acc)

  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;

  //
  // Data members
  //

  /// Initialization
  StrideA stride_A;
  StrideB stride_B;
  StrideC stride_C;
  StrideD stride_D;
  uint64_t seed = 0;

  cutlass::DeviceAllocation<ElementA> block_A;
  cutlass::DeviceAllocation<ElementB> block_B;
  cutlass::DeviceAllocation<ElementC> block_C;
  cutlass::DeviceAllocation<ElementOutput> block_D;
  cutlass::DeviceAllocation<ElementOutput> block_Aux;  
  cutlass::DeviceAllocation<ElementOutput> block_ref_D;

  //
  // Methods
  //

  bool verify(const ProblemShapeType& problem_size, ElementCompute alpha, ElementCompute beta) {
    auto [M, N, K, L] = problem_size;

    cutlass::TensorRef ref_A(block_A.get(), LayoutA::packed({M, K}));
    cutlass::TensorRef ref_B(block_B.get(), LayoutB::packed({K, N}));
    cutlass::TensorRef ref_C(block_C.get(), LayoutC::packed({M, N}));
    cutlass::TensorRef ref_D(block_ref_D.get(), LayoutD::packed({M, N}));

    cutlass::reference::device::GemmComplex(
          {M, N, K},
          alpha,                // alpha
          ref_A,
          cutlass::ComplexTransform::kNone,
          ref_B,
          cutlass::ComplexTransform::kNone,
          ElementCompute(0),    // beta = 0
          ref_C,
          ref_D,                
          ElementAccumulator(0),
          L,     // batch_count
          M * K, // batch_stride_A
          K * N, // batch_stride_B
          M * N, // batch_stride_C
          M * N  // batch_stride_D
        );

    compat::wait();

    bool aux_passed = cutlass::reference::device::BlockCompareEqual(
      block_ref_D.get(), block_Aux.get(), block_Aux.size());

    if (aux_passed) {
        std::cout << "Aux verification passed." << std::endl;
    } else {
        std::cout << "Aux verification failed." << std::endl;  
    }

    cutlass::reference::device::GemmComplex(
          {M, N, K},
          alpha,                // alpha
          ref_A,
          cutlass::ComplexTransform::kNone,
          ref_B,
          cutlass::ComplexTransform::kNone,
          ElementCompute(1),    // beta = 1 (D = Aux + 1.0 * C)
          ref_C,
          ref_D,                
          ElementAccumulator(0),
          L,     // batch_count
          M * K, // batch_stride_A
          K * N, // batch_stride_B
          M * N, // batch_stride_C
          M * N  // batch_stride_D
        );

    compat::wait();

    bool d_passed = cutlass::reference::device::BlockCompareEqual(
      block_ref_D.get(), block_D.get(), block_D.size());
    
    if (d_passed) {
        std::cout << "D verification passed." << std::endl;
    } else {
        std::cout << "D verification failed." << std::endl;
    }

    return aux_passed && d_passed;
  }

  /// Initialize operands to be used in the GEMM and reference GEMM
  void initialize(const ProblemShapeType& problem_size) {
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto [M, N, K, L] = problem_shape_MNKL;

    stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
    stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, L));
    stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, L));
    stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, L));

    block_A.reset(static_cast<std::size_t>(M) * K * L);
    block_B.reset(static_cast<std::size_t>(K) * N * L);
    block_C.reset(static_cast<std::size_t>(M) * N * L);
    block_D.reset(static_cast<std::size_t>(M) * N * L);
    block_ref_D.reset(static_cast<std::size_t>(M) * N * L);
    block_Aux.reset(static_cast<std::size_t>(M) * N * L);

    initialize_block(block_A, seed + 2023);
    initialize_block(block_B, seed + 2022);
    initialize_block(block_C, seed + 2021);
    initialize_block(block_Aux, seed + 2020);
  }

  cutlass::Status run(const Options& options, const cutlass::KernelHardwareInfo& hw_info) {
    ProblemShapeType problem_size = ProblemShapeType{options.m, options.n, options.k, options.l};

    initialize(problem_size);

  typename Alpha::Arguments alpha_args;
  alpha_args.scalars[0] = options.alpha;
  alpha_args.scalar_ptrs[0] = nullptr;
  alpha_args.dScalar[0] = {};

  typename Compute0::Arguments compute0_args{};

  typename F::Arguments f_args{};
  f_args.ptr_aux = block_Aux.get();
  f_args.dAux = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(options.m, options.n, options.l));

  typename Compute1::Arguments compute1_args{};
    
    // 1. 
    // XeEVT<Compute0(Mul), Alpha, Accum>
    // {Alpha, Accum, Compute0}
    typename EVTCompute0::Arguments evt0_args{
        alpha_args,            // Child 1: Alpha
        {},                    // Child 2: Accum (Empty)
        compute0_args          // Node: Compute0 (Empty)
    };

    // 2. 
    // XeEVT<F(AuxStore), EVTCompute0>
    // {EVTCompute0, F}
    typename EVTF::Arguments evtf_args{
        evt0_args,             // Child 1: EVTCompute0
        f_args                 // Node: F (AuxStore)
    };

    // 3. 
    // XeEVT<Compute1(Plus), EVTF, TensorC>
    // {EVTF, TensorC, Compute1}
    typename EVTCompute1::Arguments thread {
      evtf_args,               // Child 1: EVTF
      {},                      // Child 2: TensorC (Empty)
      compute1_args            // Node: Compute1 (Empty)
    };
    

    using EpilogueArguments = typename Gemm::GemmKernel::EpilogueArguments;
    EpilogueArguments epilogue_arguments{
      thread, block_C.get(), stride_C, block_D.get(), stride_D};

    typename Gemm::GemmKernel::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size,
      {block_A.get(), stride_A, block_B.get(), stride_B},
      epilogue_arguments,
      hw_info
    };

    Gemm gemm_op;

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    CUTLASS_CHECK(gemm_op.can_implement(arguments));

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
      printf("Cutlass GEMM Performance:     [%4.3f]TFlop/s  (%6.4f)ms\n", tflops / cute_time, cute_time*1000);
    }

    return cutlass::Status::kSuccess;
  }

};

int main(int argc, const char** argv)
{
  //
  // Parse options
  //

  Options options;

  options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  if (options.error) {
    std::cerr << "Aborting execution." << std::endl;
    return -1;
  }

  //
  // Run examples
  //

  // The KernelHardwareInfo struct holds the number of EUs on the GPU with a given device ID. This
  // information is used by the underlying kernel.
  cutlass::KernelHardwareInfo hw_info;

  // Change device_id to another value if you are running on a machine with multiple GPUs and wish
  // to use a GPU other than that with device ID 0.
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Operator>;

  ExampleRunner<Gemm> runner;

  CUTLASS_CHECK(runner.run(options, hw_info));

  return 0;
}
