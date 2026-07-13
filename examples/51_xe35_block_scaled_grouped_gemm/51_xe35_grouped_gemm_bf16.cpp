/***************************************************************************************************
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
\brief CUTLASS Intel xe35 BF16 Grouped GEMM (no scaling) using XE_DPAS_TT.

    To build & run this example (from your build dir):

      $ ninja 51_xe35_grouped_gemm_bf16
      $ ./examples/51_xe35_block_scaled_grouped_gemm/51_xe35_grouped_gemm_bf16

    Call with `--help` for information about available options
*/

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/collective/xe_array_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
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

#include <cfloat>

using namespace cute;
using namespace cutlass::gemm;

using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int,int,int>>;

using ElementAccumulator = float;
using ElementComputeEpilogue = float;
using ElementA = bfloat16_t;
using ElementB = bfloat16_t;
using ElementOutput = float;

///////////////////////////////////////////////////////////////////////////////////////////////////

struct Options {

  bool help = false;
  bool error = false;

  int m, n, k, groups, iterations, verify;
  float alpha, beta;
  std::vector<typename ProblemShape::UnderlyingProblemShape> problem_sizes_host;

  Options():
    m(1024), n(1024), k(64), groups(2), iterations(20), verify(1),
    alpha(1.f), beta(0.f)
  {
    problem_sizes_host.reserve(groups);
    for(int i = 0; i < groups; i++) {
      problem_sizes_host.push_back({m, n, k});
    }
  }

  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("m", m, 1024);
    cmd.get_cmd_line_argument("n", n, 1024);
    cmd.get_cmd_line_argument("k", k, 64);
    cmd.get_cmd_line_argument("groups", groups, 2);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("iterations", iterations, 1);
    cmd.get_cmd_line_argument("verify", verify, 1);

    assert(groups >= 2);
    problem_sizes_host.clear();
    problem_sizes_host.reserve(groups);
    for(int i = 0; i < groups; i++) {
      problem_sizes_host.push_back({m, n, k});
    }
  }

  std::ostream & print_usage(std::ostream &out) const {
    out << "BF16 Grouped GEMM Bandwidth Test\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --m=<int>                   Sets the M extent of the GEMM (default: 1024)\n"
      << "  --n=<int>                   Sets the N extent of the GEMM (default: 1024)\n"
      << "  --k=<int>                   Sets the K extent of the GEMM (default: 64)\n"
      << "  --groups=<int>              Sets the number of groups (default: 2)\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n"
      << "  --iterations=<int>          Iterations\n"
      << "  --verify=<int>              Specify whether to verify.\n\n";
    return out;
  }

  double tflops(double runtime_s) const {
    uint64_t fmas = uint64_t();
    for (auto const & problem : problem_sizes_host) {
      fmas += static_cast<uint64_t>(get<0>(problem)) *
              static_cast<uint64_t>(get<1>(problem)) *
              static_cast<uint64_t>(get<2>(problem));
    }
    uint64_t flop = uint64_t(2) * fmas;
    return double(flop) / double(1.0e12) / runtime_s;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <class Gemm>
struct ExampleRunner {

  using LayoutA = typename Gemm::LayoutA;
  using LayoutB = typename Gemm::LayoutB;
  using LayoutC = typename Gemm::LayoutC;
  using LayoutD = typename Gemm::LayoutD;

  using CollectiveEpilogue = typename Gemm::CollectiveEpilogue;

  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;
  using StrideD = typename Gemm::GemmKernel::InternalStrideD;

  std::vector<int64_t> offset_A;
  std::vector<int64_t> offset_B;
  std::vector<int64_t> offset_C;
  std::vector<int64_t> offset_D;

  std::vector<StrideA> stride_A_host;
  std::vector<StrideB> stride_B_host;
  std::vector<StrideC> stride_C_host;
  std::vector<StrideD> stride_D_host;

  cutlass::DeviceAllocation<typename ProblemShape::UnderlyingProblemShape> problem_sizes;
  cutlass::DeviceAllocation<ElementA> block_A;
  cutlass::DeviceAllocation<ElementB> block_B;
  cutlass::DeviceAllocation<ElementOutput> block_C;
  cutlass::DeviceAllocation<ElementOutput> block_D;
  cutlass::DeviceAllocation<ElementOutput> block_ref_D;

  cutlass::DeviceAllocation<const ElementA *> ptr_A;
  cutlass::DeviceAllocation<const ElementB *> ptr_B;
  cutlass::DeviceAllocation<const ElementOutput *> ptr_C;
  cutlass::DeviceAllocation<ElementOutput *> ptr_D;

  cutlass::DeviceAllocation<StrideA> stride_A;
  cutlass::DeviceAllocation<StrideB> stride_B;
  cutlass::DeviceAllocation<StrideC> stride_C;
  cutlass::DeviceAllocation<StrideD> stride_D;

  uint64_t seed = 0;

  bool verify(const Options &options) {
    bool passed = true;
    for (int32_t i = 0; i < options.groups; ++i) {
      auto problem = options.problem_sizes_host.at(i);
      auto M = get<0>(problem);
      auto N = get<1>(problem);
      auto K = get<2>(problem);
      cutlass::TensorRef ref_A(block_A.get() + offset_A.at(i), LayoutA::packed({M, K}));
      cutlass::TensorRef ref_B(block_B.get() + offset_B.at(i), LayoutB::packed({K, N}));
      cutlass::TensorRef ref_C(block_C.get() + offset_C.at(i), LayoutC::packed({M, N}));
      cutlass::TensorRef ref_D(block_ref_D.get() + offset_D.at(i), LayoutD::packed({M, N}));

      cutlass::reference::device::GemmComplex(
            {M, N, K},
            options.alpha,
            ref_A,
            cutlass::ComplexTransform::kNone,
            ref_B,
            cutlass::ComplexTransform::kNone,
            options.beta,
            ref_C,
            ref_D,
            ElementAccumulator(0),
            1, M * K, K * N, M * N, M * N);

      compat::wait();

      passed &= cutlass::reference::device::BlockCompareEqual(
        block_ref_D.get() + offset_D.at(i), block_D.get() + offset_D.at(i), M * N);
      if (!passed) break;
    }
    return passed;
  }

  void allocate(const Options &options) {
    int64_t total_A = 0, total_B = 0, total_C = 0, total_D = 0;

    for (int32_t i = 0; i < options.groups; ++i) {
      auto problem = options.problem_sizes_host.at(i);
      auto M = get<0>(problem);
      auto N = get<1>(problem);
      auto K = get<2>(problem);

      offset_A.push_back(total_A);
      offset_B.push_back(total_B);
      offset_C.push_back(total_C);
      offset_D.push_back(total_D);

      total_A += M * K;
      total_B += K * N;
      total_C += M * N;
      total_D += M * N;

      stride_A_host.push_back(cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1}));
      stride_B_host.push_back(cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1}));
      stride_C_host.push_back(cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1}));
      stride_D_host.push_back(cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1}));
    }

    block_A.reset(total_A);
    block_B.reset(total_B);
    block_C.reset(total_C);
    block_D.reset(total_D);
    block_ref_D.reset(total_D);
  }

  void initialize(const Options &options) {
    problem_sizes.reset(options.groups);
    problem_sizes.copy_from_host(options.problem_sizes_host.data());

    std::vector<ElementA *> ptr_A_host(options.groups);
    std::vector<ElementB *> ptr_B_host(options.groups);
    std::vector<ElementOutput *> ptr_C_host(options.groups);
    std::vector<ElementOutput *> ptr_D_host(options.groups);

    for (int32_t i = 0; i < options.groups; ++i) {
      ptr_A_host.at(i) = block_A.get() + offset_A.at(i);
      ptr_B_host.at(i) = block_B.get() + offset_B.at(i);
      ptr_C_host.at(i) = block_C.get() + offset_C.at(i);
      ptr_D_host.at(i) = block_D.get() + offset_D.at(i);
    }

    ptr_A.reset(options.groups);
    ptr_A.copy_from_host(ptr_A_host.data());
    ptr_B.reset(options.groups);
    ptr_B.copy_from_host(ptr_B_host.data());
    ptr_C.reset(options.groups);
    ptr_C.copy_from_host(ptr_C_host.data());
    ptr_D.reset(options.groups);
    ptr_D.copy_from_host(ptr_D_host.data());

    stride_A.reset(options.groups);
    stride_A.copy_from_host(stride_A_host.data());
    stride_B.reset(options.groups);
    stride_B.copy_from_host(stride_B_host.data());
    stride_C.reset(options.groups);
    stride_C.copy_from_host(stride_C_host.data());
    stride_D.reset(options.groups);
    stride_D.copy_from_host(stride_D_host.data());

    initialize_block(block_A, seed + 2023);
    initialize_block(block_B, seed + 2022);
    initialize_block(block_C, seed + 2021);
  }

  cutlass::Status run(const Options& options, const cutlass::KernelHardwareInfo& hw_info) {
    allocate(options);
    initialize(options);

    using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerXeGroup<ProblemShape>::RasterOrderOptions;

    typename Gemm::Arguments arguments {
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {options.groups, problem_sizes.get(), options.problem_sizes_host.data()},
      {ptr_A.get(), stride_A.get(), ptr_B.get(), stride_B.get()},
      {{options.alpha, options.beta}, ptr_C.get(), stride_C.get(), ptr_D.get(), stride_D.get()},
      hw_info,
      {1, RasterOrderOptions::AlongN}
    };

    Gemm gemm_op;

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    gemm_op.can_implement(arguments);
    gemm_op.initialize(arguments, workspace.get());

    #ifndef CUTLASS_TEST_FOR_CRI
    // Run warmup on real hardware (skip on CRI simulator as it's time-consuming)
#else
    gemm_op.run();
    compat::wait();
#endif

    if (options.verify != 0) {
      bool passed = verify(options);
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
      double tflops_result = options.tflops(cute_time);
      std::cout << "Problem Size: " << options.m << 'x' << options.n << 'x' << options.k << std::endl;
      std::cout << "Datatype: bfloat16_t" << std::endl;
      std::cout << "Groups: " << options.groups << std::endl;
      printf("Cutlass BF16 Grouped GEMM Performance:     [%4.3f]TFLOP/s  (%6.4f)ms\n", tflops_result, cute_time*1000);
    }
    return cutlass::Status::kSuccess;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename TileShape,
          typename ThreadLayout = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>
cutlass::Status run_bf16_case(Options & options, const cutlass::KernelHardwareInfo& hw_info) {

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using GmemTiledCopyA = void;
  using GmemTiledCopyB = void;

  using TiledMma = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, bfloat16_t>>, Layout<TileShape>, ThreadLayout>::TiledMMA;

  constexpr int PipelineStages = 2;
  using GEMMDispatchPolicy = cutlass::gemm::MainloopXeL1StagedGroup<PipelineStages>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeGenericGroup;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<ElementOutput, ElementComputeEpilogue,
          ElementAccumulator, ElementAccumulator, cutlass::FloatRoundStyle::round_to_nearest>;

  using FusionCallBacks = cutlass::epilogue::fusion::FusionCallbacks<EpilogueDispatchPolicy, EpilogueOp, TileShape,
          decltype(tile_shape(TiledMma()))>;
  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
          EpilogueDispatchPolicy,
          TileShape,
          void,
          ElementAccumulator,
          cutlass::gemm::TagToStrideC_t<LayoutC*>,
          ElementOutput,
          cutlass::gemm::TagToStrideC_t<LayoutD*>,
          FusionCallBacks,
          void, void>;

  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
          GEMMDispatchPolicy,
          TileShape,
          ElementA,
          cutlass::gemm::TagToStrideA_t<LayoutA*>,
          ElementB,
          cutlass::gemm::TagToStrideB_t<LayoutB*>,
          TiledMma,
          GmemTiledCopyA, void, void, cute::identity,
          GmemTiledCopyB, void, void, cute::identity
  >;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,
    CollectiveMainloop,
    CollectiveEpilogue,
    cutlass::gemm::GroupScheduler
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  CUTLASS_CHECK(ExampleRunner<Gemm>{}.run(options, hw_info));

  return cutlass::Status::kSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char** argv) {
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

  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  // Test: 256x256 tile, 8x4 SG layout, BW=64
  std::cout << "=== Test: Tile 256x256, SG 8x4, BW=64 ===" << std::endl;
  CUTLASS_CHECK((run_bf16_case
    <Shape<_256, _256, _32>,
     Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>(options, hw_info)));
  
  return 0;
}
