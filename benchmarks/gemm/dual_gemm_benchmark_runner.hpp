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
    \brief Benchmark runner for the Intel BMG Dual GEMM kernel.

    Mirrors the pipeline used by example 07_bmg_dual_gemm: a single shared A matrix is multiplied by
    two B matrices in one kernel (MainloopIntelXeXMX16 + two linear-combination epilogues), and the
    two results are fused through a SiLU element-wise activation epilogue producing a single D.

    The Dual GEMM kernel is not wrapped by GemmUniversalAdapter, so this runner replicates the
    example's manual SYCL kernel launch. Only the performance path is exercised (no host
    verification), matching how the regular GEMM benchmarks behave under the CRI simulator.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/device_kernel.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/epilogue/thread/activation.h"

#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/initialize_block.hpp"

// Dual GEMM application headers (available via CUTLASS_APPLICATIONS_DIR include path)
#include "dual_gemm/collective/xe_dual_gemm_mma.hpp"
#include "dual_gemm/collective/xe_dual_gemm_epilogue.hpp"
#include "dual_gemm/collective/xe_dual_gemm_epilogue_elementwise_activation.hpp"
#include "dual_gemm/kernel/xe_dual_gemm.hpp"
#include "dual_gemm/thread/xe_binary_elem_wise_op.hpp"

// Provides cutlass::benchmark::GEMMOptions and pulls in benchmark/common headers.
#include "benchmark_runner.hpp"

using namespace cute;

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::device {

// Dual GEMM configuration. Builds the full DualGemm kernel type from a small set of knobs, matching
// the no-bias path of example 07 (D = SiLU(alpha0 * A * B0) * (alpha1 * A * B1)).
template<
  class ElementA, class LayoutA,
  class ElementB, class LayoutB,
  class ElementOutput, class LayoutC,
  class TileShape,
  class TiledMma,
  class GmemTiledCopyA = XE_2D_U16x16x32_LD_N,
  class GmemTiledCopyB = XE_2D_U16x32x32_LD_V,
  int   PipelineStages_ = 2,
  class ElementAccumulator = float,
  class ElementCompute = float>
struct DualGemmConfiguration {
  static constexpr int PipelineStages = PipelineStages_;

  using LayoutD = LayoutC;

  using GEMMDispatchPolicy = cutlass::gemm::MainloopIntelXeXMX16<PipelineStages>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeXMX16;

  // Per-B-matrix linear-combination epilogue (no bias), identity activation.
  using EpilogueOp = cutlass::epilogue::fusion::LinCombEltAct<
      cutlass::epilogue::thread::Identity, ElementOutput, ElementCompute,
      ElementAccumulator, ElementAccumulator, cutlass::FloatRoundStyle::round_to_nearest>;

  using FusionCallBacks0 = cutlass::epilogue::fusion::FusionCallbacks<
      EpilogueDispatchPolicy, EpilogueOp, TileShape, decltype(tile_shape(TiledMma()))>;
  using FusionCallBacks1 = cutlass::epilogue::fusion::FusionCallbacks<
      EpilogueDispatchPolicy, EpilogueOp, TileShape, decltype(tile_shape(TiledMma()))>;

  // The two intermediate epilogue outputs are required by the SiLU fusion stage below.
  static constexpr bool WriteEpilogueOutput0 = true;
  static constexpr bool WriteEpilogueOutput1 = true;

  using CollectiveEpilogue0 = cutlass::epilogue::collective::DualGemmEpilogue<
      EpilogueDispatchPolicy, TileShape, ElementAccumulator,
      cutlass::gemm::TagToStrideC_t<LayoutC>, ElementOutput,
      cutlass::gemm::TagToStrideC_t<LayoutD>, FusionCallBacks0,
      XE_2D_U32x8x16_LD_N, XE_2D_U32x8x16_ST_N, WriteEpilogueOutput0>;
  using CollectiveEpilogue1 = cutlass::epilogue::collective::DualGemmEpilogue<
      EpilogueDispatchPolicy, TileShape, ElementAccumulator,
      cutlass::gemm::TagToStrideC_t<LayoutC>, ElementOutput,
      cutlass::gemm::TagToStrideC_t<LayoutD>, FusionCallBacks1,
      XE_2D_U32x8x16_LD_N, XE_2D_U32x8x16_ST_N, WriteEpilogueOutput1>;

  using EpilogueOutputOp2 = cutlass::epilogue::thread::FusedElementWiseOpDualGemm<
      ElementOutput, cutlass::epilogue::thread::SiLu, cutlass::epilogue::thread::Identity,
      cutlass::multiplies, ElementAccumulator, ElementAccumulator>;

  using CollectiveEpilogueActivation = cutlass::epilogue::collective::DualGemmElemActEpilogue<
      EpilogueDispatchPolicy, TileShape, void,
      cutlass::gemm::TagToStrideC_t<LayoutC>, ElementOutput,
      cutlass::gemm::TagToStrideC_t<LayoutD>, void,
      XE_2D_U32x8x16_ST_N, EpilogueOutputOp2>;

  using CollectiveMainloop = cutlass::gemm::collective::DualGemmMma<
      GEMMDispatchPolicy, TileShape,
      ElementA, cutlass::gemm::TagToStrideA_t<LayoutA>,
      ElementB, cutlass::gemm::TagToStrideB_t<LayoutB>,
      TiledMma, GmemTiledCopyA, GmemTiledCopyB>;

  using GemmKernel = cutlass::gemm::kernel::DualGemm<
      Shape<int, int, int, int>,
      CollectiveMainloop, CollectiveEpilogue0, CollectiveEpilogue1, CollectiveEpilogueActivation>;
};

} // namespace cutlass::gemm::device

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::benchmark {

template <class DualGemmConfig>
struct BenchmarkRunnerDualGemm {

  using GemmKernel = typename DualGemmConfig::GemmKernel;

  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideC = typename GemmKernel::StrideC;
  using StrideD = typename GemmKernel::StrideD;

  using ElementA = typename GemmKernel::ElementA;
  using ElementB = typename GemmKernel::ElementB;
  using ElementC = typename GemmKernel::ElementC;

  using CollectiveEpilogue0 = typename GemmKernel::CollectiveEpilogue0;
  using ElementOutput = typename CollectiveEpilogue0::ElementOutput;
  using ElementCompute = typename CollectiveEpilogue0::ElementCompute;

  using ProblemShapeType = typename GemmKernel::ProblemShape;
  using EpilogueArguments0 = typename GemmKernel::EpilogueArguments0;
  using EpilogueArguments1 = typename GemmKernel::EpilogueArguments1;

  StrideA stride_A;
  StrideB stride_B;
  StrideC stride_C;
  StrideD stride_D;
  uint64_t seed = 0;

  DeviceAllocation<ElementA> block_A;
  DeviceAllocation<ElementB> block_B0;
  DeviceAllocation<ElementB> block_B1;
  DeviceAllocation<ElementC> block_C0;
  DeviceAllocation<ElementC> block_C1;
  DeviceAllocation<ElementOutput> block_D0;
  DeviceAllocation<ElementOutput> block_D1;
  DeviceAllocation<ElementOutput> block_D2;

  BenchmarkRunnerDualGemm() : seed(0) {}

  void initialize(const ProblemShapeType& problem_size) {
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto [M, N, K, L] = problem_shape_MNKL;

    stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
    stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, L));
    stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, L));
    stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, L));

    block_A.reset(static_cast<std::size_t>(M) * K * L);
    block_B0.reset(static_cast<std::size_t>(K) * N * L);
    block_B1.reset(static_cast<std::size_t>(K) * N * L);
    block_C0.reset(static_cast<std::size_t>(M) * N * L);
    block_C1.reset(static_cast<std::size_t>(M) * N * L);
    block_D0.reset(static_cast<std::size_t>(M) * N * L);
    block_D1.reset(static_cast<std::size_t>(M) * N * L);
    block_D2.reset(static_cast<std::size_t>(M) * N * L);

    initialize_block(block_A, seed + 2023);
    initialize_block(block_B0, seed + 2022);
    initialize_block(block_B1, seed + 2021);
    initialize_block(block_C0, seed + 2020);
    initialize_block(block_C1, seed + 2019);
  }

  typename GemmKernel::Arguments get_arguments(
      const GEMMOptions& options, const cutlass::KernelHardwareInfo& hw_info) {
    ProblemShapeType problem_size = ProblemShapeType{options.m, options.n, options.k, options.l};

    EpilogueArguments0 epilogue_arguments0{
        {ElementCompute(options.alpha), ElementCompute(options.beta)},
        block_C0.get(), stride_C, block_D0.get(), stride_D};
    EpilogueArguments1 epilogue_arguments1{
        {ElementCompute(options.alpha), ElementCompute(options.beta)},
        block_C1.get(), stride_C, block_D1.get(), stride_D};

    typename GemmKernel::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem_size,
        {block_A.get(), stride_A, block_B0.get(), stride_B, block_B1.get(), stride_B},
        epilogue_arguments0,
        epilogue_arguments1,
        {block_D2.get(), stride_D},
        hw_info};

    return arguments;
  }

  // GemmUniversalAdapter does not support Dual GEMM, so launch the kernel manually (mirrors
  // ExampleRunner::run in example 07_bmg_dual_gemm).
  static cutlass::Status launch(typename GemmKernel::Params params) {
    dim3 const block = GemmKernel::get_block_shape();
    dim3 const grid = GemmKernel::get_grid_shape(params);
    int smem_size = GemmKernel::SharedStorageSize;

    const auto sycl_block = compat::dim3(block.x, block.y, block.z);
    const auto sycl_grid = compat::dim3(grid.x, grid.y, grid.z);

#if !defined(SYCL_EXT_ONEAPI_WORK_GROUP_SCRATCH_MEMORY)
    using namespace compat::experimental;
#if defined(CUTLASS_SYCL_PROFILING_ENABLED)
    auto event = launch<cutlass::device_kernel<GemmKernel>>(
        launch_policy{sycl_grid, sycl_block, local_mem_size{static_cast<std::size_t>(smem_size)},
                      kernel_properties{sycl_exp::sub_group_size<GemmKernel::DispatchPolicy::SubgroupSize>}},
        params);
    EventManager::getInstance().addEvent(event);
#else
    launch<cutlass::device_kernel<GemmKernel>, sycl::detail::auto_name, false>(
        launch_policy{sycl_grid, sycl_block, local_mem_size{static_cast<std::size_t>(smem_size)},
                      kernel_properties{sycl_exp::sub_group_size<GemmKernel::DispatchPolicy::SubgroupSize>}},
        params);
#endif
#else
    compat::experimental::launch_properties launch_props{
      sycl::ext::oneapi::experimental::work_group_scratch_size(smem_size)
    };
    compat::experimental::kernel_properties kernel_props{
      sycl::ext::oneapi::experimental::sub_group_size<GemmKernel::DispatchPolicy::SubgroupSize>
    };
    compat::experimental::launch_policy policy{sycl_grid, sycl_block, launch_props, kernel_props};
#if defined(CUTLASS_SYCL_PROFILING_ENABLED)
    auto event = compat::experimental::launch<cutlass::device_kernel<GemmKernel>, GemmKernel>(policy, params);
    EventManager::getInstance().addEvent(event);
#else
    compat::experimental::launch<cutlass::device_kernel<GemmKernel>, GemmKernel, false>(policy, params);
#endif
#endif
    return cutlass::Status::kSuccess;
  }

  void run(::benchmark::State& state, const GEMMOptions& options,
           const cutlass::KernelHardwareInfo& hw_info) {
    ProblemShapeType problem_size = ProblemShapeType{options.m, options.n, options.k, options.l};

    initialize(problem_size);

    auto arguments = get_arguments(options, hw_info);

    if (!GemmKernel::can_implement(arguments)) {
      state.SkipWithError("Dual GEMM unable to implement given args.");
      return;
    }

    size_t workspace_size = GemmKernel::get_workspace_size(arguments);
    device_memory::allocation<uint8_t> workspace(workspace_size);

    if (GemmKernel::initialize_workspace(arguments, workspace.get()) != cutlass::Status::kSuccess) {
      state.SkipWithError("Dual GEMM failed to initialize workspace.");
      return;
    }

    typename GemmKernel::Params params = GemmKernel::to_underlying_arguments(arguments, workspace.get());

#ifdef CUTLASS_TEST_FOR_CRI
    // skip warmup on the CRI simulator (it is expensive)
#else
    launch(params);
    compat::wait();
#endif

    state.counters["m"] = options.m;
    state.counters["n"] = options.n;
    state.counters["k"] = options.k;
    state.counters["l"] = options.l;

    // Dual GEMM performs two M x N x K GEMMs sharing the A operand.
    auto gflop = 2.0 * (2.0 * options.m * options.n * options.k * options.l) * 1e-9;

    initialize_counters(state);
    for (auto _ : state) {
      GPU_Clock timer;
      timer.start();
      launch(params);
      auto ms_elapsed = timer.milliseconds();
      update_counters(state, ms_elapsed);
      state.SetIterationTime(ms_elapsed / 1000);
    }
    finalize_counters(state, gflop);
  }

private:
  static void initialize_counters(::benchmark::State& state) {
    state.counters["avg_runtime_ms"] = 0;
    state.counters["total_runtime_ms"] = 0;
    state.counters["best_runtime_ms"] = std::numeric_limits<double>::max();
    state.counters["worst_runtime_ms"] = std::numeric_limits<double>::lowest();
  }

  static void update_counters(::benchmark::State& state, double ms_elapsed) {
    state.PauseTiming();
    state.counters["total_runtime_ms"] += ms_elapsed;
    state.counters["best_runtime_ms"] = std::min<double>(state.counters["best_runtime_ms"], ms_elapsed);
    state.counters["worst_runtime_ms"] = std::max<double>(state.counters["worst_runtime_ms"], ms_elapsed);
    state.ResumeTiming();
  }

  static void finalize_counters(::benchmark::State& state, double gflop) {
    state.counters["avg_runtime_ms"] =
        (state.counters["total_runtime_ms"] - state.counters["best_runtime_ms"] - state.counters["worst_runtime_ms"]) /
        static_cast<double>(state.iterations() - 2);
    state.counters["avg_tflops"] = gflop / state.counters["avg_runtime_ms"];
    state.counters["best_tflop"] = gflop / state.counters["best_runtime_ms"];
  }
};

} // namespace cutlass::benchmark

#define CUTLASS_CREATE_DUAL_GEMM_BENCHMARK(F)                    \
  static void F##_func(                                          \
      ::benchmark::State& state,                                 \
      cutlass::benchmark::GEMMOptions const& options,            \
      cutlass::KernelHardwareInfo const& hw_info) {              \
    auto bench = cutlass::benchmark::BenchmarkRunnerDualGemm<F>();\
    bench.run(state, options, hw_info);                          \
  }
