/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
/*! \file
    \brief BMG INT4 GEMM with K-blocked A and B dequantization.

    A and B use signed INT4 inputs. A has one scale per M row and K block, while B has one
    scale per N channel and K block. The mainloop drains each INT32 K-block accumulator into
    an FP32 fragment before the epilogue applies alpha and beta.
*/

#include "cutlass/epilogue/collective/xe_epilogue.hpp"
#include "cutlass/epilogue/fusion/callbacks.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/initialize_block.hpp"

#include <cute/tensor.hpp>
#include <random>
#include <vector>

#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "sycl_common.hpp"
#include "helper.h"

using namespace cute;

struct Options {
  bool help = false;
  bool error = false;
  int m = 512;
  int n = 512;
  int k = 512;
  int l = 1;
  int iterations = 20;
  int verify = 1;
  float alpha = 1.0f;
  float beta = 0.0f;

  void parse(int argc, char const** args) {
    cutlass::CommandLine cmd(argc, args);
    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }
    cmd.get_cmd_line_argument("m", m, m);
    cmd.get_cmd_line_argument("n", n, n);
    cmd.get_cmd_line_argument("k", k, k);
    cmd.get_cmd_line_argument("l", l, l);
    cmd.get_cmd_line_argument("alpha", alpha, alpha);
    cmd.get_cmd_line_argument("beta", beta, beta);
    cmd.get_cmd_line_argument("iterations", iterations, iterations);
    cmd.get_cmd_line_argument("verify", verify, verify);
  }

  std::ostream& print_usage(std::ostream& out) const {
    out << "BMG integer K-block-scaled GEMM\n\n"
        << "Options:\n\n"
        << "  --help                      If specified, displays this usage statement\n"
        << "  --m=<int>                   Sets the M extent of the GEMM\n"
        << "  --n=<int>                   Sets the N extent of the GEMM\n"
        << "  --k=<int>                   Sets the K extent of the GEMM\n"
        << "  --l=<int>                   Sets the L extent (batch count) of the GEMM\n"
        << "  --alpha=<float>             Epilogue scalar alpha\n"
        << "  --beta=<float>              Epilogue scalar beta\n"
        << "  --iterations=<int>          Iterations\n"
        << "  --verify=<int>              Specify whether to verify\n\n";
    return out;
  }
};

template <class Gemm, int GroupK>
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
  using ElementOutput = typename Gemm::ElementD;
  using ElementCompute = typename Gemm::CollectiveEpilogue::ElementCompute;
  using ElementScale = float;
  using StrideScaleA = Stride<_1, int64_t, int64_t>;
  using StrideScaleB = Stride<_1, int64_t, int64_t>;
  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;

  StrideA stride_A;
  StrideB stride_B;
  StrideC stride_C;
  StrideD stride_D;
  StrideScaleA stride_scale_A;
  StrideScaleB stride_scale_B;
  uint64_t seed = 0;

  cutlass::DeviceAllocation<ElementA> block_A;
  cutlass::DeviceAllocation<ElementB> block_B;
  cutlass::DeviceAllocation<ElementOutput> block_C;
  cutlass::DeviceAllocation<ElementOutput> block_D;
  cutlass::DeviceAllocation<ElementScale> block_scale_A;
  cutlass::DeviceAllocation<ElementScale> block_scale_B;
  cutlass::DeviceAllocation<int32_t> block_ref_accum;
  cutlass::DeviceAllocation<ElementOutput> block_ref_D;

  std::vector<ElementOutput> host_C;
  std::vector<ElementScale> host_scale_A;
  std::vector<ElementScale> host_scale_B;

  void initialize(ProblemShapeType const& problem_size) {
    auto [M, N, K, L] = cute::append<4>(problem_size, 1);
    int scale_k = K / GroupK;

    stride_A = cutlass::make_cute_packed_stride(StrideA{}, make_shape(M, K, L));
    stride_B = cutlass::make_cute_packed_stride(StrideB{}, make_shape(N, K, L));
    stride_C = cutlass::make_cute_packed_stride(StrideC{}, make_shape(M, N, L));
    stride_D = cutlass::make_cute_packed_stride(StrideD{}, make_shape(M, N, L));
    stride_scale_A = cutlass::make_cute_packed_stride(
        StrideScaleA{}, make_shape(M, scale_k, L));
    stride_scale_B = cutlass::make_cute_packed_stride(
        StrideScaleB{}, make_shape(N, scale_k, L));

      if (L > 1) {
        get<2>(stride_A) = static_cast<int64_t>(M) * K * sizeof_bits_v<ElementA> / 8;
        get<2>(stride_B) = static_cast<int64_t>(N) * K * sizeof_bits_v<ElementB> / 8;
    }

    const std::size_t elements_A = static_cast<std::size_t>(M) * K * L;
    const std::size_t elements_B = static_cast<std::size_t>(N) * K * L;
    const std::size_t elements_CD = static_cast<std::size_t>(M) * N * L;
    const std::size_t elements_scale_A = static_cast<std::size_t>(M) * scale_k * L;
    const std::size_t elements_scale_B = static_cast<std::size_t>(N) * scale_k * L;

    block_A.reset(elements_A);
    block_B.reset(elements_B);
    block_C.reset(elements_CD);
    block_D.reset(elements_CD);
    block_scale_A.reset(elements_scale_A);
    block_scale_B.reset(elements_scale_B);
    block_ref_accum.reset(elements_CD);
    block_ref_D.reset(elements_CD);

    host_C.resize(elements_CD);
    host_scale_A.resize(elements_scale_A);
    host_scale_B.resize(elements_scale_B);

    initialize_block(block_A, seed + 2023);
    initialize_block(block_B, seed + 2022);
    initialize_block(block_C, seed + 2021);
    block_C.copy_to_host(host_C.data());
    compat::wait();

    for (int batch = 0; batch < L; ++batch) {
      for (int block = 0; block < scale_k; ++block) {
        for (int m = 0; m < M; ++m) {
          host_scale_A[(static_cast<std::size_t>(batch) * scale_k + block) * M + m] =
              0.015625f * static_cast<float>(1 + ((m + 3 * block + batch) % 7));
        }
        for (int n = 0; n < N; ++n) {
          host_scale_B[(static_cast<std::size_t>(batch) * scale_k + block) * N + n] =
              0.015625f * static_cast<float>(1 + ((n + 5 * block + batch) % 9));
        }
      }
    }

    block_scale_A.copy_from_host(host_scale_A.data());
    block_scale_B.copy_from_host(host_scale_B.data());
  }

  bool verify(ProblemShapeType const& problem_size, Options const& options) {
    auto [M, N, K, L] = problem_size;
    const int scale_k = K / GroupK;
    std::vector<int32_t> host_accum(static_cast<std::size_t>(M) * N * L);
    std::vector<float> host_expected(static_cast<std::size_t>(M) * N * L);

    cutlass::TensorRef ref_A(block_A.get(), LayoutA::packed({M, K}));
    cutlass::TensorRef ref_B(block_B.get(), LayoutB::packed({K, N}));
    cutlass::TensorRef ref_C(block_C.get(), LayoutC::packed({M, N}));
    cutlass::TensorRef ref_accum(block_ref_accum.get(), LayoutD::packed({M, N}));

    for (int block = 0; block < scale_k; ++block) {
      auto block_ref_A = ref_A;
      auto block_ref_B = ref_B;
      block_ref_A.add_pointer_offset(block * GroupK);
      block_ref_B.add_pointer_offset(block * GroupK);

      cutlass::reference::device::GemmComplex(
          {M, N, GroupK},
          1.0f,
          block_ref_A,
          cutlass::ComplexTransform::kNone,
          block_ref_B,
          cutlass::ComplexTransform::kNone,
          0.0f,
          ref_C,
          ref_accum,
          int32_t(0),
          L,
          M * K,
          K * N,
          M * N,
          M * N);
      compat::wait();
      cutlass::device_memory::copy_to_host(host_accum.data(), block_ref_accum.get(), host_accum.size());
      compat::wait();

      for (int batch = 0; batch < L; ++batch) {
        for (int m = 0; m < M; ++m) {
          const float scale_a = host_scale_A[
              (static_cast<std::size_t>(batch) * scale_k + block) * M + m];
          for (int n = 0; n < N; ++n) {
            const std::size_t index =
                (static_cast<std::size_t>(batch) * M + m) * N + n;
            const float scale_b = host_scale_B[
                (static_cast<std::size_t>(batch) * scale_k + block) * N + n];
            host_expected[index] +=
              options.alpha * scale_a * scale_b * host_accum[index];
          }
        }
      }
    }

    for (std::size_t index = 0; index < host_expected.size(); ++index) {
      host_expected[index] += options.beta * static_cast<float>(host_C[index]);
    }
    std::vector<ElementOutput> host_expected_output(host_expected.size());
    for (std::size_t index = 0; index < host_expected.size(); ++index) {
      host_expected_output[index] = static_cast<ElementOutput>(host_expected[index]);
    }
    block_ref_D.copy_from_host(host_expected_output.data());
    compat::wait();
    return cutlass::reference::device::BlockCompareRelativelyEqual(
      block_ref_D.get(), block_D.get(), block_D.size(), ElementOutput(1e-2f), ElementOutput(1e-4f));
  }

  cutlass::Status run(Options const& options, cutlass::KernelHardwareInfo const& hw_info) {
    ProblemShapeType problem_size{options.m, options.n, options.k, options.l};
    if (options.k % GroupK != 0) {
      std::cerr << "K must be divisible by the block size " << GroupK << std::endl;
      return cutlass::Status::kErrorInvalidProblem;
    }
    initialize(problem_size);

    typename Gemm::GemmKernel::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem_size,
        {block_A.get(), stride_A, block_B.get(), stride_B,
         block_scale_A.get(), stride_scale_A, block_scale_B.get(), stride_scale_B},
        {{options.alpha, options.beta}, block_C.get(), stride_C, block_D.get(), stride_D},
        hw_info};

    Gemm gemm_op;
    cutlass::device_memory::allocation<uint8_t> workspace(Gemm::get_workspace_size(arguments));
    if (gemm_op.can_implement(arguments) != cutlass::Status::kSuccess) {
      std::cerr << "Invalid Problem Size: "
                << options.m << 'x' << options.n << 'x' << options.k << 'x' << options.l << std::endl;
      return cutlass::Status::kErrorInvalidProblem;
    }

    CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));
    CUTLASS_CHECK(gemm_op.run());
    compat::wait();

    if (options.verify != 0) {
      bool passed = verify(problem_size, options);
      std::cout << "Disposition: " << (passed ? "Passed" : "Failed") << std::endl;
      if (!passed) {
        return cutlass::Status::kErrorInternal;
      }
    } else {
      std::cout << "Disposition is skipped." << std::endl;
    }

    if (options.iterations > 0) {
      GPU_Clock timer;
      timer.start();
      for (int i = 0; i < options.iterations; ++i) {
        CUTLASS_CHECK(gemm_op.run());
      }
      compat::wait();
      float elapsed = timer.seconds() / options.iterations;
      double operations = (2.0 * options.m * options.n * options.k * options.l) * 1e-12;
      std::cout << "Problem Size: "
                << options.m << 'x' << options.n << 'x' << options.k << 'x' << options.l << std::endl;
      printf("Cutlass GEMM Performance:     [%4.3f]TOPS    (%6.4f)ms\n",
             operations / elapsed, elapsed * 1000);
    }
    return cutlass::Status::kSuccess;
  }
};

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

#if defined(CUTLASS_BMG_GEMM_INT8_BLOCK_SCALED)
  using ElementA = int8_t;
  using ElementB = int8_t;
  using BlockScaledMmaAtom = XE_DPAS_TT_INT8_BLOCK_SCALED<8>;
#else
  using ElementA = int4_t;
  using ElementB = int4_t;
  using BlockScaledMmaAtom = XE_DPAS_TT_INT4_BLOCK_SCALED<8>;
#endif
#if defined(CUTLASS_BMG_GEMM_BLOCK_SCALED_BF16) || \
  defined(CUTLASS_BMG_GEMM_INT4_BLOCK_SCALED_BF16)
  using ElementOutput = bfloat16_t;
#else
  using ElementOutput = float;
#endif
  using ElementCompute = float;
  using ElementScale = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;
#if defined(CUTLASS_BMG_GEMM_INT8_BLOCK_SCALED)
  using TileShape = Shape<_256, _128, _64>;
  using SubgroupLayout = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;
#else
  using TileShape = Shape<_128, _256, _128>;
  using SubgroupLayout = Layout<Shape<_4, _8, _1>, Stride<_8, _1, _0>>;
#endif
  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<BlockScaledMmaAtom>,
      Layout<TileShape>,
      SubgroupLayout>::TiledMMA;
  using GmemTiledCopyA = void;
  using GmemTiledCopyB = void;
  using GmemTiledCopyScaleA = void;
  using GmemTiledCopyScaleB = void;

#if defined(CUTLASS_BMG_GEMM_BLOCK_SCALED_256) || \
  defined(CUTLASS_BMG_GEMM_INT4_BLOCK_SCALED_256)
  constexpr int GroupK = 256;
#else
  constexpr int GroupK = 128;
#endif

  constexpr int PipelineStages = 2;
  using MainloopDispatch = cutlass::gemm::MainloopIntelXeXMX16IntBlockScaled<
      PipelineStages, GroupK>;
  using EpilogueDispatch = cutlass::epilogue::IntelXeGeneric;
  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementOutput, ElementCompute, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
  using FusionCallbacks = cutlass::epilogue::fusion::FusionCallbacks<
      EpilogueDispatch, EpilogueOp, TileShape, decltype(tile_shape(TiledMma()))>;
  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
      EpilogueDispatch,
      TileShape,
      void,
      ElementOutput,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      ElementOutput,
      cutlass::gemm::TagToStrideC_t<LayoutD>,
      FusionCallbacks,
      void,
      void>;
  using StrideScaleA = Stride<_1, int64_t, int64_t>;
  using StrideScaleB = Stride<_1, int64_t, int64_t>;
  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
      MainloopDispatch,
      TileShape,
      cute::tuple<ElementA, ElementScale>,
      cute::tuple<cutlass::gemm::TagToStrideA_t<LayoutA>, StrideScaleA>,
      cute::tuple<ElementB, ElementScale>,
      cute::tuple<cutlass::gemm::TagToStrideB_t<LayoutB>, StrideScaleB>,
      TiledMma,
      cute::tuple<GmemTiledCopyA, GmemTiledCopyScaleA>,
      void,
      void,
      cute::identity,
      cute::tuple<GmemTiledCopyB, GmemTiledCopyScaleB>,
      void,
      void,
      cute::identity>;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
  CUTLASS_CHECK((ExampleRunner<Gemm, GroupK>{}).run(options, hw_info));
  return 0;
}
