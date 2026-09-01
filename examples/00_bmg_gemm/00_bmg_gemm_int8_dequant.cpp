/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
/*! \file
    \brief BMG INT8 GEMM with epilogue dequantization.

    The GEMM computes an int32 accumulator from int8 A and B.  The epilogue then
    applies a per-tensor scale for A and a per-channel scale for B before adding
    the C tensor:

      D(m,n) = alpha * scale_a * scale_b(n) * acc(m,n) + beta * C(m,n)

    The fp32 and bf16 C/D variants are built from this source file. The per-row-A
    variants use an M-element scale_a vector instead of the scalar scale_a.
*/

#include "cutlass/epilogue/collective/xe_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/util/GPU_Clock.hpp"

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

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::epilogue::fusion {

template <
  class ElementOutput_,
  class ElementCompute_,
  class ElementSource_ = ElementOutput_,
  class ElementScalar_ = ElementCompute_,
  int AlignmentScale_ = 128 / sizeof_bits_v<ElementScalar_>,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct BmgInt8Dequant : FusionOperation {
  using ElementOutput = ElementOutput_;
  using ElementCompute = ElementCompute_;
  using ElementSource = ElementSource_;
  using ElementScalar = ElementScalar_;
  static constexpr int AlignmentScalar = AlignmentScale_;
  static constexpr bool IsSourceSupported = true;
  static constexpr bool IsPerColScaleSupported = true;
  static constexpr auto RoundStyle = RoundStyle_;
};

template <
  class ElementOutput_,
  class ElementCompute_,
  class ElementSource_ = ElementOutput_,
  class ElementScalar_ = ElementCompute_,
  int AlignmentScale_ = 128 / sizeof_bits_v<ElementScalar_>,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct BmgInt8DequantPerRowA : FusionOperation {
  using ElementOutput = ElementOutput_;
  using ElementCompute = ElementCompute_;
  using ElementSource = ElementSource_;
  using ElementScalar = ElementScalar_;
  static constexpr int AlignmentScalar = AlignmentScale_;
  static constexpr bool IsSourceSupported = true;
  static constexpr bool IsPerRowScaleSupported = true;
  static constexpr bool IsPerColScaleSupported = true;
  static constexpr auto RoundStyle = RoundStyle_;
};

template <
  class CtaTileShapeMNK,
  class ElementOutput,
  class ElementCompute,
  class ElementSource,
  class ElementScalar,
  int AlignmentScale,
  FloatRoundStyle RoundStyle
>
using BmgInt8DequantVisitor =
  XeEVT<
    XeCompute<homogeneous_multiply_add, ElementOutput, ElementCompute, RoundStyle>,
    XeScalarBroadcast<ElementScalar, Stride<_0, _0, int64_t>>,
    XeSrcFetch<ElementSource>,
    XeEVT<
      XeCompute<multiplies, ElementCompute, ElementCompute, RoundStyle>,
      XeRowBroadcast<
        0,
        CtaTileShapeMNK,
        ElementScalar,
        ElementCompute,
        Stride<_0, _1, int64_t>,
        AlignmentScale>,
      XeEVT<
        XeCompute<multiplies, ElementCompute, ElementCompute, RoundStyle>,
        XeScalarBroadcast<ElementScalar, Stride<_0, _0, int64_t>, 2>,
        XeAccFetch
      >
    >
  >;

template <
  class CtaTileShapeMNK,
  class ElementOutput,
  class ElementCompute,
  class ElementSource,
  class ElementScalar,
  int AlignmentScale,
  FloatRoundStyle RoundStyle
>
using BmgInt8DequantPerRowAVisitor =
  XeEVT<
    XeCompute<homogeneous_multiply_add, ElementOutput, ElementCompute, RoundStyle>,
    XeScalarBroadcast<ElementScalar, Stride<_0, _0, int64_t>>,
    XeSrcFetch<ElementSource>,
    XeEVT<
      XeCompute<multiplies, ElementCompute, ElementCompute, RoundStyle>,
      XeColBroadcast<
        0,
        CtaTileShapeMNK,
        ElementScalar,
        ElementCompute,
        Stride<_1, _0, int64_t>,
        AlignmentScale>,
      XeEVT<
        XeCompute<multiplies, ElementCompute, ElementCompute, RoundStyle>,
        XeRowBroadcast<
          0,
          CtaTileShapeMNK,
          ElementScalar,
          ElementCompute,
          Stride<_0, _1, int64_t>,
          AlignmentScale>,
        XeEVT<
          XeCompute<multiplies, ElementCompute, ElementCompute, RoundStyle>,
          XeScalarBroadcast<ElementScalar, Stride<_0, _0, int64_t>>,
          XeAccFetch
        >
      >
    >
  >;

template <
  class ElementOutput_,
  class ElementCompute_,
  class ElementSource_,
  class ElementScalar_,
  int AlignmentScale_,
  FloatRoundStyle RoundStyle_,
  class CtaTileShapeMNK_,
  class EpilogueTile_
>
struct FusionCallbacks<
    epilogue::IntelXeGeneric,
    fusion::BmgInt8Dequant<
      ElementOutput_, ElementCompute_, ElementSource_, ElementScalar_, AlignmentScale_, RoundStyle_>,
    CtaTileShapeMNK_,
    EpilogueTile_
> : BmgInt8DequantVisitor<
      CtaTileShapeMNK_,
      typename cutlass::detail::get_unpacked_element_type<ElementOutput_>::type,
      ElementCompute_,
      ElementSource_,
      ElementScalar_,
      AlignmentScale_,
      RoundStyle_
    > {

  using ElementOutput = ElementOutput_;
  using ElementCompute = ElementCompute_;
  using ElementSource = ElementSource_;
  using ElementScalar = ElementScalar_;
  using Operation = fusion::BmgInt8Dequant<
    ElementOutput_, ElementCompute_, ElementSource_, ElementScalar_, AlignmentScale_, RoundStyle_>;
  using Impl = BmgInt8DequantVisitor<
    CtaTileShapeMNK_,
    typename cutlass::detail::get_unpacked_element_type<ElementOutput_>::type,
    ElementCompute_,
    ElementSource_,
    ElementScalar_,
    AlignmentScale_,
    RoundStyle_
  >;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar scale_a = ElementScalar(1);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;
    ElementScalar const* scale_a_ptr = nullptr;

    ElementScalar const* scale_b_ptr = nullptr;

    using StrideAlpha = Stride<_0, _0, int64_t>;
    using StrideBeta = Stride<_0, _0, int64_t>;
    using StrideScaleB = Stride<_0, _1, int64_t>;

    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta dBeta = {_0{}, _0{}, 0};
    StrideAlpha dScaleA = {_0{}, _0{}, 0};
    StrideScaleB dScaleB = {_0{}, _1{}, 0};

    operator typename Impl::Arguments() const {
      return {
        {{beta}, {beta_ptr}, {dBeta}},
        {},
        {
          {scale_b_ptr, ElementScalar(1), dScaleB},
          {
            {{scale_a, alpha}, {scale_a_ptr, alpha_ptr}, {dScaleA, dAlpha}},
            {},
            {}
          },
          {}
        },
        {}
      };
    }
  };

  using Impl::Impl;
};

template <
  class ElementOutput_,
  class ElementCompute_,
  class ElementSource_,
  class ElementScalar_,
  int AlignmentScale_,
  FloatRoundStyle RoundStyle_,
  class CtaTileShapeMNK_,
  class EpilogueTile_
>
struct FusionCallbacks<
    epilogue::IntelXeGeneric,
    fusion::BmgInt8DequantPerRowA<
      ElementOutput_, ElementCompute_, ElementSource_, ElementScalar_, AlignmentScale_, RoundStyle_>,
    CtaTileShapeMNK_,
    EpilogueTile_
> : BmgInt8DequantPerRowAVisitor<
      CtaTileShapeMNK_,
      typename cutlass::detail::get_unpacked_element_type<ElementOutput_>::type,
      ElementCompute_,
      ElementSource_,
      ElementScalar_,
      AlignmentScale_,
      RoundStyle_
    > {

  using ElementOutput = ElementOutput_;
  using ElementCompute = ElementCompute_;
  using ElementSource = ElementSource_;
  using ElementScalar = ElementScalar_;
  using Operation = fusion::BmgInt8DequantPerRowA<
    ElementOutput_, ElementCompute_, ElementSource_, ElementScalar_, AlignmentScale_, RoundStyle_>;
  using Impl = BmgInt8DequantPerRowAVisitor<
    CtaTileShapeMNK_,
    typename cutlass::detail::get_unpacked_element_type<ElementOutput_>::type,
    ElementCompute_,
    ElementSource_,
    ElementScalar_,
    AlignmentScale_,
    RoundStyle_
  >;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;
    ElementScalar const* scale_a_ptr = nullptr;
    ElementScalar const* scale_b_ptr = nullptr;

    using StrideAlpha = Stride<_0, _0, int64_t>;
    using StrideBeta = Stride<_0, _0, int64_t>;
    using StrideScaleA = Stride<_1, _0, int64_t>;
    using StrideScaleB = Stride<_0, _1, int64_t>;

    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta dBeta = {_0{}, _0{}, 0};
    StrideScaleA dScaleA = {_1{}, _0{}, 0};
    StrideScaleB dScaleB = {_0{}, _1{}, 0};

    operator typename Impl::Arguments() const {
      return {
        {{beta}, {beta_ptr}, {dBeta}},
        {},
        {
          {scale_a_ptr, ElementScalar(1), dScaleA},
          {
            {scale_b_ptr, ElementScalar(1), dScaleB},
            {
              {{alpha}, {alpha_ptr}, {dAlpha}},
              {},
              {}
            },
            {}
          },
          {}
        },
        {}
      };
    }
  };

  using Impl::Impl;
};

} // namespace cutlass::epilogue::fusion

///////////////////////////////////////////////////////////////////////////////////////////////////

struct Options {
  bool help;
  bool error;

  int m, n, k, l, iterations, verify;
  float alpha, beta, scale_a;

  Options()
    : help(false),
      error(false),
      m(5120),
      n(4096),
      k(4096),
      l(1),
      iterations(100),
      verify(1),
      alpha(1.f),
      beta(0.f),
      scale_a(0.125f) {}

  void parse(int argc, char const** args) {
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
    cmd.get_cmd_line_argument("scale-a", scale_a, 0.125f);
    cmd.get_cmd_line_argument("iterations", iterations, 100);
    cmd.get_cmd_line_argument("verify", verify, 1);
  }

  std::ostream& print_usage(std::ostream& out) const {
    out << "BMG INT8 GEMM with epilogue dequantization\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --m=<int>                   Sets the M extent of the GEMM\n"
      << "  --n=<int>                   Sets the N extent of the GEMM\n"
      << "  --k=<int>                   Sets the K extent of the GEMM\n"
      << "  --l=<int>                   Sets the L extent (batch count) of the GEMM\n"
      << "  --alpha=<float>             Additional epilogue scale\n"
      << "  --beta=<float>              Epilogue C scale\n"
    #if defined(CUTLASS_BMG_GEMM_INT8_DEQUANT_PER_ROW_A)
      << "  --scale-a=<float>           Base value for the generated A per-row scales\n"
    #else
      << "  --scale-a=<float>           A per-tensor dequantization scale\n"
    #endif
      << "  --iterations=<int>          Iterations\n"
      << "  --verify=<int>              Specify whether to verify\n\n";
    return out;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <class Gemm>
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
  using ElementC = typename Gemm::ElementC;
  using ElementOutput = typename CollectiveEpilogue::ElementOutput;
  using ElementCompute = typename CollectiveEpilogue::ElementCompute;
  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;

  using ElementScale = float;
  using StrideScaleB = typename CollectiveEpilogue::FusionCallbacks::Arguments::StrideScaleB;
#if defined(CUTLASS_BMG_GEMM_INT8_DEQUANT_PER_ROW_A)
  using StrideScaleA = typename CollectiveEpilogue::FusionCallbacks::Arguments::StrideScaleA;
#endif

  StrideA stride_A;
  StrideB stride_B;
  StrideC stride_C;
  StrideD stride_D;
  uint64_t seed = 0;

  cutlass::DeviceAllocation<ElementA> block_A;
  cutlass::DeviceAllocation<ElementB> block_B;
  cutlass::DeviceAllocation<ElementC> block_C;
  cutlass::DeviceAllocation<ElementScale> block_scale_B;
#if defined(CUTLASS_BMG_GEMM_INT8_DEQUANT_PER_ROW_A)
  cutlass::DeviceAllocation<ElementScale> block_scale_A;
#endif
  cutlass::DeviceAllocation<ElementOutput> block_D;
  cutlass::DeviceAllocation<ElementAccumulator> block_ref_accum;
  cutlass::DeviceAllocation<ElementOutput> block_ref_D;

  std::vector<ElementA> host_A;
  std::vector<ElementB> host_B;
  std::vector<ElementC> host_C;
  std::vector<ElementScale> host_scale_B;
#if defined(CUTLASS_BMG_GEMM_INT8_DEQUANT_PER_ROW_A)
  std::vector<ElementScale> host_scale_A;
#endif

  void initialize(const ProblemShapeType& problem_size, float scale_a) {
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto [M, N, K, L] = problem_shape_MNKL;

    stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
    stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, L));
    stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, L));
    stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, L));

    std::size_t elements_A = static_cast<std::size_t>(M) * K * L;
    std::size_t elements_B = static_cast<std::size_t>(K) * N * L;
    std::size_t elements_C = static_cast<std::size_t>(M) * N * L;

    block_A.reset(elements_A);
    block_B.reset(elements_B);
    block_C.reset(elements_C);
    block_scale_B.reset(static_cast<std::size_t>(N));
  #if defined(CUTLASS_BMG_GEMM_INT8_DEQUANT_PER_ROW_A)
    block_scale_A.reset(static_cast<std::size_t>(M));
  #endif
    block_D.reset(elements_C);
    block_ref_accum.reset(elements_C);
    block_ref_D.reset(elements_C);

    host_A.resize(elements_A);
    host_B.resize(elements_B);
    host_C.resize(elements_C);
    host_scale_B.resize(static_cast<std::size_t>(N));
  #if defined(CUTLASS_BMG_GEMM_INT8_DEQUANT_PER_ROW_A)
    host_scale_A.resize(static_cast<std::size_t>(M));
  #endif

    std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
    std::uniform_int_distribution<int> quantized_distribution(-8, 8);
    std::uniform_real_distribution<float> c_distribution(-1.f, 1.f);

    for (auto& value : host_A) {
      value = static_cast<ElementA>(quantized_distribution(rng));
    }
    for (auto& value : host_B) {
      value = static_cast<ElementB>(quantized_distribution(rng));
    }
    for (auto& value : host_C) {
      value = static_cast<ElementC>(c_distribution(rng));
    }
    for (int n = 0; n < N; ++n) {
      host_scale_B[n] = 0.015625f * static_cast<float>(1 + (n % 7));
    }
#if defined(CUTLASS_BMG_GEMM_INT8_DEQUANT_PER_ROW_A)
    for (int m = 0; m < M; ++m) {
      host_scale_A[m] = scale_a * (0.5f + 0.125f * static_cast<float>(m % 5));
    }
#endif

    block_A.copy_from_host(host_A.data());
    block_B.copy_from_host(host_B.data());
    block_C.copy_from_host(host_C.data());
    block_scale_B.copy_from_host(host_scale_B.data());
#if defined(CUTLASS_BMG_GEMM_INT8_DEQUANT_PER_ROW_A)
    block_scale_A.copy_from_host(host_scale_A.data());
#endif
  }

  bool verify(const ProblemShapeType& problem_size, const Options& options) {
    auto [M, N, K, L] = problem_size;

    cutlass::TensorRef ref_A(block_A.get(), LayoutA::packed({M, K}));
    cutlass::TensorRef ref_B(block_B.get(), LayoutB::packed({K, N}));
    cutlass::TensorRef ref_C(block_C.get(), LayoutC::packed({M, N}));
    cutlass::TensorRef ref_accum(block_ref_accum.get(), LayoutD::packed({M, N}));

    cutlass::reference::device::GemmComplex(
      {M, N, K},
      ElementCompute(1),
      ref_A,
      cutlass::ComplexTransform::kNone,
      ref_B,
      cutlass::ComplexTransform::kNone,
      ElementCompute(0),
      ref_C,
      ref_accum,
      ElementAccumulator(0),
      L,
      M * K,
      K * N,
      M * N,
      M * N);

    compat::wait();

    std::size_t elements_D = static_cast<std::size_t>(M) * N * L;
    std::vector<ElementAccumulator> host_accum(elements_D);
    std::vector<ElementOutput> host_expected(elements_D);
    cutlass::device_memory::copy_to_host(host_accum.data(), block_ref_accum.get(), elements_D);

    for (int l = 0; l < L; ++l) {
      for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
          std::size_t index =
            (static_cast<std::size_t>(l) * M + m) * N + n;
          float scale_a = options.scale_a;
#if defined(CUTLASS_BMG_GEMM_INT8_DEQUANT_PER_ROW_A)
          scale_a = host_scale_A[m];
#endif
          float value =
            options.alpha * scale_a * host_scale_B[n] * static_cast<float>(host_accum[index]);
          value += options.beta * static_cast<float>(host_C[index]);
          host_expected[index] = static_cast<ElementOutput>(value);
        }
      }
    }

    block_ref_D.copy_from_host(host_expected.data());
    compat::wait();

    ElementOutput epsilon(1e-2f);
    ElementOutput non_zero_floor(1e-4f);
    bool passed = cutlass::reference::device::BlockCompareRelativelyEqual(
      block_ref_D.get(), block_D.get(), block_D.size(), epsilon, non_zero_floor);

    return passed;
  }

  cutlass::Status run(const Options& options, const cutlass::KernelHardwareInfo& hw_info) {
    ProblemShapeType problem_size = {options.m, options.n, options.k, options.l};
    initialize(problem_size, options.scale_a);

    typename Gemm::GemmKernel::EpilogueArguments epilogue_arguments{
      {}, block_C.get(), stride_C, block_D.get(), stride_D};
    epilogue_arguments.thread.alpha = static_cast<ElementCompute>(options.alpha);
    epilogue_arguments.thread.beta = static_cast<ElementCompute>(options.beta);
  #if defined(CUTLASS_BMG_GEMM_INT8_DEQUANT_PER_ROW_A)
    epilogue_arguments.thread.scale_a_ptr = block_scale_A.get();
    epilogue_arguments.thread.dScaleA = StrideScaleA{};
  #else
    epilogue_arguments.thread.scale_a = static_cast<ElementCompute>(options.scale_a);
  #endif
    epilogue_arguments.thread.scale_b_ptr = block_scale_B.get();
    epilogue_arguments.thread.dScaleB = StrideScaleB{};

    typename Gemm::GemmKernel::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size,
      {block_A.get(), stride_A, block_B.get(), stride_B},
      epilogue_arguments,
      hw_info};

    Gemm gemm_op;
    std::size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    if (gemm_op.can_implement(arguments) != cutlass::Status::kSuccess) {
      std::cout << "Invalid Problem Size: "
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
    }
    else {
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

#if defined(CUTLASS_BMG_GEMM_INT8_DEQUANT_BF16_C)
  using ElementC = bfloat16_t;
#else
  using ElementC = float;
#endif

  using ElementAccumulator = int32_t;
  using ElementComputeEpilogue = float;
  using ElementInputA = int8_t;
  using ElementInputB = int8_t;
  using ElementOutput = ElementC;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using GmemTiledCopyA = void;
  using GmemTiledCopyB = void;
  using TileShape = Shape<_256, _256, _64>;
  using SubgroupLayout = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;
  using TiledMma = typename TiledMMAHelper<
    MMA_Atom<XE_DPAS_TT<8, ElementAccumulator, ElementInputA, ElementInputB>>,
    Layout<TileShape>,
    SubgroupLayout>::TiledMMA;

  constexpr int PipelineStages = 2;
  using GEMMDispatchPolicy = cutlass::gemm::MainloopXeL1Staged<PipelineStages>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeGeneric;
#if defined(CUTLASS_BMG_GEMM_INT8_DEQUANT_PER_ROW_A)
  using EpilogueOp = cutlass::epilogue::fusion::BmgInt8DequantPerRowA<
    ElementOutput, ElementComputeEpilogue, ElementC, ElementComputeEpilogue>;
#else
  using EpilogueOp = cutlass::epilogue::fusion::BmgInt8Dequant<
    ElementOutput, ElementComputeEpilogue, ElementC, ElementComputeEpilogue>;
#endif
  using FusionCallbacks = cutlass::epilogue::fusion::FusionCallbacks<
    EpilogueDispatchPolicy,
    EpilogueOp,
    TileShape,
    decltype(tile_shape(TiledMma()))>;

  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
    EpilogueDispatchPolicy,
    TileShape,
    void,
    ElementC,
    cutlass::gemm::TagToStrideC_t<LayoutC>,
    ElementOutput,
    cutlass::gemm::TagToStrideC_t<LayoutD>,
    FusionCallbacks,
    void,
    void>;

  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
    GEMMDispatchPolicy,
    TileShape,
    ElementInputA,
    cutlass::gemm::TagToStrideA_t<LayoutA>,
    ElementInputB,
    cutlass::gemm::TagToStrideB_t<LayoutB>,
    TiledMma,
    GmemTiledCopyA,
    void,
    void,
    cute::identity,
    GmemTiledCopyB,
    void,
    void,
    cute::identity>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  ExampleRunner<Gemm> runner;
  CUTLASS_CHECK(runner.run(options, hw_info));
  return 0;
}