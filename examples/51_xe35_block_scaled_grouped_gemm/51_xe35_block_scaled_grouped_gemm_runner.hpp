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
#include <cfloat>
using namespace cute;

///////////////////////////////////////////////////////////////////////////////////////////////////

using namespace cutlass::gemm;
using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int,int,int>>; // <M,N,K> per group
// Command line options parsing
struct Options {

  bool help;
  bool error;

  int m, n, k, l, iterations, groups, verify;
  float alpha, beta;
  std::vector<typename ProblemShape::UnderlyingProblemShape> problem_sizes_host;

  Options():
    help(false),
    error(false),
    m(5120), n(4096), k(4096), l(1), iterations(20),
    groups(2),
    alpha(1.f), beta(0.f)
  {
    problem_sizes_host.reserve(groups);
    for(int i = 0; i < groups; i++) {
      problem_sizes_host.push_back({m, n, k});
    }
  }

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
    cmd.get_cmd_line_argument("groups", groups, 2);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("iterations", iterations, 100);
    cmd.get_cmd_line_argument("verify", verify, 1);

    assert(groups >= 2);
    problem_sizes_host.clear();
    problem_sizes_host.reserve(groups);
    for(int i = 0; i < groups; i++) {
      problem_sizes_host.push_back({m, n, k});
    }
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
      << "  --groups=<int>              Sets the number of individual GEMM problems for Grouped GEMM\n"
      << "  --alpha=<s32>               Epilogue scalar alpha\n"
      << "  --beta=<s32>                Epilogue scalar beta\n\n"
      << "  --iterations=<int>          Iterations\n\n"
      << "  --verify=<int>              Specify whether to verify.\n\n";

    return out;
  }

  /// Compute performance in TFLOP/s
  double tflops(double runtime_s, std::vector<typename ProblemShape::UnderlyingProblemShape> problem_sizes_host) const
  {
    // Number of real-valued multiply-adds
    uint64_t fmas = uint64_t();

    for (auto const & problem : problem_sizes_host) {
      fmas += static_cast<uint64_t>(get<0>(problem)) *
              static_cast<uint64_t>(get<1>(problem)) *
              static_cast<uint64_t>(get<2>(problem));
    }
    // Two flops per multiply-add
    uint64_t flop = static_cast<uint64_t>(2) * static_cast<uint64_t>(fmas);
    double tflop = double(flop) / double(1.0e12);
    return tflop / runtime_s;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <class DispatchPolicy>
struct RunnerScalePolicy;

template <int Stages, int GroupSize, class KernelSchedule>
struct RunnerScalePolicy<cutlass::gemm::MainloopIntelXeXMX16BlockScaledGroupImpl<Stages, cute::Int<GroupSize>, KernelSchedule>> {
  static constexpr int group_k = GroupSize;
  static constexpr int group_n = 1;
  static constexpr bool has_n_block_scale = false;

  CUTE_HOST_DEVICE
  static int scale_n_extent(int N) {
    return N;
  }

  CUTE_HOST_DEVICE
  static int scale_n_coord(int n) {
    return n;
  }
};

template <int Stages, class GroupSizeM, class GroupSizeN, class GroupSizeK, class KernelSchedule>
struct RunnerScalePolicy<cutlass::gemm::MainloopIntelXeXMX16BlockScaledGroupImpl<Stages, cute::tuple<GroupSizeM, GroupSizeN, GroupSizeK>, KernelSchedule>> {
  static constexpr int group_k = GroupSizeK::value;
  static constexpr int group_n = GroupSizeN::value;
  static constexpr bool has_n_block_scale = true;

  CUTE_HOST_DEVICE
  static int scale_n_extent(int N) {
    return cute::ceil_div(N, group_n);
  }

  CUTE_HOST_DEVICE
  static int scale_n_coord(int n) {
    return n / group_n;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <
  class Gemm
>
struct ExampleRunner {

  using CollectiveMainloop = typename Gemm::CollectiveMainloop;
  using CollectiveEpilogue = typename Gemm::CollectiveEpilogue;

  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;
  using StrideD = typename Gemm::GemmKernel::InternalStrideD;

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

  using StrideScaleA = typename CollectiveMainloop::InternalStrideScaleA;
  using StrideScaleB = typename CollectiveMainloop::InternalStrideScaleB;

  using ElementC = typename Gemm::ElementC;
  using ElementOutput = typename CollectiveEpilogue::ElementOutput;
  using ElementCompute = typename CollectiveEpilogue::ElementCompute;

  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using ScalePolicy = RunnerScalePolicy<DispatchPolicy>;
  static constexpr int scaleGroupK = ScalePolicy::group_k;
  static constexpr int scaleGroupN = ScalePolicy::group_n;
  // 2D block load requires surface width to be 4-byte aligned
  static constexpr int scaleAlignA = cute::ceil_div(4, (int)sizeof(ElementScaleA));
  static constexpr int scaleAlignB = cute::ceil_div(4, (int)sizeof(ElementScaleB));

  //
  // Data members
  //

  /// Initialization
  // // Host-side allocations
  std::vector<ElementAccumulator> alpha_host;
  std::vector<ElementAccumulator> beta_host;

  std::vector<StrideA> stride_A_host;
  std::vector<StrideB> stride_B_host;
  std::vector<StrideScaleA> stride_SFA_host;
  std::vector<StrideScaleB> stride_SFB_host;
  std::vector<StrideC> stride_C_host;
  std::vector<StrideD> stride_D_host;
  // Device-side allocations
  cutlass::DeviceAllocation<typename ProblemShape::UnderlyingProblemShape> problem_sizes;

  cutlass::DeviceAllocation<StrideA> stride_A;
  cutlass::DeviceAllocation<StrideB> stride_B;
  cutlass::DeviceAllocation<StrideC> stride_C;
  cutlass::DeviceAllocation<StrideD> stride_D;
  cutlass::DeviceAllocation<StrideScaleA> stride_SFA;
  cutlass::DeviceAllocation<StrideScaleB> stride_SFB;

  uint64_t seed = 0;

  std::vector<cutlass::DeviceAllocation<ElementA>> block_A;
  std::vector<cutlass::DeviceAllocation<ElementB>> block_B;
  std::vector<cutlass::DeviceAllocation<ElementC>> block_C;
  std::vector<cutlass::DeviceAllocation<ElementScaleA>> block_scaleA;
  std::vector<cutlass::DeviceAllocation<ElementScaleB>> block_scaleB;
  std::vector<cutlass::DeviceAllocation<ElementOutput>> block_D;

  cutlass::DeviceAllocation<const ElementA *> ptr_A;
  cutlass::DeviceAllocation<const ElementB *> ptr_B;
  cutlass::DeviceAllocation<const ElementScaleA *> ptr_SFA;
  cutlass::DeviceAllocation<const ElementScaleB *> ptr_SFB;
  cutlass::DeviceAllocation<const ElementC *> ptr_C;
  cutlass::DeviceAllocation<ElementOutput *> ptr_D;

  cutlass::DeviceAllocation<ElementAccumulator*> alpha_device;
  cutlass::DeviceAllocation<ElementAccumulator*> beta_device;
  cutlass::DeviceAllocation<ElementAccumulator> block_alpha;
  cutlass::DeviceAllocation<ElementAccumulator> block_beta;

  std::vector<cutlass::DeviceAllocation<ElementMMAVerify>> block_A_dq; // Dequantized copy of A for validation
  std::vector<cutlass::DeviceAllocation<ElementMMAVerify>> block_B_dq; // Dequantized copy of B for validation
  std::vector<cutlass::DeviceAllocation<ElementOutput>> block_ref_D;
  //
  // Methods
  //

  /// Populates a Gemm::Arguments structure from the given commandline options
  auto args_from_options(const Options &options, const cutlass::KernelHardwareInfo& hw_info)
  {
    typename Gemm::Arguments arguments;
    decltype(arguments.epilogue.thread) fusion_args;

    if (options.alpha != FLT_MAX && options.beta != FLT_MAX) {
      // If both alpha/beta are provided (via cmd line args) and are scalar, i.e., same alpha/beta applies to all batches.
      fusion_args.alpha = options.alpha;
      fusion_args.beta = options.beta;
      fusion_args.alpha_ptr = nullptr;
      fusion_args.beta_ptr = nullptr;
      fusion_args.alpha_ptr_array = nullptr;
      fusion_args.beta_ptr_array = nullptr;
      // Single alpha and beta for all groups
      fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 0};
      fusion_args.dBeta = {cute::_0{}, cute::_0{}, 0};
    }
    else {
      // If pointers to alpha/beta are provided, i.e., alpha/beta can differ between batches/groups.
      fusion_args.alpha = 0;
      fusion_args.beta = 0;
      fusion_args.alpha_ptr = nullptr;
      fusion_args.beta_ptr = nullptr;
      fusion_args.alpha_ptr_array = alpha_device.get();
      fusion_args.beta_ptr_array = beta_device.get();
      // One alpha and beta per each group
      fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 1};
      fusion_args.dBeta = {cute::_0{}, cute::_0{}, 1};
    }
    using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerXeGroup<ProblemShape>::RasterOrderOptions;

    // Per-GEMM problem shape info may only exist on the device.
    return cute::make_tuple(cutlass::gemm::GemmUniversalMode::kGrouped,
                            typename Gemm::GemmKernel::ProblemShape{options.groups, problem_sizes.get(), options.problem_sizes_host.data()},
                            fusion_args, hw_info,
                            typename Gemm::GemmKernel::TileSchedulerArguments{1, RasterOrderOptions::AlongN});

  }
  bool verify(const Options &options) {
    bool passed = true;
    ElementOutput const epsilon(1e-2f);
    ElementOutput const non_zero_floor(1e-4f);
    for (int i = 0; i < options.groups; i++){
      Shape<int, int, int, int> problem_size = append<4>(options.problem_sizes_host[i], 1);
      auto M = get<0>(problem_size);
      auto N = get<1>(problem_size);
      auto K = get<2>(problem_size);
 
      cutlass::TensorRef ref_A(block_A_dq.at(i).get(), LayoutA::packed({M, K}));
      cutlass::TensorRef ref_B(block_B_dq.at(i).get(), LayoutB::packed({K, N}));
      cutlass::TensorRef ref_C(block_C.at(i).get(), LayoutC::packed({M, N}));
      cutlass::TensorRef ref_D(block_ref_D.at(i).get(), LayoutD::packed({M, N}));

      cutlass::reference::device::GemmComplex(
            {M, N, K},
            alpha_host.at(i),
            ref_A,
            cutlass::ComplexTransform::kNone,
            ref_B,
            cutlass::ComplexTransform::kNone,
            beta_host.at(i),
            ref_C,
            ref_D,
            ElementAccumulator(0),
            1,     // batch_count
            M * K, // batch_stride_A
            K * N, // batch_stride_B
            M * N, // batch_stride_C
            M * N  // batch_stride_D
          );
  
      // CUTLASS on SYCL uses the compatibility library compat for e.g. default in-order queue
      compat::wait();

      // Check if output from CUTLASS kernel and reference kernel are equal or not
      // compare_reference
      passed &= cutlass::reference::device::BlockCompareRelativelyEqual(
        block_ref_D.at(i).get(), block_D.at(i).get(), block_D.at(i).size(), epsilon, non_zero_floor); 
      if (!passed) {
        break;
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
    const float scale_max = max_dequant_val / elt_max_f;
    const float scale_min = min_dequant_val / elt_max_f;
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
  static void apply_scale_A(DstElement* dq_buffer,
                       SrcElement const* q_buffer,
                       Layout const operand_layout,
                       ElementScale const* scale_buffer,
                       ScaleLayout const scale_layout) {
    if constexpr (std::is_same_v<DstElement, SrcElement>) {
      return;
    }

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

          auto scale_data = (ret_type)(scale_tensor(mn, k / scaleGroupK, l));

          dst_tensor(mn, k, l) = (src_data) * scale_data;
        }
      }
    }

    cutlass::device_memory::copy_to_device(dq_buffer, (DstElement*)(raw_pointer_cast(dst_tensor.data())), dst_tensor.size());
    compat::wait();
  }

  template <
  class DstElement,
  class SrcElement,
  class Layout,
  class ElementScale,
  class ScaleLayout>
  static void apply_scale_B(DstElement* dq_buffer,
                       SrcElement const* q_buffer,
                       Layout const operand_layout,
                       ElementScale const* scale_buffer,
                       ScaleLayout const scale_layout) {
    if constexpr (std::is_same_v<DstElement, SrcElement>) {
      return;
    }

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

    auto N = size<0>(src_tensor);
    auto K = size<1>(src_tensor);
    auto L = size<2>(src_tensor);

    using ret_type = float;

    for (int l = 0; l < L; l++) {
      for (int k= 0; k < K; k++) {
        for (int n = 0; n < N; n++) {
          auto src_data = [&]() {
            if constexpr (sizeof_bits_v<SrcElement> >= 8) {
              return  (ret_type)(src_tensor(n, k, l));
            } else {
              return (ret_type)(src_tensor(n, k, l).get());
            }
          }();

          auto scale_data = (ret_type)(scale_tensor(ScalePolicy::scale_n_coord(n), k / scaleGroupK, l));

          dst_tensor(n, k, l) = (src_data) * scale_data;
        }
      }
    }

    cutlass::device_memory::copy_to_device(dq_buffer, (DstElement*)(raw_pointer_cast(dst_tensor.data())), dst_tensor.size());
    compat::wait();
  }

  void allocate(const Options &options) {
    for(int32_t i = 0; i < options.groups; ++i) {
      auto problem = options.problem_sizes_host.at(i);
      auto M = get<0>(problem);
      auto N = get<1>(problem);
      auto K = get<2>(problem);

      int64_t elements_A = M * K;
      int64_t elements_B = N * K;
      int64_t elements_C = M * N;
      int64_t elements_D = M * N;
      
      const int scale_k = cute::ceil_div(K, scaleGroupK);
      int padded_M = cute::round_up(M, scaleAlignA);
      int padded_N_scale = cute::round_up(ScalePolicy::scale_n_extent(N), scaleAlignB);
      int64_t elements_SFA = static_cast<int64_t>(scale_k) * padded_M;
      int64_t elements_SFB = static_cast<int64_t>(scale_k) * padded_N_scale;
      cutlass::DeviceAllocation<ElementA> a;
      a.reset(elements_A);
      block_A.push_back(a);
      cutlass::DeviceAllocation<ElementMMAVerify> ver_a;
      ver_a.reset(elements_A);
      block_A_dq.push_back(ver_a);
      cutlass::DeviceAllocation<ElementB> b;
      b.reset(elements_B);
      block_B.push_back(b);
      cutlass::DeviceAllocation<ElementMMAVerify> ver_b;
      ver_b.reset(elements_B);
      block_B_dq.push_back(ver_b);
      cutlass::DeviceAllocation<ElementC> c;
      c.reset(elements_C);
      block_C.push_back(c);
      cutlass::DeviceAllocation<ElementOutput> d;
      d.reset(elements_D);
      block_D.push_back(d);
      cutlass::DeviceAllocation<ElementOutput> ref_d;
      ref_d.reset(elements_D);
      block_ref_D.push_back(ref_d);
      cutlass::DeviceAllocation<ElementScaleA> sa;
      sa.reset(elements_SFA);
      block_scaleA.push_back(sa);
      cutlass::DeviceAllocation<ElementScaleB> sb;
      sb.reset(elements_SFB);
      block_scaleB.push_back(sb);
    }
    block_alpha.reset(options.groups);
    block_beta.reset(options.groups);
  }

  /// Initialize operands to be used in the GEMM and reference GEMM
  void initialize(Options const& options) {
    problem_sizes.reset(options.groups);
    problem_sizes.copy_from_host(options.problem_sizes_host.data());

    std::vector<ElementA *> ptr_A_host(options.groups);
    std::vector<ElementB *> ptr_B_host(options.groups);
    std::vector<ElementC *> ptr_C_host(options.groups);
    std::vector<ElementOutput *> ptr_D_host(options.groups);
    std::vector<ElementScaleA *> ptr_SFA_host(options.groups);
    std::vector<ElementScaleB *> ptr_SFB_host(options.groups);
    std::vector<ElementAccumulator *> ptr_alpha_host(options.groups);
    std::vector<ElementAccumulator *> ptr_beta_host(options.groups);

    for(int i = 0; i < options.groups; i++) {
      auto problem = options.problem_sizes_host.at(i);
      auto M = get<0>(problem);
      auto N = get<1>(problem);
      auto K = get<2>(problem);
      auto L = 1;

      auto shape_A = cute::make_shape(M, K, L);
      auto shape_B = cute::make_shape(N, K, L);
      auto shape_CD = cute::make_shape(M, N, L);
      const int scale_k = cute::ceil_div(K, scaleGroupK);
      int padded_M = cute::round_up(M, scaleAlignA);
      int padded_N_scale = cute::round_up(ScalePolicy::scale_n_extent(N), scaleAlignB);
      auto shape_scale_A = cute::make_shape(padded_M, scale_k, L);
      auto shape_scale_B = cute::make_shape(padded_N_scale, scale_k, L);

      auto stride_a = cutlass::make_cute_packed_stride(StrideA{}, shape_A);
      auto stride_b = cutlass::make_cute_packed_stride(StrideB{}, shape_B);
      auto stride_c = cutlass::make_cute_packed_stride(StrideC{}, shape_CD);
      auto stride_d = cutlass::make_cute_packed_stride(StrideD{}, shape_CD);
      auto stride_sfa = cutlass::make_cute_packed_stride(StrideScaleA{}, shape_scale_A);
      auto stride_sfb = cutlass::make_cute_packed_stride(StrideScaleB{}, shape_scale_B);

      stride_A_host.push_back(stride_a);
      stride_B_host.push_back(stride_b);
      stride_C_host.push_back(stride_c);
      stride_D_host.push_back(stride_d);
      stride_SFA_host.push_back(stride_sfa);
      stride_SFB_host.push_back(stride_sfb);

      initialize_block(block_A.at(i), seed + 2023 + i);
      initialize_block(block_B.at(i), seed + 2022 + i);
      initialize_block(block_C.at(i), seed + 2021 + i);

      convert_dtype<ElementA, ElementMMAVerify, ExampleRunner>(
          block_A.at(i),
          block_A_dq.at(i)
      );
      convert_dtype<ElementB, ElementMMAVerify, ExampleRunner>(
          block_B.at(i),
          block_B_dq.at(i)
      );

      initialize_scale(block_scaleA.at(i), options);
      initialize_scale(block_scaleB.at(i), options);

      auto layout_A = make_layout(shape_A, stride_a);
      auto layout_B = make_layout(shape_B, stride_b);
      auto layout_scale_A = make_layout(shape_scale_A, stride_sfa);
      auto layout_scale_B = make_layout(shape_scale_B, stride_sfb);

      apply_scale_A(block_A_dq.at(i).get(), block_A.at(i).get(), layout_A, block_scaleA.at(i).get(),  layout_scale_A);
      apply_scale_B(block_B_dq.at(i).get(), block_B.at(i).get(), layout_B, block_scaleB.at(i).get(),  layout_scale_B);

      ptr_A_host.at(i) = block_A.at(i).get();
      ptr_B_host.at(i) = block_B.at(i).get();
      ptr_C_host.at(i) = block_C.at(i).get();
      ptr_D_host.at(i) = block_D.at(i).get();
      ptr_SFA_host.at(i) = block_scaleA.at(i).get();
      ptr_SFB_host.at(i) = block_scaleB.at(i).get();
      alpha_host.push_back((options.alpha == FLT_MAX) ? static_cast<ElementAccumulator>((rand() % 5) + 1) : options.alpha);
      beta_host.push_back((options.beta == FLT_MAX) ? static_cast<ElementAccumulator>(rand() % 5) : options.beta);
      // Fill host ptr vectors with offset addresses into device alpha/beta blocks
      ptr_alpha_host.at(i) = block_alpha.get() + i;
      ptr_beta_host.at(i) = block_beta.get() + i;
    }
    // Allocate device memory & copy from host
    ptr_A.reset(options.groups);
    ptr_A.copy_from_host(ptr_A_host.data());

    ptr_B.reset(options.groups);
    ptr_B.copy_from_host(ptr_B_host.data());

    ptr_SFA.reset(options.groups);
    ptr_SFA.copy_from_host(ptr_SFA_host.data());

    ptr_SFB.reset(options.groups);
    ptr_SFB.copy_from_host(ptr_SFB_host.data());
    
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
    
    stride_SFA.reset(options.groups);
    stride_SFA.copy_from_host(stride_SFA_host.data());

    stride_SFB.reset(options.groups);
    stride_SFB.copy_from_host(stride_SFB_host.data());

    // Per-group alpha and beta ptrs
    alpha_device.reset(options.groups);
    alpha_device.copy_from_host(ptr_alpha_host.data());
    beta_device.reset(options.groups);
    beta_device.copy_from_host(ptr_beta_host.data());

    block_alpha.copy_from_host(alpha_host.data());
    block_beta.copy_from_host(beta_host.data());
  }
  
  cutlass::Status run(const Options& options, const cutlass::KernelHardwareInfo& hw_info) {
    allocate(options);
    initialize(options);
    auto args_tuple = args_from_options(options, hw_info);
    typename Gemm::GemmKernel::Arguments arguments {
      get<0>(args_tuple), get<1>(args_tuple),
      typename Gemm::GemmKernel::MainloopArguments{ptr_A.get(), stride_A.get(), ptr_B.get(), stride_B.get(), ptr_SFA.get(),
      stride_SFA.get(), ptr_SFB.get(), stride_SFB.get()},
      typename Gemm::GemmKernel::EpilogueArguments{get<2>(args_tuple), ptr_C.get(), stride_C.get(), ptr_D.get(), stride_D.get()},
      get<3>(args_tuple), get<4>(args_tuple)
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
      double tflops_result = options.tflops(cute_time, options.problem_sizes_host);
      std::cout << "Problem Size: " << options.m << 'x' << options.n << 'x' << options.k << 'x' << options.l << std::endl;
      if constexpr (std::is_same_v<ElementA, float_e4m3_t>) {
        std::cout << "Datatype: float_e4m3_t"<< std::endl;
      } else if constexpr (std::is_same_v<ElementA, float_e5m2_t>) {
        std::cout << "Datatype: float_e5m2_t"<< std::endl;
      }
      std::cout << "Groups: " << options.groups << std::endl;
      if constexpr (ScalePolicy::has_n_block_scale) {
        printf("Cutlass FP8 BlockScaled Grouped GEMM Performance:     [%4.3f]TFLOP/s  (%6.4f)ms\n", tflops_result, cute_time*1000);
      } else {
        printf("Cutlass Grouped GEMM Performance:     [%4.3f]TFLOP/s  (%6.4f)ms\n", tflops_result, cute_time*1000);
      }
    }
    return cutlass::Status::kSuccess;
  }

};
