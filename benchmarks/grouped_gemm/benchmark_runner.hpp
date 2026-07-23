/***************************************************************************************************
 * Copyright (c) 2024 - 2025 Codeplay Software Ltd. All rights reserved.
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

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cute/tensor.hpp"

#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/device/tensor_silu.h"
#include "cutlass/util/initialize_block.hpp"

#include "../common.hpp"
#include <benchmark/benchmark.h>

#include <cfloat>

using namespace cute;

namespace cutlass::benchmark {

///////////////////////////////////////////////////////////////////////////////////////////////////

using namespace cutlass::gemm;
using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int,int,int>>; // <M,N,K> per group

template <class T, class = void>
struct ScaleType {
  using type = int;
};
template <class T>
struct ScaleType<T, cute::void_t<typename T::ElementScale>> {
  using type = typename T::ElementScale;
};

template <class T, class = void>
struct ZeroType {
  using type = int;
};
template <class T>
struct ZeroType<T, cute::void_t<typename T::ElementZero>> {
  using type = typename T::ElementZero;
};

template <class T, class = void>
struct ScaleStride {
  using type = int;
};
template <class T>
struct ScaleStride<T, cute::void_t<typename T::StrideScale>> {
  using type = typename T::StrideScale;
};

template <class T, class = void>
struct ZeroStride {
  using type = int;
};
template <class T>
struct ZeroStride<T, cute::void_t<typename T::StrideZero>> {
  using type = typename T::StrideZero;
};

template <class T, class = void>
static constexpr auto is_blocked_scaled = false;
template <class T>
static constexpr auto is_blocked_scaled<T, cute::void_t<typename T::ElementScaleA, typename T::ElementScaleB>> = true;

template <class T, class = void>
struct ElementScaleAType {
  using type = int;
};
template <class T>
struct ElementScaleAType<T, cute::void_t<typename T::ElementScaleA>> {
  using type = typename T::ElementScaleA;
};

template <class T, class = void>
struct ElementScaleBType {
  using type = int;
};
template <class T>
struct ElementScaleBType<T, cute::void_t<typename T::ElementScaleB>> {
  using type = typename T::ElementScaleB;
};

template <class T, class = void>
struct StrideScaleAType {
  using type = int;
};
template <class T>
struct StrideScaleAType<T, cute::void_t<typename T::InternalStrideScaleA>> {
  using type = typename T::InternalStrideScaleA;
};

template <class T, class = void>
struct StrideScaleBType {
  using type = int;
};
template <class T>
struct StrideScaleBType<T, cute::void_t<typename T::InternalStrideScaleB>> {
  using type = typename T::InternalStrideScaleB;
};

template <class T, class = void>
struct GroupSizeType {
  static constexpr int value = 32;  // Default for non-block-scaled
};
template <class T>
struct GroupSizeType<T, cute::void_t<decltype(T::GROUP_K)>> {
  static constexpr int value = T::GROUP_K;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct GroupedGEMMOptions {

  bool error;

  int m, n, k, l, groups;
  float alpha, beta;
  std::string bm_name;
  std::vector<typename ProblemShape::UnderlyingProblemShape> problem_sizes_host;

  GroupedGEMMOptions():
          error(false),
          m(5120), n(4096), k(4096), l(1),
          groups(2),
          alpha(1.f), beta(0.f),
          bm_name("GroupedGEMM")
  {
    problem_sizes_host.reserve(groups);
    for(int i = 0; i < groups; i++) {
      problem_sizes_host.push_back({m, n, k});
    }

  }

  // Parses the command line
  void parse(int argc, char const **args) {
    CommandLine cmd(argc, args);

    cmd.get_cmd_line_argument("m", m, 5120);
    cmd.get_cmd_line_argument("n", n, 4096);
    cmd.get_cmd_line_argument("k", k, 4096);
    cmd.get_cmd_line_argument("l", l, 1);
    cmd.get_cmd_line_argument("groups", groups, 2);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("bm_name", bm_name, std::string("GEMM"));

    assert(groups > 2);
    problem_sizes_host.clear();
    problem_sizes_host.reserve(groups);
    for(int i = 0; i < groups; i++) {
      problem_sizes_host.push_back({m, n, k});
    }
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

  std::string benchmark_name() const {
    std::stringstream full_name;
    full_name << bm_name << "/";
    std::string const test_name_suffix = std::to_string(m) + "x" +
                                   std::to_string(n) + "x" +
                                   std::to_string(k) + "x" +
                                   std::to_string(l);
    full_name << test_name_suffix;

    return full_name.str();
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <class GemmConfiguration>
struct BenchmarkRunnerGemm {

  using Gemm = typename GemmConfiguration::Gemm;

  using CollectiveMainloop = typename Gemm::GemmKernel::CollectiveMainloop;
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
  using ElementMMAVerify = float;

  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using ElementMma = typename CollectiveMainloop::TiledMma::ValTypeA;

  using ElementScale = typename ScaleType<CollectiveMainloop>::type;
  using ElementZero = typename ZeroType<CollectiveMainloop>::type;
  using StrideS = typename ScaleStride<CollectiveMainloop>::type;
  using StrideZ = typename ZeroStride<CollectiveMainloop>::type;

  using ElementC = typename Gemm::ElementC;
  using ElementOutput = typename CollectiveEpilogue::ElementOutput;
  using ElementCompute = typename CollectiveEpilogue::ElementCompute;

  using ElementScaleA = typename ElementScaleAType<CollectiveMainloop>::type;
  using ElementScaleB = typename ElementScaleBType<CollectiveMainloop>::type;
  using StrideScaleA = typename StrideScaleAType<CollectiveMainloop>::type;
  using StrideScaleB = typename StrideScaleBType<CollectiveMainloop>::type;

  static constexpr int GROUP_SIZE = GroupSizeType<CollectiveMainloop>::value;

  //int32_t count;

  //
  // Data members
  //

  /// Initialization
  // // Host-side allocation
  std::vector<ElementAccumulator> alpha_host;
  std::vector<ElementAccumulator> beta_host;

  std::vector<StrideA> stride_A_host;
  std::vector<StrideB> stride_B_host;
  std::vector<StrideScaleA> stride_SFA_host;
  std::vector<StrideScaleB> stride_SFB_host;
  std::vector<StrideC> stride_C_host;
  std::vector<StrideD> stride_D_host;

  StrideS stride_S;
  StrideZ stride_Z;

  // // Device-side allocations
  cutlass::DeviceAllocation<typename ProblemShape::UnderlyingProblemShape> problem_sizes;

  cutlass::DeviceAllocation<StrideA> stride_A;
  cutlass::DeviceAllocation<StrideB> stride_B;
  cutlass::DeviceAllocation<StrideC> stride_C;
  cutlass::DeviceAllocation<StrideD> stride_D;
  cutlass::DeviceAllocation<StrideScaleA> stride_SFA;
  cutlass::DeviceAllocation<StrideScaleB> stride_SFB;

  uint64_t seed = 0;

  std::vector<DeviceAllocation<ElementA>> block_A;
  std::vector<DeviceAllocation<ElementB>> block_B;
  std::vector<DeviceAllocation<ElementC>> block_C;
  std::vector<DeviceAllocation<ElementOutput>> block_D;
  std::vector<cutlass::DeviceAllocation<ElementScaleA>> block_scaleA;
  std::vector<cutlass::DeviceAllocation<ElementScaleB>> block_scaleB;
  std::vector<DeviceAllocation<ElementOutput>> block_ref_D;

  cutlass::DeviceAllocation<ElementAccumulator> block_alpha;
  cutlass::DeviceAllocation<ElementAccumulator> block_beta;
  
  cutlass::DeviceAllocation<ElementScale> block_scale;
  cutlass::DeviceAllocation<ElementZero> block_zero;

  std::vector<DeviceAllocation<ElementMma>> block_A_verify;
  std::vector<DeviceAllocation<ElementMma>> block_B_verify;

  cutlass::DeviceAllocation<const ElementA *> ptr_A;
  cutlass::DeviceAllocation<const ElementB *> ptr_B;
  cutlass::DeviceAllocation<const ElementC *> ptr_C;
  cutlass::DeviceAllocation<ElementOutput *> ptr_D;
  cutlass::DeviceAllocation<const ElementScaleA *> ptr_SFA;
  cutlass::DeviceAllocation<const ElementScaleB *> ptr_SFB;
  cutlass::DeviceAllocation<ElementAccumulator*> alpha_device;
  cutlass::DeviceAllocation<ElementAccumulator*> beta_device;

  std::vector<cutlass::DeviceAllocation<ElementMMAVerify>> block_A_dq; // Dequantized copy of A for validation
  std::vector<cutlass::DeviceAllocation<ElementMMAVerify>> block_B_dq; // Dequantized copy of B for validation

  BenchmarkRunnerGemm() : seed(0) {};

  //
  // Methods
  //

    /// Populates a Gemm::Arguments structure from the given commandline options
  auto args_from_options(const GroupedGEMMOptions &options, const cutlass::KernelHardwareInfo& hw_info)
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

  bool verify(const GroupedGEMMOptions &options) {
    bool passed = true;
    ElementOutput const epsilon(1e-2f);
    ElementOutput const non_zero_floor(1e-4f);
    for (int i = 0; i < options.groups; i++){
      Shape<int, int, int, int> problem_size = append<4>(options.problem_sizes_host[i], 1);
      auto M = get<0>(problem_size);
      auto N = get<1>(problem_size);
      auto K = get<2>(problem_size);
      TensorRef ref_A(block_A_dq.at(i).get(), LayoutA::packed({M, K}));
      TensorRef ref_B(block_B_dq.at(i).get(), LayoutB::packed({K, N}));
      TensorRef ref_C(block_C.at(i).get(), LayoutC::packed({M, N}));
      TensorRef ref_D(block_ref_D.at(i).get(), LayoutD::packed({M, N}));
      reference::device::GemmComplex(
              {M, N, K},
              alpha_host.at(i),
              ref_A,
              ComplexTransform::kNone,
              ref_B,
              ComplexTransform::kNone,
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

      compat::wait();

      passed &= cutlass::reference::device::BlockCompareRelativelyEqual(
        block_ref_D.at(i).get(), block_D.at(i).get(), block_D.at(i).size(), epsilon, non_zero_floor);

      if (!passed) break;
    }

    return passed;
  }

  template <class Element>
  bool initialize_scale(
    cutlass::DeviceAllocation<Element>& block,
    GroupedGEMMOptions const& options) {
    const float elt_max_f = float(cutlass::platform::numeric_limits<Element>::max());
    // Need to fix max_dequant_val and min_dequant_val?
    const float max_dequant_val = elt_max_f * 0.25f;
    const float min_dequant_val = 0.5f;
    const float scale_max = max_dequant_val / elt_max_f;
    const float scale_min = min_dequant_val / elt_max_f;
    cutlass::reference::device::BlockFillRandomUniform(
        block.get(), block.size(), seed, Element(scale_max), Element(scale_min));
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

          auto scale_data = (ret_type)(scale_tensor(mn, k / 32, l));

          dst_tensor(mn, k, l) = (src_data) * scale_data;
        }
      }
    }

    cutlass::device_memory::copy_to_device(dq_buffer, (DstElement*)(raw_pointer_cast(dst_tensor.data())), dst_tensor.size());
    compat::wait();
  }
  
  void allocate(const GroupedGEMMOptions &options) {
    for(int32_t i = 0; i < options.groups; ++i) {
      auto problem = options.problem_sizes_host.at(i);
      auto M = get<0>(problem);
      auto N = get<1>(problem);
      auto K = get<2>(problem);

      int64_t elements_A = M * K;
      int64_t elements_B = N * K;
      int64_t elements_C = M * N;
      int64_t elements_D = M * N;
      
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
      
      if constexpr (is_blocked_scaled<CollectiveMainloop>) {
        const int scale_k = cute::ceil_div(K, GROUP_SIZE);
        int64_t elements_SFA = scale_k * M;
        int64_t elements_SFB = scale_k * N;
        cutlass::DeviceAllocation<ElementScaleA> sa;
        sa.reset(elements_SFA);
        block_scaleA.push_back(sa);
        cutlass::DeviceAllocation<ElementScaleB> sb;
        sb.reset(elements_SFB);
        block_scaleB.push_back(sb);
      }
    }
    block_alpha.reset(options.groups);
    block_beta.reset(options.groups);
  }

  /// Initialize operands to be used in the GEMM and reference GEMM
  void initialize(::benchmark::State& state, GroupedGEMMOptions const &options) {
    problem_sizes.reset(options.groups);
    problem_sizes.copy_from_host(options.problem_sizes_host.data());

    std::vector<ElementA *> ptr_A_host(options.groups);
    std::vector<ElementB *> ptr_B_host(options.groups);
    std::vector<ElementC *> ptr_C_host(options.groups);
    std::vector<ElementOutput *> ptr_D_host(options.groups);
    std::vector<ElementAccumulator *> ptr_alpha_host(options.groups);
    std::vector<ElementAccumulator *> ptr_beta_host(options.groups);
    
    std::vector<ElementScaleA *> ptr_SFA_host;
    std::vector<ElementScaleB *> ptr_SFB_host;
    if constexpr (is_blocked_scaled<CollectiveMainloop>) {
      ptr_SFA_host.resize(options.groups);
      ptr_SFB_host.resize(options.groups);
    }

    for(int i = 0; i < options.groups; i++) {
      auto problem = options.problem_sizes_host.at(i);
      auto M = get<0>(problem);
      auto N = get<1>(problem);
      auto K = get<2>(problem);
      auto L = 1;

      auto shape_A = cute::make_shape(M, K, L);
      auto shape_B = cute::make_shape(N, K, L);
      auto shape_CD = cute::make_shape(M, N, L);

      auto stride_a = cutlass::make_cute_packed_stride(StrideA{}, shape_A);
      auto stride_b = cutlass::make_cute_packed_stride(StrideB{}, shape_B);
      auto stride_c = cutlass::make_cute_packed_stride(StrideC{}, shape_CD);
      auto stride_d = cutlass::make_cute_packed_stride(StrideD{}, shape_CD);

      stride_A_host.push_back(stride_a);
      stride_B_host.push_back(stride_b);
      stride_C_host.push_back(stride_c);
      stride_D_host.push_back(stride_d);
      
      if constexpr (is_blocked_scaled<CollectiveMainloop>) {
        const int scale_k = cute::ceil_div(K, GROUP_SIZE);
        auto shape_scale_A = cute::make_shape(M, scale_k, L);
        auto shape_scale_B = cute::make_shape(N, scale_k, L);
        auto stride_sfa = cutlass::make_cute_packed_stride(StrideScaleA{}, shape_scale_A);
        auto stride_sfb = cutlass::make_cute_packed_stride(StrideScaleB{}, shape_scale_B);
        stride_SFA_host.push_back(stride_sfa);
        stride_SFB_host.push_back(stride_sfb);
      }

      initialize_block(block_A.at(i), seed + 2023 + i);
      initialize_block(block_B.at(i), seed + 2022 + i);
      initialize_block(block_C.at(i), seed + 2021 + i);

      cutlass::benchmark::convert_dtype<ElementA, ElementMMAVerify, BenchmarkRunnerGemm>(
          block_A.at(i),
          block_A_dq.at(i)
      );
      cutlass::benchmark::convert_dtype<ElementB, ElementMMAVerify, BenchmarkRunnerGemm>(
          block_B.at(i),
          block_B_dq.at(i)
      );

      if constexpr (is_blocked_scaled<CollectiveMainloop>) {
        const int scale_k = cute::ceil_div(K, GROUP_SIZE);
        auto shape_scale_A = cute::make_shape(M, scale_k, L);
        auto shape_scale_B = cute::make_shape(N, scale_k, L);
        
        initialize_scale(block_scaleA.at(i), options);
        initialize_scale(block_scaleB.at(i), options);

        auto layout_A = make_layout(shape_A, stride_a);
        auto layout_B = make_layout(shape_B, stride_b);
        auto layout_scale_A = make_layout(shape_scale_A, stride_SFA_host.at(i));
        auto layout_scale_B = make_layout(shape_scale_B, stride_SFB_host.at(i));

        apply_scale(block_A_dq.at(i).get(), block_A.at(i).get(), layout_A, block_scaleA.at(i).get(),  layout_scale_A);
        apply_scale(block_B_dq.at(i).get(), block_B.at(i).get(), layout_B, block_scaleB.at(i).get(),  layout_scale_B);
      }

      ptr_A_host.at(i) = block_A.at(i).get();
      ptr_B_host.at(i) = block_B.at(i).get();
      ptr_C_host.at(i) = block_C.at(i).get();
      ptr_D_host.at(i) = block_D.at(i).get();
      
      if constexpr (is_blocked_scaled<CollectiveMainloop>) {
        ptr_SFA_host.at(i) = block_scaleA.at(i).get();
        ptr_SFB_host.at(i) = block_scaleB.at(i).get();
      }

      alpha_host.push_back((options.alpha == FLT_MAX) ? static_cast<ElementAccumulator>((rand() % 5) + 1) : options.alpha);
      beta_host.push_back((options.beta == FLT_MAX) ? static_cast<ElementAccumulator>(rand() % 5) : options.beta);
      // Fill host ptr vectors with offset addresses into device alpha/beta blocks
      ptr_alpha_host.at(i) = block_alpha.get() + i;
      ptr_beta_host.at(i) = block_beta.get() + i;
    }
    // Allocate device memory & copy from host
    try {
    ptr_A.reset(options.groups);
    ptr_A.copy_from_host(ptr_A_host.data());

    ptr_B.reset(options.groups);
    ptr_B.copy_from_host(ptr_B_host.data());
    
    ptr_C.reset(options.groups);
    ptr_C.copy_from_host(ptr_C_host.data());

    ptr_D.reset(options.groups);
    ptr_D.copy_from_host(ptr_D_host.data());

    if constexpr (is_blocked_scaled<CollectiveMainloop>) {
      ptr_SFA.reset(options.groups);
      ptr_SFA.copy_from_host(ptr_SFA_host.data());

      ptr_SFB.reset(options.groups);
      ptr_SFB.copy_from_host(ptr_SFB_host.data());
    }

    stride_A.reset(options.groups);
    stride_A.copy_from_host(stride_A_host.data());

    stride_B.reset(options.groups);
    stride_B.copy_from_host(stride_B_host.data());

    stride_C.reset(options.groups);
    stride_C.copy_from_host(stride_C_host.data());

    stride_D.reset(options.groups);
    stride_D.copy_from_host(stride_D_host.data());

    if constexpr (is_blocked_scaled<CollectiveMainloop>) {
      stride_SFA.reset(options.groups);
      stride_SFA.copy_from_host(stride_SFA_host.data());

      stride_SFB.reset(options.groups);
      stride_SFB.copy_from_host(stride_SFB_host.data());
    }

    // Per-group alpha and beta ptrs
    alpha_device.reset(options.groups);
    alpha_device.copy_from_host(ptr_alpha_host.data());
    beta_device.reset(options.groups);
    beta_device.copy_from_host(ptr_beta_host.data());

    block_alpha.copy_from_host(alpha_host.data());
    block_beta.copy_from_host(beta_host.data());

    } catch (std::exception const &e) {
      state.SkipWithError(e.what());
    }
  }

  void run(::benchmark::State& state, const GroupedGEMMOptions& options, const KernelHardwareInfo& hw_info) {
    allocate(options);
    initialize(state, options);

    auto args_tuple = args_from_options(options, hw_info);
    
    auto mainloop_args = [&]() {
      if constexpr (!is_blocked_scaled<CollectiveMainloop>) {
        return typename Gemm::GemmKernel::MainloopArguments{
          ptr_A.get(), stride_A.get(), ptr_B.get(), stride_B.get()
        };
      } else {
        return typename Gemm::GemmKernel::MainloopArguments{
          ptr_A.get(), stride_A.get(), ptr_B.get(), stride_B.get(),
          ptr_SFA.get(), stride_SFA.get(), ptr_SFB.get(), stride_SFB.get(),
        };
      }
    }();
    
    typename Gemm::GemmKernel::Arguments arguments {
      get<0>(args_tuple), get<1>(args_tuple),
      mainloop_args,
      typename Gemm::GemmKernel::EpilogueArguments{get<2>(args_tuple), ptr_C.get(), stride_C.get(), ptr_D.get(), stride_D.get()},
      get<3>(args_tuple), get<4>(args_tuple)
    };

    Gemm gemm_op;

    device_memory::allocation<uint8_t> workspace;
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    try {
      workspace.reset(workspace_size);
    } catch (std::exception const &e) {
      state.SkipWithError(e.what());
    }

    if (gemm_op.can_implement(arguments) != cutlass::Status::kSuccess)
      state.SkipWithError("GEMM unable to implement given args.");

    if (gemm_op.initialize(arguments, workspace.get()) != cutlass::Status::kSuccess)
      state.SkipWithError("GEMM failed to initialize.");

    if (state.error_occurred()) return;

#ifdef CUTLASS_TEST_FOR_CRI
    // disable warmup run and verification for CRI simulator as it's time-consuming
#else
    // Run the GEMM
    gemm_op.run();

    compat::wait();

    // Verify that the result is correct
    bool passed = verify(options);
    if(not passed) {
      state.SkipWithError("Disposition Failed.");
    }
#endif

    state.counters["m"] = options.m;
    state.counters["n"] = options.n;
    state.counters["k"] = options.k;
    state.counters["l"] = options.l;
    state.counters["alpha"] = options.alpha;
    state.counters["beta"] = options.beta;

    std::stringstream extra_label;
    if constexpr (cute::size<0>(StrideA{}) == 1) {
      extra_label << "layoutA=ColumnMajor ";
    } else if constexpr (cute::size<1>(StrideA{}) == 1) {
      extra_label << "layoutA=RowMajor ";
    }
    if constexpr (cute::size<0>(StrideB{}) == 1) {
      extra_label << "layoutB=RowMajor ";
    } else if constexpr (cute::size<1>(StrideB{}) == 1) {
      extra_label << "layoutB=ColumnMajor ";
    }
    if constexpr (cute::size<0>(StrideC{}) == 1) {
      extra_label << "layoutC=ColumnMajor ";
    } else if constexpr (cute::size<1>(StrideC{}) == 1) {
      extra_label << "layoutC=RowMajor ";
    }
    state.SetLabel(extra_label.str());


    // Number of real-valued multiply-adds
    uint64_t fmas = uint64_t();

    for (auto const & problem : options.problem_sizes_host) {
      fmas += static_cast<uint64_t>(get<0>(problem)) *
              static_cast<uint64_t>(get<1>(problem)) *
              static_cast<uint64_t>(get<2>(problem));
    }
    // Two flops per multiply-add
    uint64_t flop = static_cast<uint64_t>(2) * static_cast<uint64_t>(fmas);
    double gflop = double(flop) / double(1.0e9);

    // Compatible with data types smaller than 8 bits here
    constexpr double bits_per_byte = static_cast<double>(sizeof_bits_v<char>);
    constexpr double sizeof_a = sizeof_bits_v<ElementA> / bits_per_byte;
    constexpr double sizeof_b = sizeof_bits_v<ElementB> / bits_per_byte;
    constexpr double sizeof_c = sizeof_bits_v<ElementC> / bits_per_byte;
    auto mega_bytes_transferred = static_cast<double>(
        options.m * options.k * sizeof_a +
        options.k * options.n * sizeof_b +
        (options.beta != 0 ? 2 : 1) * options.m * options.n * sizeof_c
      ) * 1e-6 * options.l;

    initialize_counters(state);
    for(auto _ : state) {
      state.PauseTiming();
      auto args_tuple = args_from_options(options, hw_info);
      
      auto mainloop_args = [&]() {
        if constexpr (!is_blocked_scaled<CollectiveMainloop>) {
          return typename Gemm::GemmKernel::MainloopArguments{
            ptr_A.get(), stride_A.get(), ptr_B.get(), stride_B.get()
          };
        } else {
          return typename Gemm::GemmKernel::MainloopArguments{
            ptr_A.get(), stride_A.get(), ptr_B.get(), stride_B.get(),
            ptr_SFA.get(), stride_SFA.get(), ptr_SFB.get(), stride_SFB.get(),
          };
        }
      }();
      
      typename Gemm::GemmKernel::Arguments arguments {
        get<0>(args_tuple), get<1>(args_tuple),
        mainloop_args,
        typename Gemm::GemmKernel::EpilogueArguments{get<2>(args_tuple), ptr_C.get(), stride_C.get(), ptr_D.get(), stride_D.get()},
        get<3>(args_tuple), get<4>(args_tuple)
      };
      gemm_op.initialize(arguments, workspace.get());
      state.ResumeTiming();

      GPU_Clock timer;
      timer.start();
      gemm_op.run();
      auto ms_elapsed = timer.milliseconds();
      update_counters(state, ms_elapsed);
      state.SetIterationTime(ms_elapsed / 1000);
    }
    finalize_counters(state, gflop, mega_bytes_transferred);
  }

private:
  static void initialize_counters(::benchmark::State& state) {
    state.counters["avg_runtime_ms"] = 0;
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

  static void finalize_counters(::benchmark::State& state,  double gflop, double mega_bytes_transferred) {
    state.counters["avg_runtime_ms"] =
      (state.counters["total_runtime_ms"] -state.counters["best_runtime_ms"] - state.counters["worst_runtime_ms"] ) / static_cast<double>(state.iterations() - 2);
    state.counters["avg_tflops"] = gflop / state.counters["avg_runtime_ms"];
    state.counters["avg_throughput"] = mega_bytes_transferred / state.counters["avg_runtime_ms"];
    state.counters["best_tflop"] = gflop / state.counters["best_runtime_ms"];
    state.counters["best_bandwidth"] = mega_bytes_transferred / state.counters["best_runtime_ms"];
  }
};

}

#define CUTLASS_BENCHMARK(F) cutlass::benchmark::BenchmarkRegistry<cutlass::benchmark::GroupedGEMMOptions>::Register(#F, &F##_func)

#define CUTLASS_CREATE_GROUPED_GEMM_BENCHMARK(F)                          \
  static void F##_func(                                           \
      ::benchmark::State& state,                                  \
      cutlass::benchmark::GroupedGEMMOptions const& options,                 \
      cutlass::KernelHardwareInfo const& hw_info) {               \
    auto bench = cutlass::benchmark::BenchmarkRunnerGemm<F>();    \
    bench.run(state, options, hw_info);                           \
  }
