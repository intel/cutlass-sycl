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
#include "cutlass/util/reference/host/gemm_complex_mkl.h"
#include "cutlass/util/initialize_block.hpp"
#include "cutlass/relatively_equal.h"

#include "../common.hpp"
#include <benchmark/benchmark.h>
#include <chrono>

using namespace cute;

// ---------------------------------------------------------------------------
// GEMM benchmark naming convention
// ---------------------------------------------------------------------------
// Benchmark names registered via CUTLASS_CREATE_GEMM_BENCHMARK (see the
// `benchmarks_sycl_types_*.hpp` files) follow a fixed pattern, e.g.:
//
//   Gemm_BF16BF16FP32BF16FP32_RRR_WG128x192x64_SG16x48x64
//   BLockScalingGemm_E4M3E4M3FP32BF16FP32_RRR_WG16x256x256_SG16x16x256_GS32
//   BLockScalingGemmNonNative_E4M3E4M3FP32BF16FP32_RRR_WG8x128x256_SG8x32x256_GS32
//
// Reading left to right:
//   - "Gemm" / "BLockScalingGemm" prefix (followed by "_")
//       Plain GEMM, or an MX/OCP microscaling block-scaled GEMM
//       (see the `GroupSize`/`GS` note below).
//   - "NonNative" (block-scaled only, optional)
//       Uses software dequantization of the scale factors before the MMA,
//       as opposed to the default "native" path where the DPAS instruction
//       consumes the scale factors directly.
//   - Element/type sequence, e.g. "E4M3E4M3FP32BF16FP32"
//       Five back-to-back type tags with no separator:
//         ElementA, ElementB, ElementC, ElementD, AccType (accumulator /
//         epilogue compute type). E.g. E4M3(A) E4M3(B) FP32(C) BF16(D) FP32(Acc).
//   - "_RRR" (or "_RCR", etc.) layout tag
//       One letter per operand, in order A / B / C: 'R' = RowMajor,
//       'C' = ColumnMajor. E.g. "_RCR_" means A=RowMajor, B=ColumnMajor,
//       C=RowMajor. A and C are RowMajor in every benchmark here; only the
//       B letter actually varies.
//   - "_WGmxnxk" workgroup tile shape
//       WG_M x WG_N x WG_K: the tile computed by one workgroup/threadgroup.
//   - "_SGmxnxk" subgroup tile shape
//       SG_M x SG_N x SG_K: the tile computed by one subgroup within the
//       workgroup (WG_M/SG_M and WG_N/SG_N give the subgroup grid).
//   - "_GSn" group size (block-scaled only)
//       The MX/OCP microscaling group size: the number of contiguous
//       K-elements that share one scale factor (defaults to 32; some
//       benchmarks use 128). See MainloopIntelXeXMX16BlockScaled's
//       `GroupSize` template parameter in dispatch_policy.hpp.

// The default benchmark name is the SYCL target passed to CMake via
// -DDPCPP_SYCL_TARGET. Falls back to "GEMM" when unset/empty.
#ifndef CUTLASS_BENCHMARK_SYCL_TARGET
#define CUTLASS_BENCHMARK_SYCL_TARGET ""
#endif

namespace cutlass::benchmark {

static inline std::string default_bm_name() {
  std::string const target = CUTLASS_BENCHMARK_SYCL_TARGET;
  return target.empty() ? std::string("GEMM") : target;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

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

// Block-scaled mainloops (MainloopIntelXeXMX16BlockScaled<Stages, GroupSize>) expose
// the scale group size as CollectiveMainloop::GroupK. Default to 32 otherwise.
template <class T, class = void>
struct GroupKType {
  static constexpr int value = 32;
};
template <class T>
struct GroupKType<T, cute::void_t<decltype(T::GroupK)>> {
  static constexpr int value = static_cast<int>(T::GroupK);
};

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
struct StrideScaleAType<T, cute::void_t<typename T::StrideScaleA>> {
  using type = typename T::StrideScaleA;
};

template <class T, class = void>
struct StrideScaleBType {
  using type = int;
};
template <class T>
struct StrideScaleBType<T, cute::void_t<typename T::StrideScaleB>> {
  using type = typename T::StrideScaleB;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

// Verification mode enum
enum class VerifyMode {
  None = 0,           // No verification (skip both device and host)
  Device = 1,         // Device verification - uses reference::device::GemmComplex
  Host = 2            // Host verification - uses reference::host::GemmComplexMkl
};

// Command line options parsing
struct GEMMOptions {

  bool error;

  int m, n, k, l;
  float alpha, beta;
  std::string bm_name;
  // Selects how correctness is checked: on the device with
  // reference::device::GemmComplex, on the host with reference::host::GemmComplexMkl,
  // or skipped entirely.
  VerifyMode verify_mode;

  GEMMOptions():
          error(false),
          m(5120), n(4096), k(4096), l(1),
          alpha(1.f), beta(0.f),
          bm_name(default_bm_name()),
          verify_mode(VerifyMode::None)
  { }

  // Parses the command line
  void parse(int argc, char const **args) {
    CommandLine cmd(argc, args);

    cmd.get_cmd_line_argument("m", m, 5120);
    cmd.get_cmd_line_argument("n", n, 4096);
    cmd.get_cmd_line_argument("k", k, 4096);
    cmd.get_cmd_line_argument("l", l, 1);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("bm_name", bm_name, default_bm_name());

    // Parse verification mode. Default to device verification on real hardware,
    // but host verification on the CRI simulator where device verification +
    // warmup are too time-consuming.
#ifdef CUTLASS_TEST_FOR_CRI
    std::string default_verify = "none";
#else
    std::string default_verify = "device";
#endif
    std::string verify_str = default_verify;
    cmd.get_cmd_line_argument("verify", verify_str, default_verify);
    if (verify_str == "none") {
      verify_mode = VerifyMode::None;
    } else if (verify_str == "device") {
      verify_mode = VerifyMode::Device;
    } else if (verify_str == "host") {
      verify_mode = VerifyMode::Host;
    } else {
      std::cerr << "Invalid verify mode or mode wasn't defined, using default None" << std::endl;
      verify_mode = VerifyMode::None;
    }
  }

  std::string benchmark_name() const {
    std::stringstream full_name;
    full_name << bm_name << "/";
    std::string const test_name_suffix = "MNKL_" + std::to_string(m) + "x" +
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
  using ElementC = typename Gemm::ElementC;
  using ElementD = typename Gemm::ElementD;

  using ElementAccumulator = typename Gemm::ElementAccumulator;
  using ElementMMAVerify = float;

  using CollectiveMainloop = typename Gemm::GemmKernel::CollectiveMainloop;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using ElementMma = typename CollectiveMainloop::TiledMma::ValTypeA;

  using ElementScale = typename ScaleType<CollectiveMainloop>::type;
  using ElementZero = typename ZeroType<CollectiveMainloop>::type;
  using StrideS = typename ScaleStride<CollectiveMainloop>::type;
  using StrideZ = typename ZeroStride<CollectiveMainloop>::type;

  using ElementScaleA = typename ElementScaleAType<CollectiveMainloop>::type;
  using ElementScaleB = typename ElementScaleBType<CollectiveMainloop>::type;
  using StrideScaleA = typename StrideScaleAType<CollectiveMainloop>::type;
  using StrideScaleB = typename StrideScaleBType<CollectiveMainloop>::type;

  using CollectiveEpilogue = typename Gemm::CollectiveEpilogue;
  using ElementOutput = typename CollectiveEpilogue::ElementOutput;
  using ElementCompute = typename CollectiveEpilogue::ElementCompute;

  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;

  int32_t count;
    static constexpr int GROUP_SIZE = GroupKType<CollectiveMainloop>::value;

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

  StrideS stride_S;
  StrideZ stride_Z;


  uint64_t seed = 0;

  // TODO: Use vector of allocations to avoid reusing
  // the same memory across different benchmark iterations
  // std::vector<DeviceAllocation<ElementA>> block_A;
  DeviceAllocation<ElementA> block_A;
  DeviceAllocation<ElementB> block_B;
  DeviceAllocation<ElementC> block_C;
  DeviceAllocation<ElementOutput> block_D;
  DeviceAllocation<ElementOutput> block_ref_D;
  DeviceAllocation<ElementOutput> block_Aux;

  cutlass::DeviceAllocation<ElementScale> block_scale;
  cutlass::DeviceAllocation<ElementZero> block_zero;
  cutlass::DeviceAllocation<ElementScaleA> block_scaleA;
  cutlass::DeviceAllocation<ElementScaleB> block_scaleB;
  cutlass::DeviceAllocation<ElementMMAVerify> block_A_dq; // Dequantized copy of A for validation
  cutlass::DeviceAllocation<ElementMMAVerify> block_B_dq; // Dequantized copy of B for validation

  DeviceAllocation<ElementMma> block_A_verify;
  DeviceAllocation<ElementMma> block_B_verify;
  

  BenchmarkRunnerGemm() : seed(0) {};

  //
  // Methods
  //

  template <
  class QuantizedElement,
  class DequantizedElement,
  class OperandLayout,
  class ElementScale,
  class ElementZero,
  class ScaleLayout,
  class ZeroLayout>
  static auto dequantize_A(DequantizedElement* dq_buffer,
                       QuantizedElement const* q_buffer,
                       OperandLayout const operand_layout,
                       ElementScale const* scale_buffer,
                       ElementZero const* zero_buffer,
                       ScaleLayout const scale_layout,
                       ZeroLayout const zero_layout,
                       int const group_size) {
    if constexpr (std::is_same_v<DequantizedElement, QuantizedElement>) {
      return dq_buffer;
    }

    std::vector<uint8_t> dst(size(operand_layout) * sizeof_bits_v<DequantizedElement> / 8, 0);
    cutlass::device_memory::copy_to_host(dst.data(), (uint8_t*)dq_buffer, dst.size());

    std::vector<uint8_t> src(size(operand_layout) * sizeof_bits_v<QuantizedElement> / 8, 0);
    cutlass::device_memory::copy_to_host(src.data(), (uint8_t*)q_buffer, src.size());

    std::vector<uint8_t> scale(size(scale_layout) * sizeof_bits_v<ElementScale> / 8, 0);
    cutlass::device_memory::copy_to_host(scale.data(), (uint8_t*)scale_buffer, scale.size());

    std::vector<uint8_t> zero(size(zero_layout) * sizeof_bits_v<ElementZero> / 8, 0);
    cutlass::device_memory::copy_to_host(zero.data(), (uint8_t*)zero_buffer, zero.size());

    compat::wait();

    auto dst_tensor = make_tensor(make_gmem_ptr(reinterpret_cast<DequantizedElement*>(dst.data())), select<1, 0, 2>(operand_layout));

    auto src_tensor = [&]() {
      if constexpr (sizeof_bits_v<QuantizedElement> < 8) {
        return make_tensor(cute::subbyte_iterator<const QuantizedElement>(src.data()), operand_layout);
      } else {
        return make_tensor(make_gmem_ptr(reinterpret_cast<QuantizedElement const *>(src.data())), select<1, 0, 2>(operand_layout));
      }
    }();

    auto scale_tensor = make_tensor(make_gmem_ptr(reinterpret_cast<ElementScale const *>(scale.data())), scale_layout);

    auto zero_tensor = [&]() {
      if constexpr (sizeof_bits_v<ElementZero> < 8) {
        auto flatten_tensor = flatten(make_tensor(cute::subbyte_iterator<const ElementZero>(zero.data()), zero_layout));
        static_assert(rank(flatten_tensor.layout()) == 4);
        return make_tensor(flatten_tensor.data(), select<1, 0, 2, 3>(flatten_tensor.layout()));
      } else {
        return make_tensor(make_gmem_ptr(reinterpret_cast<ElementZero const *>(zero.data())), zero_layout);
      }
    }();

    auto M = size<1>(src_tensor);
    auto K = size<0>(src_tensor);
    auto L = size<2>(src_tensor);

    static constexpr bool is_qnt = cutlass::platform::numeric_limits<DequantizedElement>::is_integer;

    for (int l = 0; l < L; l++) {
      for (int k= 0; k < K; k++) {
        for (int m = 0; m < M; m++) {
          auto src_data = [&]() {
            if constexpr (is_qnt) {
              if constexpr (sizeof_bits_v<QuantizedElement> >= 8) {
                return  src_tensor(k, m, l);
              } else {
                return src_tensor(k, m, l).get();
              }
            } else {
              using ret_type = cute::conditional_t<sizeof_bits_v<ElementZero> >= 8, ElementZero, int8_t>;
              if constexpr (sizeof_bits_v<QuantizedElement> >= 8) {
                return  (ret_type)(src_tensor(k, m, l));
              } else {
                return (ret_type)(src_tensor(k, m, l).get());
              }
            }
          }();

          auto scale_data = scale_tensor(m, k / group_size, l);

          using ret_type = cute::conditional_t<sizeof_bits_v<ElementZero> >= 8, ElementZero, int8_t>;
          ret_type zero_data = [&]() {
            if constexpr (sizeof_bits_v<ElementZero> >= 8) {
              return zero_tensor(m, k / group_size, l);
            } else {
              auto zero_elements_packed_along_k = get<0>(zero_tensor.shape());
              return (ret_type)(zero_tensor((k / group_size) % zero_elements_packed_along_k, m, k / group_size / zero_elements_packed_along_k, l).get());
            }
          }();

          if constexpr (is_qnt) {
            dst_tensor(k, m, l) = ((int)(src_data / scale_data)) + zero_data;
          } else {
            dst_tensor(k, m, l) = (src_data - zero_data) * scale_data;
          }
        }
      }
    }

    cutlass::device_memory::copy_to_device(dq_buffer, (DequantizedElement*)(raw_pointer_cast(dst_tensor.data())), dst_tensor.size());
    compat::wait();
    return dq_buffer;
  }

  template <
  class QuantizedElement,
  class DequantizedElement,
  class OperandLayout,
  class ElementScale,
  class ElementZero,
  class ScaleLayout,
  class ZeroLayout>
  static auto dequantize_B(DequantizedElement* dq_buffer,
                       QuantizedElement const* q_buffer,
                       OperandLayout const operand_layout,
                       ElementScale const* scale_buffer,
                       ElementZero const* zero_buffer,
                       ScaleLayout const scale_layout,
                       ZeroLayout const zero_layout,
                       int const group_size) {
    std::vector<uint8_t> dst(size(operand_layout) * sizeof_bits_v<DequantizedElement> / 8, 0);
    cutlass::device_memory::copy_to_host(dst.data(), (uint8_t*)dq_buffer, dst.size());

    std::vector<uint8_t> src(size(operand_layout) * sizeof_bits_v<QuantizedElement> / 8, 0);
    cutlass::device_memory::copy_to_host(src.data(), (uint8_t*)q_buffer, src.size());

    std::vector<uint8_t> scale(size(scale_layout) * sizeof_bits_v<ElementScale> / 8, 0);
    cutlass::device_memory::copy_to_host(scale.data(), (uint8_t*)scale_buffer, scale.size());

    std::vector<uint8_t> zero(size(zero_layout) * sizeof_bits_v<ElementZero> / 8, 0);
    cutlass::device_memory::copy_to_host(zero.data(), (uint8_t*)zero_buffer, zero.size());

    compat::wait();

    auto dst_tensor = make_tensor(make_gmem_ptr(reinterpret_cast<DequantizedElement*>(dst.data())), operand_layout);

    auto src_tensor = [&]() {
      if constexpr (sizeof_bits_v<QuantizedElement> < 8) {
        return make_tensor(cute::subbyte_iterator<const QuantizedElement>(src.data()), operand_layout);
      } else {
        return make_tensor(make_gmem_ptr(reinterpret_cast<QuantizedElement const *>(src.data())), operand_layout);
      }
    }();

    auto scale_tensor = make_tensor(make_gmem_ptr(reinterpret_cast<ElementScale const *>(scale.data())), scale_layout);

    auto zero_tensor = [&]() {
      if constexpr (sizeof_bits_v<ElementZero> < 8) {
        auto flatten_tensor = flatten(make_tensor(cute::subbyte_iterator<const ElementZero>(zero.data()), zero_layout));
        static_assert(rank(flatten_tensor.layout()) == 4);
        return make_tensor(flatten_tensor.data(), select<1, 0, 2, 3>(flatten_tensor.layout()));
      } else {
        return make_tensor(make_gmem_ptr(reinterpret_cast<ElementZero const *>(zero.data())), zero_layout);
      }
    }();

    auto N = size<0>(src_tensor);
    auto K = size<1>(src_tensor);
    auto L = size<2>(src_tensor);

    for (int l = 0; l < L; l++) {
      for (int k= 0; k < K; k++) {
        for (int n = 0; n < N; n++) {
          using ret_type = cute::conditional_t<sizeof_bits_v<ElementZero> >= 8, ElementZero, int8_t>;
          ret_type a = [&]() {
            if constexpr (sizeof_bits_v<QuantizedElement> >= 8) {
              return  (ret_type)(src_tensor(n, k, l));
            } else {
              return (ret_type)(src_tensor(n, k, l).get());
            }}();

          ret_type b = [&]() {
            if constexpr (sizeof_bits_v<ElementZero> >= 8) {
              return (ret_type)(zero_tensor(n, k / group_size, l));
            } else {
              auto k_packed = get<0>(zero_tensor.shape());
              return (ret_type)(zero_tensor((k / group_size) % k_packed, n, k / group_size / k_packed, l).get());
            }
          }();

          dst_tensor(n, k, l) = ((ElementScale)(a - b)) * scale_tensor(n, k / group_size, l);
        }
      }
    }

    cutlass::device_memory::copy_to_device(dq_buffer, (DequantizedElement*)(raw_pointer_cast(dst_tensor.data())), dst_tensor.size());
    compat::wait();
    return dq_buffer;
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


  bool verify_device(const ProblemShapeType& problem_size, ElementCompute alpha, ElementCompute beta, std::ostream& label) {
    auto& M = cute::get<0>(problem_size);
    auto& N = cute::get<1>(problem_size);
    auto& K = cute::get<2>(problem_size);
    auto& L = cute::get<3>(problem_size);

    TensorRef ref_C(block_C.get(), LayoutC::packed({M, N}));
    TensorRef ref_D(block_ref_D.get(), LayoutD::packed({M, N}));

    TensorRef ref_A(block_A_dq.get(), LayoutA::packed({M, K}));
    TensorRef ref_B(block_B_dq.get(), LayoutB::packed({K, N}));

    reference::device::GemmComplex(
            {M, N, K},
            alpha,
            ref_A,
            ComplexTransform::kNone,
            ref_B,
            ComplexTransform::kNone,
            beta,
            ref_C,
            ref_D,
            ElementAccumulator(0),
            L,     // batch_count
            M * K, // batch_stride_A
            N * K, // batch_stride_B
            M * N, // batch_stride_C
            M * N  // batch_stride_D
    );

#if defined(CUTLASS_ENABLE_SYCL)
    compat::wait();
#else
    cudaDeviceSynchronize();
#endif

    compat::wait();

    // Check if output from CUTLASS kernel and reference kernel are equal or not
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

    label << "verify_status=" << (passed ? "passed" : "failed")
          << "_with_device_ref_impl_gemm_complex ";
    return passed;
  }

  bool verify_host(const ProblemShapeType& problem_size, ElementCompute alpha, ElementCompute beta, std::ostream& label) {
    auto& M = cute::get<0>(problem_size);
    auto& N = cute::get<1>(problem_size);
    auto& K = cute::get<2>(problem_size);
    auto& L = cute::get<3>(problem_size);


    std::vector<ElementMMAVerify> host_A(block_A_dq.size());
    std::vector<ElementMMAVerify> host_B(block_B_dq.size());
    std::vector<ElementC> host_C(block_C.size());
    std::vector<ElementOutput> host_D(block_D.size());

    cutlass::device_memory::copy_to_host(host_A.data(), block_A_dq.get(), block_A_dq.size());
    cutlass::device_memory::copy_to_host(host_B.data(), block_B_dq.get(), block_B_dq.size());
    cutlass::device_memory::copy_to_host(host_C.data(), block_C.get(), block_C.size());
    cutlass::device_memory::copy_to_host(host_D.data(), block_D.get(), block_D.size());
    compat::wait();

    std::vector<ElementOutput> host_ref_D(block_D.size());

    TensorRef ref_A(host_A.data(), LayoutA::packed({M, K}));
    TensorRef ref_B(host_B.data(), LayoutB::packed({K, N}));
    TensorRef ref_C(host_C.data(), LayoutC::packed({M, N}));
    TensorRef ref_D(host_ref_D.data(), LayoutD::packed({M, N}));

    bool used_mkl = reference::host::GemmComplexMkl(
            {M, N, K},
            alpha,
            ref_A,
            ComplexTransform::kNone,
            ref_B,
            ComplexTransform::kNone,
            beta,
            ref_C,
            ref_D,
            ElementAccumulator(0),
            L,     // batch_count
            M * K, // batch_stride_A
            N * K, // batch_stride_B
            M * N, // batch_stride_C
            M * N  // batch_stride_D
    );

    // Match the tolerances used by the device verification path.
    ElementOutput const epsilon(1e-2f);
    ElementOutput const non_zero_floor(1e-4f);
    bool passed = true;
    for (std::size_t i = 0; i < host_D.size(); ++i) {
      if (!cutlass::relatively_equal(host_ref_D[i], host_D[i], epsilon, non_zero_floor)) {
        passed = false;
        printf("i: %zu , ref: %f, comp: %f\n", i, float(host_ref_D[i]), float(host_D[i]));
      }
    }

    label << "verify_status=" << (passed ? "passed" : "failed")
          << "_with_host_ref_impl_" << (used_mkl ? "mkl " : "gemm_complex ");
    return passed;
  }

  template <class Element>
  bool initialize_scale(
    cutlass::DeviceAllocation<Element>& block) {
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

  /// Initialize operands to be used in the GEMM and reference GEMM
  void initialize(::benchmark::State& state, const ProblemShapeType& problem_size) {
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto [M, N, K, L] = problem_shape_MNKL;
    
    const int scale_k = cute::ceil_div(K, GROUP_SIZE);
    auto shape_A = cute::make_shape(M, K, L);
    auto shape_B = cute::make_shape(N, K, L);
    auto shape_CD = cute::make_shape(M, N, L);
    auto shape_scale_A = cute::make_shape(M, scale_k, L);
    auto shape_scale_B = cute::make_shape(N, scale_k, L);    

    stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
    stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, L));
    stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, L));
    stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, L));

    if constexpr (is_blocked_scaled<CollectiveMainloop>) {
      stride_SA = cutlass::make_cute_packed_stride(StrideScaleA{}, shape_scale_A);
      stride_SB = cutlass::make_cute_packed_stride(StrideScaleB{}, shape_scale_B);
    }

    // TODO(codeplay): cute::cosize(some_large_layout) will overflow int32. What can we do about this?
    std::size_t size_A = cute::cosize(make_layout(cute::make_shape(M, K, L), stride_A));
    std::size_t size_B = cute::cosize(make_layout(cute::make_shape(N, K, L), stride_B));
    std::size_t size_C = cute::cosize(make_layout(cute::make_shape(M, N, L), stride_C));
    std::size_t mem_occupied_ABC = ((size_A * sizeof_bits_v<ElementA>) + (size_B * sizeof_bits_v<ElementB>) +
                                   (size_C * sizeof_bits_v<ElementC>)) / sizeof_bits_v<int8_t>;
    count = std::ceil(static_cast<float>(cutlass::get_llc_size()) / static_cast<float>(mem_occupied_ABC)) + 1;

    block_A.reset(static_cast<std::size_t>(M) * K * L);
    block_A_dq.reset(static_cast<std::size_t>(M) * K * L);
    block_B.reset(static_cast<std::size_t>(K) * N * L);
    block_B_dq.reset(static_cast<std::size_t>(K) * N * L);
    block_C.reset(static_cast<std::size_t>(M) * N * L);
    block_D.reset(static_cast<std::size_t>(M) * N * L);
    block_ref_D.reset(static_cast<std::size_t>(M) * N * L);

    if constexpr (is_blocked_scaled<CollectiveMainloop>) {
      block_scaleA.reset(static_cast<std::size_t>(scale_k) * L * M);
      block_scaleB.reset(static_cast<std::size_t>(scale_k) * L * N);
    }

    initialize_block(block_A, seed + 2023);
    initialize_block(block_B, seed + 2022);
    initialize_block(block_C, seed + 2021);

    cutlass::benchmark::convert_dtype<ElementA, ElementMMAVerify, BenchmarkRunnerGemm>(
        block_A,
        block_A_dq
    );
    cutlass::benchmark::convert_dtype<ElementB, ElementMMAVerify, BenchmarkRunnerGemm>(
        block_B,
        block_B_dq
    );

    if constexpr (is_blocked_scaled<CollectiveMainloop>) {
      initialize_scale(block_scaleA);
      initialize_scale(block_scaleB);
      auto layout_A = make_layout(shape_A, stride_A);
      auto layout_B = make_layout(shape_B, stride_B);
      auto layout_scale_A = make_layout(shape_scale_A, stride_SA);
      auto layout_scale_B = make_layout(shape_scale_B, stride_SB);
      apply_scale(block_A_dq.get(), block_A.get(), layout_A, block_scaleA.get(),  layout_scale_A);
      apply_scale(block_B_dq.get(), block_B.get(), layout_B, block_scaleB.get(),  layout_scale_B);
    }
  }

  void run(::benchmark::State& state, const GEMMOptions& options, const KernelHardwareInfo& hw_info) {
    auto wall_start = std::chrono::steady_clock::now();
    ProblemShapeType problem_size = ProblemShapeType{options.m, options.n, options.k, options.l};

    initialize(state, problem_size);

    typename Gemm::GemmKernel::Arguments arguments = GemmConfiguration::defaultArguments();
    arguments.mode = gemm::GemmUniversalMode::kGemm;
    arguments.problem_shape = problem_size;

    if constexpr (!is_blocked_scaled<CollectiveMainloop>) {
      arguments.mainloop = {block_A.get(), stride_A, block_B.get(), stride_B};
    } else {
      arguments.mainloop = {block_A.get(), stride_A, block_B.get(), stride_B,
        block_scaleA.get(), stride_SA, block_scaleB.get(), stride_SB};
    }


    arguments.epilogue = {{ElementAccumulator(options.alpha), ElementAccumulator(options.beta)}, block_C.get(), stride_C, block_D.get(), stride_D};
    
    arguments.hw_info = hw_info;

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

    std::stringstream extra_label;
    if (options.verify_mode != VerifyMode::None) {
      // Run the GEMM
      gemm_op.run();

#if defined(CUTLASS_ENABLE_SYCL)
    compat::wait();
#else
    cudaDeviceSynchronize();
#endif

      // Verify that the result is correct; each verify method appends its
      // disposition (and reference implementation used) to extra_label.
      bool passed = (options.verify_mode == VerifyMode::Host)
          ? verify_host(problem_size, ElementCompute(options.alpha), ElementCompute(options.beta), extra_label)
          : verify_device(problem_size, ElementCompute(options.alpha), ElementCompute(options.beta), extra_label);
      if(not passed) {
        state.SkipWithError("Disposition Failed.");
      }
    }

    state.counters["m"] = options.m;
    state.counters["n"] = options.n;
    state.counters["k"] = options.k;
    state.counters["l"] = options.l;
    state.counters["alpha"] = options.alpha;
    state.counters["beta"] = options.beta;

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

    auto gflop = 2.0 * options.m * options.n * options.k * options.l * 1e-9;

    // Compatible with data types smaller than 8 bits here
    constexpr double bits_per_byte = static_cast<double>(sizeof_bits_v<char>);
    constexpr double sizeof_a = sizeof_bits_v<ElementA> / bits_per_byte;
    constexpr double sizeof_b = sizeof_bits_v<ElementB> / bits_per_byte;
    constexpr double sizeof_o = sizeof_bits_v<ElementOutput> / bits_per_byte;
    auto mega_bytes_transferred = static_cast<double>(
        options.m * options.k * sizeof_a +
        options.k * options.n * sizeof_b +
        (options.beta != 0 ? 2 : 1) * options.m * options.n * sizeof_o
      ) * 1e-6 * options.l;

    initialize_counters(state);
    for(auto _ : state) {
      state.PauseTiming();
      typename Gemm::GemmKernel::Arguments arguments = [&]() {
        if constexpr (!is_blocked_scaled<CollectiveMainloop>) {
          return typename Gemm::GemmKernel::Arguments{
            gemm::GemmUniversalMode::kGemm,
            problem_size,
            {block_A.get(), stride_A, block_B.get(), stride_B},
            {{ElementAccumulator(options.alpha), ElementAccumulator(options.beta)}, block_C.get(), stride_C, block_D.get(), stride_D},
            hw_info
          };
        } else {
          return typename Gemm::GemmKernel::Arguments{
            gemm::GemmUniversalMode::kGemm,
            problem_size,
            {block_A.get(), stride_A, block_B.get(), stride_B,
              block_scaleA.get(), stride_SA, block_scaleB.get(), stride_SB},
            {{ElementAccumulator(options.alpha), ElementAccumulator(options.beta)}, block_C.get(), stride_C, block_D.get(), stride_D},
            hw_info
          };
        }
      }();

      gemm_op.initialize(arguments, workspace.get());
      state.ResumeTiming();

      GPU_Clock timer;
      timer.start();
      gemm_op.run();
      auto ms_elapsed = timer.milliseconds();
      update_counters(state, ms_elapsed);
      state.SetIterationTime(ms_elapsed / 1000);
    }
    auto wall_end = std::chrono::steady_clock::now();
    finalize_counters(state, gflop, mega_bytes_transferred);
    state.counters["execution_time_s"] =
        (std::chrono::duration<double, std::milli>(wall_end - wall_start).count())/1000;
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
    state.counters["avg_bandwidth_gbs"] = mega_bytes_transferred / state.counters["avg_runtime_ms"];
    state.counters["best_tflops"] = gflop / state.counters["best_runtime_ms"];
    state.counters["best_bandwidth_gbs"] = mega_bytes_transferred / state.counters["best_runtime_ms"];
  }
};

}

#define CUTLASS_BENCHMARK(F) cutlass::benchmark::BenchmarkRegistry<cutlass::benchmark::GEMMOptions>::Register(#F, &F##_func)

#define CUTLASS_CREATE_GEMM_BENCHMARK(F)                          \
  static void F##_func(                                           \
      ::benchmark::State& state,                                  \
      cutlass::benchmark::GEMMOptions const& options,                 \
      cutlass::KernelHardwareInfo const& hw_info) {               \
    auto bench = cutlass::benchmark::BenchmarkRunnerGemm<F>();    \
    bench.run(state, options, hw_info);                           \
  }
