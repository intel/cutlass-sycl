/***************************************************************************************************
 * Copyright (C) 2024 - 2025 Codeplay Software Ltd. All rights reserved.
 * Copyright (C) 2025 - 2026 Intel Corporation, All rights reserved.
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
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/util/packed_stride.hpp"
#include "flash_attention_v2/collective/fmha_fusion.hpp"
#include "flash_attention_v2/kernel/xe_fmha_fwd_kernel.hpp"
#include "flash_attention_v2/kernel/xe_tile_scheduler.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/sycl_event_manager.hpp"
#include <cute/tensor.hpp>
#include <random>

#include "helper.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "sycl_common.hpp"

#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

using namespace cute;

// Command line options parsing
struct Options {

  bool help;
  bool error;
  bool is_causal;
  bool varlen = false;
  bool use_paged_kv = false;
  std::string scheduler;
  std::string layout;

  int batch, num_heads_q, num_heads_kv, seq_len_qo, seq_len_kv, seq_len_kv_cache, page_size, head_size_qk, head_size_vo, iterations, warmup, verify;
  float softmax_scale;

  Options()
      : help(false), error(false), is_causal(false), varlen(false), use_paged_kv(false), batch(32), num_heads_q(16), num_heads_kv(16), seq_len_qo(512), head_size_qk(128),
        seq_len_kv(512), seq_len_kv_cache(0), page_size(128), head_size_vo(128), iterations(100), warmup(100), softmax_scale(1.f), verify(1), scheduler("Individual"), layout("HND") {}

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    if (cmd.check_cmd_line_flag("is_causal")) {
      is_causal = true;
    }

    if (cmd.check_cmd_line_flag("varlen")) {
      varlen = true;
    }

    cmd.get_cmd_line_argument("scheduler", scheduler, std::string("Individual"));
    cmd.get_cmd_line_argument("layout", layout, std::string("HND"));
    if (layout == "hnd") {
      layout = "HND";
    } else if (layout == "nhd") {
      layout = "NHD";
    }
    if (layout != "HND" && layout != "NHD") {
      std::cerr << "Invalid: layout must be either HND or NHD" << std::endl;
      error = true;
      return;
    }

#ifdef PERSISTENT
    cmd.get_cmd_line_argument("batch", batch, 1);
    cmd.get_cmd_line_argument("num_heads_q", num_heads_q, 8);
    cmd.get_cmd_line_argument("num_heads_kv", num_heads_kv, 1);
    cmd.get_cmd_line_argument("seq_len_kv", seq_len_kv, 4096);
#else
    cmd.get_cmd_line_argument("batch", batch, 32);
    cmd.get_cmd_line_argument("num_heads_q", num_heads_q, 16);
    cmd.get_cmd_line_argument("num_heads_kv", num_heads_kv, num_heads_q);
    cmd.get_cmd_line_argument("seq_len_kv", seq_len_kv, 512);
    cmd.get_cmd_line_argument("seq_len_kv_cache", seq_len_kv_cache, 0);
#endif
#ifdef DECODE
    cmd.get_cmd_line_argument("seq_len_qo", seq_len_qo, 1);
#else
    cmd.get_cmd_line_argument("seq_len_qo", seq_len_qo, seq_len_kv);
#endif
    cmd.get_cmd_line_argument("head_size_vo", head_size_vo, HEAD_DIM);
    cmd.get_cmd_line_argument("head_size_qk", head_size_qk, head_size_vo);
    cmd.get_cmd_line_argument("iterations", iterations, 100);
    cmd.get_cmd_line_argument("warmup", warmup, 100);
    cmd.get_cmd_line_argument("verify", verify, 1);

    if (cmd.check_cmd_line_flag("use_paged_kv")) {
        use_paged_kv = true;
        cmd.get_cmd_line_argument("page_size", page_size, 128);

        if (page_size % 128 != 0) {
            std::cerr << "Invalid: page_size must be a multiple of 128" << std::endl;
          error = true;
          return;
        }
        if (seq_len_kv_cache % page_size != 0) {
            std::cerr << "Invalid: seq_len_kv_cache must be divisible by page_size" << std::endl;
          error = true;
            return;
        }
    }

    softmax_scale = 1 / sqrt(static_cast<float>(head_size_qk));
  }

  /// Prints the usage statement.
  std::ostream &print_usage(std::ostream &out) const {

    out << "Xe FMHA Example\n\n"
        << "Options:\n\n"
        << "  --help                      If specified, displays this usage statement\n\n"
        << "  --is_causal                 Apply Causal Mask to the output of first Matmul\n"
        << "  --varlen                    Enable variable sequence length\n"
        << "  --layout                    Choose between HND or NHD layouts\n"
        << "  --scheduler=\"Value\"       Choose between Individual or Persistent Scheduler\n"
        << "  --batch=<int>               Sets the Batch Size of the Multi-Head Self Attention module\n"
        << "  --num_heads_q=<int>         Sets the Number of Attention Heads for Key-Value pair the Multi-Head Self Attention module\n"
        << "  --num_heads_kv=<int>        Sets the Number of Attention Heads for Query input in the Multi-Head Self Attention module\n"
        << "  --seq_len_qo=<int>          Sets the Sequence length of the Query input in Multi-Head Self Attention module\n"
        << "  --seq_len_kv=<int>          Sets the Sequence length of the Key-Value pair in Multi-Head Self Attention module\n"
        << "  --seq_len_kv_cache=<int>    Sets the Sequence length of the cached Key-Value pair in Multi-Head Self Attention module\n"
        << "  --use_paged_kv              Use paged (non-contiguous) KV cache. Default is contiguous KV Cache\n"
        << "  --page_size=<int>           Block size for paged KV cache. Default is 128\n"
        << "  --head_size_qk=<int>        Sets the Attention Head dimension of the 1st Matrix Multiplication in Multi-Head Self Attention module\n"
        << "  --head_size_vo=<int>        Sets the Attention Head dimension of the 2nd Matrix Multiplication in Multi-Head Self Attention module\n"
        << "  --iterations=<int>          Iterations\n"
        << "  --warmup=<int>              Warmup iterations before timing\n"
        << "  --verify=<int>              Specify whether to verify.\n\n";
    return out;
  }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
// Helpers

template <typename SrcT, typename DstT> class ConvertTensorKernelTag{};

template <typename SrcT, typename DstT>
void convert_tensor(const SrcT* d_src, DstT* d_dst, size_t size) {
  using Tag = ConvertTensorKernelTag<SrcT, DstT>;
  compat::get_default_queue().parallel_for<Tag>(size, [=](auto indx) {
    d_dst[indx] = static_cast<DstT>(d_src[indx]);
  }).wait();
}

template <typename InT> inline auto in_memory(cutlass::DeviceAllocation<InT>& in) {
  using OutT = cute::conditional_t<(sizeof_bits_v<InT> <= 8), half_t, InT>;
  if constexpr (!is_same_v<InT, OutT>) {
    cutlass::DeviceAllocation<OutT> out(in.size());
    convert_tensor<InT, OutT>(in.get(), out.get(), in.size());
    return out;
  } else {
    return in;
  };
}

///////////////////////////////////////////////////////////////////////////////////////////////////

// 3 input matrices: (K)eys, (Q)ueries and (V)alues.
using LayoutQ = cutlass::layout::RowMajor;
using LayoutK = cutlass::layout::ColumnMajor;
using LayoutV = cutlass::layout::RowMajor;
using LayoutO = cutlass::layout::RowMajor;

using HNDQLayoutStep = Step<_1, _0, _2, _3>;
using HNDVLayoutStep = Step<_0, _1, _2, _3>;
using NHDQLayoutStep = Step<_2, _0, _1, _3>;
using NHDVLayoutStep = Step<_0, _2, _1, _3>;

enum class CopySliceOp { Q, KCache, VCache, K, V, O };

template <typename Owner, CopySliceOp Op, typename SrcT, typename DstT, typename Stride>
class CopySliceKernelTag {};

template <typename Owner, CopySliceOp Op, typename SrcT, typename DstT, typename Stride>
void copy_slice(const SrcT* src, DstT* dst, int seq_start, int seq_len, int head_size,
                int head_idx, int batch_idx, Stride stride, bool is_v = false,
                bool store = false, int page_size = 0, const int* page_table = nullptr,
                int start_page_idx = 0) {
  std::size_t size = static_cast<std::size_t>(seq_len) * head_size;
  if (size == 0) {
    return;
  }
  using Tag = CopySliceKernelTag<Owner, Op, SrcT, DstT, Stride>;
  compat::get_default_queue().parallel_for<Tag>(size, [=](auto indx) {
    int linear_idx = indx;
    int row = linear_idx / head_size;
    int col = linear_idx % head_size;
    int seq_idx = seq_start + row;
    if (page_table != nullptr && page_size > 0) {
      int page_offset = row % page_size;
      int logical_page = row / page_size;
      seq_idx = seq_start + page_table[start_page_idx + logical_page] * page_size + page_offset;
    }
    int tensor_idx = is_v ? col * get<0>(stride) + seq_idx * get<1>(stride)
                          : seq_idx * get<0>(stride) + col * get<1>(stride);
    tensor_idx += head_idx * get<2>(stride) + batch_idx * get<3>(stride);
    if (store) {
      dst[tensor_idx] = src[linear_idx];
    } else {
      dst[linear_idx] = src[tensor_idx];
    }
  }).wait();
}

template <class FMHAKernel, bool isVarLen = false> struct ExampleRunner {

  using StrideQ = typename FMHAKernel::StrideQ;
  using StrideK = typename FMHAKernel::StrideK;
  using StrideV = typename FMHAKernel::StrideV;
  using StrideO = typename FMHAKernel::StrideO;

  using ElementQ = typename FMHAKernel::ElementQ;
  using ElementK = typename FMHAKernel::ElementK;
  using ElementV = typename FMHAKernel::ElementV;
  using ElementO = typename FMHAKernel::ElementO;

  using CollectiveMainloop = typename FMHAKernel::CollectiveMainloop;
  using ElementS = typename CollectiveMainloop::ElementS;

  using ProblemShapeType = cutlass::fmha::kernel::FMHAProblemShape<isVarLen>;

  //
  // Data members
  //

  /// Initialization
  StrideQ stride_Q;
  StrideK stride_K;
  StrideV stride_V;
  StrideK stride_K_cache;
  StrideV stride_V_cache;
  StrideO stride_O;
  uint64_t seed = 0;

  cutlass::DeviceAllocation<ElementQ> block_Q;
  cutlass::DeviceAllocation<ElementK> block_K;
  cutlass::DeviceAllocation<ElementV> block_V;
  cutlass::DeviceAllocation<ElementK> block_K_cache;
  cutlass::DeviceAllocation<ElementV> block_V_cache;
  cutlass::DeviceAllocation<ElementO> block_O;
  cutlass::DeviceAllocation<ElementO> block_ref_O;

  std::vector<int> cumulative_seqlen_q;
  std::vector<int> cumulative_seqlen_kv;
  std::vector<int> cumulative_seqlen_kv_cache;
  cutlass::DeviceAllocation<int> device_cumulative_seqlen_q;
  cutlass::DeviceAllocation<int> device_cumulative_seqlen_kv;
  cutlass::DeviceAllocation<int> device_cumulative_seqlen_kv_cache;

  struct PagedKVParams {
    cutlass::DeviceAllocation<int> page_table;
    int page_size = 0;
    cutlass::DeviceAllocation<int> num_pages_per_seq;
  };
  PagedKVParams paged_kv_cache;

  //
  // Methods
  //

  template<class ProblemShape>
  auto initialize_varlen(const ProblemShape& problem_size) {
    int num_batches = get<0>(problem_size);

    // generate Q as --b times
    //    gaussian (--Q, --Q / 2) sampled positive
    //    track cumulative 
    std::mt19937 rng(0x202305151552ull);
    std::normal_distribution<double> dist_q(get<3>(problem_size), get<3>(problem_size) / 2);
    std::normal_distribution<double> dist_kv(get<4>(problem_size), get<4>(problem_size) / 2);
    std::normal_distribution<double> dist_kv_cache(get<5>(problem_size), get<5>(problem_size) / 2);

    // Use Cacheline Size to calculate alignment
    constexpr int cacheline_bytes = 64;
    constexpr int AlignmentQ = cacheline_bytes * 8 /  sizeof_bits_v<ElementQ> ;    // Alignment of Q matrix in units of elements
    constexpr int AlignmentKV = cacheline_bytes * 8 / sizeof_bits_v<ElementK> ;   // Alignment of Kand V matrix in units of elements
    constexpr int AlignmentKVCache = 128; //Page size must be a multiple of 128
    auto generate_positive_int = [](auto& dist, auto& gen) {
      int result = 0;
      do {
        result = static_cast<int>(dist(gen));
      } while (result <= 0);
      return result;
    };

    cumulative_seqlen_q = {0};
    cumulative_seqlen_kv = {0};
    cumulative_seqlen_kv_cache = {0};

    int total_seqlen_q = 0;
    int total_seqlen_kv = 0;
    int total_seqlen_kv_cache = 0;
    int max_seqlen_q = 0;
    int max_seqlen_kv = 0;
    int max_seqlen_kv_cache = 0;

    for (int i = 0; i < num_batches; i++) {
      int seqlen_q = cutlass::round_up(generate_positive_int(dist_q, rng), AlignmentQ);
      int seqlen_kv = cutlass::round_up(generate_positive_int(dist_kv, rng), AlignmentKV);
      int seqlen_kv_cache = get<5>(problem_size) == 0 ? 0 : cutlass::round_up(generate_positive_int(dist_kv_cache, rng), AlignmentKVCache);

      total_seqlen_q += seqlen_q;
      total_seqlen_kv += seqlen_kv;
      total_seqlen_kv_cache += seqlen_kv_cache;

      max_seqlen_q = std::max(max_seqlen_q, seqlen_q);
      max_seqlen_kv = std::max(max_seqlen_kv, seqlen_kv);
      max_seqlen_kv_cache = std::max(max_seqlen_kv_cache, seqlen_kv_cache);

      cumulative_seqlen_q.push_back(cumulative_seqlen_q.back() + seqlen_q);
      cumulative_seqlen_kv.push_back(cumulative_seqlen_kv.back() + seqlen_kv);
      cumulative_seqlen_kv_cache.push_back(cumulative_seqlen_kv_cache.back() + seqlen_kv_cache);
    }

    ProblemShape problem_size_for_init = problem_size;
    get<0>(problem_size_for_init) = 1;
    get<3>(problem_size_for_init) = total_seqlen_q;
    get<4>(problem_size_for_init) = total_seqlen_kv;
    get<5>(problem_size_for_init) = total_seqlen_kv_cache;

    ProblemShapeType problem_size_for_launch;
    problem_size_for_launch.batch = get<0>(problem_size);
    problem_size_for_launch.num_heads_q = get<1>(problem_size);
    problem_size_for_launch.num_heads_kv = get<2>(problem_size);
    problem_size_for_launch.seq_len_qo = cutlass::fmha::collective::VariableLength{max_seqlen_q};
    problem_size_for_launch.seq_len_kv = cutlass::fmha::collective::VariableLength{max_seqlen_kv};
    problem_size_for_launch.seq_len_kv_cache = cutlass::fmha::collective::VariableLength{max_seqlen_kv_cache};
    problem_size_for_launch.head_size_qk = get<6>(problem_size);
    problem_size_for_launch.head_size_vo = get<7>(problem_size);

    return cute::make_tuple(problem_size_for_init, problem_size_for_launch);
  }

  bool verify(ProblemShapeType shape, const Options& options) {

    if constexpr (isVarLen) {
      int max_seq_len_q = shape.seq_len_qo;
      int max_seq_len_kv = shape.seq_len_kv;
      int max_seq_len_kv_cache = shape.seq_len_kv_cache;
      shape.seq_len_qo = cutlass::fmha::collective::VariableLength{max_seq_len_q, cumulative_seqlen_q.data()};
      shape.seq_len_kv = cutlass::fmha::collective::VariableLength{max_seq_len_kv, cumulative_seqlen_kv.data()};
      shape.seq_len_kv_cache = cutlass::fmha::collective::VariableLength{max_seq_len_kv_cache, cumulative_seqlen_kv_cache.data()};
    }

    auto batch = shape.batch;
    auto num_heads_q = shape.num_heads_q;
    auto num_heads_kv = shape.num_heads_kv;
    auto head_size_qk = shape.head_size_qk;
    auto head_size_vo = shape.head_size_vo;
    int seq_len_qo, seq_len_kv, seq_len_kv_cache;

    auto block_Q_ = in_memory(block_Q);
    auto block_K_ = in_memory(block_K);
    auto block_V_ = in_memory(block_V);
    auto block_K_cache_ = in_memory(block_K_cache);
    auto block_V_cache_ = in_memory(block_V_cache);

    using ElementV_ = std::remove_pointer_t<decltype(block_V_.get())>;
    using ElementK_ = std::remove_pointer_t<decltype(block_K_.get())>;

    std::vector<int> page_table_host;
    std::vector<int> num_pages_per_seq_host;
    if (paged_kv_cache.page_size > 0) {
      page_table_host.resize(paged_kv_cache.page_table.size());
      compat::memcpy(page_table_host.data(), paged_kv_cache.page_table.get(), page_table_host.size() * sizeof(int));
      num_pages_per_seq_host.resize(paged_kv_cache.num_pages_per_seq.size());
      compat::memcpy(num_pages_per_seq_host.data(), paged_kv_cache.num_pages_per_seq.get(), num_pages_per_seq_host.size() * sizeof(int));
      compat::wait();
    }

    compat::memset(block_ref_O.get(), 0, block_ref_O.size() * sizeof(ElementO));

    // loop over the batch dimension to compute the output
    // to avoid the risk of running out of device memory
    int q_group_size = num_heads_q / num_heads_kv;
    for (int b = 0; b < batch; b++) {
      if constexpr (isVarLen) {
        auto logical_seq_shape = cutlass::fmha::collective::apply_variable_length(
            make_shape(shape.seq_len_qo, shape.seq_len_kv, shape.seq_len_kv_cache), b);
        seq_len_qo = get<0>(logical_seq_shape);
        seq_len_kv = get<1>(logical_seq_shape);
        seq_len_kv_cache = get<2>(logical_seq_shape);
      } else {
        seq_len_qo = shape.seq_len_qo;
        seq_len_kv = shape.seq_len_kv;
        seq_len_kv_cache = shape.seq_len_kv_cache;
      }

      int seq_len_kv_total = seq_len_kv + seq_len_kv_cache;
      int batch_idx = isVarLen ? 0 : b;
      int q_base = isVarLen ? cumulative_seqlen_q[b] : 0;
      int kv_base = isVarLen ? cumulative_seqlen_kv[b] : 0;
      int kv_cache_base = isVarLen ? cumulative_seqlen_kv_cache[b] : 0;
      auto q = block_Q_.get();
      auto k = block_K_.get();
      auto v = block_V_.get();
      auto k_cache = block_K_cache_.get();
      auto v_cache = block_V_cache_.get();
      auto o = block_ref_O.get();
      auto dQ = stride_Q;
      auto dK = stride_K;
      auto dV = stride_V;
      auto dK_cache = stride_K_cache;
      auto dV_cache = stride_V_cache;
      auto dO = stride_O;

      if constexpr (isVarLen) {
        if (options.layout == "HND") {
          q += static_cast<std::size_t>(q_base) * num_heads_q * get<0>(stride_Q);
          k += static_cast<std::size_t>(kv_base) * num_heads_kv * get<0>(stride_K);
          v += static_cast<std::size_t>(kv_base) * num_heads_kv * get<1>(stride_V);
          o += static_cast<std::size_t>(q_base) * num_heads_q * get<0>(stride_O);
          if (seq_len_kv_cache > 0) {
            k_cache += static_cast<std::size_t>(kv_cache_base) * num_heads_kv * get<0>(stride_K_cache);
            v_cache += static_cast<std::size_t>(kv_cache_base) * num_heads_kv * get<1>(stride_V_cache);
            dK_cache = make_stride(get<0>(stride_K_cache), get<1>(stride_K_cache), int(seq_len_kv_cache * get<0>(stride_K_cache)), int(num_heads_kv * seq_len_kv_cache * get<0>(stride_K_cache)));
            dV_cache = make_stride(get<0>(stride_V_cache), get<1>(stride_V_cache), int(seq_len_kv_cache * get<1>(stride_V_cache)), int(num_heads_kv * seq_len_kv_cache * get<1>(stride_V_cache)));
          }
          dQ = make_stride(get<0>(stride_Q), get<1>(stride_Q), int(seq_len_qo * get<0>(stride_Q)), int(num_heads_q * seq_len_qo * get<0>(stride_Q)));
          dK = make_stride(get<0>(stride_K), get<1>(stride_K), int(seq_len_kv * get<0>(stride_K)), int(num_heads_kv * seq_len_kv * get<0>(stride_K)));
          dV = make_stride(get<0>(stride_V), get<1>(stride_V), int(seq_len_kv * get<1>(stride_V)), int(num_heads_kv * seq_len_kv * get<1>(stride_V)));
          dO = make_stride(get<0>(stride_O), get<1>(stride_O), int(seq_len_qo * get<0>(stride_O)), int(num_heads_q * seq_len_qo * get<0>(stride_O)));
          q_base = kv_base = kv_cache_base = 0;
        }
      }

      for (int h = 0; h < num_heads_q; h++) {
        int head_kv = h / q_group_size;

        cutlass::DeviceAllocation<ElementK_> block_K_concat;
        cutlass::DeviceAllocation<ElementV_> block_V_concat;
        block_K_concat.reset(static_cast<std::size_t>(head_size_qk) * seq_len_kv_total);
        block_V_concat.reset(static_cast<std::size_t>(seq_len_kv_total) * head_size_vo);

        if (seq_len_kv_cache > 0) {
          int page_size = paged_kv_cache.page_size;
          const int* page_table = page_size > 0 ? paged_kv_cache.page_table.get() : nullptr;
          int start_page_idx = 0;
          if (page_table != nullptr) {
            start_page_idx = isVarLen ? num_pages_per_seq_host[b] : b * (seq_len_kv_cache / page_size);
          }

          copy_slice<ExampleRunner, CopySliceOp::KCache>(
            k_cache,
            block_K_concat.get(),
            kv_cache_base,
            seq_len_kv_cache,
            head_size_qk,
            head_kv,
            batch_idx,
            dK_cache,
            false,
            false,
            page_size,
            page_table,
            start_page_idx);

          copy_slice<ExampleRunner, CopySliceOp::VCache>(
            v_cache,
            block_V_concat.get(),
            kv_cache_base,
            seq_len_kv_cache,
            head_size_vo,
            head_kv,
            batch_idx,
            dV_cache,
            true,
            false,
            page_size,
            page_table,
            start_page_idx);
        }

        copy_slice<ExampleRunner, CopySliceOp::K>(
          k,
          block_K_concat.get() + static_cast<std::size_t>(head_size_qk) * seq_len_kv_cache,
          kv_base,
          seq_len_kv,
          head_size_qk,
          head_kv,
          batch_idx,
          dK);

        copy_slice<ExampleRunner, CopySliceOp::V>(
          v,
          block_V_concat.get() + static_cast<std::size_t>(seq_len_kv_cache) * head_size_vo,
          kv_base,
          seq_len_kv,
          head_size_vo,
          head_kv,
          batch_idx,
          dV,
          true);

        int q_chunk_size = 128;
        for (int q_start = 0; q_start < seq_len_qo; q_start += q_chunk_size) {
          int q_end = std::min(seq_len_qo, q_start + q_chunk_size);
          int current_q_len = q_end - q_start;
          int q_seq_start = q_base + q_start;

          cutlass::DeviceAllocation<std::remove_pointer_t<decltype(block_Q_.get())>> block_Q_chunk;
          block_Q_chunk.reset(static_cast<std::size_t>(current_q_len) * head_size_qk);
          copy_slice<ExampleRunner, CopySliceOp::Q>(
            q, block_Q_chunk.get(), q_seq_start, current_q_len, head_size_qk, h, batch_idx, dQ);

          cutlass::DeviceAllocation<ElementS> block_S;
          block_S.reset(current_q_len * seq_len_kv_total);

          cutlass::TensorRef ref_Q(block_Q_chunk.get(), LayoutQ::packed({current_q_len, head_size_qk}));
          cutlass::TensorRef ref_K(block_K_concat.get(), LayoutK::packed({head_size_qk, seq_len_kv_total}));
          cutlass::TensorRef ref_V(block_V_concat.get(), LayoutV::packed({seq_len_kv_total, head_size_vo}));
          cutlass::TensorRef ref_S(block_S.get(), LayoutQ::packed({current_q_len, seq_len_kv_total}));

          // GEMM 1: S = Q_chunk * K^T
          cutlass::reference::device::GemmComplex({current_q_len, seq_len_kv_total, head_size_qk}, 1.f, ref_Q,
                                                  cutlass::ComplexTransform::kNone, ref_K, cutlass::ComplexTransform::kNone,
                                                  0.f, ref_S, ref_S, ElementS(0),
                                                  1,                               // batch_count
                                                  current_q_len * head_size_qk,    // batch_stride_Q
                                                  seq_len_kv_total * head_size_qk, // batch_stride_K
                                                  current_q_len * seq_len_kv_total,// batch_stride_S
                                                  current_q_len * seq_len_kv_total // batch_stride_S
          );

          compat::wait();

          std::vector<ElementS> host_S(block_S.size());
          compat::memcpy<ElementS>(host_S.data(), block_S.get(), host_S.size());

          // delete this memory as it is no longer needed
          block_S.reset();
          auto offset = cute::min(seq_len_qo, seq_len_kv);
          auto discard_seq_coord = seq_len_qo - offset;
          auto full_tile_offset = seq_len_kv - offset;
          if (options.is_causal) {
            // apply mask to S
            for (int row = 0; row < current_q_len; row++) {
              int global_row = row + q_start; // Use global row index for masking check
              for (int col = seq_len_kv_cache; col < seq_len_kv_total; col++) {
                if ((col - seq_len_kv_cache - full_tile_offset) > (global_row - discard_seq_coord)) {
                  host_S[col + row * seq_len_kv_total] = ElementS{-INFINITY};
                }
              }
            }
          }

          // compute max element per row of S
          std::vector<ElementS> max_vec(current_q_len, ElementS{-INFINITY});
          for (int row = 0; row < current_q_len; row++) {
            int idx = row * seq_len_kv_total;
            int max_idx = row;
            max_vec[max_idx] = host_S[idx++];
            for (int col = 1; col < seq_len_kv_total; col++, idx++) {
              if (max_vec[max_idx] < host_S[idx]) {
                max_vec[max_idx] = host_S[idx];
              }
            }
          }

          // compute exp of S
          for (int row = 0; row < current_q_len; row++) {
            int idx = row * seq_len_kv_total;
            int max_idx = row;
            for (int col = 0; col < seq_len_kv_total; col++, idx++) {
              /* FIXME: use softmax_scale instead of assuming its value here */
              host_S[idx] = expf((host_S[idx] - max_vec[max_idx]) / sqrt(static_cast<ElementS>((head_size_qk))));
            }
          }

          // compute sum per row of S
          std::vector<ElementS> sum_vec(current_q_len, ElementS{0});
          for (int row = 0; row < current_q_len; row++) {
            int idx = row * seq_len_kv_total;
            int sum_idx = row;
            for (int col = 0; col < seq_len_kv_total; col++, idx++) {
              sum_vec[sum_idx] += host_S[idx];
            }

            // scale each row with the sum to compute softmax
            idx = row * seq_len_kv_total;
            sum_idx = row;
            for (int col = 0; col < seq_len_kv_total; col++, idx++) {
              int global_row = row + q_start;
              if (options.is_causal && global_row < discard_seq_coord) {
                host_S[idx] = 0;
              } else {
                host_S[idx] /= sum_vec[sum_idx];
              }
            }
          }

          std::vector<ElementV_> host_P(host_S.size());
          for (int p = 0; p < host_P.size(); p++) {
            host_P[p] = static_cast<ElementV_>(host_S[p]);
          }

          cutlass::DeviceAllocation<ElementV_> block_P;
          block_P.reset(host_P.size());

          compat::memcpy<ElementV_>(block_P.get(), host_P.data(), host_P.size());

          cutlass::TensorRef ref_P(block_P.get(), LayoutQ::packed({current_q_len, seq_len_kv_total}));

          cutlass::DeviceAllocation<ElementS> block_acc;
          block_acc.reset(current_q_len * head_size_vo);
          cutlass::TensorRef ref_acc(block_acc.get(), LayoutO::packed({current_q_len, head_size_vo}));

          // GEMM 2: O = P_chunk * V
          cutlass::reference::device::GemmComplex({current_q_len, head_size_vo, seq_len_kv_total}, ElementS{1}, ref_P,
                                                  cutlass::ComplexTransform::kNone, ref_V, cutlass::ComplexTransform::kNone,
                                                  ElementS{0}, ref_acc, ref_acc, ElementS{0},
                                                  1,                               // batch_count
                                                  current_q_len * seq_len_kv_total,// batch_stride_P
                                                  seq_len_kv_total * head_size_vo, // batch_stride_V
                                                  current_q_len * head_size_vo,    // batch_stride_O
                                                  current_q_len * head_size_vo     // batch_stride_O
            );

          compat::wait();
          block_P.reset();

          std::vector<ElementS> vec_acc(block_acc.size());
          compat::memcpy<ElementS>(vec_acc.data(), block_acc.get(), vec_acc.size());

          block_acc.reset();
          std::vector<ElementO> vec_out(vec_acc.size());
          for (int i = 0; i < vec_out.size(); i++) {
            vec_out[i] = static_cast<ElementO>(vec_acc[i]);
          }

          cutlass::DeviceAllocation<ElementO> block_O_chunk;
          block_O_chunk.reset(vec_out.size());
          compat::memcpy<ElementO>(block_O_chunk.get(), vec_out.data(), vec_out.size());
          copy_slice<ExampleRunner, CopySliceOp::O>(
            block_O_chunk.get(), o, q_seq_start, current_q_len, head_size_vo, h, batch_idx, dO, false, true);
        }
      }
    }

    compat::wait();
    // Check if output from CUTLASS kernel and reference kernel are equal within tolerance.
    return cutlass::reference::device::BlockCompareRelativelyEqual(
        block_ref_O.get(), block_O.get(), block_O.size(), ElementO{0.05}, ElementO{0.05});
  }

  /// Initialize operands to be used in the GEMM and reference GEMM
  ProblemShapeType initialize(const Options &options) {
    auto problem_shape_in = cute::make_tuple(options.batch, options.num_heads_q, options.num_heads_kv, options.seq_len_qo, options.seq_len_kv, options.seq_len_kv_cache, options.head_size_qk, options.head_size_vo);
    ProblemShapeType shape;

    decltype(problem_shape_in) problem_size;

    if constexpr (isVarLen) {
      auto [problem_shape_init, problem_shape_launch] = initialize_varlen(problem_shape_in);
      problem_size = problem_shape_init;
      shape = problem_shape_launch;
    } else {
      problem_size = problem_shape_in;
      shape.batch        = options.batch;
      shape.num_heads_q  = options.num_heads_q;
      shape.num_heads_kv = options.num_heads_kv;
      shape.seq_len_qo   = options.seq_len_qo;
      shape.seq_len_kv   = options.seq_len_kv;
      shape.seq_len_kv_cache = options.seq_len_kv_cache;
      shape.head_size_qk = options.head_size_qk;
      shape.head_size_vo = options.head_size_vo;
    }

    auto [batch, num_heads_q, num_heads_kv, seq_len_qo, seq_len_kv, seq_len_kv_cache, head_size_qk, head_size_vo] = problem_size;
    auto shape_Q = cute::make_shape(seq_len_qo, head_size_qk, num_heads_q,  batch);
    auto shape_K = cute::make_shape(seq_len_kv, head_size_qk, num_heads_kv, batch);
    auto shape_V = cute::make_shape(head_size_vo, seq_len_kv, num_heads_kv, batch);
    auto shape_K_cache = cute::make_shape(seq_len_kv_cache, head_size_qk, num_heads_kv, batch);
    auto shape_V_cache = cute::make_shape(head_size_vo, seq_len_kv_cache, num_heads_kv, batch);
    auto shape_O = cute::make_shape(seq_len_qo, head_size_vo, num_heads_q,  batch);

    if (options.layout == "NHD") {
      stride_Q = stride(make_ordered_layout(shape_Q, NHDQLayoutStep{}));
      stride_K = stride(make_ordered_layout(shape_K, NHDQLayoutStep{}));
      stride_V = stride(make_ordered_layout(shape_V, NHDVLayoutStep{}));
      stride_K_cache = stride(make_ordered_layout(shape_K_cache, NHDQLayoutStep{}));
      stride_V_cache = stride(make_ordered_layout(shape_V_cache, NHDVLayoutStep{}));
      stride_O = stride(make_ordered_layout(shape_O, NHDQLayoutStep{}));
    } else {
      stride_Q = stride(make_ordered_layout(shape_Q, HNDQLayoutStep{}));
      stride_K = stride(make_ordered_layout(shape_K, HNDQLayoutStep{}));
      stride_V = stride(make_ordered_layout(shape_V, HNDVLayoutStep{}));
      stride_K_cache = stride(make_ordered_layout(shape_K_cache, HNDQLayoutStep{}));
      stride_V_cache = stride(make_ordered_layout(shape_V_cache, HNDVLayoutStep{}));
      stride_O = stride(make_ordered_layout(shape_O, HNDQLayoutStep{}));
    }

    block_Q.reset(static_cast<std::size_t>(batch) * num_heads_q * seq_len_qo * head_size_qk);
    block_K.reset(static_cast<std::size_t>(batch) * num_heads_kv * seq_len_kv * head_size_qk);
    block_V.reset(static_cast<std::size_t>(batch) * num_heads_kv * seq_len_kv * head_size_vo);
    block_K_cache.reset(static_cast<std::size_t>(batch) * num_heads_kv * seq_len_kv_cache * head_size_qk);
    block_V_cache.reset(static_cast<std::size_t>(batch) * num_heads_kv * seq_len_kv_cache * head_size_vo);
    block_O.reset(static_cast<std::size_t>(batch) * num_heads_q * seq_len_qo * head_size_vo);
    block_ref_O.reset(static_cast<std::size_t>(batch) * num_heads_q * seq_len_qo * head_size_vo);
    // Zero-initialize output buffer for the kernel result
    // block_ref_O is fully written in verify() before being read, so no initialization needed
    compat::memset(block_O.get(), 0, block_O.size() * sizeof_bits_v<ElementO> / 8);
    if (options.use_paged_kv) {
      paged_kv_cache.page_size = options.page_size;
      std::vector<int> num_pages_per_seq{0};
      int num_pages = 0;
      for(int b = 0; b < shape.batch; b++) {
        int seq_len_cache = isVarLen ? cumulative_seqlen_kv_cache[b + 1] - cumulative_seqlen_kv_cache[b] : seq_len_kv_cache;
        int pages_per_seq = ceil_div(seq_len_cache, paged_kv_cache.page_size);
        num_pages_per_seq.push_back(num_pages_per_seq.back() + pages_per_seq);
        num_pages += pages_per_seq;
      }
      paged_kv_cache.page_table.reset(num_pages);

      // initialize block table with random mapping for non-contiguous layout
      std::vector<int> page_mapping(num_pages);
      for (int b = 0; b < shape.batch; ++b) {
        std::vector<int> physical_pages(num_pages_per_seq[b + 1] - num_pages_per_seq[b]);
        std::iota(physical_pages.begin(), physical_pages.end(), 0);
        // shuffle physical pages
        std::shuffle(physical_pages.begin(), physical_pages.end(), std::mt19937{ std::random_device{}() });
        for (int blk = 0; blk < physical_pages.size(); ++blk) {
          int logical_idx = num_pages_per_seq[b] + blk;
          page_mapping[logical_idx] = physical_pages[blk];
        }
      }
      compat::memcpy(paged_kv_cache.page_table.get(), page_mapping.data(), page_mapping.size() * sizeof(int));

      paged_kv_cache.num_pages_per_seq.reset(num_pages_per_seq.size());
      compat::memcpy(paged_kv_cache.num_pages_per_seq.get(), num_pages_per_seq.data(), num_pages_per_seq.size() * sizeof(int));
    }

    initialize_block(block_Q, seed + 2023);
    initialize_block(block_K, seed + 2022);
    initialize_block(block_V, seed + 2021);
    initialize_block(block_K_cache, seed + 2024);
    initialize_block(block_V_cache, seed + 2025);
    
    if (!cumulative_seqlen_q.empty()) {
      device_cumulative_seqlen_q.reset(cumulative_seqlen_q.size());
      device_cumulative_seqlen_q.copy_from_host(cumulative_seqlen_q.data(), cumulative_seqlen_q.size());
    }

    if (!cumulative_seqlen_kv.empty()) {
      device_cumulative_seqlen_kv.reset(cumulative_seqlen_kv.size());
      device_cumulative_seqlen_kv.copy_from_host(cumulative_seqlen_kv.data(), cumulative_seqlen_kv.size());
    }

    if (!cumulative_seqlen_kv_cache.empty()) {
      device_cumulative_seqlen_kv_cache.reset(cumulative_seqlen_kv_cache.size());
      device_cumulative_seqlen_kv_cache.copy_from_host(cumulative_seqlen_kv_cache.data(), cumulative_seqlen_kv_cache.size());
    }

    if constexpr (isVarLen) {
      shape.seq_len_qo.cumulative_length = device_cumulative_seqlen_q.get();
      shape.seq_len_kv.cumulative_length = device_cumulative_seqlen_kv.get();
      shape.seq_len_kv_cache.cumulative_length = device_cumulative_seqlen_kv_cache.get();
    }
    return shape;
  }

  // Note that the GemmUniversalAdapter currently doesn't support flash attention, which is why this
  // secondary `run` function is required to launch the kernel.
  static void run(typename FMHAKernel::Params params)
  {
    namespace syclex = sycl::ext::oneapi::experimental;
    namespace intelex = sycl::ext::intel::experimental;

    dim3 const block = FMHAKernel::get_block_shape();
    dim3 const grid = FMHAKernel::get_grid_shape(params);

    // configure smem size and carveout
    int smem_size = FMHAKernel::SharedStorageSize;

    const auto sycl_block = compat::dim3(block.x, block.y, block.z);
    const auto sycl_grid = compat::dim3(grid.x, grid.y, grid.z);

    // Launch parameters depend on whether SYCL compiler supports work-group scratch memory extension
    compat::experimental::launch_properties launch_props {
      syclex::work_group_scratch_size(smem_size),
    };
    compat::experimental::kernel_properties kernel_props{
      syclex::sub_group_size<cute::intel::sg_size>,
      intelex::grf_size<256>
    };
    compat::experimental::launch_policy policy{sycl_grid, sycl_block, launch_props, kernel_props};
#if defined(CUTLASS_SYCL_PROFILING_ENABLED)
    auto event = compat::experimental::launch<cutlass::device_kernel<FMHAKernel>, FMHAKernel>(policy, params);
    EventManager::getInstance().addEvent(event);
#else
    compat::experimental::launch<cutlass::device_kernel<FMHAKernel>, FMHAKernel, false>(policy, params);
#endif
  }

  cutlass::Status run(const Options &options, const cutlass::KernelHardwareInfo &hw_info) {

    ProblemShapeType shape = initialize(options);

    typename FMHAKernel::Arguments arguments{
      {
        shape,
        block_Q.get(), stride_Q,
        block_K.get(), stride_K,
        block_V.get(), stride_V,
        block_O.get(), stride_O,
        block_K_cache.get(), stride_K_cache,
        block_V_cache.get(), stride_V_cache,
      },
      {
        options.softmax_scale,
        options.use_paged_kv ? paged_kv_cache.page_table.get() : nullptr,
        options.use_paged_kv ? paged_kv_cache.page_size : 0,
        options.use_paged_kv ? paged_kv_cache.num_pages_per_seq.get() : nullptr
      },
      {},
      hw_info
    };

    // Define device-global scratch memory
    size_t workspace_size = FMHAKernel::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    if (!FMHAKernel::can_implement(arguments)) {
      std::cout << "Invalid Problem Size: " << options.batch << 'x' << options.num_heads_q << 'x' <<
        options.seq_len_qo << 'x' << options.seq_len_kv << 'x' << options.head_size_qk << 'x'  << options.head_size_vo
        << (options.is_causal ? "xCausal" : "xNonCausal") << std::endl;
      return cutlass::Status::kErrorInvalidProblem;
    }

    // Initialize the workspace
    CUTLASS_CHECK(FMHAKernel::initialize_workspace(arguments, workspace.get()));

    // Convert host-side arguments to device-side arguments to be passed to the kernel
    auto params = FMHAKernel::to_underlying_arguments(arguments, workspace.get());

    // Run the GEMM
    // Warmup runs
    for (int i = 0; i < options.warmup; ++i) {
      run(params);
    }
    compat::wait();

    if (options.verify != 0) {
      // Verify that the result is correct
      bool passed = verify(shape, options);
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
        run(params);
      }
      compat::wait();
      double cute_time = timer.seconds() / options.iterations;
      double batched_effective_seq_len_qo_x_kv = 0.0;  // flops_qk and flops_pv
      double batched_effective_seq_len_qo = 0.0;       // gbps_q and gbps_o
      double batched_effective_seq_len_kv = 0.0;       // gbps_kv
      double batched_seq_len_kv_cache = 0.0;           // gbps_kv_cache
      if constexpr (isVarLen) {
        // For varlen, we need to iterate over each batch and compute per-batch flops
        for (int b = 0; b < options.batch; ++b) {
          int seq_len_qo = cumulative_seqlen_q[b + 1] - cumulative_seqlen_q[b];
          int seq_len_kv = cumulative_seqlen_kv[b + 1] - cumulative_seqlen_kv[b];
          int seq_len_kv_cache = 0;
          if (options.seq_len_kv_cache > 0) {
            seq_len_kv_cache = cumulative_seqlen_kv_cache[b + 1] - cumulative_seqlen_kv_cache[b];
          }
          auto offset = cute::min(seq_len_qo, seq_len_kv);
          auto discard_seq_coord = seq_len_qo - offset;
          auto full_tile_offset = seq_len_kv - offset;
          // offset + 1 is going to be ceil_div
          auto per_batch_effective_seq_len_kv = options.is_causal ? full_tile_offset + ((offset + 1) / 2.0) : static_cast<double>(seq_len_kv);
          auto per_batch_effective_seq_len_qo = options.is_causal ? seq_len_qo - discard_seq_coord : seq_len_qo;
          double per_batch_qo_x_kv_cache = static_cast<double>(seq_len_qo) * seq_len_kv_cache;  // Full attention to cache
          double per_batch_qo_x_kv_new = per_batch_effective_seq_len_qo * per_batch_effective_seq_len_kv;  // Causal-adjusted for new KV

          batched_effective_seq_len_qo_x_kv += per_batch_qo_x_kv_cache + per_batch_qo_x_kv_new;
          batched_effective_seq_len_qo += per_batch_effective_seq_len_qo;
          batched_effective_seq_len_kv += per_batch_effective_seq_len_kv;
          batched_seq_len_kv_cache += seq_len_kv_cache;
        }
      } else {
        // Non-varlen
        int seq_len_kv_cache = options.seq_len_kv_cache;
        auto offset = cute::min(options.seq_len_qo, options.seq_len_kv);
        auto discard_seq_coord = options.seq_len_qo - offset;
        auto full_tile_offset = options.seq_len_kv - offset;
        auto effective_seq_len_kv = options.is_causal ? full_tile_offset + ((offset + 1) / 2.0) : static_cast<double>(options.seq_len_kv);
        auto effective_seq_len_qo = options.is_causal ? options.seq_len_qo - discard_seq_coord : options.seq_len_qo;
        double qo_x_kv_cache = static_cast<double>(options.seq_len_qo) * seq_len_kv_cache;  // Full attention to cache
        double qo_x_kv_new = effective_seq_len_qo * effective_seq_len_kv;  // Causal-adjusted for new KV

        batched_effective_seq_len_qo_x_kv = options.batch * (qo_x_kv_cache + qo_x_kv_new);
        batched_effective_seq_len_qo = options.batch * effective_seq_len_qo;
        batched_effective_seq_len_kv = options.batch * effective_seq_len_kv;
        batched_seq_len_kv_cache = options.batch * seq_len_kv_cache;
      }

      double flops_qk = 2.0 * options.num_heads_q * batched_effective_seq_len_qo_x_kv * options.head_size_qk;
      double flops_pv = 2.0 * options.num_heads_q * batched_effective_seq_len_qo_x_kv * options.head_size_vo;
      double tflops = ((flops_qk + flops_pv) * 1e-12) / cute_time;
      
      double batched_seq_len_kv_total = batched_effective_seq_len_kv + batched_seq_len_kv_cache;
      double gbps_qk = options.num_heads_q * batched_effective_seq_len_qo * options.head_size_qk * sizeof_bits_v<ElementQ> / 8 +
                       options.num_heads_kv * batched_seq_len_kv_total * options.head_size_qk * sizeof_bits_v<ElementK> / 8;
      double gbps_pv = options.num_heads_kv * batched_seq_len_kv_total * options.head_size_vo * sizeof_bits_v<ElementV> / 8 +
                       options.num_heads_q * batched_effective_seq_len_qo * options.head_size_vo * sizeof_bits_v<ElementO> / 8;
      double gbps = ((gbps_qk + gbps_pv) * 1e-9) / cute_time;
      std::cout << "Batch: " << options.batch << "\tNumHeads_q: " << options.num_heads_q  << "\tNumHeads_kv: " << options.num_heads_kv  << "\tSeq Length QO: " << options.seq_len_qo
                << "\tSeq Length KV: " << options.seq_len_kv << "\tHead Size QK: " << options.head_size_qk << "\tHead Size VO: " << options.head_size_vo
                << "\tCausal Mask: " << (options.is_causal ? "true" : "false") << "\tVariable Sequence Length: " << (options.varlen ? "true" : "false")
                << "\t Scheduler: " << options.scheduler;
      printf("\nPerformance:   %4.3f  GB/s,    %4.3f  TFlop/s,   %6.4f  ms\n\n", gbps, tflops, cute_time * 1000);
    }

    return cutlass::Status::kSuccess;
  }
};

template <bool Causal,
          typename TileShapeQK,
          typename TileShapePV,
          typename TileShapeOutput,
          typename SubgroupLayoutQK,
          typename SubgroupLayoutPV_,      /* void -> default */
          int PipelineStages,
          bool persistent,
          typename ElementQ = bfloat16_t,
          typename ElementK = bfloat16_t,
          typename ElementV = bfloat16_t,
          typename ElementO = float,
          typename MMAOperation_ = void,    /* void -> default */
          typename StrideQ = Stride<int, _1, int, int>,
          typename StrideK = Stride<int, _1, int, int>,
          typename StrideV = Stride<_1, int, int, int>,
          typename StrideO = Stride<int, _1, int, int>,
          typename GmemTiledCopyQ = void,   /* void -> default block 2D */
          typename GmemTiledCopyK = void,
          typename GmemTiledCopyV = void,
          typename GmemTiledCopyO = void>
struct FMHAConfig {

  static constexpr int SGTileQ = get<0>(shape_div(TileShapeQK{}, shape(SubgroupLayoutQK{})))();
  using MMAOperation = cute::conditional_t<is_void_v<MMAOperation_>,
                                           typename cute::conditional_t<
                                               cute::is_same_v<ElementQ, cutlass::float_e5m2_t> || cute::is_same_v<ElementQ, cutlass::float_e4m3_t>,
                                               XE_DPAS_TT<cute::gcd(SGTileQ, 8), float, half_t>,
                                               XE_DPAS_TT<cute::gcd(SGTileQ, 8), float, ElementQ> 
                                           >,
                                           MMAOperation_>;
  using SubgroupLayoutPV = cute::conditional_t<is_void_v<SubgroupLayoutPV_>,
                                               decltype(cutlass::fmha::collective::get_sg_layout_pv(SubgroupLayoutQK{})),
                                               SubgroupLayoutPV_>;

  template <bool isVarLen, bool CachedKV, bool PagedKV, class Scheduler>
  static int run(const Options &options) {
    //
    // Run examples
    //

    // The KernelHardwareInfo struct holds the number of EUs on the GPU with a given device ID. This
    // information is used by the underlying kernel.
    cutlass::KernelHardwareInfo hw_info;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

    using ProblemShapeType = cutlass::fmha::kernel::FMHAProblemShape<isVarLen>;

    using TiledMMAQK = typename TiledMMAHelper<MMA_Atom<MMAOperation>, Layout<TileShapeQK>, SubgroupLayoutQK>::TiledMMA;
    using TiledMMAPV = typename TiledMMAHelper<MMA_Atom<MMAOperation>, Layout<TileShapePV>, SubgroupLayoutPV>::TiledMMA;

    static_assert(get<0>(TileShapeOutput{}) == get<0>(TileShapePV{}),
        "Output tile and P*V tile have different sizes in Q dimension");
    constexpr int VTiles = get<1>(TileShapeOutput{}) / get<1>(TileShapePV{});

    auto make_dummy_tensor = [&](auto val, auto stride) {
      return make_tensor(make_gmem_ptr(&val),
                         make_layout(repeat<rank_v<decltype(stride)>>(1), stride));
    };

    using TensorQ = decltype(make_dummy_tensor(ElementQ{}, StrideQ{}));
    using TensorK = decltype(make_dummy_tensor(ElementK{}, StrideK{}));
    using TensorV = decltype(make_dummy_tensor(ElementV{}, StrideV{}));
    using TensorO = decltype(make_dummy_tensor(ElementO{}, StrideO{}));
    using TensorK_cache = TensorK;
    using TensorV_cache = TensorV;
    using GmemTiledCopyK_cache = GmemTiledCopyK;
    using GmemTiledCopyV_cache = GmemTiledCopyV;

    // Mainloop
    using MainloopDispatchPolicy = cutlass::fmha::XeDefault<PipelineStages>;
    using CollectiveMainloop = cutlass::fmha::collective::FMHAFwdMainloop<
        MainloopDispatchPolicy, Causal, CachedKV, PagedKV,
        TiledMMAQK, TiledMMAPV, VTiles,
        TensorQ, TensorK, TensorV,
        TensorK_cache, TensorV_cache,
        GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV,
        GmemTiledCopyK_cache, GmemTiledCopyV_cache
    >;

    // Epilogue
    using CollectiveEpilogue = cutlass::fmha::collective::FMHAFwdEpilogue<
        CollectiveMainloop,
        TileShapeOutput,
        TensorO,
        GmemTiledCopyO
    >;

    static_assert(!(persistent & Causal), "persistent SDPA kernel not support Causal yet");
    if (options.layout == "NHD") {
      using FMHAKernel = cute::conditional_t<
          is_same_v<Scheduler, cutlass::fmha::kernel::XeFHMAIndividualPersistentTileScheduler>,
          cutlass::fmha::kernel::XeFMHAFwdDynamicSplitKernel<
            ProblemShapeType,
            CollectiveMainloop,
            CollectiveEpilogue,
            Scheduler,
            NHDQLayoutStep,
            NHDQLayoutStep,
            NHDVLayoutStep,
            NHDQLayoutStep>,
          cutlass::fmha::kernel::XeFMHAFwdKernel<
            ProblemShapeType,
            CollectiveMainloop,
            CollectiveEpilogue,
            Scheduler,
            NHDQLayoutStep,
            NHDQLayoutStep,
            NHDVLayoutStep,
            NHDQLayoutStep>>;
      ExampleRunner<FMHAKernel, isVarLen> runner;
      CUTLASS_CHECK(runner.run(options, hw_info));
      return 0;
    } else {
      using FMHAKernel = cute::conditional_t<
          is_same_v<Scheduler, cutlass::fmha::kernel::XeFHMAIndividualPersistentTileScheduler>,
          cutlass::fmha::kernel::XeFMHAFwdDynamicSplitKernel<
            ProblemShapeType,
            CollectiveMainloop,
            CollectiveEpilogue,
            Scheduler,
            HNDQLayoutStep,
            HNDQLayoutStep,
            HNDVLayoutStep,
            HNDQLayoutStep>,
          cutlass::fmha::kernel::XeFMHAFwdKernel<
            ProblemShapeType,
            CollectiveMainloop,
            CollectiveEpilogue,
            Scheduler,
            HNDQLayoutStep,
            HNDQLayoutStep,
            HNDVLayoutStep,
            HNDQLayoutStep>>;
      ExampleRunner<FMHAKernel, isVarLen> runner;
      CUTLASS_CHECK(runner.run(options, hw_info));
      return 0;
    }
  }

  static int run(const Options &options) {
    bool cached_kv = options.seq_len_kv_cache > 0;
    if constexpr (persistent) {
      if (options.use_paged_kv || options.seq_len_kv_cache > 0) {
        std::cerr << "Error: Persistent kernel does not support paged/cached KV cache (use_paged_kv or seq_len_kv_cache > 0)." << std::endl;
        return -1;
      }
      return run<false, false, false, cutlass::fmha::kernel::XeFHMAIndividualPersistentTileScheduler>(options);
    } else if (options.use_paged_kv && !options.varlen) {
      return run<false, true, true, cutlass::fmha::kernel::XeFHMAIndividualTileScheduler>(options);
    } else if(!options.use_paged_kv && options.varlen && !cached_kv) {
      return run<true, false, false, cutlass::fmha::kernel::XeFHMAIndividualTileScheduler>(options);
    } else if(!options.use_paged_kv && !options.varlen && !cached_kv) {
      return run<false, false, false, cutlass::fmha::kernel::XeFHMAIndividualTileScheduler>(options);
    } else if (!options.use_paged_kv && options.varlen && cached_kv) {
      return run<true, true, false, cutlass::fmha::kernel::XeFHMAIndividualTileScheduler>(options);
    } else if (!options.use_paged_kv && !options.varlen && cached_kv) {
      return run<false, true, false, cutlass::fmha::kernel::XeFHMAIndividualTileScheduler>(options);
    } else {
      return run<true, true, true, cutlass::fmha::kernel::XeFHMAIndividualTileScheduler>(options);
    }
  }
};
