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

#include <sycl/sycl.hpp>
#include <cute/util/compat.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <cute/tensor.hpp>
#include <cute/quantization/quantize.hpp>

#include "cutlass/kernel_hardware_info.h"
#include "cutlass/platform/platform.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/sycl_event_manager.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/command_line.h"

#include "../../common/sycl_cute_common.hpp"

#if defined(__clang__)
  #pragma clang diagnostic ignored "-Wpass-failed"
  #pragma clang diagnostic ignored "-Wdeprecated-declarations"
#elif defined(__GNUC__)
  #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

using namespace cute;

template <class ATensor, class BTensor, class CTensor,
          class QTensor, class STensor,
          class TiledMMA, int BlockSize = 32>
void
gemm_device(ATensor   const& A,         // (M,K)
            BTensor   const& B,         // (N,K)
            CTensor        & C,         // (M,N)
            QTensor        & Q,         // (M,N)
            STensor        & S,         // (M, N/BlockSize)
            TiledMMA const & mma,
            int              verify)
{
  // -----
  // Setup
  // -----

  using QType = typename QTensor::value_type;
  using SType = typename STensor::value_type;

  /* Get workgroup and local IDs */
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<2>();
  auto wg_m = int(item.get_group(1));
  auto wg_n = int(item.get_group(0));
  auto local_id = int(item.get_local_id(0));

  /* Create proxy coordinate tensors for each global tensor */
  Tensor cA = make_identity_tensor(A.shape());   // (M,K)
  Tensor cB = make_identity_tensor(B.shape());   // (N,K)
  Tensor cC = make_identity_tensor(C.shape());   // (M,N)
  Tensor cQ = make_identity_tensor(Q.shape());   // (M,N)

  /* Split GEMM into workgroup tiles, and identify our workgroup's tile (wg_coord) */
  auto wg_tile = mma.tile_mnk();
  auto wg_coord = make_coord(wg_m, wg_n, 0);

  Tensor gA = local_tile(cA, select<0,2>(wg_tile), make_coord(wg_m,_));  // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(cB, select<1,2>(wg_tile), make_coord(wg_n,_));  // (BLK_N,BLK_K,k)
  Tensor gC = local_tile(cC, wg_tile, wg_coord, Step<_1,_1, X>{});       // (BLK_M,BLK_N)
  Tensor gQ = local_tile(cQ, wg_tile, wg_coord, Step<_1,_1, X>{});       // (BLK_M,BLK_N)
  Tensor gS = local_tile(S, make_tile(get<0>(wg_tile), get<1>(wg_tile) / BlockSize, _1{}), wg_coord, Step<_1,_1, X>{});

  /* Create block 2D TiledCopies */
  auto copy_a = make_block_2d_copy_A(mma, A);
  auto copy_b = make_block_2d_copy_B(mma, B);
  auto copy_c = make_block_2d_copy_D(mma, C);

  /* Slice TiledCopy/TiledMMA operations to thread (work-item) level */
  auto thr_mma    =    mma.get_slice(local_id);
  auto thr_copy_a = copy_a.get_slice(local_id);
  auto thr_copy_b = copy_b.get_slice(local_id);
  auto thr_copy_c = copy_c.get_slice(local_id);

  /* Register fragments for MMA */
  auto tCrA = thr_mma.partition_sg_fragment_A(gA(_,_,0));
  auto tCrB = thr_mma.partition_sg_fragment_B(gB(_,_,0));

  /* Register fragments for copies */
  auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_,_,0));
  auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_,_,0));

  /* Partition global tensor (proxies) for copies */
  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tBgB = thr_copy_b.partition_S(gB);

  /* Partition C */
  auto tCrC = thr_mma.partition_sg_fragment_C(gC);
  Tensor tCgC = thr_mma.partition_C(gC);    /* also matches copy_c's source layout */

  /* Create prefetch TiledCopy instances */
  auto prefetch_a = make_block_2d_prefetch(copy_a);
  auto prefetch_b = make_block_2d_prefetch(copy_b);

  auto thr_prefetch_A = prefetch_a.get_slice(local_id);
  auto thr_prefetch_B = prefetch_b.get_slice(local_id);

  /* Partition global tensor (proxies) for prefetch */
  auto pAgA = thr_prefetch_A.partition_S(gA);
  auto pBgB = thr_prefetch_B.partition_S(gB);

  /* Prefetch distance, in units of k tiles */
  const int prefetch_dist = 3;

  // ------
  // Kernel
  // ------

  constexpr auto barrier_scope = SPIRVScope::ScopeWorkgroup;

  int k_tile_count = ceil_div(shape<1>(A), get<2>(wg_tile));
  int k_tile_prefetch = 0;

  /* Clear the accumulators */
  clear(tCrC);

  /* Warm up loops with prefetch to L1 */
  CUTE_UNROLL
  for (; k_tile_prefetch < prefetch_dist; k_tile_prefetch++) {
    prefetch(prefetch_a, pAgA(_,_,_,k_tile_prefetch));
    prefetch(prefetch_b, pBgB(_,_,_,k_tile_prefetch));
  }

  /* Main loop */
  for (int k_tile = 0; k_tile < k_tile_count; k_tile++, k_tile_prefetch++) {
    /* Split barrier keeping threads loosely together */
    barrier_arrive(barrier_scope);

    /* Copy A/B from global memory (ideally L1 cache) to registers */
    copy(copy_a, tAgA(_,_,_,k_tile), tArA);
    copy(copy_b, tBgB(_,_,_,k_tile), tBrB);

    /* Prefetch A/B tiles to L1 */
    prefetch(prefetch_a, pAgA(_,_,_,k_tile_prefetch));
    prefetch(prefetch_b, pBgB(_,_,_,k_tile_prefetch));

    /* Shuffle data from copy fragments to MMA fragments */
    reorder(tArA, tCrA);
    reorder(tBrB, tCrB);

    /* Accumulate C += A * B */
    gemm(mma, tCrA, tCrB, tCrC);

    /* Other half of split barrier */
    barrier_wait(barrier_scope);
  }

  if (verify != 0) {
    /* Write C (GEMM result) to global memory*/
    copy(copy_c, tCrC, tCgC);
  }

  /* Partition quantization output */
  auto copy_q = make_block_2d_copy_D(mma, Q);
  auto thr_copy_q = copy_q.get_slice(local_id);
  auto trQ = thr_copy_q.partition_sg_fragment_S(gQ);
  auto tgQ = thr_copy_q.partition_D(gQ);

  /* Partition scale output */
  using ThrLayout = typename TiledMMA::ThrLayoutVMNK;
  static constexpr int THR_M = get<1>(ThrLayout{}.shape());
  static constexpr int THR_N = get<2>(ThrLayout{}.shape());
  auto c_shape = atuple_coshape(tCrC.tv_layout());
  constexpr auto M = decltype(get<0>(c_shape))::value;
  constexpr auto N = decltype(get<1>(c_shape))::value;
  constexpr int NumScaleBlocks = N / BlockSize;
  constexpr int rows_per_thr = M / intel::sg_size;
  constexpr int per_thr = rows_per_thr * NumScaleBlocks;
  auto s_tensor = make_tensor<SType>(Layout<Shape<Int<per_thr>>>{});
  auto s_tv_layout = make_layout(
      make_shape(Int<intel::sg_size>{}, make_shape(Int<rows_per_thr>{}, Int<NumScaleBlocks>{})),
      make_stride(ScaledBasis<Int<1>,0>{},
                  make_stride(ScaledBasis<Int<intel::sg_size>,0>{}, ScaledBasis<Int<1>,1>{})));
  auto trS = make_subgroup_tensor(s_tensor, s_tv_layout);

  quantize<BlockSize>(tCrC, trQ, trS);

  /* Write LP quantize output to global memory */
  copy(copy_q, trQ, tgQ);

  /* Compute per-SG tile of gS using SGLayout convention:
  SG index = sm * THR_N + sn, so sm = idx / THR_N, sn = idx % THR_N.
  local_partition can't be used here because it always decomposes the
  thread index in column-major order, which doesn't match the n-major
  SGLayout.  Use local_tile with explicitly computed SG coordinates. */
  int sg_idx = local_id / intel::sg_size;
  int sg_m = sg_idx / int(THR_N);
  int sg_n = sg_idx % int(THR_N);
  auto tgS = local_tile(gS, make_shape(Int<M>{}, Int<NumScaleBlocks>{}), make_coord(sg_m, sg_n));
  /* Write scale output to global memory.
      s_tensor holds per_thr = (M/16) * NumScaleBlocks values per thread.
      The distributed TV layout assigns each lane its own rows:
        lane t → rows {t, t+16, t+32, ...}.
      s_tensor is indexed colexicographically within the per-thread subset:
        s_tensor(r + b * rows_per_thr) = scale for row (lane + r*16), block b.
      Bounds checks guard against out-of-bounds SGs when problem < WG tile. */
  int lane = local_id % intel::sg_size;
  int row_base = wg_m * int(get<0>(wg_tile)) + sg_m * int(M);
  int col_base = wg_n * (int(get<1>(wg_tile)) / BlockSize) + sg_n * NumScaleBlocks;
  int s_rows = int(size<0>(S));
  int s_cols = int(size<1>(S));
  for (int r = 0; r < rows_per_thr; ++r) {
    int row = lane + r * int(intel::sg_size);
    if (row_base + row < s_rows) {
      for (int b = 0; b < NumScaleBlocks; ++b) {
        if (col_base + b < s_cols) {
          tgS(row, b) = s_tensor(r + b * rows_per_thr);
        }
      }
    }
  }
}

template <typename TA, typename TB, typename TC>
auto
choose_mma_op()
{
  if constexpr (is_complete_v<XE_DPAS_TT<8, TC, TA, TB>>)
    return XE_DPAS_TT<8, TC, TA, TB>{};
  else if constexpr (is_same_v<TA, cute::bfloat16_t>)
    return XE_DPAS_TT<8, float, cute::bfloat16_t>{};
  else  /* Use f16 by default as upconversion sequences are typically faster */
    return XE_DPAS_TT<8, float, cute::half_t>{};
}

template <class ATensor, class BTensor, class CTensor>
auto
choose_tiled_mma(ATensor const& A, BTensor const& B, CTensor const&)
{
  using TA = typename ATensor::element_type;
  using TB = typename BTensor::element_type;
  using TC = typename CTensor::element_type;

  auto op = choose_mma_op<TA,TB,TC>();

  constexpr bool byte = (cute::max(sizeof_bits_v<TA>, sizeof_bits_v<TB>) <= 8);
  constexpr bool a_t = is_constant_v<1, decltype(stride<0>(A))>;
  constexpr bool b_n = is_constant_v<1, decltype(stride<0>(B))>;

  constexpr bool use_1x_dpas_per_k = a_t                                  // Use one DPAS in k dimension for A^T case
                                  || (byte && b_n);                       //  pending compiler improvements (also int8 B^N).

  using _K = conditional_t<use_1x_dpas_per_k,
                           C<op.K>, C<op.K*2>>;

  using WGTile = Shape<_256, _256, _K>;                               // 256x256 WG tile size
  using SGLayout = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;  // 8x4 SG tiling, n-major

  using MMA = typename TiledMMAHelper<MMA_Atom<decltype(op)>, Layout<WGTile>, SGLayout>::TiledMMA;

  return MMA{};
}

template <class, class, class, char, char> class GemmCuteName;
template <class ATensor, class BTensor, class CTensor, class QTensor, class STensor, typename TA, typename TB, typename TC, char layoutA, char layoutB>
void
gemm_cute(sycl::queue &Queue,
          ATensor   const& A,         // (M,K)
          BTensor   const& B,         // (N,K)
          CTensor        & C,         // (M,N)
          QTensor        & Q,
          STensor        & S,
          int              verify = 0)
{
  auto mma = choose_tiled_mma(A, B, C);

  sycl::range<2> local = {size(mma), 1};
  sycl::range<2> global = {local[0] * ceil_div(shape<0>(B), get<1>(mma.tile_mnk())),
                           local[1] * ceil_div(shape<0>(A), get<0>(mma.tile_mnk()))};

  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;

  syclex::properties kernel_props {
    syclex::sub_group_size<intel::sg_size>,
#if (SYCL_INTEL_TARGET == 35)
    intelex::grf_size<512>
#else
    intelex::grf_size<256>
#endif
  };

  auto event = Queue.parallel_for<GemmCuteName<TA, TB, TC, layoutA, layoutB>>(sycl::nd_range<2>(global, local), kernel_props,
    [=](auto) {
      gemm_device(A, B, C, Q, S, mma, verify);
    }
  );

  EventManager::getInstance().addEvent(event);
}

template <class...> class GemmVerifyKernelName;
template <class ATensor, class BTensor, class CTensor>
bool
gemm_verify(sycl::queue &Q,
            ATensor const& A,         // (M,K)
            BTensor const& B,         // (N,K)
            CTensor const& C)         // (M,N)
{
  int m = size<0>(A);
  int n = size<0>(B);
  int k = size<1>(A);

  auto ok = sycl::malloc_shared<bool>(1, Q);
  *ok = true;

  Q.parallel_for<GemmVerifyKernelName<ATensor, BTensor, CTensor>>(sycl::range<2>(m, n), [=](sycl::item<2> id) {
    int i = id[0], j = id[1];

    using AccType = typename CTensor::element_type;
    using SignedAccType = ensure_signed_t<AccType>;

    auto c = AccType(0);
    for (int h = 0; h < k; h++)
      c += AccType(A(i,h)) * AccType(B(j,h));

    auto tol = AccType(static_cast<float>(std::numeric_limits<AccType>::epsilon()) * 2 * k);
    if constexpr (std::is_same_v<AccType, float>)
    {
      //loose tolerance for float AccType
      tol = 1e-5f * k;
    }

    if (std::abs(SignedAccType(c - AccType(C(i,j)))) > tol) {
#ifdef SHOW_DIFF
      printf("Error at (%d,%d): got %f, expected %f\n", i, j, double(C(i,j)), double(c));
#endif
      *ok = false;
    }
  }).wait();

  bool read_ok = *ok;

  sycl::free(ok, Q);

  return read_ok;
}

// ============================================================================
// Host-side verification for block-wise quantization
// ============================================================================
//
// Reads the GEMM result C (M,N), the quantized output Q (M,N), and the
// scale factors S (M, N/BlockSize) from shared USM memory, then performs
// the same block-wise quantization algorithm on C and compares the results.
//
// Algorithm (matches the GPU implementation in quantize.hpp):
//   1. For each (row m, block b): block_amax = max(|C(m, b*BS + k)|) for k in [0, BS)
//   2. scale = round_toward_zero(target_max / block_amax)
//   3. q[i] = round_to_nearest(C[i] * scale)
//
template <int BlockSize, class CTensor, class QTensor, class STensor>
bool
quantize_verify(CTensor const& C,     // (M, N) — GEMM result (float)
                QTensor const& Q,     // (M, N) — quantized output
                STensor const& S)     // (M, N/BlockSize) — scale factors
{
  using SrcType   = typename CTensor::element_type;
  using DstType   = typename QTensor::element_type;
  using ScaleType = typename STensor::element_type;

  using FloatToScale = cutlass::NumericConverter<ScaleType, float,
                        cutlass::FloatRoundStyle::round_toward_zero>;
  using HalfToDst    = cutlass::NumericConverter<DstType, cutlass::half_t,
                        cutlass::FloatRoundStyle::round_to_nearest>;

  const float target_max = static_cast<float>(
      cutlass::platform::numeric_limits<DstType>::max());

  int m = size<0>(C);
  int n = size<1>(C);
  int num_blocks = n / BlockSize;

  bool ok = true;

  // --- Verify scales ---
  for (int row = 0; row < m; ++row) {
    for (int b = 0; b < num_blocks; ++b) {
      // Phase 1: block_amax
      float block_amax = 0.0f;
      for (int k = 0; k < BlockSize; ++k) {
        float val = static_cast<float>(C(row, b * BlockSize + k));
        block_amax = std::max(block_amax, std::abs(val));
      }

      // Phase 2: compute expected scale
      float s = (block_amax > 0.0f) ? (target_max / block_amax) : 0.0f;
      ScaleType ref_scale = FloatToScale{}(s);

      ScaleType gpu_scale = S(row, b);
      if (static_cast<float>(gpu_scale) != static_cast<float>(ref_scale)) {
        printf("Scale mismatch at (%d, %d): got %f, expected %f\n",
               row, b, double(gpu_scale), double(ref_scale));
        return false;
      }

      // Phase 3: verify quantized elements in this block
      // GPU ASM does: F32 mul → F32→HF mov → HF→FP8 fcvt
      for (int k = 0; k < BlockSize; ++k) {
        int col = b * BlockSize + k;
        float val    = static_cast<float>(C(row, col));
        float scaled = val * s;
        cutlass::half_t h = static_cast<cutlass::half_t>(scaled);
        DstType ref_q = HalfToDst{}(h);

        DstType gpu_q = Q(row, col);
        if (static_cast<float>(gpu_q) != static_cast<float>(ref_q)) {
          printf("Quantize mismatch at (%d, %d): got %f, expected %f\n",
                 row, col, double(gpu_q), double(ref_q));
          ok = false;
        }
      }
    }
  }

  return ok;
}

template <typename TA, typename TB, typename TC, typename TQ, typename TS, int BlockSize = 32,
          char layoutA = 'R', char layoutB = 'R'>
void
test_case(sycl::queue &Queue, int m, int n, int k, int iterations, int verify)
{
  std::cout << type_str<TA>() << " (" << layoutA << ") x "
            << type_str<TB>() << " (" << layoutB << ") -> "
            << type_str<TC>() << std::endl;

  // Transpose B to match CuTe conventions
  constexpr char tlayoutB = layoutB ^ ('R' ^ 'C');

  // Prepare data:
  auto A = make_shared_usm_tensor<TA,  layoutA>(Queue, m, k);
  auto B = make_shared_usm_tensor<TB, tlayoutB>(Queue, n, k);
  auto C = make_shared_usm_tensor<TC,      'R'>(Queue, m, n);
  auto Q = make_shared_usm_tensor<TQ,      'R'>(Queue, m, n);
  auto S = make_shared_usm_tensor<TS,      'R'>(Queue, m, n / BlockSize);

  random_fill(A);
  random_fill(B);
  zero_fill(C);

  bool ok = true;
  
  auto A_ref = make_shared_usm_tensor<float,  layoutA>(Queue, m, k);
  auto B_ref = make_shared_usm_tensor<float, tlayoutB>(Queue, n, k);

  copy(A, A_ref);
  copy(B, B_ref);

  subbyte_pack(A);
  subbyte_pack(B);

  // Run the GEMM
  gemm_cute<decltype(A), decltype(B), decltype(C), decltype(Q), decltype(S), TA, TB, TC, layoutA, layoutB>(Queue, A, B, C, Q, S, verify);
  Queue.wait_and_throw();

  if (verify != 0) {  
    ok = gemm_verify(Queue, A_ref, B_ref, C);
    std::cout << (ok ? "gemm passed" : "gemm failed") << std::endl;

    bool qok = quantize_verify<BlockSize>(C, Q, S);
    std::cout << (qok ? "quantize passed" : "quantize failed");
    ok = ok && qok;
    // TODO: Throw exception or error when verification fails, this requires refactor for the whole example.
  } else {
    std::cout << "verification skipped";
  }

  free_usm_tensor(A_ref, Queue);
  free_usm_tensor(B_ref, Queue);

  if (ok) { 
    // If verification passed or skipped, run performance test
    if (iterations > 0) {
      // Test performance:
      GPU_Clock timer;

      timer.start();
      for (int i = 0; i < iterations; ++i)
        gemm_cute<decltype(A), decltype(B), decltype(C), decltype(Q), decltype(S), TA, TB, TC, layoutA, layoutB>(Queue, A, B, C, Q, S);
      Queue.wait_and_throw();

      double avg = timer.seconds() / iterations;
#if defined(CUTLASS_TEST_FOR_CRI)      
      // Use MF/s instead of TF/s as we always use small problem size on CRI 
      // simulator, will remove this when HW is available
      double tops = (2.0*m*n*k) * 1e-12 * 1e6;
      printf(", %4.3f MF/s", tops / avg, avg*1000);
#else
      double tops = (2.0*m*n*k) * 1e-12;
      printf(", %4.3f TF/s", tops / avg, avg*1000);
#endif
    } else {
      printf(", performance benchmark skipped due to 0 iterations");
    }
  } else {
    printf(", performance benchmark skipped due to verification failure");
  }

  free_usm_tensor(A, Queue);
  free_usm_tensor(B, Queue);
  free_usm_tensor(C, Queue);
  free_usm_tensor(Q, Queue);
  free_usm_tensor(S, Queue);

  std::cout << '\n';

  // Pause for a short period of time to allow the GPU to cool.
  static bool first = true;
  if (first)
    first = false;
  else
    sleep(1);
}


int main(int argc, char** argv)
{
  int m, n, k, iterations, verify;
  cutlass::CommandLine cmd(argc, const_cast<const char**>(argv));
  cmd.get_cmd_line_argument("m", m, 1024);
  cmd.get_cmd_line_argument("n", n, 1024);
  cmd.get_cmd_line_argument("k", k, 4096);
  cmd.get_cmd_line_argument("iterations", iterations, 100);
  cmd.get_cmd_line_argument("verify", verify, 1);

  sycl::queue Q = compat::get_default_queue();

  test_case<bfloat16_t, bfloat16_t, float, float_e5m2_t, float_ue8m0_t, 32, 'R', 'R'>(Q, m, n, k, iterations, verify);
}
