/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
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
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#include <sycl/sycl.hpp>
#include <cute/util/compat.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <cute/tensor.hpp>

#include "cutlass/kernel_hardware_info.h"
#include "cutlass/platform/platform.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/sycl_event_manager.hpp"

#include "../../common/sycl_cute_common.hpp"

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

using namespace cute;

template <typename TB> CUTE_DEVICE TB apply_scale(TB &x, float &y) {
  static_assert(is_any_of_v<TB, bfloat16_t, half_t>, "Only BF16 & FP16 are supported");
  uint16_t z = sycl::bit_cast<uint16_t>(x);
#if defined(__SYCL_DEVICE_ONLY__) && defined(SYCL_INTEL_TARGET)
  if constexpr (is_same_v<TB, half_t>) {
    asm("{\n"
        ".decl Z_FP16 v_type=G type=HF num_elts=16 alias=<%0,0>\n"
        ".decl Y_FP32 v_type=G type=F num_elts=16 alias=<%1,0>\n"
        "mul (M1, 16) Z_FP16(0,0)<1> Z_FP16(0,0)<1;1,0> Y_FP32(0,0)<1;1,0>\n"
        "}\n"
        : "+rw"(z)
        : "rw"(y));
  } else {
    asm("{\n"
        ".decl Z_BF16 v_type=G type=BF num_elts=16 alias=<%0,0>\n"
        ".decl Y_FP32 v_type=G type=F num_elts=16 alias=<%1,0>\n"
        "mul (M1, 16) Z_BF16(0,0)<1> Z_BF16(0,0)<1;1,0> Y_FP32(0,0)<1;1,0>\n"
        "}\n"
        : "+rw"(z)
        : "rw"(y));
  }
#endif
  return sycl::bit_cast<TB>(z);
}

template <int WG_N, int SG_N, int q_group_size, class ATensor, class BTensor,
          class STensor, class CTensor,
          class TiledMMA>
CUTE_DEVICE void gemm_device(ATensor const &A, // (M,K)
                             BTensor const &B, // (N,K)
                             STensor const &S, // (K/q_group_size, N)
                             CTensor &C,       // (M,N)
                             TiledMMA const &mma) {
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<2>();
  auto wg_m = int(item.get_group(1));
  auto wg_n = int(item.get_group(0));
  auto local_id = int(item.get_local_id(0));
  auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
  uint32_t sg_id = sg.get_group_linear_id();
  uint32_t lane = sg.get_local_linear_id();

  auto total_N = get<0>(B.shape());

  Tensor cA = make_identity_tensor(A.shape()); // (M,K)
  Tensor cB = make_identity_tensor(B.shape()); // (N,K)
  Tensor cC = make_identity_tensor(C.shape()); // (M,N)
  Tensor cS = make_identity_tensor(S.shape()); // (K/q_group_size,N)
  Tensor cScales_per_sg =
      make_identity_tensor(make_shape(Int<1>{}, Int<SG_N>{}));
  auto wg_tile = mma.tile_mnk();
  auto wg_coord = make_coord(wg_m, wg_n, 0);

  Tensor gA = local_tile(cA, select<0, 2>(wg_tile),
                         make_coord(wg_m, _)); // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(cB, select<1, 2>(wg_tile),
                         make_coord(wg_n, _)); // (BLK_N,BLK_K,k)
  Tensor gC =
      local_tile(cC, wg_tile, wg_coord, Step<_1, _1, X>{}); // (BLK_M,BLK_N)

  constexpr int num_N_SG_tiles = WG_N / SG_N;
  constexpr int num_scales_per_col = (SG_N == 32) ? 4 : 2;

  // When we use E8M0, the compiler behaves differently & loads more data than
  // needed. The rest is discarded.
  // BF16 or FP16 scales are also supported, but please choose a suitable
  // tiling scheme that'd avoid register spills.
  using scaleLoadType =
      conditional_t<is_same_v<typename STensor::element_type, float_ue8m0_t>,
                    int8_t, int16_t>;

  auto S_tile = coalesce(
      local_tile(S, make_shape(Int<1>{}, Int<WG_N>{}), make_coord(_, wg_n)));

  auto copy_a = make_block_2d_copy_A(mma, A);
  auto copy_b = make_block_2d_copy_B(mma, B);
  auto copy_c = make_block_2d_copy_D(mma, C);

  auto thr_mma = mma.get_slice(local_id);
  auto thr_copy_a = copy_a.get_slice(local_id);
  auto thr_copy_b = copy_b.get_slice(local_id);
  auto thr_copy_c = copy_c.get_slice(local_id);

  auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
  auto tCrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));

  auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
  auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_, _, 0));
  auto tCrC = thr_mma.partition_sg_fragment_C(gC);

  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tBgB = thr_copy_b.partition_S(gB);
  Tensor tCgC = thr_copy_c.partition_D(gC);

  auto prefetch_a = make_block_2d_prefetch(copy_a);
  auto prefetch_b = make_block_2d_prefetch(copy_b);

  auto thr_prefetch_A = prefetch_a.get_slice(local_id);
  auto thr_prefetch_B = prefetch_b.get_slice(local_id);

  auto pAgA = thr_prefetch_A.partition_S(gA);
  auto pBgB = thr_prefetch_B.partition_S(gB);

  const int prefetch_dist = 3;

  constexpr int barrier_scope = 2;

  int k_tile_count = ceil_div(shape<1>(A), get<2>(wg_tile));
  int k_tile_prefetch = 0;
  constexpr int num_threads_per_sg = 16;

  typename STensor::element_type
      frag[num_scales_per_col / 2]; // per-thread registers (compiler
                                    // will keep in regs)
  float frag_fp32[num_scales_per_col];
  // assuming SG_K = WG_K = 32
  constexpr int frequency_scale_change = q_group_size / get<2>(wg_tile);
  Tensor scales_e8m0 =
      make_tensor(make_rmem_ptr(frag),
                  make_layout(make_shape(Int<num_scales_per_col / 2>{})));
  Tensor scales_float =
      make_tensor(make_rmem_ptr(frag_fp32),
                  make_layout(make_shape(Int<num_scales_per_col>{})));

  auto srcTVLayout = make_layout(
      make_shape(Int<num_threads_per_sg>{}, Int<num_scales_per_col / 2>{}),
      make_stride(Int<1>{}, Int<num_threads_per_sg>{}));
  auto dstTVLayout = make_layout(
      make_shape(make_shape(Int<2>{}, Int<num_threads_per_sg / 2>{}),
                 make_shape(Int<num_scales_per_col / 2>{})),
      make_stride(make_stride(Int<0>{}, Int<1>{}), make_stride(Int<8>{})));
  auto scales_e8m0_sg_tensor = make_subgroup_tensor(scales_e8m0, srcTVLayout);
  auto scales_float_sg_tensor = make_subgroup_tensor(scales_float, dstTVLayout);

  /* Warm up loops with prefetch to L1 */
  CUTE_UNROLL
  for (; k_tile_prefetch < prefetch_dist; k_tile_prefetch++) {
    prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
    prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
  }
  /* Main loop */
  for (int k_tile = 0; k_tile < k_tile_count; k_tile++, k_tile_prefetch++) {
    barrier_arrive(barrier_scope);

    copy(copy_b, tBgB(_, _, _, k_tile), tBrB);
    prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
    reorder(tBrB, tCrB);

    if (k_tile % frequency_scale_change == 0) {
      auto scales_tensor = make_tensor(
          make_gmem_ptr(reinterpret_cast<scaleLoadType *>(
              static_cast<void *>(cute::raw_pointer_cast(
                  S_tile.data() + (SG_N * (sg_id % num_N_SG_tiles)) +
                  (k_tile / frequency_scale_change) * total_N)))),
          make_layout(make_shape(Int<1>{}, Int<SG_N>{})));
      auto copy_scales = make_block_2d_copy(
          XE_LOAD_2D<sizeof_bits_v<typename STensor::element_type>, 1, SG_N,
                     SG_N>{},
          scales_tensor);
      auto thr_copy_scales = copy_scales.get_slice(lane);
      auto scales_per_thread = thr_copy_scales.partition_S(cScales_per_sg);
      copy(copy_scales, scales_per_thread(_, 0, 0), scales_e8m0);
      reorder(scales_e8m0_sg_tensor, scales_float_sg_tensor);
      // For non MX-format scaledMM, it'd be better if prefetch is outside
      // Even in this file, we could use if constexpr to add conditionals
      // for that case, but it'd make the code messy because it requires
      // duplication
      if (k_tile != (k_tile_count - frequency_scale_change)) {
        auto next_scales_tensor = make_tensor(
            make_gmem_ptr(reinterpret_cast<scaleLoadType *>(
                static_cast<void *>(cute::raw_pointer_cast(
                    S_tile.data() + (SG_N * (sg_id % num_N_SG_tiles)) +
                    ((k_tile / frequency_scale_change) + 1) * total_N)))),
            make_layout(make_shape(Int<1>{}, Int<SG_N>{})));
        auto prefetch_scales = make_block_2d_prefetch<1>(
            make_shape(Int<1>{}, Int<SG_N>{}), next_scales_tensor);
        auto thr_prefetch_scales = prefetch_scales.get_slice(lane);
        auto pSgS = thr_prefetch_scales.partition_S(cScales_per_sg);
        prefetch(prefetch_scales, pSgS(_, 0, 0));
      }
    }

    copy(copy_a, tAgA(_, _, _, k_tile), tArA);
    prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
    reorder(tArA, tCrA);

    // Instead of hardcoding, figure out CuTe algebra based
    // transformations that can lead to generic code.
    auto scale0 = scales_float_sg_tensor[0];
    auto scale1 = scales_float_sg_tensor[1];
    if (num_scales_per_col == 4) {
      auto scale2 = scales_float_sg_tensor[2];
      auto scale3 = scales_float_sg_tensor[3];
      CUTE_UNROLL
      for (int i = 0; i < 16; i += 2) {
        tCrB[i] = apply_scale(tCrB[i], scale0);
        tCrB[i + 1] = apply_scale(tCrB[i + 1], scale1);
      }
      CUTE_UNROLL
      for (int i = 16; i < 32; i += 2) {
        tCrB[i] = apply_scale(tCrB[i], scale2);
        tCrB[i + 1] = apply_scale(tCrB[i + 1], scale3);
      }
      CUTE_UNROLL
      for (int i = 32; i < 48; i += 2) {
        tCrB[i] = apply_scale(tCrB[i], scale0);
        tCrB[i + 1] = apply_scale(tCrB[i + 1], scale1);
      }
      CUTE_UNROLL
      for (int i = 48; i < 64; i += 2) {
        tCrB[i] = apply_scale(tCrB[i], scale2);
        tCrB[i + 1] = apply_scale(tCrB[i + 1], scale3);
      }
    } else {
      CUTE_UNROLL
      for (int i = 0; i < 32; i += 2) {
        tCrB[i] = apply_scale(tCrB[i], scale0);
        tCrB[i + 1] = apply_scale(tCrB[i + 1], scale1);
      }
    }

    gemm(mma, tCrA, tCrB, tCrC);
    barrier_wait(barrier_scope);
  }
  auto tCrC_final = thr_copy_c.partition_sg_fragment_S(gC);
  reorder(tCrC, tCrC_final);
  copy(copy_c, tCrC_final, tCgC);
}

template <typename TA, typename TB, typename TC> auto choose_mma_op() {
  if constexpr (is_complete_v<XE_DPAS_TT<8, TC, TA, TB>>) {
    return XE_DPAS_TT<8, TC, TA, TB>{};
  } else if constexpr (is_same_v<TA, cute::bfloat16_t>) {
    return XE_DPAS_TT<8, float, cute::bfloat16_t>{};
  } else { /* Use f16 by default as upconversion sequences are typically faster
            */
    return XE_DPAS_TT<8, float, cute::half_t>{};
  }
}

template <class ATensor, class BTensor, class CTensor>
auto choose_tiled_mma(ATensor const &A, BTensor const &B, CTensor const &) {
  using TA = typename ATensor::element_type;
  using TB = typename BTensor::element_type;
  using TC = typename CTensor::element_type;

  auto op = choose_mma_op<TA, TB, TC>();

  using WGTile = Shape<_256, _256, _32>; // 256x256 WG tile size
  using SGLayout4x8 =
      Layout<Shape<_4, _8, _1>, Stride<_8, _1, _0>>; // 4x8 SG tiling, n-major

  using MMA = typename TiledMMAHelper<MMA_Atom<decltype(op)>, Layout<WGTile>,
                                      SGLayout4x8>::TiledMMA;

  return MMA{};
}

template <class, class, char, char, int> class GemmCuteName;
template <class ATensor, class BTensor, class STensor, class CTensor,
          typename TA, typename TB, char layoutA, char layoutB,
          int q_group_size = 32>
void gemm_cute(sycl::queue &Q,
               ATensor const &A, // (M,K)
               BTensor const &B, // (N,K)
               STensor const &S,
               CTensor &C) // (M,N)
{

  auto mma = choose_tiled_mma(A, B, C);
  auto wg_m = get<0>(mma.tile_mnk());
  auto wg_n = get<1>(mma.tile_mnk());
  auto wg_k = get<2>(mma.tile_mnk());
  using SGLayout =
      Layout<Shape<_4, _8, _1>, Stride<_8, _1, _0>>; // 4x8 SG tiling, n-major
  auto sg_n = wg_n / get<1>(SGLayout{}.shape());
  auto sg_k = wg_k / get<2>(SGLayout{}.shape());
  sycl::range<2> local = {size(mma), 1};
  sycl::range<2> global = {
      local[0] * ceil_div(shape<0>(B), get<1>(mma.tile_mnk())),
      local[1] * ceil_div(shape<0>(A), get<0>(mma.tile_mnk()))};

  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;

  syclex::properties kernel_props{syclex::sub_group_size<16>,
                                  intelex::grf_size<256>};

  auto event =
      Q.parallel_for<GemmCuteName<TA, TB, layoutA, layoutB, q_group_size>>(
          sycl::nd_range<2>(global, local), kernel_props, [=](auto) {
            gemm_device<wg_n, sg_n, q_group_size>(A, B, S, C, mma);
          });

  EventManager::getInstance().addEvent(event);
}

// #define SHOW_DIFF 1

template <class...> class GemmVerifyKernelName;
template <class ATensor, class BTensor, class STensor, class CTensor>
bool gemm_verify(sycl::queue &Q,
                 ATensor const &A, // (M,K)
                 BTensor const &B, // (N,K)
                 STensor const &S, // (K/group_size, N)
                 CTensor const &C) // (M,N)
{
  int m = size<0>(A);
  int n = size<0>(B);
  int k = size<1>(A);
  int q_group_size = k / size<0>(S);

  auto ok = sycl::malloc_shared<bool>(1, Q);
  *ok = true;

  Q.parallel_for<GemmVerifyKernelName<ATensor, BTensor, STensor, CTensor>>(
       sycl::range<2>(m, n),
       [=](sycl::item<2> id) {
         int i = id[0], j = id[1];

         using AccType = float;
         using SignedAccType = ensure_signed_t<AccType>;

         auto c = AccType(0);
         for (int h = 0; h < k; h++)
           c += AccType(static_cast<typename ATensor::element_type>(
               AccType(static_cast<typename ATensor::element_type>(A(i, h))) *
               (AccType(static_cast<typename ATensor::element_type>(B(j, h))) *
                static_cast<AccType>(S(h / q_group_size, j)))));
         auto atol = AccType(1e-2f);
         auto rtol = std::abs(SignedAccType(AccType(C(i, j))) * AccType(1e-2f));
         auto tol = atol + rtol;
         if (std::abs(SignedAccType(c - AccType(C(i, j)))) > tol) {
#ifdef SHOW_DIFF
           printf("Error at (%d,%d): got %f, expected %f\n", i, j,
                  double(C(i, j)), double(c));
#endif
           *ok = false;
         }
       })
      .wait();

  bool read_ok = *ok;

  sycl::free(ok, Q);

  return read_ok;
}

template <typename TA, typename TB, typename TC, char layoutA = 'R',
          char layoutB = 'R', int q_group_size = 32>
void test_case(sycl::queue &Q, int m, int n, int k) {
  std::cout << type_str<TA>() << " (" << layoutA << ") x " << type_str<TB>()
            << " (" << layoutB << ") -> " << type_str<TC>()
            << "\tq_group_size=" << q_group_size << ": \t";
  if (k % q_group_size != 0) {
    std::cout << "Invalid run-time K for fixed q_group_size\n";
    return;
  }
  // Transpose B to match CuTe conventions
  constexpr char tlayoutB = layoutB ^ ('R' ^ 'C');
  using scaleType = cute::conditional_t<
      is_any_of_v<TB, float_e4m3_t, float_e5m2_t, float_e2m1_t>, float_ue8m0_t,
      TA>;
  // Prepare data:
  auto A = make_shared_usm_tensor<TA, layoutA>(Q, m, k);
  auto B = make_shared_usm_tensor<TB, tlayoutB>(Q, n, k);
  auto S = make_shared_usm_tensor<scaleType, 'R'>(Q, k / q_group_size, n);
  auto C = make_shared_usm_tensor<TC, 'R'>(Q, m, n);

  random_fill(A);
  random_fill(B);
  random_fill(S);
  zero_fill(C);

// #define SKIP_VERIFY 1
#ifndef SKIP_VERIFY
  auto A_ref = make_shared_usm_tensor<float, layoutA>(Q, m, k);
  auto B_ref = make_shared_usm_tensor<float, tlayoutB>(Q, n, k);

  copy(A, A_ref);
  copy(B, B_ref);
#endif

  subbyte_pack(A);
  subbyte_pack(B);

  // Test accuracy:
  gemm_cute<decltype(A), decltype(B), decltype(S), decltype(C), TA, TB, layoutA,
            layoutB, q_group_size>(Q, A, B, S, C);
  Q.wait_and_throw();

#ifdef SKIP_VERIFY
  const bool ok = true;
  std::cout << "verification skipped";
#else
  bool ok = true;
  if constexpr (!(is_same_v<TB, int4_t> && is_same_v<TB, int4_t> &&
                  (q_group_size == 128))) {
    ok = gemm_verify(Q, A_ref, B_ref, S, C);
    std::cout << (ok ? "passed" : "failed");
  } else {
    // int4 weights with BF16/half scales have poor accuracy with atol=1e-2,
    // rtol=1e-2 because we don't apply zero points
    std::cout << "atol=rtol=1e-2 has poor accuracy";
  }

#endif

  if (ok) {
    // Test performance:
    const int timing_iterations = 100;
    GPU_Clock timer;

    timer.start();
    for (int i = 0; i < timing_iterations; ++i)
      gemm_cute<decltype(A), decltype(B), decltype(S), decltype(C), TA, TB,
                layoutA, layoutB, q_group_size>(Q, A, B, S, C);
    Q.wait_and_throw();

    double avg = timer.seconds() / timing_iterations;
    double tops = (2.0 * m * n * k) * 1e-12;

    printf(", %4.3f TF/s", tops / avg, avg * 1000);
  }

  free_usm_tensor(A, Q);
  free_usm_tensor(B, Q);
  free_usm_tensor(S, Q);
  free_usm_tensor(C, Q);

#ifndef SKIP_VERIFY
  free_usm_tensor(A_ref, Q);
  free_usm_tensor(B_ref, Q);
#endif

  std::cout << '\n';

  // Pause for a short period of time to allow the GPU to cool.
  static bool first = true;
  if (first)
    first = false;
  else
    sleep(1);
}

int main(int argc, char **argv) {
  auto shift = [&] { return (argc-- > 0) ? *argv++ : nullptr; };

  auto parse_size = [&] {
    static constexpr int default_size = 4096;
    if (auto e = shift())
      return atoi(e);
    else
      return default_size;
  };

  (void)shift();

  auto m = parse_size();
  auto n = parse_size();
  auto k = parse_size();

  sycl::queue Q = compat::get_default_queue();

  // weights will be converted to either BF16 or FP16
  test_case<bfloat16_t, float_e4m3_t, bfloat16_t, 'R', 'R'>(Q, m, n, k);
  test_case<bfloat16_t, float_e4m3_t, bfloat16_t, 'R', 'C'>(Q, m, n, k);
  test_case<half_t, float_e4m3_t, half_t, 'R', 'R'>(Q, m, n, k);
  test_case<half_t, float_e4m3_t, half_t, 'R', 'C'>(Q, m, n, k);
  test_case<bfloat16_t, float_e5m2_t, bfloat16_t, 'R', 'R'>(Q, m, n, k);
  test_case<bfloat16_t, float_e5m2_t, bfloat16_t, 'R', 'C'>(Q, m, n, k);
  test_case<half_t, float_e5m2_t, half_t, 'R', 'R'>(Q, m, n, k);
  test_case<half_t, float_e5m2_t, half_t, 'R', 'C'>(Q, m, n, k);
  test_case<bfloat16_t, float_e2m1_t, bfloat16_t, 'R', 'R'>(Q, m, n, k);
  test_case<bfloat16_t, float_e2m1_t, bfloat16_t, 'R', 'C'>(Q, m, n, k);
  test_case<half_t, float_e2m1_t, half_t, 'R', 'R'>(Q, m, n, k);
  test_case<half_t, float_e2m1_t, half_t, 'R', 'C'>(Q, m, n, k);
}
