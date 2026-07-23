/***************************************************************************************************
 * Copyright (c) 2025 - 2025 Codeplay Software Ltd. All rights reserved.
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

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/fp8_to_fp16.h"
#include "cutlass/gemm/collective/xe_common_blockscaled_mxfp.hpp"
#include "cute/algorithm/functional.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {
using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  int Stages,
  int GroupSize,
  class KernelSchedule,
  class TileShape_,
  class ElementPairA_,
  class StridePairA_,
  class ElementPairB_,
  class StridePairB_,
  class TiledMma_,
  class GmemTiledCopyPairA_,
  class SmemLayoutAtomA_,
  class SmemCopyAtomA_,
  class TransformA_,
  class GmemTiledCopyPairB_,
  class SmemLayoutAtomB_,
  class SmemCopyAtomB_,
  class TransformB_>
struct CollectiveMma<
  MainloopIntelXeXMX16BlockScaledImpl<Stages, cute::Int<GroupSize>, KernelSchedule>,
    TileShape_,
    ElementPairA_,
    StridePairA_,
    ElementPairB_,
    StridePairB_,
    TiledMma_,
    GmemTiledCopyPairA_,
    SmemLayoutAtomA_,
    SmemCopyAtomA_,
    TransformA_,
    GmemTiledCopyPairB_,
    SmemLayoutAtomB_,
    SmemCopyAtomB_,
    TransformB_>
{
public:
  //
  // Type Aliases
  //
  using DispatchPolicy = MainloopIntelXeXMX16BlockScaledImpl<Stages, cute::Int<GroupSize>, KernelSchedule>;
  using WorkgroupTileShape = TileShape_;

  using GmemTiledCopyPairA = GmemTiledCopyPairA_;
  using GmemTiledCopyPairB = GmemTiledCopyPairB_;

  using TiledMma = TiledMma_;
  using ElementPairA = ElementPairA_;
  using ElementPairB = ElementPairB_;
  using ElementAMma = typename TiledMma::ValTypeA;
  using ElementBMma = typename TiledMma::ValTypeB;
  using StridePairA = StridePairA_;
  using StridePairB = StridePairB_;

  using ElementMMA = typename TiledMma_::ValTypeA;

    // A and B matrices
  using ElementA = remove_cvref_t<decltype(get<0>(ElementPairA{}))>;
  using StrideA  = cute::remove_pointer_t<remove_cvref_t<decltype(get<0>(StridePairA{}))>>;

  using ElementB = remove_cvref_t<decltype(get<0>(ElementPairB{}))>;
  using StrideB  = cute::remove_pointer_t<remove_cvref_t<decltype(get<0>(StridePairB{}))>>;

    // SFA and SFB
  using ElementSF = remove_cvref_t<decltype(get<1>(ElementPairA{}))>;
  using StrideScaleA = cute::remove_pointer_t<remove_cvref_t<decltype(get<1>(StridePairA{}))>>;
  using StrideScaleB = cute::remove_pointer_t<remove_cvref_t<decltype(get<1>(StridePairB{}))>>;

  using ElementScaleA = ElementSF;
  using ElementScaleB = ElementSF;

  using ElementAccumulator = typename TiledMma::ValTypeC;


  // using GmemTiledCopyA = void;
  // using GmemTiledCopyB = void;
  using GmemTiledCopyA = typename std::tuple_element<0, GmemTiledCopyPairA>::type;
  using GmemTiledCopyB = typename std::tuple_element<0, GmemTiledCopyPairB>::type;
  using GmemTiledCopyScaleA = typename std::tuple_element<1, GmemTiledCopyPairA>::type;
  using GmemTiledCopyScaleB = typename std::tuple_element<1, GmemTiledCopyPairB>::type;

  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;
  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using ArchTag = typename DispatchPolicy::ArchTag;
  using MmaType = typename TiledMma::ValTypeA; // ValTypeA and ValTypeB are always same and reflects MMA type on intel Xe

  static constexpr bool kSupportedElementA =
      cute::is_same_v<ElementA, float> ||
      cute::is_same_v<ElementA, cutlass::half_t> ||
      cute::is_same_v<ElementA, cutlass::float_e5m2_t> ||
      cute::is_same_v<ElementA, cutlass::float_e4m3_t> ||
      cute::is_same_v<ElementA, cutlass::float_e2m1_t>;

  static constexpr bool kSupportedElementB =
      cute::is_same_v<ElementB, cutlass::float_e5m2_t> ||
      cute::is_same_v<ElementB, cutlass::float_e4m3_t> ||
      cute::is_same_v<ElementB, cutlass::float_e2m1_t>;

   static constexpr bool kScaleALeftmostUnitStride = [] {
    if constexpr (cute::is_same_v<StrideScaleA, void>) {
      return false;
    } else {
      using LeftmostStrideA = remove_cvref_t<decltype(get<0>(StrideScaleA{}))>;
      return cute::is_same_v<LeftmostStrideA, _1>;
    }
  }();

  static constexpr bool kScaleBLeftmostUnitStride = [] {
    if constexpr (cute::is_same_v<StrideScaleB, void>) {
      return false;
    } else {
      using LeftmostStrideB = remove_cvref_t<decltype(get<0>(StrideScaleB{}))>;
      return cute::is_same_v<LeftmostStrideB, _1>;
    }
  }();

  static_assert(!(cute::is_same_v<ElementA, cutlass::float_e2m1_t> || 
                  cute::is_same_v<ElementB, cutlass::float_e2m1_t>) || (GroupSize == 32), 
                "Intel Xe blockscaled MMA only supports GroupSize=32 for e2m1 inputs.");

  static_assert(std::is_same_v<TransformA, cute::identity>, "Transformation for A is not currently supported on Intel PVC");
  static_assert(std::is_same_v<TransformB, cute::identity>, "Transformation for B is not currently supported on Intel PVC");
  static_assert(kSupportedElementA && kSupportedElementB,
                "Intel Xe blockscaled MMA only supports bf8 and hf8 operand types.");
  static_assert(kScaleALeftmostUnitStride,
                "Intel Xe blockscaled MMA requires scale A leftmost stride to be _1.");
  static_assert(kScaleBLeftmostUnitStride,
                "Intel Xe blockscaled MMA requires scale B leftmost stride to be _1.");

public:
  static constexpr int SubgroupSize = DispatchPolicy::SubgroupSize;

  using MmaAtomShape = typename TiledMma::AtomShape_MNK;
  static constexpr int ATOM_M = get<0>(MmaAtomShape{});
  static constexpr int ATOM_N = get<1>(MmaAtomShape{});

  static constexpr int BLK_M = get<0>(WorkgroupTileShape{});
  static constexpr int BLK_N = get<1>(WorkgroupTileShape{});
  static constexpr int BLK_K = get<2>(WorkgroupTileShape{});

  static constexpr int SG_NUMS_M = get<1>(typename TiledMma::ThrLayoutVMNK{}.shape());
  static constexpr int SG_NUMS_N = get<2>(typename TiledMma::ThrLayoutVMNK{}.shape());
  static constexpr int SG_NUMS_K = get<3>(typename TiledMma::ThrLayoutVMNK{}.shape());

  static constexpr int MMA_M = get<0>(typename TiledMma::Shape_MNK{});
  static constexpr int MMA_N = get<1>(typename TiledMma::Shape_MNK{});
  static constexpr int MMA_K = get<2>(typename TiledMma::Shape_MNK{});

  static constexpr int SG_M = ceil_div(BLK_M, SG_NUMS_M);
  static constexpr int SG_N = ceil_div(BLK_N, SG_NUMS_N);
  static constexpr int SG_K = ceil_div(BLK_K, SG_NUMS_K);
  using SubgroupTileShape = Shape<C<SG_M>, C<SG_N>, C<SG_K>>;

  static constexpr auto GroupK = GroupSize;

  static_assert(SG_K >= 32, "Intel Xe blockscaled MMA requires SG_K to be at least 32.");

  static constexpr auto Num_SGs = SG_NUMS_M * SG_NUMS_N * SG_NUMS_K;
  static constexpr uint32_t MaxThreadsPerBlock = size(TiledMma{});

  using CopyThreadShape = Shape<_1, Int<SubgroupSize>>;
  using CopyThreadShapeRev = decltype(cute::reverse(CopyThreadShape{}));

  // Helper to get tensor types
  template<class Element, class Stride>
  using TensorType = decltype(make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)),
                                        make_layout(make_shape(int{}, int{}, int{}), Stride{})));

  using DefScaleType = cutlass::float_ue8m0_t;
  using NonVoidElementScaleA = cute::conditional_t<cute::is_void_v<ElementScaleA>, DefScaleType, ElementScaleA>;
  using NonVoidElementScaleB = cute::conditional_t<cute::is_void_v<ElementScaleB>, DefScaleType, ElementScaleB>;

  static_assert(sizeof_bits_v<NonVoidElementScaleA> == 8 && sizeof_bits_v<NonVoidElementScaleB> == 8);

  // 2D block load requires surface width to be 4-byte aligned.
  // Physical scale extents must be multiples of ScaleAlignElems; callers may pad scale storage.
  static constexpr int ScaleAlignElems = cute::ceil_div(4, (int)(sizeof_bits_v<NonVoidElementScaleA> / 8));

  // Host side kernel arguments
  struct Arguments {
    ElementA const* ptr_A;
    StrideA dA;
    ElementB const* ptr_B;
    StrideB dB;
    ElementScaleA const* ptr_SA = nullptr;
    StrideScaleA dSA{};
    ElementScaleB const* ptr_SB = nullptr;
    StrideScaleB dSB{};
  };

  struct Params {
    TensorType<ElementA, StrideA> mA_mkl;
    TensorType<ElementB, StrideB> mB_nkl;
    TensorType<ElementScaleA, StrideScaleA> mAscale;
    TensorType<ElementScaleB, StrideScaleB> mBscale;
  };

  //
  // Methods
  //

  CollectiveMma() = default;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const &problem_shape,
                          Arguments const &args, void *workspace) {
    (void)workspace;

    auto [M, N, K, L] = problem_shape;

    auto mA_mkl =
        make_tensor(make_gmem_ptr(args.ptr_A), make_layout(make_shape(M, K, L), args.dA));
    auto mB_nkl =
        make_tensor(make_gmem_ptr(static_cast<ElementB const *>(args.ptr_B)), make_layout(make_shape(N, K, L), args.dB));

    auto scale_k = cute::ceil_div(K, GroupK);
    auto mScaleA = make_tensor(make_gmem_ptr(static_cast<ElementScaleA const *>(args.ptr_SA)),
                               make_layout(make_shape(M, scale_k, L), args.dSA));
    auto mScaleB = make_tensor(make_gmem_ptr(static_cast<ElementScaleB const *>(args.ptr_SB)),
                               make_layout(make_shape(N, scale_k, L), args.dSB));

    return Params{mA_mkl, mB_nkl, mScaleA, mScaleB};
  }

  template<class ProblemShape>
  static bool
  can_implement(
      ProblemShape problem_shapes,
      Arguments const& args) {
    constexpr int copy_alignment_bits = 128;
    constexpr int batch_alignment_bits = 512;
    auto problem_shape_MNKL = append<4>(problem_shapes, 1);
    auto [M,N,K,L] = problem_shape_MNKL;

    bool implementable = true;

    if constexpr (cute::is_same_v<ElementA, cutlass::float_e2m1_t> ||
                  cute::is_same_v<ElementB, cutlass::float_e2m1_t>) {
      if (GroupSize != 32) {
        CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Intel Xe blockscaled MMA only supports GroupSize=32 for e2m1 inputs.\n");
        implementable = false;
      }
    }

    constexpr int min_aligned_elements_A = copy_alignment_bits / sizeof_bits<ElementA>::value;
    implementable &= cutlass::detail::check_alignment<min_aligned_elements_A>(cute::make_shape(M,K,L), args.dA);
    constexpr int min_aligned_elements_B = copy_alignment_bits / sizeof_bits<ElementB>::value;
    implementable &= cutlass::detail::check_alignment<min_aligned_elements_B>(cute::make_shape(N,K,L), args.dB);

    if (L > 1) {
      constexpr int min_batch_aligned_elements_A = batch_alignment_bits / sizeof_bits<ElementA>::value;
      implementable &= get<2>(args.dA) % min_batch_aligned_elements_A == 0;
      constexpr int min_batch_aligned_elements_B = batch_alignment_bits / sizeof_bits<ElementB>::value;
      implementable &= get<2>(args.dB) % min_batch_aligned_elements_B == 0;
    }

    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for XE 2D copy.\n");
    }

    int scale_m_extent = static_cast<int>(M);
    int scale_n_extent = static_cast<int>(N);
    if constexpr (!cute::is_void_v<StrideScaleA>) {
      scale_m_extent = cute::max(scale_m_extent, static_cast<int>(get<1>(args.dSA)));
    }
    if constexpr (!cute::is_void_v<StrideScaleB>) {
      scale_n_extent = cute::max(scale_n_extent, static_cast<int>(get<1>(args.dSB)));
    }

    // 2D block load requires physical scale extents to be multiples of ScaleAlignElems
    // (4 for 8-bit scales). Logical M/N may be unaligned if scale storage is padded.
    if (scale_m_extent % ScaleAlignElems != 0 || scale_n_extent % ScaleAlignElems != 0) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: physical scale extents are not aligned for 2D block load. "
                         "Pad scale storage for MXFP scale factors.\n");
      implementable = false;
    }

    return implementable;
  }

  /// Perform a subgroup-scoped matrix multiply-accumulate
  template <class FrgTensorD,
    class TensorA,
    class TensorB,
    class FrgTensorC,
    class KTileIterator,
    class BlkCoord
  >
  CUTLASS_DEVICE void
  operator() (
      FrgTensorD &accum,
      TensorA gA,
      TensorB gB,
      FrgTensorC const &src_accum,
      KTileIterator k_tile_iter, int k_tile_count,
      BlkCoord const &blk_coord,
      int const &K_start,
      int thread_idx,
      Params const& mainloop) 
  {
    static_assert(is_rmem<FrgTensorD>::value, "D tensor must be rmem resident.");
    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");

    // Partition the copying of A and B tiles across the threads
    (void)blk_coord;
    auto batch_idx = get<3>(blk_coord);
    auto copy_a = get_block_2d_copy_A<GmemTiledCopyA>(TiledMma{}, mainloop.mA_mkl(_,_,batch_idx));
    auto copy_b = get_block_2d_copy_B<GmemTiledCopyB>(TiledMma{}, mainloop.mB_nkl(_,_,batch_idx));

    auto thr_copy_a = copy_a.get_slice(thread_idx);
    auto thr_copy_b = copy_b.get_slice(thread_idx);

    // Instantiate the MMA object and get thread slice
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_slice(thread_idx);

    /* Register fragments for MMA */
    auto tCrA = thr_mma.partition_sg_fragment_A(gA(_,_,0));
    auto tCrB = thr_mma.partition_sg_fragment_B(gB(_,_,0));

    /* Register fragments for copies */
    auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_,_,0));
    auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_,_,0));

    /* Partition global tensor (proxies) for copies */
    Tensor tAgA = thr_copy_a.partition_S(gA);
    Tensor tBgB = thr_copy_b.partition_S(gB);
    
    /* Create prefetch TiledCopy instances */
    auto prefetch_a = make_block_2d_prefetch(copy_a);
    auto prefetch_b = make_block_2d_prefetch(copy_b);
      
    auto thr_prefetch_A = prefetch_a.get_slice(thread_idx);
    auto thr_prefetch_B = prefetch_b.get_slice(thread_idx);

    /* Partition global tensor (proxies) for prefetch */
    auto pAgA = thr_prefetch_A.partition_S(gA);
    auto pBgB = thr_prefetch_B.partition_S(gB);

    using GemmIterM = Int<decltype(size<1>(tCrA.shape()))::value>;
    using GemmIterN = Int<decltype(size<1>(tCrB.shape()))::value>;
    using GemmIterK = Int<decltype(size<2>(tCrB.shape()))::value>;

    auto [m_idx, n_idx, k_idx, l_idx] = blk_coord;
    const int m_coord = m_idx * BLK_M + (get_sub_group_id() / SG_NUMS_N) * SG_M;
    const int n_coord = n_idx * BLK_N + (get_sub_group_id() % SG_NUMS_N) * SG_N;
    const int l_coord = l_idx;

    const int k_start_idx = crd2idx((*k_tile_iter), make_shape(K_start));

    constexpr int k_reload_factor = cute::max(GroupK / BLK_K, 1);

    auto [tiled_copy_scaleA, copy_iter_scaleA, fragment_scaleA] = make_scaled_copy<GmemTiledCopyScaleA, NonVoidElementScaleA,
                                              SG_M, SG_K, GroupK>(mainloop.mAscale, m_coord, l_coord, k_tile_count);
    auto [tiled_copy_scaleB, copy_iter_scaleB, fragment_scaleB] = make_scaled_copy<GmemTiledCopyScaleB, NonVoidElementScaleB,
                                              SG_N, SG_K, GroupK>(mainloop.mBscale, n_coord, l_coord, k_tile_count);
    auto [scale_m_offsets, scale_n_offsets, scale_ak_offsets, scale_bk_offsets] = make_scaled_offsets<
                                                  GemmIterM::value, GemmIterN::value, GemmIterK::value, MMA_K, GroupK,
                                                  typename decltype(tiled_copy_scaleA)::BlockShape,
                                                  typename decltype(tiled_copy_scaleB)::BlockShape>();
    auto [tiled_prefetch_scaleA, prefetch_iter_scaleA] = make_scaled_prefetch<decltype(tiled_copy_scaleA),
                                                           SG_M, SG_K, GroupK>(tiled_copy_scaleA, m_coord, l_coord, k_tile_count);
    auto [tiled_prefetch_scaleB, prefetch_iter_scaleB] = make_scaled_prefetch<decltype(tiled_copy_scaleB),
                                                           SG_N, SG_K, GroupK>(tiled_copy_scaleB, n_coord, l_coord, k_tile_count);

    using scaleA_vec_t = intel::vector_t<ElementScaleA, decltype(size(fragment_scaleA))::value>;
    using scaleB_vec_t = intel::vector_t<ElementScaleB, decltype(size(fragment_scaleB))::value>;

    int prefetch_k = k_start_idx;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < DispatchPolicy::Stages; i++, prefetch_k++) {
        prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
        prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
        prefetch(tiled_prefetch_scaleA, prefetch_iter_scaleA(_, _, _, prefetch_k / k_reload_factor));
        prefetch(tiled_prefetch_scaleB, prefetch_iter_scaleB(_, _, _, prefetch_k / k_reload_factor));
    }

    for (int k_tile = k_start_idx; k_tile < k_tile_count + k_start_idx; k_tile++, prefetch_k++) {
      copy(copy_a, tAgA(_,_,_,k_tile), tArA);
      copy(copy_b, tBgB(_,_,_,k_tile), tBrB);

      copy(tiled_copy_scaleA, copy_iter_scaleA(_, _, _, k_tile / k_reload_factor), fragment_scaleA);
      copy(tiled_copy_scaleB, copy_iter_scaleB(_, _, _, k_tile / k_reload_factor), fragment_scaleB);

      prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
      prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
      prefetch(tiled_prefetch_scaleA, prefetch_iter_scaleA(_, _, _, prefetch_k / k_reload_factor));
      prefetch(tiled_prefetch_scaleB, prefetch_iter_scaleB(_, _, _, prefetch_k / k_reload_factor));
      reorder(tArA, tCrA);
      reorder(tBrB, tCrB);

      Tensor scaleA = make_tensor(recast<scaleA_vec_t>(fragment_scaleA).data(), make_layout(Shape<_1, GemmIterM, _1>{}, Stride<_1, _0, _0>{}));
      Tensor scaleB = make_tensor(recast<scaleB_vec_t>(fragment_scaleB).data(), make_layout(Shape<_1, GemmIterN, _1>{}, Stride<_1, _0, _0>{}));

      cute::gemm(tiled_mma, make_zip_tensor(tCrA, scaleA, scale_m_offsets, scale_ak_offsets),
                make_zip_tensor(tCrB, scaleB, scale_n_offsets, scale_bk_offsets), accum);
    }
  }
};

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
