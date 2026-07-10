/***************************************************************************************************

 * Copyright (c) 2026 Intel Corporation, All rights reserved.
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
  \brief Visitor tree operations for the Xe epilogue
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cute/arch/copy_xe_2d.hpp"


namespace cutlass::epilogue::fusion {

namespace detail {

enum class XeReductionAxis { Column, Row };

template <XeReductionAxis Axis>
struct XeReductionAxisTraits;

template <>
struct XeReductionAxisTraits<XeReductionAxis::Column> {
  template <class MShape, class NShape>
  CUTLASS_HOST_DEVICE static auto axis_extent(MShape const& M, NShape const&) {
    return M;
  }
  template <class TileM, class TileN>
  CUTLASS_HOST_DEVICE static auto tile_extent(TileM const& tile_M, TileN const&) {
    return tile_M;
  }
};

template <>
struct XeReductionAxisTraits<XeReductionAxis::Row> {
  template <class MShape, class NShape>
  CUTLASS_HOST_DEVICE static auto axis_extent(MShape const&, NShape const& N) {
    return N;
  }
  template <class TileM, class TileN>
  CUTLASS_HOST_DEVICE static auto tile_extent(TileM const&, TileN const& tile_N) {
    return tile_N;
  }
};

struct XeReductionWorkspaceLayout {
  size_t reduction_buffer_bytes = 0;
  size_t tile_counter_count = 0;
};

template <class T>
CUTLASS_HOST_DEVICE constexpr size_t
xe_to_size_t(T const& value) {
  return static_cast<size_t>(static_cast<long long>(value));
}

template <XeReductionAxis Axis, class CtaTileShapeMNK>
struct XeReductionWorkspaceHelper {
  template <class ElementCompute, class ProblemShape>
  CUTLASS_HOST_DEVICE static XeReductionWorkspaceLayout
  layout(ProblemShape const& problem_shape) {
    auto problem_shape_mnkl = append<4>(problem_shape, 1);
    auto [M, N, K, L] = problem_shape_mnkl;
    auto [tile_M, tile_N, tile_K] = CtaTileShapeMNK{};

    auto ceil_tiles = ceil_div(make_shape(M, N, L), make_shape(tile_M, tile_N));
    using AxisTraits = XeReductionAxisTraits<Axis>;
    auto reduction_axis = AxisTraits::axis_extent(M, N);
    auto tile_axis = AxisTraits::tile_extent(tile_M, tile_N);
    auto tile_counter_tiles = cute::ceil_div(reduction_axis, tile_axis);

    XeReductionWorkspaceLayout workspace{};
    workspace.reduction_buffer_bytes =
      xe_to_size_t(product(ceil_tiles)) * xe_to_size_t(tile_axis) * sizeof(ElementCompute);
    workspace.tile_counter_count = xe_to_size_t(tile_counter_tiles);
    return workspace;
  }
};

template <class SyncFn>
CUTLASS_DEVICE bool
xe_signal_final_reduction(
    int* tile_counters,
    int axis_index,
    int total_tiles,
    int thread_idx,
    void* shared_storage,
    SyncFn const& sync_fn) {
  int* prev_tile_count = reinterpret_cast<int*>(shared_storage);
  if (thread_idx == 0) {
    *prev_tile_count = atomicAdd(&tile_counters[axis_index], 1);
  }
  sync_fn();
  bool do_final = *prev_tile_count == total_tiles - 1;
  sync_fn();
  return do_final;
}

} // namespace detail

/***************************************************************************************************
 *                                    ----------------
 *                                       XeAuxStore 
 *                                    ----------------
 * Execution Paths:
 *   1. Scalar/Vectorized Copy Path [DEFAULT] (when CopyOpR2G_ == void AND UseBlock2DCopy == false):
 *      - Uses copy_if with predicate-based bounds checking
 *      - Automatically vectorizes using max_common_layout when Alignment allows (V > 1)
 *      - Falls back to scalar copy if vectorization constraints not met
 *
 *   2. Block 2D Copy Path (when CopyOpR2G_ != void OR UseBlock2DCopy == true):
 *      - Uses Intel Xe block_2d_store operations for optimal bandwidth
 *      - Tile sizes auto-deduced from MMATile using gcd if CopyOpR2G_ is void
 *      - Recommended for aligned, contiguous memory access patterns
 *
 * Template Parameters:
 *   Element         Element type of auxiliary tensor (e.g., float, bfloat16_t)
 *   StrideMNL       Stride type for 3D tensor (M, N, L) layout
 *   CopyOpR2G_      Copy operation for register-to-global transfer
 *                           - void (default): Uses scalar/vectorized copy_if path
 *                           - XE_STORE_2D<...>: Uses Block 2D copy path with explicit tile sizes
 *   Alignment       Maximum vectorization width in elements (default: 128 bits / sizeof(Element))
 *                           - Controls vector size V = min(Alignment, max_common_layout_size)
 *   EnableNullptr   Allow nullptr for aux tensor (uses null_default value if true)
 *   UseBlock2DCopy  Enable Block 2D copy with auto-deduced tile sizes when CopyOpR2G_ == void
 *                           - true: Auto-deduce tile sizes and use block_2d_store
 *                           - false (default): Use scalar/vectorized copy_if path
 *
 **************************************************************************************************/

template <
  class Element,
  class StrideMNL,
  class CopyOpR2G_ = void,
  int Alignment = 128 / sizeof_bits_v<Element>,
  bool EnableNullptr = true,
  bool UseBlock2DCopy = false
>
struct XeAuxStore {
  using SharedStorage = Element;

  struct Arguments {
    Element* ptr_aux = nullptr;
    Element null_default = Element(0);
    StrideMNL dAux = {};
  };

  static constexpr int CopyBits = cute::min(sizeof_bits_v<Element>, 64);

  // Define 3D tensor type for aux tensor (M, N, L)
  using TensorAux = decltype(make_tensor(make_gmem_ptr(static_cast<Element*>(nullptr)),
                                         Layout<Shape<int,int,int>, StrideMNL>{}));

  struct Params {
    TensorAux mAux;            // 3D tensor (M, N, L)
    Element null_default = Element(0);
    bool use_default = false;
  };

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    // Optionally append 1s until problem shape is rank-4 in case its is only rank-3 (MNK)
    auto problem_shape_mnkl = append<4>(problem_shape, 1);
    auto [M, N, K, L] = problem_shape_mnkl;

    // Create 3D tensor with shape (M, N, L) and stride (stride_M, stride_N, stride_L)
    auto shape_MNL = make_shape(int(M), int(N), int(L));
    auto mAux = make_tensor(make_gmem_ptr(args.ptr_aux),
                           make_layout(shape_MNL, args.dAux));

    bool use_default = false;
    if constexpr (EnableNullptr) {
      use_default = args.ptr_aux == nullptr;
    }

    return Params{mAux, args.null_default, use_default};
  }

  template <class ProblemShape>
  static bool
  can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    return true;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream,
    CudaHostAdapter* cuda_adapter = nullptr) {
    return cutlass::Status::kSuccess;
  }

  CUTLASS_HOST_DEVICE
  XeAuxStore() { }

  CUTLASS_HOST_DEVICE
  XeAuxStore(Params const& params, SharedStorage const&) : params_ptr(&params) { }

  Params const* params_ptr;

  CUTLASS_DEVICE bool
  is_producer_load_needed() const {
    return false;
  }

  CUTLASS_DEVICE bool
  is_C_load_needed() const {
    return false;
  }

  CUTLASS_DEVICE bool
  is_zero() const {
    return (params_ptr->use_default && params_ptr->null_default == Element(0));
  }

  template <class... Args>
  CUTLASS_DEVICE auto
  get_producer_load_callbacks(ProducerLoadArgs<Args...> const&) {
    return EmptyProducerLoadCallbacks{};
  }

  // Scalar/vectorized path callback
  template <class GTensor, class CTensor, class RTensor>
  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
    GTensor tCgAux_epi;      // Gmem tensor ((mma_v,mma_m,mma_n),epi_m,epi_n)
    CTensor gAux_coord;       // Coordinate tensor for bounds checking
    RTensor tC_rAux;          // Register fragment (mma_v,mma_m,mma_n)
    Params const* params_ptr;

    CUTLASS_DEVICE
    ConsumerStoreCallbacks(GTensor tCgAux_epi, CTensor gAux_coord, RTensor&& tC_rAux, Params const* params_ptr)
      : tCgAux_epi(tCgAux_epi), 
        gAux_coord(gAux_coord),
        tC_rAux(cute::forward<RTensor>(tC_rAux)), 
        params_ptr(params_ptr) { }

    CUTLASS_DEVICE void
    end_loop(int epi_m, int epi_n) {
        if constexpr (EnableNullptr) {
            if (params_ptr->use_default) {
                return; // Skip store if pointer is nullptr
            }
        }

        auto tCgAux_mn = tCgAux_epi(_,epi_m,epi_n);
        auto coord_mn = gAux_coord(_,_,epi_m,epi_n);
        auto [M, N, L] = params_ptr->mAux.shape();
        auto problem_shape_mn = make_coord(M, N);
        
        constexpr auto MCL = decltype(max_common_layout(tCgAux_mn, tC_rAux)){};
        constexpr int V = cute::min(Alignment, size(MCL));
        if constexpr (V > 1) {
          // vectorized store
          using VecType = uint_bit_t<V * sizeof_bits_v<Element>>;
          Tensor tCgAux_vec = recast<VecType>(coalesce(tCgAux_mn));
          Tensor tCrAux_vec = recast<VecType>(coalesce(tC_rAux));
          Tensor coord_vec = tensor<1>(zipped_divide(coord_mn, MCL.compose(Int<V>{})));
          
          auto pred_fn = [&](auto const&... coords) CUTLASS_LAMBDA_FUNC_INLINE {
            return elem_less(coord_vec(coords...), problem_shape_mn);
          };
          copy_if(pred_fn, tCrAux_vec, tCgAux_vec);
        } else {
          // scalar store
          auto pred_fn = [&](auto const&... coords) CUTLASS_LAMBDA_FUNC_INLINE {
            return elem_less(coord_mn(coords...), problem_shape_mn);
          };
          copy_if(pred_fn, tC_rAux, tCgAux_mn);
        }
    }

    template <typename ElementAccumulator, typename ElementInput, int FragmentSize>
    CUTLASS_DEVICE Array<ElementInput, FragmentSize>
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, 
      int epi_n, Array<ElementInput, FragmentSize> const& frg_input) {
      for(int i = 0; i < FragmentSize; ++i) {
        tC_rAux(epi_v * FragmentSize + i) = static_cast<Element>(frg_input.data()[i]);
      }
      return frg_input;
    }
  };

  // Block 2D path callback
  template <class CTensor, class RTensor, class AuxCopy>
  struct ConsumerStoreCallbacksBlock2D : EmptyConsumerStoreCallbacks {
    CTensor rw_coord;
    AuxCopy xe_copy_aux;
    RTensor tC_rAux;
    Params const* params_ptr;
    
    CUTLASS_DEVICE
    ConsumerStoreCallbacksBlock2D(CTensor rw_coord_, AuxCopy xe_copy_aux_, 
                                  RTensor&& tC_rAux_, Params const* params_ptr_)
      : rw_coord(rw_coord_), xe_copy_aux(xe_copy_aux_), tC_rAux(cute::move(tC_rAux_)), params_ptr(params_ptr_) { }
    
    CUTLASS_DEVICE void
    end_loop(int epi_m, int epi_n) {
      if constexpr (EnableNullptr) {
        if (params_ptr->use_default) {
          return;
        }
      }
      copy(xe_copy_aux, tC_rAux, rw_coord(_, _, _, epi_m, epi_n));
    }
    
    template <typename ElementAccumulator, typename ElementInput, int FragmentSize>
    CUTLASS_DEVICE Array<ElementInput, FragmentSize>
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, 
          int epi_n, Array<ElementInput, FragmentSize> const& frg_input) {
      for(int i = 0; i < FragmentSize; ++i) {
        tC_rAux(epi_v * FragmentSize + i) = static_cast<Element>(frg_input.data()[i]);
      }
      return frg_input;
    }
  };

  template <
    bool ReferenceSrc,
    class... Args
  >
  CUTLASS_DEVICE auto
  get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const& args) {
    auto [M, N, K, L] = args.problem_shape_mnkl;
    auto [m_coord, n_coord, k_coord, l_coord] = args.tile_coord_mnkl;

    auto mAux_batch = params_ptr->mAux(_,_,l_coord);

    auto thr_mma = args.tiled_mma.get_slice(args.thread_idx);

    // Calculate MMA tiles per epilogue tile
    using MMATile = decltype(take<0,2>(typename cute::remove_cvref_t<decltype(args.tiled_mma)>::AtomShape_MNK{}));
    
    auto mma_per_epi = shape_div(args.epi_tile, MMATile{});

    if constexpr (!cute::is_void_v<CopyOpR2G_> || UseBlock2DCopy) {
      // Block 2D copy path (explicit CopyOp or auto-deduced)
      using ActualCopyOpR2G = cute::conditional_t<
        cute::is_void_v<CopyOpR2G_>,
        XE_STORE_2D<CopyBits, 
                   cute::gcd(8, get<0>(MMATile{})), 
                   cute::gcd(512 / CopyBits, get<1>(MMATile{}))>,
        CopyOpR2G_
      >;
      
      auto tCDgCD = thr_mma.partition_C(args.cD);
      auto sg_v_coord = prepend(flat_divide(remove<0>(tCDgCD.layout()), mma_per_epi),
                                get<0>(tCDgCD.layout()));
      auto gAux_epi_layout = append(append(make_identity_layout(args.epi_tile),
                                          get<3>(sg_v_coord)), get<4>(sg_v_coord));
      auto gAux_epi = make_tensor(tCDgCD.data(), gAux_epi_layout);
      
      auto xe_copy_aux = make_block_2d_copy(ActualCopyOpR2G{}, mAux_batch);
      auto thr_copy_aux = xe_copy_aux.get_slice(args.thread_idx % intel::sg_size);
      auto tCgAux = thr_copy_aux.partition_D(gAux_epi);  // (atom_v,atom_m,atom_n,epi_m,epi_n)
      auto trAux = thr_copy_aux.partition_sg_fragment_S(gAux_epi(_,_,0,0));  // (atom_v,atom_m,atom_n)
      
      return ConsumerStoreCallbacksBlock2D<decltype(tCgAux), decltype(trAux), decltype(xe_copy_aux)>(
        tCgAux, xe_copy_aux, cute::move(trAux), params_ptr);
    } else {
      // Scalar/vectorized copy path (default)
      auto tCDgCD = thr_mma.partition_C(args.cD);
      auto sg_v_coord = prepend(flat_divide(remove<0>(tCDgCD.layout()), mma_per_epi),
                                get<0>(tCDgCD.layout()));
      auto gAux_epi_layout = append(append(make_identity_layout(args.epi_tile),
                                          get<3>(sg_v_coord)), get<4>(sg_v_coord));
      auto gAux_epi = make_tensor(tCDgCD.data(), gAux_epi_layout);
      
      auto gAux_data = local_tile(mAux_batch, select<0,1>(args.tile_shape_mnk), make_coord(m_coord, n_coord));
      auto tCgAux = thr_mma.partition_C(gAux_data);  // (mma_v,mma_m,mma_n)
      auto tiled_tCgAux = group<0,3>(prepend(flat_divide(remove<0>(tCgAux.layout()), mma_per_epi),
                                             get<0>(tCgAux.layout())));
      auto tCgAux_epi = make_tensor(tCgAux.data(), tiled_tCgAux);  // ((mma_v,mma_m,mma_n),epi_m,epi_n)
      auto trAux = make_tensor_like(tCgAux);  // (mma_v,mma_m,mma_n)

      return ConsumerStoreCallbacks<decltype(tCgAux_epi), decltype(gAux_epi), decltype(trAux)>(
        tCgAux_epi, gAux_epi, cute::move(trAux), params_ptr);
    }
  }
};

/***************************************************************************************************
 *                                   -----------------
 *                                      XeAuxLoad
 *                                   -----------------
 * Execution Paths:
 *   1. Scalar/Vectorized Copy Path [DEFAULT] (when CopyOpG2R_ == void AND UseBlock2DCopy == false):
 *      - Uses copy_if with predicate-based bounds checking
 *      - Automatically vectorizes using max_common_layout when Alignment allows (V > 1)
 *      - Falls back to scalar copy if vectorization constraints not met
 *      - Zero-fills out-of-bounds elements for safe computation
 *
 *   2. Block 2D Copy Path (when CopyOpG2R_ != void OR UseBlock2DCopy == true):
 *      - Uses Intel Xe block_2d_load operations for optimal bandwidth
 *      - Tile sizes auto-deduced from MMATile using gcd if CopyOpG2R_ is void
 *      - Recommended for aligned, contiguous memory access patterns
 *
 * Template Parameters:
 *   Element         Element type of auxiliary tensor (e.g., float, bfloat16_t)
 *   StrideMNL       Stride type for 3D tensor (M, N, L) layout
 *   CopyOpG2R_      Copy operation for global-to-register transfer
 *                           - void (default): Uses scalar/vectorized copy_if path
 *                           - XE_LOAD_2D<...>: Uses Block 2D copy path with explicit tile sizes
 *   Alignment       Maximum vectorization width in elements (default: 128 bits / sizeof(Element))
 *                           - Controls vector size V = min(Alignment, max_common_layout_size)
 *   EnableNullptr   Allow nullptr for aux tensor (uses null_default value if true)
 *   UseBlock2DCopy  Enable Block 2D copy with auto-deduced tile sizes when CopyOpG2R_ == void
 *                           - true: Auto-deduce tile sizes and use block_2d_load
 *                           - false (default): Use scalar/vectorized copy_if path
 *
 **************************************************************************************************/

template <
  class Element,
  class StrideMNL,
  class CopyOpG2R_ = void,
  int Alignment = 128 / sizeof_bits_v<Element>,
  bool EnableNullptr = true,
  bool UseBlock2DCopy = false
>
struct XeAuxLoad {
  using SharedStorage = Element;

  struct Arguments {
    Element const* ptr_aux = nullptr;
    Element null_default = Element(0);
    StrideMNL dAux = {};
  };

  static constexpr int CopyBits = cute::min(sizeof_bits_v<Element>, 64);

  // Define 3D tensor type for aux tensor (M, N, L)
  using TensorAux = decltype(make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)),
                                         Layout<Shape<int,int,int>, StrideMNL>{}));

  struct Params {
    TensorAux mAux;            // 3D tensor (M, N, L)
    Element null_default = Element(0);
    bool use_default = false;
  };

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    // Optionally append 1s until problem shape is rank-4 in case its is only rank-3 (MNK)
    auto problem_shape_mnkl = append<4>(problem_shape, 1);
    auto [M, N, K, L] = problem_shape_mnkl;

    // Create 3D tensor with shape (M, N, L) and stride (stride_M, stride_N, stride_L)
    auto shape_MNL = make_shape(int(M), int(N), int(L));
    auto mAux = make_tensor(make_gmem_ptr(args.ptr_aux),
                           make_layout(shape_MNL, args.dAux));

    bool use_default = false;
    if constexpr (EnableNullptr) {
      use_default = args.ptr_aux == nullptr;
    }

    return Params{mAux, args.null_default, use_default};
  }

  template <class ProblemShape>
  static bool
  can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    return true;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream,
    CudaHostAdapter* cuda_adapter = nullptr) {
    return cutlass::Status::kSuccess;
  }

  CUTLASS_HOST_DEVICE
  XeAuxLoad() { }

  CUTLASS_HOST_DEVICE
  XeAuxLoad(Params const& params, SharedStorage const&) : params_ptr(&params) { }

  Params const* params_ptr;

  CUTLASS_DEVICE bool
  is_producer_load_needed() const {
    return false;
  }

  CUTLASS_DEVICE bool
  is_C_load_needed() const {
    return false;
  }

  CUTLASS_DEVICE bool
  is_zero() const {
    return (params_ptr->use_default && params_ptr->null_default == Element(0));
  }

  template <class... Args>
  CUTLASS_DEVICE auto
  get_producer_load_callbacks(ProducerLoadArgs<Args...> const&) {
    return EmptyProducerLoadCallbacks{};
  }

  // Scalar/vectorized path callback
  template <class GTensor, class CTensor, class RTensor>
  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
    GTensor tCgAux_epi;      // Gmem tensor ((mma_v,mma_m,mma_n),epi_m,epi_n)
    CTensor gAux_coord;       // Coordinate tensor for bounds checking
    RTensor tC_rAux;          // Register fragment (mma_v,mma_m,mma_n)
    Params const* params_ptr;

    CUTLASS_DEVICE
    ConsumerStoreCallbacks(GTensor tCgAux_epi, CTensor gAux_coord, RTensor&& tC_rAux, Params const* params_ptr)
      : tCgAux_epi(tCgAux_epi), 
        gAux_coord(gAux_coord),
        tC_rAux(cute::forward<RTensor>(tC_rAux)), 
        params_ptr(params_ptr) { }

    // Load aux data for epilogue tile (epi_m, epi_n)
    CUTLASS_DEVICE void
    previsit(int epi_m, int epi_n, int load_iteration, bool is_producer_load_needed) {
       if constexpr (EnableNullptr) {
         if (params_ptr->use_default) {
           fill(tC_rAux, params_ptr->null_default);
           return;
         }
       }

       // Load from global memory using vectorized copy_if with bounds checking (matches XeRowBroadcastLegacy pattern)
       auto tCgAux_mn = tCgAux_epi(_,epi_m,epi_n);
       auto coord_mn = gAux_coord(_,_,epi_m,epi_n);
       auto [M, N, L] = params_ptr->mAux.shape();
       auto problem_shape_mn = make_coord(M, N);
       
       // Zero-fill out-of-bounds elements first
       clear(tC_rAux);
       
       constexpr auto MCL = decltype(max_common_layout(tCgAux_mn, tC_rAux)){};
       constexpr int V = cute::min(Alignment, size(MCL));
       if constexpr (V > 1) {
        // vectorized load
         using VecType = uint_bit_t<V * sizeof_bits_v<Element>>;
         Tensor tCgAux_vec = recast<VecType>(coalesce(tCgAux_mn));
         Tensor tCrAux_vec = recast<VecType>(coalesce(tC_rAux));
         Tensor coord_vec = tensor<1>(zipped_divide(coord_mn, MCL.compose(Int<V>{})));
         
         auto pred_fn = [&](auto const&... coords) CUTLASS_LAMBDA_FUNC_INLINE {
           return elem_less(coord_vec(coords...), problem_shape_mn);
         };
         copy_if(pred_fn, tCgAux_vec, tCrAux_vec);
       } else {
        // scalar load
         auto pred_fn = [&](auto const&... coords) CUTLASS_LAMBDA_FUNC_INLINE {
           return elem_less(coord_mn(coords...), problem_shape_mn);
         };
         copy_if(pred_fn, tCgAux_mn, tC_rAux);
       }
    }

    // Return loaded aux values for epilogue computation
    template <typename ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE Array<Element, FragmentSize>
    visit(Array<ElementAccumulator, FragmentSize> const&, int epi_v, int epi_m, int epi_n) {
       Array<Element, FragmentSize> frg_aux;
       CUTLASS_PRAGMA_UNROLL
       for (int i = 0; i < FragmentSize; ++i) {
         frg_aux[i] = tC_rAux(epi_v * FragmentSize + i);
       }
       return frg_aux;
    }
  };

  // Block 2D path callback
  template <class CTensor, class RTensor, class AuxCopy>
  struct ConsumerStoreCallbacksBlock2D : EmptyConsumerStoreCallbacks {
    CTensor rw_coord;
    AuxCopy xe_copy_aux;
    RTensor tC_rAux;
    Params const* params_ptr;
    
    CUTLASS_DEVICE
    ConsumerStoreCallbacksBlock2D(CTensor rw_coord_, AuxCopy xe_copy_aux_, 
                                  RTensor&& tC_rAux_, Params const* params_ptr_)
      : rw_coord(rw_coord_), xe_copy_aux(xe_copy_aux_), tC_rAux(cute::move(tC_rAux_)), params_ptr(params_ptr_) { }
    
    CUTLASS_DEVICE void
    previsit(int epi_m, int epi_n, int load_iteration, bool is_producer_load_needed) {
      if constexpr (EnableNullptr) {
        if (params_ptr->use_default) {
          fill(tC_rAux, params_ptr->null_default);
          return;
        }
      }
      copy(xe_copy_aux, rw_coord(_,_,_,epi_m,epi_n), tC_rAux);
    }
    
    template <typename ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE Array<Element, FragmentSize>
    visit(Array<ElementAccumulator, FragmentSize> const&, int epi_v, int epi_m, int epi_n) {
      Tensor tC_rAux_frg = recast<Array<Element, FragmentSize>>(coalesce(tC_rAux.tensor()));  // (EPI_V)
      return tC_rAux_frg(epi_v);
    }
  };

  template <
    bool ReferenceSrc,
    class... Args
  >
  CUTLASS_DEVICE auto
  get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const& args) {
    auto [M, N, K, L] = args.problem_shape_mnkl;
    auto [m_coord, n_coord, k_coord, l_coord] = args.tile_coord_mnkl;

    auto mAux_batch = params_ptr->mAux(_,_,l_coord);

    auto thr_mma = args.tiled_mma.get_slice(args.thread_idx);

    // Calculate MMA tiles per epilogue tile
    using MMATile = decltype(take<0,2>(typename cute::remove_cvref_t<decltype(args.tiled_mma)>::AtomShape_MNK{}));
    
    auto mma_per_epi = shape_div(args.epi_tile, MMATile{});

    if constexpr (!cute::is_void_v<CopyOpG2R_> || UseBlock2DCopy) {
      // Block 2D copy path (explicit CopyOp or auto-deduced)
      using ActualCopyOpG2R = cute::conditional_t<
        cute::is_void_v<CopyOpG2R_>,
        XE_LOAD_2D<CopyBits, 
                   cute::gcd(8, get<0>(MMATile{})), 
                   cute::gcd(512 / CopyBits, get<1>(MMATile{}))>,
        CopyOpG2R_
      >;
      
      auto tCDgCD = thr_mma.partition_C(args.cD);
      auto sg_v_coord = prepend(flat_divide(remove<0>(tCDgCD.layout()), mma_per_epi),
                                get<0>(tCDgCD.layout()));
      auto gAux_epi_layout = append(append(make_identity_layout(args.epi_tile),
                                          get<3>(sg_v_coord)), get<4>(sg_v_coord));
      auto gAux_epi = make_tensor(tCDgCD.data(), gAux_epi_layout);
      
      auto xe_copy_aux = make_block_2d_copy(ActualCopyOpG2R{}, mAux_batch);
      auto thr_copy_aux = xe_copy_aux.get_slice(args.thread_idx % intel::sg_size);
      auto tCgAux = thr_copy_aux.partition_S(gAux_epi);  // (atom_v,atom_m,atom_n,epi_m,epi_n)
      auto trAux = thr_copy_aux.partition_sg_fragment_D(gAux_epi(_,_,0,0));  // (atom_v,atom_m,atom_n)
      
      return ConsumerStoreCallbacksBlock2D<decltype(tCgAux), decltype(trAux), decltype(xe_copy_aux)>(
        tCgAux, xe_copy_aux, cute::move(trAux), params_ptr);
    } else {
      // Scalar/vectorized copy path (default)
      auto tCDgCD = thr_mma.partition_C(args.cD);
      auto sg_v_coord = prepend(flat_divide(remove<0>(tCDgCD.layout()), mma_per_epi),
                                get<0>(tCDgCD.layout()));
      auto gAux_epi_layout = append(append(make_identity_layout(args.epi_tile),
                                          get<3>(sg_v_coord)), get<4>(sg_v_coord));
      auto gAux_epi = make_tensor(tCDgCD.data(), gAux_epi_layout);
      
      auto gAux_data = local_tile(mAux_batch, select<0,1>(args.tile_shape_mnk), make_coord(m_coord, n_coord));
      auto tCgAux = thr_mma.partition_C(gAux_data);  // (mma_v,mma_m,mma_n)
      auto tiled_tCgAux = group<0,3>(prepend(flat_divide(remove<0>(tCgAux.layout()), mma_per_epi),
                                             get<0>(tCgAux.layout())));
      auto tCgAux_epi = make_tensor(tCgAux.data(), tiled_tCgAux);  // ((mma_v,mma_m,mma_n),epi_m,epi_n)
      auto trAux = make_tensor_like(tCgAux);  // (mma_v,mma_m,mma_n)

      return ConsumerStoreCallbacks<decltype(tCgAux_epi), decltype(gAux_epi), decltype(trAux)>(
        tCgAux_epi, gAux_epi, cute::move(trAux), params_ptr);
    }
  }
};

// Row Broadcast
template<
  int Stages,
  class CtaTileShapeMNK,
  class ElementInput_,
  class ElementCompute = cute::remove_pointer_t<ElementInput_>,
  class StrideMNL_ = Stride<_0,_1,_0>,
  int Alignment = 128 / sizeof_bits_v<cute::remove_pointer_t<ElementInput_>>,
  bool EnableNullptr = true // Fallback scalar broadcast for nullptr params
>
struct XeRowBroadcast {
  using StrideMNL = StrideMNL_;
  // Get base element input type.
  using ElementInput = cute::remove_pointer_t<ElementInput_>;
  // Check if input is an array of pointers.
  static constexpr bool IsArrayOfPointers = is_same_v<ElementInput*, ElementInput_>;
  using PtrRowType = cute::conditional_t<IsArrayOfPointers, ElementInput const* const*, ElementInput const*>;

  static_assert(Stages == 0, "Row broadcast doesn't support smem pipelining");

  static constexpr bool IsDynamicBroadcast = is_same_v<remove_cvref_t<decltype(get<1>(StrideMNL{}))>, bool>; // row vector or scalar broadcast
  static_assert(is_static_v<decltype(take<0,2>(StrideMNL{}))> || IsDynamicBroadcast, "XeRowBroadcast requires static MN stride for non-dynamic broadcast case."); // batch stride can be dynamic or static
  static_assert(take<0,2>(StrideMNL{}) == Stride<_0,_1>{} || IsDynamicBroadcast, "XeRowBroadcast requires MN stride=(0,1) for non-dynamic broadcast case.");

  struct SharedStorage { };

  struct Arguments {
    PtrRowType ptr_row = nullptr;
    ElementInput null_default = ElementInput(0);
    StrideMNL dRow = {};
  };

  struct Params {
    PtrRowType ptr_row = nullptr;
    ElementCompute null_default = ElementCompute(0);
    StrideMNL dRow = {};
  };

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    return {args.ptr_row, ElementCompute(args.null_default), args.dRow};
  }

  template <class ProblemShape>
  static bool
  can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    return true;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream,
    CudaHostAdapter* cuda_adapter = nullptr) {
    return cutlass::Status::kSuccess;
  }

  CUTLASS_HOST_DEVICE
  XeRowBroadcast() { }

  CUTLASS_HOST_DEVICE
  XeRowBroadcast(Params const& params, SharedStorage const& shared_storage)
      : params(params), is_zero_(false) {
    auto const& [stride_M, stride_N, stride_L] = params.dRow;
    // Nullptr default
    if (EnableNullptr && params.ptr_row == nullptr) {
      is_zero_ = params.null_default == ElementCompute(0);
    }
    // Dynamic non-batched scalar broadcast
    else if (IsDynamicBroadcast && stride_N == bool(0) && stride_L == repeat_like(stride_L, 0)) {
       if constexpr (!IsArrayOfPointers) {
         is_zero_ = params.ptr_row[0] == ElementInput(0);
       }
    }
  }

  Params params;
  bool is_zero_ = false;
  ElementInput *smem = nullptr;

  CUTLASS_DEVICE bool
  is_producer_load_needed() const {
    return false;
  }

  CUTLASS_DEVICE bool
  is_C_load_needed() const {
    return false;
  }

  CUTLASS_DEVICE bool
  is_zero() const {
    return is_zero_;
  }

  template <class... Args>
  CUTLASS_DEVICE auto
  get_producer_load_callbacks(ProducerLoadArgs<Args...> const& args) {
    return EmptyProducerLoadCallbacks{};
  }

  template<class MRow, class CTensor>
  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
    CUTLASS_DEVICE
    ConsumerStoreCallbacks(MRow mRow_, CTensor tCcRow_, Params const& params_)
      : mRow(mRow_),
        tCcRow(tCcRow_),
        params(params_) { }

    MRow mRow;
    CTensor tCcRow;
    Params const& params;

    CUTLASS_DEVICE void
    begin() { }

    template <typename ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE Array<ElementCompute, FragmentSize>
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n) {
      Array<ElementCompute, FragmentSize> frg_row;

      if (EnableNullptr && params.ptr_row == nullptr) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < FragmentSize; ++i) {
          frg_row[i] = params.null_default;
        }
        return frg_row;
      }

      Tensor tCcRow_mn = tCcRow(_,_,_,epi_m,epi_n);

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < FragmentSize; ++i) {
        int idx = epi_v * FragmentSize + i;
        auto coord = tCcRow_mn(idx);
        int n_coord = get<1>(coord);
        if (n_coord < get<1>(shape(mRow))) {
          frg_row[i] = ElementCompute(mRow(_0{}, n_coord));
        } else {
          frg_row[i] = ElementCompute(0);
        }
      }

      return frg_row;
    }
  };

  template <
    bool ReferenceSrc, // do register tensors reference the src or dst layout of the tiled copy
    class... Args
  >
  CUTLASS_DEVICE auto
  get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const& args) {
    auto [M, N, K, L] = args.problem_shape_mnkl;
    auto [m, n, k, l] = args.tile_coord_mnkl;

    auto layout_N = [&] () CUTLASS_LAMBDA_FUNC_INLINE {
      auto shape_N = get<1>(args.problem_shape_mnkl);
      if constexpr (IsDynamicBroadcast) {
        auto stride_N = repeat_like(shape_N, int(0));
        if (get<1>(params.dRow) == bool(1)) {
          stride_N = transform_leaf(compact_major<LayoutLeft>(shape_N),
            [] (auto const& stride) { return static_cast<int>(stride); }
          );
        }
        return make_layout(shape_N, stride_N);
      }
      else {
        return make_layout(shape_N);
      }
    }();

    auto layout_M = make_layout(M, repeat_like(M, _0{}));
    ElementInput const* ptr_row;
    if constexpr(IsArrayOfPointers) {
      ptr_row = params.ptr_row[l];  // Array-of-pointers: each batch has separate allocation
    } else {
      // When single contiguous allocation with shape (L, 1, N) or (L, N),
      // compute byte offset: batch_stride elements per batch
      auto batch_stride = size_t(get<2>(params.dRow));
      ptr_row = params.ptr_row + l * batch_stride;  // Advance pointer to batch l's data
    }
    // Create 2D tensor (M, N) with pre-offset pointer
    // Pointer already points to correct batch, so layout should NOT include batch dimension.
    Tensor mRow = make_tensor(make_gmem_ptr(ptr_row), make_layout(layout_M, layout_N));

    // Use global output coordinates from epilogue - these are already partitioned per thread
    // and tiled by (epi_m, epi_n). Row broadcast uses these to index bias vector.
    auto tCcRow = args.tCcD;

    return ConsumerStoreCallbacks(mRow, tCcRow, params);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template<
  int Stages,
  class CtaTileShapeMNK,
  class ElementInput_,
  class ElementCompute = cute::remove_pointer_t<ElementInput_>,
  class StrideMNL_ = Stride<_1,_0,_0>,
  int Alignment = 128 / sizeof_bits_v<cute::remove_pointer_t<ElementInput_>>,
  bool EnableNullptr = true // Fallback scalar broadcast for nullptr params
>
struct XeColBroadcast {
  using StrideMNL = StrideMNL_;
  // Get base element input type.
  using ElementInput = cute::remove_pointer_t<ElementInput_>;
  // Check if input is an array of pointers.
  static constexpr bool IsArrayOfPointers = is_same_v<ElementInput*, ElementInput_>;
  using PtrColType = cute::conditional_t<IsArrayOfPointers, ElementInput const* const*, ElementInput const*>;

  static_assert(Stages == 0, "Column broadcast doesn't support smem pipelining");

  static constexpr bool IsDynamicBroadcast = is_same_v<remove_cvref_t<decltype(get<0>(StrideMNL{}))>, bool>; // column vector or scalar broadcast
  static_assert(is_static_v<decltype(take<0,2>(StrideMNL{}))> || IsDynamicBroadcast, "XeColBroadcast requires static MN stride for non-dynamic broadcast case."); // batch stride can be dynamic or static
  static_assert(take<0,2>(StrideMNL{}) == Stride<_1,_0>{} || IsDynamicBroadcast, "XeColBroadcast requires MN stride=(1,0) for non-dynamic broadcast case.");

  struct SharedStorage { };

  struct Arguments {
    PtrColType ptr_col = nullptr;
    ElementInput null_default = ElementInput(0);
    StrideMNL dCol = {};
  };

  struct Params {
    PtrColType ptr_col = nullptr;
    ElementCompute null_default = ElementCompute(0);
    StrideMNL dCol = {};
  };

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    return {args.ptr_col, ElementCompute(args.null_default), args.dCol};
  }

  template <class ProblemShape>
  static bool
  can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    return true;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream,
    CudaHostAdapter* cuda_adapter = nullptr) {
    return cutlass::Status::kSuccess;
  }

  CUTLASS_HOST_DEVICE
  XeColBroadcast() { }

  CUTLASS_HOST_DEVICE
  XeColBroadcast(Params const& params, SharedStorage const& shared_storage)
      : params(params), is_zero_(false) {
    auto const& [stride_M, stride_N, stride_L] = params.dCol;
    // Nullptr default
    if (EnableNullptr && params.ptr_col == nullptr) {
      is_zero_ = params.null_default == ElementCompute(0);
    }
    // Dynamic non-batched scalar broadcast
    else if (IsDynamicBroadcast && stride_M == bool(0) && stride_L == repeat_like(stride_L, 0)) {
       if constexpr (!IsArrayOfPointers) {
         is_zero_ = params.ptr_col[0] == ElementInput(0);
       }
    }
  }

  Params params;
  bool is_zero_ = false;
  ElementInput *smem = nullptr;

  CUTLASS_DEVICE bool
  is_producer_load_needed() const {
    return false;
  }

  CUTLASS_DEVICE bool
  is_C_load_needed() const {
    return false;
  }

  CUTLASS_DEVICE bool
  is_zero() const {
    return is_zero_;
  }

  template <class... Args>
  CUTLASS_DEVICE auto
  get_producer_load_callbacks(ProducerLoadArgs<Args...> const& args) {
    return EmptyProducerLoadCallbacks{};
  }

  template<class MCol, class CTensor>
  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
    CUTLASS_DEVICE
    ConsumerStoreCallbacks(MCol mCol_, CTensor tCcCol_, Params const& params_)
      : mCol(mCol_),
        tCcCol(tCcCol_),
        params(params_) { }

    MCol mCol;
    CTensor tCcCol;
    Params const& params;

    CUTLASS_DEVICE void
    begin() { }

    template <typename ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE Array<ElementCompute, FragmentSize>
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n) {
      Array<ElementCompute, FragmentSize> frg_col;

      if (EnableNullptr && params.ptr_col == nullptr) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < FragmentSize; ++i) {
          frg_col[i] = params.null_default;
        }
        return frg_col;
      }

      Tensor tCcCol_mn = tCcCol(_,_,_,epi_m,epi_n);

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < FragmentSize; ++i) {
        int idx = epi_v * FragmentSize + i;
        auto coord = tCcCol_mn(idx);
        int m_coord = get<0>(coord);
        if (m_coord < get<0>(shape(mCol))) {
          frg_col[i] = ElementCompute(mCol(m_coord, _0{}));
        } else {
          frg_col[i] = ElementCompute(0);
        }
      }

      return frg_col;
    }
  };

  template <
    bool ReferenceSrc, // do register tensors reference the src or dst layout of the tiled copy
    class... Args
  >
  CUTLASS_DEVICE auto
  get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const& args) {
    auto [M, N, K, L] = args.problem_shape_mnkl;
    auto [m, n, k, l] = args.tile_coord_mnkl;

    auto layout_M = [&] () CUTLASS_LAMBDA_FUNC_INLINE {
      auto shape_M = get<0>(args.problem_shape_mnkl);
      if constexpr (IsDynamicBroadcast) {
        auto stride_M = repeat_like(shape_M, int(0));
        if (get<0>(params.dCol) == bool(1)) {
          stride_M = transform_leaf(compact_major<LayoutLeft>(shape_M),
            [] (auto const& stride) { return static_cast<int>(stride); }
          );
        }
        return make_layout(shape_M, stride_M);
      }
      else {
        return make_layout(shape_M);
      }
    }();

    auto layout_N = make_layout(N, repeat_like(N, _0{}));
    ElementInput const* ptr_col;
    if constexpr(IsArrayOfPointers) {
      ptr_col = params.ptr_col[l];  // Array-of-pointers: each batch has separate allocation
    } else {
      // BATCHED BIAS: Single contiguous allocation with shape (L, M, 1) or (L, M)
      // Compute byte offset: batch_stride elements per batch
      auto batch_stride = size_t(get<2>(params.dCol));
      ptr_col = params.ptr_col + l * batch_stride;  // Advance pointer to batch l's data
    }
    // Create 2D tensor (M, N) with pre-offset pointer - avoids double-offset bug.
    // Key: Pointer already points to correct batch, so layout should NOT include batch dimension.
    Tensor mCol = make_tensor(make_gmem_ptr(ptr_col), make_layout(layout_M, layout_N));

    // Use global output coordinates from epilogue - these are already partitioned per thread
    // and tiled by (epi_m, epi_n). Column broadcast uses these to index bias vector.
    auto tCcCol = args.tCcD;

    return ConsumerStoreCallbacks(mCol, tCcCol, params);
  }
};

// Scalar reduction
template <
  template <class> class RegReduceFn,
  template <class> class GmemReduceFn,
  class ElementOutput,
  class ElementCompute,
  FloatRoundStyle RoundStyle,
  class StrideMNL = Stride<_0,_0,_0>,
  bool EnableNullptr = true // Noop on nullptr params
>
struct XeScalarReduction {
private:
  static_assert(is_static_v<decltype(take<0,2>(StrideMNL{}))>); // batch stride can be dynamic or static
  static_assert(take<0,2>(StrideMNL{}) == Stride<_0,_0>{});
  static constexpr bool IsAtomic = is_atomic<GmemReduceFn<ElementCompute>>::value;
  static_assert(IsAtomic, "non-atomic scalar reduction not supported yet");

public:
  struct SharedStorage { };

  struct Arguments {
    ElementOutput* ptr_scalar = nullptr;
    ElementCompute reduction_identity = ElementCompute(0);
    StrideMNL dScalar = {};
  };

  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    return args;
  }

  template <class ProblemShape>
  static bool
  can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    return true;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream,
    CudaHostAdapter* cuda_adapter = nullptr) {
    if constexpr (IsAtomic) {
      auto problem_shape_mnkl = append<4>(problem_shape, 1);
      auto [M, N, K, L] = problem_shape_mnkl;
      Layout mScalar_layout = make_layout(make_shape(M,N,L), args.dScalar);
      if (args.ptr_scalar != nullptr) {
        return fill_workspace(args.ptr_scalar, ElementOutput(args.reduction_identity), cosize(mScalar_layout), stream, cuda_adapter);
      }
    }

    return cutlass::Status::kSuccess;
  }

  CUTLASS_DEVICE bool
  is_producer_load_needed() const {
    return false;
  }

  CUTLASS_DEVICE bool
  is_C_load_needed() const {
    return false;
  }

  CUTLASS_HOST_DEVICE
  XeScalarReduction() { }

  CUTLASS_HOST_DEVICE
  XeScalarReduction(Params const& params, SharedStorage const& shared_storage)
      : params(params) { }

  Params const params;

  template <class... Args>
  CUTLASS_DEVICE auto
  get_producer_load_callbacks(ProducerLoadArgs<Args...> const& args) {
    return EmptyProducerLoadCallbacks{};
  }

  template<class CTensor, class ThrResidue>
  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
    CUTLASS_DEVICE
    ConsumerStoreCallbacks(
        int l_coord,
        CTensor tCcScalar,
        ThrResidue residue_tCcScalar,
        Params const& params)
      : scalar(params.reduction_identity),
        l_coord(l_coord),
        tCcScalar(tCcScalar),
        residue_tCcScalar(residue_tCcScalar),
        params(params) {}

    ElementCompute scalar;
    int l_coord;
    CTensor tCcScalar;                                                                 // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)
    ThrResidue residue_tCcScalar;
    Params params;

    template <typename ElementAccumulator, typename ElementInput, int FragmentSize>
    CUTLASS_DEVICE auto
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n,
          Array<ElementInput, FragmentSize> const& frg_input) {
      if constexpr (EnableNullptr) {
        if (params.ptr_scalar == nullptr) {
          return frg_input;
        }
      }

      using ConvertInput = NumericArrayConverter<ElementCompute, ElementInput, FragmentSize, RoundStyle>;
      using ReduceInput = RegReduceFn<ElementCompute>;
      ConvertInput convert_input{};
      ReduceInput reduce_input{};

      Array frg_I = convert_input(frg_input);
      Tensor tCcScalar_mn = tCcScalar(_,_,_,epi_m,epi_n);

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < FragmentSize; ++i) {
        if (elem_less(tCcScalar_mn(epi_v * FragmentSize + i), residue_tCcScalar)) {
          scalar = reduce_input(scalar, frg_I[i]);
        }
      }

      return frg_input;
    }

    CUTLASS_DEVICE void
    end() {
      if constexpr (EnableNullptr) {
        if (params.ptr_scalar == nullptr) {
          return;
        }
      }

      using ConvertI = NumericConverter<ElementOutput, ElementCompute, RoundStyle>;
      using ReduceInput = GmemReduceFn<ElementOutput>;

      ConvertI convert_I{};
      ReduceInput reduce_input{};

      ElementOutput* ptr_scalar = params.ptr_scalar + l_coord * get<2>(params.dScalar);
      reduce_input(ptr_scalar, convert_I(scalar));
    }

  };

  template <
    bool ReferenceSrc, // do register tensors reference the src or dst layout of the tiled copy
    class... Args
  >
  CUTLASS_DEVICE auto
  get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const& args) {
    // For scalar reduction, we need to process ALL elements in the output tensor.
    // Use the full problem M,N dimensions for residue checking (not per-tile residue).
    auto residue_mnk = make_coord(size<0>(args.problem_shape_mnkl), size<1>(args.problem_shape_mnkl));
    return ConsumerStoreCallbacks<decltype(args.tCcD), decltype(residue_mnk)>(
      get<3>(args.tile_coord_mnkl), args.tCcD, residue_mnk, params);
  }

};
///////////////////////////////////////////////////////////////////////////////////////////////////////

// Col vector reduction
template <
  template <class> class RegReduceFn,
  template <class> class ShuffleReduceFn,
  template <class> class GmemReduceFn,
  int Stages,
  class CtaTileShapeMNK,
  class ElementOutput,
  class ElementCompute,
  FloatRoundStyle RoundStyle,
  class StrideMNL = Stride<_1,_0,_0>,
  int Alignment = 128 / sizeof_bits_v<ElementOutput>,
  bool EnableNullptr = true, // Noop on nullptr params
  // If this is false, ptr_col is assumed to point to a compact m-major (round_nearest(M,CTA_M), ceil_div(N,CTA_N), L)
  // tensor of ElementCompute. It is the user's responsibility to reduce this to a (M, L) tensor of ElementOutput
  bool FinalReduction = true,
  // False means skip OOB predication if OOB inputs are known to be the reduction identity
  bool VisitCheckOOB = true
>
struct XeColReduction {
private:
  static_assert(Stages == 0, "Smem usage not supported yet");
  static_assert(Alignment * sizeof_bits_v<ElementOutput> % 128 == 0, "sub-16B alignment not supported yet");
  static_assert(is_static_v<decltype(take<0,2>(StrideMNL{}))>); // batch stride can be dynamic or static
  static_assert(take<0,2>(StrideMNL{}) == Stride<_1,_0>{});
  static constexpr bool IsAtomic = is_atomic<GmemReduceFn<ElementCompute>>::value;
  static_assert(not (IsAtomic && not FinalReduction), "atomic reduction must be final");
  using WorkspaceHelper = detail::XeReductionWorkspaceHelper<detail::XeReductionAxis::Column, CtaTileShapeMNK>;

public:
  struct SharedStorage { };

  struct Arguments {
    void* ptr_col = nullptr; // ElementOutput* if FinalReduction, else ElementCompute*
    ElementCompute reduction_identity = ElementCompute(0);
    StrideMNL dCol = {};
  };

  struct Params {
    void* ptr_col = nullptr;
    ElementCompute reduction_identity = ElementCompute(0);
    StrideMNL dCol = {};
    ElementCompute* reduction_buffer = nullptr;
    int* tile_counters = nullptr;
  };

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    ElementCompute* reduction_buffer;
    int* tile_counters = nullptr;
    if constexpr (IsAtomic) {
      reduction_buffer = nullptr;
    }
    else if constexpr (FinalReduction) {
      auto workspace_layout = WorkspaceHelper::template layout<ElementCompute>(problem_shape);
      size_t tile_counters_offset = round_nearest(workspace_layout.reduction_buffer_bytes, MinWorkspaceAlignment);

      reduction_buffer = reinterpret_cast<ElementCompute*>(workspace);
      tile_counters = reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(workspace) + tile_counters_offset);
    }
    else {
      reduction_buffer = reinterpret_cast<ElementCompute*>(args.ptr_col);
    }

    return {
      args.ptr_col,
      args.reduction_identity,
      args.dCol,
      reduction_buffer,
      tile_counters
    };
  }

  template <class ProblemShape>
  static bool
  can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    return true;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    if constexpr (IsAtomic || not FinalReduction) {
      return 0;
    }

    auto workspace_layout = WorkspaceHelper::template layout<ElementCompute>(problem_shape);
    size_t workspace_size = workspace_layout.reduction_buffer_bytes;
    workspace_size = round_nearest(workspace_size, MinWorkspaceAlignment);
    workspace_size += workspace_layout.tile_counter_count * sizeof(int);
    return workspace_size;
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream,
    CudaHostAdapter* cuda_adapter = nullptr) {
    if constexpr (IsAtomic) {
      auto problem_shape_mnkl = append<4>(problem_shape, 1);
      auto [M, N, K, L] = problem_shape_mnkl;
      Layout mCol_layout = make_layout(make_shape(size<>(M),size<>(N),size<>(L)), args.dCol);
      if (args.ptr_col != nullptr) {
        return fill_workspace(args.ptr_col, ElementOutput(args.reduction_identity), cosize(mCol_layout), stream, cuda_adapter);
      }
      return Status::kSuccess;
    }
    else if constexpr (FinalReduction) {
      auto workspace_layout = WorkspaceHelper::template layout<ElementCompute>(problem_shape);
      size_t tile_counters_offset = round_nearest(workspace_layout.reduction_buffer_bytes, MinWorkspaceAlignment);

      int* tile_counters = reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(workspace) + tile_counters_offset);
      size_t tile_counters_size = workspace_layout.tile_counter_count * sizeof(int);
      return zero_workspace(tile_counters, tile_counters_size, stream, cuda_adapter);
    }
    else {
      return Status::kSuccess;
    }
  }

  CUTLASS_DEVICE bool
  is_producer_load_needed() const {
    return false;
  }

  CUTLASS_DEVICE bool
  is_C_load_needed() const {
    return false;
  }

  CUTLASS_HOST_DEVICE
  XeColReduction() { }

  CUTLASS_HOST_DEVICE
  XeColReduction(Params const& params, SharedStorage const& shared_storage)
      : params(params) { }

  Params params;

  template <class... Args>
  CUTLASS_DEVICE auto
  get_producer_load_callbacks(ProducerLoadArgs<Args...> const& args) {
    return EmptyProducerLoadCallbacks{};
  }

  template<class ArgsTuple>
  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
    CUTLASS_DEVICE
    ConsumerStoreCallbacks(ArgsTuple&& args_tuple, Params const& params)
      : args_tuple(cute::forward<ArgsTuple>(args_tuple)),
        params(params) {}

    ArgsTuple args_tuple;
    Params const& params;
    bool do_final_reduction = false;

    template <typename ElementAccumulator, typename ElementInput, int FragmentSize>
    CUTLASS_DEVICE auto
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n,
          Array<ElementInput, FragmentSize> const& frg_input) {
      if constexpr (EnableNullptr) {
        if (params.ptr_col == nullptr) {
          return frg_input;
        }
      }

      auto& [ref_src, tCrCol, tCcCol, gCol_l, cCol, gBuf_nl, sBuf_layout,
              lane_layout_MN, lane_mn, warp_layout_MN, warp_mn,
              tile_coord_mnkl, residue_cCol, residue_tCcCol, epi_tile, tiled_copy, thread_idx] = args_tuple;
      Tensor tCrCol_mn = tCrCol(_,_,_,epi_m,epi_n);
      Tensor tCcCol_mn = tCcCol(_,_,_,epi_m,epi_n);

      using ConvertInput = NumericArrayConverter<ElementCompute, ElementInput, FragmentSize, RoundStyle>;
      using ReduceInput = RegReduceFn<ElementCompute>;
      ConvertInput convert_input{};
      ReduceInput reduce_input{};

      Array frg_I = convert_input(frg_input);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < FragmentSize; ++i) {
        if (!VisitCheckOOB || elem_less(tCcCol_mn(epi_v * FragmentSize + i), residue_tCcCol)) {
          ElementCompute& tCrCol_vmn = tCrCol_mn(epi_v * FragmentSize + i);
          tCrCol_vmn = reduce_input(tCrCol_vmn, frg_I[i]);
        }
      }

      return frg_input;
    }

    template <class STensor, class SyncFn, class VTensor>
    CUTLASS_DEVICE void
    reduce(STensor&& smem_buffer, SyncFn const& sync_fn, int epi_m, int epi_n, bool is_last_iteration, VTensor visit_results) {
      auto& [ref_src, tCrCol, tCcCol, gCol_l, cCol, gBuf_nl, sBuf_layout,
              lane_layout_MN, lane_mn, warp_layout_MN, warp_mn,
              tile_coord_mnkl, residue_cCol, residue_tCcCol, epi_tile, tiled_copy, thread_idx] = args_tuple;
      auto [m, n, k, l] = tile_coord_mnkl;

      if (not is_last_iteration) {
        return;
      }

      constexpr bool ReferenceSrc = decltype(ref_src)::value;

      // Runtime nullptr is noop
      if constexpr (EnableNullptr) {
        if (params.ptr_col == nullptr) {
          return;
        }
      }

      // fully OOB CTA in partially OOB cluster
      // Use residue_tCcCol (full problem size) instead of residue_cCol (clipped) since cCol contains global coordinates
      if (not elem_less(cCol(_0{},_0{}), residue_tCcCol)) {
        return;
      }

      //
      // 1. Warp shuffle reduction
      //
      using FragmentShuffle = Array<ElementCompute, sizeof(uint64_t) / sizeof(ElementCompute)>;
      using ReduceShuffle = ShuffleReduceFn<FragmentShuffle>;
      ReduceShuffle reduce_shuffle{};
      Tensor tCrCol_frg = recast<FragmentShuffle>(filter(tCrCol));
      CUTLASS_PRAGMA_UNROLL
      for (int reduction_cols = size<1>(lane_layout_MN) / 2; reduction_cols > 0; reduction_cols /= 2) {
        CUTLASS_PRAGMA_UNROLL
        for (int frg_idx = 0; frg_idx < size(tCrCol_frg); ++frg_idx) {
          uint64_t frg_shfl = reinterpret_cast<uint64_t&>(tCrCol_frg(frg_idx));
          frg_shfl = shfl_down_sync(0xFFFFFFFF, frg_shfl, lane_layout_MN(_0{},reduction_cols));
          tCrCol_frg(frg_idx) = reduce_shuffle(tCrCol_frg(frg_idx), reinterpret_cast<FragmentShuffle&>(frg_shfl));
        }
      }
      bool is_reduced_lane = get<1>(lane_mn) == 0;

      //
      // 2. Atomic reduction
      //
      if constexpr (IsAtomic) {
        // Filter so we don't issue redunant copies over stride-0 modes
        Tensor tCrCol_flt = filter_zeros(tCrCol);
        Tensor tCcCol_flt = make_tensor(tCcCol.data(), make_layout(tCrCol_flt.shape(), tCcCol.stride()));
        Tensor tCgCol = sm90_partition_for_epilogue<ReferenceSrc>(gCol_l(_,_,l), epi_tile, tiled_copy, thread_idx);
        Tensor tCgCol_flt = filter_zeros(tCgCol);

        // NOTE: atomic reduction is performed in the output type
        using ConvertOutput = NumericConverter<ElementOutput, ElementCompute, RoundStyle>;
        using ReduceOutput = GmemReduceFn<ElementOutput>;
        ConvertOutput convert_output{};
        ReduceOutput reduce_output{};

        if (is_reduced_lane) {
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(tCrCol_flt); ++i) {
            if (elem_less(tCcCol_flt(i), residue_tCcCol)) {
              // tCcCol_flt reports global (M,N) coordinates; convert to tile-local before indexing gCol_l
              auto [tile_m, tile_n, tile_k, tile_l] = tile_coord_mnkl;
              auto tile_shape = shape(gCol_l);  // (CTA_M, CTA_N, L)
              int tile_extent_m = int(size<0>(tile_shape));
              int tile_extent_n = int(size<1>(tile_shape));
              int global_m = int(get<0>(tCcCol_flt(i)));
              int global_n = int(get<1>(tCcCol_flt(i)));
              int local_m = global_m - tile_m * tile_extent_m;
              int local_n = global_n - tile_n * tile_extent_n;
              auto* output_ptr = &gCol_l(local_m, local_n, tile_l);
              reduce_output(output_ptr, convert_output(tCrCol_flt(i)));
            }
          }
        }
        
        sync_fn();
      }

      //
      // 2. One warp in N, skip threadblock smem reduction
      //
      else if constexpr (decltype(size<1>(warp_layout_MN))::value <= 1) {
        // Dump warp reduction to gmem workspace
        using ElementGmem = cute::conditional_t<FinalReduction, ElementCompute volatile, ElementCompute>;
        Tensor tCgBuf = sm90_partition_for_epilogue<ReferenceSrc>(gBuf_nl(_,_,n,l), epi_tile, tiled_copy, thread_idx);
        if (is_reduced_lane) {
          copy_aligned(tCrCol, recast<ElementGmem>(tCgBuf));
        }
        sync_fn();
      }

      //
      // 2. Multiple warps in N, do threadblock smem reduction
      //
      else {
        Tensor sBuf = make_tensor(make_smem_ptr<ElementCompute>(raw_pointer_cast(smem_buffer.data())), sBuf_layout);
        static_assert(decltype(cosize(sBuf.layout()))::value * sizeof(ElementCompute) <=
                      decltype(cosize(smem_buffer.layout()))::value * sizeof(typename remove_cvref_t<STensor>::value_type),
                      "smem reduction buffer not large enough, use a larger epilogue tile");
        sync_fn();

        // Dump warp reduction to smem workspace
        Tensor tCsBuf = sm90_partition_for_epilogue<ReferenceSrc>(sBuf(_,_,get<1>(warp_mn)), epi_tile, tiled_copy, thread_idx);
        if (is_reduced_lane) {
          copy_aligned(tCrCol, tCsBuf);
        }
        sync_fn();

        constexpr int SmemFragSize = cute::max(size_t{1}, sizeof(uint32_t) / sizeof(ElementCompute));
        using FragmentSmem = Array<ElementCompute, SmemFragSize>;
        using VectorSmem = uint_bit_t<sizeof_bits_v<FragmentSmem>>;
        using ReduceSmem = GmemReduceFn<FragmentSmem>;
        ReduceSmem reduce_smem{};

        Tensor sBuf_frg = recast<FragmentSmem>(filter_zeros(sBuf));
        Tensor sBuf_vec = recast<VectorSmem>(filter_zeros(sBuf));
        constexpr int FragsPerCol = decltype(size<0>(sBuf_frg))::value;

        // Do the threadblock smem reduction
        CUTLASS_PRAGMA_UNROLL
        for (int reduction_cols = size<1>(warp_layout_MN) / 2; reduction_cols > 1; reduction_cols /= 2) {
          int FragsPerReduction = reduction_cols * FragsPerCol;
          CUTLASS_PRAGMA_NO_UNROLL
          for (int frg_idx = thread_idx; frg_idx < FragsPerReduction; frg_idx += size(tiled_copy)) {
            FragmentSmem frg_smem = reduce_smem(sBuf_frg(frg_idx), sBuf_frg(frg_idx + FragsPerReduction));
            sBuf_vec(frg_idx) = reinterpret_cast<VectorSmem&>(frg_smem);
          }
          sync_fn();
        }

        // Do final smem reduction and dump to gmem workspace
        using VectorGmem = cute::conditional_t<FinalReduction, VectorSmem volatile, VectorSmem>;
        Tensor gBuf_vec = recast<VectorGmem>(filter(gBuf_nl(_,_,n,l)));
        CUTLASS_PRAGMA_NO_UNROLL
        for (int frg_idx = thread_idx; frg_idx < FragsPerCol; frg_idx += size(tiled_copy)) {
          FragmentSmem frg_smem = reduce_smem(sBuf_frg(frg_idx), sBuf_frg(frg_idx + FragsPerCol));
          gBuf_vec(frg_idx) = reinterpret_cast<VectorSmem&>(frg_smem);
        }
        sync_fn();
      }

      //
      // 3. Increment atomic counters to signal final gmem reduction
      //
      if constexpr (not IsAtomic && FinalReduction) {
        // Ensure gmem writes are visible to other threads before incrementing counter
        threadfence();
        sync_fn();
        int axis_index = int(m);
        int total_tiles = size<2>(gBuf_nl) * size<3>(gBuf_nl);
        do_final_reduction = detail::xe_signal_final_reduction(
          params.tile_counters,
          axis_index,
          total_tiles,
          thread_idx,
          raw_pointer_cast(smem_buffer.data()),
          sync_fn);
      }
    }

    CUTLASS_DEVICE void
    end() {
      //
      // 4. Do final gmem reduction if necessary
      //
      if constexpr (not IsAtomic && FinalReduction) {
        if (not do_final_reduction) {
          return;
        }

        auto& [ref_src, tCrCol, tCcCol, gCol_l, cCol, gBuf_nl, sBuf_layout,
                lane_layout_MN, lane_mn, warp_layout_MN, warp_mn,
                tile_coord_mnkl, residue_cCol, residue_tCcCol, epi_tile, tiled_copy, thread_idx] = args_tuple;

        using ReduceOutput = GmemReduceFn<ElementCompute>;
        using ConvertOutput = NumericConverter<ElementOutput, ElementCompute, RoundStyle>;
        ReduceOutput reduce_output{};
        ConvertOutput convert_output{};

        // Reduction over batches
        if (size<2>(stride(gCol_l)) == 0) {
          CUTLASS_PRAGMA_NO_UNROLL
          for (int m = thread_idx; m < size<0>(gBuf_nl); m += size(tiled_copy)) {
            Tensor tRgBuf_nl = gBuf_nl(m,_0{},_,_);
            ElementCompute output = tRgBuf_nl(_0{});
            CUTLASS_PRAGMA_NO_UNROLL
            for (int nl = 1; nl < size(tRgBuf_nl); ++nl) {
              output = reduce_output(output, tRgBuf_nl(nl));
            }
            if (elem_less(cCol(m,_0{}), residue_tCcCol)) {
              gCol_l(m,_0{},_0{}) = convert_output(output);
            }
          }
        }
        // No reduction over batches
        else {
          CUTLASS_PRAGMA_NO_UNROLL
          for (int m = thread_idx; m < size<0>(gBuf_nl); m += size(tiled_copy)) {
            bool do_store = elem_less(cCol(m,_0{}), residue_tCcCol);
            CUTLASS_PRAGMA_NO_UNROLL
            for (int l = 0; l < size<3>(gBuf_nl); ++l) {
              Tensor tRgBuf_n = gBuf_nl(m,_0{},_,l);
              ElementCompute output = tRgBuf_n(_0{});
              CUTLASS_PRAGMA_NO_UNROLL
              for (int n = 1; n < size(tRgBuf_n); ++n) {
                output = reduce_output(output, tRgBuf_n(n));
              }
              if (do_store) {
                gCol_l(m,_0{},l) = convert_output(output);
              }
            }
          }
        }

      }
    }

  };

  template <
    bool ReferenceSrc, // do register tensors reference the src or dst layout of the tiled copy
    class... Args
  >
  CUTLASS_DEVICE auto
  get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const& args) {
    Layout ref_layout_MN = [&] () {
      auto mn_shape = shape(typename decltype(args.tiled_copy)::Tiler_MN{});
      if constexpr (ReferenceSrc) { return right_inverse(args.tiled_copy.get_layoutS_TV()).with_shape(mn_shape); }
      else                        { return right_inverse(args.tiled_copy.get_layoutD_TV()).with_shape(mn_shape); }
    }();                                                                                         // tile_mn -> tv_idx

    // Get the MN layout + coord of lanes to determine shuffle reduction iterations
    using _W = Int<decltype(args.tiled_copy)::TiledNumThr::value / NumThreadsPerWarp>;
    Layout tv2lane = Layout<Shape<Int<NumThreadsPerWarp>,_W,_1>,Stride<_1,_0,_0>>{};            //   tv_idx -> lane_idx
    Layout ref2lane = composition(tv2lane, ref_layout_MN);                                      //  tile_mn -> lane_idx
    Layout lane_layout_MN = make_layout(filter(get<0>(ref2lane)), filter(get<1>(ref2lane)));    //  lane_mn -> lane_idx
    Layout inv_lane_layout_MN = right_inverse(lane_layout_MN);                                  // lane_idx -> lane_mn
    int lane_idx = canonical_lane_idx();
    auto lane_mn = idx2crd(inv_lane_layout_MN(lane_idx), shape(lane_layout_MN));

    // Get the MN layout + coord of warps to determine smem reduction iterations
    Layout tv2warp = Layout<Shape<Int<NumThreadsPerWarp>,_W,_1>,Stride<_0,_1,_0>>{};            //   tv_idx -> warp_idx
    Layout ref2warp = composition(tv2warp, ref_layout_MN);                                      //  tile_mn -> warp_idx
    Layout warp_layout_MN = make_layout(filter(get<0>(ref2warp)), filter(get<1>(ref2warp)));    //  warp_mn -> warp_idx
    Layout inv_warp_layout_MN = right_inverse(warp_layout_MN);                                  // warp_idx -> warp_mn
    int warp_idx = args.thread_idx / NumThreadsPerWarp;
    auto warp_mn = idx2crd(inv_warp_layout_MN(warp_idx), shape(warp_layout_MN));

    // Partition output gmem and register tensors
    auto [tile_M, tile_N, tile_K] = args.tile_shape_mnk;
    auto [M, N, K, L] = args.problem_shape_mnkl;
    auto [m, n, k, l] = args.tile_coord_mnkl;

    Tensor mCol = make_tensor(make_gmem_ptr<ElementOutput>(params.ptr_col), make_shape(M,N,L), params.dCol); // (M,N,L)
    Tensor gCol_l = local_tile(mCol, take<0,2>(args.tile_shape_mnk), make_coord(m,n,_));             // (CTA_M,CTA_N,L)
    Tensor tCgCol = sm90_partition_for_epilogue<ReferenceSrc>(                         // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)
                      gCol_l(_,_,l), args.epi_tile, args.tiled_copy, args.thread_idx);
    Tensor tCrCol = make_tensor_like<ElementCompute>(tCgCol);                          // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)
    fill(tCrCol, params.reduction_identity);

    // Partition gmem+smem reduction buffer tensors
    Layout gBuf_layout = make_layout(take<0,2>(args.tile_shape_mnk), make_stride(_1{}, _0{}));
    Layout mBuf_layout = blocked_product(gBuf_layout, make_layout(ceil_div(make_shape(M,N,L), shape(gBuf_layout))));
    Tensor mBuf = make_tensor(make_gmem_ptr(params.reduction_buffer), mBuf_layout);                // (ceil_M,ceil_N,L)
    Tensor gBuf_nl = local_tile(mBuf, take<0,2>(args.tile_shape_mnk), make_coord(m,_,_));     // (CTA_M,CTA_N,REST_N,L)
    Layout sBuf_layout = blocked_product(gBuf_layout,make_layout(make_shape(_1{},_1{},size<1>(warp_layout_MN)))); // (CTA_M,CTA_N,WARPS_N)

    // For column reduction, use full problem dimensions for residue checking
    auto residue_mnk = make_coord(size<0>(args.problem_shape_mnkl), size<1>(args.problem_shape_mnkl));
    auto args_tuple = make_tuple(
        bool_constant<ReferenceSrc>{}, cute::move(tCrCol), args.tCcD, gCol_l, args.cD, gBuf_nl, sBuf_layout,
        lane_layout_MN, lane_mn, warp_layout_MN, warp_mn,
        args.tile_coord_mnkl, args.residue_cD, residue_mnk, args.epi_tile, args.tiled_copy, args.thread_idx);
    return ConsumerStoreCallbacks<decltype(args_tuple)>(std::move(args_tuple), params);
  }
};
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Row vector reduction
template <
  template <class> class RegReduceFn,
  template <class> class ShuffleReduceFn,
  template <class> class GmemReduceFn,
  int Stages,
  class CtaTileShapeMNK,
  class ElementOutput,
  class ElementCompute,
  FloatRoundStyle RoundStyle,
  class StrideMNL = Stride<_0,_1,_0>,
  int Alignment = 128 / sizeof_bits_v<ElementOutput>,
  bool EnableNullptr = true, // Noop on nullptr params
  // If this is false, ptr_row is assumed to point to a compact n-major (ceil_div(M,CTA_M), round_nearest(N,CTA_N), L)
  // tensor of ElementCompute. It is the user's responsibility to reduce this to a (N, L) tensor of ElementOutput
  bool FinalReduction = true,
  // False means skip OOB predication if OOB inputs are known to be the reduction identity
  bool VisitCheckOOB = true,
  // Indicate the parameter order when calling RegReduceFn
  // Seq length equals the number of RegReduceFn parameters
  // No.0 represents tCrRow; No.1 and subsequent numbers sequentially represent frg_inputs in `visit`
  class RegReduceSeq = cute::seq<0, 1>
>
struct XeRowReduction {
private:
  static_assert(Stages == 0, "Smem usage not supported yet");
  static_assert(Alignment * sizeof_bits_v<ElementOutput> % 128 == 0, "sub-16B alignment not supported yet");
  static_assert(is_static_v<decltype(take<0,2>(StrideMNL{}))>); // batch stride can be dynamic or static
  static_assert(take<0,2>(StrideMNL{}) == Stride<_0,_1>{}); // Tensor column major
  static constexpr bool IsAtomic = is_atomic<GmemReduceFn<ElementCompute>>::value;
  static_assert(not (IsAtomic && not FinalReduction), "atomic reduction must be final");
  using WorkspaceHelper = detail::XeReductionWorkspaceHelper<detail::XeReductionAxis::Row, CtaTileShapeMNK>;

public:
  struct SharedStorage { };

  struct Arguments {
    void* ptr_row = nullptr; // ElementOutput* if FinalReduction, else ElementCompute*
    ElementCompute reduction_identity = ElementCompute(0);
    StrideMNL dRow = {};
  };

  struct Params {
    void* ptr_row = nullptr;
    ElementCompute reduction_identity = ElementCompute(0);
    StrideMNL dRow = {};
    ElementCompute* reduction_buffer = nullptr;
    int* tile_counters = nullptr;
  };

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    ElementCompute* reduction_buffer;
    int* tile_counters = nullptr;
    if constexpr (IsAtomic) {
      reduction_buffer = nullptr; // shared local memory
    }
    else if constexpr (FinalReduction) {
      auto workspace_layout = WorkspaceHelper::template layout<ElementCompute>(problem_shape);
      size_t tile_counters_offset = round_nearest(workspace_layout.reduction_buffer_bytes, MinWorkspaceAlignment);

      reduction_buffer = reinterpret_cast<ElementCompute*>(workspace);
      tile_counters = reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(workspace) + tile_counters_offset);
    }
    else {
      reduction_buffer = reinterpret_cast<ElementCompute*>(args.ptr_row);
    }

    return {
      args.ptr_row,
      args.reduction_identity,
      args.dRow,
      reduction_buffer,
      tile_counters
    };
  }

  template <class ProblemShape>
  static bool
  can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    return true;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    if constexpr (IsAtomic || not FinalReduction) {
      return 0;
    }

    auto workspace_layout = WorkspaceHelper::template layout<ElementCompute>(problem_shape);
    size_t workspace_size = workspace_layout.reduction_buffer_bytes;
    workspace_size = round_nearest(workspace_size, MinWorkspaceAlignment);
    workspace_size += workspace_layout.tile_counter_count * sizeof(int);
    return workspace_size;
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream,
    CudaHostAdapter* cuda_adapter = nullptr) {
    if constexpr (IsAtomic) {
      auto problem_shape_mnkl = append<4>(problem_shape, 1);
      auto [M, N, K, L] = problem_shape_mnkl;
      Layout mRow_layout = make_layout(make_shape(size<>(M),size<>(N),size<>(L)), args.dRow);
      if (args.ptr_row != nullptr) {
        return fill_workspace(args.ptr_row, ElementOutput(args.reduction_identity), cosize(mRow_layout), stream, cuda_adapter);
      }
      return Status::kSuccess;
    }
    else if constexpr (FinalReduction) {
      auto workspace_layout = WorkspaceHelper::template layout<ElementCompute>(problem_shape);
      size_t tile_counters_offset = round_nearest(workspace_layout.reduction_buffer_bytes, MinWorkspaceAlignment);

      int* tile_counters = reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(workspace) + tile_counters_offset);
      size_t tile_counters_size = workspace_layout.tile_counter_count * sizeof(int);
      return zero_workspace(tile_counters, tile_counters_size, stream, cuda_adapter);
    }
    else {
      return Status::kSuccess;
    }
  }

  CUTLASS_DEVICE bool
  is_producer_load_needed() const {
    return false;
  }

  CUTLASS_DEVICE bool
  is_C_load_needed() const {
    return false;
  }

  CUTLASS_HOST_DEVICE
  XeRowReduction() { }

  CUTLASS_HOST_DEVICE
  XeRowReduction(Params const& params, SharedStorage const& shared_storage)
      : params(params) { }

  Params params;

  template <class... Args>
  CUTLASS_DEVICE auto
  get_producer_load_callbacks(ProducerLoadArgs<Args...> const& args) {
    return EmptyProducerLoadCallbacks{};
  }

  template<class ArgsTuple>
  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
    CUTLASS_DEVICE
    ConsumerStoreCallbacks(ArgsTuple&& args_tuple, Params const& params)
      : args_tuple(cute::forward<ArgsTuple>(args_tuple)),
        params(params) {}

    ArgsTuple args_tuple;
    Params const& params;
    bool do_final_reduction = false;

    template <typename ElementAccumulator, typename... ElementInputs, int FragmentSize>
    CUTLASS_DEVICE auto
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n,
          Array<ElementInputs, FragmentSize> const&... frg_inputs) {
      
      if constexpr (EnableNullptr) {
        if (params.ptr_row == nullptr) {
          return cute::get<0>(cute::make_tuple(frg_inputs...));
        }
      }

      auto& [ref_src, tCrRow, tCcRow, gRow_l, cRow, gBuf_ml, sBuf_layout,
        lane_layout_MN, lane_mn, warp_layout_MN, warp_mn,
        tile_coord_mnkl, residue_cRow, residue_tCcRow, epi_tile, tiled_copy, thread_idx] = args_tuple;
      Tensor tCrRow_mn = tCrRow(_,_,_,epi_m,epi_n);
      Tensor tCcRow_mn = tCcRow(_,_,_,epi_m,epi_n);

      if constexpr (VisitCheckOOB) {
        using ReduceInput = RegReduceFn<ElementCompute>;
        ReduceInput reduce_input{};

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < FragmentSize; ++i) {
          if (elem_less(tCcRow_mn(epi_v * FragmentSize + i), residue_tCcRow)) {
            ElementCompute& tCrRow_vmn = tCrRow_mn(epi_v * FragmentSize + i);
            tCrRow_vmn = transform_apply(cute::make_tuple(frg_inputs...),
                [&] (auto&& frg_input) {
                  return ElementCompute(frg_input[i]);
                },
                [&] (auto&&... cvt_frg_inputs) {
                  auto frg_compute_tuple = cute::make_tuple(tCrRow_vmn, cvt_frg_inputs...);
                  return cute::detail::apply(frg_compute_tuple, reduce_input, RegReduceSeq{});
                });
          }
        }
      }
      else {
        constexpr int RegFragSize = cute::max(1, static_cast<int>(sizeof(uint32_t) / sizeof(ElementCompute)));
        using ReduceInput = RegReduceFn<Array<ElementCompute, RegFragSize>>;
        ReduceInput reduce_input{};
        Tensor tCrRow_mn_frg = recast<Array<ElementCompute, RegFragSize>>(tCrRow_mn);

        constexpr int RegFragArraySize = FragmentSize / RegFragSize;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < RegFragArraySize; ++i) {
          Array<ElementCompute, RegFragSize>& tCrRow_vmn_frg = tCrRow_mn_frg(epi_v * RegFragArraySize + i);
          tCrRow_vmn_frg = transform_apply(cute::make_tuple(frg_inputs...),
              [&] (auto&& frg_input) {
                using ElementInput = typename cute::remove_cvref_t<decltype(frg_input)>::Element;
                using ConvertInput = NumericArrayConverter<ElementCompute, ElementInput, RegFragSize, RoundStyle>;
                using RegFragArr = Array<Array<ElementCompute, RegFragSize>, RegFragArraySize>;
                ConvertInput convert_input{};
                return convert_input(reinterpret_cast<RegFragArr&>(frg_input)[i]);
              },
              [&] (auto&&... cvt_frg_inputs) {
                auto frg_compute_tuple = cute::make_tuple(tCrRow_vmn_frg, cvt_frg_inputs...);
                return cute::detail::apply(frg_compute_tuple, reduce_input, RegReduceSeq{});
              });
        }
      }
      return cute::get<0>(cute::make_tuple(frg_inputs...));
    }

    template <class STensor, class SyncFn, class VTensor>
    CUTLASS_DEVICE void
    reduce(STensor&& smem_buffer, SyncFn const& sync_fn, int epi_m, int epi_n, bool is_last_iteration, VTensor visit_results) {
      if (not is_last_iteration) {
        return;
      }

      auto& [ref_src, tCrRow, tCcRow, gRow_l, cRow, gBuf_ml, sBuf_layout,
        lane_layout_MN, lane_mn, warp_layout_MN, warp_mn,
        tile_coord_mnkl, residue_cRow, residue_tCcRow, epi_tile, tiled_copy, thread_idx] = args_tuple;
      auto [m, n, k, l] = tile_coord_mnkl;
      constexpr bool ReferenceSrc = decltype(ref_src)::value;
      if constexpr (EnableNullptr) {
        if (params.ptr_row == nullptr) {
          return;
        }
      }

      // fully OOB CTA in partially OOB cluster
      auto residue_zero = repeat_like(residue_cRow, _0{});
      if (not elem_less(residue_zero, residue_cRow)) {
        return;
      }

      int lane_m = get<0>(lane_mn);
      [[maybe_unused]] bool is_reduced_lane = lane_m == 0;

      //
      // 1. Warp shuffle reduction
      //
      using FragmentShuffle = Array<ElementCompute, sizeof(uint64_t) / sizeof(ElementCompute)>;
      Tensor tCrRow_frg = recast<FragmentShuffle>(filter(tCrRow));
      using ReduceShuffle = ShuffleReduceFn<FragmentShuffle>;
      ReduceShuffle reduce_shuffle{};

      auto FrgSizePerLaneM = size(tCrRow_frg) / size<0>(lane_layout_MN);
      constexpr bool SwapShuffle = (FrgSizePerLaneM > 0) && (!IsAtomic);

      //
      // Swap Shuffle
      //
      // The normal way to reduction among threads:
      // use shuffle to let *** the first half of threads *** have *** whole data *** from the second half of threads.
      // After each step of reduction, a half of threads won't work in the following steps.
      // That is, as the reduction progresses, the efficiency of shuffle & reduction instructions gradually change from 1/2, 1/4 to 1/32 (the worst case).
      //
      // To overcome this shortcoming, for a NxN matrix to be reduced among N threads as a 1XN vectors,
      // we use swap & shuffle aiming to let *** each half of threads *** have *** a half of data *** from the other half of threads.
      // After reduction, each half of threads should deal with a (N/2)x(N/2) sub-matrix independently in the following step.
      // We can recursively do this until the problem size is 1.
      //
      if constexpr (SwapShuffle) { // for a NxN matrix to be reduced among N threads as a 1XN vectors
        Tensor tCrRow_frg_ = logical_divide(tCrRow_frg, FrgSizePerLaneM);                       // (FrgSizePerLaneM, M)
        CUTLASS_PRAGMA_UNROLL
        for (int m = size<1>(tCrRow_frg_) / 2; m > 0; m /= 2) {
          CUTLASS_PRAGMA_UNROLL
          for (int r = 0; r < m; ++r) {
            auto frg_A = tCrRow_frg_(_,r);
            auto frg_B = tCrRow_frg_(_,r + m);
            CUTLASS_PRAGMA_UNROLL
            for (int v = 0; v < size(frg_A); ++v) {
              // Step1: swap
              if (not (lane_m & m)) { // the first half of threads swap fragments from the first half of data to the second
                cutlass::swap(frg_A(v), frg_B(v));
              }

              // Step2: shuffle
              uint64_t frg_shfl = reinterpret_cast<uint64_t&>(frg_A(v));
              // each half of threads get a half of data from the other half of threads
              frg_shfl = shfl_xor_sync(0xFFFFFFFF, frg_shfl, lane_layout_MN(m, _0{}));

              // Step3: reduction
              frg_A(v) = reduce_shuffle(frg_B(v), reinterpret_cast<FragmentShuffle&>(frg_shfl));
            }
          }
        }
      }
      else {
        CUTLASS_PRAGMA_UNROLL
        for (int reduction_rows = size<0>(lane_layout_MN) / 2; reduction_rows > 0; reduction_rows /= 2) {
          CUTLASS_PRAGMA_UNROLL
          for (int frg_idx = 0; frg_idx < size(tCrRow_frg); ++frg_idx) {
            uint64_t frg_shfl = reinterpret_cast<uint64_t&>(tCrRow_frg(frg_idx));
            frg_shfl = shfl_down_sync(0xFFFFFFFF, frg_shfl, lane_layout_MN(reduction_rows, _0{}));
            tCrRow_frg(frg_idx) = reduce_shuffle(tCrRow_frg(frg_idx), reinterpret_cast<FragmentShuffle&>(frg_shfl));
          }
        }
      }

      //
      // 2. Atomic reduction
      //
      if constexpr (IsAtomic) {
        // Filter so we don't issue redunant copies over stride-0 modes
        Tensor tCrRow_flt = filter_zeros(tCrRow);
        Tensor tCcRow_flt = make_tensor(tCcRow.data(), make_layout(tCrRow_flt.shape(), tCcRow.stride()));
        auto FltFrgSizePerLaneM = size(tCrRow_flt) / size<0>(lane_layout_MN);

        Tensor tCgRow = sm90_partition_for_epilogue<ReferenceSrc>(gRow_l(_,_,l), epi_tile, tiled_copy, thread_idx);
        int tile_extent_m = int(size<0>(gRow_l));
        int tile_extent_n = int(size<1>(gRow_l));
        int tile_m = int(m);
        int tile_n = int(n);
        int tile_l = int(l);
        // NOTE: atomic reduction is performed in the output type
        using ConvertOutput = NumericConverter<ElementOutput, ElementCompute, RoundStyle>;
        using ReduceOutput = GmemReduceFn<ElementOutput>;
        ConvertOutput convert_output{};
        ReduceOutput reduce_output{};

        if constexpr (SwapShuffle) {
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < FltFrgSizePerLaneM; ++i) {
            int idx = lane_m * FltFrgSizePerLaneM + i;
            // Row reduction produces output indexed by M (rows), so check M coordinate
            if (get<1>(tCcRow_flt(idx)) < get<1>(residue_tCcRow)) {
              int global_m = int(get<0>(tCcRow_flt(idx)));
              int global_n = int(get<1>(tCcRow_flt(idx)));
              int local_m = global_m - tile_m * tile_extent_m;
              int local_n = global_n - tile_n * tile_extent_n;
              auto* output_ptr = &gRow_l(local_m, local_n, tile_l);
              reduce_output(output_ptr, convert_output(tCrRow_flt(i)));
            } 
          }
        }
        else {
          if (is_reduced_lane) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(tCrRow_flt); ++i) {
              if (elem_less(tCcRow_flt(i), residue_tCcRow)) {
                int global_m = int(get<0>(tCcRow_flt(i)));
                int global_n = int(get<1>(tCcRow_flt(i)));
                int local_m = global_m - tile_m * tile_extent_m;
                int local_n = global_n - tile_n * tile_extent_n;
                auto* output_ptr = &gRow_l(local_m, local_n, tile_l);
                reduce_output(output_ptr, convert_output(tCrRow_flt(i)));
              } 
            }
          }
        }
        sync_fn();
      }

      //
      // 2. One warp in M, skip threadblock smem reduction
      //
      else if constexpr (decltype(size<0>(warp_layout_MN))::value <= 1) {
        // Dump warp reduction to gmem workspace
        using ElementGmem = cute::conditional_t<FinalReduction, ElementCompute volatile, ElementCompute>;
        Tensor tCgBuf = sm90_partition_for_epilogue<ReferenceSrc>(gBuf_ml(_,_,m,l), epi_tile, tiled_copy, thread_idx);

        if constexpr (SwapShuffle) {
          Tensor tCrRow_flt = filter(tCrRow);
          Tensor tCgBuf_flt = recast<ElementGmem>(filter(tCgBuf));
          auto FltFrgSizePerLaneM = size(tCrRow_flt) / size<0>(lane_layout_MN);
          Tensor tCgBuf_flt_ = logical_divide(tCgBuf_flt, FltFrgSizePerLaneM);               // (FltFrgSizePerLaneM, M)
          Tensor tCrRow_flt_ = logical_divide(tCrRow_flt, FltFrgSizePerLaneM);               // (FltFrgSizePerLaneM, M)
          copy_aligned(tCrRow_flt_(_,_0{}), tCgBuf_flt_(_,lane_m));
        }
        else {
          if (is_reduced_lane) {
            copy_aligned(tCrRow, recast<ElementGmem>(tCgBuf));
          }
        }
        sync_fn();
      }

      //
      // 2. Multiple warps in M, do threadblock smem reduction
      //
      else {
        Tensor sBuf = make_tensor(make_smem_ptr<ElementCompute>(raw_pointer_cast(smem_buffer.data())), sBuf_layout);
        static_assert(decltype(cosize(sBuf.layout()))::value * sizeof(ElementCompute) <=
                      decltype(cosize(smem_buffer.layout()))::value * sizeof(typename remove_cvref_t<STensor>::value_type),
                      "smem reduction buffer not large enough, use a larger epilogue tile");
        sync_fn();

        // Dump warp reduction to smem workspace
        Tensor tCsBuf = sm90_partition_for_epilogue<ReferenceSrc>(sBuf(_,_,get<0>(warp_mn)), epi_tile, tiled_copy, thread_idx);

        if constexpr (SwapShuffle) {
          Tensor tCrRow_flt = filter(tCrRow);
          Tensor tCsBuf_flt = filter(tCsBuf);
          auto FltFrgSizePerLaneM = size(tCrRow_flt) / size<0>(lane_layout_MN);
          Tensor tCsBuf_flt_ = logical_divide(tCsBuf_flt, FltFrgSizePerLaneM);               // (FltFrgSizePerLaneM, M)
          Tensor tCrRow_flt_ = logical_divide(tCrRow_flt, FltFrgSizePerLaneM);               // (FltFrgSizePerLaneM, M)
          copy_aligned(tCrRow_flt_(_,_0{}), tCsBuf_flt_(_,lane_m));
        }
        else {
          if (is_reduced_lane) {
            copy_aligned(tCrRow, tCsBuf);
          }
        }
        sync_fn();

        constexpr int SmemFragSize = cute::max(size_t{1}, sizeof(uint32_t) / sizeof(ElementCompute));
        using FragmentSmem = Array<ElementCompute, SmemFragSize>;
        using VectorSmem = uint_bit_t<sizeof_bits_v<FragmentSmem>>;
        using ReduceSmem = GmemReduceFn<FragmentSmem>;
        ReduceSmem reduce_smem{};

        Tensor sBuf_frg = recast<FragmentSmem>(filter_zeros(sBuf));
        Tensor sBuf_vec = recast<VectorSmem>(filter_zeros(sBuf));
        constexpr int FragsPerRow = decltype(size<1>(sBuf_frg))::value;

        constexpr int RowNum = decltype(size<0>(warp_layout_MN))::value;
        using FragmentSmemArray = Array<FragmentSmem, RowNum>;

        // Do the threadblock smem reduction
        using VectorGmem = cute::conditional_t<FinalReduction, VectorSmem volatile, VectorSmem>;
        Tensor gBuf_vec = recast<VectorGmem>(filter(gBuf_ml(_,_,m,l)));
        CUTLASS_PRAGMA_UNROLL
        for (int frg_idx = thread_idx; frg_idx < FragsPerRow; frg_idx += size(tiled_copy)) {
          FragmentSmemArray frg_smem;

          CUTLASS_PRAGMA_UNROLL
          for (int reduction_rows = 0; reduction_rows < RowNum; ++reduction_rows) {
            int FragsCurrRows = reduction_rows * FragsPerRow;
            frg_smem[reduction_rows] = sBuf_frg(FragsCurrRows + frg_idx);
          }

          CUTLASS_PRAGMA_UNROLL
          for (int reduction_rows = RowNum / 2; reduction_rows > 0; reduction_rows /= 2) {
            CUTLASS_PRAGMA_UNROLL
            for (int row_idx = 0; row_idx < reduction_rows; ++row_idx) {
              frg_smem[row_idx] = reduce_smem(frg_smem[row_idx], frg_smem[row_idx + reduction_rows]);
            }
          }
          gBuf_vec(frg_idx) = reinterpret_cast<VectorSmem&>(frg_smem[0]);
        }
        sync_fn();
      }

      //
      // 3. Increment atomic counters to signal final gmem reduction
      //
      if constexpr (not IsAtomic && FinalReduction) {
        // Ensure gmem writes are visible to other threads before incrementing counter
        threadfence();
        sync_fn();
        int axis_index = int(n);
        int total_tiles = size<2>(gBuf_ml) * size<3>(gBuf_ml);
        do_final_reduction = detail::xe_signal_final_reduction(
          params.tile_counters,
          axis_index,
          total_tiles,
          thread_idx,
          raw_pointer_cast(smem_buffer.data()),
          sync_fn);
      }
    }

    CUTLASS_DEVICE void
    end() {
      //
      // 4. Do final gmem reduction if necessary
      //
      if constexpr (not IsAtomic && FinalReduction) {
        if (not do_final_reduction) {
          return;
        }

        auto& [ref_src, tCrRow, tCcRow, gRow_l, cRow, gBuf_ml, sBuf_layout,
          lane_layout_MN, lane_mn, warp_layout_MN, warp_mn,
          tile_coord_mnkl, residue_cRow, residue_tCcRow, epi_tile, tiled_copy, thread_idx] = args_tuple;

        using ReduceOutput = GmemReduceFn<ElementCompute>;
        using ConvertOutput = NumericConverter<ElementOutput, ElementCompute, RoundStyle>;
        ReduceOutput reduce_output{};
        ConvertOutput convert_output{};

        // Reduction over batches
        if (size<2>(stride(gRow_l)) == 0) {
          CUTLASS_PRAGMA_NO_UNROLL
          for (int n = thread_idx; n < size<1>(gBuf_ml); n += size(tiled_copy)) {
            Tensor tRgBuf_ml = gBuf_ml(_0{},n,_,_);
            ElementCompute output = tRgBuf_ml(_0{});
            CUTLASS_PRAGMA_NO_UNROLL
            for (int ml = 1; ml < size(tRgBuf_ml); ++ml) {
              output = reduce_output(output, tRgBuf_ml(ml));
            }
            if (elem_less(cRow(_0{},n), residue_cRow)) {
              gRow_l(_0{},n,_0{}) = convert_output(output);
            }
          }
        }
        // No reduction over batches
        else {
          CUTLASS_PRAGMA_NO_UNROLL
          for (int n = thread_idx; n < size<1>(gBuf_ml); n += size(tiled_copy)) {
            bool do_store = elem_less(cRow(_0{},n), residue_cRow);
            CUTLASS_PRAGMA_NO_UNROLL
            for (int l = 0; l < size<3>(gBuf_ml); ++l) {
              Tensor tRgBuf_m = gBuf_ml(_0{},n,_,l);
              ElementCompute output = tRgBuf_m(_0{});
              CUTLASS_PRAGMA_NO_UNROLL
              for (int m = 1; m < size(tRgBuf_m); ++m) {
                output = reduce_output(output, tRgBuf_m(m));
              }
              if (do_store) {
                gRow_l(_0{},n,l) = convert_output(output);
              }
            }
          }
        }

      }
    }
  };

  template <
    bool ReferenceSrc, // do register tensors reference the src or dst layout of the tiled copy
    class... Args
  >
  CUTLASS_DEVICE auto
  get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const& args) {
    Layout ref_layout_MN = [&] () {
      auto mn_shape = shape(typename decltype(args.tiled_copy)::Tiler_MN{});
      if constexpr (ReferenceSrc) { return right_inverse(args.tiled_copy.get_layoutS_TV()).with_shape(mn_shape); }
      else                        { return right_inverse(args.tiled_copy.get_layoutD_TV()).with_shape(mn_shape); }
    }();                                                                                         // tile_mn -> tv_idx

    // Get the MN layout + coord of lanes to determine shuffle reduction iterations
    using _W = Int<decltype(args.tiled_copy)::TiledNumThr::value / NumThreadsPerWarp>;
    Layout tv2lane = Layout<Shape<Int<NumThreadsPerWarp>,_W,_1>,Stride<_1,_0,_0>>{};            //   tv_idx -> lane_idx
    Layout ref2lane = composition(tv2lane, ref_layout_MN);                                      //  tile_mn -> lane_idx
    Layout lane_layout_MN = make_layout(filter(get<0>(ref2lane)), filter(get<1>(ref2lane)));    //  lane_mn -> lane_idx
    Layout inv_lane_layout_MN = right_inverse(lane_layout_MN);                                  // lane_idx -> lane_mn
    int lane_idx = canonical_lane_idx();
    auto lane_mn = idx2crd(inv_lane_layout_MN(lane_idx), shape(lane_layout_MN));

    // Get the MN layout + coord of warps to determine smem reduction iterations
    Layout tv2warp = Layout<Shape<Int<NumThreadsPerWarp>,_W,_1>,Stride<_0,_1,_0>>{};            //   tv_idx -> warp_idx
    Layout ref2warp = composition(tv2warp, ref_layout_MN);                                      //  tile_mn -> warp_idx
    Layout warp_layout_MN = make_layout(filter(get<0>(ref2warp)), filter(get<1>(ref2warp)));    //  warp_mn -> warp_idx
    Layout inv_warp_layout_MN = right_inverse(warp_layout_MN);                                  // warp_idx -> warp_mn

    int warp_idx = args.thread_idx / NumThreadsPerWarp;
    auto warp_mn = idx2crd(inv_warp_layout_MN(warp_idx), shape(warp_layout_MN));

    // Partition output gmem and register tensors
    auto [tile_M, tile_N, tile_K] = args.tile_shape_mnk;
    auto [M, N, K, L] = args.problem_shape_mnkl;
    auto [m, n, k, l] = args.tile_coord_mnkl;

    Tensor mRow = make_tensor(make_gmem_ptr<ElementOutput>(params.ptr_row), make_shape(M,N,L), params.dRow); // (M,N,L)
    Tensor gRow_l = local_tile(mRow, take<0,2>(args.tile_shape_mnk), make_coord(m,n,_));             // (CTA_M,CTA_N,L)
    Tensor tCgRow = sm90_partition_for_epilogue<ReferenceSrc>(                         // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)
      gRow_l(_,_,l), args.epi_tile, args.tiled_copy, args.thread_idx);
    Tensor tCrRow = make_tensor_like<ElementCompute>(tCgRow);                          // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)

    fill(tCrRow, params.reduction_identity);

    // Partition gmem+smem reduction buffer tensors
    Layout gBuf_layout = make_layout(take<0,2>(args.tile_shape_mnk), make_stride(_0{}, _1{}));
    auto block_shape = ceil_div(make_shape(M,N,L), shape(gBuf_layout)); // (M_CNT, N_CNT, L_CNT)

    // Let the M_CNT (the num of partial reduction results) become the outer mode
    Layout block_layout = make_layout(block_shape, make_stride(get<1>(block_shape), _1{}, get<0>(block_shape) * get<1>(block_shape)));
    Layout mBuf_layout = blocked_product(gBuf_layout, block_layout);
    Tensor mBuf = make_tensor(make_gmem_ptr(params.reduction_buffer), mBuf_layout);                // (ceil_M,ceil_N,L)
    Tensor gBuf_ml = local_tile(mBuf, take<0,2>(args.tile_shape_mnk), make_coord(_,n,_));     // (CTA_M,CTA_N,REST_M,L)
    Layout sBuf_layout = blocked_product(gBuf_layout,                                          // (CTA_M,CTA_N,WARPS_M)
      make_layout(make_shape(_1{},_1{},size<0>(warp_layout_MN))));

    // For row reduction, use full problem dimensions for residue checking
    auto residue_mnk = make_coord(size<0>(args.problem_shape_mnkl), size<1>(args.problem_shape_mnkl));
    auto args_tuple = make_tuple(
        bool_constant<ReferenceSrc>{}, cute::move(tCrRow), args.tCcD, gRow_l, args.cD, gBuf_ml, sBuf_layout,
        lane_layout_MN, lane_mn, warp_layout_MN, warp_mn,
        args.tile_coord_mnkl, args.residue_cD, residue_mnk, args.epi_tile, args.tiled_copy, args.thread_idx);
    return ConsumerStoreCallbacks<decltype(args_tuple)>(cute::move(args_tuple), params);
  }
};


} // namespace cutlass::epilogue::fusion
