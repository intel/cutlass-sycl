/***************************************************************************************************
 * Copyright (c) 2024 - 2024 Codeplay Software Ltd. All rights reserved.
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

/*! \file
  \brief Fusion callbacks specializations for the Intel Xe epilogue
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"

#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/fusion/callbacks.hpp"
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/xe_visitor.hpp"
#include "cutlass/epilogue/fusion/xe_visitor_legacy.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_store_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_compute_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/xe_visitor_softmax.hpp"
#include "cutlass/epilogue/fusion/xe_visitor_splitk.hpp"
#include "cutlass/epilogue/fusion/xe_visitor_topk_softmax.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::epilogue::fusion {

/////////////////////////////////////////////////////////////////////////////////////////////////


template <
  class ElementOutput_,
  class ElementCompute_,
  class ElementSource_,
  class ElementScalar_,
  FloatRoundStyle RoundStyle_,
  class CtaTileShapeMNK_,
  class EpilogueTile_
>
struct FusionCallbacks<
    epilogue::IntelXeGeneric,
    fusion::LinearCombination<ElementOutput_, ElementCompute_, ElementSource_, ElementScalar_, RoundStyle_>,
    CtaTileShapeMNK_,
    EpilogueTile_
> : Sm90LinearCombination<typename cutlass::detail::get_unpacked_element_type<ElementOutput_>::type, ElementCompute_, ElementSource_, ElementScalar_, RoundStyle_> {

  using Impl = Sm90LinearCombination<typename cutlass::detail::get_unpacked_element_type<ElementOutput_>::type, ElementCompute_, ElementSource_, ElementScalar_, RoundStyle_>;
  using ElementOutput = ElementOutput_;
  using ElementCompute = ElementCompute_;
  using ElementSource = ElementSource_;
  using ElementScalar = ElementScalar_;
  using Operation = fusion::LinearCombination<ElementOutput, ElementCompute, ElementSource_, ElementScalar, RoundStyle_>;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;

    using StrideAlpha = Stride<_0,_0,int64_t>;
    using StrideBeta  = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta  dBeta  = {_0{}, _0{}, 0};

    operator typename Impl::Arguments() const {
      return
        {    // ternary op : beta * C + (alpha * acc)
          {{beta}, {beta_ptr}, {dBeta}}, // leaf args : beta
          {},                   // leaf args : C
          {                     // binary op : alpha * acc
            {{alpha}, {alpha_ptr}, {dAlpha}}, // leaf args : alpha
            {},                     // leaf args : acc
            {}                  // binary args : multiplies
          },                    // end binary op
          {} // ternary args : multiply_add
        };   // end ternary op
    }
  };

  // Check if the epilogue is an identity operation (D = acc).
  // True when alpha == 1 and beta == 0.
  CUTLASS_DEVICE bool is_identity() const {
    using OuterVisitorImpl = typename Impl::Impl;
    auto const& outer_ops = static_cast<OuterVisitorImpl const&>(*this).ops;
    if (!get<0>(outer_ops).is_zero()) return false;
    auto const& inner_tree = get<2>(outer_ops);
    using InnerTreeType = cute::remove_cvref_t<decltype(inner_tree)>;
    using InnerVisitorImpl = typename InnerTreeType::Impl;
    auto const& alpha_op = get<0>(static_cast<InnerVisitorImpl const&>(inner_tree).ops);
    return alpha_op.scalar == ElementScalar(1);
  }

  // Ctor inheritance
  using Impl::Impl;
};

template <
  template <class> class ActivationFn_,
  class ElementOutput_,
  class ElementCompute_,
  class ElementSource_,
  class ElementScalar_,
  FloatRoundStyle RoundStyle_,
  class CtaTileShapeMNK_,
  class EpilogueTile_
>
struct FusionCallbacks<
    epilogue::IntelXeGeneric,
    fusion::LinCombEltAct<ActivationFn_, ElementOutput_, ElementCompute_, ElementSource_, ElementScalar_, RoundStyle_>,
    CtaTileShapeMNK_,
    EpilogueTile_
> : Sm90LinCombEltAct<ActivationFn_, ElementOutput_, ElementCompute_, ElementSource_, ElementScalar_, RoundStyle_> {

  using Impl = Sm90LinCombEltAct<ActivationFn_, typename cutlass::detail::get_unpacked_element_type<ElementOutput_>::type, ElementCompute_, ElementSource_, ElementScalar_, RoundStyle_>;
  using ElementOutput = ElementOutput_;
  using ElementCompute = ElementCompute_;
  using ElementSource = ElementSource_;
  using ElementScalar = ElementScalar_;
  using Operation = fusion::LinCombEltAct<ActivationFn_, ElementOutput_, ElementCompute_, ElementSource_, ElementScalar_, RoundStyle_>;

  struct Arguments {
    ElementScalar_ alpha = ElementScalar_(1);
    ElementScalar_ beta = ElementScalar_(0);
    ElementScalar_ const* alpha_ptr = nullptr;
    ElementScalar_ const* beta_ptr = nullptr;

    using StrideAlpha = Stride<_0,_0,int64_t>;
    using StrideBeta  = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta  dBeta  = {_0{}, _0{}, 0};

    using ActivationArguments = typename Sm90Compute<ActivationFn_, ElementOutput_, ElementCompute_, RoundStyle_>::Arguments;
    ActivationArguments activation = ActivationArguments();

    operator typename Impl::Arguments() const {
      return
              {    // unary op: activation(beta * C + (alpha * acc))
                        {    // ternary op : beta * C + (alpha * acc)
                          {{beta}, {beta_ptr}, {dBeta}}, // leaf args : beta
                          {},                   // leaf args : C
                          {                     // binary op : alpha * acc
                                        {{alpha}, {alpha_ptr}, {dAlpha}}, // leaf args : alpha
                                        {},                     // leaf args : acc
                                        {}                  // binary args : multiplies
                          },                    // end binary op
                          {} // ternary args : multiply_add
                        },   // end ternary op
                        activation // unary args: activation
                };   // end unary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

// D = splitk(alpha * acc + beta * C)
template<
  // int FragmentSize,
  class CtaTileShapeMNK,
  class EpilogueTile,
  class ElementOutput,
  class ElementCompute,
  class CopyOpR2G,
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using XeLinCombSplitK =
  Sm90EVT<XeSplitK<CtaTileShapeMNK, EpilogueTile, ElementOutput, ElementCompute, CopyOpR2G, RoundStyle>, // splitk(beta * C + (alpha * acc))
    Sm90LinearCombination<ElementCompute, ElementCompute, ElementSource, ElementScalar, RoundStyle> // beta * C + (alpha * acc)
  >;

// TODO: update split-k for new epilogues
template <
  // int FragmentSize,
  class ElementOutput_,
  class ElementCompute_,
  class ElementSource_,
  class ElementScalar_,
  class CopyOpR2G_,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile
>
struct FusionCallbacks<
    epilogue::IntelXeXMX16,
    fusion::LinCombSplitK<ElementOutput_, ElementCompute_, CopyOpR2G_, ElementSource_, ElementScalar_, RoundStyle>,
    CtaTileShapeMNK,
    EpilogueTile
> : XeLinCombSplitK<CtaTileShapeMNK, EpilogueTile, ElementOutput_, ElementCompute_, CopyOpR2G_, ElementSource_, ElementScalar_, RoundStyle> {

  using ElementOutput = ElementOutput_;
  using ElementCompute = ElementCompute_;
  using ElementSource = ElementSource_;
  using ElementScalar = ElementScalar_;
  using Impl = XeLinCombSplitK<CtaTileShapeMNK, EpilogueTile, typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, ElementCompute, CopyOpR2G_, ElementSource, ElementScalar, RoundStyle>;
  using Operation = fusion::LinCombSplitK<ElementOutput_, ElementCompute, CopyOpR2G_, ElementSource, ElementScalar, RoundStyle>;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;
    ElementOutput* output_ptr = nullptr;
    ElementOutput *output_ptr1 = nullptr;
    ElementOutput *output_ptr2 = nullptr;
    size_t NUM_HEAD = 0;
    size_t NOPE_DIM = 0;
    size_t ROPE_DIM = 0;
    operator typename Impl::Arguments() const {
      return
        {    // unary op: activation(beta * C + (alpha * acc))
          {    // ternary op : beta * C + (alpha * acc)
            {{beta}, {beta_ptr}}, // leaf args : beta
            {},                   // leaf args : C
            {                     // binary op : alpha * acc
              {{alpha}, {alpha_ptr}}, // leaf args : alpha
              {},                     // leaf args : acc
              {}                  // binary args : multiplies
            },                    // end binary op
            {} // ternary args : multiply_add
          },   // end ternary op
          {output_ptr, output_ptr1, output_ptr2, NUM_HEAD, NOPE_DIM, ROPE_DIM} // unary args: activation
        };   // end unary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

// TODO: update softmax for new epilogues
// D = softmax(alpha * acc + beta * C)
template<
  // int FragmentSize,
  class CtaTileShapeMNK,
  class EpilogueTile,
  class ElementOutput,
  class ElementCompute,
  class CopyOpR2G,
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using XeLinCombSoftmaxRow =
  Sm90EVT<XeSoftmaxRowReduction<CtaTileShapeMNK, EpilogueTile, ElementOutput, ElementCompute, CopyOpR2G, RoundStyle>, // softmax(beta * C + (alpha * acc))
    Sm90LinearCombination<ElementCompute, ElementCompute, ElementSource, ElementScalar, RoundStyle> // beta * C + (alpha * acc)
  >;

template <
  // int FragmentSize,
  class ElementOutput_,
  class ElementCompute_,
  class ElementSource_,
  class ElementScalar_,
  class CopyOpR2G_,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile
>
struct FusionCallbacks<
    epilogue::IntelXeXMX16,
    fusion::LinCombSoftmaxRow<ElementOutput_, ElementCompute_, CopyOpR2G_, ElementSource_, ElementScalar_, RoundStyle>,
    CtaTileShapeMNK,
    EpilogueTile
> : XeLinCombSoftmaxRow<CtaTileShapeMNK, EpilogueTile, ElementOutput_, ElementCompute_, CopyOpR2G_, ElementSource_, ElementScalar_, RoundStyle> {

  using ElementOutput = ElementOutput_;
  using ElementCompute = ElementCompute_;
  using ElementSource = ElementSource_;
  using ElementScalar = ElementScalar_;
  using Impl = XeLinCombSoftmaxRow<CtaTileShapeMNK, EpilogueTile, typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, ElementCompute, CopyOpR2G_, ElementSource, ElementScalar, RoundStyle>;
  using Operation = fusion::LinCombSoftmaxRow<ElementOutput_, ElementCompute, CopyOpR2G_, ElementSource, ElementScalar, RoundStyle>;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;
    ElementOutput* output_ptr = nullptr;

    operator typename Impl::Arguments() const {
      return
        {    // unary op: activation(beta * C + (alpha * acc))
          {    // ternary op : beta * C + (alpha * acc)
            {{beta}, {beta_ptr}}, // leaf args : beta
            {},                   // leaf args : C
            {                     // binary op : alpha * acc
              {{alpha}, {alpha_ptr}}, // leaf args : alpha
              {},                     // leaf args : acc
              {}                  // binary args : multiplies
            },                    // end binary op
            {} // ternary args : multiply_add
          },   // end ternary op
          {output_ptr} // unary args: activation
        };   // end unary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

// XeLinCombDeEltActLegacy - uses old XeAuxLoad for backward compatibility with examples
// EVT tests bypass this and directly instantiate XeAuxLoad via Python code generation
template<
  class StrideAux,
  class CopyOpG2R,
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  class ElementAux = ElementOutput,
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using XeLinCombDeEltActLegacy =
  Sm90EVT<Sm90Compute<ActivationFn, ElementOutput, ElementCompute, RoundStyle>, // activation(beta * C + (alpha * acc), aux)
    Sm90LinearCombination<ElementCompute, ElementCompute, ElementSource, ElementScalar, RoundStyle>, // beta * C + (alpha * acc)
    XeAuxLoadLegacy<ElementAux, StrideAux, CopyOpG2R> // aux (legacy version with explicit CopyOpG2R)
  >;

// Z = Aux
// dY = alpha * acc + beta * C
// D = activation(dY, Z)
//
// Specialization with explicit CopyOpG2R - uses legacy XeAuxLoad for backward compatibility
template <
  class GmemLayoutTagAux,
  template <class> class ActivationFn,
  class ElementOutput_,
  class ElementCompute_,
  class ElementAux,
  class ElementSource,
  class ElementScalar,
  int AlignmentAux,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile,
  class CopyOpG2R
>
struct FusionCallbacks<
    epilogue::IntelXeGeneric,
    fusion::LinCombDeEltAct<
      GmemLayoutTagAux, ActivationFn, ElementOutput_, ElementCompute_,
      ElementAux, ElementSource, ElementScalar, AlignmentAux, RoundStyle
    >,
    CtaTileShapeMNK,
    EpilogueTile,
    CopyOpG2R
> : XeLinCombDeEltActLegacy<
      cutlass::gemm::TagToStrideC_t<GmemLayoutTagAux>, CopyOpG2R, ActivationFn, ElementOutput_,
      ElementCompute_, ElementAux, ElementSource, ElementScalar, RoundStyle
    > {

  using ElementOutput = ElementOutput_;
  using ElementCompute = ElementCompute_;

  using Impl =
    XeLinCombDeEltActLegacy<
      cutlass::gemm::TagToStrideC_t<GmemLayoutTagAux>, CopyOpG2R, ActivationFn, ElementOutput,
      ElementCompute, ElementAux, ElementSource, ElementScalar, RoundStyle
    >;
  using Operation =
    fusion::LinCombDeEltAct<
      GmemLayoutTagAux, ActivationFn, ElementOutput, ElementCompute,
      ElementAux, ElementSource, ElementScalar, AlignmentAux, RoundStyle
    >;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;

    using StrideAlpha = Stride<_0,_0,int64_t>;
    using StrideBeta  = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta  dBeta  = {_0{}, _0{}, 0};

    using ActivationArguments = typename Sm90Compute<ActivationFn, ElementOutput, ElementCompute, RoundStyle>::Arguments;
    ActivationArguments activation = ActivationArguments();

    using StrideAux = cutlass::gemm::TagToStrideC_t<GmemLayoutTagAux>;
    ElementAux const* aux_ptr = nullptr;
    StrideAux dAux = {};

    operator typename Impl::Arguments() const {
      return
        {    // binary op : activation(beta * C + (alpha * acc), aux)
          {                  // ternary op : beta * C + (alpha * acc)
            {{beta}, {beta_ptr}, {dBeta}}, // leaf args : beta
            {},                   // leaf args : C
            {                     // binary op : alpha * acc
              {{alpha}, {alpha_ptr}, {dAlpha}}, // leaf args : alpha
              {},                     // leaf args : acc
              {}                  // binary args : multiplies
            },                    // end binary op
            {}               // ternary args : multiply_add
          },                 // end ternary op
          {aux_ptr, ElementAux(0), dAux}, // leaf args : aux
          activation // binary args : activation
        };   // end binary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

// Temporary: provide default G2R operation for LinCombDeEltAct, for CollectiveBuilder.
// Used by examples and non-EVT tests that don't specify CopyOpG2R template parameter
template <typename ElementSource>
using XeAuxLoadDefaultCopyOpG2R = conditional_t<sizeof_bits_v<ElementSource> == 32, XE_2D_U32x8x16_LD_N, XE_2D_U16x8x16_LD_N>;

template <
  class GmemLayoutTagAux, template <class> class ActivationFn,
  class ElementOutput_, class ElementCompute_, class ElementAux, class ElementSource, class ElementScalar,
  int AlignmentAux, FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK, class EpilogueTile
>
struct FusionCallbacks<
    epilogue::IntelXeGeneric,
    fusion::LinCombDeEltAct<
      GmemLayoutTagAux, ActivationFn, ElementOutput_, ElementCompute_,
      ElementAux, ElementSource, ElementScalar, AlignmentAux, RoundStyle
    >,
    CtaTileShapeMNK,
    EpilogueTile
> : FusionCallbacks<
    epilogue::IntelXeGeneric,
    fusion::LinCombDeEltAct<
      GmemLayoutTagAux, ActivationFn, ElementOutput_, ElementCompute_,
      ElementAux, ElementSource, ElementScalar, AlignmentAux, RoundStyle
    >,
    CtaTileShapeMNK,
    EpilogueTile,
    XeAuxLoadDefaultCopyOpG2R<ElementSource>
> {
  using FusionCallbacks<
    epilogue::IntelXeGeneric, fusion::LinCombDeEltAct<
      GmemLayoutTagAux, ActivationFn, ElementOutput_, ElementCompute_,
      ElementAux, ElementSource, ElementScalar, AlignmentAux, RoundStyle
    >,
    CtaTileShapeMNK, EpilogueTile, XeAuxLoadDefaultCopyOpG2R<ElementSource>
  >::FusionCallbacks;
};

/////////////////////////////////////////////////////////////////////////////////////////////////
// New XeLinCombDeEltAct - uses new XeAuxLoad with auto-deduced copy operation
/////////////////////////////////////////////////////////////////////////////////////////////////

// FusionCallbacks specialization for XeLinCombDeEltAct using new XeAuxLoad
template <
  class GmemLayoutTagAux,
  template <class> class ActivationFn,
  class ElementOutput_,
  class ElementCompute_,
  class ElementAux,
  class ElementSource,
  class ElementScalar,
  int AlignmentAux,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile
>
struct FusionCallbacks<
    epilogue::IntelXeGeneric,
    fusion::XeLinCombDeEltAct<
      GmemLayoutTagAux, ActivationFn, ElementOutput_, ElementCompute_,
      ElementAux, ElementSource, ElementScalar, AlignmentAux, RoundStyle
    >,
    CtaTileShapeMNK,
    EpilogueTile
> : Sm90EVT<
      Sm90Compute<ActivationFn, ElementOutput_, ElementCompute_, RoundStyle>,
      Sm90LinearCombination<ElementCompute_, ElementCompute_, ElementSource, ElementScalar, RoundStyle>,
      XeAuxLoad<ElementAux, cutlass::gemm::TagToStrideC_t<GmemLayoutTagAux>>
    > {

  using ElementOutput = ElementOutput_;
  using ElementCompute = ElementCompute_;

  using Impl =
    Sm90EVT<
      Sm90Compute<ActivationFn, ElementOutput, ElementCompute, RoundStyle>,
      Sm90LinearCombination<ElementCompute, ElementCompute, ElementSource, ElementScalar, RoundStyle>,
      XeAuxLoad<ElementAux, cutlass::gemm::TagToStrideC_t<GmemLayoutTagAux>>
    >;
  using Operation =
    fusion::XeLinCombDeEltAct<
      GmemLayoutTagAux, ActivationFn, ElementOutput, ElementCompute,
      ElementAux, ElementSource, ElementScalar, AlignmentAux, RoundStyle
    >;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;

    using StrideAlpha = Stride<_0,_0,int64_t>;
    using StrideBeta  = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta  dBeta  = {_0{}, _0{}, 0};

    using ActivationArguments = typename Sm90Compute<ActivationFn, ElementOutput, ElementCompute, RoundStyle>::Arguments;
    ActivationArguments activation = ActivationArguments();

    using StrideAux = cutlass::gemm::TagToStrideC_t<GmemLayoutTagAux>;
    ElementAux const* aux_ptr = nullptr;
    StrideAux dAux = {};

    operator typename Impl::Arguments() const {
      return
        {    // binary op : activation(beta * C + (alpha * acc), aux)
          {                  // ternary op : beta * C + (alpha * acc)
            {{beta}, {beta_ptr}, {dBeta}}, // leaf args : beta
            {},                   // leaf args : C
            {                     // binary op : alpha * acc
              {{alpha}, {alpha_ptr}, {dAlpha}}, // leaf args : alpha
              {},                     // leaf args : acc
              {}                  // binary args : multiplies
            },                    // end binary op
            {}               // ternary args : multiply_add
          },                 // end ternary op
          {aux_ptr, ElementAux(0), dAux}, // leaf args : aux
          activation // binary args : activation
        };   // end binary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// D = alpha * acc + beta * C + per-row bias
template <
  class ElementOutput_,
  class ElementCompute_,
  class ElementBias_,
  class ElementSource_,
  class ElementScalar_,
  int AlignmentBias_,
  FloatRoundStyle RoundStyle_,
  class CtaTileShapeMNK_,
  class EpilogueTile_
>
struct FusionCallbacks<
    epilogue::IntelXeGeneric,
    fusion::LinCombPerRowBias<ElementOutput_, ElementCompute_, ElementBias_, ElementSource_, ElementScalar_, AlignmentBias_, RoundStyle_>,
    CtaTileShapeMNK_,
    EpilogueTile_
> : Sm90LinCombPerRowBias<CtaTileShapeMNK_, ElementOutput_, ElementCompute_, ElementBias_, ElementSource_, ElementScalar_, AlignmentBias_, RoundStyle_> {

  using Impl = Sm90LinCombPerRowBias<
      CtaTileShapeMNK_,
      typename cutlass::detail::get_unpacked_element_type<ElementOutput_>::type,
      ElementCompute_, ElementBias_, ElementSource_, ElementScalar_,
      AlignmentBias_, RoundStyle_>;
  using ElementOutput = ElementOutput_;
  using ElementCompute = ElementCompute_;
  using ElementBias = ElementBias_;
  using ElementSource = ElementSource_;
  using ElementScalar = ElementScalar_;
  static constexpr int AlignmentBias = AlignmentBias_;
  using Operation = fusion::LinCombPerRowBias<ElementOutput_, ElementCompute_, ElementBias_, ElementSource_, ElementScalar_, AlignmentBias_, RoundStyle_>;

  struct Arguments {
    ElementScalar_ alpha = ElementScalar_(1);
    ElementScalar_ beta = ElementScalar_(0);
    ElementScalar_ const* alpha_ptr = nullptr;
    ElementScalar_ const* beta_ptr = nullptr;

    using StrideAlpha = Stride<_0,_0,int64_t>;
    using StrideBeta  = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta  dBeta  = {_0{}, _0{}, 0};

    using StrideBias = Stride<_1, _0, int64_t>;
    ElementBias const* bias_ptr = nullptr;
    StrideBias dBias = {};

    operator typename Impl::Arguments() const {
      return
        {     // ternary op : beta * C + (alpha * acc + bias)
          {{beta}, {beta_ptr}, {dBeta}}, // leaf args : beta
          {},                   // leaf args : C
          {                     // ternary op : alpha * acc + bias
            {{alpha}, {alpha_ptr}, {dAlpha}}, // leaf args : alpha
            {},                     // leaf args : acc
            {bias_ptr, ElementBias(0), dBias}, // leaf args : bias
            {}                  // ternary args : multiply_add
          },                    // end ternary op
          {} // ternary args : multiply_add
        };   // end ternary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

// D = alpha * acc + beta * C + per-column bias
// XeLinCombPerColBiasLegacy - uses XeRowBroadcastLegacy for backward compatibility with examples
// EVT tests bypass this and directly instantiate XeRowBroadcast via Python code generation
template<
  int StagesC,
  class CtaTileShapeMNK,
  class EpilogueTile,
  class ElementOutput,
  class ElementCompute,
  class ElementBias = ElementOutput,
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  int AlignmentBias = 128 / sizeof_bits_v<ElementBias>,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using XeLinCombPerColBiasLegacy =
  Sm90EVT<Sm90Compute<homogeneous_multiply_add, ElementOutput, ElementCompute, RoundStyle>, // beta * C + (alpha * acc + bias)
    Sm90ScalarBroadcast<ElementScalar, Stride<_0,_0,int64_t>>, // beta
    Sm90SrcFetch<ElementSource>, // C
    Sm90EVT<Sm90Compute<homogeneous_multiply_add, ElementCompute, ElementCompute, RoundStyle>, // alpha * acc + bias
      Sm90ScalarBroadcast<ElementScalar, Stride<_0,_0,int64_t>>, // alpha
      Sm90AccFetch, // acc
      XeRowBroadcastLegacy<0, CtaTileShapeMNK, ElementBias, ElementCompute, Stride<_0,_1,int64_t>, AlignmentBias> // bias (legacy version)
    >
  >;

template <
  class ElementOutput_,
  class ElementCompute_,
  class ElementBias_,
  class ElementSource_,
  class ElementScalar_,
  int AlignmentBias_,
  FloatRoundStyle RoundStyle_,
  class CtaTileShapeMNK_,
  class EpilogueTile_
>
struct FusionCallbacks<
    epilogue::IntelXeGeneric,
    fusion::LinCombPerColBias<ElementOutput_, ElementCompute_, ElementBias_, ElementSource_, ElementScalar_, AlignmentBias_, RoundStyle_>,
    CtaTileShapeMNK_,
    EpilogueTile_
> : XeLinCombPerColBiasLegacy<_1{} /* Stages */, CtaTileShapeMNK_, EpilogueTile_, ElementOutput_, ElementCompute_, ElementBias_, ElementSource_, ElementScalar_, AlignmentBias_, RoundStyle_> {

  using Impl = XeLinCombPerColBiasLegacy<
      _1{},
      CtaTileShapeMNK_,
      EpilogueTile_,
      typename cutlass::detail::get_unpacked_element_type<ElementOutput_>::type,
      ElementCompute_, ElementBias_, ElementSource_, ElementScalar_,
      AlignmentBias_, RoundStyle_>;
  using ElementOutput = ElementOutput_;
  using ElementCompute = ElementCompute_;
  using ElementBias = ElementBias_;
  using ElementSource = ElementSource_;
  using ElementScalar = ElementScalar_;
  static constexpr int AlignmentBias = AlignmentBias_;
  using Operation = fusion::LinCombPerColBias<ElementOutput_, ElementCompute_, ElementBias_, ElementSource_, ElementScalar_, AlignmentBias_, RoundStyle_>;

  struct Arguments {
    ElementScalar_ alpha = ElementScalar_(1);
    ElementScalar_ beta = ElementScalar_(0);
    ElementScalar_ const* alpha_ptr = nullptr;
    ElementScalar_ const* beta_ptr = nullptr;

    using StrideAlpha = Stride<_0,_0,int64_t>;
    using StrideBeta  = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta  dBeta  = {_0{}, _0{}, 0};

    using StrideBias = Stride<_0, _1, int64_t>;
    ElementBias const* bias_ptr = nullptr;
    StrideBias dBias = {};

    operator typename Impl::Arguments() const {
      return
        {     // ternary op : beta * C + (alpha * acc + bias)
          {{beta}, {beta_ptr}, {dBeta}}, // leaf args : beta
          {},                   // leaf args : C
          {                     // ternary op : alpha * acc + bias
            {{alpha}, {alpha_ptr}, {dAlpha}}, // leaf args : alpha
            {},                     // leaf args : acc
            {bias_ptr, ElementBias(0), dBias}, // leaf args : bias
            {}                  // ternary args : multiply_add
          },                    // end ternary op
          {} // ternary args : multiply_add
        };   // end ternary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

// XE-native TopK + Column Softmax: visitor does its own output store
template <
  class CtaTileShapeMNK,
  class EpilogueTile,
  class ElementOutput,
  class ElementCompute,
  class CopyOpR2G,
  int TopK,
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using XeLinCombTopKSoftmaxColImpl =
  Sm90EVT<XeTopKSoftmaxColReduction<TopK, CtaTileShapeMNK, EpilogueTile, ElementOutput, ElementCompute, CopyOpR2G, RoundStyle>,
    Sm90LinearCombination<ElementCompute, ElementCompute, ElementSource, ElementScalar, RoundStyle>
  >;

template <
  int TopK,
  class ElementOutput_,
  class ElementCompute_,
  class CopyOpR2G_,
  class ElementSource_,
  class ElementScalar_,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile
>
struct FusionCallbacks<
  epilogue::IntelXeXMX16,
  fusion::XeLinCombTopKSoftmaxCol<TopK, ElementOutput_, ElementCompute_, CopyOpR2G_, ElementSource_, ElementScalar_, RoundStyle>,
  CtaTileShapeMNK,
  EpilogueTile
> : XeLinCombTopKSoftmaxColImpl<CtaTileShapeMNK, EpilogueTile, ElementOutput_, ElementCompute_, CopyOpR2G_, TopK, ElementSource_, ElementScalar_, RoundStyle> {

  using ElementOutput = ElementOutput_;
  using ElementCompute = ElementCompute_;
  using ElementSource = ElementSource_;
  using ElementScalar = ElementScalar_;
  using Impl = XeLinCombTopKSoftmaxColImpl<CtaTileShapeMNK, EpilogueTile,
                                           typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type,
                                           ElementCompute, CopyOpR2G_, TopK, ElementSource, ElementScalar, RoundStyle>;
  using Operation = fusion::XeLinCombTopKSoftmaxCol<TopK, ElementOutput, ElementCompute, CopyOpR2G_, ElementSource, ElementScalar, RoundStyle>;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;
    ElementOutput* ptr_D = nullptr;

    operator typename Impl::Arguments() const {
      return
        {    // child op_0: Sm90LinearCombination args
          {    // ternary op : beta * C + (alpha * acc)
            {{beta}, {beta_ptr}}, // leaf args : beta
            {},                   // leaf args : C
            {                     // binary op : alpha * acc
              {{alpha}, {alpha_ptr}}, // leaf args : alpha
              {},                     // leaf args : acc
              {}                  // binary args : multiplies
            },                    // end binary op
            {} // ternary args : multiply_add
          },   // end ternary op
          {ptr_D} // node op_1: XeTopKSoftmaxColReduction args
        };
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// D = activation(alpha * acc + beta * C + per-row bias)

template <
 // int FragmentSize,
  //bool ReuseSmemC,
 // bool DelayTmaStore,
  template <class> class ActivationFn_,
  class ElementOutput_,
  class ElementCompute_,
  class ElementBias_,
  class ElementSource_,
  class ElementScalar_,
  int AlignmentBias_,
  FloatRoundStyle RoundStyle_,
  class CtaTileShapeMNK_,
  class EpilogueTile_
>
struct FusionCallbacks<
     epilogue::IntelXeGeneric,
    fusion::LinCombPerRowBiasEltAct<
      ActivationFn_, ElementOutput_, ElementCompute_, ElementBias_, ElementSource_, ElementScalar_, AlignmentBias_, RoundStyle_
    >,
    CtaTileShapeMNK_,
    EpilogueTile_
> : Sm90LinCombPerRowBiasEltAct<
      CtaTileShapeMNK_, ActivationFn_, ElementOutput_, ElementCompute_, ElementBias_, ElementSource_, ElementScalar_, AlignmentBias_, RoundStyle_
    > {

  using ElementOutput = ElementOutput_;
  using ElementCompute = ElementCompute_;
  using ElementBias = ElementBias_;
  using ElementSource = ElementSource_;
  using ElementScalar = ElementScalar_;
  static constexpr int AlignmentBias = AlignmentBias_;
  using Impl =
    Sm90LinCombPerRowBiasEltAct<
      CtaTileShapeMNK_, ActivationFn_, ElementOutput, ElementCompute, ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle_
    >;
  using Operation =
    fusion::LinCombPerRowBiasEltAct<
      ActivationFn_, ElementOutput, ElementCompute, ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle_
    >;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;

    using StrideAlpha = Stride<_0,_0,int64_t>;
    using StrideBeta  = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta  dBeta  = {_0{}, _0{}, 0};

    using StrideBias = Stride<_1,_0,int64_t>;
    ElementBias const* bias_ptr = nullptr;
    StrideBias dBias = {};

    using ActivationArguments = typename Sm90Compute<ActivationFn_, ElementOutput, ElementCompute, RoundStyle_>::Arguments;
    ActivationArguments activation = ActivationArguments();

    operator typename Impl::Arguments() const {
      return
        {    // unary op : activation(beta * C + (alpha * acc + bias))
          {    // ternary op : beta * C + (alpha * acc + bias)
            {{beta}, {beta_ptr}, {dBeta}}, // leaf args : beta
            {},                   // leaf args : C
            {                     // ternary op : alpha * acc + bias
              {{alpha}, {alpha_ptr}, {dAlpha}}, // leaf args : alpha
              {},                     // leaf args : acc
              {bias_ptr, ElementBias(0), dBias}, // leaf args : bias
              {}                  // ternary args : multiply_add
            },                    // end ternary op
            {} // ternary args : multiply_add
          },   // end ternary op
          activation // unary args : activation
        };   // end unary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

/////////////////////////////////////////////////////////////////////////////////////////////
// D = alpha * acc + beta * C, where beta and alpha can be vectors for each batch
template <
  class ElementOutput_,
  class ElementCompute_,
  class ElementSource_,
  class ElementScalar_,
  FloatRoundStyle RoundStyle_,
  class CtaTileShapeMNK_,
  class EpilogueTile_
>
struct FusionCallbacks<
    epilogue::IntelXeGenericGroup,
    fusion::LinearCombination<ElementOutput_, ElementCompute_, ElementSource_, ElementScalar_, RoundStyle_>,
    CtaTileShapeMNK_,
    EpilogueTile_
> : Sm90LinearCombinationPtrArray<typename cutlass::detail::get_unpacked_element_type<ElementOutput_>::type, ElementCompute_, ElementSource_, ElementScalar_, RoundStyle_> {

  using Impl = Sm90LinearCombinationPtrArray<typename cutlass::detail::get_unpacked_element_type<ElementOutput_>::type, ElementCompute_, ElementSource_, ElementScalar_, RoundStyle_>;
  using ElementOutput = ElementOutput_;
  using ElementCompute = ElementCompute_;
  using ElementSource = ElementSource_;
  using ElementScalar = ElementScalar_;
  using Operation = fusion::LinearCombination<ElementOutput, ElementCompute, ElementSource, ElementScalar, RoundStyle_>;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;
    ElementScalar const* const* alpha_ptr_array = nullptr;
    ElementScalar const* const* beta_ptr_array = nullptr;

    using StrideAlpha = Stride<_0,_0,int64_t>;
    using StrideBeta  = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta  dBeta  = {_0{}, _0{}, 0};

    operator typename Impl::Arguments() const {
      return
        {    // ternary op : beta * C + (alpha * acc)
          {{beta}, {beta_ptr}, {beta_ptr_array}, {dBeta}}, // leaf args : beta
          {},                   // leaf args : C
          {                     // binary op : alpha * acc
            {{alpha}, {alpha_ptr}, {alpha_ptr_array}, {dAlpha}}, // leaf args : alpha
            {},                     // leaf args : acc
            {}                  // binary args : multiplies
          },                    // end binary op
          {} // ternary args : multiply_add
        };   // end ternary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

// D = activation(alpha * acc + beta * C + per-row bias)
// Aux = alpha * acc + beta * C + per-row bias)
template<
  class CtaTileShapeMNK,
  class EpilogueTile,
  class StrideAux,
  class CopyOpR2G,
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  class ElementAux = ElementOutput,
  class ElementBias = ElementOutput,
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  int AlignmentAux = 128 / sizeof_bits_v<ElementAux>,
  int AlignmentBias = 128 / sizeof_bits_v<ElementBias>,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using XeLinCombPerRowBiasEltActAux =
  Sm90EVT<Sm90Compute<ActivationFn, ElementOutput, ElementCompute, RoundStyle>,
    Sm90EVT<XeAuxStore<ElementAux, StrideAux, CopyOpR2G>,
      Sm90LinCombPerRowBias<CtaTileShapeMNK, ElementCompute, ElementCompute, ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle>
    >
  >;

template <
  class GmemLayoutTagAux,
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  class ElementAux,
  class ElementBias,
  class ElementSource,
  class ElementScalar,
  int AlignmentAux,
  int AlignmentBias,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile,
  class CopyOpR2G
>
struct FusionCallbacks<
    epilogue::IntelXeGeneric,
    fusion::LinCombPerRowBiasEltActAux<
      GmemLayoutTagAux, ActivationFn, ElementOutput, ElementCompute,
      ElementAux, ElementBias, ElementSource, ElementScalar, AlignmentAux, AlignmentBias, RoundStyle
    >,
    CtaTileShapeMNK,
    EpilogueTile,
    CopyOpR2G
> : XeLinCombPerRowBiasEltActAux<
      CtaTileShapeMNK, EpilogueTile, cutlass::gemm::TagToStrideC_t<GmemLayoutTagAux>, CopyOpR2G, ActivationFn,
      ElementOutput, ElementCompute, ElementAux, ElementBias, ElementSource, ElementScalar, AlignmentAux, AlignmentBias, RoundStyle
    > {

  using Impl =
    XeLinCombPerRowBiasEltActAux<
      CtaTileShapeMNK, EpilogueTile, cutlass::gemm::TagToStrideC_t<GmemLayoutTagAux>, CopyOpR2G, ActivationFn,
      ElementOutput, ElementCompute, ElementAux, ElementBias, ElementSource, ElementScalar, AlignmentAux, AlignmentBias, RoundStyle
    >;
  using Operation =
    fusion::LinCombPerRowBiasEltActAux<
      GmemLayoutTagAux, ActivationFn,
      ElementOutput, ElementCompute, ElementAux, ElementBias, ElementSource, ElementScalar, AlignmentAux, AlignmentBias, RoundStyle
    >;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;

    using StrideAlpha = Stride<_0,_0,int64_t>;
    using StrideBeta  = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta  dBeta  = {_0{}, _0{}, 0};

    using StrideBias = Stride<_1,_0,int64_t>;
    ElementBias const* bias_ptr = nullptr;
    StrideBias dBias = {};

    using ActivationArguments = typename Sm90Compute<ActivationFn, ElementOutput, ElementCompute, RoundStyle>::Arguments;
    ActivationArguments activation = ActivationArguments();

    using StrideAux = cutlass::gemm::TagToStrideC_t<GmemLayoutTagAux>;
    ElementAux* aux_ptr = nullptr;
    StrideAux dAux = {};

    operator typename Impl::Arguments() const {
      return
        {    // unary op : activation(store(beta * C + (alpha * acc + bias)))
          {                 // unary op : store(beta * C + (alpha * acc + bias))
            {                  // ternary op : beta * C + (alpha * acc + bias)
              {{beta}, {beta_ptr}, {dBeta}}, // leaf args : beta
              {},                   // leaf args : C
              {                     // ternary op : alpha * acc + bias
                {{alpha}, {alpha_ptr}, {dAlpha}}, // leaf args : alpha
                {},                     // leaf args : acc
                {bias_ptr, ElementBias(0), dBias}, // leaf args : bias
                {}                  // ternary args : multiply_add
              },                    // end ternary op
              {}               // ternary args : multiply_add
            },                 // end ternary op
            {aux_ptr, ElementAux(0), dAux} // unary args : store
          },                // end unary op
          activation // unary args : activation
        };   // end unary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

/////////////////////////////////////////////////////////////////////////////////////////////////
// FusionCallbacks specialization for per-row bias with activation using XeColBroadcast
/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  template <class> class ActivationFn_,
  class ElementOutput_,
  class ElementCompute_,
  class ElementBias_,
  class ElementSource_,
  class ElementScalar_,
  int AlignmentBias_,
  FloatRoundStyle RoundStyle_,
  class CtaTileShapeMNK_,
  class EpilogueTile_
>
struct FusionCallbacks<
     epilogue::IntelXeGeneric,
    fusion::XeLinCombPerRowBiasEltAct<
      ActivationFn_, ElementOutput_, ElementCompute_, ElementBias_, ElementSource_, ElementScalar_, AlignmentBias_, RoundStyle_
    >,
    CtaTileShapeMNK_,
    EpilogueTile_
> : Sm90EVT<
      Sm90Compute<ActivationFn_, ElementOutput_, ElementCompute_, RoundStyle_>,
      Sm90EVT<Sm90Compute<homogeneous_multiply_add, ElementCompute_, ElementCompute_, RoundStyle_>,
        Sm90ScalarBroadcast<ElementScalar_, Stride<_0,_0,int64_t>>,
        Sm90SrcFetch<ElementSource_>,
        Sm90EVT<Sm90Compute<homogeneous_multiply_add, ElementCompute_, ElementCompute_, RoundStyle_>,
          Sm90ScalarBroadcast<ElementScalar_, Stride<_0,_0,int64_t>>,
          Sm90AccFetch,
          XeColBroadcast<0, CtaTileShapeMNK_, ElementBias_, ElementCompute_, Stride<_1,_0,int64_t>, AlignmentBias_>
        >
      >
    > {

  using ElementOutput = ElementOutput_;
  using ElementCompute = ElementCompute_;
  using ElementBias = ElementBias_;
  using ElementSource = ElementSource_;
  using ElementScalar = ElementScalar_;
  static constexpr int AlignmentBias = AlignmentBias_;
  using Impl =
    Sm90EVT<
      Sm90Compute<ActivationFn_, ElementOutput, ElementCompute, RoundStyle_>,
      Sm90EVT<Sm90Compute<homogeneous_multiply_add, ElementCompute, ElementCompute, RoundStyle_>,
        Sm90ScalarBroadcast<ElementScalar, Stride<_0,_0,int64_t>>,
        Sm90SrcFetch<ElementSource>,
        Sm90EVT<Sm90Compute<homogeneous_multiply_add, ElementCompute, ElementCompute, RoundStyle_>,
          Sm90ScalarBroadcast<ElementScalar, Stride<_0,_0,int64_t>>,
          Sm90AccFetch,
          XeColBroadcast<0, CtaTileShapeMNK_, ElementBias, ElementCompute, Stride<_1,_0,int64_t>, AlignmentBias_>
        >
      >
    >;
  using Operation =
    fusion::XeLinCombPerRowBiasEltAct<
      ActivationFn_, ElementOutput, ElementCompute, ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle_
    >;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;

    using StrideAlpha = Stride<_0,_0,int64_t>;
    using StrideBeta  = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta  dBeta  = {_0{}, _0{}, 0};

    using StrideBias = Stride<_1,_0,int64_t>;
    ElementBias const* bias_ptr = nullptr;
    StrideBias dBias = {};

    using ActivationArguments = typename Sm90Compute<ActivationFn_, ElementOutput, ElementCompute, RoundStyle_>::Arguments;
    ActivationArguments activation = ActivationArguments();

    operator typename Impl::Arguments() const {
      return
        {    // unary op : activation(beta * C + (alpha * acc + bias))
          {    // ternary op : beta * C + (alpha * acc + bias)
            {{beta}, {beta_ptr}, {dBeta}}, // leaf args : beta
            {},                   // leaf args : C
            {                     // ternary op : alpha * acc + bias
              {{alpha}, {alpha_ptr}, {dAlpha}}, // leaf args : alpha
              {},                     // leaf args : acc
              {bias_ptr, ElementBias(0), dBias}, // leaf args : bias
              {}                  // ternary args : multiply_add
            },                    // end ternary op
            {} // ternary args : multiply_add
          },   // end ternary op
          activation // unary args : activation
        };   // end unary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

/////////////////////////////////////////////////////////////////////////////////////////////////
// FusionCallbacks specialization for per-column bias with activation using XeRowBroadcast
/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  template <class> class ActivationFn_,
  class ElementOutput_,
  class ElementCompute_,
  class ElementBias_,
  class ElementSource_,
  class ElementScalar_,
  int AlignmentBias_,
  FloatRoundStyle RoundStyle_,
  class CtaTileShapeMNK_,
  class EpilogueTile_
>
struct FusionCallbacks<
     epilogue::IntelXeGeneric,
    fusion::XeLinCombPerColBiasEltAct<
      ActivationFn_, ElementOutput_, ElementCompute_, ElementBias_, ElementSource_, ElementScalar_, AlignmentBias_, RoundStyle_
    >,
    CtaTileShapeMNK_,
    EpilogueTile_
> : Sm90EVT<
      Sm90Compute<ActivationFn_, ElementOutput_, ElementCompute_, RoundStyle_>,
      Sm90EVT<Sm90Compute<homogeneous_multiply_add, ElementCompute_, ElementCompute_, RoundStyle_>,
        Sm90ScalarBroadcast<ElementScalar_, Stride<_0,_0,int64_t>>,
        Sm90SrcFetch<ElementSource_>,
        Sm90EVT<Sm90Compute<homogeneous_multiply_add, ElementCompute_, ElementCompute_, RoundStyle_>,
          Sm90ScalarBroadcast<ElementScalar_, Stride<_0,_0,int64_t>>,
          Sm90AccFetch,
          XeRowBroadcast<0, CtaTileShapeMNK_, ElementBias_, ElementCompute_, Stride<_0,_1,int64_t>, AlignmentBias_>
        >
      >
    > {

  using ElementOutput = ElementOutput_;
  using ElementCompute = ElementCompute_;
  using ElementBias = ElementBias_;
  using ElementSource = ElementSource_;
  using ElementScalar = ElementScalar_;
  static constexpr int AlignmentBias = AlignmentBias_;
  using Impl =
    Sm90EVT<
      Sm90Compute<ActivationFn_, ElementOutput, ElementCompute, RoundStyle_>,
      Sm90EVT<Sm90Compute<homogeneous_multiply_add, ElementCompute, ElementCompute, RoundStyle_>,
        Sm90ScalarBroadcast<ElementScalar, Stride<_0,_0,int64_t>>,
        Sm90SrcFetch<ElementSource>,
        Sm90EVT<Sm90Compute<homogeneous_multiply_add, ElementCompute, ElementCompute, RoundStyle_>,
          Sm90ScalarBroadcast<ElementScalar, Stride<_0,_0,int64_t>>,
          Sm90AccFetch,
          XeRowBroadcast<0, CtaTileShapeMNK_, ElementBias, ElementCompute, Stride<_0,_1,int64_t>, AlignmentBias_>
        >
      >
    >;
  using Operation =
    fusion::XeLinCombPerColBiasEltAct<
      ActivationFn_, ElementOutput, ElementCompute, ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle_
    >;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;

    using StrideAlpha = Stride<_0,_0,int64_t>;
    using StrideBeta  = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta  dBeta  = {_0{}, _0{}, 0};

    using StrideBias = Stride<_0,_1,int64_t>;
    ElementBias const* bias_ptr = nullptr;
    StrideBias dBias = {};

    using ActivationArguments = typename Sm90Compute<ActivationFn_, ElementOutput, ElementCompute, RoundStyle_>::Arguments;
    ActivationArguments activation = ActivationArguments();

    operator typename Impl::Arguments() const {
      return
        {    // unary op : activation(beta * C + (alpha * acc + bias))
          {    // ternary op : beta * C + (alpha * acc + bias)
            {{beta}, {beta_ptr}, {dBeta}}, // leaf args : beta
            {},                   // leaf args : C
            {                     // ternary op : alpha * acc + bias
              {{alpha}, {alpha_ptr}, {dAlpha}}, // leaf args : alpha
              {},                     // leaf args : acc
              {bias_ptr, ElementBias(0), dBias}, // leaf args : bias
              {}                  // ternary args : multiply_add
            },                    // end ternary op
            {} // ternary args : multiply_add
          },   // end ternary op
          activation // unary args : activation
        };   // end unary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// Callbacks for legacy epilogues (deprecated).

// D = activation(alpha * acc + beta * C + per-row bias)
// Aux = alpha * acc + beta * C + per-row bias)
template<
  class CtaTileShapeMNK,
  class EpilogueTile,
  class StrideAux,
  class CopyOpR2G,
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  class ElementAux = ElementOutput,
  class ElementBias = ElementOutput,
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  int AlignmentAux = 128 / sizeof_bits_v<ElementAux>,
  int AlignmentBias = 128 / sizeof_bits_v<ElementBias>,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using XeLinCombPerRowBiasEltActAuxLegacy =
  Sm90EVT<Sm90Compute<ActivationFn, ElementOutput, ElementCompute, RoundStyle>,
    Sm90EVT<XeAuxStoreLegacy<ElementAux, StrideAux, CopyOpR2G>,
      Sm90LinCombPerRowBias<CtaTileShapeMNK, ElementCompute, ElementCompute, ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle>
    >
  >;

template <
  class GmemLayoutTagAux,
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  class ElementAux,
  class ElementBias,
  class ElementSource,
  class ElementScalar,
  int AlignmentAux,
  int AlignmentBias,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile,
  class CopyOpR2G
>
struct FusionCallbacks<
    epilogue::IntelXeXMX16,
    fusion::LinCombPerRowBiasEltActAux<
      GmemLayoutTagAux, ActivationFn, ElementOutput, ElementCompute,
      ElementAux, ElementBias, ElementSource, ElementScalar, AlignmentAux, AlignmentBias, RoundStyle
    >,
    CtaTileShapeMNK,
    EpilogueTile,
    CopyOpR2G
> : XeLinCombPerRowBiasEltActAuxLegacy<
      CtaTileShapeMNK, EpilogueTile, cutlass::gemm::TagToStrideC_t<GmemLayoutTagAux>, CopyOpR2G, ActivationFn,
      ElementOutput, ElementCompute, ElementAux, ElementBias, ElementSource, ElementScalar, AlignmentAux, AlignmentBias, RoundStyle
    > {

  using Impl =
    XeLinCombPerRowBiasEltActAuxLegacy<
      CtaTileShapeMNK, EpilogueTile, cutlass::gemm::TagToStrideC_t<GmemLayoutTagAux>, CopyOpR2G, ActivationFn,
      ElementOutput, ElementCompute, ElementAux, ElementBias, ElementSource, ElementScalar, AlignmentAux, AlignmentBias, RoundStyle
    >;
  using Operation =
    fusion::LinCombPerRowBiasEltActAux<
      GmemLayoutTagAux, ActivationFn,
      ElementOutput, ElementCompute, ElementAux, ElementBias, ElementSource, ElementScalar, AlignmentAux, AlignmentBias, RoundStyle
    >;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;

    using StrideAlpha = Stride<_0,_0,int64_t>;
    using StrideBeta  = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta  dBeta  = {_0{}, _0{}, 0};

    using StrideBias = Stride<_1,_0,int64_t>;
    ElementBias const* bias_ptr = nullptr;
    StrideBias dBias = {};

    using ActivationArguments = typename Sm90Compute<ActivationFn, ElementOutput, ElementCompute, RoundStyle>::Arguments;
    ActivationArguments activation = ActivationArguments();

    using StrideAux = cutlass::gemm::TagToStrideC_t<GmemLayoutTagAux>;
    ElementAux* aux_ptr = nullptr;
    StrideAux dAux = {};

    operator typename Impl::Arguments() const {
      return
        {    // unary op : activation(store(beta * C + (alpha * acc + bias)))
          {                 // unary op : store(beta * C + (alpha * acc + bias))
            {                  // ternary op : beta * C + (alpha * acc + bias)
              {{beta}, {beta_ptr}, {dBeta}}, // leaf args : beta
              {},                   // leaf args : C
              {                     // ternary op : alpha * acc + bias
                {{alpha}, {alpha_ptr}, {dAlpha}}, // leaf args : alpha
                {},                     // leaf args : acc
                {bias_ptr, ElementBias(0), dBias}, // leaf args : bias
                {}                  // ternary args : multiply_add
              },                    // end ternary op
              {}               // ternary args : multiply_add
            },                 // end ternary op
            {aux_ptr, ElementAux(0), dAux} // unary args : store
          },                // end unary op
          activation // unary args : activation
        };   // end unary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

template <
  class ElementOutput_,
  class ElementCompute_,
  class ElementSource_,
  class ElementScalar_,
  FloatRoundStyle RoundStyle_,
  class CtaTileShapeMNK_,
  class EpilogueTile_
>
struct FusionCallbacks<
    epilogue::IntelXeXMX16,
    fusion::LinearCombination<ElementOutput_, ElementCompute_, ElementSource_, ElementScalar_, RoundStyle_>,
    CtaTileShapeMNK_,
    EpilogueTile_
> : FusionCallbacks<
    epilogue::IntelXeGeneric,
    fusion::LinearCombination<ElementOutput_, ElementCompute_, ElementSource_, ElementScalar_, RoundStyle_>,
    CtaTileShapeMNK_,
    EpilogueTile_
> {
  using FusionCallbacks<
    epilogue::IntelXeGeneric,
    fusion::LinearCombination<ElementOutput_, ElementCompute_, ElementSource_, ElementScalar_, RoundStyle_>,
    CtaTileShapeMNK_,
    EpilogueTile_>::FusionCallbacks;
};

template <
  template <class> class ActivationFn_,
  class ElementOutput_,
  class ElementCompute_,
  class ElementSource_,
  class ElementScalar_,
  FloatRoundStyle RoundStyle_,
  class CtaTileShapeMNK_,
  class EpilogueTile_
>
struct FusionCallbacks<
    epilogue::IntelXeXMX16,
    fusion::LinCombEltAct<ActivationFn_, ElementOutput_, ElementCompute_, ElementSource_, ElementScalar_, RoundStyle_>,
    CtaTileShapeMNK_,
    EpilogueTile_
> : FusionCallbacks<
    epilogue::IntelXeGeneric,
    fusion::LinCombEltAct<ActivationFn_, ElementOutput_, ElementCompute_, ElementSource_, ElementScalar_, RoundStyle_>,
    CtaTileShapeMNK_,
    EpilogueTile_
> {
  using FusionCallbacks<
    epilogue::IntelXeGeneric,
    fusion::LinCombEltAct<ActivationFn_, ElementOutput_, ElementCompute_, ElementSource_, ElementScalar_, RoundStyle_>,
    CtaTileShapeMNK_,
    EpilogueTile_>::FusionCallbacks;
};

template <
  class GmemLayoutTagAux,
  template <class> class ActivationFn,
  class ElementOutput_,
  class ElementCompute_,
  class ElementAux,
  class ElementSource,
  class ElementScalar,
  int AlignmentAux,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile,
  class CopyOpG2R
>
struct FusionCallbacks<
    epilogue::IntelXeXMX16,
    fusion::LinCombDeEltAct<
      GmemLayoutTagAux, ActivationFn, ElementOutput_, ElementCompute_,
      ElementAux, ElementSource, ElementScalar, AlignmentAux, RoundStyle
    >,
    CtaTileShapeMNK,
    EpilogueTile,
    CopyOpG2R
> : FusionCallbacks<
    epilogue::IntelXeGeneric,
    fusion::LinCombDeEltAct<
      GmemLayoutTagAux, ActivationFn, ElementOutput_, ElementCompute_,
      ElementAux, ElementSource, ElementScalar, AlignmentAux, RoundStyle
    >,
    CtaTileShapeMNK,
    EpilogueTile,
    CopyOpG2R
> {
  using FusionCallbacks<
    epilogue::IntelXeGeneric,
    fusion::LinCombDeEltAct<
      GmemLayoutTagAux, ActivationFn, ElementOutput_, ElementCompute_,
      ElementAux, ElementSource, ElementScalar, AlignmentAux, RoundStyle
    >,
    CtaTileShapeMNK,
    EpilogueTile,
    CopyOpG2R
  >::FusionCallbacks;
};

template <
  class ElementOutput_,
  class ElementCompute_,
  class ElementBias_,
  class ElementSource_,
  class ElementScalar_,
  int AlignmentBias_,
  FloatRoundStyle RoundStyle_,
  class CtaTileShapeMNK_,
  class EpilogueTile_
>
struct FusionCallbacks<
    epilogue::IntelXeXMX16,
    fusion::LinCombPerRowBias<ElementOutput_, ElementCompute_, ElementBias_, ElementSource_, ElementScalar_, AlignmentBias_, RoundStyle_>,
    CtaTileShapeMNK_,
    EpilogueTile_
> : FusionCallbacks<
    epilogue::IntelXeGeneric,
    fusion::LinCombPerRowBias<ElementOutput_, ElementCompute_, ElementBias_, ElementSource_, ElementScalar_, AlignmentBias_, RoundStyle_>,
    CtaTileShapeMNK_,
    EpilogueTile_
> {
  using FusionCallbacks<
    epilogue::IntelXeGeneric,
    fusion::LinCombPerRowBias<ElementOutput_, ElementCompute_, ElementBias_, ElementSource_, ElementScalar_, AlignmentBias_, RoundStyle_>,
    CtaTileShapeMNK_,
    EpilogueTile_
  >::FusionCallbacks;
};

template <
  class ElementOutput_,
  class ElementCompute_,
  class ElementBias_,
  class ElementSource_,
  class ElementScalar_,
  int AlignmentBias_,
  FloatRoundStyle RoundStyle_,
  class CtaTileShapeMNK_,
  class EpilogueTile_
>
struct FusionCallbacks<
    epilogue::IntelXeXMX16,
    fusion::LinCombPerColBias<ElementOutput_, ElementCompute_, ElementBias_, ElementSource_, ElementScalar_, AlignmentBias_, RoundStyle_>,
    CtaTileShapeMNK_,
    EpilogueTile_
> : FusionCallbacks<
    epilogue::IntelXeGeneric,
    fusion::LinCombPerColBias<ElementOutput_, ElementCompute_, ElementBias_, ElementSource_, ElementScalar_, AlignmentBias_, RoundStyle_>,
    CtaTileShapeMNK_,
    EpilogueTile_
> {
  using FusionCallbacks<
    epilogue::IntelXeGeneric,
    fusion::LinCombPerColBias<ElementOutput_, ElementCompute_, ElementBias_, ElementSource_, ElementScalar_, AlignmentBias_, RoundStyle_>,
    CtaTileShapeMNK_,
    EpilogueTile_
  >::FusionCallbacks;
};

template <
  template <class> class ActivationFn_,
  class ElementOutput_,
  class ElementCompute_,
  class ElementBias_,
  class ElementSource_,
  class ElementScalar_,
  int AlignmentBias_,
  FloatRoundStyle RoundStyle_,
  class CtaTileShapeMNK_,
  class EpilogueTile_
>
struct FusionCallbacks<
    epilogue::IntelXeXMX16,
    fusion::LinCombPerRowBiasEltAct<
      ActivationFn_, ElementOutput_, ElementCompute_, ElementBias_, ElementSource_, ElementScalar_, AlignmentBias_, RoundStyle_
    >,
    CtaTileShapeMNK_,
    EpilogueTile_
> : FusionCallbacks<
    epilogue::IntelXeGeneric,
    fusion::LinCombPerRowBiasEltAct<
      ActivationFn_, ElementOutput_, ElementCompute_, ElementBias_, ElementSource_, ElementScalar_, AlignmentBias_, RoundStyle_
    >,
    CtaTileShapeMNK_,
    EpilogueTile_
> {
  using FusionCallbacks<
    epilogue::IntelXeGeneric,
    fusion::LinCombPerRowBiasEltAct<
      ActivationFn_, ElementOutput_, ElementCompute_, ElementBias_, ElementSource_, ElementScalar_, AlignmentBias_, RoundStyle_
    >,
    CtaTileShapeMNK_,
    EpilogueTile_
  >::FusionCallbacks;
};

template <
  class ElementOutput_,
  class ElementCompute_,
  class ElementSource_,
  class ElementScalar_,
  FloatRoundStyle RoundStyle_,
  class CtaTileShapeMNK_,
  class EpilogueTile_
>
struct FusionCallbacks<
    epilogue::IntelXeXMX16Group,
    fusion::LinearCombination<ElementOutput_, ElementCompute_, ElementSource_, ElementScalar_, RoundStyle_>,
    CtaTileShapeMNK_,
    EpilogueTile_
> : FusionCallbacks<
    epilogue::IntelXeGenericGroup,
    fusion::LinearCombination<ElementOutput_, ElementCompute_, ElementSource_, ElementScalar_, RoundStyle_>,
    CtaTileShapeMNK_,
    EpilogueTile_
> {
  using FusionCallbacks<
    epilogue::IntelXeGenericGroup,
    fusion::LinearCombination<ElementOutput_, ElementCompute_, ElementSource_, ElementScalar_, RoundStyle_>,
    CtaTileShapeMNK_,
    EpilogueTile_
  >::FusionCallbacks;
};

} // namespace cutlass::epilogue::fusion

/////////////////////////////////////////////////////////////////////////////////////////////////
