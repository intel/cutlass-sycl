/***************************************************************************************************
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

#include <cute/arch/mma_xe.hpp>
#include <cute/atom/mma_traits.hpp>
#include "cute/arch/util.hpp"

#include <cute/layout.hpp>

namespace cute
{

namespace detail
{

template <typename ValType, typename LayoutIn>
CUTE_HOST_DEVICE
constexpr auto
wi_interleave(LayoutIn const&)
{
  using namespace intel;
  constexpr LayoutIn layout{};
  constexpr int per_byte = ceil_div(8, sizeof_bits_v<ValType>);
  constexpr int vals = ceil_div(size(layout), sg_size);
  auto tv_interleaved = Layout<Shape<_16,          Shape<C<per_byte>, C<vals/per_byte>>>,
                              Stride<C<per_byte>, Stride<_1,          C<sg_size*per_byte>>>>{};
  return coalesce(composition(layout, tv_interleaved), Step<_1,_1>{});
}

template <typename ValType, typename LayoutIn>
using wi_interleave_t = remove_cvref_t<decltype(wi_interleave<ValType>(LayoutIn{}))>;

} // end namespace detail


template <int M, typename TD, typename TA, typename TB, typename TC>
struct MMA_Traits<XE_DPAS_TT<M, TD, TA, TB, TC>>
{
  using Op = XE_DPAS_TT<M, TD, TA, TB, TC>;

  static constexpr int BV = 32 / sizeof_bits_v<TB>;
  static constexpr int K = Op::K;

  using ValTypeD = TD;
  using ValTypeA = TA;
  using ValTypeB = TB;
  using ValTypeC = TC;
  using _M = Int<M>;
  using _K = Int<K>;

  using Shape_MNK = Shape<_M, _16, _K>;
  using ThrID = Layout<intel::_SGSize>;

  // A layout: (T,V) -> (M,K)
  //   M x K row major, work-items interleaved.
  using ALayout = detail::wi_interleave_t<TA, Layout<Shape<_K, _M>, Stride<_M, _1>>>;

  // B layout: (T,V) -> (N,K)
  //   K x 16 VNNI-transformed row major, work-items interleaved.
  using BLayout = detail::wi_interleave_t<TB, Layout<Shape<Int<BV>, _16, Int<K/BV>>,
                                                     Stride<_16,    _1,  Int<16*BV>>>>;

  // C layout: (T,V) -> (M,N)
  //   M x 16 row major, work-items interleaved.
  using CLayout = Layout<Shape<_16, _M>, Stride<_M, _1>>;
};

template <int M, typename TD, typename TA, typename TB, typename TC>
struct MMA_Traits<XE_BDPAS_TT<M, TD, TA, TB, TC>> : public MMA_Traits<XE_DPAS_TT<M, TD, TA, TB, TC>>
{
  using MMAOp = XE_BDPAS_TT<M, TD, TA, TB, TC>;
  using BaseOp = XE_DPAS_TT<M, TD, TA, TB, TC>;

  template <bool NoAcc = false,
            class TD1, class DLayout,
            class TA1, class ALayout,
            class TB1, class BLayout,
            class TC1, class CLayout>
  CUTE_DEVICE friend void
  mma_unpack(MMA_Traits<MMAOp>    const& traits,
            Tensor<TD1, DLayout>      & D,
            Tensor<TA1, ALayout> const& A_zipped,
            Tensor<TB1, BLayout> const& B_zipped,
            Tensor<TC1, CLayout> const& C)
  {
    static_assert(is_rmem<TD>::value, "Expected registers in MMA_Atom::call");
    static_assert(is_rmem<TA>::value, "Expected registers in MMA_Atom::call");
    static_assert(is_rmem<TB>::value, "Expected registers in MMA_Atom::call");
    static_assert(is_rmem<TC>::value, "Expected registers in MMA_Atom::call");

    using RegTypeD = typename remove_extent<typename MMAOp::DRegisters>::type;
    using RegTypeA = typename remove_extent<typename MMAOp::ARegisters>::type;
    using RegTypeB = typename remove_extent<typename MMAOp::BRegisters>::type;
    using RegTypeC = typename remove_extent<typename MMAOp::CRegisters>::type;

    constexpr int RegNumD = extent<typename MMAOp::DRegisters>::value;
    constexpr int RegNumA = extent<typename MMAOp::ARegisters>::value;
    constexpr int RegNumB = extent<typename MMAOp::BRegisters>::value;
    constexpr int RegNumC = extent<typename MMAOp::CRegisters>::value;

    Tensor rD = recast<RegTypeD>(D);
    Tensor rC = recast<RegTypeC>(C);

    CUTE_STATIC_ASSERT_V(size(rD) == Int<RegNumD>{});
    CUTE_STATIC_ASSERT_V(size(rC) == Int<RegNumC>{});

    // Detect zip payload to choose between hardware BDPAS, software-scaled DPAS, and plain DPAS.
    // Hardware BDPAS (MX path):          4-element zip (data + scale + m_offset + k_offset)
    // Software DPAS  (FP8/BF16/FP16...): 2-element zip (data + scale)
    // Plain DPAS (no scaling):           non-zip tensor (data only)
    // The zip arity is set by the mainloop (xe_blockscaled_mma vs xe_fp8_blockscaled_mma)
    using AValType = typename remove_cvref_t<decltype(A_zipped)>::value_type;
    constexpr bool is_zip_input = is_tuple<AValType>::value;

    if constexpr (!is_zip_input) {
      // === Plain DPAS path (no scaling) ===
      Tensor rA = recast<RegTypeA>(A_zipped);
      Tensor rB = recast<RegTypeB>(B_zipped);

      CUTE_STATIC_ASSERT_V(size(rA) == Int<RegNumA>{});
      CUTE_STATIC_ASSERT_V(size(rB) == Int<RegNumB>{});

      cute::detail::explode_mma<BaseOp, NoAcc>(
              rD, make_int_sequence<RegNumD>{},
              rA, make_int_sequence<RegNumA>{},
              rB, make_int_sequence<RegNumB>{},
              rC, make_int_sequence<RegNumC>{});
    } else {
      auto unzipped_A = unzip_tensor(A_zipped);
      auto unzipped_B = unzip_tensor(B_zipped);

      auto& A = get<0>(unzipped_A);
      auto& B = get<0>(unzipped_B);

      Tensor rA = recast<RegTypeA>(A);
      Tensor rB = recast<RegTypeB>(B);

      CUTE_STATIC_ASSERT_V(size(rA) == Int<RegNumA>{});
      CUTE_STATIC_ASSERT_V(size(rB) == Int<RegNumB>{});

      constexpr auto zip_arity = tuple_size<decltype(unzipped_A)>::value;

      if constexpr (zip_arity == 4) {
        // === Hardware BDPAS path ===
        auto& SFA = get<1>(unzipped_A);
        auto& SFB = get<1>(unzipped_B);
        auto& SFA_M_OFFSET = get<2>(unzipped_A);
        auto& SFA_K_OFFSET = get<3>(unzipped_A);
        auto& SFB_N_OFFSET = get<2>(unzipped_B);
        auto& SFB_K_OFFSET = get<3>(unzipped_B);

        auto sfa_offset = SFA_M_OFFSET[0] + SFA_K_OFFSET[0];
        auto sfb_offset = SFB_N_OFFSET[0] + SFB_K_OFFSET[0];

        cute::detail::explode_mma<MMAOp, NoAcc>(
                rD,   make_int_sequence<RegNumD>{},
                rA,   make_int_sequence<RegNumA>{},
                rB,   make_int_sequence<RegNumB>{},
                rC,   make_int_sequence<RegNumC>{},
                SFA, make_int_sequence<1>{},
                SFB, make_int_sequence<1>{},
                sfa_offset,
                sfb_offset);
      } else {
        // === Software-scaled DPAS path ===
        static_assert(zip_arity == 2, "Unsupported zip arity");
        auto& SFA = get<1>(unzipped_A);
        auto& SFB = get<1>(unzipped_B);

        RegTypeD product{};
        RegTypeC zero{};
        // Inner DPAS already has zero accumulator, use null-src0 to elide it.
        BaseOp::template fma<true>(product, rA[0], rB[0], zero);

        RegTypeD out{};
        for (int i = 0; i < M; ++i) {
          float const scale = static_cast<float>(SFA(i)) * static_cast<float>(SFB(i));
          float const scaled = static_cast<float>(product[i]) * scale;
          if constexpr (NoAcc) {
            out[i] = static_cast<TD>(scaled);
          } else {
            out[i] = static_cast<TD>(scaled + static_cast<float>(rC[0][i]));
          }
        }

        rD[0] = out;
      }
    }
  }

};

template <int M, typename TypeA, typename TypeB>
struct MMA_Traits<XE_DPAS_TT_INT_BLOCK_SCALED<M, TypeA, TypeB>>
{
  using MMAOp = XE_DPAS_TT_INT_BLOCK_SCALED<M, TypeA, TypeB>;
  using RawOp = XE_DPAS_TT<M, dpas_type::d, TypeA, TypeB, dpas_type::d>;
  using RawTraits = MMA_Traits<RawOp>;

  static_assert(
    (std::is_same_v<TypeA, dpas_type::s4> && std::is_same_v<TypeB, dpas_type::s4>) ||
    (std::is_same_v<TypeA, dpas_type::s8> && std::is_same_v<TypeB, dpas_type::s8>),
    "Integer block-scaled DPAS traits support signed INT4xINT4 or INT8xINT8.");

  using ValTypeD = float;
  using ValTypeA = typename RawTraits::ValTypeA;
  using ValTypeB = typename RawTraits::ValTypeB;
  using ValTypeC = float;

  using FrgTypeD = float;
  using FrgTypeC = float;

  using Shape_MNK = typename RawTraits::Shape_MNK;
  using ThrID = typename RawTraits::ThrID;
  using ALayout = typename RawTraits::ALayout;
  using BLayout = typename RawTraits::BLayout;
  using CLayout = typename RawTraits::CLayout;

  template <bool NoAcc = false,
            class TD1, class DLayout,
            class TA1, class ALayout1,
            class TB1, class BLayout1,
            class TC1, class CLayout1>
  CUTE_DEVICE friend void
  mma_unpack(MMA_Traits const&,
             Tensor<TD1, DLayout>      & D,
             Tensor<TA1, ALayout1> const& A,
             Tensor<TB1, BLayout1> const& B,
             Tensor<TC1, CLayout1> const& C)
  {
    using RegTypeD = typename remove_extent<typename RawOp::DRegisters>::type;
    using RegTypeA = typename remove_extent<typename RawOp::ARegisters>::type;
    using RegTypeB = typename remove_extent<typename RawOp::BRegisters>::type;
    using RegTypeC = typename remove_extent<typename RawOp::CRegisters>::type;

    using DValue = typename Tensor<TD1, DLayout>::value_type;
    using CValue = typename Tensor<TC1, CLayout1>::value_type;

    if constexpr (is_same_v<DValue, int32_t> && is_same_v<CValue, int32_t>) {
      Tensor rA = recast<RegTypeA>(A);
      Tensor rB = recast<RegTypeB>(B);
      Tensor rD = recast<RegTypeD>(D);
      Tensor rC = recast<RegTypeC>(C);
      cute::detail::explode_mma<RawOp, NoAcc>(
          rD, make_int_sequence<extent<typename RawOp::DRegisters>::value>{},
          rA, make_int_sequence<extent<typename RawOp::ARegisters>::value>{},
          rB, make_int_sequence<extent<typename RawOp::BRegisters>::value>{},
          rC, make_int_sequence<extent<typename RawOp::CRegisters>::value>{});
    }
    else {
      Tensor rA = recast<RegTypeA>(A);
      Tensor rB = recast<RegTypeB>(B);
      RegTypeD product{};
      RegTypeC zero{};
      RawOp::template fma<true>(product, rA[0], rB[0], zero);

      for (int i = 0; i < M; ++i) {
        float value = static_cast<float>(product[i]);
        if constexpr (!NoAcc) {
          value += static_cast<float>(C(i));
        }
        D(i) = static_cast<DValue>(value);
      }
    }
  }
};

} /* namespace cute */
