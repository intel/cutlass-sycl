/***************************************************************************************************
 * Copyright (c) 2024 - 2025 Codeplay Software Ltd. All rights reserved.
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
    \brief Tests for Xe flash attention decode fp16
*/

#include "flash_decode_testbed_3x.hpp"

namespace cutlass {

using MMAOperationFP16 = XE_1x16x16_F32F16F16F32_TT;

#define EXECUTE_TEST_FP16(NAME, NAME_CAUSAL_VARLEN, DTYPE_IN, DTYPE_ACCUM, DTYPE_OUT, MMAOperation, CAUSAL, VARLEN, HEADSIZE, KVTILE, NUMSG) \
TEST(NAME##HEADSIZE, NAME_CAUSAL_VARLEN) { \
  using Shape_h = test::flash_attention::Shape_h##HEADSIZE<KVTILE, NUMSG>; \
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<DTYPE_IN, DTYPE_ACCUM, DTYPE_OUT, typename Shape_h::ShapeQK, typename Shape_h::ShapePV, \
                                            typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, MMAOperation, CAUSAL, VARLEN>::Kernel; \
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(HEADSIZE)); \
}

#define EXECUTE_TEST_HEAD_SIZE_FP16(NAME, CAUSAL, VARLEN) \
EXECUTE_TEST_FP16(XE_Flash_Attention_Decode_fp16_fp32_fp32_KVTile1024_h, NAME, half_t, float, float, MMAOperationFP16, CAUSAL, VARLEN, 64, 1024, 16)


EXECUTE_TEST_HEAD_SIZE_FP16(causal, true, false)
EXECUTE_TEST_HEAD_SIZE_FP16(noncausal, false, false)
EXECUTE_TEST_HEAD_SIZE_FP16(varlen_causal, true, true)
EXECUTE_TEST_HEAD_SIZE_FP16(varlen_noncausal, false, true)

#undef EXECUTE_TEST_HEAD_SIZE_FP16
#undef EXECUTE_TEST_FP16

} // namespace cutlass
