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

#include "pvc_flash_attn_decode_runner.hpp"

template <int KV_Tile, int NumSGs, bool Varlen>
int run_decode(Options const& options) {
  if (options.head_size_vo == 64) {

    using ShapeQK = Shape<_1, Int<KV_Tile>, _64>;
    using ShapePV = Shape<_1, _32, Int<KV_Tile>>;
    using ShapeOutput = Shape<_1, _64, Int<KV_Tile>>;
    using SubgroupLayout = Layout<Shape<Int<NumSGs>, _1, _1>, Stride<_1, _1, _1>>;

    return options.is_causal ? FMHAConfig<true, ShapeQK, ShapePV, ShapeOutput, SubgroupLayout, Varlen>::run(options)
                             : FMHAConfig<false, ShapeQK, ShapePV, ShapeOutput, SubgroupLayout, Varlen>::run(options);
  } else if (options.head_size_vo == 96) {

    using ShapeQK = Shape<_1, Int<KV_Tile>, _64>;
    using ShapePV = Shape<_1, _32, Int<KV_Tile>>;
    using ShapeOutput = Shape<_1, _96, Int<KV_Tile>>;
    using SubgroupLayout = Layout<Shape<Int<NumSGs>, _1, _1>, Stride<_1, _1, _1>>;

    return options.is_causal ? FMHAConfig<true, ShapeQK, ShapePV, ShapeOutput, SubgroupLayout, Varlen>::run(options)
                             : FMHAConfig<false, ShapeQK, ShapePV, ShapeOutput, SubgroupLayout, Varlen>::run(options);
  } else if (options.head_size_vo == 128) {

    using ShapeQK = Shape<_1, Int<KV_Tile>, _64>;
    using ShapePV = Shape<_1, _32, Int<KV_Tile>>;
    using ShapeOutput = Shape<_1, _128, Int<KV_Tile>>;
    using SubgroupLayout = Layout<Shape<Int<NumSGs>, _1, _1>, Stride<_1, _1, _1>>;

    return options.is_causal ? FMHAConfig<true, ShapeQK, ShapePV, ShapeOutput, SubgroupLayout, Varlen>::run(options)
                             : FMHAConfig<false, ShapeQK, ShapePV, ShapeOutput, SubgroupLayout, Varlen>::run(options);
  } else if (options.head_size_vo == 192) {

    using ShapeQK = Shape<_1, Int<KV_Tile>, _64>;
    using ShapePV = Shape<_1, _32, Int<KV_Tile>>;
    using ShapeOutput = Shape<_1, _192, Int<KV_Tile>>;
    using SubgroupLayout = Layout<Shape<Int<NumSGs>, _1, _1>, Stride<_1, _1, _1>>;

    return options.is_causal ? FMHAConfig<true, ShapeQK, ShapePV, ShapeOutput, SubgroupLayout, Varlen>::run(options)
                             : FMHAConfig<false, ShapeQK, ShapePV, ShapeOutput, SubgroupLayout, Varlen>::run(options);
  } else {
    std::cerr << "Aborting execution." << std::endl;
    return -1;
  }
}


int main(int argc, const char **argv) {
  //
  // Parse options
  //

  Options options;

  options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  if (options.error) {
    std::cerr << "Aborting execution." << std::endl;
    return -1;
  }

  const int seq_len_kv_total = options.seq_len_kv + options.seq_len_kv_cache;
  const bool kv_tile_block = (seq_len_kv_total % 1024) == 0;

  if(!kv_tile_block && !options.varlen) {
    return run_decode<512, 8, false>(options);
  } else if(kv_tile_block && !options.varlen) {
    return run_decode<1024, 16, false>(options);
  } else if(!kv_tile_block && options.varlen) {
    return run_decode<512, 8, true>(options);
  } else if(kv_tile_block && options.varlen) {
    return run_decode<1024, 16, true>(options);
  }
}
