/***************************************************************************************************
 * Copyright (C) 2025 - 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
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
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/util/command_line.h"

#include "moe_benchmark_runner.hpp"

int main(int argc, const char **argv) {

  BenchmarkOptions options;

  options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  if (options.config_file.empty()) {
    std::cerr << "Benchmark configuration file not found." << std::endl;
    options.error = true;
  }

  if (options.error) {
    std::cerr << "Aborting execution." << std::endl;
    return -1;
  }

  std::ifstream file(options.config_file);

  if (!file.is_open()) {
    std::cerr << "Failed to open configuration file: " << options.config_file
              << std::endl;
    return 1;
  }

  register_grouped_gemm_benchmarks();

  // Each binary registered only the (dtype,tile) names it compiled (see
  // register_grouped_gemm_benchmarks). A shared .in file may list lines for all
  // dtypes, so skip any line whose first token this binary does not hold —
  // otherwise get_benchmark throws "Benchmark not found" and aborts the run.
  // This lets one .in serve every per-dtype binary (and the merged binary).
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line.find("#") == 0) {
      continue;
    }
    std::istringstream iss(line);
    std::string first_token;
    iss >> first_token;
    try {
      cutlass::benchmark::BenchmarkRegistry<
          cutlass::benchmark::MoEBenchmarkOptions>::get_benchmark(first_token);
    } catch (std::exception const &) {
      continue;  // not a name this binary holds
    }
    register_benchmarks<cutlass::benchmark::MoEBenchmarkOptions>(line);
  }
  file.close();

  int argc_bm = 0;
  ::benchmark::SetDefaultTimeUnit(::benchmark::kMillisecond);
  ::benchmark::Initialize(&argc_bm, nullptr);

  ::benchmark::RunSpecifiedBenchmarks();
  ::benchmark::Shutdown();

  return 0;
}
