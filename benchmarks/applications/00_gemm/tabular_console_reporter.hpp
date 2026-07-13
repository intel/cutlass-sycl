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

#pragma once

// A Google Benchmark console reporter that prints every UserCounters entry
// (and every "key=value" token set via State::SetLabel()) as its own
// aligned column, instead of squashing them all into a single trailing
// "UserCounters..." blob. Columns are printed in a fixed, readable order
// (problem shape -> runtime -> throughput -> remaining GEMM parameters) so
// the most useful numbers are easy to scan without reformatting the log
// afterwards.
//
// The column set is fixed (not discovered per-run) on purpose: this keeps
// results streaming to stdout one benchmark at a time -- exactly like the
// stock ConsoleReporter -- which is required for progress monitoring /
// stuck-case detection while a long sweep is running on the CRI simulator.
// A benchmark that doesn't set a given counter (e.g. DualGemm has no
// alpha/beta/bandwidth counters) simply prints a blank cell for it.

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstring>
#include <functional>
#include <iomanip>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

namespace cutlass::benchmark {

class TabularConsoleReporter : public ::benchmark::BenchmarkReporter {
 public:
  bool ReportContext(const Context& context) override {
    PrintBasicContext(&GetErrorStream(), context);
    name_field_width_ = std::max<size_t>(context.name_field_width, std::strlen(kNameHeader));
    return true;
  }

  void ReportRuns(const std::vector<Run>& reports) override {
    if (!printed_header_) {
      PrintHeader();
      printed_header_ = true;
    }
    for (auto const& run : reports) {
      PrintRunData(run);
    }
  }

 private:
  struct Column {
    const char* header;
    std::function<std::optional<std::string>(Run const&)> value;
  };

  static constexpr const char* kNameHeader = "Benchmark";

  // Columns are streamed one benchmark at a time (see file header comment),
  // so widths can't be derived from the actual longest value in the run --
  // there's no "measure everything, then print" pass. Instead each column's
  // printed width is derived from its header length, widened up to
  // kMinValueFieldWidth for anything that holds a formatted double: doubles
  // are printed with std::defaultfloat at 6 significant digits, whose
  // longest representations (e.g. "-1.23457e-05") run to ~12 characters.
  // std::setw() never truncates, so a value longer than this is still
  // printed in full -- it just breaks alignment for that one row/column.
  static constexpr size_t kMinValueFieldWidth = std::char_traits<char>::length("-1.23457e-05");

  static size_t ColumnWidth(Column const& col) {
    return std::max(std::strlen(col.header), kMinValueFieldWidth);
  }

  static std::optional<double> GetCounter(Run const& run, const char* key) {
    auto it = run.counters.find(key);
    if (it == run.counters.end()) return std::nullopt;
    return static_cast<double>(it->second);
  }

  // Parses "key1=val1 key2=val2 ..." tokens set via State::SetLabel(), e.g.
  // the "layoutA=RowMajor layoutB=RowMajor layoutC=RowMajor" label emitted
  // by BenchmarkRunnerGemm::run().
  static std::optional<std::string> GetLabelField(Run const& run, const char* key) {
    std::istringstream iss(run.report_label);
    std::string token;
    std::string const prefix = std::string(key) + "=";
    while (iss >> token) {
      if (token.compare(0, prefix.size(), prefix) == 0) {
        return token.substr(prefix.size());
      }
    }
    return std::nullopt;
  }

  static std::string FormatDouble(double v) {
    std::ostringstream os;
    os << std::defaultfloat << std::setprecision(6) << v;
    return os.str();
  }

  // Builds a value-extractor for a plain numeric UserCounters entry.
  static std::function<std::optional<std::string>(Run const&)> CounterColumn(const char* key) {
    return [key](Run const& run) -> std::optional<std::string> {
      auto v = GetCounter(run, key);
      if (!v) return std::nullopt;
      return FormatDouble(*v);
    };
  }

  static std::function<std::optional<std::string>(Run const&)> LabelColumn(const char* key) {
    return [key](Run const& run) { return GetLabelField(run, key); };
  }

  std::vector<Column> const& Columns() const {
    static const std::vector<Column> columns = {
        {"Time_ms", [](Run const& run) { return FormatDouble(run.GetAdjustedRealTime()); }},
        {"CPU_ms", [](Run const& run) { return FormatDouble(run.GetAdjustedCPUTime()); }},
        {"Iterations", [](Run const& run) { return std::to_string(run.iterations); }},
        {"m", CounterColumn("m")},
        {"n", CounterColumn("n")},
        {"k", CounterColumn("k")},
        {"l", CounterColumn("l")},
        {"avg_runtime_ms", CounterColumn("avg_runtime_ms")},
        {"best_runtime_ms", CounterColumn("best_runtime_ms")},
        {"worst_runtime_ms", CounterColumn("worst_runtime_ms")},
        {"total_runtime_ms", CounterColumn("total_runtime_ms")},
        {"avg_tflops", CounterColumn("avg_tflops")},
        {"best_tflops", CounterColumn("best_tflops")},
        {"avg_bandwidth_gbs", CounterColumn("avg_bandwidth_gbs")},
        {"best_bandwidth_gbs", CounterColumn("best_bandwidth_gbs")},
        {"alpha", CounterColumn("alpha")},
        {"beta", CounterColumn("beta")},
        {"layoutA", LabelColumn("layoutA")},
        {"layoutB", LabelColumn("layoutB")},
        {"layoutC", LabelColumn("layoutC")},
        {"execution_time_s", CounterColumn("execution_time_s")},
    };
    return columns;
  }

  void PrintHeader() {
    std::ostream& out = GetOutputStream();

    // Separation line width: name field + all column widths (each followed by a space).
    size_t total_width = name_field_width_ + 1;
    for (auto const& col : Columns()) {
      total_width += ColumnWidth(col) + 1;
    }
    std::string const separator(total_width, '-');

    // Separation line above the column names.
    out << separator << "\n";

    out << std::left << std::setw(static_cast<int>(name_field_width_)) << kNameHeader << " ";
    for (auto const& col : Columns()) {
      out << std::left << std::setw(static_cast<int>(ColumnWidth(col))) << col.header << " ";
    }
    out << "\n";

    // Separation line between the column names and the values.
    out << separator << "\n";
  }

  void PrintRunData(Run const& run) {
    std::ostream& out = GetOutputStream();
    out << std::left << std::setw(static_cast<int>(name_field_width_)) << run.benchmark_name() << " ";
    for (auto const& col : Columns()) {
      auto value = col.value(run);
      out << std::left << std::setw(static_cast<int>(ColumnWidth(col))) << value.value_or("") << " ";
    }
    out << "\n";
  }

  size_t name_field_width_ = 0;
  bool printed_header_ = false;
};

}  // namespace cutlass::benchmark
