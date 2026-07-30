#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF POSSIBILITY OF SUCH DAMAGE.
#
#################################################################################################


"""
Unit tests for test/benchmarks/run_benchmarks.py result reporting.

These guard the CI failure-reporting contract: the benchmark runner must exit
non-zero when benchmarks fail, when a suite's binary dies, or when a suite
reports no results at all. Regressions here make the CI job green while
benchmarks are actually failing.
"""

import unittest
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

import run_benchmarks


# A trimmed but verbatim-format sample of real BMG benchmark output.
GEMM_PASS_LOG = """
BmgGemmBF16BF16FP32_RRR_TileShape_512_256_32/bmg_mm/1024x1024x1024x1/manual_time      0.389 ms        0.390 ms         1797 alpha=1 avg_runtime_ms=0.388481 avg_tflops=5.5279 avg_throughput=21.5934 beta=0 k=1.024k l=1 m=1.024k n=1.024k
BmgGemmBF16BF16FP32_RRR_TileShape_512_256_32/bmg_mm/512x256x1024x1/manual_time        0.364 ms        0.365 ms         1927 alpha=1 avg_runtime_ms=0.364043 avg_tflops=0.737374 avg_throughput=5.76073 beta=0 k=1.024k l=1 m=512 n=256
"""

# Verification failures surface as "ERROR OCCURRED" while the binary still exits 0.
FMHA_MIXED_LOG = """
BmgFMHAPrefill_BF16_BF16_BF16_BF16_RCR_WgQ128K64V32_SgQ8K64_HDimQK32V64_NonCausal_FixedLen/attention_prefill_bf16/1x4x4x512x128x512x0x128/manual_time     ERROR OCCURRED: 'Disposition Failed.'
BmgFMHAPrefill_BF16_BF16_BF16_BF16_RCR_WgQ128K64V32_SgQ8K64_HDimQK32V64_NonCausal_FixedLen/attention_prefill_bf16/1x8x8x512x128x512x0x128/manual_time     ERROR OCCURRED: 'Disposition Failed.'
BmgFMHAPrefill_BF16_BF16_BF16_BF16_RCR_h64_NonCausal_FixedLen_CachedKV_PagedKV/attention_prefill_bf16/1x4x4x256x64x256x128x64/manual_time                      0.018 ms        0.018 ms        40158 avg_runtime_ms=0.0175107 avg_tflops=3.83244 avg_throughput=29.9409 batch=1
"""


def parse(text):
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "bench.log"
        log.write_text(text)
        return run_benchmarks.parse_benchmark_log(log)


class TestParseBenchmarkLog(unittest.TestCase):
    def test_all_passing(self):
        records = parse(GEMM_PASS_LOG)
        self.assertEqual(len(records), 2)
        self.assertEqual(run_benchmarks.count_passed(records), 2)
        self.assertEqual(run_benchmarks.count_failed(records), 0)

    def test_error_occurred_counts_as_failure(self):
        records = parse(FMHA_MIXED_LOG)
        self.assertEqual(len(records), 3)
        self.assertEqual(run_benchmarks.count_passed(records), 1)
        self.assertEqual(run_benchmarks.count_failed(records), 2)

    def test_counts_always_sum_to_total(self):
        for text in (GEMM_PASS_LOG, FMHA_MIXED_LOG):
            records = parse(text)
            self.assertEqual(
                run_benchmarks.count_passed(records) + run_benchmarks.count_failed(records),
                len(records),
            )

    def test_missing_log_yields_no_records(self):
        self.assertEqual(run_benchmarks.parse_benchmark_log(Path("/nonexistent/bench.log")), [])


class TestReportSummary(unittest.TestCase):
    def summarize(self, **kwargs):
        suite = {"name": "suite", "returncode": 0, "total": 1, "passed": 1, "failed": 0, "errors": []}
        suite.update(kwargs)
        return run_benchmarks.report_summary([suite])

    def test_clean_suite_passes(self):
        self.assertTrue(self.summarize())

    def test_suite_with_errors_fails(self):
        self.assertFalse(self.summarize(errors=["3 of 10 benchmarks failed"]))

    def test_no_step_summary_env_is_tolerated(self):
        # GITHUB_STEP_SUMMARY is absent outside GitHub Actions; must not raise.
        self.assertTrue(self.summarize())

    def test_suite_reporting_no_results_fails(self):
        # A suite that ran nothing must not read as a pass: this is how a stale
        # --config_file path silently dropped a whole suite from CI.
        self.assertFalse(self.summarize(total=0, passed=0, errors=["no benchmarks ran"]))


class TestSuiteConfigPaths(unittest.TestCase):
    """Every configured .in file must exist, relative to the build directory.

    A stale path here does not fail loudly at runtime -- the benchmark binary just
    reports no results -- so it is checked statically instead. This is a real
    regression: `all_in_one.in` moved into `legacy/` and the suite ran zero
    benchmarks for weeks while CI stayed green.
    """

    def test_all_config_files_exist(self):
        repo_root = Path(__file__).resolve().parents[2]
        for suite in run_benchmarks.TEST_SUITES:
            # config_file paths are relative to the build directory.
            resolved = (repo_root / "build" / suite["config_file"]).resolve()
            with self.subTest(suite=suite["name"]):
                self.assertTrue(
                    resolved.is_file(),
                    f"config file for suite '{suite['name']}' not found: {suite['config_file']} "
                    f"(resolved to {resolved})",
                )


if __name__ == "__main__":
    unittest.main()
