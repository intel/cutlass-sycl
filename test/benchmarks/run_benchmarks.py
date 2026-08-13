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

import csv
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
import sys
import argparse

TEST_SUITES = [
    {
        "name": "gemm_sycl",
        "executable": "./benchmarks/gemm/cutlass_benchmarks_gemm_sycl",
        "config_file": "../benchmarks/device/bmg/input_files/bmg_small_input.in",
    },
    {
        "name": "flash_attention_prefill",
        "executable": "./benchmarks/flash_attention/cutlass_benchmarks_flash_attention_prefill_xe",
        "config_file": "../benchmarks/device/bmg/input_files/input_flash_attention_prefill_bf16.in",
    },
    {
        "name": "flash_attention_decode",
        "executable": "./benchmarks/flash_attention/cutlass_benchmarks_flash_attention_decode_xe",
        "config_file": "../benchmarks/device/bmg/input_files/input_flash_attention_decode_bf16.in",
    },
    {
        "name": "cutlass_benchmarks_gemm_sycl_legacy",
        "executable": "./benchmarks/gemm/legacy/cutlass_benchmarks_gemm_sycl_legacy",
        "config_file": "../benchmarks/device/bmg/input_files/legacy/all_in_one.in",
    }
]

def run_command(command, cwd, log_path=None):
    """Run a benchmark binary, tee its output to log_path, and return its exit code."""
    print(f"\n$ {' '.join(command)}")
    if log_path is None:
        return subprocess.run(command, cwd=cwd, check=True).returncode

    with open(log_path, "w") as log_file:
        try:
            results = subprocess.run(command, cwd=cwd, text=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            output = results.stdout
            returncode = 0
        except subprocess.CalledProcessError as e:
            # Still tee whatever the binary produced before dying, otherwise the
            # uploaded log is empty and the failure cannot be triaged after the fact.
            output = e.stdout or ""
            returncode = e.returncode
            print(f"Error: Command failed with return code {e.returncode}")
            if not output:
                print("No output captured.")
        for line in output.splitlines(keepends=True):
            sys.stdout.write(line)
            log_file.write(line)
    print(f"Log written to: {log_path}")
    return returncode

def parse_benchmark_log(log_path):
    records = []
    if not log_path.exists():
        return records

    with open(log_path, "r") as handle:
        for line in handle:
            if not re.search(r"(Gemm|manual_time)", line):
                continue

            parts = line.strip().split()
            if not parts:
                continue

            benchmark_token = parts[0]
            tokens = benchmark_token.split("/")
            if len(tokens) < 3:
                continue

            kernel_name = tokens[0]
            dimensions = tokens[2]
            result = "Fail"
            reason=""
            avg_tflops = ""
            avg_throughput = ""

            if any(sub in line for sub in ["ERROR OCCURRED", "ERROR"]):
                result = "Fail"
                reason=line.strip()
            elif "avg_tflops" in line:
                result = "Pass"
                tflops_match = re.search(r"avg_tflops=([0-9.]+[a-z]*)", line)
                throughput_match = re.search(r"avg_throughput=([0-9.]+)", line)
                if tflops_match:
                    avg_tflops = tflops_match.group(1)
                if throughput_match:
                    avg_throughput = throughput_match.group(1)
            else:
                # Neither a reported error nor a line carrying timing counters. Treat
                # as a failure rather than dropping it, so it cannot pass unnoticed.
                reason = "No timing counters reported"
            records.append({
                "Kernel": kernel_name,
                "Shape": dimensions,
                "Result": result,
                "Tflops": avg_tflops,
                "Throughput": avg_throughput,
                "Reason": reason
            })
    # Counts are derived from the records so the CSV and the pass/fail verdict can
    # never disagree.
    print("failed: ", count_failed(records))
    print("passed: ", count_passed(records))
    print("total: ", len(records))
    return records

def count_failed(records):
    return sum(1 for r in records if r["Result"] != "Pass")

def count_passed(records):
    return sum(1 for r in records if r["Result"] == "Pass")

def write_report_csv(path, records):
    with open(path, "w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["Kernel", "Shape", "Result", "Tflops", "Throughput","Reason"],
        )
        writer.writeheader()
        writer.writerows(records)
        print(f"csv written to: {path}")

def run_tests(logs_dir, branch, build_dir, repo_root):
    """Run every suite and return a per-suite summary list.

    Every suite is run even if an earlier one fails, so a single broken suite does
    not hide the results of the others.
    """
    summaries = []
    work_dir = Path(repo_root, build_dir)
    for test_suite in TEST_SUITES:
        run_cmd = [
            test_suite["executable"],
            f"--config_file={test_suite['config_file']}",
        ]
        test_name = test_suite["name"]
        test_log = logs_dir / f"{test_name}_run_{branch}.log"
        test_report = logs_dir / f"{test_name}_report_{branch}.csv"

        # Preflight the config path so a stale entry is reported by name up front,
        # rather than as an opaque non-zero exit from the benchmark binary.
        missing_config = not (work_dir / test_suite["config_file"]).exists()
        if missing_config:
            print(f"\nError: config file not found for suite '{test_name}': {test_suite['config_file']}")

        returncode = run_command(run_cmd, cwd=work_dir, log_path=test_log)
        main_records = parse_benchmark_log(test_log)
        write_report_csv(test_report, main_records)

        failed = count_failed(main_records)
        total = len(main_records)
        errors = []
        if missing_config:
            errors.append(f"config file not found: {test_suite['config_file']}")
        if returncode != 0:
            errors.append(f"binary exited with code {returncode}")
        if total == 0:
            # A suite that reports nothing is a failure, not a pass. This is what
            # let a stale --config_file path silently drop a whole suite from CI.
            errors.append("no benchmarks ran (missing config file or empty result set)")
        if failed:
            errors.append(f"{failed} of {total} benchmarks failed")

        summaries.append({
            "name": test_name,
            "returncode": returncode,
            "total": total,
            "passed": count_passed(main_records),
            "failed": failed,
            "errors": errors,
        })
    return summaries


def report_summary(summaries):
    """Print a per-suite verdict and emit a GitHub step summary. Returns True if all suites passed."""
    all_ok = all(not s["errors"] for s in summaries)

    print("\n" + "=" * 72)
    print("BENCHMARK SUMMARY")
    print("=" * 72)
    for s in summaries:
        status = "PASS" if not s["errors"] else "FAIL"
        print(f"[{status}] {s['name']}: {s['passed']} passed, {s['failed']} failed, {s['total']} total")
        for error in s["errors"]:
            print(f"         -> {error}")
    print("=" * 72)
    print("OVERALL: " + ("PASS" if all_ok else "FAIL"))

    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a") as handle:
            handle.write("## SYCL-TLA Benchmark Results\n\n")
            handle.write(f"**Overall: {'✅ PASS' if all_ok else '❌ FAIL'}**\n\n")
            handle.write("| Suite | Status | Passed | Failed | Total | Details |\n")
            handle.write("|---|---|---|---|---|---|\n")
            for s in summaries:
                status = "✅ PASS" if not s["errors"] else "❌ FAIL"
                details = "; ".join(s["errors"]) or "-"
                handle.write(
                    f"| {s['name']} | {status} | {s['passed']} | {s['failed']} | {s['total']} | {details} |\n"
                )

    return all_ok

def main():
    parser = argparse.ArgumentParser(description="Benchmarking script for cutlass kernels.")
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Report benchmark failures but still exit 0 (for local experimentation)"
    )
    args = parser.parse_args()

    build_dir = "build"
    branch = "main"
    # This file is expected to be run from the root of the repository,
    # so we can directly use relative paths to access logs and benchmarks.
    repo_root = Path.cwd()
    logs_root = repo_root / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    workdir = f"{datetime.now().strftime('%Y%m%d%I%M')}_benchmarks_{branch}"
    logs_dir = logs_root / workdir
    logs_dir.mkdir(parents=True, exist_ok=True)
    summaries = run_tests(logs_dir, branch, build_dir, repo_root)
    all_ok = report_summary(summaries)
    if not all_ok and not args.allow_failures:
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())

