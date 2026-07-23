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
import re
import subprocess
from datetime import datetime
from pathlib import Path
import sys
import argparse
from utils.grafana_push import push_results

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
        "config_file": "../benchmarks/device/bmg/input_files/all_in_one.in",
    }
]

def run_command(command, cwd, log_path=None):
    print(f"\n$ {' '.join(command)}")
    if log_path:
        with open(log_path, "w") as log_file:
            try:
                results = subprocess.run(command, cwd=cwd, text=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                for line in results.stdout.splitlines(keepends=True):
                    sys.stdout.write(line)
                    log_file.write(line)
            except subprocess.CalledProcessError as e:
                print(f"Error: Command failed with return code {e.returncode}")
                if e.stdout:
                    print("Stdout:", e.stdout)
                    print(e.stdout)
                else:
                    print("No output captured.")
        print(f"Log written to: {log_path}")
    else:
        subprocess.run(command, cwd=cwd, check=True)

def parse_benchmark_log(log_path):
    records = []
    total=0
    failed=0
    passed=0
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
                failed+=1
            elif "avg_tflops" in line:
                result = "Pass"
                passed+=1
                tflops_match = re.search(r"avg_tflops=([0-9.]+[a-z]*)", line)
                throughput_match = re.search(r"avg_throughput=([0-9.]+)", line)
                if tflops_match:
                    avg_tflops = tflops_match.group(1)
                if throughput_match:
                    avg_throughput = throughput_match.group(1)
            total+=1
            records.append({
                "Kernel": kernel_name,
                "Shape": dimensions,
                "Result": result,
                "Tflops": avg_tflops,
                "Throughput": avg_throughput,
                "Reason": reason
            })
    print("failed: ", failed)
    print("passed: ", passed)
    print("total: ", total)
    return records

def write_report_csv(path, records):
    with open(path, "w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["Kernel", "Shape", "Result", "Tflops", "Throughput","Reason"],
        )
        writer.writeheader()
        writer.writerows(records)
        print(f"csv written to: {path}")

def get_git_commit_id(repo_root):
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def run_tests(logs_dir, branch, build_dir, repo_root, grafana, git_commit_id):
    for test_suite in TEST_SUITES:
        run_cmd = [
            test_suite["executable"],
            f"--config_file={test_suite['config_file']}",
        ]
        test_name = test_suite["name"]
        test_log = logs_dir / f"{test_name}_run_{branch}.log"
        test_report = logs_dir / f"{test_name}_report_{branch}.csv"
        run_command(run_cmd, cwd=Path(repo_root, build_dir), log_path=test_log)
        main_records = parse_benchmark_log(test_log)
        if grafana:
            push_results(main_records, test_name, git_commit_id=git_commit_id)
        write_report_csv(test_report, main_records)

def main():
    parser = argparse.ArgumentParser(description="Benchmarking script for cutlass kernels.")
    parser.add_argument(
        "--push-to-dashboard",
        dest="grafana",
        action="store_true",
        help="Push benchmark results to the Grafana/InfluxDB dashboard"
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
    git_commit_id = get_git_commit_id(repo_root)
    run_tests(logs_dir, branch, build_dir, repo_root, args.grafana, git_commit_id)

if __name__ == "__main__":
    main()

