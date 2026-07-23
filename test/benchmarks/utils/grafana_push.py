#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

import os
import time
import urllib.error
import urllib.request


def _escape_tag(value):
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace(" ", "\\ ")
        .replace(",", "\\,")
        .replace("=", "\\=")
    )


def _build_line_protocol(records, suite_name, timestamp_ns, git_commit_id=""):
    lines = []
    commit_tag = _escape_tag(git_commit_id) if git_commit_id else ""

    for record in records:
        kernel = (record.get("Kernel", ""))
 
        suite = (suite_name)
        result = record.get("Result", "")

        fields = []
        tflops = record.get("Tflops",0)
        throughput = record.get("Throughput",0)
        shape = record.get("Shape", "")

        if isinstance(shape, (list, tuple)):
            shape = "x".join(str(dim) for dim in shape)
        else:
            shape = str(shape)

        kernel = kernel + f"_{shape}"

        if tflops:
            fields.append(f"tflops={float(tflops)}")
        else:
            fields.append(f"tflops=0")
        if throughput:
            fields.append(f"throughput={float(throughput)}")
        else:
            fields.append(f"throughput=0")

        if result:
            fields.append(f"result=\"{str(result).replace('"', '\\"')}\"")

        if not fields:
            continue

        tag_set = f"suite={suite},kernel={kernel}"
        if commit_tag:
            fields.append(f"commit_id=\"{commit_tag}\"")

        line = f"cutlass_benchmarks,{tag_set} {','.join(fields)} {timestamp_ns}"

        lines.append(line)

    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def push_results(records, suite_name, timestamp_ns=None, git_commit_id=""):
    influx_url = os.getenv("INFLUX_URL")
    influx_org = os.getenv("INFLUX_ORG")
    influx_bucket = os.getenv("INFLUX_BUCKET")
    influx_token = os.getenv("INFLUX_TOKEN")

    if not all([influx_url, influx_org, influx_bucket, influx_token]):
        print("InfluxDB config missing; skipping push (set INFLUX_URL/ORG/BUCKET/TOKEN).")
        return

    if timestamp_ns is None:
        timestamp_ns = int(time.time() * 1e9)

    payload = _build_line_protocol(records, suite_name, timestamp_ns, git_commit_id)
    if not payload:
        print("No records to push to InfluxDB.")
        return

    write_url = (
        f"{influx_url.rstrip('/')}/api/v2/write"
        f"?org={influx_org}&bucket={influx_bucket}&precision=ns"
    )

    request = urllib.request.Request(write_url, data=payload.encode("utf-8"), method="POST")
    request.add_header("Authorization", f"Token {influx_token}")
    request.add_header("Content-Type", "text/plain; charset=utf-8")

    try:
        with urllib.request.urlopen(request) as response:
            if response.status >= 300:
                print(f"InfluxDB write failed: HTTP {response.status}")
            else:
                print("InfluxDB write succeeded.")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8") if exc.fp else ""
        print(f"InfluxDB write error: HTTP {exc.code} {error_body}")
    except urllib.error.URLError as exc:
        print(f"InfluxDB connection error: {exc.reason}")

