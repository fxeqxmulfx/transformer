#!/usr/bin/env bash
set -euo pipefail
test "$#" -eq 5
task_workspace=$1
task_storage=$2
task_freeze=$3
task_cohort=$4
task_output=$5
ps -e -o rss= | awk '{s+=$1*1024}END{if(s+34359738368>110000000000)exit 1}'
test "$(df -B1 --output=avail "$task_workspace" | tail -n 1)" -gt 2147483648
test "$(df -B1 --output=avail "$task_storage" | tail -n 1)" -gt 17179869184
test ! -e "$task_output"
(cd "$task_freeze" && sha256sum -c SHA256SUMS)
prlimit --as=34359738368:34359738368 -- "$task_freeze/convex-hilbert-ablation" run \
 "$task_workspace" "$task_storage" "$task_cohort" "$task_output"
