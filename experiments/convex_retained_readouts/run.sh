#!/usr/bin/env bash
set -euo pipefail
test "$#" -eq 6
task_workspace=$1
task_storage=$2
task_sources=$3
task_freeze=$4
task_cohort=$5
task_diagnostics=$6
admit() {
 ps -e -o rss= | awk '{s+=$1*1024}END{if(s+34359738368>110000000000)exit 1}'
 test "$(df -B1 --output=avail "$task_workspace" | tail -n 1)" -gt 2147483648
 test "$(df -B1 --output=avail "$task_storage" | tail -n 1)" -gt 17179869184
}
test ! -e "$task_cohort"
test ! -e "$task_diagnostics"
for task_arm in instantaneous-small retained-large retained-small; do
 for task_variant in cautious refined; do
  for task_mode in mini forward; do
   admit
   prlimit --as=34359738368:34359738368 -- "$task_freeze/convex-retained-readouts" train \
    "$task_workspace" "$task_storage" "$task_sources" "$task_cohort" "$task_arm" "$task_variant" "$task_mode"
  done
 done
done
admit
prlimit --as=34359738368:34359738368 -- "$task_freeze/convex-retained-readouts" diagnose \
 "$task_workspace" "$task_storage" "$task_cohort" "$task_diagnostics"
admit
prlimit --as=34359738368:34359738368 -- "$task_freeze/convex-retained-readouts" prior \
 "$task_workspace" "$task_storage" "$task_storage/transformer-runs/joint-readouts/original-v1" "$task_diagnostics/prior.json"
for task_arm in instantaneous-small retained-large retained-small; do
 for task_variant in cautious refined; do
  for task_mode in mini forward; do
   for task_action in score replay; do
    admit
    prlimit --as=34359738368:34359738368 -- "$task_freeze/convex-retained-readouts" "$task_action" \
     "$task_workspace" "$task_storage" "$task_cohort" "$task_arm" "$task_variant" "$task_mode"
   done
  done
 done
done
