#!/usr/bin/env bash
# Seal all four complete endpoints before any new held-out scoring.
set -euo pipefail
if test "$#" -lt 5 || test "$#" -gt 6; then
  echo 'usage: bash run.sh WORKSPACE STORAGE SOURCE_RUNS FREEZE FRESH_COHORT [-long|-projected]' >&2
  exit 2
fi
task_workspace=$1
task_storage=$2
task_sources=$3
task_freeze=$4
task_cohort=$5
task_suffix=${6:-}
case "$task_suffix" in ''|-long|-projected) ;; *) echo 'unknown frozen protocol' >&2; exit 2 ;; esac
test -x "$task_freeze/stress"
test -f "$task_freeze/protocol.json"
test "$(df -B1 --output=avail "$task_workspace" | tail -1)" -gt 2147483648
test "$(df -B1 --output=avail "$task_storage" | tail -1)" -gt 17179869184
ps -e -o rss= | awk '{s+=$1*1024}END{if(s+68719476736>110000000000)exit 1}'
mkdir "$task_cohort"
mkdir "$task_cohort/logs"
run_cell() {
 local task_action=$1 task_variant=$2 task_mode=$3 task_result=0
 local task_log=$task_cohort/logs/$task_action-$task_variant-$task_mode
 prlimit --as=34359738368:34359738368 -- "$task_freeze/stress" "$task_action$task_suffix" \
  "$task_workspace" "$task_storage" "$task_sources" "$task_cohort" "$task_variant" "$task_mode" \
  > "$task_log.log" 2>&1 || task_result=$?
 printf '%s\n' "$task_result" > "$task_log.exit"
 return "$task_result"
}
run_mode() {
 local task_action=$1 task_mode=$2 task_variant
 for task_variant in cautious refined; do
  run_cell "$task_action" "$task_variant" "$task_mode" || return "$?"
  if test "$task_action" = score; then
   run_cell verify "$task_variant" "$task_mode" || return "$?"
  fi
 done
}
for task_action in train score; do
 run_mode "$task_action" mini &
 task_mini_pid=$!
 run_mode "$task_action" forward &
 task_forward_pid=$!
 task_result=0
 wait "$task_mini_pid" || task_result=$?
 wait "$task_forward_pid" || task_result=$?
 printf '%s\n' "$task_result" > "$task_cohort/logs/$task_action-queue.exit"
 test "$task_result" -eq 0
done
