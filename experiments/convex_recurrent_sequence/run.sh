#!/usr/bin/env bash
# Run a previously frozen native producer; retain every endpoint and failure.
set -euo pipefail
if test "$#" -ne 4; then
  echo 'usage: bash run.sh WORKSPACE STORAGE FREEZE COHORT' >&2
  exit 2
fi
task_workspace=$1
task_storage=$2
task_freeze=$3
task_cohort=$4
task_binary=$task_freeze/stress
test -x "$task_binary"
test -f "$task_freeze/protocol.json"
test "$(df -B1 --output=avail "$task_workspace" | tail -1)" -gt 2147483648
test "$(df -B1 --output=avail "$task_storage" | tail -1)" -gt 17179869184
ps -e -o rss= | awk '{sum += $1 * 1024} END {if (sum + 68719476736 > 110000000000) exit 1}'
mkdir "$task_cohort"
mkdir "$task_cohort/logs"
mapfile -t task_cases < <(jq -r '.cases[].name' "$task_freeze/protocol.json")
mapfile -t task_seeds < <(jq -r '.seeds[]' "$task_freeze/protocol.json")
test "${#task_cases[@]}" -eq 3
test "${#task_seeds[@]}" -eq 1

run_cell() {
  local task_action=$1 task_case=$2 task_mode=$3 task_seed=$4 task_result=0
  local task_log=$task_cohort/logs/$task_action-$task_case-$task_mode-seed$task_seed
  prlimit --as=34359738368:34359738368 -- "$task_binary" "$task_action" \
    "$task_workspace" "$task_storage" "$task_cohort" "$task_case" "$task_mode" "$task_seed" \
    > "$task_log.log" 2>&1 || task_result=$?
  printf '%s\n' "$task_result" > "$task_log.exit"
  return "$task_result"
}

run_mode() {
  local task_phase=$1 task_mode=$2 task_case task_seed
  for task_case in "${task_cases[@]}"; do
    for task_seed in "${task_seeds[@]}"; do
      run_cell "$task_phase" "$task_case" "$task_mode" "$task_seed" || return "$?"
      if test "$task_phase" = score; then
        run_cell verify "$task_case" "$task_mode" "$task_seed" || return "$?"
      fi
    done
  done
}

for task_phase in train score; do
  run_mode "$task_phase" mini &
  task_mini_pid=$!
  run_mode "$task_phase" forward &
  task_forward_pid=$!
  task_result=0
  wait "$task_mini_pid" || task_result=$?
  wait "$task_forward_pid" || task_result=$?
  printf '%s\n' "$task_result" > "$task_cohort/logs/$task_phase-queue.exit"
  test "$task_result" -eq 0
done
