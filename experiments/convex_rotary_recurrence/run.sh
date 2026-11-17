#!/usr/bin/env bash
# Preserve every declared family and failure; seal every endpoint before scoring.
set -euo pipefail
if test "$#" -lt 4 || test "$#" -gt 5; then
  echo 'usage: bash run.sh WORKSPACE STORAGE FREEZE COHORT [-cautious|-refined]' >&2
  exit 2
fi
task_workspace=$1
task_storage=$2
task_freeze=$3
task_cohort=$4
task_suffix=${5:-}
case "$task_suffix" in
  ''|-cautious|-refined) ;;
  *) echo 'unknown frozen cohort variant' >&2; exit 2 ;;
esac
task_binary=$task_freeze/stress
test -x "$task_binary"
test -f "$task_freeze/protocol.json"
test "$(df -B1 --output=avail "$task_workspace" | tail -1)" -gt 2147483648
test "$(df -B1 --output=avail "$task_storage" | tail -1)" -gt 17179869184
ps -e -o rss= | awk '{sum += $1 * 1024} END {if (sum + 68719476736 > 110000000000) exit 1}'
mkdir "$task_cohort"
mkdir "$task_cohort/logs"
mapfile -t task_cases < <(jq -r '.cases[].name' "$task_freeze/protocol.json")
mapfile -t task_families < <(jq -r '.families[]' "$task_freeze/protocol.json")
mapfile -t task_seeds < <(jq -r '.seeds[]' "$task_freeze/protocol.json")
test "${#task_cases[@]}" -eq 1
case "$task_suffix" in
  -refined) test "${#task_families[@]}" -eq 1 ;;
  -cautious) test "${#task_families[@]}" -eq 3 ;;
  '') test "${#task_families[@]}" -eq 2 ;;
esac
test "${#task_seeds[@]}" -eq 1
run_cell() {
  local task_action=$1 task_case=$2 task_family=$3 task_mode=$4 task_seed=$5 task_result=0
  local task_log=$task_cohort/logs/$task_action-$task_family-$task_case-$task_mode-seed$task_seed
  prlimit --as=34359738368:34359738368 -- "$task_binary" "$task_action$task_suffix" \
    "$task_workspace" "$task_storage" "$task_cohort" "$task_case" "$task_family" "$task_mode" "$task_seed" \
    > "$task_log.log" 2>&1 || task_result=$?
  printf '%s\n' "$task_result" > "$task_log.exit"
  return "$task_result"
}
run_mode() {
  local task_phase=$1 task_mode=$2 task_family task_case task_seed
  for task_family in "${task_families[@]}"; do
    for task_case in "${task_cases[@]}"; do
      for task_seed in "${task_seeds[@]}"; do
        run_cell "$task_phase" "$task_case" "$task_family" "$task_mode" "$task_seed" || return "$?"
        if test "$task_phase" = score; then
          run_cell verify "$task_case" "$task_family" "$task_mode" "$task_seed" || return "$?"
        fi
      done
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
