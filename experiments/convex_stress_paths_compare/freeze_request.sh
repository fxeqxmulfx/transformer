#!/usr/bin/env bash
# Generate immutable input pins only after complete numerical reviews.
set -euo pipefail
[[ $# == 3 ]] || { echo 'usage: freeze_request.sh REPOSITORY_ROOT PRODUCER_COMMIT NEW_REQUEST' >&2; exit 2; }
repository=$(realpath "$1")
producer=$2
destination=$3
test ! -e "$destination"
for task in 175 191 225 227; do
  agent-kanban show "$task" | sed -n '1p' | rg -q "^#$task [^ ]+ done "
done
digest() { sha256sum "$1" | cut -d ' ' -f 1; }
{
  for name in consequences depth_original depth_retained transformer; do
    case "$name" in
      consequences) parent=$repository/experiments/convex_stress_consequences/runs/original-v1; review=$repository/.kanban/artifacts/225/review-complete; summary=$review/summary.json ;;
      depth_original) parent=$repository/experiments/convex_stress_depth/runs/cohort-v1; review=$repository/.kanban/artifacts/175/review-complete; summary=$review/cohort-v1/summary.json ;;
      depth_retained) parent=$repository/experiments/convex_stress_depth/runs/retained-v1; review=$repository/.kanban/artifacts/175/review-complete; summary=$review/retained-v1/summary.json ;;
      transformer) parent=$repository/experiments/convex_stress_transformer/runs/cohort-v1; review=$repository/.kanban/artifacts/191/review-complete; summary=$review/summary.json ;;
    esac
    test "$(cat "$review/process.exit")" = 0
    cmp "$summary" "$parent/summary.json"
    jq -n --arg name "$name" --arg root "$parent" --arg summary_hash "$(digest "$parent/summary.json")" \
      --arg inventory_hash "$(digest "$parent/inventory.json")" --arg review_summary "$summary" \
      --arg review_exit "$review/process.exit" --arg review_commit "$review/reviewed.commit" \
      '{key:$name,value:{root:$root,summary_sha256:$summary_hash,inventory_sha256:$inventory_hash,
        review:{summary:$review_summary,exit:$review_exit,commit:$review_commit}}}'
  done
} | jq -s --arg producer "$producer" --arg suite_hash "$(digest "$repository/experiments/convex_stress/suite.json")" \
  '{producer_commit:$producer,suite_sha256:$suite_hash,parents:from_entries}' > "$destination"
