#!/usr/bin/env bash
set -euo pipefail
[[ $# == 3 ]] || { echo 'usage: freeze_original.sh REPOSITORY_ROOT PRODUCER_COMMIT NEW_REQUEST' >&2; exit 2; }
repository=$(realpath "$1")
producer=$2
destination=$3
test ! -e "$destination"
[[ "$producer" =~ ^[[:xdigit:]]{40}$ ]]
for task in 232 233 234; do agent-kanban show "$task" | sed -n '1p' | rg -q "^#$task [^ ]+ done "; done
parent=$repository/experiments/convex_stress_quotient/cohort/runs/original-v1
review=$repository/.kanban/artifacts/232/review-complete
binary=$repository/.kanban/artifacts/232/freeze-v1/quotient-v1
test "$(cat "$review/process.exit")" = 0
cmp "$review/summary.json" "$parent/summary.json"
digest() { sha256sum "$1" | cut -d ' ' -f 1; }
jq -n --arg producer "$producer" --arg suite "$(digest "$repository/experiments/convex_stress/suite.json")" \
  --arg root "$parent" --arg binary "$binary" --arg summary "$(digest "$parent/summary.json")" \
  --arg inventory "$(digest "$parent/inventory.json")" \
  --arg review_exit "$review/process.exit" --arg review_commit "$review/reviewed.commit" --arg review_summary "$review/summary.json" \
  '{producer_commit:$producer,suite_sha256:$suite,parent:{root:$root,binary:$binary,summary_sha256:$summary,
    inventory_sha256:$inventory,review:{exit:$review_exit,commit:$review_commit,summary:$review_summary}}}' > "$destination"
