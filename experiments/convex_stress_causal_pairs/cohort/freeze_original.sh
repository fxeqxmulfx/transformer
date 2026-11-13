#!/usr/bin/env bash
set -euo pipefail
[[ $# == 3 || $# == 4 ]] || { echo 'usage: freeze_original.sh REPOSITORY_ROOT PRODUCER_COMMIT NEW_REQUEST [STAGED_ROOT]' >&2; exit 2; }
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
staged_root=
staged_summary=
staged_inventory=
if [[ $# == 4 ]]; then
  agent-kanban show 236 | sed -n '1p' | rg -q '^#236 [^ ]+ done '
  staged_root=$(realpath "$4")
  staged_summary=$(digest "$staged_root/summary.json")
  staged_inventory=$(digest "$staged_root/inventory.json")
fi
jq -n --arg producer "$producer" --arg suite "$(digest "$repository/experiments/convex_stress/suite.json")" \
  --arg root "$parent" --arg binary "$binary" --arg summary "$(digest "$parent/summary.json")" \
  --arg inventory "$(digest "$parent/inventory.json")" \
  --arg review_exit "$review/process.exit" --arg review_commit "$review/reviewed.commit" --arg review_summary "$review/summary.json" \
  --arg staged_root "$staged_root" --arg staged_summary "$staged_summary" --arg staged_inventory "$staged_inventory" \
  '{producer_commit:$producer,suite_sha256:$suite,parent:{root:$root,binary:$binary,summary_sha256:$summary,
    inventory_sha256:$inventory,review:{exit:$review_exit,commit:$review_commit,summary:$review_summary}}}
    + if $staged_root=="" then {} else {staging:{root:$staged_root,summary_sha256:$staged_summary,inventory_sha256:$staged_inventory}} end' > "$destination"
