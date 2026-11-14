#!/usr/bin/env bash
set -euo pipefail
[[ $# == 3 ]] || { echo 'usage: freeze_original.sh PRODUCER_CHECKOUT PRODUCER_COMMIT NEW_REQUEST' >&2; exit 2; }
repository=$(realpath "$1")
producer=$2
destination=$3
test ! -e "$destination"
[[ "$producer" =~ ^[[:xdigit:]]{40}$ ]]
for task in 237 246 252 274 275 276; do agent-kanban show "$task" | sed -n '1p' | rg -q "^#$task [^ ]+ done "; done
test "$(git -C "$repository" rev-parse HEAD)" = "$producer"
test -z "$(git -C "$repository" status --porcelain)"
root=$(dirname "$(git -C "$repository" rev-parse --path-format=absolute --git-common-dir)")
storage=/run/media/misha/storage/transformer-runs
reference() {
  local original=$1 binary=$2 review=$3
  test "$(cat "$review/process.exit")" = 0
  cmp "$review/summary.json" "$original/summary.json"
  jq -n --arg root "$original" --arg binary "$binary" \
    --arg summary "$(sha256sum "$original/summary.json" | cut -d ' ' -f 1)" \
    --arg inventory "$(sha256sum "$original/inventory.json" | cut -d ' ' -f 1)" \
    --arg exit "$review/process.exit" --arg commit "$review/reviewed.commit" --arg reviewed "$review/summary.json" \
    --arg exit_hash "$(sha256sum "$review/process.exit" | cut -d ' ' -f 1)" \
    --arg commit_hash "$(sha256sum "$review/reviewed.commit" | cut -d ' ' -f 1)" \
    '{root:$root,binary:$binary,summary_sha256:$summary,inventory_sha256:$inventory,
      review:{exit:$exit,exit_sha256:$exit_hash,commit:$commit,commit_sha256:$commit_hash,summary:$reviewed}}'
}
parent=$(reference "$storage/convex_stress_refined_compare/original-v1" "$root/.kanban/artifacts/246/freeze-v1/refined-compare-v1" "$root/.kanban/artifacts/246/review")
raw=$(reference "$root/experiments/convex_stress_kernel/cohort/runs/training-v1" "$root/.kanban/artifacts/252/freeze-v1/kernel-training-v1" "$root/.kanban/artifacts/252/review")
shared=$(reference "$storage/convex_stress_shared_roles/training-v1" "$root/.kanban/artifacts/274/freeze-v1/shared-roles-training-v1" "$root/.kanban/artifacts/274/review")
causal=$(reference "$root/experiments/convex_stress_causal_pairs/cohort/runs/training-v1" "$root/.kanban/artifacts/237/freeze-v1/causal-pairs-staging-v1" "$root/.kanban/artifacts/237/review")
jq -n --arg producer "$producer" \
  --arg suite "$(sha256sum "$repository/experiments/convex_stress/suite.json" | cut -d ' ' -f 1)" \
  --arg checkpoints "$root/experiments/convex_stress_hull/runs/suite-v1" \
  --argjson parent "$parent" --argjson raw "$raw" --argjson shared "$shared" --argjson causal "$causal" \
  '{producer_commit:$producer,suite_sha256:$suite,checkpoint_root:$checkpoints,parents:{parent:$parent,raw:$raw,shared:$shared,causal:$causal}}' > "$destination"
