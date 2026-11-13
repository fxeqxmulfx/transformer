#!/usr/bin/env bash
set -euo pipefail
[[ $# == 3 ]] || { echo 'usage: freeze_original.sh REPOSITORY_ROOT PRODUCER_COMMIT NEW_REQUEST' >&2; exit 2; }
repository=$(realpath "$1")
producer=$2
destination=$3
test ! -e "$destination"
for task in 226 230 231; do agent-kanban show "$task" | sed -n '1p' | rg -q "^#$task [^ ]+ done "; done
parent=$repository/experiments/convex_stress_paths_compare/runs/original-v1
review=$repository/.kanban/artifacts/226/review-complete
binary=$repository/.kanban/artifacts/226/freeze-v1/paths-compare-v1
test "$(cat "$review/process.exit")" = 0
cmp "$review/replay.json" "$parent/replay.json"
digest() { sha256sum "$1" | cut -d ' ' -f 1; }
jq -n --arg producer "$producer" --arg suite "$(digest "$repository/experiments/convex_stress/suite.json")" \
  --arg root "$parent" --arg binary "$binary" --arg summary "$(digest "$parent/summary.json")" \
  --arg manifest "$(digest "$parent/manifest.json")" --arg replay "$(digest "$parent/replay.json")" \
  --arg review_exit "$review/process.exit" --arg review_commit "$review/reviewed.commit" --arg review_replay "$review/replay.json" \
  '{producer_commit:$producer,suite_sha256:$suite,parent:{root:$root,binary:$binary,summary_sha256:$summary,
    manifest_sha256:$manifest,replay_sha256:$replay,review:{exit:$review_exit,commit:$review_commit,replay:$review_replay}}}' > "$destination"
