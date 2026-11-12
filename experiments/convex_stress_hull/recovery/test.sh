#!/usr/bin/env bash
# Check inventory guards without constructing examples or running a learner.
set -euo pipefail
recovery_source=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
recovery_repo=$(git -C "$recovery_source" rev-parse --show-toplevel)
recovery_root=$(git -C "$recovery_repo" rev-parse --path-format=absolute --git-common-dir)
recovery_root=${recovery_root%/.git}
recovery_temp=$(mktemp -d)
trap 'rm -rf -- "$recovery_temp"' EXIT
protocol="$recovery_source/protocol.json"
prepared="$recovery_root/experiments/convex_stress_hull/runs/suite-v1/prepared.json"
jq --slurpfile recovery "$protocol" '
  . as $p | [$p.tasks[] as $task | $p.protocol.arms[] as $arm |
  $p.protocol.candidates[] as $candidate |
  select(($recovery[0].missing|index($arm+"--"+$task.task+"--"+$candidate))==null) |
  {task:$task.task,arm:$arm,candidate:$candidate,source_fingerprint:$p.source_fingerprint,
   protocol:$p.protocol,data_fingerprints:$task.data_fingerprints,vocab:$task.vocab,
   scale:$task.scale,seed:$p.protocol.seed,status:"complete",model_sha256:("0"*64)}]
' "$prepared" > "$recovery_temp/valid.json"
validate() {
  jq --slurpfile prep "$prepared" --slurpfile recovery "$protocol" \
    -f "$recovery_source/inventory.jq" "$1" > /dev/null 2> "$recovery_temp/error.log"
}
validate "$recovery_temp/valid.json"
for change in '.[1: ]' '.+[.[0]]' '.[0].status="interrupted"' \
  '.[0].source_fingerprint="changed"' '.[0].protocol.seed=99' \
  '.[0].task="unplanned"' '.[0].data_fingerprints={}' '.[0].model_sha256=null' \
  '.[0].vocab=999' '.[0].arm="invalid_mode"'; do
  jq "$change" "$recovery_temp/valid.json" > "$recovery_temp/changed.json"
  if validate "$recovery_temp/changed.json"; then
    echo "Inventory guard accepted mutation: $change" >&2; exit 1
  fi
done
if HULL_RECOVERY_COMMIT=0000000000000000000000000000000000000000 \
  bash "$recovery_source/run.sh" prepare "$recovery_temp" > "$recovery_temp/guard.log" 2>&1; then
  echo 'Existing destination was accepted.' >&2; exit 1
fi
if HULL_RECOVERY_COMMIT=invalid bash "$recovery_source/run.sh" prepare "$recovery_temp/fresh" \
  > "$recovery_temp/guard.log" 2>&1; then
  echo 'Missing committed producer was accepted.' >&2; exit 1
fi
test ! -e "$recovery_temp/fresh"
printf '%s\n' 'Valid 311-cell fixture accepted; ten inventory mutations and two unsafe preparation requests refused.'
