#!/usr/bin/env bash
# Recover only the four missing slots; the original cohort is read-only input.
set -euo pipefail
ulimit -c 0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_root=$(git rev-parse --path-format=absolute --git-common-dir)
project_root=${project_root%/.git}
protocol="$source_dir/protocol.json"
field() { jq -r "$1" "$protocol"; }
parent="$project_root/$(field .parent)"
producer="$project_root/$(field .producer)"
counts_auditor="$project_root/$(field .counts_auditor)"
moments_auditor="$project_root/$(field .moments_auditor)"
fingerprint=$(field .source_fingerprint)
limit=$(field .address_space_limit_bytes)
scratch=$(mktemp -d)
trap 'rm -rf -- "$scratch"' EXIT
fail() { printf '%s\n' "$*" >&2; exit 1; }
digest() { local actual; actual=$(sha256sum -- "$1"); printf '%s\n' "${actual%% *}"; }
check_digest() { [[ $(digest "$1") == "$2" ]] || fail "Changed frozen input: $1"; }

check_frozen() {
  [[ "$limit" == 68719476736 ]] || fail 'Recovery requires the frozen 64 GiB cap.'
  while IFS=$'\t' read -r path sha; do
    check_digest "$project_root/$path" "$sha"
  done < <(jq -r '.frozen_inputs|to_entries[]|[.key,.value]|@tsv' "$protocol")
  while IFS=$'\t' read -r path sha; do check_digest "$path" "$sha"; done \
    < <(jq -r '.native_libraries|to_entries[]|[.key,.value]|@tsv' "$protocol")
  [[ $("$producer" fingerprint) == "$fingerprint" ]] || fail 'Producer fingerprint changed.'
  local interruptions="$project_root/$(field .interruption_manifest)"
  cmp -s <(jq -c '[.interrupted[].cell]|sort' "$interruptions") \
    <(jq -c '.missing|sort' "$protocol") || fail 'Interruption scope changed.'
}

check_parent() {
  check_frozen
  sha256sum --check --quiet "$project_root/$(field .completed_manifest)"
  jq -s --slurpfile prep "$parent/prepared.json" --slurpfile recovery "$protocol" \
    -f "$source_dir/inventory.jq" "$parent"/*/result.json > "$scratch/inventory.json"
  while IFS= read -r cell; do
    [[ -d "$parent/$cell" && ! -e "$parent/$cell/result.json" ]] || fail "Missing slot was changed: $cell"
    [[ -s "$parent/$cell/metadata.json" && -s "$parent/$cell/trace.jsonl" ]] || fail "Interrupted trace missing: $cell"
  done < <(jq -r '.missing[]' "$protocol")
}

# Print the exact existing evidence files after checking model/state bindings,
# replay identities and the successful independent audit of each saved model.
retained_files() {
  local result directory cell status arm task candidate model_sha state_sha audit found
  printf '%s\n' "$parent/prepared.json" "$parent/planned.tsv" "$parent/protocol.json" "$parent/source_fingerprint.txt"
  printf '%s\n' "$project_root/$(field .interruption_manifest)" "$project_root/$(field .completed_manifest)"
  for result in "$parent"/*/result.json; do
    directory=$(dirname -- "$result"); cell=$(basename -- "$directory")
    IFS=$'\t' read -r status arm task candidate model_sha state_sha < <(
      jq -r '[.status,.arm,.task,.candidate,(.model_sha256//"none"),(.diagnostic_state_sha256//"none")]|@tsv' "$result")
    [[ "$cell" == "$arm--$task--$candidate" ]] || fail 'Retained directory identity changed.'
    jq -e --arg task "$task" --arg arm "$arm" --arg candidate "$candidate" --arg status "$status" \
      '.replayed==true and .task==$task and .arm==$arm and .candidate==$candidate and .status==$status' \
      "$directory/replay.json" > /dev/null
    printf '%s\n' "$result" "$directory/metadata.json" "$directory/replay.json"
    if [[ -f "$directory/trace.jsonl" ]]; then printf '%s\n' "$directory/trace.jsonl"; fi
    if [[ "$status" != complete ]]; then continue; fi
    check_digest "$directory/model.json" "$model_sha"
    printf '%s\n' "$directory/model.json"
    if [[ "$state_sha" != none ]]; then
      check_digest "$directory/diagnostic_state.json" "$state_sha"
      printf '%s\n' "$directory/diagnostic_state.json"
    fi
    found=false
    for audit in "$project_root/.kanban/artifacts/177/pending-audits/$cell" \
                 "$project_root/.kanban/artifacts/166/independent-audits/$cell"; do
      if [[ -f "$audit.exit" && $(cat "$audit.exit") == 0 ]]; then
        jq -e --arg task "$task" --arg arm "$arm" --arg candidate "$candidate" \
          '.independent_audit==true and .task==$task and .arm==$arm and .candidate==$candidate' "$audit.json" > /dev/null
        if [[ "$candidate" == counts ]]; then
          [[ $(jq -r .model_sha256 "$audit.json") == "$model_sha" ]] || fail 'Count audit model changed.'
        else
          [[ $(jq -r .source_fingerprint "$audit.json") == "$fingerprint" ]] || fail 'Moment audit source changed.'
        fi
        printf '%s\n' "$audit.json" "$audit.exit"
        found=true; break
      fi
    done
    [[ "$found" == true ]] || fail "No successful independent audit: $cell"
  done
  while IFS= read -r cell; do
    printf '%s\n' "$parent/$cell/metadata.json" "$parent/$cell/trace.jsonl"
  done < <(jq -r '.missing[]' "$protocol")
}

prepare() {
  local destination=$1
  [[ ! -e "$destination" ]] || fail 'Recovery destination already exists.'
  destination=$(realpath -m -- "$destination")
  [[ "$destination" != "$parent" && "$destination" != "$parent/"* ]] || fail 'Recovery must be outside the original cohort.'
  [[ ${HULL_RECOVERY_COMMIT:-} =~ ^[0-9a-f]{40}$ ]] || fail 'Set HULL_RECOVERY_COMMIT to the checked harness commit.'
  check_parent
  retained_files > "$scratch/retained-paths.txt"
  mkdir -- "$destination"
  mkdir -- "$destination/cells"
  cp -- "$scratch/inventory.json" "$destination/inventory.json"
  cp -- "$protocol" "$destination/recovery-protocol.json"
  cp -- "$scratch/retained-paths.txt" "$destination/retained-paths.txt"
  while IFS= read -r path; do sha256sum -- "$path"; done < "$scratch/retained-paths.txt" > "$destination/retained.sha256"
  sha256sum -- "$source_dir/run.sh" "$source_dir/inventory.jq" "$protocol" \
    "$producer" "$counts_auditor" "$moments_auditor" > "$destination/frozen.sha256"
  jq -n --arg commit "$HULL_RECOVERY_COMMIT" --arg parent "$parent" \
    --arg producer "$producer" --arg fp "$fingerprint" --argjson cap "$limit" \
    '{harness_commit:$commit,parent:$parent,producer:$producer,source_fingerprint:$fp,
      address_space_limit_bytes:$cap,original_retained:311,fresh_attempts:4,
      original_cohort_overwritten:false,solver_resume:false}' > "$destination/provenance.json"
  printf '%s\n' 'Prepared exactly four fresh attempts; retained evidence hashes are frozen.'
}

check_recovery() {
  local destination=$1
  cmp -- "$protocol" "$destination/recovery-protocol.json"
  sha256sum --check --quiet "$destination/frozen.sha256"
  sha256sum --check --quiet "$destination/retained.sha256"
  check_parent
  cmp -- "$scratch/inventory.json" "$destination/inventory.json"
}

measure() {
  local prefix=$1; shift
  local result
  if /usr/bin/time -v -o "$prefix.resources" prlimit --as="$limit:$limit" -- "$@" > "$prefix.stdout" 2> "$prefix.log"; then
    result=0
  else result=$?; fi
  printf '%s\n' "$result" > "$prefix.exit"
  return "$result"
}

finish() {
  local destination=$1 cell arm task candidate path view
  check_recovery "$destination"
  while IFS= read -r cell; do
    for stage in train replay audit; do
      [[ -f "$destination/$cell-$stage.exit" && $(cat "$destination/$cell-$stage.exit") == 0 ]] || fail "Unfinished or failed recovery: $cell $stage"
    done
    sha256sum --check --quiet "$destination/$cell-seal.sha256"
  done < <(jq -r '.missing[]' "$protocol")
  view="$destination/view"
  [[ ! -e "$view" ]] || fail 'Comparison view already exists; verify it instead of overwriting.'
  mkdir -- "$view"
  cp -- "$parent/prepared.json" "$parent/planned.tsv" "$parent/protocol.json" "$parent/source_fingerprint.txt" "$view/"
  while IFS=$'\t' read -r arm task candidate; do
    cell="$arm--$task--$candidate"
    if jq -e --arg cell "$cell" '.missing|index($cell)!=null' "$protocol" > /dev/null; then
      path="$destination/cells/$cell"
    else path="$parent/$cell"; fi
    [[ -s "$path/result.json" ]] || fail "Missing endpoint for comparison: $cell"
    ln -s -- "$path" "$view/$cell"
  done < "$parent/planned.tsv"
  jq -s --slurpfile prep "$view/prepared.json" -f "$project_root/experiments/convex_stress_hull/summarize.jq" \
    "$view"/*/result.json > "$view/summary.json"
  measure "$destination/selection" "$producer" select "$view"
  measure "$destination/selection-replay" "$producer" replay-selection "$view"
  while IFS=$'\t' read -r arm task candidate; do
    cell="$arm--$task--$candidate"
    sha256sum -- "$view/$cell/result.json"
    if [[ -f "$view/$cell/model.json" ]]; then sha256sum -- "$view/$cell/model.json"; fi
  done < "$parent/planned.tsv" > "$destination/completed-endpoints.sha256"
  sha256sum -- "$view/summary.json" "$view/selected.json" > "$destination/comparison.sha256"
}

run() {
  local destination=$1 cell arm task candidate rest auditor available failed=0
  check_recovery "$destination"
  while IFS= read -r cell; do
    [[ ! -e "$destination/cells/$cell" && ! -e "$destination/$cell-train.exit" ]] || fail 'Recovery refuses to resume or overwrite a started attempt.'
  done < <(jq -r '.missing[]' "$protocol")
  while IFS= read -r cell; do
    available=$(df --output=avail -B1 "$destination" | tail -n 1)
    (( available >= $(field .disk_reserve_bytes) )) || fail 'Disk reserve reached; retained evidence is unchanged.'
    arm=${cell%%--*}; rest=${cell#*--}; task=${rest%--*}; candidate=${rest##*--}
    printf '%s\n' "Starting $cell"
    if ! measure "$destination/$cell-train" "$producer" run "$arm" "$task" "$candidate" "$destination/cells/$cell"; then
      failed=$((failed+1)); continue
    fi
    if ! measure "$destination/$cell-replay" "$producer" replay "$destination/cells/$cell"; then
      failed=$((failed+1)); continue
    fi
    cp -- "$destination/$cell-replay.stdout" "$destination/cells/$cell/replay.json"
    if [[ "$candidate" == counts ]]; then auditor=$counts_auditor; else auditor=$moments_auditor; fi
    if ! measure "$destination/$cell-audit" "$auditor" "$destination/cells/$cell" "$fingerprint"; then
      failed=$((failed+1))
      continue
    fi
    jq -e '.independent_audit==true' "$destination/$cell-audit.stdout" > /dev/null
    sha256sum -- "$destination/cells/$cell"/*.json "$destination/cells/$cell/trace.jsonl" \
      "$destination/$cell-audit.stdout" "$destination/$cell-replay.stdout" \
      "$destination/$cell-train.exit" "$destination/$cell-replay.exit" "$destination/$cell-audit.exit" \
      > "$destination/$cell-seal.sha256"
  done < <(jq -r '.missing[]' "$protocol")
  (( failed == 0 )) || fail "$failed recovery attempts failed; all artifacts retained."
  finish "$destination"
}

case "${1:-}" in
  preflight) [[ $# == 1 ]] || fail 'usage: run.sh preflight'; check_parent; retained_files > "$scratch/retained-paths.txt"; printf '%s\n' 'Original 311 results, four interruptions, models, replays and audits verified.' ;;
  prepare) [[ $# == 2 ]] || fail 'usage: run.sh prepare FRESH_ROOT'; prepare "$2" ;;
  run) [[ $# == 2 ]] || fail 'usage: run.sh run ROOT'; run "$(realpath -- "$2")" ;;
  finish) [[ $# == 2 ]] || fail 'usage: run.sh finish ROOT'; finish "$(realpath -- "$2")" ;;
  verify) [[ $# == 2 ]] || fail 'usage: run.sh verify ROOT'; check_recovery "$2"; sha256sum --check --quiet "$2/completed-endpoints.sha256" "$2/comparison.sha256"; prlimit --as="$limit:$limit" -- "$producer" replay-selection "$2/view" ;;
  *) fail 'usage: run.sh preflight | prepare FRESH_ROOT | run ROOT | finish ROOT | verify ROOT' ;;
esac
