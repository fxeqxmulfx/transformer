#!/usr/bin/env bash
# A transparent view of 70 endpoints, with the two original failures retained.
set -euo pipefail
set_binary=${1:?frozen producer required}
set_original=$(realpath "${2:?original cohort required}")
set_recovery=$(realpath "${3:?recovery cohort required}")
set_output=${4:?fresh parent view required}
set_expected=$(jq -r '.producer_fingerprint' "$set_recovery/resource_override.json")
[[ "$("$set_binary" fingerprint)" == "$set_expected" ]]
[[ "$(cat "$set_original/fingerprint.txt")" == "$set_expected" ]]
mkdir -- "$set_output"
cp -- "$set_original/jobs.tsv" "$set_output/jobs.tsv"
cp -- "$set_original/fingerprint.txt" "$set_output/fingerprint.txt"
cp -- "$set_recovery/resource_override.json" "$set_output/resource_override.json"
set_attempts="$set_output/failed_attempts.jsonl"
: > "$set_attempts"
while IFS=$'\t' read -r arm task; do
  id="$arm--$task"
  source="$set_original"
  if [[ "$(cat "$source/$id.exit")" != 0 ]]; then
    jq -e --arg id "$id" '.cells | index($id) != null' "$set_recovery/resource_override.json" >/dev/null
    elapsed=$(awk '/Elapsed \(wall clock\)/ { n=split($NF,a,":"); s=0; for(i=1;i<=n;i++) s=60*s+a[i]; print s }' "$source/$id.log")
    digest=$(sha256sum "$source/$id.log" | cut -d ' ' -f 1)
    jq -n --arg id "$id" --arg log "$source/$id.log" --arg digest "$digest" --argjson seconds "$elapsed" \
      '{id:$id,log:$log,log_sha256:$digest,wall_seconds:$seconds,exit_code:134}' >> "$set_attempts"
    source="$set_recovery"
  fi
  [[ "$(cat "$source/$id.exit")" == 0 && "$(cat "$source/$id.replay.exit")" == 0 ]]
  for suffix in '' '.exit' '.replay.exit' '.log' '.replay.log'; do
    ln -s -- "$source/$id$suffix" "$set_output/$id$suffix"
  done
done < "$set_output/jobs.tsv"
jq -s --arg original "$set_original" --arg recovery "$set_recovery" \
  '{original_cohort:$original,recovery_cohort:$recovery,failed_attempts:.,
    failed_attempt_seconds:(map(.wall_seconds)|add // 0),
    description:"68 original endpoints and two resource-only retries; 72 total fit attempts. Original failures remain retained and costed."}' \
  "$set_attempts" > "$set_output/provenance.json"
"$set_binary" summary "$set_output" > "$set_output/summary.json"
