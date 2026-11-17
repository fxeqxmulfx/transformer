#!/usr/bin/env bash
# Fresh replay and deliberate corruption controls. Original evidence is read-only.
set -euo pipefail
test "$#" -eq 6
task_workspace=$1
task_storage=$2
task_freeze=$3
task_cohort=$4
task_diagnostics=$5
task_review=$6
admit() {
 ps -e -o rss= | awk '{s+=$1*1024}END{if(s+34359738368>110000000000)exit 1}'
 test "$(df -B1 --output=avail "$task_workspace" | tail -n 1)" -gt 2147483648
 test "$(df -B1 --output=avail "$task_storage" | tail -n 1)" -gt 17179869184
}
admit
test ! -e "$task_review"
mkdir "$task_review"
(cd "$task_freeze" && sha256sum -c SHA256SUMS) > "$task_review/freeze-check.log"
for task_variant in cautious refined; do
 for task_mode in mini forward; do
  for task_arm in old hilbert; do
   jq -c '{update,batch_indices_sha256,record_indices_sha256}' \
    "$task_cohort/$task_arm/$task_variant/$task_mode/training.jsonl" > "$task_review/sampler-$task_arm.jsonl"
   head -n 1 "$task_cohort/$task_arm/$task_variant/$task_mode/training.jsonl" | \
    jq '.centre_ce' > "$task_review/initial-$task_arm.json"
  done
  diff "$task_review/sampler-old.jsonl" "$task_review/sampler-hilbert.jsonl"
  diff "$task_review/initial-old.json" "$task_review/initial-hilbert.json"
 done
done
cp -a --reflink=auto "$task_cohort" "$task_review/cohort"
for task_arm in old hilbert; do
 for task_variant in cautious refined; do
  for task_mode in mini forward; do
   admit
   rm "$task_review/cohort/$task_arm/$task_variant/$task_mode/replay.json"
   prlimit --as=34359738368:34359738368 -- "$task_freeze/convex-hilbert-content" replay \
    "$task_workspace" "$task_storage" "$task_review/cohort" "$task_arm" "$task_variant" "$task_mode"
  done
 done
done
admit
prlimit --as=34359738368:34359738368 -- "$task_freeze/convex-hilbert-content" diagnose \
 "$task_workspace" "$task_storage" "$task_review/cohort" "$task_review/diagnostics"
diff -r "$task_diagnostics" "$task_review/diagnostics"
cp -a --reflink=auto "$task_review/cohort" "$task_review/faults"
task_original=$task_cohort/old/cautious/mini
task_cell=$task_review/faults/old/cautious/mini
refuse_score() {
 local task_name=$1 task_pattern=$2
 admit
 if prlimit --as=34359738368:34359738368 -- "$task_freeze/convex-hilbert-content" score \
  "$task_workspace" "$task_storage" "$task_review/faults" old cautious mini > "$task_review/$task_name.log" 2>&1; then
  printf 'Fault accepted: %s\n' "$task_name" >&2; exit 1
 fi
 rg -F "$task_pattern" "$task_review/$task_name.log"
}
refuse_diagnostic() {
 local task_name=$1 task_pattern=$2
 admit
 if prlimit --as=34359738368:34359738368 -- "$task_freeze/convex-hilbert-content" diagnose \
  "$task_workspace" "$task_storage" "$task_review/faults" "$task_review/$task_name-output" > "$task_review/$task_name.log" 2>&1; then
  printf 'Fault accepted: %s\n' "$task_name" >&2; exit 1
 fi
 rg -F "$task_pattern" "$task_review/$task_name.log"
}
rehash() {
 local task_file=$1 task_key=$2
 local task_hash
 task_hash=$(sha256sum "$task_cell/$task_file" | cut -d' ' -f1)
 jq --arg key "$task_key" --arg hash "$task_hash" '.[$key]=$hash' "$task_cell/training_receipt.json" > "$task_review/receipt.tmp"
 mv "$task_review/receipt.tmp" "$task_cell/training_receipt.json"
}
jq '.source_fingerprint="changed"' "$task_original/metadata.json" > "$task_cell/metadata.json"
refuse_score wrong-producer 'incomplete or changed Hilbert cohort seal'
cp "$task_original/metadata.json" "$task_cell/metadata.json"
printf '\n' >> "$task_cell/endpoint.json"
refuse_score changed-endpoint 'changed sealed Hilbert artifact'
cp "$task_original/endpoint.json" "$task_cell/endpoint.json"
printf '\n' >> "$task_cell/cache.json"
refuse_score changed-cache-hash 'changed sealed Hilbert artifact'
cp "$task_original/cache.json" "$task_cell/cache.json"
jq '.mean[0] += 0.001' "$task_original/cache.json" > "$task_cell/cache.json"
rehash cache.json cache_sha256
refuse_diagnostic wrong-retained-mean 'retained finite estimate cache drift'
cp "$task_original/cache.json" "$task_cell/cache.json"
cp "$task_original/training_receipt.json" "$task_cell/training_receipt.json"
jq '.values[0][0] += 0.001' "$task_original/cache.json" > "$task_cell/cache.json"
rehash cache.json cache_sha256
refuse_diagnostic wrong-historical-estimate 'cached estimate disagrees with its actual recorded origin'
cp "$task_original/cache.json" "$task_cell/cache.json"
cp "$task_original/training_receipt.json" "$task_cell/training_receipt.json"
printf '\n' >> "$task_cell/history.jsonl"
refuse_score changed-history-hash 'changed sealed Hilbert artifact'
cp "$task_original/history.jsonl" "$task_cell/history.jsonl"
jq -c 'if .version==1 then .block_sha256="changed" else . end' "$task_original/history.jsonl" > "$task_cell/history.jsonl"
rehash history.jsonl history_sha256
refuse_diagnostic wrong-history-version-hash 'historical parameter hash or version mismatch'
cp "$task_original/history.jsonl" "$task_cell/history.jsonl"
cp "$task_original/training_receipt.json" "$task_cell/training_receipt.json"
diff -r "$task_review/cohort" "$task_review/faults"
printf 'All eight raw replays, eight historical/current diagnostics and seven corruption refusals pass.\n'
