#!/usr/bin/env bash
# Fresh complete replays and corruption controls on copies, never on inputs.
set -euo pipefail
test "$#" -eq 6
task_workspace=$1
task_storage=$2
task_freeze=$3
task_cohort=$4
task_diagnostics=$5
task_review=$6
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
admit() {
 ps -e -o rss= | awk '{s+=$1*1024}END{if(s+34359738368>110000000000)exit 1}'
 test "$(df -B1 --output=avail "$task_workspace" | tail -n 1)" -gt 2147483648
 test "$(df -B1 --output=avail "$task_storage" | tail -n 1)" -gt 17179869184
}
admit
test ! -e "$task_review"
mkdir "$task_review"
(cd "$task_freeze" && sha256sum -c SHA256SUMS) > "$task_review/freeze-check.log"
for task_stage in {1..8}; do
 task_reference=$task_cohort/continue/old/cautious/mini/stage-$task_stage
 jq -c '{update,batch_indices_sha256,record_indices_sha256}' "$task_reference/training.jsonl" > "$task_review/sampler-reference.jsonl"
 for task_arm in continue grow; do
  for task_geometry in old hilbert; do
   for task_variant in cautious refined; do
    for task_mode in mini forward; do
     jq -c '{update,batch_indices_sha256,record_indices_sha256}' \
      "$task_cohort/$task_arm/$task_geometry/$task_variant/$task_mode/stage-$task_stage/training.jsonl" > "$task_review/sampler.jsonl"
     diff "$task_review/sampler-reference.jsonl" "$task_review/sampler.jsonl"
    done
   done
  done
 done
done
cp -a --reflink=auto "$task_cohort" "$task_review/cohort"
for task_arm in continue grow; do
 for task_geometry in old hilbert; do
  for task_variant in cautious refined; do
   for task_mode in mini forward; do
    rm "$task_review/cohort/$task_arm/$task_geometry/$task_variant/$task_mode/replay.json"
   done
  done
 done
done
admit
prlimit --as=34359738368:34359738368 -- "$task_freeze/convex-content-growth" replay \
 "$task_workspace" "$task_storage" "$task_review/cohort"
admit
prlimit --as=34359738368:34359738368 -- "$task_freeze/convex-content-growth" diagnose \
 "$task_workspace" "$task_storage" "$task_review/cohort" "$task_review/diagnostics"
diff -r "$task_diagnostics" "$task_review/diagnostics"
cp -a --reflink=auto "$task_review/cohort" "$task_review/faults"
task_original=$task_cohort/continue/old/cautious/mini
task_cell=$task_review/faults/continue/old/cautious/mini
task_original_stage=$task_original/stage-1
task_stage=$task_cell/stage-1
refuse() {
 local task_name=$1 task_action=$2 task_pattern=$3
 admit
 local task_args=("$task_workspace" "$task_storage" "$task_review/faults")
 if test "$task_action" = diagnose; then task_args+=("$task_review/$task_name-output"); fi
 if prlimit --as=34359738368:34359738368 -- "$task_freeze/convex-content-growth" "$task_action" \
  "${task_args[@]}" > "$task_review/$task_name.log" 2>&1; then
  printf 'Fault accepted: %s\n' "$task_name" >&2; exit 1
 fi
 rg -F "$task_pattern" "$task_review/$task_name.log"
}
rehash() {
 local task_file=$1 task_key=$2 task_hash
 task_hash=$(sha256sum "$task_stage/$task_file" | cut -d' ' -f1)
 jq --arg key "$task_key" --arg hash "$task_hash" '.[$key]=$hash' "$task_stage/training_receipt.json" > "$task_review/stage-receipt.tmp"
 mv "$task_review/stage-receipt.tmp" "$task_stage/training_receipt.json"
 task_hash=$(sha256sum "$task_stage/training_receipt.json" | cut -d' ' -f1)
 jq --arg hash "$task_hash" '.stage_receipts[0]=$hash' "$task_cell/training_receipt.json" > "$task_review/receipt.tmp"
 mv "$task_review/receipt.tmp" "$task_cell/training_receipt.json"
}
restore_receipts() {
 cp "$task_original_stage/training_receipt.json" "$task_stage/training_receipt.json"
 cp "$task_original/training_receipt.json" "$task_cell/training_receipt.json"
}
jq '.source_fingerprint="changed"' "$task_original/metadata.json" > "$task_cell/metadata.json"
refuse wrong-producer score 'incomplete or changed growth cohort seal'
cp "$task_original/metadata.json" "$task_cell/metadata.json"
printf '\n' >> "$task_stage/endpoint.json"
refuse changed-endpoint score 'changed sealed growth artifact'
cp "$task_original_stage/endpoint.json" "$task_stage/endpoint.json"
printf '\n' >> "$task_stage/cache.json"
refuse changed-cache-hash score 'changed sealed growth artifact'
cp "$task_original_stage/cache.json" "$task_stage/cache.json"
jq '.mean[0] += 0.001' "$task_original_stage/cache.json" > "$task_stage/cache.json"
rehash cache.json cache_sha256
refuse wrong-retained-mean diagnose 'retained finite estimate cache drift'
cp "$task_original_stage/cache.json" "$task_stage/cache.json"
restore_receipts
jq '.values[0][0] += 0.001' "$task_original_stage/cache.json" > "$task_stage/cache.json"
rehash cache.json cache_sha256
refuse wrong-historical-estimate diagnose 'cached estimate disagrees with its actual recorded origin'
cp "$task_original_stage/cache.json" "$task_stage/cache.json"
restore_receipts
jq -c 'if .version==1 then .block_sha256="changed" else . end' "$task_original_stage/history.jsonl" > "$task_stage/history.jsonl"
rehash history.jsonl history_sha256
refuse wrong-history-version-hash diagnose 'historical parameter hash or version mismatch'
cp "$task_original_stage/history.jsonl" "$task_stage/history.jsonl"
restore_receipts
jq '.parent_fraction -= 0.0009765625 | .blocks[0].weight += 0.015625' "$task_original_stage/endpoint.json" > "$task_stage/endpoint.json"
rehash endpoint.json endpoint_sha256
refuse wrong-held-base diagnose 'stage initial state or held mixture changed'
cp "$task_original_stage/endpoint.json" "$task_stage/endpoint.json"
restore_receipts
diff -r "$task_review/cohort" "$task_review/faults"
printf 'All80 checkpoint reports,128 historical caches,16 final current diagnostics and seven corruption refusals pass.\n'
