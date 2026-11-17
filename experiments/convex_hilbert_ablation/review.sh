#!/usr/bin/env bash
# Repeat every original-score replay and ablation, then reject changed input copies.
set -euo pipefail
test "$#" -eq 6
task_workspace=$1
task_storage=$2
task_freeze=$3
task_cohort=$4
task_original=$5
task_review=$6
test ! -e "$task_review"
mkdir "$task_review"
bash "$task_freeze/run.sh" "$task_workspace" "$task_storage" "$task_freeze" "$task_cohort" "$task_review/report"
diff -r "$task_original" "$task_review/report"
for task_arm in old hilbert; do for task_variant in cautious refined; do for task_mode in mini forward; do
 task_from=$task_cohort/$task_arm/$task_variant/$task_mode
 task_to=$task_review/inputs/$task_arm/$task_variant/$task_mode
 mkdir -p "$task_to"
 for task_file in metadata.json training_receipt.json endpoint.json scores.json; do
  cp --reflink=auto "$task_from/$task_file" "$task_to/$task_file"
 done
done; done; done
task_cell=$task_review/inputs/old/cautious/mini
jq '.source_fingerprint="changed"' "$task_cell/metadata.json" > "$task_review/metadata.tmp"
mv "$task_review/metadata.tmp" "$task_cell/metadata.json"
if prlimit --as=34359738368:34359738368 -- "$task_freeze/convex-hilbert-ablation" verify \
 "$task_workspace" "$task_storage" "$task_review/inputs" > "$task_review/wrong-producer.log" 2>&1; then exit 1; fi
rg -F 'immutable ablation input hash changed' "$task_review/wrong-producer.log"
cp "$task_cohort/old/cautious/mini/metadata.json" "$task_cell/metadata.json"
printf '\n' >> "$task_cell/endpoint.json"
if prlimit --as=34359738368:34359738368 -- "$task_freeze/convex-hilbert-ablation" verify \
 "$task_workspace" "$task_storage" "$task_review/inputs" > "$task_review/wrong-endpoint.log" 2>&1; then exit 1; fi
rg -F 'immutable ablation input hash changed' "$task_review/wrong-endpoint.log"
cp "$task_cohort/old/cautious/mini/endpoint.json" "$task_cell/endpoint.json"
prlimit --as=34359738368:34359738368 -- "$task_freeze/convex-hilbert-ablation" verify \
 "$task_workspace" "$task_storage" "$task_review/inputs"
printf 'Eight intact original replays and32 ablation reports agree; both input corruptions refused.\n'
