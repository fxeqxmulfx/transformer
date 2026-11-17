#!/usr/bin/env bash
# Repeat every report and corrupt only a separate minimal copy of the inputs.
set -euo pipefail
test "$#" -eq 6
task_workspace=$1
task_storage=$2
task_freeze=$3
task_cohort=$4
task_reports=$5
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
prlimit --as=34359738368:34359738368 -- "$task_freeze/convex-binding-diagnostic" run \
 "$task_workspace" "$task_storage" "$task_cohort" "$task_review/reports"
diff -r "$task_reports" "$task_review/reports"
for task_arm in continue grow; do
 for task_geometry in old hilbert; do
  for task_variant in cautious refined; do
   for task_mode in mini forward; do
    admit
    task_rel=$task_arm/$task_geometry/$task_variant/$task_mode
    task_original=$task_cohort/$task_rel
    task_copy=$task_review/faults/$task_rel
    mkdir -p "$task_copy"
    cp "$task_original"/{metadata,training_receipt,scores,replay}.json "$task_copy/"
    for task_stage in {1..8}; do
     mkdir "$task_copy/stage-$task_stage"
     cp "$task_original/stage-$task_stage"/{training_receipt,endpoint}.json "$task_copy/stage-$task_stage/"
    done
   done
  done
 done
done
admit
prlimit --as=34359738368:34359738368 -- "$task_freeze/convex-binding-diagnostic" verify \
 "$task_workspace" "$task_storage" "$task_review/faults"
task_original=$task_cohort/continue/old/cautious/mini
task_copy=$task_review/faults/continue/old/cautious/mini
for task_file in metadata.json scores.json stage-8/endpoint.json stage-1/endpoint.json; do
 printf '\n' >> "$task_copy/$task_file"
 task_name=${task_file//\//-}
 admit
 if prlimit --as=34359738368:34359738368 -- "$task_freeze/convex-binding-diagnostic" verify \
  "$task_workspace" "$task_storage" "$task_review/faults" > "$task_review/$task_name.log" 2>&1; then
  printf 'Changed input accepted: %s\n' "$task_file" >&2; exit 1
 fi
 if test "$task_file" = stage-1/endpoint.json; then
  rg -F 'binding input stage changed' "$task_review/$task_name.log"
 else
  rg -F 'immutable binding input hash changed' "$task_review/$task_name.log"
 fi
 cp "$task_original/$task_file" "$task_copy/$task_file"
done
admit
prlimit --as=34359738368:34359738368 -- "$task_freeze/convex-binding-diagnostic" verify \
 "$task_workspace" "$task_storage" "$task_review/faults"
printf 'All24 complete reports repeat identically; all four changed-input controls are refused and restored.\n'
