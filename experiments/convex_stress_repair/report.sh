#!/usr/bin/env bash
# Summarize every planned cell, including process errors and pending replays.
set -euo pipefail
repair_cohort=${1:?supply a cohort directory}
repair_protocol=${2:-"$(dirname -- "${BASH_SOURCE[0]}")/protocol.json"}
repair_plan=${3:-"$repair_protocol"}
repair_rows=$(mktemp "$repair_cohort/report-rows.XXXXXX")
trap 'rm -f -- "$repair_rows"' EXIT
while IFS=$'\t' read -r arm task seed scale; do
    cell="$repair_cohort/scale$scale/$arm/$task/seed$seed"
    result_file=/dev/null
    run_exit=null
    replay_exit=null
    if test -f "$cell/result.json"; then result_file="$cell/result.json"; fi
    if test -f "$cell.exit"; then run_exit=$(cat "$cell.exit"); fi
    if test -f "$cell.replay.exit"; then replay_exit=$(cat "$cell.replay.exit"); fi
    jq -nc --arg arm "$arm" --arg task "$task" --argjson seed "$seed" --argjson scale "$scale" \
        --slurpfile result "$result_file" --argjson run_exit "$run_exit" --argjson replay_exit "$replay_exit" \
        '{arm:$arm,task:$task,seed:$seed,scale:$scale,run_exit:$run_exit,replay_exit:$replay_exit,result:($result[0]//null)}' >> "$repair_rows"
done < "$repair_cohort/jobs.tsv"
jq -s --slurpfile protocol "$repair_protocol" --slurpfile plan "$repair_plan" --arg fingerprint "$(cat "$repair_cohort/source_fingerprint.txt")" '
  def identity: [.arm,.task,.seed,.scale];
  ([$plan[0] | .pilot_tasks[] as $task | .arms[] as $arm | .scales[] as $scale |
    [$arm,$task,.pilot_seed,$scale]]|sort) as $expected |
  if (map(identity)|sort) != $expected then error("planned inventory differs from protocol") else . end |
  if (map(identity)|unique|length) != length then error("duplicate planned cells") else . end |
  map(. as $cell | if .result != null and
      ((.result|identity) != identity or .result.source_fingerprint != $fingerprint or
       .result.protocol != $protocol[0] or .result.status != "complete")
      then error("result identity or source mismatch") else . end |
      . + {status:(if .run_exit != null and .run_exit != 0 then "execution_error"
        elif .result == null or .run_exit == null then "pending"
        elif .replay_exit == null then "replay_pending"
        elif .replay_exit != 0 then "replay_error"
        elif .result.splits == null then "no_consistent_key"
        else "replayed" end)} |
      . + {primary:(if .result.splits == null then null else .result.splits[.result.selection] end)} |
      . + {passed:(.status == "replayed" and .primary != null and .primary.exact * 100 >= .primary.examples * 99)} ) |
  {source_fingerprint:$fingerprint,planned:length,
    statuses:(group_by(.status)|map({key:.[0].status,value:length})|from_entries),
    totals:(group_by([.arm,.scale])|map({arm:.[0].arm,scale:.[0].scale,planned:length,
      passed:map(select(.passed))|length,replay_verified:map(select(.replay_exit==0))|length})),
    cells:.}' "$repair_rows" > "$repair_cohort/summary.json"
