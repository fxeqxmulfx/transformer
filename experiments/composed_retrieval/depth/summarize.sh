#!/usr/bin/env bash
set -euo pipefail
(( $# == 1 )) || { echo 'usage: bash summarize.sh DIAGNOSTIC_OUTPUT' >&2; exit 2; }
for arm in original repeat; do test -d "$1/$arm/results"; done
diff -qr "$1/original/results" "$1/repeat/results" >&2
for mode in mini forward; do for seed in 0 1 2; do
  cell="$1/original/results/$mode-$seed"
  jq -c --slurpfile log "$cell/updates.json" '
    $log[0] as $s | select(($s|length)==1024) |
    {mode,seed,initial:$s[0],second:$s[1],final_train,
     first_distinct:([$s[]|select(.batch.distinguishable_rows>0)][0].update),
     first_zero:([$s[]|select(.batch.depth==0)][0].update),
     last_above_zero:([$s[]|select(.batch.depth>0)][-1].update),
     at_zero:([$s[]|select(.batch.depth==0)]|length),
     positive_at_zero:([$s[]|select(.batch.depth==0 and .batch.slope>1e-10)]|length),
     negative_at_zero:([$s[]|select(.batch.depth==0 and .batch.slope < -1e-10)]|length)}' "$cell/endpoint.json"
done; done | jq -s -e 'if length==6 then . else error("incomplete original trajectories") end'
