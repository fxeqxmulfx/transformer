#!/usr/bin/env bash
# Stage both canonical tiers at scheduled observations and independently replay endpoints.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
suite="$here/../convex_stress/suite.json"
bin=${INKLING_BIN:-"$here/target/release/inkling-stress"}
case "${1:-}" in
  cell)
    out=$2
    case_name=$3
    seed=$4
    limit=$5
    dir="$out/$case_name/seed$seed"
    mkdir -p "$dir"
    # Several launchers may advance the same cell. Inspect and publish under one lease.
    exec 8>"$dir/queue.lock"
    flock 8
    if [[ ! -s "$dir/latest.json" ]] ||
       jq -e --argjson limit "$limit" '.status=="in progress" and .step<$limit' "$dir/latest.json" >/dev/null; then
      if ! "$bin" run "$case_name" "$seed" "$out" "$limit" >/dev/null; then
        # A separate direct run may have advanced while this process waited for the cell.
        jq -e --argjson limit "$limit" '.step>=$limit or .status=="passed"' "$dir/latest.json" >/dev/null || exit 1
      fi
    fi
    if jq -e '.status!="in progress"' "$dir/latest.json" >/dev/null && [[ ! -s "$dir/replay.json" ]]; then
      "$bin" replay "$case_name" "$seed" "$out" >"$dir/replay.$$.pending"
      mv "$dir/replay.$$.pending" "$dir/replay.json"
    fi
    ;;
  run)
    out=${2:?usage: cohort.sh run OUT [max observation multiple]}
    stop=${3:-32768}
    jobs=${INKLING_JOBS:-1}
    [[ $stop =~ ^[1-9][0-9]*$ && $jobs =~ ^[1-9][0-9]*$ ]]
    mkdir -p "$out"
    for ((round=1; round<=stop; round*=2)); do
      jq -r --argjson round "$round" '
        (.tasks|map(select(.tier=="short"))) as $s |
        (.tasks|map(select(.tier=="long"))) as $l |
        range(0;3) as $seed | range(0; $s|length) as $i | ($s[$i],$l[$i]) |
        select(.!=null) | [.name,$seed,([.case.observe*$round,.case.max_updates]|min)] | @tsv
      ' "$suite" | xargs -r -n3 -P "$jobs" bash "$here/cohort.sh" cell "$out"
    done
    ;;
  report)
    out=${2:?usage: cohort.sh report OUT}
    jq -r '.tasks[]|. as $c|range(0;3)|[$c.name,$c.tier,.]|@tsv' "$suite" |
      while IFS=$'\t' read -r case_name tier seed; do
        dir="$out/$case_name/seed$seed"
        if [[ ! -s "$dir/latest.json" ]]; then
          jq -nc --arg name "$case_name" --arg tier "$tier" --argjson seed "$seed" \
            '{case:$name,tier:$tier,seed:$seed,status:"unstarted",verified:false}'
          continue
        fi
        replay=/dev/null
        if [[ -s "$dir/replay.json" ]]; then replay="$dir/replay.json"; fi
        jq -ce --slurpfile replay "$replay" --arg name "$case_name" --arg tier "$tier" --argjson seed "$seed" '
          . as $v | ($replay[0]//{}) as $r |
          if ([$v.case,$v.tier,$v.seed,$v.model]!=[$name,$tier,$seed,"inkling2-text-8layer-8expert-top2-3pass-tied-fp32-v1"]
              or $v.total<=0 or $v.exact<0 or $v.exact>$v.total
              or ($v.status=="passed")!=($v.exact*100>=$v.total*99)
              or ([$v.observations[] | select(.exact*100>=.total*99)|.step] != (if $v.status=="passed" then [$v.step] else [] end))
              or ($v.status=="budget miss" and $v.step!=$v.original_cap))
          then error("invalid training endpoint for "+$name) else . end |
          ([$r.model,$r.case,$r.seed,$r.step,$r.checkpoint_sha256,$r.binding,$r.primary,$r.splits[$v.primary].exact,$r.splits[$v.primary].total]==
           [$v.model,$v.case,$v.seed,$v.step,$v.checkpoint_sha256,$v.binding,$v.primary,$v.exact,$v.total]) as $verified |
          if ($replay|length)>0 and ($verified|not) then error("inconsistent replay for "+$name) else
          {case,tier,seed,step,exact,total,status,verified:$verified,splits:$r.splits} end' "$dir/latest.json"
      done | jq -s '
        {short:map(select(.tier=="short")),long:map(select(.tier=="long"))} |
        with_entries(.value |= (. as $all | {total:length,passed:map(select(.verified and .status=="passed"))|length,
          awaiting_replay:map(select(.status=="passed" and (.verified|not)))|length,
          in_progress:map(select(.status=="in progress"))|length,budget_misses:map(select(.status=="budget miss"))|length,
          unstarted:map(select(.status=="unstarted"))|length,cells:$all}))'
    ;;
  *) echo "usage: cohort.sh run OUT [max observation multiple] | report OUT" >&2; exit 2 ;;
esac
