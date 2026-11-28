#!/usr/bin/env bash
# Plot recorded trajectories and independently replayed fixed-horizon scores.
set -euo pipefail
if (( $# != 2 )); then
    echo 'usage: bash plot_division_cohort.sh COHORT OUTPUT_DIRECTORY' >&2
    exit 2
fi
dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cohort=$(cd -- "$1" && pwd)
mkdir -p -- "$2"
out=$(cd -- "$2" && pwd)
tmp=$(mktemp -d "$out/.division-cohort.XXXXXX")
trap 'rm -f -- "$tmp"/*.tsv "$tmp"/snapshot.{pdf,png} "$tmp/verified.json"; rmdir -- "$tmp"' EXIT
cd -- "$tmp"
plan="$cohort/protocol.json"
budget=$(jq -er 'select(.case=="grok-division-p97" and .correction=="adamw-wsd"
    and .n==12 and (.seeds|length)==12) | .updates | select(.>0)' "$plan")
done_count=0
successes=0
started=0
labels=''
last_draw=0
last_step=0
archives=0
touch recent.tsv complete.tsv
trajectory_tsv() {
    # The native writer terminates each evaluation with a newline.
    # A partial final record can already end in a nested closing brace.
    sed -z 's/[^\n]*$//' "$1" |
        jq -r --argjson seed "$2" --argjson budget "$budget" '
          if .identity.case=="grok-division-p97" and .identity.optimizer=="adamw-wsd"
            and .identity.seed==$seed and .identity.order_seed==(($seed+10000)%4294967296)
            and .identity.fixed_horizon and .identity.budget==$budget and .total==4656
            and .identity.split_sha256==["53e9d2fb9336a74529e88bf582a2fb37483c1f7e79ab8f1fd670de565baf0eef",
                "1f679f635bd75862e28f1f6dec8cc121a2963f2fa258a0d8d291df0c8b9acc7d"]
            and (.step>=0 and .step<=$budget) and (.correct>=0 and .correct<=.total)
          then [.step,100*.correct/.total,.total-.correct]|@tsv else error("unexpected trajectory") end'
}
for ((i=0; i<12; i++)); do
    cell="$cohort/draw-$i"
    seed=$(jq -er --argjson i "$i" '.seeds[$i][0]' "$plan")
    draw=$((i+1))
    labels+="${labels:+,}'$draw    $seed' $draw"
    for prefix in "$cell"/interrupted/step-*/observations.jsonl; do
        [[ -s "$prefix" ]] || continue
        trajectory_tsv "$prefix" "$seed" >> interrupted.tsv
        printf '\n' >> interrupted.tsv
        ((archives+=1))
    done
    [[ -s "$cell/observations.jsonl" ]] || continue
    trajectory_tsv "$cell/observations.jsonl" "$seed" > "$draw.tsv"
    [[ -s "$draw.tsv" ]] || continue
    read -r step _ errors < <(tail -1 "$draw.tsv")
    ((started+=1))
    last_draw=$draw
    last_step=$step
    if [[ -s "$cell/replay.json" ]]; then
        read -r strict ok < <(jq -er --argjson seed "$seed" --argjson budget "$budget" '
          if .step==$budget and .identity.seed==$seed and .identity.budget==$budget
            and .identity.case=="grok-division-p97" and .identity.optimizer=="adamw-wsd"
            and .identity.fixed_horizon and .strict.total==4656 and .splits.hidden.total==4656
          then [(.strict.total-.strict.strict_correct), (if .strict.all_strict and .splits.hidden.correct==4656
            then 1 else 0 end)]|@tsv else error("unexpected terminal replay") end' "$cell/replay.json")
        ((done_count+=1))
        successes=$((successes+ok))
        echo "$draw $((1+strict))" >> complete.tsv
    else
        echo "$draw $((1+errors))" >> recent.tsv
    fi
done
if (( started == 0 )); then echo 'no recorded evaluations yet' >&2; exit 1; fi
[[ -s recent.tsv ]] || echo '1 NaN' > recent.tsv
[[ -s complete.tsv ]] || echo '1 NaN' > complete.tsv
[[ -s interrupted.tsv ]] || echo '0 NaN NaN' > interrupted.tsv
series=$(for f in {1..12}; do if [[ -s "$f.tsv" ]]; then echo "$f"; fi; done | paste -sd' ')
gnuplot -e "series='$series'; labels=\"$labels\"; budget=$budget; done_count=$done_count; successes=$successes; archives=$archives" - <<'GP'
set terminal pdfcairo color size 12in,6.4in font "DejaVu Sans,10"
set output 'snapshot.pdf'
set multiplot
set label 1 'Division modulo 97 across 12 initializations' at screen 0.073,0.946 font ",18" tc rgb '#172c38'
set label 2 sprintf('Argand   |   AdamW, weight decay 0.1   |   %d updates per seed   |   4,656 held-out equations',budget) at screen 0.073,0.886 tc rgb '#405563'
set lmargin at screen 0.073
set rmargin at screen 0.536
set bmargin at screen 0.535
set tmargin at screen 0.77
set border 3 lc rgb '#87959e'
set tics nomirror out scale 0.5
set grid ytics lc rgb '#e1e6e9'
set format x ''
set ylabel 'Held-out accuracy (%)'
set title 'Recorded accuracy during training'
set yrange [-2:103]
set ytics 25
set xrange [0:budget/1000.]
cooldown=0.8*budget/1000.
set arrow 1 from first cooldown,graph 0 to first cooldown,graph 1 nohead dt 2 lw 1 lc rgb '#81929b' back
unset key
colors='#176a73 #b55724 #604aa5 #a22c59 #397b3c #335ea8 #9a6b17 #026b9c #7d476c #737c22 #3c595d #a03432'
if (archives>0) { set label 6 'Dashed gray = interrupted attempt' at screen 0.073,0.038 font ",9" tc rgb '#536572' }
plot for [j=1:(archives>0 ? 1 : 0)] 'interrupted.tsv' u ($1/1000.):2 w l dt 3 lw 2 lc rgb '#748591', \
     for [k in series] sprintf('%s.tsv',k) u ($1/1000.):2 w l lw 1.8 lc rgb word(colors,int(k))
unset label
set bmargin at screen 0.135
set tmargin at screen 0.45
set xlabel 'Minibatch updates (thousands)'
set format x '%g'
set ylabel 'Held-out errors'
unset title
set logscale y
set yrange [0.78:7000]
set ytics ('0' 1,'1' 2,'10' 11,'100' 101,'1,000' 1001,'4,656' 4657)
set label 5 'WSD cooldown' at first cooldown,graph 1.025 center font ",8" tc rgb '#455864'
plot for [j=1:(archives>0 ? 1 : 0)] 'interrupted.tsv' u ($1/1000.):(1+$3) w l dt 3 lw 1.6 lc rgb '#748591', \
     for [k in series] sprintf('%s.tsv',k) u ($1/1000.):(1+$3) w l lw 1.4 lc rgb word(colors,int(k))
unset arrow 1
unset label 5
unset logscale y
set bmargin at screen 0.135
set tmargin at screen 0.77
set lmargin at screen 0.715
set rmargin at screen 0.965
unset ylabel
set title (done_count==0 ? 'Awaiting the first terminal evaluation' : sprintf('Terminal successes: %d of %d evaluated',successes,done_count))
set xlabel 'Held-out errors (logarithmic spacing)'
set yrange [12.65:0.35]
set xrange [0.76:6500]
set logscale x
set xtics ('0' 1,'1' 2,'10' 11,'100' 101,'1,000' 1001,'4,656' 4657) font ",8"
eval 'set ytics ('.labels.') font ",8"'
set label 3 'Draw      Seed' at graph -0.055,1.045 right font ",8" tc rgb '#435563'
set label 4 'Open circle = in progress     Diamond = terminal replay' at screen 0.610,0.038 font ",9" tc rgb '#405563'
plot 'recent.tsv' u 2:1:(rgbcolor(word(colors,int($1)))) w p pt 6 ps 1.5 lw 2 lc rgb variable, \
     'complete.tsv' u 2:1:(rgbcolor(word(colors,int($1)))) w p pt 13 ps 1.5 lc rgb variable
unset multiplot
GP
pdftoppm -singlefile -r 135 -png snapshot.pdf snapshot
base="division-${budget}-cohort"
if [[ $budget == 100000 || $budget == 200000 ]]; then
    bash "$dir/report_division_horizon.sh" "$cohort" > verified.json
    mv -- verified.json "$out/$base-verified.json"
fi
if [[ ${ARGAND_PLOT_SNAPSHOT:-0} == 1 ]]; then
    tag="$base-draw-$last_draw-step-$last_step-done-$done_count"
    if ((archives>0)); then tag+="-retry-$archives"; fi
    for ext in pdf png; do
        [[ -e "$out/$tag.$ext" ]] || cp -- "snapshot.$ext" "$out/$tag.$ext"
    done
fi
mv -- snapshot.pdf "$out/$base.pdf"
mv -- snapshot.png "$out/$base.png"
echo "$last_draw draws started, latest step $last_step; $successes strict successes in $done_count terminal replays"
