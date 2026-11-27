#!/usr/bin/env bash
# Compare completed native AdamW trajectories with the same mask and WSD budget.
set -euo pipefail
if (( $# != 3 )); then
    echo 'usage: bash plot_division_seeds.sh FIRST_CELL SECOND_CELL OUTPUT_DIRECTORY' >&2
    exit 2
fi
mkdir -p -- "$3"
out=$(cd -- "$3" && pwd)
read -r seed_a seed_b steps cool exact second < <(
    jq -esr '
      def settings: .identity | del(.seed,.order_seed);
      if length==2 and (.[0]|settings)==(.[1]|settings)
          and (.[0].identity.case=="grok-division-p97") and (.[0].identity.optimizer=="adamw-wsd")
          and all(.[]; .step==.identity.budget and .total==4656 and .training_rows==4656)
      then [.[0].identity.seed,.[1].identity.seed,.[0].step,.[0].identity.adamw.cooldown_start,
            (.[0].first_strict_step // -1), (100*.[1].correct/.[1].total)] | @tsv
      else error("incompatible division endpoints") end' "$1/result.json" "$2/result.json"
)
for i in 1 2; do
    jq -r '[.step/1000,100*.correct/.total,100*.trained/.training_rows] | @tsv' \
        "${!i}/observations.jsonl" > "$out/division-seed-$i.tsv"
done
cd -- "$out"
gnuplot -e "s1=$seed_a; s2=$seed_b; budget=$steps/1000.; cool=$cool/1000.; exact=$exact/1000.; second=$second" - <<'GP'
set terminal pdfcairo color size 9.6in,5.5in font "DejaVu Sans,10"
set output 'division-seeds.pdf'
set multiplot
set lmargin at screen 0.09
set rmargin at screen 0.95
set tmargin at screen 0.84
set bmargin at screen 0.41
set label 1 'Division modulo 97' at screen 0.09,0.955 front font ",20" tc rgb '#172c38'
set label 2 'Argand  |  AdamW, weight decay 0.1  |  4,656 observed and 4,656 held-out equations' at screen 0.09,0.902 font ",10" tc rgb '#3e5261'
set xrange [0:budget]
set yrange [-1:105]
set ytics 25
set border 3 lc rgb '#8e9ba4'
set tics nomirror out scale 0.5
set grid ytics lc rgb '#e0e5e8' lw 0.5
set object 1 rect from first cool,graph 0 to graph 1,graph 1 behind fc rgb '#eef1f5' fs solid 1.0 noborder
set arrow 1 from first cool,graph 0 to first cool,graph 1 nohead dt 3 lc rgb '#8e9ba4' back
set ylabel 'Held-out accuracy (%)'
set format x ''
set style line 1 lc rgb '#006d73' lw 2.6 pt 7 ps 0.24
set style line 2 lc rgb '#b54d23' lw 2.6 pt 7 ps 0.24
set key top left reverse Left samplen 2.5 spacing 1.15
if (exact >= 0) {set label 3 sprintf('100%% at %gk',exact) at first exact+0.18, first 94 left tc rgb '#006d73' font ",9"}
set label 4 sprintf('%.2f%%',second) at first budget-0.2,first second-7 right tc rgb '#963f1b' font ",9"
set label 5 'WSD cooldown' at first cool+0.25,first 12 tc rgb '#546574' font ",9"
plot 'division-seed-1.tsv' u 1:2 w lp ls 1 title sprintf('seed %.0f',s1), \
     'division-seed-2.tsv' u 1:2 w lp ls 2 title sprintf('seed %.0f',s2)
unset label
set tmargin at screen 0.32
set bmargin at screen 0.13
set format x '%g'
set xlabel 'Minibatch updates (thousands)'
set ylabel 'Train (%)'
set ytics 50
unset key
plot 'division-seed-1.tsv' u 1:3 w l ls 1, 'division-seed-2.tsv' u 1:3 w l ls 2 dt 2
unset multiplot
GP
pdftoppm -singlefile -r 150 -png division-seeds.pdf division-seeds
