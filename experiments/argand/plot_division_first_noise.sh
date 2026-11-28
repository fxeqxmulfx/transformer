#!/usr/bin/env bash
# Plot two matched, independently replayed fixed-horizon division trajectories.
set -euo pipefail
if (( $# != 3 )); then
    echo 'usage: bash plot_division_first_noise.sh STOCK_CELL FIRST_NOISE_CELL OUTPUT_DIRECTORY' >&2
    exit 2
fi
mkdir -p -- "$3"
out=$(cd -- "$3" && pwd)
tmp=$(mktemp -d "$out/.division-first-noise.XXXXXX")
trap 'rm -f -- "$tmp"/{1,2}.tsv "$tmp"/figure.{pdf,png}; rmdir -- "$tmp"' EXIT
jq -en --slurpfile a "$1/result.json" --slurpfile b "$2/result.json" '
  ($a[0].identity) as $x | ($b[0].identity) as $y |
  $x.case=="grok-division-p97" and $x.optimizer=="adamw-wsd" and $x.fixed_horizon and
  $y.optimizer=="adamw-wsd-first-noise" and $y.gaussian_weight_noise.after_update==1 and
  ($x | del(.optimizer)) == ($y | del(.optimizer,.gaussian_weight_noise))' > /dev/null
for i in 1 2; do
    cell=${!i}
    jq -res --slurpfile r "$cell/result.json" --slurpfile v "$cell/replay.json" '
      ($r[0]) as $r | ($v[0]) as $v |
      if .[0].step==0 and .[-1]==$r and $r.step==$r.identity.budget and
         ($r|{identity,step,weights_sha256,strict})==($v|{identity,step,weights_sha256,strict}) and
         $r.correct==$v.splits.hidden.correct and $r.total==$v.splits.hidden.total and
         all(.[]; .identity==$r.identity and .total==4656 and .training_rows==4656)
      then .[] | [.step,100*.correct/.total,.total-.correct] | @tsv
      else error("inconsistent trajectory or terminal replay") end' "$cell/observations.jsonl" > "$tmp/$i.tsv"
done
jq -en --slurpfile a "$1/observations.jsonl" --slurpfile b "$2/observations.jsonl" '
  def initial: {step,correct,trained,total,training_rows,loss};
  ($a | map(.step)) == ($b | map(.step)) and ($a[0] | initial) == ($b[0] | initial)' >/dev/null
read -r budget cool model_seed noise_seed sigma decay < <(jq -r '.identity |
  [.budget,.adamw.cooldown_start,.seed,.gaussian_weight_noise.seed,.gaussian_weight_noise.stddev,.adamw.weight_decay]|@tsv' "$2/result.json")
cd -- "$tmp"
gnuplot -e "budget=$budget; cool=$cool; model_seed=$model_seed; noise_seed=$noise_seed; sigma=$sigma; decay=$decay" - <<'GP'
set terminal pdfcairo color size 10.8in,4.8in font 'DejaVu Sans,10'
set output 'figure.pdf'
set multiplot layout 1,2 margins 0.08,0.975,0.15,0.76 spacing 0.10,0 title sprintf("Argand on division modulo 97   |   model seed %d\n4,656 held-out equations   ·   matched WSD   ·   weight decay %g",model_seed,decay) font ',14'
set border 3 lc rgb '#788895'
set tics nomirror out scale 0.5
set grid ytics lc rgb '#e0e6e9'
set xlabel 'Minibatch updates (thousands)'
set xrange [0:budget/1000.*1.02]
set arrow 1 from first cool/1000.,graph 0 to first cool/1000.,graph 1 nohead dt 3 lc rgb '#8b989f' back
set label 1 'WSD cooldown' at first cool/1000.,graph 1.03 center font ',8' tc rgb '#536471'
set style line 1 lc rgb '#13617a' lw 2.5
set style line 2 lc rgb '#b44328' lw 2.4 dt 2
set key bottom right samplen 2 font ',9'
set title 'Exact hidden accuracy'
set ylabel 'Correct (%)'
set yrange [-3:104]
set ytics 20
plot '1.tsv' u ($1/1000.):2 w l ls 1 t 'AdamW', \
     '2.tsv' u ($1/1000.):2 w l ls 2 t sprintf('AdamW + noise (σ = %g, seed %d)',sigma,noise_seed)
unset key
set title 'Errors near full recovery'
set ylabel 'Incorrect equations'
set logscale y
set yrange [0.77:6800]
set ytics ('0' 1,'1' 2,'10' 11,'100' 101,'1,000' 1001,'4,656' 4657)
plot '1.tsv' u ($1/1000.):(1+$3) w l ls 1, '2.tsv' u ($1/1000.):(1+$3) w l ls 2
unset multiplot
GP
pdftoppm -singlefile -r 150 -png figure.pdf figure
base="division-${budget}-adamw-first-noise-seed${model_seed}-noise${noise_seed}"
mv -- figure.pdf "$out/$base.pdf"
mv -- figure.png "$out/$base.png"
