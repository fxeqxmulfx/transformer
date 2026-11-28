#!/usr/bin/env bash
# Compare one registered native draw on the same WSD horizon. The rank-one
# cell may be in progress. Its trace contains only completed evaluations.
set -euo pipefail
if (( $# != 3 )); then echo 'usage: bash plot_sum8_rank_one.sh STOCK_CELL RANK_ONE_CELL OUTPUT' >&2; exit 2; fi
stock=$(realpath -e -- "$1")
cell=$(realpath -e -- "$2")
mkdir -p -- "$3"
out=$(realpath -e -- "$3")
idx=${cell##*/draw-}
[[ $idx =~ ^[0-7]$ ]]
tmp=$(mktemp -d)
trap 'rm -r -- "$tmp"' EXIT
cd -- "$tmp"
jq -en --slurpfile a "$stock/result.json" --slurpfile b "$cell/../protocol.json" --argjson i "$idx" '
    $a[0].protocol=="argand-sum8-adamw-wsd-v1" and $b[0].protocol=="argand-sum8-adamw-rank-one-gap-v1" and
    $a[0].seeds==$b[0].seeds[$i] and $a[0].updates==$b[0].updates' >/dev/null
jq -r '([.initial+{step:0}] + .milestones + [. + {step:.optimizer.step}])
    | .[] | [.step,.observed.correct,.hidden.correct] | @tsv' "$stock/result.json" > ordinary.tsv
{
    if [[ -s "$cell/result.json" ]]; then
        jq -r '.initial | [0,.observed.correct,.hidden.correct] | @tsv' "$cell/result.json"
    fi
    jq -r 'select(.observed!=null and .hidden!=null) | [.step,.observed.correct,.hidden.correct] | @tsv' \
        "$cell/training.jsonl"
} > penalty.tsv
last=$(tail -1 penalty.tsv | cut -f1)
(( last > 0 ))
seed=$(jq -r '.seeds[0]' "$stock/result.json")
gnuplot -e "last=$last; seed='$seed'" - <<'PLOT'
set terminal pdfcairo color size 9in,3.9in font 'DejaVu Sans,10'
set output 'figure.pdf'
set multiplot layout 1,2 margins 0.08,0.975,0.18,0.76 spacing 0.09,0.05 title sprintf('Argand on the eight-bit sum   |   seed %s   |   through update %d',seed,last)
set border 3
set tics nomirror
set grid ytics lc rgb '#e1e5e8'
set yrange [0:103]
set xrange [0:last]
set xlabel 'AdamW updates'
set ylabel 'Strict accuracy (%)'
set key bottom left font ',9' samplen 1.5
set title '31 observed words'
plot 'ordinary.tsv' u 1:(100*$2/31) w lines lw 1.8 lc rgb '#596974' title 'Observed CE', \
     'penalty.tsv' u 1:(100*$2/31) w lines lw 2.2 lc rgb '#b6552c' title 'CE + rank-one loss'
set title '225 hidden words'
unset ylabel
unset key
plot 50 dt 3 lc rgb '#777777' notitle, \
     'ordinary.tsv' u 1:(100*$3/225) w lines lw 1.8 lc rgb '#596974' notitle, \
     'penalty.tsv' u 1:(100*$3/225) w lines lw 2.2 lc rgb '#b6552c' notitle
unset multiplot
PLOT
pdftoppm -r 150 -png -singlefile figure.pdf figure
for ext in pdf png; do
    cp -- "figure.$ext" "$out/sum8-rank-one-draw-$idx-step-$last.$ext"
done
echo "sum8-rank-one-draw-$idx-step-$last"
