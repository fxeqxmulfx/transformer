#!/usr/bin/env bash
# Accuracy per ordinary execution, plus the cumulative count of verified first hits.
set -euo pipefail
if (( $# != 1 )); then echo 'usage: bash plot_stock_first_hit.sh COHORT' >&2; exit 2; fi
out=$(realpath -e -- "$1")
tmp=$(mktemp -d)
trap 'rm -f -- "$tmp"/{curves.tsv,hits.tsv,figure.pdf,figure.png}; rmdir -- "$tmp"' EXIT
cd -- "$tmp"
task=$(jq -er .case "$out/protocol.json")
n=$(jq -er '.total | select(.>0)' "$out/summary.json")
for ((i=0;i<n;i++)); do
    seed=$(jq -er --argjson i "$i" '.trials[$i].seed' "$out/summary.json")
    src="$out/trial-$i/$task/seed$seed/observations.jsonl"
    if [[ -s $src ]]; then
        jq -r '[.step,100*.correct/.total]|@tsv' "$src" >> curves.tsv
        printf '\n\n' >> curves.tsv
    fi
done
jq -r '[[0,0]]+((.trials|map(select(.learned))|group_by(.step))|
    reduce .[] as $g ([]; .+[[$g[0].step,((.[-1][1]//0)+($g|length))]])) | .[]|@tsv' "$out/summary.json" >hits.tsv
gnuplot -e "n=$n" - <<'GP'
set terminal pdfcairo size 10in,4.1in font 'DejaVu Sans,10'
set output 'figure.pdf'
set multiplot layout 1,2 margins 0.075,0.975,0.17,0.80 spacing 0.12,0.0 title 'Ordinary AdamW on division modulo 97'
set key off
set xlabel 'Parameter updates'
set border 3 lc rgb '#52616a'
set tics nomirror out
set grid ytics lc rgb '#dde4e6'
set ylabel 'Hidden accuracy (%)'
set yrange [0:103]
set title 'One curve per stock initialization' font ',11'
plot 'curves.tsv' using 1:2 with lines lw 1.5 lc rgb '#318090'
set ylabel 'Runs with a strict first hit'
set yrange [0:n+0.5]
set ytics 1
set title 'All 4,656 hidden equations correct' font ',11'
plot 'hits.tsv' using 1:2 with steps lw 2.4 lc rgb '#185968', '' using 1:2 with points pt 7 ps .5 lc rgb '#185968'
unset multiplot
GP
pdftoppm -singlefile -r 150 -png figure.pdf figure
mv -- figure.pdf "$out/ordinary-adamw-accuracy.pdf"
mv -- figure.png "$out/ordinary-adamw-accuracy.png"
