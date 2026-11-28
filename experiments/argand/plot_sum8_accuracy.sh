#!/usr/bin/env bash
# Plot the replayed eight-run staircase cohort on its actual observation grid.
set -euo pipefail
if (( $# != 1 )); then
    echo 'usage: bash plot_sum8_accuracy.sh RUN_DIRECTORY' >&2
    exit 2
fi
dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd -- "$1"
jq -er -f "$dir/sum8_accuracy.jq" summary.json > accuracy.tsv
horizon=$(jq -er '.protocol.updates' summary.json)
optimizer=$(jq -er '.protocol.protocol | {
  "argand-sum8-adamw-wsd-v1":"AdamW + WSD", "argand-sum8-adaptive-page-v1":"PAGE",
  "argand-sum8-adamw-first-noise-v1":"AdamW + WSD, first-update noise",
  "argand-sum8-adaptive-page-sqrt-v1":"PAGE (square root)", "argand-sum8-adaptive-page-retry-v1":"PAGE (retry)"}[.]' summary.json)
gnuplot -e "horizon=$horizon; optimizer='$optimizer'" - <<'PLOT'
set terminal pdfcairo color size 9.2in,4.0in font "DejaVu Sans,10"
set output "accuracy.pdf"
set multiplot layout 1,2 margins 0.075,0.975,0.17,0.80 spacing 0.08,0.05 title sprintf("Eight-bit modular sum   |   %s   |   8 initializations, 31 observed and 225 hidden words",optimizer)
set border 3
set tics nomirror
set grid ytics lc rgb "#e4e7ec"
set yrange [0:103]
set ytics 20
set xlabel "Accepted optimizer updates"
set ylabel "Strictly correct words (%)"
set style fill transparent solid 0.22 noborder
set style line 1 lc rgb "#1763a0" lw 2 pt 7 ps 0.32
set style line 2 lc rgb "#a3451d" lw 2 pt 7 ps 0.32
set key left bottom samplen 2 font ",8.5"
set title "First 2,048 updates" font ",11"
set xrange [0:2048]
set xtics 512
plot "accuracy.tsv" u 1:3:4 w filledcurves lc rgb "#519ad2" notitle, \
     "" u 1:6:7 w filledcurves lc rgb "#e58756" notitle, \
     50 dt 3 lc rgb "#666666" notitle, \
     "accuracy.tsv" u 1:2 w linespoints ls 1 title "Observed, mean", \
     "" u 1:5 w linespoints ls 2 title "Hidden, mean"
unset key
unset ylabel
set title "Entire run, bands show the seed range" font ",11"
set xrange [0:horizon]
stride=horizon/4
set xtics 0,stride,horizon
replot
unset multiplot
PLOT
pdftoppm -r 150 -png -singlefile accuracy.pdf accuracy
