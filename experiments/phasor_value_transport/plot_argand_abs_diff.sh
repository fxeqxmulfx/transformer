#!/usr/bin/env bash
# Plot the paired 988-update blind absolute-difference comparison from native runner logs.
set -euo pipefail
if (( $# != 1 )); then
    echo 'usage: plot_argand_abs_diff.sh RUN_DIRECTORY' >&2
    exit 2
fi
cd -- "$1"
mkdir -p curves
task=arithmetic-abs-diff-blind
for arm in single norm-phase write-gate read-write-gate skip-write-gate entry-phase; do
    jq -ers --arg task "$task" '
        group_by(.step) | if map(.[0].step) == [range(0;989;19)]
            and all(.[]; (map(.identity.seed)|sort) == [0,1,2]
                and all(.[]; .identity.case == $task and .total == 4705))
        then .[] | [.[0].step, (map(100*.correct/.total)|add/length),
                    (map(.segment_seconds)|add/length)] | @tsv
        else error("expected all 53 observations for each of three seeds") end
    ' "$arm/$task"/seed{0,1,2}/observations.jsonl > "curves/$arm.tsv"
done

gnuplot <<'GNUPLOT'
set terminal pdfcairo color size 9in,4.1in font "DejaVu Sans,9"
set output "learning-curves.pdf"
arms = "single norm-phase write-gate read-write-gate skip-write-gate entry-phase"
array labels[6] = ["Ordinary", "Norm phase", "Write", "Read + write", "Skip + write", "Complex mHC"]
file(i) = "curves/".word(arms,i).".tsv"
set style line 1 lc rgb "#757575" lw 1.8
set style line 2 lc rgb "#252525" dt 3 lw 1.5
set style line 3 lc rgb "#0072B2" lw 2
set style line 4 lc rgb "#A34789" dt 2 lw 1.8
set style line 5 lc rgb "#00816B" lw 2
set style line 6 lc rgb "#C14D00" lw 2
set border 3
set tics nomirror
set grid ytics lc rgb "#dddddd"
set yrange [0:100]
set key left top samplen 1.6 spacing 0.95
set multiplot layout 1,2 margins 0.08,0.98,0.15,0.83 spacing 0.1,0.04 title "Blind absolute difference, mean across three seeds"
set ylabel "Held-out exact accuracy (%)"
set title "Equal optimizer updates"
set xlabel "Updates"
set xrange [0:1000]
plot for [i=1:6] file(i) using 1:2 with lines ls i title labels[i]
unset key
unset ylabel
set title "Elapsed training and scoring time"
set xlabel "Seconds on RTX 3050 Laptop GPU"
set xrange [0:*]
plot for [i=1:6] file(i) using 3:2 with lines ls i
unset multiplot
GNUPLOT
