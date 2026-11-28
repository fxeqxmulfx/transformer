#!/usr/bin/env bash
# Plot each task's mean terminal primary-split accuracy across its three seeds.
set -euo pipefail
if (( $# < 3 )); then
    echo 'usage: bash plot_stress_accuracy.sh OUTPUT_DIR ALL_SUMMARY_JSON ALL_SUMMARY_JSON ...' >&2
    exit 2
fi
dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
mkdir -p -- "$1"
out=$(realpath -e -- "$1")
shift
tmp=$(mktemp -d "$out/.stress-figure.XXXXXXXX")
trap 'rm -f -- "$tmp"/{points.tsv,plot.gp,figure.pdf,figure.png}; rmdir -- "$tmp"' EXIT
for other in "${@:2}"; do
    jq -ne --slurpfile control "$1" --slurpfile candidate "$other" \
        -f "$dir/compare_optimizer_screens.jq" >/dev/null
done
for tier in short long; do
    jq -ers --arg tier "$tier" --slurpfile suite "$dir/../convex_stress/suite.json" '
      def caption:
        if .optimizer=="equal-moment-wsd" then "Equal decay"
        elif .optimizer=="uncorrected-adam-wsd" then "Uncorrected Adam"
        elif .optimizer=="normalized-adamw-wsd" then "Normalized AdamW"
        elif .optimizer=="adamw-wsd" then "AdamW 0.001"
        elif .optimizer=="adamw-scaled-wsd" then "AdamW 0.003"
        elif .optimizer=="adamw-wsd-first-noise" then "AdamW + noise"
        elif .optimizer=="batch-reference-adaptive-page-v1" then
          (if .correction=="one" then "PAGE" else "PAGE ("+.correction+")" end)
        else .optimizer end;
      [$suite[0].tasks[] | select(.tier==$tier)] as $tasks | $suite[0].seeds as $seeds |
      to_entries[] | .key as $column | .value as $screen | ($screen | caption) as $name |
      ($screen.cases | INDEX(.name)) as $cases |
      if ($screen.cases|length)!=($suite[0].tasks|length) then error("incomplete screen") else . end |
      $tasks | to_entries[] | .key as $row | .value as $task | $cases[$task.name] as $case |
      if $case.tier!=$tier or $case.total!=$task.heldout_examples or
         ($case.cells|map(.seed)|sort)!=($seeds|sort) then error("screen design mismatch at "+$task.name) else . end |
      ($case.cells|map(.correct)|add) as $correct | ($case.total*($seeds|length)) as $total |
      [$column,$row,(100*$correct/$total),$name,$case.name,($correct==$total)] | @tsv' "$@" > "$tmp/points.tsv"
    rows=$(awk -F '\t' '$1==0{n++} END{print n}' "$tmp/points.tsv")
    width=$(awk -v n="$#" 'BEGIN{print 3.2+1.35*n}')
    height=$(awk -v n="$rows" 'BEGIN{print 1.7+0.28*n}')
    (
        cd -- "$tmp"
        cat > plot.gp <<GP
set terminal pdfcairo noenhanced size ${width}in,${height}in color font 'DejaVu Sans,10'
set output 'figure.pdf'
set title 'GPT stress tests | $tier' font ',16'
set datafile separator '\t'
set xrange [-0.5:$#-0.5]
set yrange [$rows-0.5:-0.5]
set cbrange [0:100]
set palette defined (0 '#f2f5f5',25 '#c9e2e0',50 '#86c2b9',75 '#318c80',100 '#07574f')
unset colorbox
unset key
unset border
set tics scale 0
set xtics nomirror
set x2range [-0.5:$#-0.5]
set x2tics
unset xtics
set ytics nomirror
set lmargin 32
set rmargin 2
set bmargin 3.1
set tmargin 4.8
set label 1 'Mean held-out accuracy (%) at the recorded endpoint, three seeds per task.' at screen 0.03,0.04 left
plot 'points.tsv' using 1:2:(0.49):(0.48):3:x2ticlabels(4):yticlabels(5) \
     with boxxyerrorbars fs solid 1.0 noborder lc palette, \
     '' using 1:2:(strcol(6) eq 'true' ? '100' : (\$3>99.95 ? '>99.9' : sprintf('%.1f',\$3))):(\$3>=72 ? 0xffffff : 0x132c2c) \
     with labels tc rgb variable
GP
        gnuplot plot.gp
        pdftoppm -singlefile -r 150 -png figure.pdf figure
        mv -- figure.pdf "$out/$tier-accuracy.pdf"
        mv -- figure.png "$out/$tier-accuracy.png"
    )
done
