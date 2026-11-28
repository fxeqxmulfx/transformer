#!/usr/bin/env bash
# Compare the two optimizers at the same budget and from the same native seeds.
set -euo pipefail
if (( $# != 3 )); then
    echo 'usage: bash plot_sum8_pair.sh PAGE_COHORT ADAMW_COHORT OUTPUT_DIRECTORY' >&2
    exit 2
fi
dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
page=$(realpath -e -- "$1/summary.json")
adamw=$(realpath -e -- "$2/summary.json")
mkdir -p -- "$3"
out=$(cd -- "$3" && pwd)
temp=$(mktemp -d "$out/.sum8-pair.XXXXXXXX")
trap 'rm -f -- "$temp"/*.tsv "$temp"/pair.{pdf,png}; rmdir -- "$temp"' EXIT
cd -- "$temp"
jq -ern --slurpfile a "$page" --slurpfile b "$adamw" '
  $a[0] as $p | $b[0] as $w |
  def design: del(.protocol,.seed_order,.adamw);
  if $p.protocol.protocol!="argand-sum8-adaptive-page-v1" or $w.protocol.protocol!="argand-sum8-adamw-wsd-v1" or
     ($p.protocol|design)!=($w.protocol|design) or $p.protocol.open_words!=31 or $p.protocol.hidden_words!=225 or
     ($p.protocol.seeds|length)!=8 or ($p.protocol.seeds|map(.[0])|unique|length)!=8
  then error("different sum8 designs") else . end |
  range(8) as $i | $p.records[$i] as $r | $w.records[$i] as $s |
  if $r.seeds!=$p.protocol.seeds[$i] or $s.seeds!=$r.seeds or $r.initial!=$s.initial or
     $r.updates!=$p.protocol.updates or $s.updates!=$p.protocol.updates or
     $r.hidden.total!=225 or $s.hidden.total!=225 or $r.observed.total!=31 or $s.observed.total!=31
  then error("unmatched initial conditions or evaluation domains") else
    [$i+1,$r.seeds[0],$r.hidden.correct,$s.hidden.correct]|@tsv end' > pairs.tsv
jq -er -f "$dir/sum8_accuracy.jq" "$page" > page.tsv
jq -er -f "$dir/sum8_accuracy.jq" "$adamw" > adamw.tsv
budget=$(jq -er '.protocol.updates' "$page")
ticks=$(awk '{printf "%s\047%s\047 %d", (NR>1 ? "," : ""), $2, $1}' pairs.tsv)
gnuplot -e "budget=$budget; ticks=\"$ticks\"" - <<'PLOT'
set terminal pdfcairo color size 10.8in,4.6in font 'DejaVu Sans,10'
set output 'pair.pdf'
set multiplot
set label 1 'Eight-bit modular sum on the 31-word staircase' at screen 0.075,0.922 font ',17' tc rgb '#243746'
set label 2 sprintf('Eight matched initializations   |   %d updates   |   225 held-out words',budget) at screen 0.075,0.85 tc rgb '#455b65'
set border 3 lc rgb '#7c8990'
set tics nomirror out scale 0.4
set grid ytics lc rgb '#e0e5e8'
set lmargin at screen 0.078
set rmargin at screen 0.52
set bmargin at screen 0.20
set tmargin at screen 0.72
set title 'Mean accuracy over the eight seeds'
set xlabel 'Optimizer updates (logarithmic spacing)'
set ylabel 'Correct words (%)'
set yrange [0:105]
set ytics 25
set xrange [1:budget+1]
set logscale x
set xtics ('0' 1)
do for [t in '8 64 512 4096'] { if (int(t)<budget/1.5) {set xtics add (t int(t)+1)} }
set xtics add (sprintf('%d',budget) budget+1)
set style line 1 lc rgb '#176a73' lw 2
set style line 2 lc rgb '#a84d24' lw 2
set key left bottom font ',8' samplen 2 maxcols 2
plot 'page.tsv' u (1+$1):2 w l ls 1 dt 3 title 'PAGE observed', \
     'page.tsv' u (1+$1):5 w l ls 1 title 'PAGE hidden', \
     'adamw.tsv' u (1+$1):2 w l ls 2 dt 3 title 'AdamW observed', \
     'adamw.tsv' u (1+$1):5 w l ls 2 title 'AdamW hidden'
unset label
unset logscale x
set autoscale xy
set xtics autofreq
stats 'pairs.tsv' using 3:4 nooutput
lo = STATS_min_x < STATS_min_y ? STATS_min_x : STATS_min_y
hi = STATS_max_x > STATS_max_y ? STATS_max_x : STATS_max_y
set xrange [lo-2:hi+2]
set lmargin at screen 0.71
set rmargin at screen 0.96
set title 'Hidden accuracy at the endpoint'
set xlabel 'Correct words out of 225'
unset ylabel
set yrange [8.6:0.4]
eval 'set ytics ('.ticks.') font ",8"'
set label 3 'Seed' at graph -0.03,1.025 right font ',8'
set key at screen 0.74,0.082 left center horizontal font ',8' samplen 1.5 width 2
plot 'pairs.tsv' u 3:1:($4-$3):(0) w vectors nohead lw 1 lc rgb '#98a4ab' notitle, \
     'pairs.tsv' u 3:1 w p ls 1 pt 6 ps 1.35 title 'PAGE', \
     'pairs.tsv' u 4:1 w p ls 2 pt 13 ps 0.85 title 'AdamW + WSD'
unset multiplot
PLOT
pdftoppm -r 145 -png -singlefile pair.pdf pair
for ext in pdf png; do mv -- "pair.$ext" "$out/sum8-page-adamw.$ext"; done
