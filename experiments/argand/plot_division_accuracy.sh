#!/usr/bin/env bash
# Plot three matched modular-division runs, retaining their actual observation grids.
set -euo pipefail
if (( $# != 4 )); then
    echo 'usage: bash plot_division_accuracy.sh PAGE_CELL SHORT_ADAMW_CELL LONG_ADAMW_CELL OUTPUT_DIR' >&2
    exit 2
fi
mkdir -p -- "$4"
out=$(cd -- "$4" && pwd)
series=(page short long)
files=()
for i in 0 1 2; do
    j=$((i+1))
    files+=("${!j}/observations.jsonl")
done
jq -ens --slurpfile p "${files[0]}" --slurpfile s "${files[1]}" --slurpfile l "${files[2]}" '
  def base: {case, model, seed, order_seed, split_sha256, width, heads, layers, repeat, batch, vocab, kernite};
  def init: {correct, trained, loss, total, training_rows};
  (($p[0].identity | base) == ($s[0].identity | base) and
   ($p[0].identity | base) == ($l[0].identity | base) and
   ($p[0] | init) == ($s[0] | init) and ($p[0] | init) == ($l[0] | init) and
   $p[0].identity.case == "grok-division-p97" and
   $p[0].identity.optimizer == "batch-reference-adaptive-page-v1" and
   $s[0].identity.optimizer == "adamw-wsd" and $l[0].identity.optimizer == "adamw-wsd" and
   $s[0].identity.adamw.budget < $l[0].identity.adamw.budget)
' >/dev/null
for i in 0 1 2; do
    jq -r '[.step,100*.trained/.training_rows,100*.correct/.total,.optimizer.rate//0] | @tsv' \
      "${files[i]}" > "$out/${series[i]}.tsv"
done
read -r short short_cool < <(jq -sr 'first.identity.adamw|[.budget,.cooldown_start]|@tsv' "${files[1]}")
read -r long long_cool < <(jq -sr 'first.identity.adamw|[.budget,.cooldown_start]|@tsv' "${files[2]}")
cd -- "$out"
gnuplot -e "short=$short; long=$long; short_cool=$short_cool; long_cool=$long_cool" - <<'GP'
set terminal pdfcairo size 10.6in,4.25in color font "DejaVu Sans,10"
set output 'division-wsd.pdf'
set multiplot layout 1,2 margins 0.07,0.982,0.18,0.82 spacing 0.08,0.0 title "Division modulo 97  |  Argand, seed 0, 4,656 observed and 4,656 held-out equations"
set border 3 lc rgb "#5b6065"
set tics nomirror
set grid ytics lc rgb "#dee1e3"
set xlabel 'Minibatch updates'
set xrange [0:long*1.02]
set yrange [0:104]
set ytics 20
set ylabel 'Accuracy (%)'
set key bottom right font ",8" opaque
set title 'Held-out exact answers'
set arrow 1 from short_cool,0 to short_cool,102 nohead dt 3 lc rgb "#735791" back
set arrow 2 from long_cool,0 to long_cool,102 nohead dt 3 lc rgb "#145a88" back
set style line 1 lc rgb "#af5328" lw 2 pt 7 ps 0.25
set style line 2 lc rgb "#735791" lw 2 pt 7 ps 0.25
set style line 3 lc rgb "#145a88" lw 2 pt 7 ps 0.25
plot 'page.tsv' u 1:3 w lp ls 1 title 'PAGE', \
     'short.tsv' u 1:3 w lp ls 2 title sprintf('AdamW, %d-step WSD',short), \
     'long.tsv' u 1:3 w lp ls 3 title sprintf('AdamW, %d-step WSD',long)
unset ylabel
set title 'Training exact answers'
set key center right
plot 'page.tsv' u 1:2 w lp ls 1 title 'PAGE', \
     'short.tsv' u 1:2 w lp ls 2 title sprintf('AdamW, %d-step WSD',short), \
     'long.tsv' u 1:2 w lp ls 3 title sprintf('AdamW, %d-step WSD',long)
unset multiplot
GP
pdftoppm -singlefile -r 150 -png division-wsd.pdf division-wsd
