#!/usr/bin/env bash
# Render only recorded measurements, including a still running 100k trajectory.
set -euo pipefail
if (( $# != 4 )); then
    echo 'usage: bash plot_division_progress.sh FIRST_20K_CELL SECOND_20K_CELL LONG_CELL OUTPUT_DIRECTORY' >&2
    exit 2
fi
mkdir -p -- "$4"
out=$(cd -- "$4" && pwd)
tmp=$(mktemp -d "$out/.division-progress.XXXXXX")
trap 'rm -f -- "$tmp"/{1,2,3}.{jsonl,tsv} "$tmp"/snapshot.{pdf,png}; rmdir -- "$tmp"' EXIT
for i in 1 2 3; do
    sed '${/}$/!d;}' "${!i}/observations.jsonl" > "$tmp/$i.jsonl"
    jq -er '[.step/1000,100*.correct/.total,100*.trained/.training_rows]|@tsv' \
        "$tmp/$i.jsonl" > "$tmp/$i.tsv"
done
cd -- "$tmp"
read -r a b last correct train total < <(
    jq -enr --slurpfile a 1.jsonl --slurpfile b 2.jsonl --slurpfile c 3.jsonl '
      def base: .identity|{model,split_sha256,width,heads,layers,repeat,vocab,batch,optimizer};
      if ($a[-1]|base)==($b[-1]|base) and ($b[-1]|base)==($c[-1]|base)
        and $a[-1].identity.case=="grok-division-p97" and $a[-1].identity.optimizer=="adamw-wsd"
        and $a[-1].step==20000 and $b[-1].step==20000 and $c[-1].identity.adamw.budget==100000
        and $b[-1].identity.seed==$c[-1].identity.seed and $b[-1].identity.order_seed==$c[-1].identity.order_seed
      then [$a[-1].identity.seed,$b[-1].identity.seed,$c[-1].step,$c[-1].correct,$c[-1].trained,$c[-1].total]|@tsv
      else error("incompatible trajectories") end'
)
gnuplot -e "a=$a; b=$b; last=$last; correct=$correct; train=$train; total=$total" - <<'GP'
set terminal pdfcairo color size 11.7in,4.9in font "DejaVu Sans,10"
set output 'snapshot.pdf'
set multiplot layout 1,2 margins 0.075,0.978,0.16,0.77 spacing 0.10,0.0
set label 1 'Division modulo 97' at screen 0.075,0.944 font ",19" tc rgb '#172c38'
set label 2 'Argand  |  AdamW, weight decay 0.1  |  4,656 held-out equations' at screen 0.075,0.871 font ",11" tc rgb '#3e5261'
set grid ytics lc rgb '#dde2e5'
set border 3 lc rgb '#85949f'
set tics nomirror out scale 0.5
set xlabel 'Minibatch updates (thousands)'
set ylabel 'Accuracy (%)'
set yrange [-2:105]
set ytics 25
set xrange [0:(last<20000?20:last/1000.)*1.02]
set key top left reverse Left font ",9" samplen 1.7
set style line 1 lc rgb '#006d73' lw 2.1
set style line 2 lc rgb '#ac471c' lw 2.1
set style line 3 lc rgb '#6446a4' lw 2.4 pt 7 ps 0.22
set title 'Held-out accuracy across runs' offset 0,0.2
plot '1.tsv' u 1:2 w l ls 1 title sprintf('seed %.0f, 20k WSD',a), \
     '2.tsv' u 1:2 w l ls 2 title sprintf('seed %.0f, 20k WSD',b), \
     '3.tsv' u 1:2 w lp ls 3 title sprintf('seed %.0f, 100k WSD',b)
unset label
unset ylabel
set title sprintf('100k WSD: latest recorded update %.0f',last) offset 0,0.2
set xrange [0:(last>256?last/1000.:0.256)*1.04]
set key center right opaque
plot '3.tsv' u 1:3 w l lw 2 dt 2 lc rgb '#617a8c' title sprintf('Train: %.2f%%',100.*train/total), \
     '3.tsv' u 1:2 w lp ls 3 title sprintf('Held out: %.2f%% (%d/%d)',100.*correct/total,correct,total)
unset multiplot
GP
pdftoppm -singlefile -r 135 -png snapshot.pdf snapshot
if [[ ${ARGAND_PLOT_SNAPSHOT:-0} == 1 ]]; then
    cp -- snapshot.png "$out/division-100k-step-$last.png"
    cp -- snapshot.pdf "$out/division-100k-step-$last.pdf"
fi
mv -- snapshot.pdf "$out/division-100k-progress.pdf"
mv -- snapshot.png "$out/division-100k-progress.png"
echo "$last updates: $correct/$total held out, $train/$total train"
