#!/usr/bin/env bash
# Display completed first-passage trajectories and the latest active seed.
set -euo pipefail
if (( $# < 2 || $# > 3 )); then echo 'usage: bash plot_division_to_perfect.sh COHORT OUTPUT_DIRECTORY [PROSPECTIVE_EXTENSION]' >&2; exit 2; fi
dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
cohort=$(realpath -e -- "$1")
mkdir -p -- "$2"
out=$(cd -- "$2" && pwd)
extension=''
first=0
end=12
title='Argand on division modulo 97'
if (( $# == 3 )); then
    extension=$(realpath -e -- "$3")
    cmp -- "$dir/division-first-hit-registration.json" "$extension/registration.json"
    jq -ne --slurpfile a "$extension/registration.json" --slurpfile p "$cohort/protocol.json" '
      $a[0].model_seeds==($p[0].seeds|map(.[0])) and $a[0].three_quarters_test.wsd_budget==$p[0].updates' >/dev/null
    read -r first end < <(jq -r '[.first_prospective_index, ((.model_seeds|length)+(.three_quarters_test.additional_model_seeds|length))]|@tsv' "$extension/registration.json")
    title='Argand on division modulo 97  |  prospective seeds and extension'
fi
tmp=$(mktemp -d "$out/.division-first100.XXXXXX")
trap 'rm -f -- "$tmp"/*.tsv "$tmp"/figure.{pdf,png} "$tmp/report.json"; rmdir -- "$tmp"' EXIT
cd -- "$tmp"
touch successes.tsv running.tsv missed.tsv
cap=$(jq -er 'select(.stop=="first-strict-100-percent" and .n==12) | .updates' "$cohort/protocol.json")
labels=''
series=''
done_count=0
started=0
last=0
max=0
for ((i=first;i<end;i++)); do
    draw=$((i-first+1))
    if (( i<12 )); then
        seed=$(jq -er --argjson i "$i" '.seeds[$i][0]' "$cohort/protocol.json")
        cell="$cohort/grok-division-p97/seed$seed"
        record="$cohort/draw-$i.json"
    else
        j=$((i-12))
        seed=$(jq -er --argjson j "$j" '.three_quarters_test.additional_model_seeds[$j]' "$extension/registration.json")
        cell="$extension/wsd-$cap/grok-division-p97/seed$seed"
        record="$extension/draw-$j.json"
    fi
    if [[ -s "$record" ]] && jq -e '.status=="interrupted"' "$record" >/dev/null; then
        labels+="${labels:+,}'$draw    $seed (interrupted)' $draw"
        continue
    fi
    labels+="${labels:+,}'$draw    $seed' $draw"
    stop=$cap
    if [[ -s "$record" ]]; then
        if [[ $(jq -er --argjson seed "$seed" 'select(.seed==$seed and .total==4656)|.checkpoint' "$record") == first-strict ]]; then
            cell=$(jq -er '.cell' "$record")
        fi
        stop=$(jq -r '.step' "$record")
        if jq -e '.success' "$record" >/dev/null; then file=successes.tsv; else file=missed.tsv; fi
        echo "$draw $stop" >> "$file"
        ((done_count+=1))
    fi
    [[ -s "$cell/observations.jsonl" ]] || continue
    sed -z 's/[^\n]*$//' "$cell/observations.jsonl" | jq -r --argjson seed "$seed" --argjson cap "$cap" --argjson stop "$stop" '
      if .identity.seed==$seed and .identity.case=="grok-division-p97" and .identity.optimizer=="adamw-wsd" and
         .identity.budget==$cap and .total==4656 then select(.step<=$stop)|[.step,100*.correct/.total]|@tsv
      else error("unexpected trajectory") end' > "$draw.tsv"
    [[ -s "$draw.tsv" ]] || continue
    read -r last _ < <(tail -1 "$draw.tsv")
    if ((last > max)); then max=$last; fi
    if [[ ! -s "$record" ]]; then echo "$draw $last" >> running.tsv; fi
    series+=" $draw"
    started=$draw
done
((started > 0))
hits=$(wc -l < successes.tsv)
for f in successes.tsv running.tsv missed.tsv; do [[ -s $f ]] || echo '1 NaN' > "$f"; done
gnuplot -e "series='$series'; labels=\"$labels\"; cap=$cap; hits=$hits; done_count=$done_count; xmax=$max; n=$((end-first)); heading='$title'" - <<'GP'
set terminal pdfcairo color size 11.5in,5.1in font 'DejaVu Sans,10'
set output 'figure.pdf'
set multiplot
set label 1 heading at screen 0.072,0.937 font ',17' tc rgb '#203340'
set label 2 sprintf('First evaluation with all 4,656 held-out answers strictly correct   |   AdamW, decay 0.1   |   WSD cap %d',cap) at screen 0.072,0.872 tc rgb '#425b6b'
set border 3 lc rgb '#86959f'
set tics nomirror out scale 0.4
set grid ytics lc rgb '#dee5e9'
set lmargin at screen 0.072
set rmargin at screen 0.51
set bmargin at screen 0.16
set tmargin at screen 0.73
set title 'Hidden accuracy through the first perfect evaluation'
set xlabel 'Minibatch updates (thousands)'
set ylabel 'Correct (%)'
set yrange [-3:105]
set ytics 25
set xrange [0:(ceil((xmax+1)/5000.)*5)]
colors='#156a71 #b25727 #604ba3 #a72953 #3e7439 #335da6 #946b17 #056993 #764969 #76762b #445e63 #9b302d'
unset key
plot for [j in series] sprintf('%s.tsv',j) u ($1/1000.):2 w l lw 2 lc rgb word(colors,int(j))
unset label
set lmargin at screen 0.70
set rmargin at screen 0.965
set title sprintf('Verified first hits: %d of %d evaluated',hits,done_count)
unset ylabel
set yrange [n+0.65:0.35]
eval 'set ytics ('.labels.') font ",8"'
set label 3 'Draw      Seed' at graph -0.05,1.04 right font ',8'
set label 4 'Diamond = first 100%     Circle = in progress     Cross = budget exhausted' at screen 0.53,0.045 font ',8'
plot 'successes.tsv' u ($2/1000.):1:(rgbcolor(word(colors,int($1)))) w p pt 13 ps 1.45 lc rgb variable, \
     'running.tsv' u ($2/1000.):1:(rgbcolor(word(colors,int($1)))) w p pt 6 ps 1.3 lw 2 lc rgb variable, \
     'missed.tsv' u ($2/1000.):1:(rgbcolor(word(colors,int($1)))) w p pt 2 ps 1.4 lw 2 lc rgb variable
unset multiplot
GP
pdftoppm -singlefile -r 140 -png figure.pdf figure
base="division-first100-$cap"
if [[ -n "$extension" ]]; then
    base="division-first100-prospective-$cap"
    if [[ -s "$extension/summary.json" ]]; then
        cp -- "$extension/summary.json" report.json
        mv -- report.json "$out/$base-inference.json"
    fi
elif [[ -s "$cohort/summary.json" ]]; then
    bash "$dir/report_division_first_hits.sh" "$cohort" > report.json
    mv -- report.json "$out/$base-inference.json"
fi
tag="$base-draw-$started-step-$last-done-$done_count"
for ext in pdf png; do
    if [[ ${ARGAND_PLOT_SNAPSHOT:-1} == 1 && ! -e "$out/$tag.$ext" ]]; then
        cp -- "figure.$ext" "$out/$tag.$ext"
    fi
    mv -- "figure.$ext" "$out/$base.$ext"
done
if [[ ${ARGAND_PLOT_SNAPSHOT:-1} == 1 ]]; then echo "$tag"; else echo "$base"; fi
