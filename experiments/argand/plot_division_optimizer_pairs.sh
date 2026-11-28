#!/usr/bin/env bash
# Compare registered first hits at identical seeds. An absent arm stays blank.
set -euo pipefail
if (( $# < 1 || $# > 2 )); then echo 'usage: bash plot_division_optimizer_pairs.sh PRIMARY [HIGHER_RATE_CONTROL]' >&2; exit 2; fi
dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
out=$(realpath -e -- "$1")
tmp=$(mktemp -d "$out/.pair-figure.XXXXXXXX")
trap 'rm -f -- "$tmp"/{report.json,points.tsv,plot.gp,figure.pdf,figure.png}; rmdir -- "$tmp"' EXIT
bash "$dir/report_division_optimizer_pairs.sh" "$@" > "$tmp/report.json"
jq -r '.pairs[] | . as $p | ["control","candidate","higher_rate_control"] | to_entries[] |
    . as $a | $p[$a.value] as $v | select($v!=null) |
    [$v.step,$p.index,$a.key,(if $v.first_attempt!="complete" then -1 elif $v.success then 1 else 0 end)] | @tsv' \
    "$tmp/report.json" > "$tmp/points.tsv"
[[ -s "$tmp/points.tsv" ]] || { echo 'no completed arm yet' >&2; exit 1; }
ticks=$(jq -r '[.pairs[]|select(.control!=null or .candidate!=null or .higher_rate_control!=null)|
    "\""+(.seed|tostring)+"\" "+(.index|tostring)]|join(", ")' "$tmp/report.json")
read -r last max < <(awk 'BEGIN{x=0;y=0}{if($1>x)x=$1;if($2>y)y=$2}END{print y,x}' "$tmp/points.tsv")
arms=$(awk '{seen[$3]=1} END{for(i=0;i<3;i++) if(seen[i]) printf "%d ",i}' "$tmp/points.tsv")
control=$(jq -r '"AdamW "+(.primary_rate|tostring)' "$tmp/report.json")
counts=$(jq -r --arg ctrl "$control" --argjson n "$#" '"Candidate "+(.candidate.hits|tostring)+"/"+(.candidate.evaluated|tostring)+
    "     "+$ctrl+" "+(.control.hits|tostring)+"/"+(.control.evaluated|tostring)+
    (if $n==2 then "     AdamW 0.003 "+(.higher_rate_control.hits|tostring)+"/"+(.higher_rate_control.evaluated|tostring) else "" end)' "$tmp/report.json")
name=$(jq -r '(if .candidate_optimizer|startswith("equal-moment") then "Equal-decay Adam"
    elif .candidate_optimizer=="uncorrected-adam-wsd" then "Uncorrected Adam"
    elif .candidate_optimizer=="normalized-adamw-wsd" then "Normalized AdamW"
    elif .candidate_optimizer|contains("softsign") then "Softsign Lion"
    elif .candidate_optimizer|contains("rmsprop") then "RMSPropW" else "Complex Muon" end)
    +" "+(.candidate_peak|tostring)' "$tmp/report.json")
(
cd -- "$tmp"
cat > plot.gp <<GP
set terminal pdfcairo size 8.4in,4.6in color font 'DejaVu Sans,11'
set output 'figure.pdf'
set title 'First strict success on division modulo 97' font ',16'
set xlabel 'Minibatch updates (thousands)'
set ylabel 'Registered model seed'
set xrange [0:ceil(($max+1)/1000.)]
set yrange [$last+0.85:-0.85]
set ytics ($ticks)
set border 3 lc rgb '#718183'
set grid xtics lc rgb '#dce4e6'
set tics out nomirror
set key outside below center horizontal
set label 1 '$counts' at graph 0.02,0.97 font ',9'
colors='#3774a1 #16715d #c25628'
array names[3] = ['$control','$name','AdamW 0.003']
do for [i=1:3] {set style line i lc rgb word(colors,i)}
set datafile missing 'NaN'
plot for [arm in '$arms'] 'points.tsv' u (\$3==int(arm) ? \$1/1000. : 1/0):(\$2+(int(arm)-1)*0.19):(\$4==1?7:(\$4==0?2:4)) \
    w p ls (int(arm)+1) pt variable ps 1.15 title names[int(arm)+1]
GP
gnuplot plot.gp
pdftoppm -singlefile -r 150 -png figure.pdf figure
mv -- figure.pdf "$out/paired-first-hits.pdf"
mv -- figure.png "$out/paired-first-hits.png"
)
