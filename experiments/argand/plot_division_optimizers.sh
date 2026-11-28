#!/usr/bin/env bash
# Overlay paired first-passage trajectories. Never extend a control past its first strict hit.
set -euo pipefail
if (( $# < 3 )); then echo 'usage: bash plot_division_optimizers.sh OUTPUT_DIRECTORY CELL CELL [...]' >&2; exit 2; fi
out=$(realpath -m -- "$1")
shift
mkdir -p -- "$out"
tmp=$(mktemp -d "$out/.optimizers.XXXXXXXX")
trap 'rm -f -- "$tmp"/*.tsv "$tmp"/figure.pdf "$tmp"/figure.png "$tmp"/plot.gp; rmdir -- "$tmp"' EXIT
ref=''
curves=()
max=0
for cell in "$@"; do
    cell=$(realpath -e -- "$cell")
    id=$(head -n 1 "$cell/observations.jsonl" | jq -ce '.identity | select(.case=="grok-division-p97")')
    paired=$(jq -cS '(.adamw // .softsign_lion // .muon) as $o | {case,seed,order_seed,budget,wsd_budget:$o.budget,warmup:$o.warmup,cooldown:$o.cooldown_start,observe,split_sha256,suite_sha256,model,repeat,width,heads,layers,vocab,batch,kernite,gradient_reduction}' <<< "$id")
    if [[ -z "$ref" ]]; then ref=$paired; elif [[ $paired != "$ref" ]]; then echo "unpaired trajectory: $cell" >&2; exit 1; fi
    peak=$(jq -er '(.adamw // .softsign_lion // .muon).peak' <<< "$id")
    decay=$(jq -er '(.adamw // .softsign_lion // .muon).weight_decay' <<< "$id")
    case $(jq -r .optimizer <<< "$id") in
        adamw-wsd|adamw-scaled-wsd) name=AdamW ;;
        softsign-lion-wsd) name='Softsign Lion' ;;
        softsign-lion-wsd-checkpoint-proposals) name='Softsign Lion + verifier' ;;
        complex-muon-wsd) name='Complex Muon' ;;
        *) echo "unknown optimizer in $cell" >&2; exit 1 ;;
    esac
    idx=${#curves[@]}
    stop=$(jq -r .budget <<< "$id")
    status=running
    hit=''
    cert=observed-certificate.json
    replay=replay.json
    if [[ -s "$cell/first-strict.json" ]]; then
        checkpoint=first-strict.json
        cert=first-strict-certificate.json
        replay=first-strict-replay.json
    else
        checkpoint=result.json
    fi
    if [[ -s "$cell/$checkpoint" ]]; then
        stop=$(jq -er .step "$cell/$checkpoint")
        status='budget exhausted'
        if jq -e '.strict.all_strict' "$cell/$checkpoint" > /dev/null; then
            status='awaiting replay'
            if [[ -s "$cell/$cert" && -s "$cell/$replay" ]]; then
                jq -es '.[0] as $c | [.[1],.[2]] | all(.step==$c.step and .weights_sha256==$c.weights_sha256) and
                    .[0].strict.all_strict and .[1].accept and .[1].hidden.strict_correct==4656 and .[1].hidden.minimum_gap>0' \
                    "$cell/$checkpoint" "$cell/$replay" "$cell/$cert" > /dev/null
                status="100% at $stop"
                hit=$stop
            fi
        fi
    fi
    sed -z 's/[^\n]*$//' "$cell/observations.jsonl" | jq -er --argjson id "$id" --argjson stop "$stop" '
        if .identity!=$id or .total!=4656 then error("inconsistent trajectory")
        else select(.step<=$stop) | [.step,100*.correct/.total] | @tsv end' > "$tmp/$idx.tsv"
    read -r last _ < <(tail -n 1 "$tmp/$idx.tsv")
    if (( last > max )); then max=$last; fi
    curves+=("'$idx.tsv' u (\$1/1000.):2 w l ls $((idx+1)) lw 2.5 title '$name   η=$peak   λ=$decay   ($status)'")
    if [[ -n $hit ]]; then
        echo "$hit 100" > "$tmp/hit-$idx.tsv"
    fi
done
seed=$(jq -r .seed <<< "$ref")
cap=$(jq -r .wsd_budget <<< "$ref")
(
cd -- "$tmp"
cat > plot.gp <<GP
set terminal pdfcairo size 9in,4.9in color font 'DejaVu Sans,10'
set output 'figure.pdf'
set title 'Argand on division modulo 97  |  seed $seed' font ',16'
set xlabel 'Minibatch updates (thousands)'
set ylabel 'Held-out equations correct (%)'
set xrange [0:ceil(($max+1)/2000.)*2]
set yrange [-3:108]
set ytics 25
set border 3 lc rgb '#6e7d80'
set tics out nomirror scale 0.4
set grid ytics lc rgb '#dce4e6'
set key outside below left Left reverse horizontal maxcols 2
set label 1 '4,656 held-out equations    |    WSD budget $cap' at graph 0.02,0.98 font ',9'
colors='#3774a1 #c25628 #16715d #7955a4 #956400 #536f47 #a03e77'
do for [i=1:words(colors)] {set style line i lc rgb word(colors,i) dt 1}
GP
for i in "${!curves[@]}"; do
    if [[ -s "hit-$i.tsv" ]]; then curves+=("'hit-$i.tsv' u (\$1/1000.):2 w p ls $((i+1)) pt 7 ps 0.7 notitle"); fi
done
(IFS=,; echo "plot ${curves[*]}") >> plot.gp
gnuplot plot.gp
pdftoppm -singlefile -r 145 -png figure.pdf figure
mv -- figure.pdf "$out/accuracy.pdf"
mv -- figure.png "$out/accuracy.png"
)
