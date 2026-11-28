#!/usr/bin/env bash
# Plot measured hidden accuracy and compare each pair on identical held-out words.
set -euo pipefail
cd -- "${1:?usage: bash plot_sum8_half.sh RUN_DIRECTORY}"
if [[ ${2:-} == --live ]]; then
    draw=${3:-0}
    [[ $draw =~ ^[0-7]$ ]]
    seed=$(jq -er --argjson i "$draw" '.seeds[$i][0]' protocol.json)
    budget=$(jq -er '.updates' protocol.json)
    control=$(jq -er --argjson i "$draw" '.records[$i].common_hidden.correct' control-common.json)
    cd "draw-$draw"
    # Only parse lines whose terminating newline is already on disk.
    jq -Rrs 'split("\n")[:-1][] | fromjson | select(.observed and .hidden) |
      [.step,.observed.correct,.hidden.correct]|@tsv' training.jsonl > progress.tsv
    read -r last train hidden < <(tail -1 progress.tsv)
    gnuplot -e "budget=$budget; last=$last; seed=$seed; control=$control; obs=$train; held=$hidden" - <<'PLOT'
set terminal pngcairo size 1320,575 font 'DejaVu Sans,14' background rgb 'white'
set output sprintf('accuracy-step-%d.png',last)
set lmargin at screen 0.085
set rmargin at screen 0.955
set bmargin at screen 0.16
set tmargin at screen 0.80
set label 1 'Eight-bit parity   |   50% of inputs observed' at screen 0.085,0.945 font ',22'
set label 2 sprintf('AdamW + WSD, seed %.0f   |   Step %d:  %d/128 train,  %d/128 hidden',seed,last,obs,held) at screen 0.085,0.875
set border 3
set tics out nomirror
set grid ytics lc rgb '#e1e5e5'
set xlabel 'Optimizer updates (log scale)'
set ylabel 'Strict accuracy (%)'
set logscale x
set xrange [1:budget*1.12]
set xtics ('1' 1, '8' 8, '32' 32, '128' 128, '512' 512, '2,048' 2048, '8,192' 8192, '20,480' 20480)
set yrange [0:104]
set ytics 20
set arrow from budget,0 to budget,102 nohead dt 2 lc rgb '#777777'
set label 3 'WSD budget' at budget,9 right offset -0.3,0 tc rgb '#666666' font ',11'
set key left bottom horizontal font ',12' samplen 1.7
plot 'progress.tsv' u 1:($2*100./128) w lp pt 5 ps 0.55 lw 2 lc rgb '#b45f29' title 'Train', \
     '' u 1:($3*100./128) w lp pt 7 ps 0.7 lw 2.5 lc rgb '#12767d' title 'Hidden', \
     (control*100./128) w l dt 3 lw 1.5 lc rgb '#666666' title sprintf('31-train control at 20,480 (%d/128 hidden)',control)
PLOT
    echo "draw-$draw/accuracy-step-$last.png"
    exit 0
fi
jq -ner --slurpfile half summary.json --slurpfile control control-common.json '
  $half[0] as $h | $control[0] as $c |
  if $h.replayed != true or $h.protocol.protocol != "argand-sum8-adamw-half-v1" or
     $c.control_protocol.protocol != "argand-sum8-adamw-wsd-v1" or
     $c.half_train_ids != $h.protocol.train_ids or $h.protocol.seeds != $c.control_protocol.seeds or
     $h.protocol.adamw != $c.control_protocol.adamw or
     ($h.records|length) != 8 or ($c.records|length) != 8
  then error("expected a replayed pair of matched cohorts") else . end |
  range(8) as $i | $h.records[$i] as $r | $c.records[$i] as $s |
  if $s.seeds != $r.seeds or $r.seeds != $h.protocol.seeds[$i] or
     $s.common_hidden.total != 128 or $r.hidden.total != 128
  then error("unmatched checkpoints or hidden sets") else
    [$i+1,$s.common_hidden.correct,$r.hidden.correct]|@tsv end' > half-pairs.tsv
jq -r '.records[] |
  (([.initial+{step:0}] + .milestones + [{step:.optimizer.step,hidden:.hidden}]) |
    unique_by(.step)[] | [.step,.hidden.correct]|@tsv), "", ""' summary.json > half-trajectories.tsv
last=$(jq -er '[.records[].optimizer.step]|max' summary.json)
gnuplot -e "last=$last" - <<'PLOT'
set terminal pngcairo size 1460,570 font 'DejaVu Sans,13' background rgb 'white'
set output 'half-accuracy.png'
set multiplot layout 1,2 margins 0.075,0.98,0.15,0.81 spacing 0.10,0.05
set label 1 'Eight-bit parity   |   Stock Argand + AdamW' at screen 0.075,0.945 font ',20'
set label 2 'Eight paired initializations. Both splits are scored on the same 128 held-out words.' at screen 0.075,0.885 font ',13'
set border 3
set tics out nomirror
set title '50% train, one curve per initialization'
set xlabel 'Optimizer updates (log scale after initialization)'
set ylabel 'Strictly correct hidden words (%)'
set logscale x
set xrange [1:last+1]
set xtics ('0' 1, '8' 9, '64' 65, '512' 513, '4k' 4097, '16k' 16385)
set yrange [0:104]
set ytics 20
set grid ytics lc rgb '#e3e6e7'
set key off
plot for [i=0:7] 'half-trajectories.tsv' index i using ($1+1):($2*100./128) with linespoints pt 7 ps 0.25 lw 1.2 lc rgb '#187d83'
unset label
unset logscale x
unset ylabel
set title 'Accuracy at the stopping checkpoint'
set xlabel 'Strictly correct hidden words (%)'
set xrange [0:103]
set xtics 20
set yrange [8.8:0.2]
set ytics 1,1,8
set ylabel 'Paired run'
set grid xtics lc rgb '#e3e6e7'
set key bottom right horizontal font ',11' samplen 1.2
plot 'half-pairs.tsv' using ($2*100./128):1:(($3-$2)*100./128):(0) with vectors nohead lw 1.2 lc rgb '#a6afb1' notitle, \
     '' using ($2*100./128):1 with points pt 5 ps 1.25 lc rgb '#ad642c' title '31 train', \
     '' using ($3*100./128):1 with points pt 7 ps 1.25 lc rgb '#187d83' title '128 train'
unset multiplot
PLOT
