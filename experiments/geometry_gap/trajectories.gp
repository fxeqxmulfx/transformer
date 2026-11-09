# Native gnuplot export. Bands are ranges over eight seeds at measured probes.
# Usage: gnuplot -c trajectories.gp DATA.tsv OUTPUT_PREFIX
data_path = ARG1
output_prefix = ARG2
array names[3] = ["Peak 0.001", "Peak 0.01", "Early reduction"]
array colors[3] = ["#3568b0", "#d47725", "#21845b"]
do for [format=0:1] {
  if (format==0) {
    set terminal svg size 1000,800 font "sans,14"
    set output output_prefix . ".svg"
  } else {
    set terminal pngcairo size 1500,1200 font "sans,17"
    set output output_prefix . ".png"
  }
  set multiplot layout 2,1 title "Earlier LR reduction improves sampled persistence"
  set xrange [0:10000]
  set xtics 2000
  set grid back
  set border 3
  set tics nomirror
  set key outside top center horizontal
  set ylabel "Validation accuracy (%)"
  set yrange [90:100]
  set title "Accuracy: 90-100% range"
  unset xlabel
  plot for [i=0:2] data_path using ($1==i?$2:1/0):($4*100):($5*100) with filledcurves lc rgb colors[i+1] fs transparent solid 0.12 notitle, \
       for [i=0:2] data_path using ($1==i?$2:1/0):($3*100) with linespoints lw 2 pt 7 ps 0.35 lc rgb colors[i+1] title names[i+1]
  unset key
  set title "Original all-class rank-32 certificate; bands show seed ranges"
  set ylabel "Certificate coverage (%)"
  set xlabel "Completed training updates"
  set yrange [0:100]
  plot for [i=0:2] data_path using ($1==i?$2:1/0):($7*100):($8*100) with filledcurves lc rgb colors[i+1] fs transparent solid 0.12 notitle, \
       for [i=0:2] data_path using ($1==i?$2:1/0):($6*100) with linespoints lw 2 pt 7 ps 0.35 lc rgb colors[i+1] notitle
  unset multiplot
  unset output
}
