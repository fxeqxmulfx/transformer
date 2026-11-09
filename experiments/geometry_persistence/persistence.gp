# Usage: gnuplot -e "data='/path/trajectories.tsv'; forecast='/path/forecasts.tsv'; output='/path/persistence.svg'" persistence.gp
if (!exists('raster')) raster=0
if (raster) {
    set terminal pngcairo size 1200,520 enhanced font 'sans,12'
} else {
    set terminal svg size 1200,520 dynamic enhanced font 'sans,12'
}
set output output
set multiplot layout 1,2 title 'Observed persistence under continued AdamW updates'
set border 3
set tics nomirror
set grid ytics lc rgb '#dddddd'
set xlabel 'Training update'
set ylabel 'Active share of centered readout energy (%)'
set xrange [4000:8000]
set yrange [0:80]
set key bottom right font ',10'
set title 'Worst seed at each update: solid bound, dashed observed'
plot data using (strcol(1) eq 'settle' ? $2 : 1/0):3 with lines lw 2 lc rgb '#167d52' title 'Early settling: bound', \
     data using (strcol(1) eq 'settle' ? $2 : 1/0):5 with lines lw 2 dt 2 lc rgb '#167d52' title 'Early settling: observed', \
     data using (strcol(1) eq 'lr001' ? $2 : 1/0):3 with lines lw 2 lc rgb '#306eb0' title '.001 WSD: bound', \
     data using (strcol(1) eq 'lr001' ? $2 : 1/0):5 with lines lw 2 dt 2 lc rgb '#306eb0' title '.001 WSD: observed', \
     data using (strcol(1) eq 'lr01' ? $2 : 1/0):3 with lines lw 2 lc rgb '#c34c42' title '.01 WSD: bound', \
     data using (strcol(1) eq 'lr01' ? $2 : 1/0):5 with lines lw 2 dt 2 lc rgb '#c34c42' title '.01 WSD: observed'
set title 'Early settling: certificates at the window endpoint'
set xlabel 'Window length (updates; different anchor sets)'
set ylabel 'Validation coverage (%)'
set logscale x 2
set xrange [210:4700]
set xtics ('250' 250, '1000' 1000, '4000' 4000)
set yrange [0:103]
set key bottom left
plot forecast using 1:2 with linespoints lw 2 pt 7 lc rgb '#333333' title 'Direct nonlinear forecast', \
     forecast using 1:3 with linespoints lw 2 pt 5 lc rgb '#167d52' title 'Transported rank-33 certificate', \
     forecast using 1:4 with linespoints lw 2 dt 2 pt 4 lc rgb '#999999' title 'Observed correctness'
unset multiplot
