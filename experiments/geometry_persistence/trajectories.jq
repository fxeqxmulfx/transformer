# jq -nr -f trajectories.jq RUN_DIRECTORY/*-trace.jsonl
[inputs | . as $r |
  {cohort:(input_filename|capture("(?<name>lr001|lr01|settle)-seed").name),
   step:$r[0], bound:$r[4], actual:$r[6]}] |
group_by([.cohort,.step])[] |
if length != 8 then error("missing fixed seed at trajectory prefix") else . end |
[.[0].cohort, .[0].step, (map(.bound)|min)*100, (map(.bound)|max)*100,
 (map(.actual)|min)*100, (map(.actual)|max)*100] | @tsv
