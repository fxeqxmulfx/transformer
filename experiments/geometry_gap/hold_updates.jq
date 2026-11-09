# Diagnostic of actual class-contrast motion during the declared nonzero hold.
# Run with jq -n -f hold_updates.jq runs/settle-01/seed*/updates.jsonl.
reduce inputs as $r ({};
  if $r.step > 4000 and $r.step <= 8000 then
    (input_filename | capture("seed(?<seed>[0-9]+)/") | .seed) as $seed |
    ($r.quadratic | add) as $motion |
    .[$seed] //= {updates:0, nonzero_contrast_updates:0, min_lr:1, max_lr:0,
                 cumulative_squared_motion:0, max_squared_motion:0} |
    .[$seed].updates += 1 |
    .[$seed].nonzero_contrast_updates += (if $motion > 0 then 1 else 0 end) |
    .[$seed].min_lr = ([.[$seed].min_lr,$r.eta]|min) |
    .[$seed].max_lr = ([.[$seed].max_lr,$r.eta]|max) |
    .[$seed].cumulative_squared_motion += $motion |
    .[$seed].max_squared_motion = ([.[$seed].max_squared_motion,$motion]|max)
  else . end)
