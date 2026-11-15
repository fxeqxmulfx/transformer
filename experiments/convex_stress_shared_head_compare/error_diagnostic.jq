# Read only already-scored raw logits and the complete sealed state.
# Invocation: jq -s --slurpfile state MODEL.json -f error_diagnostic.jq LOGITS.jsonl
map(select(.new_prediction != .target)
  | . as $r
  | $state[0][$r.role] as $m
  | $m.model.input.route[($r.tokens[0] * $m.model.input.n + $r.tokens[1])] as $block
  | $m.initialization.blocks[$block] as $b
  | {role, block:$block, labels:$b.labels, method:$b.method, selected:$b.selected,
     target_known:($b.labels | index($r.target) != null),
     target:$r.target, prediction:$r.new_prediction, tokens})
| {by_method:(group_by([.method,.target_known])
     | map({method:.[0].method,target_known:.[0].target_known,count:length})),
   by_direction:(group_by(.selected) | map({selected:.[0].selected,count:length})),
   missing:[.[] | select(.target_known == false)]}
