# Post-run structural check for the two MQAR tasks, never used by either learner.
# Given the frozen generator's unique bindings and nonrepeated query keys, this
# route reads the queried binding. Stored token-value rows must copy that value.
.adapter as $a
| ($a.vocab + 2) as $values
| .queries | to_entries as $queries
| {
    query_count: ($queries | length),
    queried_successor_routes: all($queries[];
      (.key | tonumber) as $query
      | (.value.routes | keys) == [(($a.context + $query) | tostring)]
        and .value.routes[(($a.context + $query) | tostring)] == 1),
    identity_on_observed_values: all($queries[];
      .value as $channel
      | all($channel.rows | to_entries[] | select((.key | tonumber) % $values < $a.vocab);
        (.value | index(1)) as $at
        | $at != null and $channel.classes[$at] == ((.key | tonumber) % $values))),
    stored_token_value_rows: ([$queries[].value.rows | to_entries[]
      | select((.key | tonumber) % $values < $a.vocab)] | length)
  }
| if .queried_successor_routes and .identity_on_observed_values then .
  else error("saved recall channel does not have the claimed structure") end
