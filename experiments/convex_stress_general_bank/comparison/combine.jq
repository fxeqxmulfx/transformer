# Inputs have already passed their native replay/hash gates. Check the complete
# identity sets again, retain every parent, and choose solely by training CE.
def require($ok; $why): if $ok then . else error($why) end;
def identity: [.task, .training_mode];
def finite_number: type == "number" and isfinite;
def once($rows; $task; $mode; $mode_key):
  [$rows[] | select(.task == $task and .[$mode_key] == $mode)]
  | require(length == 1; "missing or duplicate parent choice") | .[0];
def check_candidate($c; $vocab):
  require($c.vocab == $vocab and ($c.train_ce | finite_number)
    and ($c.primary_pass | type == "boolean")
    and ($c.all_splits_pass | type == "boolean")
    and ($c.metrics | type == "object"); "invalid comparable candidate");
def choose($rows; $ideal; $tol):
  [$rows | to_entries[] | select(.value.train_ce - $ideal <= $tol) | .key] as $near
  | if $near | length > 0 then $near[0]
    else ($rows | to_entries | min_by([.value.train_ce, .key]) | .key) end;

$general[0] as $g | $certified[0] as $c | $hull[0] as $h
| $prepared[0] as $inventory | $protocol[0] as $p
| [$p.tasks[] as $t | $p.arms[] | [$t, .]] as $planned
| [$planned[] | select(.[0] != $p.added_task)] as $old
| [$suite[0].tasks[].name as $t | $p.arms[] | [$t, .]] as $all
| require(($planned | length) == 27 and ($old | length) == 24
    and ($all | unique | length) == 105; "comparison scope changed")
| require($g.source_fingerprint == $p.general_fingerprint
    and $c.source_fingerprint == $p.certified_fingerprint
    and $h.source_fingerprint == $p.hull_fingerprint
    and $inventory.source_fingerprint == $p.hull_fingerprint;
    "parent fingerprint mismatch")
| require($g.cases == 27 and $g.planned_decoder_slots == 108
    and ([$g.records[] | [.case.task, .case.arm]] | sort) == ($planned | sort)
    and ([$g.choices[] | identity] | sort) == ($planned | sort);
    "general records or choices incomplete")
| require(([$c.choices[] | identity] | sort) == ($old | sort)
    and ([$h.choices[] | [.task, .arm]] | sort) == ($all | sort)
    and ([$inventory.tasks[].task] | sort) == ([$suite[0].tasks[].name] | sort);
    "complete parent scope required")
| require(all($g.candidates[]; ([.task, .training_mode] as $id
      | any($planned[]; . == $id)) and (.decoder_index | floor == . and . >= 0 and . < 4))
    and ([$g.candidates[] | [.task, .training_mode, .decoder_index]] | length)
      == ([$g.candidates[] | [.task, .training_mode, .decoder_index]] | unique | length);
    "duplicate or unplanned general candidate")
| [$planned[] as $id | $id[0] as $task | $id[1] as $mode
    | once($g.choices; $task; $mode; "training_mode") as $own
    | once($g.records | map(. + {task:.case.task, training_mode:.case.arm});
        $task; $mode; "training_mode") as $record
    | if $task == $p.added_task then
        once($h.choices; $task; $mode; "arm") as $prior
        | [$inventory.tasks[] | select(.task == $task)] as $metadata
        | require(($metadata | length) == 1 and $metadata[0].scale == 1;
            "blind baseline has a different CE scale")
        | {prior:$prior, candidate:(if $prior.status == "no_model" then null
          elif $prior.status == "selected" then
            ($prior + {kind:"hull", vocab:$metadata[0].vocab, train_ce:$prior.training_ce})
          else error("nonterminal hull choice") end)}
      else
        once($c.choices; $task; $mode; "training_mode") as $prior
        | {prior:$prior, candidate:$prior.selected}
      end as $previous
    | [$g.candidates[] | select(.task == $task and .training_mode == $mode)]
        | sort_by(.decoder_index) as $fresh
    | require($own.status == $record.status
        and (if $record.status == "process_failure" then
          ($fresh | length) == 0 and $record.planned_decoder_slots == 4
        elif $record.status == "complete" or $record.status == "bank_inconsistent"
            or $record.status == "no_candidate" then
          ($record.candidates | map(.decoder_index)) == [0,1,2,3]
          and all($record.candidates[]; .status == "complete" or .status == "refused")
          and ([$record.candidates[] | select(.status == "complete") | .decoder_index] | sort)
              == ([$fresh[].decoder_index] | sort)
        else false end); "unaccounted general outcome")
    | ($fresh + [$previous.candidate | select(. != null)]) as $comparison
    | if $comparison | length == 0 then
        {task:$task, training_mode:$mode, status:"no_model", selected:null,
          general_selected:false, candidates:[], baseline:$previous.prior, general_record:$record}
      else
        $comparison[0].vocab as $vocab
        | require(($vocab | type == "number") and $vocab >= 2 and $vocab <= 1024
            and ($vocab | floor) == $vocab; "invalid comparison vocabulary")
        | reduce $comparison[] as $candidate (.; check_candidate($candidate; $vocab))
        | ((1 + ($vocab - 1) * ((-1) | exp)) | log) as $ideal
        | require(all($comparison[]; .train_ce + $p.ideal_ce_tolerance >= $ideal);
            "CE below the common lower bound")
        | choose($comparison; $ideal; $p.ideal_ce_tolerance) as $winner
        | {task:$task, training_mode:$mode, status:"selected", selected:$comparison[$winner],
            general_selected:($winner < ($fresh | length)), candidates:$comparison,
            baseline:$previous.prior, general_record:$record}
      end] as $choices
| {protocol:$p, choices_compared:($choices | length), choices:$choices,
    general_selected_choices:([$choices[] | select(.general_selected)] | length),
    primary_pass_choices:([$choices[] | select(.selected.primary_pass == true)] | length),
    all_split_pass_choices:([$choices[] | select(.selected.all_splits_pass == true)] | length),
    by_arm:[$p.arms[] as $mode | [$choices[] | select(.training_mode == $mode)] as $rows
      | {training_mode:$mode, cases:($rows | length),
          no_model:([$rows[] | select(.status == "no_model")] | length),
          primary_passes:([$rows[] | select(.selected.primary_pass == true)] | length),
          all_split_passes:([$rows[] | select(.selected.all_splits_pass == true)] | length)}],
    parents:{general:$g, certified:$c, full_suite_hull:$h, hull_inventory:$inventory},
    integrated_with_older_candidates:true, full35_result:false, fresh_transformer_comparison:false}
