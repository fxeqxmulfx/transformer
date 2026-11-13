# Native summaries first pass their frozen producer's complete artifact gates.
# This layer additionally checks full identity sets, common data and CE scale.
def require($ok; $why): if $ok then . else error($why) end;
def number: type == "number" and isfinite;
def one($rows; $task): [$rows[] | select(.task == $task)]
  | require(length == 1; "missing or duplicate task") | .[0];
def choice($rows; $task; $mode; $key):
  [$rows[] | select(.task == $task and .[$key] == $mode)]
  | require(length == 1; "missing or duplicate mode choice") | .[0];
def select_ce($rows; $tol): reduce $rows[] as $c (null;
  if . == null or $c.train_ce + $tol < .train_ce then $c else . end);
def audit_vocabulary($audit; $vocab):
  [$audit.vocabulary, $audit.original_vocabulary] | map(select(. != null))
  | length > 0 and all(. == $vocab);
def valid_candidate($c; $expected_splits; $selection):
  require(($c.train_ce | number) and ($c.metrics | type == "object")
    and ($c.primary_pass | type == "boolean") and ($c.all_splits_pass | type == "boolean")
    and ($c.metrics | keys) == $expected_splits
    and ($c.metrics.train.loss - $c.train_ce | fabs) < 1e-9
    and all($c.metrics[]; (.sequence_accuracy | number) and .sequence_accuracy >= 0 and .sequence_accuracy <= 1)
    and $c.primary_pass == ($c.metrics[$selection].sequence_accuracy >= 0.99)
    and $c.all_splits_pass == ([$c.metrics | to_entries[] | select(.key != "train") | .value.sequence_accuracy] | all(. >= 0.99));
    "invalid common-split CE or pass flags");

$general[0] as $g | $families[0] as $fs | $protocol[0] as $p
| $g.parents.full_suite_hull as $h | $g.parents.hull_inventory as $inventory
| [$suite[0].tasks[].name as $task | $p.modes[] | [$task,.]] as $all
| [$p.tasks[] as $task | $p.modes[] | [$task,.]] as $pairs
| require(($all | length) == 105 and ($all | unique | length) == 105
    and ($pairs | length) == 27 and ($pairs | unique | length) == 27
    and ([$h.choices[] | [.task,.arm]] | sort) == ($all | sort)
    and ([$g.choices[] | [.task,.training_mode]] | sort) == ($pairs | sort)
    and ([$inventory.tasks[].task] | sort) == ([$suite[0].tasks[].name] | sort)
    and ([$fs[].name]) == ([$p.families[].name]); "complete parent scope or declared family order changed")
| reduce ($fs | to_entries[]) as $entry (.;
    $entry.value as $f | $f.summary as $s | $p.families[$entry.key] as $pin
    | require($s.source_fingerprint == $pin.source_fingerprint and $s.producer_commit == $pin.producer_commit
        and $s.tasks == 9 and ([$s.records[].task] | sort) == ($p.tasks | sort)
        and ([$s.mode_references[] | [.task,.mode]] | sort) == ($pairs | sort)
        and all($s.mode_references[]; .standalone_constructor_reference == true and .gradient_training_run == false)
        and $s.process_failures == ([$s.records[] | select(.status == "process_failure")] | length);
        "family source, constructor references or terminal coverage changed")
    | reduce $s.records[] as $r (.;
        one($inventory.tasks; $r.task) as $meta
        | [$suite[0].tasks[] | select(.name == $r.task)][0] as $spec
        | [$s.mode_references[] | select(.task == $r.task)] as $refs
        | if $r.status == "process_failure" then
            require(all($refs[]; .status == "process_failure" and .selected == null);
              "failed constructor supplied mode scores")
          else
            $r.result as $result | $result.evaluation as $evaluation
            | require($result.metadata.source_fingerprint == $s.source_fingerprint
                and $result.metadata.producer_commit == $s.producer_commit
                and $result.metadata.input.task == $r.task and $result.metadata.input.vocab == $meta.vocab
                and $result.metadata.input.data_fingerprints == $meta.data_fingerprints
                and $result.metadata.input.selection == $spec.selection and $meta.scale == 1
                and $result.metadata.training_gradient_evaluations == 0
                and ($result.construction_seconds | number) and $result.construction_seconds >= 0
                and ($evaluation.candidates | map(.id) | length) == ($evaluation.candidates | map(.id) | unique | length)
                and ($evaluation.status == "complete" or $evaluation.status == "proposal_ineligible")
                and ($evaluation.status == "complete") == (($evaluation.candidates|length)>0);
                "family data, scale, candidate identities or construction cost changed")
            | reduce $evaluation.candidates[] as $c (.; valid_candidate($c; ($meta.data_fingerprints|keys); $spec.selection)
                | require(($c.id | type == "string") and ($c.model_sha256 | type == "string")
                    and audit_vocabulary($c.audit; $meta.vocab)
                    and ($c.audit.train_ce - $c.train_ce | fabs) < 1e-10;
                    "candidate raw CE binding differs"))
            | select_ce($evaluation.candidates; $p.ce_tie_tolerance) as $selected
            | require($evaluation.selected == $selected.id
                and all($refs[]; .selected == $selected); "family CE choice or three mode references differ")
          end))
| [$all[] as $id | $id[0] as $task | $id[1] as $mode
    | choice($h.choices; $task; $mode; "arm") as $old
    | one($inventory.tasks; $task) as $meta
    | [$suite[0].tasks[] | select(.name == $task)][0] as $spec
    | ($p.tasks | index($task) != null) as $updated
    | if $updated then choice($g.choices; $task; $mode; "training_mode")
      else {status:$old.status,selected:(if $old.status == "selected" then $old + {train_ce:$old.training_ce,vocab:$meta.vocab} else null end)} end as $baseline
    | require(($baseline.status == "selected" or $baseline.status == "no_model")
        and ($baseline.status == "selected") == ($baseline.selected != null);
        "nonterminal earlier choice")
    | [($baseline.selected | select(. != null) | . + {family:"earlier",constructor_reference:false}),
        (if $updated then $fs[] as $f | one($f.summary.records; $task) as $r
          | if $r.status == "process_failure" then empty
            else $r.result.evaluation.candidates[] + {family:$f.name,vocab:$meta.vocab,
              constructor_reference:true,gradient_training_run:false} end
          else empty end)] as $candidates
    # Sequence parents can derive novel subsets from frozen raw data. Their
    # independently replayed metric inventory supplies the exact split set.
    | (if $updated then ($meta.data_fingerprints|keys) else (($old.metrics // {})|keys) end) as $expected_splits
    | reduce $candidates[] as $c (.; valid_candidate($c; $expected_splits; $spec.selection)
        | if $updated then
            require($c.vocab == $meta.vocab and $meta.scale == 1
                and $c.train_ce + 1e-8 >= ((1 + ($meta.vocab-1)*((-1)|exp))|log);
                "incomparable vocabulary, scale or CE lower bound")
          else . end)
    | select_ce($candidates; $p.ce_tie_tolerance) as $winner
    | {task:$task,training_mode:$mode,tier:$old.tier,selection:$spec.selection,
        status:(if $winner == null then "no_model" else "selected" end),selected:$winner,
        updated_scope:$updated,candidates:$candidates,baseline:$baseline,
        primary_pass:($winner.primary_pass == true),all_splits_pass:($winner.all_splits_pass == true)}] as $choices
| {protocol:$p,choices_compared:105,choices:$choices,
    by_mode:[$p.modes[] as $mode | [$choices[] | select(.training_mode == $mode)] as $rows
      | {training_mode:$mode,cases:35,no_model:([$rows[]|select(.status == "no_model")]|length),
          primary_passes:([$rows[]|select(.primary_pass)]|length),all_split_passes:([$rows[]|select(.all_splits_pass)]|length)}],
    constructor_accounting:[$fs[] | {family:.name,planned_constructions:9,mode_references:27,
      process_failures:.summary.process_failures,
      completed_seconds:([.summary.records[]|select(.status != "process_failure")|.result.construction_seconds]|add // 0),
      ce_gradient_evaluations:0,charged_once:true}],
    parents:{complete_general_comparison:$g,complete_constructor_families:$fs},
    full35_result:true,full35_pass:all($choices[];.all_splits_pass),fresh_transformer_comparison:false,
    pending_followups:["retained_depth_integration","composite_code_original_cohort","fresh_transformer_comparison"]}
