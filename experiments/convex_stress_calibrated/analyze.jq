# Post-run analysis only. Do not select a predictor or change a training budget.
def absolute: if . < 0 then -. else . end;
if .planned != 70 or (.records | length) != 70
  or (.records | group_by(.task) | length) != 35
  or any(.records | group_by(.task)[]; (map(.arm) | sort) != ["calibrated", "scaled"])
then error("expected both readouts on every original task")
else . end
| [.records | group_by(.task)[] |
    (map(select(.arm == "scaled"))[0]) as $old |
    (map(select(.arm == "calibrated"))[0]) as $new |
    {task:$old.task, tier:$old.tier, scaled:$old, calibrated:$new}]
  as $paired
| {
    fingerprint, totals,
    execution_failures:[.records[] | select(.status != "complete")],
    new_primary_passes:[$paired[] | select(.calibrated.primary_pass and (.scaled.primary_pass|not)) | .task],
    lost_primary_passes:[$paired[] | select(.scaled.primary_pass and (.calibrated.primary_pass|not)) | .task],
    empirical_optima_with_failed_primary:[.records[] |
      select(.status == "complete" and .train_accuracy == 1 and (.primary_pass|not)
        and ((.fresh_fw_gap|absolute) < 1e-7)) |
      {task,arm,primary_accuracy,training_ce,fresh_fw_gap}],
    budget_endpoints:[.records[] | select(.stop_reason == "time_cap" or .stop_reason == "update_cap") |
      {task,arm,updates,seconds,stop_reason,train_accuracy,primary_accuracy,fresh_fw_gap}],
    paired:$paired
  }
