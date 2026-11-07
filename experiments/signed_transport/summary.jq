# Presentation only: preserve every case/method and summarize detailed geometry by fold.
# The complete unfiltered report is retained as report.json.gz.
def geometry_by_fold:
  .groups as $groups |
  [range(0;3) as $fold |
   [$groups[].folds[]? | select(.fold == $fold and .geometry != null)] as $folds |
   {fold:$fold,
    groups:($folds|length),
    queries:([$folds[].query_count]|add),
    ranks:
      [([0,1,2,4,8,16,32,64,128][]) as $rank |
       [$folds[] | .query_count as $n | .geometry.ranks[] |
        select(.requested_rank == $rank) | . + {queries:$n}] as $rows |
       ([$rows[].queries]|add) as $n |
       {requested_rank:$rank,
        achieved_rank_min:([$rows[].achieved_rank]|min),
        achieved_rank_max:([$rows[].achieved_rank]|max),
        groups_clamped:([$rows[]|select(.achieved_rank < $rank)]|length),
        retained_training_energy_mean:(([$rows[]|.queries*.retained_training_energy]|add)/$n),
        residual_norm_mean:(([$rows[]|.queries*.residual_norm.mean]|add)/$n),
        relative_residual_norm_mean:(([$rows[]|.queries*.relative_residual_norm.mean]|add)/$n),
        coefficient_norm_mean:(([$rows[]|.queries*.coefficient_norm.mean]|add)/$n),
        readout_residual_norm_mean:(([$rows[]|.queries*.readout_residual_norm.mean]|add)/$n),
        relative_readout_residual_norm_mean:(([$rows[]|.queries*.relative_readout_residual_norm.mean]|add)/$n),
        orthogonality_defect_penalty_max:([$rows[].orthogonality_defect_penalty.max]|max),
        basis_reconstruction_max_error:([$rows[].basis_reconstruction_max_error]|max),
        basis_orthogonality_max_error:([$rows[].basis_orthogonality_max_error]|max)}]}];

.cases |= map(
  .diagnostic.sum |= (. + {geometry_by_fold:geometry_by_fold} | del(.groups)) |
  .diagnostic.orbit |= (. + {geometry_by_fold:geometry_by_fold} | del(.groups))
) |
. + {detailed_report:"report.json.gz",presentation:"summary.jq"}
