# Reduce one sealed forecast at a time before slurping the small case summaries.
def stats: {min:min,mean:(add/length),max:max};
{cohort,seed,anchor,old_q,old_contrast_norm,s,metrics,
 forecasts:[.forecasts[]|{horizon,q,coefficients,
   feature_radius:(.feature_radii|stats),
   contrast_budget:(.contrast_budgets|stats),
   probability_ceiling:(.probability_ceilings|stats),
   row_cap:(.parameter_caps[:97]|stats),matrix_caps:.parameter_caps[97:]}]}
