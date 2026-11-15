# V2 keeps every v1 fit, decoder, parameter vector, score and product audit.
# The complete model hash changes because its protocol is explicitly versioned.
def old_fields:
  {fold,bank,ridge,direction,refusal,
   state:(if .state == null then null else
      (.state | {bank,ridge,direction,decoder,theta}) end),
   gains:[.gains[] | {gain,score,weighted_mass,complete_mass,product_residual}]};
. as $new | $old[0] as $before |
if (.initialization.fold_record_ids == $before.initialization.fold_record_ids)
   and (.initialization.local_controls == $before.initialization.local_controls)
   and ([.initialization.candidates[] | select(.ridge > 0 and (.calibrated|not)) | old_fields]
        == [$before.initialization.candidates[] | old_fields])
then {retained:true,old_fold_states:($before.initialization.candidates|length),
      new_fold_states:($new.initialization.candidates|length),original_heldout_evaluations:0}
else error("a retained v1 candidate, decoder, parameter, fold or local control changed") end
