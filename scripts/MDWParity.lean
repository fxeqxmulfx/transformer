import Transformer.MDW

/-!
# Axiom audit of the MDW recurrence, storage and shared numerical checkpoints

The corresponding literal values are frozen in test_mdw_state.py. This
checks exact-real definitions and does not model PyTorch floating-point kernels.
-/

#print axioms Transformer.MDW.correctWeight_weight
#print axioms Transformer.MDW.correctWeight_norm
#print axioms Transformer.MDW.correctWeight_zero
#print axioms Transformer.MDW.correctWeight_zero_decay
#print axioms Transformer.MDW.adam_auxiliary_native
#print axioms Transformer.MDW.freeStepWith_adam
#print axioms Transformer.MDW.freeStepWith_amsgrad
#print axioms Transformer.MDW.freeMDStep_base_weight
#print axioms Transformer.MDW.freeMDStep_base_norm
#print axioms Transformer.MDW.freeMDStep_histories
#print axioms Transformer.MDW.adam_callback_checkpoint
#print axioms Transformer.MDW.adamw_auxiliary_checkpoint
#print axioms Transformer.MDW.amsgrad_callback_checkpoint
#print axioms Transformer.MDW.free_variance_checkpoint
#print axioms Transformer.MDW.free_mdw_decay_checkpoint
