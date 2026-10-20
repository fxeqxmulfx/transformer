import Transformer.Normalization.Section3_CoupledJacobian

/-!
Axiom checks for the actual coupled-field derivative and the block
eigenvector calculation of arXiv:2510.22026v2, Appendix D, Step 4.
No declaration may use `sorryAx` or an unproved synchronization input.
-/

open Transformer.Normalization

#print axioms coupledJacobian_apply
#print axioms coupledField_eq_zero_iff
#print axioms hasFDerivAt_coupledField_of_critical
#print axioms coupledJacobian_lift_eigenvector
#print axioms coupledJacobian_eigen_iff
#print axioms coupledJacobian_unstable
#print axioms fderiv_coupledField_unstable
