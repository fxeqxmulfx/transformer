import Transformer.MeanField.Clustering

/-!
Axiom checks for the original SA/USA clustering theorem of
arXiv:2512.01868v4, Section 4, and its new constructive USA existence proof.
No statement below may depend on `sorryAx` or a nonstandard extra axiom.
-/

open Transformer

#print axioms Perspective.usaExtensionAverage_eq
#print axioms Perspective.usaGlobalField_apply
#print axioms Perspective.usaExtensionExponent_le
#print axioms Perspective.contDiff_usaExtensionAverage
#print axioms Perspective.contDiff_usaGlobalField
#print axioms Perspective.norm_usaExtensionAverage_le
#print axioms Perspective.norm_usaGlobalField_le
#print axioms Perspective.norm_eq_one_of_usaGlobalField
#print axioms Perspective.exists_USA
#print axioms Perspective.hasDerivAt_usaGlobalField_of_USA
#print axioms Perspective.USA_unique
#print axioms Perspective.exists_unique_USA
#print axioms MeanField.global_clustering
