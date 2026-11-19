module

public import Transformer.Grokking.Convex.CoupledMoments.ReadoutKernel.TieWitness
public import Transformer.Grokking.Convex.CoupledMoments.ReadoutKernel.Gauges

/-!
# Finite readout constraints for complete observed kernels

Source: `review/convex/moments.md`, CM35--CM37. Equal code/answer cardinalities
derive the declared readout's fiber equalities from compatible unknown codes.
Complete grouped-kernel matches therefore identify original answers under the
existing target-family assumptions. A strict unseen-corner witness gains
coverage while preserving raw ambiguity. Cardinality, surjectivity and
compatibility counterexamples delimit the claim. The actual complex head
retains raw-vocabulary CE, finite updates and its budget; no native benchmark
or general unknown-decoder discovery is asserted. Grouping preserves complete
kernel match coverage. Actual refined complex heads can have exactly tied
training CE while making opposite unseen pooled predictions; the three-corner
witness retains full, mini and zero-update schedules and both budgets.
Coverage and ordered-search results survive independently chosen gauge anchors.
-/
