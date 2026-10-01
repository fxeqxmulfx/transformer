/-
# Entropy dependency audit for arXiv:2505.24832v3

Run with `lake env lean scripts/MemorizationAxioms.lean` after building
`Transformer.Memorization`. Besides the paper declarations, this checks
every public declaration from the vendored PFR entropy import closure.
The audit follows the kernel's transitive axiom dependencies; a hidden
`sorryAx`, including one in an imported proof, makes the command fail.
-/
import Transformer.Memorization
import Lean.Util.FoldConsts

open Lean Elab Command

run_cmd do
  let env ← getEnv
  let standard : List Name := [``propext, ``Classical.choice, ``Quot.sound]
  let mut paper : Nat := 0
  let mut entropyAPI : Nat := 0
  let mut unexpected : Nat := 0
  for (name, _) in env.constants.toList do
    if name.isInternal then continue
    let isPaper := (`Transformer.Memorization).isPrefixOf name
    let isEntropy := match env.getModuleIdxFor? name with
      | some index => (`PFR).isPrefixOf env.header.moduleNames[index]!
      | none => false
    unless isPaper || isEntropy do continue
    if isPaper then paper := paper + 1
    if isEntropy then entropyAPI := entropyAPI + 1
    let axioms ← liftCoreM (collectAxioms name)
    for axiomName in axioms do
      unless standard.contains axiomName do
        unexpected := unexpected + 1
        logError m!"Unexpected transitive axiom: {name}: {axiomName}"
  logInfo m!"{paper} paper declarations; {entropyAPI} entropy-library declarations; \
    {unexpected} unexpected transitive axioms"

#print axioms Transformer.Memorization.proposition1_superadditivity
#print axioms Transformer.Memorization.proposition4_arbitrary_decoder_counterexample
#print axioms Transformer.Memorization.learningCapacity_le_storage_bits
#print axioms Transformer.Memorization.mixture_code_overhead
#print axioms Transformer.Memorization.membership_limit
#print axioms Transformer.Memorization.fitted_limit_above_0835
#print axioms Transformer.Memorization.table_precision_means
#print axioms Transformer.Memorization.validation_discrepancy_exceeds_uncertainty
