import Transformer

/-! The statements that the claim of `review/grok_division_convergence.md`, section 6, rests on:
what is decided about "the held-out accuracy of the division transformer tends to 1". Each is
printed with its hypotheses, then with the axioms it depends on. The statements of the claim
itself are then printed with the modules of this library that they rest on, and with what they
use of the one such module that also holds a theorem with a hypothesis on the future of a run.

`LAKE_ARTIFACT_CACHE=false lake env lean scripts/GrokConvergenceClaim.lean` -/

open Lean Elab Command

/-- The constants of this library that a declaration depends on, through every constant that
its statement and its proof use. Constants of other libraries are not entered. The constants
are read from the kernel's environment, as `#print axioms` reads them, so that proofs are
seen. -/
partial def libraryConstants (env : Environment) (kenv : Kernel.Environment) (todo : Array Name)
    (seen used : NameSet) : NameSet :=
  if todo.isEmpty then used else
    let n := todo.back!
    let todo := todo.pop
    if seen.contains n then libraryConstants env kenv todo seen used else
      let seen := seen.insert n
      match env.getModuleIdxFor? n, kenv.find? n with
      | some idx, some info =>
        if (`Transformer).isPrefixOf env.header.moduleNames[idx.toNat]! then
          let next := info.type.getUsedConstants ++
            (match info.value? (allowOpaque := true) with
              | some v => v.getUsedConstants
              | none => #[])
          libraryConstants env kenv (todo ++ next) seen (used.insert n)
        else libraryConstants env kenv todo seen used
      | _, _ => libraryConstants env kenv todo seen used

/-- The module that declares a constant. -/
def moduleOf (env : Environment) (c : Name) : Option Name :=
  (env.getModuleIdxFor? c).map fun idx => env.header.moduleNames[idx.toNat]!

/-- `#library_modules c` prints the modules of this library that `c` rests on. -/
elab "#library_modules " id:ident : command => do
  let n ← liftCoreM <| realizeGlobalConstNoOverloadWithInfo id
  let env ← getEnv
  let used := libraryConstants env env.checked.get #[n] {} {}
  let mods := (used.toList.filterMap (moduleOf env)).eraseDups
  let names := mods.map toString |>.toArray.qsort (· < ·)
  logInfo m!"{n} rests on {names.size} modules of this library: {names.toList}"

/-- `#library_constants M of c₁ c₂ …` prints, for each `cᵢ`, the named constants of the module
`M` that it rests on. -/
elab "#library_constants " m:ident " of " ids:ident+ : command => do
  let env ← getEnv
  for id in ids do
    let n ← liftCoreM <| realizeGlobalConstNoOverloadWithInfo id
    let used := libraryConstants env env.checked.get #[n] {} {}
    let here := used.toList.filter fun c =>
      moduleOf env c == some m.getId && !c.isInternalDetail
    let names := here.map toString |>.toArray.qsort (· < ·)
    logInfo m!"{n} uses {names.size} named constants of {m.getId}: {names.toList}"

open Transformer.Grokking.Protocol

/-! The target, and the event that it speaks of. -/
/-! The target, and the event that it speaks of. -/
#check division_generalizes_iff
#check measurableSet_eventuallyRight
#check division_generalizes_iff_law_eq_one
#print axioms division_generalizes_iff
#print axioms measurableSet_eventuallyRight
#print axioms division_generalizes_iff_law_eq_one

/-! The declared target is false: without a condition on the split, in three modes. -/
#check not_division_generalizes
#check not_division_tendsToOneInProbability
#check not_division_tendsToOneInMean
#print axioms not_division_generalizes
#print axioms not_division_tendsToOneInProbability
#print axioms not_division_tendsToOneInMean

/-! It is false under every condition that admits a split that is not labelled, and on the
labelled splits, in the same three modes. -/
#check not_ae_eventually_of_not_mem_labelled
#check not_tendsto_law_inter_wrongAt
#check not_tendsto_setIntegral_divisionAccuracy
#check not_ae_eventually_of_twisted
#check not_ae_eventually_labelled
#check not_ae_eventually_of_twin_subset
#check not_tendsto_law_of_twin_subset
#check not_tendsto_setIntegral_of_twin_subset
#print axioms not_ae_eventually_of_not_mem_labelled
#print axioms not_tendsto_law_inter_wrongAt
#print axioms not_tendsto_setIntegral_divisionAccuracy
#print axioms not_ae_eventually_of_twisted
#print axioms not_ae_eventually_labelled
#print axioms not_ae_eventually_of_twin_subset
#print axioms not_tendsto_law_of_twin_subset
#print axioms not_tendsto_setIntegral_of_twin_subset

/-! The probability of the event is bounded from above. -/
#check law_eventuallyRight_le
#check law_eventuallyRight_le_of_twisted
#check not_le_law_eventuallyRight
#check law_eventuallyRight_le_explicit
#check not_le_law_eventuallyRight_explicit
#print axioms law_eventuallyRight_le
#print axioms law_eventuallyRight_le_of_twisted
#print axioms not_le_law_eventuallyRight
#print axioms law_eventuallyRight_le_explicit
#print axioms not_le_law_eventuallyRight_explicit

/-! The probabilities of the conditions on the split. -/
#check uniformPerm_unlabelled_eq
#check uniformPerm_compl_labelled_mem_Icc
#check uniformPerm_labelled_mem_Icc
#check uniformPerm_anchored_eq
#print axioms uniformPerm_unlabelled_eq
#print axioms uniformPerm_compl_labelled_mem_Icc
#print axioms uniformPerm_labelled_mem_Icc
#print axioms uniformPerm_anchored_eq

/-! Routes to a positive statement that the runs of the protocol exclude. -/
#check division_eventually_margin_lt
#check exists_margin_le_zero_of_tendsto_division
#check ae_abs_logits_divisionRun_le
#check ae_floor_le_batchLoss_divisionRun
#check ae_margin_divisionRun_le
#check not_tendsto_adamRun_factor
#check Transformer.Grokking.FlatSpectrum.Training.not_exists_forall_ceLoss_le
#print axioms division_eventually_margin_lt
#print axioms exists_margin_le_zero_of_tendsto_division
#print axioms ae_abs_logits_divisionRun_le
#print axioms ae_floor_le_batchLoss_divisionRun
#print axioms ae_margin_divisionRun_le
#print axioms not_tendsto_adamRun_factor
#print axioms Transformer.Grokking.FlatSpectrum.Training.not_exists_forall_ceLoss_le

/-! What the trajectory is: the code's update, how far it moves, and that it is measurable. -/
#check adamStep_eq_sourceStep
#check abs_adamRun_params_sub_le
#check abs_adamRun_params_le
#check measurable_divisionRun
#check measurable_divisionAccuracy
#check measurableSet_wrongAt
#check hasDerivAt_batchLoss_sourceGrad
#print axioms adamStep_eq_sourceStep
#print axioms abs_adamRun_params_sub_le
#print axioms abs_adamRun_params_le
#print axioms measurable_divisionRun
#print axioms measurable_divisionAccuracy
#print axioms measurableSet_wrongAt
#print axioms hasDerivAt_batchLoss_sourceGrad

/-! A structure of the logits that gives held-out accuracy 1, as a sufficient condition; built
parameters have it, trained ones are not proved to. -/
#check heldout_accuracy_eq_one_of_training_anchors
#check heldout_accuracy_solver
#check exists_fit_training_wrong_heldout
#check heldout_accuracy_eq_one_of_enclosed_anchors
#print axioms heldout_accuracy_eq_one_of_training_anchors
#print axioms heldout_accuracy_solver
#print axioms exists_fit_training_wrong_heldout
#print axioms heldout_accuracy_eq_one_of_enclosed_anchors

/-! The modules of this library that the statements of the claim rest on. -/
#library_modules measurableSet_eventuallyRight
#library_modules division_generalizes_iff_law_eq_one
#library_modules not_division_generalizes
#library_modules not_division_tendsToOneInProbability
#library_modules not_division_tendsToOneInMean
#library_modules not_ae_eventually_of_not_mem_labelled
#library_modules not_tendsto_law_inter_wrongAt
#library_modules not_tendsto_setIntegral_divisionAccuracy
#library_modules not_ae_eventually_of_twisted
#library_modules not_ae_eventually_labelled
#library_modules not_ae_eventually_of_twin_subset
#library_modules not_tendsto_law_of_twin_subset
#library_modules not_tendsto_setIntegral_of_twin_subset
#library_modules law_eventuallyRight_le_explicit
#library_modules not_le_law_eventuallyRight_explicit

/-! Three of those modules belong to section 5 of the review, whose theorems speak of the
future of a run. `ReadoutStationary` holds `division_eventually_margin_lt`, whose hypothesis is
that the parameters converge. What the claim uses of the three: -/
#library_constants Transformer.Grokking.Protocol.ReadoutStationary of
  measurableSet_eventuallyRight division_generalizes_iff_law_eq_one not_division_generalizes
  not_division_tendsToOneInProbability not_division_tendsToOneInMean
  not_ae_eventually_of_not_mem_labelled not_tendsto_law_inter_wrongAt
  not_tendsto_setIntegral_divisionAccuracy not_ae_eventually_of_twisted
  not_ae_eventually_labelled not_ae_eventually_of_twin_subset not_tendsto_law_of_twin_subset
  not_tendsto_setIntegral_of_twin_subset law_eventuallyRight_le_explicit
  not_le_law_eventuallyRight_explicit
#library_constants Transformer.Grokking.Protocol.TargetMean of
  measurableSet_eventuallyRight division_generalizes_iff_law_eq_one not_division_generalizes
  not_division_tendsToOneInProbability not_division_tendsToOneInMean
  not_ae_eventually_of_not_mem_labelled not_tendsto_law_inter_wrongAt
  not_tendsto_setIntegral_divisionAccuracy not_ae_eventually_of_twisted
  not_ae_eventually_labelled not_ae_eventually_of_twin_subset not_tendsto_law_of_twin_subset
  not_tendsto_setIntegral_of_twin_subset law_eventuallyRight_le_explicit
  not_le_law_eventuallyRight_explicit
#library_constants Transformer.Grokking.Protocol.Continuity of
  measurableSet_eventuallyRight division_generalizes_iff_law_eq_one not_division_generalizes
  not_division_tendsToOneInProbability not_division_tendsToOneInMean
  not_ae_eventually_of_not_mem_labelled not_tendsto_law_inter_wrongAt
  not_tendsto_setIntegral_divisionAccuracy not_ae_eventually_of_twisted
  not_ae_eventually_labelled not_ae_eventually_of_twin_subset not_tendsto_law_of_twin_subset
  not_tendsto_setIntegral_of_twin_subset law_eventuallyRight_le_explicit
  not_le_law_eventuallyRight_explicit
