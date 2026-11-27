module

public import Transformer.Grokking.FiniteRecovery.ArgandNative
public import Transformer.Grokking.FiniteRecovery.ArgandPredictionContinuity
public import Transformer.AMSGradW.KerniteLimits

/-! What the initialized, jointly trained model can guarantee. The WSD
floor is positive and imposes a common-gradient condition on a weight
limit. A summable continuation gives a weight and loss limit for every
minibatch stream. Positive decay rules out zero real cross-entropy. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open Filter Transformer.AMSGradW
open scoped Topology

variable {A : Type*} {H d layers T : ℕ}

theorem wsdRate_tendsto_floor (budget : ℕ) (hw : budget * 2 / 100 ≤ budget - 1)
    (hd : budget * 80 / 100 < budget - 1) :
    Tendsto (wsdRate budget) atTop (𝓝 (1 / 100000000)) := by
  apply tendsto_const_nhds.congr'
  exact (eventually_ge_atTop (budget - 1)).mono fun _ hs => (wsdRate_eq_floor hw hd hs).symm

/-- A WSD parameter limit would require consensus of the clipped batch
gradients, with the common value balancing positive decoupled decay. -/
theorem nativeTrainingRun_clippedGrad_of_weightLimit [Fintype A] (repeats budget : ℕ)
    (batches : ℕ → List (LabeledPosition A)) (draw : ComplexParameter A H d layers → ℂ)
    (hw : budget * 2 / 100 ≤ budget - 1) (hd : budget * 80 / 100 < budget - 1)
    (w : Weights A H d layers)
    (h : Tendsto (fun t => (nativeTrainingRun repeats batches (wsdRate budget) draw t).weights)
      atTop (𝓝 w)) :
    let fit := nativeTrainingRun repeats batches (wsdRate budget) draw
    ∃ V, Tendsto (fun t => (fit t).maximum) atTop (𝓝 V) ∧
      Tendsto (fun t i => Kernite.clip (batchGradient repeats (batches t) (fit t).weights) i) atTop
        (𝓝 (fun i => -(1 / 10 : ℝ) * (Real.sqrt (V i) + 1 / 100000000) * w i)) :=
  Kernite.run_clippedGrad_tendsto_of_weights_tendsto _ _ _ w (by norm_num)
    (wsdRate_tendsto_floor budget hw hd) h

/-- No fixed token's exact cross-entropy can tend to zero under the
documented positive-decay WSD run, even when every prediction is correct. -/
theorem nativeTrainingRun_loss_not_tendsto_zero [Fintype A] [Nontrivial A] (repeats budget : ℕ)
    (batches : ℕ → List (LabeledPosition A)) (draw : ComplexParameter A H d layers → ℂ)
    (word : Word A T) (i : Fin T) (a : A) :
    let fit := nativeTrainingRun repeats batches (wsdRate budget) draw
    ¬ Tendsto (fun t => -Real.log ((Model.fromWeights (fit t).weights).probability repeats word i a))
      atTop (𝓝 0) := by
  obtain ⟨δ, hδ, hb⟩ := nativeTrainingRun_loss_floor (T := T) repeats budget batches draw
  exact fun h => hδ.not_ge (ge_of_tendsto h (Eventually.of_forall fun t => hb t word i a))

/-- The unchanged WSD iterates have a common convergent subsequence of
weights and per-position losses for any sequence of minibatches. -/
theorem nativeTrainingRun_subseq [Fintype A] (repeats budget : ℕ)
    (batches : ℕ → List (LabeledPosition A)) (draw : ComplexParameter A H d layers → ℂ) :
    let fit := nativeTrainingRun repeats batches (wsdRate budget) draw
    ∃ (w : Weights A H d layers) (subseq : ℕ → ℕ), StrictMono subseq ∧
      Tendsto (fun t => (fit (subseq t)).weights) atTop (𝓝 w) ∧
      ∀ (word : Word A T) (i : Fin T) (a : A),
        Tendsto (fun t => -Real.log ((Model.fromWeights (fit (subseq t)).weights).probability repeats word i a))
          atTop (𝓝 (-Real.log ((Model.fromWeights w).probability repeats word i a))) := by
  let fit := nativeTrainingRun repeats batches (wsdRate budget) draw
  let B (i : Parameter A H d layers) := max |initialWeights draw i| (10000000000 : ℝ)
  have hrate (t : ℕ) : 0 ≤ wsdRate budget t ∧ wsdRate budget t ≤ 10 :=
    ⟨(wsdRate_bounds budget t).1, (wsdRate_bounds budget t).2.trans (by norm_num)⟩
  have hb (t : ℕ) : (fit t).weights ∈ Set.Icc (fun i => -B i) B := by
    have hi (i : Parameter A H d layers) : -(B i) ≤ (fit t).weights i ∧ (fit t).weights i ≤ B i :=
      abs_le.mp (Kernite.run_abs_weight_le _ _ _ hrate i t)
    exact ⟨fun i => (hi i).1, fun i => (hi i).2⟩
  obtain ⟨w, _, u, hu, ht⟩ := isCompact_Icc.tendsto_subseq hb
  exact ⟨w, u, hu, ht, fun word i a => (continuous_labelLoss_fromWeights repeats word i a).tendsto w |>.comp ht⟩

/-- Retain all scheduled WSD updates. After the budget, halve the final
rate after each additional update instead of holding it indefinitely. -/
noncomputable def summableWsdRate (budget completed : ℕ) : ℝ :=
  if completed < budget then wsdRate budget completed
  else (1 / 100000000 : ℝ) * (1 / 2 : ℝ) ^ (completed - budget)

/-- The continuation has exactly the native state through the entire
scheduled prefix, including all moment buffers. -/
theorem nativeTrainingRun_summableWsd_eq_prefix [Fintype A] (repeats budget : ℕ)
    (batches : ℕ → List (LabeledPosition A)) (draw : ComplexParameter A H d layers → ℂ)
    (t : ℕ) (ht : t ≤ budget) :
    nativeTrainingRun repeats batches (summableWsdRate budget) draw t =
      nativeTrainingRun repeats batches (wsdRate budget) draw t := by
  unfold nativeTrainingRun
  induction t with
  | zero => rfl
  | succ t ih =>
    simp only [Kernite.run, ih ((Nat.le_succ _).trans ht), summableWsdRate, ite_eq_left (Nat.lt_of_succ_le ht)]

/-- The continued run starts at the same initialized model and updates
every parameter. All weights and all per-position losses converge. -/
theorem nativeTrainingRun_tendsto_with_summableWsd [Fintype A] (repeats budget : ℕ)
    (batches : ℕ → List (LabeledPosition A)) (draw : ComplexParameter A H d layers → ℂ) :
    let fit := nativeTrainingRun repeats batches (summableWsdRate budget) draw
    ∃ w : Weights A H d layers, Tendsto (fun t => (fit t).weights) atTop (𝓝 w) ∧
      ∀ (word : Word A T) (i : Fin T) (a : A),
        Tendsto (fun t => -Real.log ((Model.fromWeights (fit t).weights).probability repeats word i a))
          atTop (𝓝 (-Real.log ((Model.fromWeights w).probability repeats word i a))) := by
  have hb (n : ℕ) : 0 ≤ summableWsdRate budget n ∧ summableWsdRate budget n ≤ 10 := by
    unfold summableWsdRate
    split
    · exact ⟨(wsdRate_bounds _ _).1, (wsdRate_bounds _ _).2.trans (by norm_num)⟩
    · exact ⟨by positivity, (mul_le_mul_of_nonneg_left
          (pow_le_one₀ (by norm_num) (by norm_num)) (by norm_num)).trans (by norm_num)⟩
  have hs : Summable (summableWsdRate budget) := by
    refine (summable_nat_add_iff budget).mp ?_
    have he (n : ℕ) : summableWsdRate budget (n + budget) =
        (1 / 100000000 : ℝ) * (1 / 2 : ℝ) ^ n := by
      simp [summableWsdRate, Nat.not_lt.mpr (Nat.le_add_left _ _)]
    simpa only [he] using (summable_geometric_of_lt_one (by norm_num : (0 : ℝ) ≤ 1 / 2)
      (by norm_num)).mul_left (1 / 100000000)
  obtain ⟨w, ht⟩ := Kernite.run_weights_tendsto_of_summable (summableWsdRate budget)
    (fun t => batchGradient repeats (batches t)) (initialWeights draw) hb hs
  exact ⟨w, ht, fun word i a => (continuous_labelLoss_fromWeights repeats word i a).tendsto w |>.comp ht⟩

example := wsdRate_tendsto_floor 20000 (by decide) (by decide)

example (batches : ℕ → List (LabeledPosition (Fin 64))) (draw : ComplexParameter (Fin 64) 4 16 1 → ℂ) :=
  nativeTrainingRun_tendsto_with_summableWsd (T := 32) 1 20000 batches draw

example (batches : ℕ → List (LabeledPosition (Fin 256))) (draw : ComplexParameter (Fin 256) 4 32 1 → ℂ) :=
  nativeTrainingRun_loss_not_tendsto_zero 1 20000 batches draw (fun _ : Fin 32 => 0) 31 0

example (batches : ℕ → List (LabeledPosition (Fin 256))) (draw : ComplexParameter (Fin 256) 4 32 1 → ℂ) :=
  nativeTrainingRun_summableWsd_eq_prefix 1 20000 batches draw 20000 le_rfl

end Transformer.Grokking.FiniteRecovery.Argand
