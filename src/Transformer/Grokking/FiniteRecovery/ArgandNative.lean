module

public import Transformer.Grokking.FiniteRecovery.ArgandParameters

/-! Bounds for the actual jointly trained Argand parameterization and the
warmup, stable and decay rates documented in ARGAND.md. Proofs are in real
arithmetic and include any order of the observed minibatches. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open Filter Transformer.PhasorV2 Transformer.AMSGradW
open scoped Topology InnerProductSpace

variable {A : Type*} {H d layers T : ℕ}

/-- The runner's WSD rate clamps the completed update count to the last
scheduled update. In particular its floor is positive. -/
noncomputable def wsdRate (budget completed : ℕ) : ℝ :=
  let k := min completed (budget - 1)
  let warmup := budget * 2 / 100
  let decayStart := budget * 80 / 100
  if k < warmup then (1 / 1000 : ℝ) * k / warmup
  else if k < decayStart then 1 / 1000
  else
    let u := ((k - decayStart : ℕ) : ℝ) / (budget - 1 - decayStart : ℕ)
    1 / 1000 + (1 / 100000000 - 1 / 1000) * ((1 - Real.cos (Real.pi * u)) / 2)

theorem wsdRate_bounds (budget step : ℕ) : 0 ≤ wsdRate budget step ∧ wsdRate budget step ≤ 1 / 1000 := by
  dsimp only [wsdRate]
  split
  · rename_i h
    have hpos : (0 : ℝ) < (budget * 2 / 100 : ℕ) := by exact_mod_cast (Nat.zero_le _).trans_lt h
    exact ⟨by positivity, (div_le_iff₀ hpos).2
      (mul_le_mul_of_nonneg_left (by exact_mod_cast h.le) (by norm_num))⟩
  · split
    · constructor <;> norm_num
    · constructor <;> linarith [Real.neg_one_le_cos
        (Real.pi * ((↑(min step (budget - 1) - budget * 80 / 100) : ℝ) / ↑(budget - 1 - budget * 80 / 100))),
        Real.cos_le_one
        (Real.pi * ((↑(min step (budget - 1) - budget * 80 / 100) : ℝ) / ↑(budget - 1 - budget * 80 / 100)))]

theorem wsdRate_eq_floor {budget step : ℕ} (hw : budget * 2 / 100 ≤ budget - 1)
    (hd : budget * 80 / 100 < budget - 1) (hs : budget - 1 ≤ step) :
    wsdRate budget step = 1 / 100000000 := by
  simp only [wsdRate, min_eq_right hs, ite_eq_right (Nat.not_lt.mpr hw), ite_eq_right (Nat.not_lt.mpr hd.le)]
  rw [div_self (Nat.cast_ne_zero.mpr (Nat.sub_ne_zero_of_lt hd)), mul_one, Real.cos_pi]
  norm_num

private theorem norm_argandRMS_bound {c : ℕ} {ε : ℝ} (hε : 0 < ε) (x : CSpace c) :
    ‖argandRMS ε x‖ ≤ Real.sqrt c := by
  have hD : 0 < ‖x‖ + ε := by positivity
  rw [argandRMS, norm_smul, Complex.norm_real, Real.norm_of_nonneg (by positivity), div_mul_eq_mul_div]
  apply (div_le_iff₀ hD).2
  exact mul_le_mul_of_nonneg_left (le_add_of_nonneg_right hε.le) (Real.sqrt_nonneg _)

private theorem abs_logits_le (model : Model A H d layers) (heps : 0 < model.eps)
    (repeats : ℕ) (word : Word A T) (i : Fin T) (a : A) :
    |model.logits repeats word i a| ≤ ‖model.embedding a‖ * Real.sqrt (H * d : ℕ) / 2 := by
  have hs : ‖model.hidden repeats word i‖ ≤ Real.sqrt (H * d : ℕ) :=
    norm_argandRMS_bound heps _
  rw [Model.logits, Model.readout, abs_div, abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 2)]
  apply div_le_div_of_nonneg_right _ (by norm_num)
  exact (Complex.abs_re_le_norm _).trans
    ((norm_inner_le_norm (𝕜 := ℂ) _ _).trans (mul_le_mul_of_nonneg_left hs (norm_nonneg _)))

/-- Bounded tied embeddings alone bound the achievable cross-entropy
away from zero for every supervised token when a competitor exists. -/
theorem Model.lossGe_of_embeddingBound [Fintype A] [Nontrivial A] (model : Model A H d layers)
    (heps : 0 < model.eps) {B : ℝ} (hB : ∀ a, ‖model.embedding a‖ ≤ B)
    (repeats : ℕ) (word : Word A T) (i : Fin T) (a : A) :
    Real.log (1 + Real.exp (-B * Real.sqrt (H * d : ℕ))) ≤
      -Real.log (model.probability repeats word i a) := by
  classical
  let W := Real.sqrt (H * d : ℕ)
  let z := model.logits repeats word i
  let p := fun t => Real.exp (z t)
  have hz (b : A) : |z b| ≤ B * W / 2 :=
    (abs_logits_le model heps repeats word i b).trans
      (div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_right (hB b) (Real.sqrt_nonneg _)) (by norm_num))
  obtain ⟨b, hb⟩ := exists_ne a
  have hgap : z a + -B * W ≤ z b := by
    linarith [(abs_le.mp (hz a)).2, (abs_le.mp (hz b)).1]
  have hfit : p a * (1 + Real.exp (-B * W)) ≤ ∑ j, p j := by
    calc
      _ = p a + Real.exp (z a + -B * W) := by
        rw [mul_add, mul_one, Real.exp_add]
      _ ≤ p a + p b := add_le_add_right (Real.exp_le_exp.mpr hgap) _
      _ ≤ ∑ j, p j := by
        rw [← Finset.sum_erase_add Finset.univ p (Finset.mem_univ a), add_comm (p a)]
        exact add_le_add_left (Finset.single_le_sum
          (fun j _ => (Real.exp_pos (z j)).le) (Finset.mem_erase.mpr ⟨hb, Finset.mem_univ _⟩)) _
  have hp : 0 < p a := Real.exp_pos _
  have hsum : 0 < ∑ j, p j :=
    (mul_pos hp (by positivity : 0 < 1 + Real.exp (-B * W))).trans_le hfit
  have hlog := Real.log_le_log (mul_pos hp (by positivity : 0 < 1 + Real.exp (-B * W))) hfit
  rw [Real.log_mul hp.ne' (by positivity : 0 < 1 + Real.exp (-B * W)).ne'] at hlog
  change Real.log (1 + Real.exp (-B * W)) ≤ -Real.log (p a / ∑ j, p j)
  rw [Real.log_div hp.ne' hsum.ne']
  linarith

/-- Every native iterate has a uniform positive loss floor. The same
bound holds on training positions and held-out positions for any task. -/
theorem nativeTrainingRun_loss_floor [Fintype A] [Nontrivial A] (repeats budget : ℕ)
    (batches : ℕ → List (LabeledPosition A)) (draw : ComplexParameter A H d layers → ℂ) :
    let fit := nativeTrainingRun repeats batches (wsdRate budget) draw
    ∃ δ : ℝ, 0 < δ ∧ ∀ (step : ℕ) (word : Word A T) (i : Fin T) (a : A),
      δ ≤ -Real.log ((Model.fromWeights (fit step).weights).probability repeats word i a) := by
  dsimp only
  let fit := nativeTrainingRun repeats batches (wsdRate budget) draw
  let C := ‖initialWeights draw‖ + 10000000000
  have hstep (step : ℕ) : 0 ≤ wsdRate budget step ∧ wsdRate budget step ≤ 10 :=
    ⟨(wsdRate_bounds budget step).1, (wsdRate_bounds budget step).2.trans (by norm_num)⟩
  have hw (step : ℕ) (p : Parameter A H d layers) : |(fit step).weights p| ≤ C := by
    apply (Kernite.run_abs_weight_le _ _ _ hstep p step).trans
    exact max_le ((norm_le_pi_norm (initialWeights draw) p).trans
      (by dsimp only [C]; linarith)) (le_add_of_nonneg_left (norm_nonneg _))
  let E := Real.sqrt ((H * d : ℕ) * (2 * C) ^ 2)
  have hE (step : ℕ) (a : A) : ‖(Model.fromWeights (fit step).weights).embedding a‖ ≤ E := by
    have he (j : Fin (H * d)) : ‖(Model.fromWeights (fit step).weights).embedding a j‖ ≤ 2 * C :=
      (Complex.norm_le_abs_re_add_abs_im _).trans (by
        change |(fit step).weights (.inl (.embedding a j, false))| +
          |(fit step).weights (.inl (.embedding a j, true))| ≤ 2 * C
        linarith [hw step (.inl (.embedding a j, false)), hw step (.inl (.embedding a j, true))])
    rw [PiLp.norm_eq_of_L2]
    apply Real.sqrt_le_sqrt
    calc
      _ ≤ ∑ _ : Fin (H * d), (2 * C) ^ 2 :=
        Finset.sum_le_sum fun j _ => pow_le_pow_left₀ (norm_nonneg _) (he j) _
      _ = _ := by simp
  refine ⟨Real.log (1 + Real.exp (-E * Real.sqrt (H * d : ℕ))),
    Real.log_pos (lt_add_of_pos_right 1 (Real.exp_pos _)), ?_⟩
  intro step word i a
  exact Model.lossGe_of_embeddingBound _ (by norm_num [Model.fromWeights]) (hE step) repeats word i a

example := wsdRate_eq_floor (budget := 20000) (step := 20000) (by decide) (by decide) (by decide)

example (batches : ℕ → List (LabeledPosition (Fin 256))) (draw : ComplexParameter (Fin 256) 4 16 1 → ℂ) :=
  nativeTrainingRun_loss_floor (T := 32) 1 20000 batches draw

end Transformer.Grokking.FiniteRecovery.Argand
