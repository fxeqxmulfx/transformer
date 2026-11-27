module

public import Transformer.Grokking.FiniteRecovery.ArgandBinaryInitialization
public import Transformer.Grokking.FiniteRecovery.ArgandCausality
public import Transformer.Grokking.FiniteRecovery.ArgandEnergyRecovery

/-! A binary recovery guarantee with two observations and a positive
probability of a strict limit under the full clipped momentum update.
The observed tokens exhaust the contexts at the selected readout. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open Filter MeasureTheory Transformer.Optimization
open scoped Topology BigOperators

variable {T H d layers : ℕ}

private theorem binary_labelLoss (model : Model Bool H d layers) (repeats : ℕ)
    (word : Word Bool (T + 1)) (i : Fin (T + 1)) (a : Bool) :
    -Real.log (model.probability repeats word i a) =
      Real.log (1 + Real.exp (model.logits repeats word i (!a) - model.logits repeats word i a)) := by
  have hratio (x y : ℝ) : Real.exp x / (Real.exp x + Real.exp y) =
      (1 + Real.exp (y - x))⁻¹ := by
    rw [Real.exp_sub]
    field_simp [Real.exp_ne_zero, ne_of_gt (by positivity : 0 < Real.exp x + Real.exp y)]
  have hprob : model.probability repeats word i a =
      (1 + Real.exp (model.logits repeats word i (!a) - model.logits repeats word i a))⁻¹ := by
    dsimp only [Model.probability]
    rw [Fintype.sum_bool]
    cases a
    · simpa only [Bool.not_false, add_comm (Real.exp (model.logits repeats word i false))]
        using hratio (model.logits repeats word i false) (model.logits repeats word i true)
    · exact hratio _ _
  rw [hprob, Real.log_inv, neg_neg]

/-- At the readout position, observe one constant word for each token. -/
def binaryPositionBatch (i : Fin (T + 1)) (mode : Option Bool) : List (LabeledPosition Bool) :=
  [⟨T, fun _ => false, i, mode.getD false⟩, ⟨T, fun _ => true, i, mode.getD true⟩]

private noncomputable def siteLoss (model : Model Bool H d layers) (repeats : ℕ)
    (p : LabeledPosition Bool) : ℝ :=
  -Real.log (model.probability repeats p.2.1 p.2.2.1 p.2.2.2)

private theorem siteLoss_nonneg (model : Model Bool H d layers) (repeats : ℕ) (p : LabeledPosition Bool) :
    0 ≤ siteLoss model repeats p := by
  unfold siteLoss
  rw [binary_labelLoss]
  exact Real.log_nonneg (le_add_of_nonneg_right (Real.exp_pos _).le)

/-- If the target logit does not beat its competitor, that observation
costs at least log two. Its loss is bounded by the total batch loss. -/
theorem strict_binary_site_of_totalLoss (repeats : ℕ) (batch : List (LabeledPosition Bool))
    (w : Weights Bool H d layers) (ht : batch.length * batchLoss repeats batch w < Real.log 2)
    {p : LabeledPosition Bool} (hp : p ∈ batch) :
    let z := (Model.fromWeights w).logits repeats p.2.1 p.2.2.1
    z (!p.2.2.2) < z p.2.2.2 := by
  let model := Model.fromWeights w
  let losses := batch.map (siteLoss model repeats)
  have hlen : (batch.length : ℝ) ≠ 0 :=
    Nat.cast_ne_zero.mpr (List.length_pos_of_mem hp).ne'
  change (batch.length : ℝ) * (losses.sum / batch.length) < _ at ht
  rw [mul_div_cancel₀ _ hlen] at ht
  have hsmall : siteLoss model repeats p < Real.log 2 :=
    (List.single_le_sum (l := losses) (by
      intro a ha
      obtain ⟨q, _, rfl⟩ := List.mem_map.mp ha
      exact siteLoss_nonneg _ _ q) _ (List.mem_map_of_mem hp)).trans_lt ht
  dsimp only
  by_contra! h
  unfold siteLoss at hsmall
  rw [binary_labelLoss] at hsmall
  apply not_le_of_gt hsmall
  apply Real.log_le_log (by norm_num)
  have he := Real.one_le_exp (sub_nonneg.mpr h)
  change _ ≤ 1 + Real.exp ((Model.fromWeights w).logits repeats p.2.1 p.2.2.1 (!p.2.2.2) -
    (Model.fromWeights w).logits repeats p.2.1 p.2.2.1 p.2.2.2)
  linarith

/-- The fixed inputs admit arbitrary strict margins before the optimizer
runs. This property concerns only the finite observed batch. -/
def NativeBinaryMargins (H d layers repeats : ℕ) (batch : List (LabeledPosition Bool)) : Prop :=
  ∀ M : ℝ, 0 < M → ∃ draw : ComplexParameter Bool H d layers → ℂ, ∀ p ∈ batch,
    let z := (Model.fromWeights (sgdInitialWeights true draw)).logits repeats p.2.1 p.2.2.1
    M ≤ z p.2.2.2 - z (!p.2.2.2)

theorem nativeBinaryMargins_of_token_labels (layers repeats : ℕ) (batch : List (LabeledPosition Bool))
    (mode : Option Bool) (hlabels : ∀ p ∈ batch, p.2.2.2 = mode.getD (p.2.1 p.2.2.1)) :
    NativeBinaryMargins 1 1 layers repeats batch := by
  intro M hM
  let r := 4 * (M + 1)
  have hr : 1 ≤ r := by dsimp [r]; linarith
  refine ⟨binaryWitnessDraw r mode, fun p hp => ?_⟩
  dsimp only
  rw [hlabels p hp]
  exact (show M ≤ r / 4 by dsimp [r]; linarith).trans
    (binaryWitness_gap hr mode repeats p.2.1 p.2.2.1)

private theorem exists_draw_small_binary_loss (repeats : ℕ) (batch : List (LabeledPosition Bool))
    (hfit : NativeBinaryMargins H d layers repeats batch)
    {δ : ℝ} (hδ : 0 < δ) :
    ∃ draw : ComplexParameter Bool H d layers → ℂ,
      batchLoss repeats batch (sgdInitialWeights true draw) ≤ δ := by
  obtain ⟨draw, hgaps⟩ := hfit (δ⁻¹) (inv_pos.mpr hδ)
  let model := Model.fromWeights (sgdInitialWeights true draw)
  let losses := batch.map (siteLoss model repeats)
  have hsite (p : LabeledPosition Bool) (hp : p ∈ batch) : siteLoss model repeats p ≤ δ := by
    let a := p.2.2.2
    let m := model.logits repeats p.2.1 p.2.2.1 a - model.logits repeats p.2.1 p.2.2.1 (!a)
    have hg : δ⁻¹ ≤ m := hgaps p hp
    have hbig : δ⁻¹ ≤ Real.exp m := by
      linarith [Real.add_one_le_exp m]
    have he : Real.exp (-m) ≤ δ := by
      rw [Real.exp_neg]
      exact ((inv_le_inv₀ (Real.exp_pos _) (inv_pos.mpr hδ)).mpr hbig).trans_eq (inv_inv _)
    unfold siteLoss
    rw [binary_labelLoss]
    change Real.log (1 + Real.exp (_ - _)) ≤ δ
    rw [show model.logits repeats p.2.1 p.2.2.1 (!a) - model.logits repeats p.2.1 p.2.2.1 a =
      -m by dsimp only [m]; ring]
    have hl := Real.log_le_sub_one_of_pos (by positivity : 0 < 1 + Real.exp (-m))
    linarith
  refine ⟨draw, ?_⟩
  by_cases hb : batch = []
  · simp [hb, batchLoss, hδ.le]
  have hlen : 0 < (batch.length : ℝ) := by exact_mod_cast List.length_pos_iff.mpr hb
  change losses.sum / batch.length ≤ δ
  apply (div_le_iff₀ hlen).mpr
  have hs := losses.sum_le_length_nsmul δ (by
    intro a ha
    obtain ⟨q, hq, rfl⟩ := List.mem_map.mp ha
    exact hsite q hq)
  simpa [losses, nsmul_eq_mul, mul_comm δ] using hs

/-- Arbitrarily large native margins on the observed sites give a
positive-measure set of runs with arbitrarily small limiting batch loss
when the positive coupled decay is chosen sufficiently small. -/
theorem binaryBatch_momentum_lowLoss
    (mu : Measure (ComplexParameter Bool H d layers → ℂ)) [mu.IsOpenPosMeasure]
    (repeats : ℕ) (batch : List (LabeledPosition Bool)) (hfit : NativeBinaryMargins H d layers repeats batch)
    {δ : ℝ} (hδ : 0 < δ) :
    ∃ lam₀ > 0, ∀ lam, 0 < lam → lam ≤ lam₀ →
      ∃ U : Set (ComplexParameter Bool H d layers → ℂ), IsOpen U ∧ 0 < mu U ∧
        ∃ η₀ > 0, ∀ η, 0 < η → η ≤ η₀ → ∀ draw ∈ U,
          ∃ star : Weights Bool H d layers,
            Tendsto (fun t => (fullGradientMomentumRun true repeats batch lam (fun _ => η) draw t).weights)
              atTop (𝓝 star) ∧ batchLoss repeats batch star ≤ δ := by
  obtain ⟨draw, hce⟩ := exists_draw_small_binary_loss repeats batch hfit (half_pos hδ)
  let x : WeightSpace Bool H d layers := WithLp.toLp 2 (sgdInitialWeights true draw)
  refine ⟨(δ / 2) / (‖x‖ ^ 2 + 1), div_pos (half_pos hδ) (by positivity), fun lam hlam hlam₀ => ?_⟩
  have hreg := (le_div_iff₀ (by positivity : 0 < ‖x‖ ^ 2 + 1)).mp hlam₀
  have hx : fullGradientObjective repeats batch lam x < δ := by
    change batchLoss repeats batch (sgdInitialWeights true draw) + (lam / 2) * ‖x‖ ^ 2 < _
    nlinarith only [hce, hreg, hlam, hδ]
  obtain ⟨U, hU, _, hpos, η₀, hη₀, hr⟩ := fullGradientMomentumRun_posMeasure_of_sublevel
    mu true repeats batch hlam (fun _ => 0) 0 δ draw hx (fun w => batchLoss repeats batch w ≤ δ)
    (fun w hw => (le_add_of_nonneg_right (by positivity : 0 ≤ (lam / 2) * ‖w‖ ^ 2)).trans hw)
  exact ⟨U, hU, hpos, η₀, hη₀, fun η hη hle draw hd => by
    simpa only [Nat.not_lt_zero, ite_false] using hr η hη hle draw hd⟩

/-- This guarantee uses only observed positions. A prediction on a
hidden word can be transferred from an observed site when the model
has identical logits there for every weight vector. -/
theorem binaryBatch_momentum_posMeasure
    (mu : Measure (ComplexParameter Bool H d layers → ℂ)) [mu.IsOpenPosMeasure]
    (repeats : ℕ) (batch : List (LabeledPosition Bool)) (hfit : NativeBinaryMargins H d layers repeats batch)
    (i : Fin (T + 1)) (target : Word Bool (T + 1) → Bool)
    (hcover : ∀ word, ∃ p ∈ batch, p.2.2.2 = target word ∧
      ∀ (w : Weights Bool H d layers) a,
        (Model.fromWeights w).logits repeats word i a =
        (Model.fromWeights w).logits repeats p.2.1 p.2.2.1 a) :
    ∃ lam₀ > 0, ∀ lam, 0 < lam → lam ≤ lam₀ →
      ∃ U : Set (ComplexParameter Bool H d layers → ℂ), IsOpen U ∧ 0 < mu U ∧
        ∃ η₀ > 0, ∀ η, 0 < η → η ≤ η₀ → ∀ draw ∈ U,
          ∃ star : Weights Bool H d layers,
            Tendsto (fun t => (fullGradientMomentumRun true repeats batch lam (fun _ => η) draw t).weights)
              atTop (𝓝 star) ∧ ∀ word,
              (Model.fromWeights star).logits repeats word i (!(target word)) <
              (Model.fromWeights star).logits repeats word i (target word) := by
  let n : ℝ := batch.length
  have hn : 0 ≤ n := Nat.cast_nonneg _
  let δ := Real.log 2 / (4 * (n + 1))
  have hden : 0 < 4 * (n + 1) := by linarith
  have hδ : 0 < δ := div_pos (Real.log_pos (by norm_num)) hden
  have htotal : n * (2 * δ) < Real.log 2 := by
    have he := (div_mul_cancel₀ (Real.log 2) hden.ne' : δ * (4 * (n + 1)) = Real.log 2)
    nlinarith only [he, hδ, mul_nonneg hn hδ.le]
  obtain ⟨lam₀, hlam₀, hr⟩ := binaryBatch_momentum_lowLoss mu repeats batch hfit (show 0 < 2 * δ by positivity)
  refine ⟨lam₀, hlam₀, fun lam hlam hle => ?_⟩
  obtain ⟨U, hU, hp, η₀, hη₀, ht⟩ := hr lam hlam hle
  refine ⟨U, hU, hp, η₀, hη₀, fun η hη hηle draw hd => ?_⟩
  obtain ⟨w, hw, hb⟩ := ht η hη hηle draw hd
  refine ⟨w, hw, fun word => ?_⟩
  obtain ⟨p, hp, hlabel, heq⟩ := hcover word
  have h := strict_binary_site_of_totalLoss repeats batch w
    ((mul_le_mul_of_nonneg_left hb hn).trans_lt htotal) hp
  simpa only [heq _ _, ← hlabel] using h

end Transformer.Grokking.FiniteRecovery.Argand
