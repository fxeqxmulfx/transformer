module

public import Transformer.Grokking.Protocol.ReadoutWitness

/-!
# AdamW with weight decay 1 keeps the logits bounded

Source: arXiv:2201.02177v1 (Power et al.), appendix *Additional experimental details*: "For
most experiments we used AdamW optimizer with learning rate `10⁻³`, weight decay `1`"; openai/grok
at 3d64b1d8, `grok/training.py`, `CustomAdamW.step`, and PyTorch's default initialization of
`Linear` and `LayerNorm`; `experiments/grok_division/PROTOCOL.md`, *The optimizer*;
`review/grok_division_convergence.md`, bridge 4.

With weight decay `1` every parameter of the division run stays within `max(|θ₀|, 17)` of zero
(`abs_trajectory_division_le`). Almost every initialization has its readout and its second
normalizations within `1` of zero (`ae_initMeasure_abs_le_one`: the readout is uniform on
`[−1/√128, 1/√128]`, the gains start at `1` and the biases at `0`), so they stay within `17`
of zero, and by `abs_logits_le` every logit of the decoder, for all tokens read and at every
update, is at most

    logitBox = 128 · 17 · (17 √128 + 17) < 455 600

in absolute value: pathwise in `abs_logits_divisionRun_le`, for almost every sample of the
protocol in `ae_abs_logits_divisionRun_le`. Bounded logits keep the cross-entropy of a target
above `log(1 + e^{−2Z})` (`log_le_crossEntropy_of_abs_le`) and its margin below `2Z`
(`margin_le_of_abs_le`). So for almost every sample, at every update, the loss of every batch
with a target is above one positive number (`ae_floor_le_batchLoss_divisionRun`,
`floor_pos`) and no answer leads by more than `2 · logitBox` (`ae_margin_divisionRun_le`).

The bound is uniform in the update and far from what a run shows; it is what the box of the
parameters gives without looking at a gradient. It excludes two ways of proving a
convergence for the runs with weight decay: through a loss that tends to zero, and through
margins that grow without bound. It says nothing of where the predictions go. Without weight
decay no such box is proved: the bound of `abs_trajectory_division_sub_le` grows with the
update.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

open Filter Topology MeasureTheory ProbabilityTheory
open Transformer.Landscape.Neural (crossEntropy)

/-- **Bounded logits keep the cross-entropy above a floor**: with logits at most `Z` in
absolute value and another class, the loss of a target is at least `log(1 + e^{−2Z})`. -/
theorem log_le_crossEntropy_of_abs_le {V : Type*} [Fintype V] {z : V → ℝ} {y : V} {Z : ℝ}
    (hZ : ∀ c, |z c| ≤ Z) (hV : ∃ c, c ≠ y) :
    Real.log (1 + Real.exp (-(2 * Z))) ≤ crossEntropy z y := by
  classical
  obtain ⟨c₀, hc₀⟩ := hV
  have h1 : Real.exp (z y) + Real.exp (z c₀) ≤ ∑ q, Real.exp (z q) :=
    (Finset.sum_pair (f := fun q => Real.exp (z q)) hc₀.symm).symm.trans_le
      (Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ _) fun q _ _ =>
        (Real.exp_pos _).le)
  have h2 : Real.exp (z y) * Real.exp (-(2 * Z)) ≤ Real.exp (z c₀) := by
    rw [← Real.exp_add]
    exact Real.exp_le_exp.2 (by linarith [le_of_abs_le (hZ y), neg_le_of_abs_le (hZ c₀)])
  have h3 : Real.log (Real.exp (z y) * (1 + Real.exp (-(2 * Z)))) ≤
      Real.log (∑ q, Real.exp (z q)) :=
    Real.log_le_log (by positivity) (by rw [mul_add, mul_one]; linarith)
  rw [Real.log_mul (Real.exp_pos _).ne' (by positivity), Real.log_exp] at h3
  unfold crossEntropy
  linarith

/-- **Bounded logits bound the margin**: it is at most `2Z`. -/
theorem margin_le_of_abs_le {V : Type*} [Fintype V] {z : V → ℝ} {y : V} {Z : ℝ}
    (hZ : ∀ c, |z c| ≤ Z) (hV : ∃ c, c ≠ y) : margin z y ≤ 2 * Z := by
  obtain ⟨c, hc⟩ := hV
  linarith [margin_le z hc, le_of_abs_le (hZ y), neg_le_of_abs_le (hZ c)]

/-- The bound of the logits of the division decoder whose readout and second normalizations
are within `17` of zero. -/
noncomputable def logitBox : ℝ := 128 * (17 * (√128 * 17 + 17))

lemma logitBox_lt : logitBox < 455600 := by
  have h : √(128 : ℝ) < 11.314 := (Real.sqrt_lt' (by norm_num)).2 (by norm_num)
  unfold logitBox
  linarith

/-- The floor of the loss is positive. -/
lemma floor_pos : 0 < Real.log (1 + Real.exp (-(2 * logitBox))) :=
  Real.log_pos (by linarith [Real.exp_pos (-(2 * logitBox))])

/-- **Under AdamW with weight decay 1 the logits stay bounded along the whole run**, on a
sample whose initial readout and second normalizations are within `17` of zero. -/
theorem abs_logits_divisionRun_le (ω : Sample (97 * 96) 4656 (Param paperSizes))
    (hout : ∀ c i, |ω.2.1 (.out c i)| ≤ 17) (hgain : ∀ l i, |ω.2.1 (.gain2 l i)| ≤ 17)
    (hbias : ∀ l i, |ω.2.1 (.bias2 l i)| ≤ 17) (t : ℕ) (act : Activation paperSizes)
    (tokens : Fin 6 → Fin 239) (pos : Fin 6) (c : Fin 239) :
    |logits (divisionRun 1 ω t) act tokens pos c| ≤ logitBox := by
  have hp : ∀ p, |ω.2.1 p| ≤ 17 → |divisionRun 1 ω t p| ≤ 17 := fun p hp =>
    (abs_trajectory_division_le ω t p).trans (max_le hp le_rfl)
  have h := abs_logits_le (S := paperSizes) (by decide) (θ := divisionRun 1 ω t)
    (fun c i => hp _ (hout c i)) (fun l i => hp _ (hgain l i)) (fun l i => hp _ (hbias l i))
    act tokens pos c
  have hw : ((paperSizes.width : ℕ) : ℝ) = 128 := by norm_num
  rw [hw] at h
  exact h

/-- **Almost every initialization has its readout and its second normalizations within `1`
of zero.** -/
theorem ae_initMeasure_abs_le_one : ∀ᵐ θ ∂initMeasure paperSizes,
    (∀ c i, |θ (.out c i)| ≤ 1) ∧ (∀ l i, |θ (.gain2 l i)| ≤ 1) ∧
      ∀ l i, |θ (.bias2 l i)| ≤ 1 := by
  have hev : ∀ p : Param paperSizes, (∀ᵐ x ∂initLaw paperSizes p, |x| ≤ 1) →
      ∀ᵐ θ ∂initMeasure paperSizes, |θ p| ≤ 1 := fun p h =>
    (Measure.tendsto_eval_ae_ae (μ := initLaw paperSizes) (i := p)).eventually h
  have hu : ∀ᵐ x ∂uniformInit 128, |x| ≤ 1 := by
    have hs : (1 : ℝ) ≤ √(128 : ℕ) := Real.one_le_sqrt.2 (by norm_num)
    have h1 : 1 / √(128 : ℕ) ≤ 1 := (div_le_one (by positivity)).2 hs
    filter_upwards [ae_cond_mem (μ := volume)
      (measurableSet_Icc (a := -(1 / √(128 : ℕ))) (b := 1 / √(128 : ℕ)))] with x hx
    exact abs_le.2 ⟨by linarith [hx.1], by linarith [hx.2]⟩
  refine eventually_and.2 ⟨?_, eventually_and.2 ⟨?_, ?_⟩⟩
  · simp only [eventually_all]
    exact fun c i => hev (.out c i) hu
  · simp only [eventually_all]
    exact fun l i => hev (.gain2 l i) (by simp [initLaw])
  · simp only [eventually_all]
    exact fun l i => hev (.bias2 l i) (by simp [initLaw])

/-- What holds for almost every initialization holds for almost every sample. -/
lemma ae_law_init {ι : Type*} {N k : ℕ} {ν : Measure (ι → ℝ)} [SFinite ν]
    {P : (ι → ℝ) → Prop} (h : ∀ᵐ θ ∂ν, P θ) : ∀ᵐ ω ∂law N k ν, P ω.2.1 :=
  (Measure.quasiMeasurePreserving_fst.comp Measure.quasiMeasurePreserving_snd).ae h

/-- **Division under AdamW: for almost every sample the logits are bounded along the whole
run**, for all tokens read, all activations and every update. -/
theorem ae_abs_logits_divisionRun_le : ∀ᵐ ω ∂law (97 * 96) 4656 (initMeasure paperSizes),
    ∀ (t : ℕ) (act : Activation paperSizes) (tokens : Fin 6 → Fin 239) (pos : Fin 6)
      (c : Fin 239), |logits (divisionRun 1 ω t) act tokens pos c| ≤ logitBox := by
  filter_upwards [ae_law_init ae_initMeasure_abs_le_one] with ω hω t act tokens pos c
  exact abs_logits_divisionRun_le ω (fun c i => (hω.1 c i).trans (by norm_num))
    (fun l i => (hω.2.1 l i).trans (by norm_num)) (fun l i => (hω.2.2 l i).trans (by norm_num))
    t act tokens pos c

/-- **Division under AdamW: for almost every sample the loss of every batch with a target
stays above one positive number**, at every update. -/
theorem ae_floor_le_batchLoss_divisionRun :
    ∀ᵐ ω ∂law (97 * 96) 4656 (initMeasure paperSizes), ∀ (t : ℕ) (B : List (Example paperSizes)),
      0 < targetCount B →
        Real.log (1 + Real.exp (-(2 * logitBox))) ≤ batchLoss (divisionRun 1 ω t) B := by
  have hnt : Nontrivial (Fin paperSizes.vocab) := Fin.nontrivial_iff_two_le.2 (by decide)
  filter_upwards [ae_abs_logits_divisionRun_le] with ω hω t B hB
  rw [batchLoss_eq_targetMean]
  exact le_targetMean hB fun e _ pc _ =>
    log_le_crossEntropy_of_abs_le (fun c => hω t _ _ _ c) (exists_ne pc.2)

/-- **Division under AdamW: for almost every sample no answer ever leads by more than
`2 · logitBox`.** -/
theorem ae_margin_divisionRun_le : ∀ᵐ ω ∂law (97 * 96) 4656 (initMeasure paperSizes),
    ∀ (t : ℕ) (x : Equation),
      margin (answerLogits (divisionRun 1 ω t) x) (residueToken x.1) ≤ 2 * logitBox := by
  filter_upwards [ae_abs_logits_divisionRun_le] with ω hω t x
  exact margin_le_of_abs_le (fun c => hω t _ _ _ c)
    ⟨eosToken, (residueToken_ne_eosToken x.1).symm⟩

/-! The witnesses: two classes with the logits `2` and `0`; the sample `zeroSample`, whose
initial parameters are `0`; and the initialization law for `ae_law_init`. -/

example : Real.log (1 + Real.exp (-(2 * 2))) ≤ crossEntropy ![(2 : ℝ), 0] 0 :=
  log_le_crossEntropy_of_abs_le (fun c => by fin_cases c <;> simp) ⟨1, by decide⟩

example : margin ![(2 : ℝ), 0] 0 ≤ 2 * 2 :=
  margin_le_of_abs_le (fun c => by fin_cases c <;> simp) ⟨1, by decide⟩

example (t : ℕ) (x : Equation) (c : Fin 239) :
    |answerLogits (divisionRun 1 zeroSample t) x c| ≤ logitBox :=
  abs_logits_divisionRun_le zeroSample (by simp [zeroSample]) (by simp [zeroSample])
    (by simp [zeroSample]) t _ _ _ c

example : ∀ᵐ ω ∂law (97 * 96) 4656 (initMeasure paperSizes), ∀ c i, |ω.2.1 (.out c i)| ≤ 1 :=
  ae_law_init (ae_initMeasure_abs_le_one.mono fun _ h => h.1)

end Transformer.Grokking.Protocol
