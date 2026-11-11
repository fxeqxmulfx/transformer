module

public import Transformer.Grokking.FlatSpectrum.Training.Scope.MarginContinuity
public import Transformer.Grokking.FlatSpectrum.Training.Scope.SignRun

/-!
# Loss, margin and direction of the run without weight decay on the sign sample

Source: `review/flat_spectrum.md`, *Training with cross-entropy* (added), item XE19; the item
"Not covered" of XE17.

For the sample of XE3 and XE17, without weight decay, for every step `η > 0` and every start
with `U₀ + V₀ ≠ 0`, with `x_t = U_t V_tᴴ`:

* the margin `2 Re x_t` tends to infinity (`tendsto_margin_signProduct_atTop`) and the
  cross-entropy tends to `0` (`tendsto_ceLoss_signProduct`);
* the margin for the weight norm, `μ(x_t) / (½ (‖U_t‖² + ‖V_t‖²))`, tends to `2`
  (`tendsto_margin_div_frobeniusSq_signProduct`), and so does the margin for the nuclear norm,
  `μ(x_t) / ‖x_t‖_S1 → μ* = 2` (`tendsto_margin_div_signProduct`): the largest margin of XE6;
* the direction `x_t / ‖x_t‖_S1` tends to `1`, the product of largest margin
  (`tendsto_inv_smul_signProduct`).

From a start with `U₀ + V₀ = 0` the sum stays `0` and the loss is at least `log 2` at every
step, for every real `η` (`log_two_le_ceLoss_signProduct`). So the loss tends to `0` exactly
from the starts with `U₀ + V₀ ≠ 0` (`tendsto_ceLoss_signProduct_iff`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training.Scope

open Matrix Filter Topology

variable {k : Type*} [Fintype k] {η : ℝ} {U₀ V₀ : Matrix (Fin 1) k ℂ}

local notation "R" => decayRun (ceGrad signProbe fun _ => true)

/-- **The margin of the product tends to infinity** (`review/flat_spectrum.md`, item XE19,
statement 2). -/
theorem tendsto_margin_signProduct_atTop (hη : 0 < η) (h0 : U₀ + V₀ ≠ 0) :
    Tendsto (fun t => margin signProbe (fun _ => true) (signProduct η U₀ V₀ t)) atTop atTop :=
  ((tendsto_modes_decayRun_signProbe hη h0).2.1.const_mul_atTop
    (by norm_num : (0 : ℝ) < 1 / 2)).congr fun t => by
      rw [margin_signProduct]
      ring

/-- **Without weight decay the cross-entropy tends to zero** (`review/flat_spectrum.md`, item
XE19, statement 2): on the sign sample, for every step `η > 0` and every start with
`U₀ + V₀ ≠ 0`. -/
theorem tendsto_ceLoss_signProduct (hη : 0 < η) (h0 : U₀ + V₀ ≠ 0) :
    Tendsto (fun t => ceLoss signProbe (fun _ => true) (signProduct η U₀ V₀ t)) atTop
      (𝓝 0) := by
  refine squeeze_zero (fun t => ceLoss_nonneg _ _ _) (fun t => ?_)
    (Real.tendsto_exp_neg_atTop_nhds_zero.comp (tendsto_margin_signProduct_atTop hη h0))
  have h := ceLoss_le_mul_exp (A := signProbe) (y := fun _ => true)
    (X := signProduct η U₀ V₀ t) fun s c hc => margin_le_gap _ hc
  norm_num at h
  exact h

/-- **The margin for the weight norm tends to `2`** (`review/flat_spectrum.md`, item XE19,
statement 3): `μ(x_t) / (½ (‖U_t‖² + ‖V_t‖²)) → 2`, so the weight norm of the factors is
asymptotically the nuclear norm of their product. -/
theorem tendsto_margin_div_frobeniusSq_signProduct (hη : 0 < η) (h0 : U₀ + V₀ ≠ 0) :
    Tendsto (fun t => margin signProbe (fun _ => true) (signProduct η U₀ V₀ t) /
      ((frobeniusSq (R η 0 U₀ V₀ t).1 + frobeniusSq (R η 0 U₀ V₀ t).2) / 2)) atTop (𝓝 2) := by
  obtain ⟨ha, -, T, hT⟩ := tendsto_modes_decayRun_signProbe hη h0
  have hlow : Tendsto (fun t => 2 - 4 * frobeniusSq ((R η 0 U₀ V₀ T).1 - (R η 0 U₀ V₀ T).2) /
      frobeniusSq ((R η 0 U₀ V₀ t).1 + (R η 0 U₀ V₀ t).2)) atTop (𝓝 2) := by
    simpa using (tendsto_const_nhds.div_atTop ha).const_sub 2
  have key : ∀ᶠ t in atTop,
      2 - 4 * frobeniusSq ((R η 0 U₀ V₀ T).1 - (R η 0 U₀ V₀ T).2) /
          frobeniusSq ((R η 0 U₀ V₀ t).1 + (R η 0 U₀ V₀ t).2) ≤
        margin signProbe (fun _ => true) (signProduct η U₀ V₀ t) /
          ((frobeniusSq (R η 0 U₀ V₀ t).1 + frobeniusSq (R η 0 U₀ V₀ t).2) / 2) ∧
      margin signProbe (fun _ => true) (signProduct η U₀ V₀ t) /
          ((frobeniusSq (R η 0 U₀ V₀ t).1 + frobeniusSq (R η 0 U₀ V₀ t).2) / 2) ≤ 2 := by
    filter_upwards [eventually_ge_atTop T, ha.eventually_gt_atTop 0] with t ht hat
    have hbt := frobeniusSq_nonneg ((R η 0 U₀ V₀ t).1 - (R η 0 U₀ V₀ t).2)
    have hbT := hT t ht
    rw [margin_signProduct, frobeniusSq_add_frobeniusSq]
    generalize frobeniusSq ((R η 0 U₀ V₀ t).1 + (R η 0 U₀ V₀ t).2) = a at hat ⊢
    generalize frobeniusSq ((R η 0 U₀ V₀ t).1 - (R η 0 U₀ V₀ t).2) = b at hbt hbT ⊢
    have h1 := div_le_div₀ (hbt.trans hbT) hbT hat (by linarith : a ≤ a + b)
    have h2 : (a - b) / 2 / ((a + b) / 2 / 2) = 2 - 4 * (b / (a + b)) := by
      field_simp
      ring
    have h3 : 0 ≤ b / (a + b) := div_nonneg hbt (by linarith)
    rw [h2, mul_div_assoc]
    exact ⟨by linarith, by linarith⟩
  exact tendsto_of_tendsto_of_tendsto_of_le_of_le' hlow tendsto_const_nhds
    (key.mono fun _ h => h.1) (key.mono fun _ h => h.2)

/-- **Without weight decay the margin for the nuclear norm tends to the largest margin**
(`review/flat_spectrum.md`, item XE19, statement 3): on the sign sample, for every step
`η > 0` and every start with `U₀ + V₀ ≠ 0`, `μ(x_t) / ‖x_t‖_S1 → μ*`, the limit of XE6. -/
theorem tendsto_margin_div_signProduct (hη : 0 < η) (h0 : U₀ + V₀ ≠ 0) :
    Tendsto (fun t => margin signProbe (fun _ => true) (signProduct η U₀ V₀ t) /
      nuclearNorm (signProduct η U₀ V₀ t)) atTop (𝓝 (maxMargin signProbe fun _ => true)) := by
  rw [maxMargin_signProbe]
  refine tendsto_of_tendsto_of_tendsto_of_le_of_le'
    (tendsto_margin_div_frobeniusSq_signProduct hη h0) tendsto_const_nhds ?_
    (Eventually.of_forall fun t => (margin_div_le_maxMargin _).trans_eq maxMargin_signProbe)
  filter_upwards [(tendsto_margin_signProduct_atTop hη h0).eventually_gt_atTop 0] with t ht
  exact div_le_div_of_nonneg_left ht.le (nuclearNorm_pos_of_margin_pos ht)
    (nuclearNorm_le_of_eq_mul _ _ rfl)

/-- **Without weight decay the direction of the product tends to the product of largest
margin** (`review/flat_spectrum.md`, item XE19, statement 3; XE10 along one run):
`x_t / ‖x_t‖_S1 → 1`. -/
theorem tendsto_inv_smul_signProduct (hη : 0 < η) (h0 : U₀ + V₀ ≠ 0) :
    Tendsto (fun t => (nuclearNorm (signProduct η U₀ V₀ t))⁻¹ • signProduct η U₀ V₀ t) atTop
      (𝓝 1) :=
  tendsto_inv_smul_of_tendsto_margin_div (tendsto_margin_div_signProduct hη h0)
    fun _ => eq_one_of_margin_signProbe

/-- **The exceptional starts** (`review/flat_spectrum.md`, item XE19, statement 4): from a
start with `U₀ + V₀ = 0` the sum of the factors stays `0` and the cross-entropy is at least
`log 2`, its value at the product `0`, at every step and for every real step size. -/
theorem log_two_le_ceLoss_signProduct (η : ℝ) (h0 : U₀ + V₀ = 0) (t : ℕ) :
    (R η 0 U₀ V₀ t).1 + (R η 0 U₀ V₀ t).2 = 0 ∧
      Real.log 2 ≤ ceLoss signProbe (fun _ => true) (signProduct η U₀ V₀ t) := by
  have hsum : ∀ t, (R η 0 U₀ V₀ t).1 + (R η 0 U₀ V₀ t).2 = 0 := fun t => by
    induction t with
    | zero => exact h0
    | succ t ih => rw [decayRun_signProbe_add, ih, smul_zero]
  refine ⟨hsum t, ?_⟩
  have hm : margin signProbe (fun _ => true) (signProduct η U₀ V₀ t) ≤ 0 := by
    have h1 := margin_signProduct η U₀ V₀ t
    have h2 := frobeniusSq_nonneg ((R η 0 U₀ V₀ t).1 - (R η 0 U₀ V₀ t).2)
    have h3 : frobeniusSq (0 : Matrix (Fin 1) k ℂ) = 0 := by simp [frobeniusSq]
    rw [hsum t, h3] at h1
    linarith
  have hg : gap signProbe (fun _ => true) (signProduct η U₀ V₀ t) 0 false ≤ 0 := by
    rw [gap_signProbe]
    have h1 := margin_signProbe (signProduct η U₀ V₀ t)
    have h2 : pairing 1 (signProduct η U₀ V₀ t) = ((signProduct η U₀ V₀ t) 0 0).re := by
      simp [pairing, Matrix.trace_fin_one]
    linarith
  refine le_trans (Real.log_le_log (by norm_num) ?_)
    (log_le_ceLoss signProbe (fun _ => true) (signProduct η U₀ V₀ t) (s := 0) (c := false)
      (by simp))
  linarith [Real.one_le_exp (neg_nonneg.2 hg)]

/-- **Without weight decay the loss tends to zero exactly from the starts with `U₀ + V₀ ≠ 0`**
(`review/flat_spectrum.md`, item XE19, statement 4), for every step `η > 0`. -/
theorem tendsto_ceLoss_signProduct_iff (hη : 0 < η) :
    Tendsto (fun t => ceLoss signProbe (fun _ => true) (signProduct η U₀ V₀ t)) atTop (𝓝 0) ↔
      U₀ + V₀ ≠ 0 := by
  refine ⟨fun h h0 => ?_, tendsto_ceLoss_signProduct hη⟩
  have h1 := ge_of_tendsto' h fun t => (log_two_le_ceLoss_signProduct η h0 t).2
  linarith [Real.log_pos (by norm_num : (1 : ℝ) < 2)]

/-! The witnesses: the start `U₀ = [1 0]`, `V₀ = [0 1]` of the example of XE17 with the step
`1/10`; and the start `u₀ = 1`, `v₀ = −1` of width one, which is exceptional. -/

example := tendsto_margin_signProduct_atTop (η := 1 / 10) (by norm_num)
  fromCols_one_zero_add_ne_zero

example := tendsto_ceLoss_signProduct (η := 1 / 10) (by norm_num) fromCols_one_zero_add_ne_zero

example := tendsto_margin_div_frobeniusSq_signProduct (η := 1 / 10) (by norm_num)
  fromCols_one_zero_add_ne_zero

example := tendsto_margin_div_signProduct (η := 1 / 10) (by norm_num)
  fromCols_one_zero_add_ne_zero

example := tendsto_inv_smul_signProduct (η := 1 / 10) (by norm_num)
  fromCols_one_zero_add_ne_zero

example := log_two_le_ceLoss_signProduct (1 / 10) (add_neg_cancel (1 : Matrix (Fin 1) (Fin 1) ℂ))
  3

example := tendsto_ceLoss_signProduct_iff (η := 1 / 10) (U₀ := (1 : Matrix (Fin 1) (Fin 1) ℂ))
  (V₀ := 1) (by norm_num)

end Transformer.Grokking.FlatSpectrum.Training.Scope
