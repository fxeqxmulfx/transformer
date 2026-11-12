module

public import Transformer.Grokking.FlatSpectrum.Training.Scope.SeparateCellsRows

/-!
# Samples of the code `±1` in distinct rows and columns: the run without weight decay

Source: `review/flat_spectrum.md`, *Training with cross-entropy* (added), item XE20,
statements 2 and 3; the items "Not covered" of XE17 and XE19.

Let no two samples lie in one row or in one column, and let `η > 0` be any step. The rows
`u = U_i`, `v = y V_j` of a sample follow the run on the sign sample (`SeparateCellsRows`), so
by XE19, without weight decay and at every width:

* if `U_i + y V_j ≠ 0` at the rows of some sample, the norm of the factors tends to infinity
  (`tendsto_frobeniusSq_decayRun_signCode_atTop`);
* the cross-entropy tends to `0` if and only if `U_i + y V_j ≠ 0` at the rows of every sample
  (`tendsto_ceLoss_decayRun_signCode_iff`), and then the margin tends to infinity
  (`tendsto_margin_decayRun_signCode_atTop`);
* a sample with `U_i + y V_j = 0` keeps the loss at least `log 2` at every step, for every
  real step size (`log_two_le_ceLoss_decayRun_signCode`).

The direction of the product and the margin for the nuclear norm are not treated: they depend
on how fast the modes of different samples grow.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training.Scope

open Matrix Filter Topology

variable {m n k S : Type*} [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n] [Fintype k]
  [Fintype S] {cell : S → m × n} {y : S → Bool} {η : ℝ}

local notation "RC" => decayRun (ceGrad (codeProbe signCode cell) y)

/-- **Without weight decay the norm of the factors tends to infinity**
(`review/flat_spectrum.md`, item XE20, statement 2): for samples of the code `±1`, no two in
one row or in one column, every step `η > 0` and every start in which the rows of some sample
have `U_i + y V_j ≠ 0`. -/
theorem tendsto_frobeniusSq_decayRun_signCode_atTop
    (hrow : Function.Injective fun s => (cell s).1)
    (hcol : Function.Injective fun s => (cell s).2) (hη : 0 < η) {U₀ : Matrix m k ℂ}
    {V₀ : Matrix n k ℂ} {s : S}
    (hs : ∃ b, U₀ (cell s).1 b + classSign (y s) • V₀ (cell s).2 b ≠ 0) :
    Tendsto (fun t => frobeniusSq (RC η 0 U₀ V₀ t).1 + frobeniusSq (RC η 0 U₀ V₀ t).2)
      atTop atTop := by
  refine tendsto_atTop_mono (fun t => ?_) (tendsto_frobeniusSq_decayRun_signProbe_atTop hη
    ((rowOf_add_ne_zero_iff _ _ _ U₀ V₀).2 hs))
  rw [rowOf_decayRun_signCode hrow hcol, frobeniusSq_smul, sq, classSign_mul_self, one_mul]
  exact add_le_add (frobeniusSq_rowOf_le _ _) (frobeniusSq_rowOf_le _ _)

/-- **A sample whose rows cancel keeps the loss above `log 2`** (`review/flat_spectrum.md`,
item XE20, statement 3), at every step and for every real step size. -/
theorem log_two_le_ceLoss_decayRun_signCode (hrow : Function.Injective fun s => (cell s).1)
    (hcol : Function.Injective fun s => (cell s).2) (η : ℝ) {U₀ : Matrix m k ℂ}
    {V₀ : Matrix n k ℂ} {s : S}
    (hs : ∀ b, U₀ (cell s).1 b + classSign (y s) • V₀ (cell s).2 b = 0) (t : ℕ) :
    Real.log 2 ≤ ceLoss (codeProbe signCode cell) y
      ((RC η 0 U₀ V₀ t).1 * (RC η 0 U₀ V₀ t).2ᴴ) := by
  have h0 : rowOf (cell s).1 U₀ + classSign (y s) • rowOf (cell s).2 V₀ = 0 := by
    by_contra hne
    obtain ⟨b, hb⟩ := (rowOf_add_ne_zero_iff _ _ _ U₀ V₀).1 hne
    exact hb (hs b)
  rw [ceLoss_decayRun_signCode hrow hcol]
  exact (log_two_le_ceLoss_signProduct η h0 t).2.trans
    (Finset.single_le_sum (f := fun s => ceLoss signProbe (fun _ => true)
      (signProduct η (rowOf (cell s).1 U₀) (classSign (y s) • rowOf (cell s).2 V₀) t))
      (fun s _ => ceLoss_nonneg _ _ _) (Finset.mem_univ s))

/-- **Without weight decay the loss tends to zero exactly when the rows of no sample cancel**
(`review/flat_spectrum.md`, item XE20, statement 3): for samples of the code `±1`, no two in
one row or in one column, and every step `η > 0`, the cross-entropy tends to `0` if and only
if `U_i + y V_j ≠ 0` at the rows of every sample. -/
theorem tendsto_ceLoss_decayRun_signCode_iff (hrow : Function.Injective fun s => (cell s).1)
    (hcol : Function.Injective fun s => (cell s).2) (hη : 0 < η) {U₀ : Matrix m k ℂ}
    {V₀ : Matrix n k ℂ} :
    Tendsto (fun t => ceLoss (codeProbe signCode cell) y
        ((RC η 0 U₀ V₀ t).1 * (RC η 0 U₀ V₀ t).2ᴴ)) atTop (𝓝 0) ↔
      ∀ s, ∃ b, U₀ (cell s).1 b + classSign (y s) • V₀ (cell s).2 b ≠ 0 := by
  constructor
  · intro h s
    by_contra hs
    have hall : ∀ b, U₀ (cell s).1 b + classSign (y s) • V₀ (cell s).2 b = 0 := fun b =>
      by_contra fun hb => hs ⟨b, hb⟩
    have h1 := ge_of_tendsto' h (log_two_le_ceLoss_decayRun_signCode hrow hcol η hall)
    linarith [Real.log_pos (by norm_num : (1 : ℝ) < 2)]
  · intro h
    have hlim := tendsto_finsetSum (Finset.univ : Finset S) fun s _ =>
      tendsto_ceLoss_signProduct hη ((rowOf_add_ne_zero_iff _ _ _ U₀ V₀).2 (h s))
    rw [Finset.sum_const_zero] at hlim
    exact hlim.congr fun t => (ceLoss_decayRun_signCode hrow hcol η U₀ V₀ t).symm

/-- **Without weight decay the margin tends to infinity** (`review/flat_spectrum.md`, item
XE20, statement 3): for samples of the code `±1`, no two in one row or in one column, every
step `η > 0` and every start with `U_i + y V_j ≠ 0` at the rows of every sample. -/
theorem tendsto_margin_decayRun_signCode_atTop [Nonempty S]
    (hrow : Function.Injective fun s => (cell s).1)
    (hcol : Function.Injective fun s => (cell s).2) (hη : 0 < η) {U₀ : Matrix m k ℂ}
    {V₀ : Matrix n k ℂ}
    (h : ∀ s, ∃ b, U₀ (cell s).1 b + classSign (y s) • V₀ (cell s).2 b ≠ 0) :
    Tendsto (fun t => margin (codeProbe signCode cell) y
      ((RC η 0 U₀ V₀ t).1 * (RC η 0 U₀ V₀ t).2ᴴ)) atTop atTop := by
  refine tendsto_atTop.2 fun B => ?_
  have hs : ∀ s, ∀ᶠ t in atTop, B ≤ margin signProbe (fun _ => true)
      (signProduct η (rowOf (cell s).1 U₀) (classSign (y s) • rowOf (cell s).2 V₀) t) :=
    fun s => tendsto_atTop.1 (tendsto_margin_signProduct_atTop hη
      ((rowOf_add_ne_zero_iff _ _ _ U₀ V₀).2 (h s))) B
  filter_upwards [eventually_all.2 hs] with t ht
  refine le_margin fun s c hc => (ht s).trans_eq ?_
  rw [margin_signProbe, signProduct, rowOf_decayRun_signCode hrow hcol, pairing_one_rowOf, gap,
    pairing_codeProbe_signCode, pairing_codeProbe_signCode]
  cases hy : y s <;> cases c <;> simp_all [classSign, signCode] <;> ring

/-! The witnesses: the two samples at the diagonal cells of a `2 × 2` product, with the step
`1/10`. From `U₀ = 1`, `V₀ = 0` with the labels `(+, −)` no rows cancel; from `U₀ = 1`,
`V₀ = −1` with the labels `(+, +)` the rows of both samples cancel. -/

/-- From `U₀ = 1`, `V₀ = 0` no rows cancel, for any labels. -/
lemma diagCell_one_add_zero_ne (y : Fin 2 → Bool) (s : Fin 2) :
    ∃ b, (1 : Matrix (Fin 2) (Fin 2) ℂ) (diagCell s).1 b +
      classSign (y s) • (0 : Matrix (Fin 2) (Fin 2) ℂ) (diagCell s).2 b ≠ 0 := by
  refine ⟨s, ?_⟩
  change (1 : Matrix (Fin 2) (Fin 2) ℂ) s s +
    classSign (y s) • (0 : Matrix (Fin 2) (Fin 2) ℂ) s s ≠ 0
  simp

/-- From `U₀ = 1`, `V₀ = −1` with the labels `+` the rows of every sample cancel. -/
lemma diagCell_one_add_neg_one_eq (s b : Fin 2) :
    (1 : Matrix (Fin 2) (Fin 2) ℂ) (diagCell s).1 b +
      classSign ((fun _ : Fin 2 => true) s) • (-1 : Matrix (Fin 2) (Fin 2) ℂ) (diagCell s).2 b
        = 0 := by
  change (1 : Matrix (Fin 2) (Fin 2) ℂ) s b +
    classSign true • (-1 : Matrix (Fin 2) (Fin 2) ℂ) s b = 0
  simp [classSign]

example := tendsto_frobeniusSq_decayRun_signCode_atTop (y := ![true, false])
  injective_diagCell_fst injective_diagCell_snd (η := 1 / 10) (by norm_num)
  (diagCell_one_add_zero_ne ![true, false] 0)

example := (tendsto_ceLoss_decayRun_signCode_iff (y := ![true, false]) injective_diagCell_fst
  injective_diagCell_snd (η := 1 / 10) (by norm_num)).2 (diagCell_one_add_zero_ne ![true, false])

example := tendsto_margin_decayRun_signCode_atTop (y := ![true, false]) injective_diagCell_fst
  injective_diagCell_snd (η := 1 / 10) (by norm_num) (diagCell_one_add_zero_ne ![true, false])

example := log_two_le_ceLoss_decayRun_signCode (y := fun _ => true) injective_diagCell_fst
  injective_diagCell_snd (1 / 10) (s := 0) (diagCell_one_add_neg_one_eq 0) 3

end Transformer.Grokking.FlatSpectrum.Training.Scope
