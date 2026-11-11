module

public import Transformer.Grokking.FlatSpectrum.FlatKernel
public import Transformer.Grokking.FlatSpectrum.Training.CrossEntropyRun
public import Transformer.Grokking.FlatSpectrum.Training.Margin

/-!
# A trained fit against a competitor of unit nuclear norm

Source: `review/flat_spectrum.md`, *Training with cross-entropy* (added), item XE5.
A trained fit is a factorization of least weight-decayed cross-entropy among those of every
width (`IsDecayedMin`), where a run ends (item XE3). Write `X = U Vᴴ` and `B = ‖X‖_S1`. Let `X'`
have `‖X'‖_S1 ≤ 1` and every gap at least `μ'`. For `λ ≥ 0` and every `t ≥ 0`
(`IsDecayedMin.ceLoss_add_le`)

    L X + λ B ≤ L (t X') + λ t,

so the trained fit has no more loss than the competitor scaled to its own norm,
`L X ≤ L (B X')` (`IsDecayedMin.ceLoss_le_smul`). With the two bounds of item XE4, every gap
`g` of the trained fit has (`IsDecayedMin.log_le_mul_exp`)

    log (1 + exp (−g)) ≤ κ exp (−B μ'),    κ = |S| (|C| − 1),

and once `log (2κ) ≤ B μ'` every gap is at least `B μ' − log (2κ)`
(`IsDecayedMin.sub_log_le_gap`): the margin of the trained fit is within a constant of the
margin of the competitor at the same norm. The witness is `signProbe` with the competitor `1`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix

variable {m n k S C : Type*} [Fintype m] [Fintype n] [Fintype k] [Fintype S] [Fintype C]
  [DecidableEq n] {A : S → C → Matrix m n ℂ} {y : S → C} {lam μ' : ℝ} {U : Matrix m k ℂ}
  {V : Matrix n k ℂ} {X' : Matrix m n ℂ}

/-- **A trained fit against a scaled competitor** (`review/flat_spectrum.md`, item XE5): for
`λ ≥ 0`, `‖X'‖_S1 ≤ 1` and `t ≥ 0`, `L (U Vᴴ) + λ ‖U Vᴴ‖_S1 ≤ L (t X') + λ t`. -/
theorem IsDecayedMin.ceLoss_add_le (hlam : 0 ≤ lam) (h : IsDecayedMin (ceLoss A y) lam U V)
    (hX' : nuclearNorm X' ≤ 1) {t : ℝ} (ht : 0 ≤ t) :
    ceLoss A y (U * Vᴴ) + lam * nuclearNorm (U * Vᴴ) ≤ ceLoss A y (t • X') + lam * t := by
  have h1 := h.add_nuclearNorm_le hlam (t • X')
  rw [nuclearNorm_real_smul ht] at h1
  have h2 := mul_le_mul_of_nonneg_left hX' (mul_nonneg hlam ht)
  linarith

/-- **A trained fit has no more loss than the competitor at its own norm**
(`review/flat_spectrum.md`, item XE5): `L (U Vᴴ) ≤ L (‖U Vᴴ‖_S1 X')` for `λ ≥ 0` and
`‖X'‖_S1 ≤ 1`. -/
theorem IsDecayedMin.ceLoss_le_smul (hlam : 0 ≤ lam) (h : IsDecayedMin (ceLoss A y) lam U V)
    (hX' : nuclearNorm X' ≤ 1) :
    ceLoss A y (U * Vᴴ) ≤ ceLoss A y (nuclearNorm (U * Vᴴ) • X') := by
  have h1 := h.ceLoss_add_le hlam hX' (nuclearNorm_nonneg (U * Vᴴ))
  linarith

/-- **The gaps of a trained fit against the margin of a competitor**
(`review/flat_spectrum.md`, item XE5): if `‖X'‖_S1 ≤ 1` and every gap of `X'` is at least
`μ'`, every gap `g` of a trained fit has
`log (1 + exp (−g)) ≤ |S| (|C| − 1) exp (−‖U Vᴴ‖_S1 μ')`. -/
theorem IsDecayedMin.log_le_mul_exp (hlam : 0 ≤ lam) (h : IsDecayedMin (ceLoss A y) lam U V)
    (hX' : nuclearNorm X' ≤ 1) (hμ' : ∀ s c, c ≠ y s → μ' ≤ gap A y X' s c) {s : S} {c : C}
    (hc : c ≠ y s) :
    Real.log (1 + Real.exp (-(gap A y (U * Vᴴ) s c))) ≤
      Fintype.card S * (Fintype.card C - 1) * Real.exp (-(nuclearNorm (U * Vᴴ) * μ')) :=
  (log_le_ceLoss A y _ hc).trans ((h.ceLoss_le_smul hlam hX').trans
    (ceLoss_le_mul_exp fun s c hc => by
      rw [gap_smul]
      exact mul_le_mul_of_nonneg_left (hμ' s c hc) (nuclearNorm_nonneg _)))

/-- **A trained fit has the margin of the competitor up to a constant**
(`review/flat_spectrum.md`, item XE5): with `κ = |S| (|C| − 1)`, `‖X'‖_S1 ≤ 1` and every gap of
`X'` at least `μ'`, once `log (2κ) ≤ ‖U Vᴴ‖_S1 μ'` every gap of a trained fit is at least
`‖U Vᴴ‖_S1 μ' − log (2κ)`. -/
theorem IsDecayedMin.sub_log_le_gap (hlam : 0 ≤ lam) (h : IsDecayedMin (ceLoss A y) lam U V)
    (hX' : nuclearNorm X' ≤ 1) (hμ' : ∀ s c, c ≠ y s → μ' ≤ gap A y X' s c)
    (hB : Real.log (2 * (Fintype.card S * (Fintype.card C - 1))) ≤
      nuclearNorm (U * Vᴴ) * μ') {s : S} {c : C} (hc : c ≠ y s) :
    nuclearNorm (U * Vᴴ) * μ' - Real.log (2 * (Fintype.card S * (Fintype.card C - 1))) ≤
      gap A y (U * Vᴴ) s c := by
  have hS : (1 : ℝ) ≤ Fintype.card S := Nat.one_le_cast.mpr (Fintype.card_pos_iff.mpr ⟨s⟩)
  have hC : (2 : ℝ) ≤ Fintype.card C := by
    exact_mod_cast Fintype.one_lt_card_iff.mpr ⟨c, y s, hc⟩
  have hκ : (0 : ℝ) < Fintype.card S * (Fintype.card C - 1) :=
    mul_pos (by linarith) (by linarith)
  have hm := h.log_le_mul_exp hlam hX' hμ' hc
  generalize (Fintype.card S : ℝ) * (Fintype.card C - 1) = κ at hκ hm hB ⊢
  generalize nuclearNorm (U * Vᴴ) * μ' = b at hm hB ⊢
  generalize gap A y (U * Vᴴ) s c = g at hm ⊢
  have hb : Real.exp b * Real.exp (-b) = 1 := by
    rw [← Real.exp_add, add_neg_cancel, Real.exp_zero]
  have h2κ := (Real.log_le_iff_le_exp (by positivity)).mp hB
  have hε0 : 0 ≤ κ * Real.exp (-b) := by positivity
  have h2ε : 2 * (κ * Real.exp (-b)) ≤ 1 := by nlinarith [Real.exp_pos (-b)]
  have e1 := (Real.log_le_iff_le_exp (by positivity)).mp hm
  have e2 := Real.one_sub_le_exp_neg (κ * Real.exp (-b))
  have e3 : Real.exp (κ * Real.exp (-b)) * Real.exp (-(κ * Real.exp (-b))) = 1 := by
    rw [← Real.exp_add, add_neg_cancel, Real.exp_zero]
  have hE : Real.exp (-g) ≤ 2 * κ * Real.exp (-b) := by
    have a1 := mul_le_mul_of_nonneg_right e1 (by linarith : (0 : ℝ) ≤ 1 - κ * Real.exp (-b))
    have a2 := mul_le_mul_of_nonneg_left e2 (Real.exp_pos (κ * Real.exp (-b))).le
    have a3 := mul_nonneg (Real.exp_pos (-g)).le
      (by linarith : (0 : ℝ) ≤ 1 - 2 * (κ * Real.exp (-b)))
    nlinarith
  have h3 := Real.log_le_log (Real.exp_pos (-g)) hE
  rw [Real.log_exp, Real.log_mul (by positivity) (Real.exp_pos _).ne', Real.log_exp] at h3
  linarith

/-! The witness: one sample of a `1 × 1` product read with the code `±1` and labeled `true`,
`κ = 1`, against the competitor `X' = 1` with `‖X'‖_S1 = 1` and gap `2`. -/

lemma nuclearNorm_one_fin_one : nuclearNorm (1 : Matrix (Fin 1) (Fin 1) ℂ) = 1 := by
  simpa using (isFlat_one (n := Fin 1)).nuclearNorm_eq one_ne_zero

/-- A trained fit of the witness at width one and weight decay `λ > 0`. -/
lemma exists_isDecayedMin_signProbe {lam : ℝ} (hlam : 0 < lam) :
    ∃ U V : Matrix (Fin 1) (Fin 1) ℂ, IsDecayedMin (ceLoss signProbe fun _ => true) lam U V :=
  exists_isDecayedMin_ceLoss signProbe (fun _ => true) hlam (Or.inl le_rfl)

example : ∃ U V : Matrix (Fin 1) (Fin 1) ℂ,
    ceLoss signProbe (fun _ => true) (U * Vᴴ) + 1 * nuclearNorm (U * Vᴴ) ≤
      ceLoss signProbe (fun _ => true) ((3 : ℝ) • (1 : Matrix (Fin 1) (Fin 1) ℂ)) + 1 * 3 := by
  obtain ⟨U, V, h⟩ := exists_isDecayedMin_signProbe one_pos
  exact ⟨U, V, h.ceLoss_add_le zero_le_one nuclearNorm_one_fin_one.le (by norm_num)⟩

example : ∃ U V : Matrix (Fin 1) (Fin 1) ℂ, ceLoss signProbe (fun _ => true) (U * Vᴴ) ≤
    ceLoss signProbe (fun _ => true) (nuclearNorm (U * Vᴴ) • (1 : Matrix (Fin 1) (Fin 1) ℂ)) := by
  obtain ⟨U, V, h⟩ := exists_isDecayedMin_signProbe one_pos
  exact ⟨U, V, h.ceLoss_le_smul zero_le_one nuclearNorm_one_fin_one.le⟩

example : ∃ U V : Matrix (Fin 1) (Fin 1) ℂ,
    Real.log (1 + Real.exp (-(gap signProbe (fun _ => true) (U * Vᴴ) 0 false))) ≤
      (Fintype.card (Fin 1) : ℝ) * (Fintype.card Bool - 1) *
        Real.exp (-(nuclearNorm (U * Vᴴ) * 2)) := by
  obtain ⟨U, V, h⟩ := exists_isDecayedMin_signProbe one_pos
  exact ⟨U, V, h.log_le_mul_exp zero_le_one nuclearNorm_one_fin_one.le le_gap_signProbe_one
    (by simp)⟩

/-! The margin bound `IsDecayedMin.sub_log_le_gap` asks a trained fit whose norm has passed
`log (2κ) / μ'`. The witness has one for a small weight decay: the example is in
`MaxMarginLimit`, after the norms of its trained fits are shown to grow. -/

end Transformer.Grokking.FlatSpectrum.Training
