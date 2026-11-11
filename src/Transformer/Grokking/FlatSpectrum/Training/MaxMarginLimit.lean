module

public import Mathlib.Analysis.SpecificLimits.Basic
public import Transformer.Grokking.FlatSpectrum.Training.MaxMargin

/-!
# Vanishing weight decay under the cross-entropy: the norm grows and the margin is maximal

Source: `review/flat_spectrum.md`, *Training with cross-entropy* (added), item XE6.
Let the weight decays `λ_j > 0` tend to `0` and let `(U_j, V_j)` be trained fits of the
cross-entropy (`IsDecayedMin`), `X_j = U_j V_jᴴ`. If some product has a positive margin:

* `‖X_j‖_S1 → ∞` (`tendsto_nuclearNorm_atTop`): the loss of a trained fit is at most
  `κ exp (−t μ') + λ_j t` for every `t ≥ 0` (item XE5), below every positive level in the end,
  while a product of bounded norm has a gap bounded above and so a loss bounded below (XE4);
* the normalized margin `μ(X_j) / ‖X_j‖_S1` is in the end above every `θ ≥ 0` below the margin
  of a product of nuclear norm at most one (`eventually_lt_margin_div`), by item XE5;
* so it converges to the largest margin of the unit ball of the nuclear norm, `maxMargin`
  (`tendsto_margin_div_nuclearNorm`); it never exceeds it (`margin_div_le_maxMargin`).

In this limit the squared loss has the exact fit of least nuclear norm instead (item WD7).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix Filter Topology

variable {m n k S C : Type*} [Fintype m] [Fintype n] [Fintype k] [Fintype S] [Fintype C]
  [DecidableEq n] {A : S → C → Matrix m n ℂ} {y : S → C} {μ' : ℝ} {X' : Matrix m n ℂ}
  {lams : ℕ → ℝ} {Us : ℕ → Matrix m k ℂ} {Vs : ℕ → Matrix n k ℂ}

/-- A product of positive margin is not zero: `0 < ‖X‖_S1`. -/
lemma nuclearNorm_pos_of_margin_pos [Nonempty S] [Nontrivial C] {X : Matrix m n ℂ}
    (h : 0 < margin A y X) : 0 < nuclearNorm X := by
  obtain ⟨a, ha0, ha⟩ := exists_forall_frobeniusSq_le A
  obtain ⟨s, c, -, hg⟩ := exists_gap_eq_margin A y X
  have h1 := gap_le_mul_nuclearNorm (y := y) ha0 ha X s c
  rw [hg] at h1
  by_contra h0
  rw [le_antisymm (not_lt.mp h0) (nuclearNorm_nonneg X), mul_zero] at h1
  linarith

/-- **The norm of trained fits grows without bound as the weight decay vanishes**
(`review/flat_spectrum.md`, item XE6): if a product `X'` with `‖X'‖_S1 ≤ 1` has every gap at
least `μ' > 0` and `λ_j → 0⁺`, the trained fits of the cross-entropy have `‖X_j‖_S1 → ∞`. -/
theorem tendsto_nuclearNorm_atTop [Nonempty S] [Nontrivial C] (hX' : nuclearNorm X' ≤ 1)
    (hμ' : 0 < μ') (hgap : ∀ s c, c ≠ y s → μ' ≤ gap A y X' s c) (hpos : ∀ j, 0 < lams j)
    (hlam : Tendsto lams atTop (𝓝 0))
    (hmin : ∀ j, IsDecayedMin (ceLoss A y) (lams j) (Us j) (Vs j)) :
    Tendsto (fun j => nuclearNorm (Us j * (Vs j)ᴴ)) atTop atTop := by
  rw [tendsto_atTop]
  intro b
  obtain ⟨a, ha0, ha⟩ := exists_forall_frobeniusSq_le A
  obtain ⟨s, c, hc, -⟩ := exists_gap_eq_margin A y X'
  have hℓ : 0 < Real.log (1 + Real.exp (-(2 * a * b))) :=
    Real.log_pos (by linarith [Real.exp_pos (-(2 * a * b))])
  have h1 : Tendsto (fun t : ℝ => (Fintype.card S : ℝ) * (Fintype.card C - 1) *
      Real.exp (-(t * μ'))) atTop (𝓝 (Fintype.card S * (Fintype.card C - 1) * 0)) :=
    (Real.tendsto_exp_neg_atTop_nhds_zero.comp (tendsto_id.atTop_mul_const hμ')).const_mul _
  rw [mul_zero] at h1
  obtain ⟨t, ht, ht0⟩ :=
    ((h1.eventually_lt_const (half_pos hℓ)).and (eventually_ge_atTop 0)).exists
  have h2 : Tendsto (fun j => lams j * t) atTop (𝓝 (0 * t)) := hlam.mul_const t
  rw [zero_mul] at h2
  filter_upwards [h2.eventually_lt_const (half_pos hℓ)] with j hj
  by_contra hb
  rw [not_le] at hb
  have h3 := log_le_ceLoss A y (Us j * (Vs j)ᴴ) hc
  have h4 := (gap_le_mul_nuclearNorm (y := y) ha0 ha (Us j * (Vs j)ᴴ) s c).trans
    (mul_le_mul_of_nonneg_left hb.le (by positivity))
  have h5 : Real.log (1 + Real.exp (-(2 * a * b))) ≤
      Real.log (1 + Real.exp (-(gap A y (Us j * (Vs j)ᴴ) s c))) :=
    Real.log_le_log (by positivity) (by linarith [Real.exp_le_exp.mpr (neg_le_neg h4)])
  have h6 := (hmin j).ceLoss_add_le (hpos j).le hX' ht0
  have h7 := ceLoss_le_mul_exp (A := A) (y := y) (X := t • X') (μ := t * μ') fun s c hc => by
    rw [gap_smul]
    exact mul_le_mul_of_nonneg_left (hgap s c hc) ht0
  have h8 := mul_nonneg (hpos j).le (nuclearNorm_nonneg (Us j * (Vs j)ᴴ))
  linarith

/-- **The normalized margin of trained fits passes the margins of the unit ball**
(`review/flat_spectrum.md`, item XE6): if `‖X'‖_S1 ≤ 1`, `0 ≤ θ < μ(X')` and `λ_j → 0⁺`, the
trained fits of the cross-entropy have `θ < μ(X_j) / ‖X_j‖_S1` in the end. -/
theorem eventually_lt_margin_div [Nonempty S] [Nontrivial C] (hX' : nuclearNorm X' ≤ 1)
    {θ : ℝ} (hθ0 : 0 ≤ θ) (hθ : θ < margin A y X') (hpos : ∀ j, 0 < lams j)
    (hlam : Tendsto lams atTop (𝓝 0))
    (hmin : ∀ j, IsDecayedMin (ceLoss A y) (lams j) (Us j) (Vs j)) :
    ∀ᶠ j in atTop, θ < margin A y (Us j * (Vs j)ᴴ) / nuclearNorm (Us j * (Vs j)ᴴ) := by
  have hgap : ∀ s c, c ≠ y s → margin A y X' ≤ gap A y X' s c := fun s c hc =>
    margin_le_gap X' hc
  have hδ : 0 < margin A y X' - θ := sub_pos.mpr hθ
  filter_upwards [(tendsto_nuclearNorm_atTop hX' (hθ0.trans_lt hθ) hgap hpos hlam
    hmin).eventually_ge_atTop ((|Real.log (2 * (Fintype.card S * (Fintype.card C - 1)))| + 1) /
      (margin A y X' - θ))] with j hj
  have h1 := (div_le_iff₀ hδ).mp hj
  have h2 := le_abs_self (Real.log (2 * ((Fintype.card S : ℝ) * (Fintype.card C - 1))))
  have h3 := abs_nonneg (Real.log (2 * ((Fintype.card S : ℝ) * (Fintype.card C - 1))))
  have hB : 0 < nuclearNorm (Us j * (Vs j)ᴴ) := by
    by_contra h0
    rw [le_antisymm (not_lt.mp h0) (nuclearNorm_nonneg _), zero_mul] at h1
    linarith
  have h4 := mul_nonneg hB.le hθ0
  have h5 := le_margin fun s c hc =>
    (hmin j).sub_log_le_gap (hpos j).le hX' hgap (by linarith) hc
  rw [lt_div_iff₀ hB]
  linarith

/-- The largest margin of a product of nuclear norm at most one, item XE6. -/
noncomputable def maxMargin (A : S → C → Matrix m n ℂ) (y : S → C) : ℝ :=
  sSup (margin A y '' {X : Matrix m n ℂ | nuclearNorm X ≤ 1})

lemma margin_le_maxMargin [Nonempty S] [Nontrivial C] {X : Matrix m n ℂ}
    (hX : nuclearNorm X ≤ 1) : margin A y X ≤ maxMargin A y := by
  obtain ⟨a, ha0, ha⟩ := exists_forall_frobeniusSq_le A
  obtain ⟨s, c, hc, -⟩ := exists_gap_eq_margin A y X
  refine le_csSup ⟨2 * a, ?_⟩ ⟨X, hX, rfl⟩
  rintro _ ⟨Y, hY, rfl⟩
  exact (margin_le_gap Y hc).trans ((gap_le_mul_nuclearNorm ha0 ha Y s c).trans
    (mul_le_of_le_one_right (by positivity) hY))

/-- The normalized margin of every product is at most the largest margin of the unit ball. -/
lemma margin_div_le_maxMargin [Nonempty S] [Nontrivial C] (X : Matrix m n ℂ) :
    margin A y X / nuclearNorm X ≤ maxMargin A y := by
  rcases (nuclearNorm_nonneg X).eq_or_lt with h0 | h0
  · have h := margin_le_maxMargin (A := A) (y := y) (X := 0)
      (nuclearNorm_zero.trans_le zero_le_one)
    have e : margin A y (0 : Matrix m n ℂ) = 0 := by
      simpa using margin_smul (A := A) (y := y) le_rfl (0 : Matrix m n ℂ)
    rwa [← h0, div_zero, ← e]
  · have h := margin_le_maxMargin (A := A) (y := y) (X := (nuclearNorm X)⁻¹ • X) (by
      rw [nuclearNorm_real_smul (inv_nonneg.mpr h0.le), inv_mul_cancel₀ h0.ne'])
    rwa [margin_smul (inv_nonneg.mpr h0.le), inv_mul_eq_div] at h

/-- **The normalized margin of trained fits converges to the largest margin**
(`review/flat_spectrum.md`, item XE6): if some product has a positive margin and `λ_j → 0⁺`,
the trained fits of the cross-entropy have `μ(X_j) / ‖X_j‖_S1 → max {μ(X) : ‖X‖_S1 ≤ 1}`. -/
theorem tendsto_margin_div_nuclearNorm [Nonempty S] [Nontrivial C] {X₀ : Matrix m n ℂ}
    (hsep : 0 < margin A y X₀) (hpos : ∀ j, 0 < lams j) (hlam : Tendsto lams atTop (𝓝 0))
    (hmin : ∀ j, IsDecayedMin (ceLoss A y) (lams j) (Us j) (Vs j)) :
    Tendsto (fun j => margin A y (Us j * (Vs j)ᴴ) / nuclearNorm (Us j * (Vs j)ᴴ)) atTop
      (𝓝 (maxMargin A y)) := by
  have hM : 0 < maxMargin A y :=
    (div_pos hsep (nuclearNorm_pos_of_margin_pos hsep)).trans_le (margin_div_le_maxMargin X₀)
  refine tendsto_order.mpr ⟨fun θ hθ => ?_, fun θ hθ => Eventually.of_forall fun j =>
    (margin_div_le_maxMargin _).trans_lt hθ⟩
  have hne : (margin A y '' {X : Matrix m n ℂ | nuclearNorm X ≤ 1}).Nonempty :=
    ⟨_, ⟨0, nuclearNorm_zero.trans_le zero_le_one, rfl⟩⟩
  obtain ⟨_, ⟨X', hX', rfl⟩, hlt⟩ := exists_lt_of_lt_csSup hne (max_lt hθ hM)
  filter_upwards [eventually_lt_margin_div hX' (le_max_right θ 0) hlt hpos hlam hmin] with j hj
  exact (le_max_left θ 0).trans_lt hj

/-! The witness: trained fits of one sample of a `1 × 1` product read with the code `±1`, along
the weight decays `1 / (j + 1)`. The competitor `1` has `‖1‖_S1 = 1` and margin `2`. -/

lemma exists_seq_isDecayedMin_signProbe : ∃ Us Vs : ℕ → Matrix (Fin 1) (Fin 1) ℂ,
    ∀ j : ℕ, IsDecayedMin (ceLoss signProbe fun _ => true) (1 / ((j : ℝ) + 1)) (Us j) (Vs j) :=
  exists_seq_isDecayedMin_ceLoss signProbe (fun _ => true) (fun j => by positivity)
    (Or.inl le_rfl)

example := nuclearNorm_pos_of_margin_pos
  (two_pos.trans_le (le_margin (A := signProbe) (y := fun _ => true) le_gap_signProbe_one))

/-- Some trained fit of the witness has the norm that `IsDecayedMin.sub_log_le_gap` asks, and
so the margin bound of item XE5. -/
lemma exists_isDecayedMin_signProbe_log_le : ∃ (lam : ℝ) (U V : Matrix (Fin 1) (Fin 1) ℂ),
    0 < lam ∧ IsDecayedMin (ceLoss signProbe fun _ => true) lam U V ∧
      Real.log (2 * ((Fintype.card (Fin 1) : ℝ) * (Fintype.card Bool - 1))) ≤
        nuclearNorm (U * Vᴴ) * 2 ∧
      nuclearNorm (U * Vᴴ) * 2
          - Real.log (2 * ((Fintype.card (Fin 1) : ℝ) * (Fintype.card Bool - 1))) ≤
        gap signProbe (fun _ => true) (U * Vᴴ) 0 false := by
  obtain ⟨Us, Vs, h⟩ := exists_seq_isDecayedMin_signProbe
  obtain ⟨j, hj⟩ := ((tendsto_nuclearNorm_atTop nuclearNorm_one_fin_one.le two_pos
    le_gap_signProbe_one (fun j => by positivity) tendsto_one_div_add_atTop_nhds_zero_nat
    h).eventually_ge_atTop (Real.log (2 * ((Fintype.card (Fin 1) : ℝ) *
      (Fintype.card Bool - 1))) / 2)).exists
  have hB := (div_le_iff₀ two_pos).mp hj
  exact ⟨_, Us j, Vs j, by positivity, h j, hB, (h j).sub_log_le_gap (by positivity)
    nuclearNorm_one_fin_one.le le_gap_signProbe_one hB (by simp)⟩

example : ∃ Us Vs : ℕ → Matrix (Fin 1) (Fin 1) ℂ, ∀ᶠ j in atTop, (1 : ℝ) <
    margin signProbe (fun _ => true) (Us j * (Vs j)ᴴ) / nuclearNorm (Us j * (Vs j)ᴴ) := by
  obtain ⟨Us, Vs, h⟩ := exists_seq_isDecayedMin_signProbe
  exact ⟨Us, Vs, eventually_lt_margin_div nuclearNorm_one_fin_one.le zero_le_one
    (one_lt_two.trans_le (le_margin le_gap_signProbe_one)) (fun j => by positivity)
    tendsto_one_div_add_atTop_nhds_zero_nat h⟩

example := margin_le_maxMargin (A := signProbe) (y := fun _ => true) nuclearNorm_one_fin_one.le

example : ∃ Us Vs : ℕ → Matrix (Fin 1) (Fin 1) ℂ, Tendsto (fun j =>
    margin signProbe (fun _ => true) (Us j * (Vs j)ᴴ) / nuclearNorm (Us j * (Vs j)ᴴ)) atTop
      (𝓝 (maxMargin signProbe fun _ => true)) := by
  obtain ⟨Us, Vs, h⟩ := exists_seq_isDecayedMin_signProbe
  exact ⟨Us, Vs, tendsto_margin_div_nuclearNorm
    (two_pos.trans_le (le_margin le_gap_signProbe_one)) (fun j => by positivity)
    tendsto_one_div_add_atTop_nhds_zero_nat h⟩

end Transformer.Grokking.FlatSpectrum.Training
