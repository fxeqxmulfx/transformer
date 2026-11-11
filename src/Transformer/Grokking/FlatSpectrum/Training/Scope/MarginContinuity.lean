module

public import Transformer.Grokking.FlatSpectrum.FlatCapacity
public import Transformer.Grokking.FlatSpectrum.Training.MarginDirection

/-!
# The margin for the nuclear norm is continuous, and it decides the direction

Source: `review/flat_spectrum.md`, *Training with cross-entropy* (added), item XE18, with the
items XE6 and XE10.

* The nuclear norm moves by at most `√|m| ‖X − Y‖_S2` (`abs_nuclearNorm_sub_le`), so it is
  continuous (`continuous_nuclearNorm`). The margin is the least of finitely many gaps, which
  are continuous (`continuous_margin`). So the margin for the nuclear norm,
  `μ(X) / ‖X‖_S1`, is continuous at every matrix of nonzero nuclear norm
  (`continuousAt_margin_div_nuclearNorm`).
* Item XE10 for an arbitrary sequence of matrices: if `μ(X_j) / ‖X_j‖_S1 → μ*` and one product
  `Z₀` of the unit ball has the margin `μ*`, then `X_j / ‖X_j‖_S1 → Z₀`
  (`tendsto_inv_smul_of_tendsto_margin_div`). The trained fits of item XE6 are one such
  sequence; the phase ends of the run of item XE18 are another.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training.Scope

open Matrix Filter Topology

variable {m n S C : Type*} [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n]
  [Fintype S] [Fintype C] {A : S → C → Matrix m n ℂ} {y : S → C}

omit [Fintype S] [Fintype C] [DecidableEq m] in
/-- **The nuclear norm is Lipschitz for the Frobenius norm**: it moves by at most
`√|m| ‖X − Y‖_S2`, by the triangle inequality and `‖D‖_S1² ≤ rank D · ‖D‖_S2²`. -/
lemma abs_nuclearNorm_sub_le (X Y : Matrix m n ℂ) :
    |nuclearNorm X - nuclearNorm Y| ≤ √(Fintype.card m * frobeniusSq (X - Y)) := by
  have hD : nuclearNorm (X - Y) ≤ √(Fintype.card m * frobeniusSq (X - Y)) := by
    refine Real.le_sqrt_of_sq_le ((nuclearNorm_sq_le_rank_mul_frobeniusSq _).trans ?_)
    exact mul_le_mul_of_nonneg_right (Nat.cast_le.2 (rank_le_card_height _))
      (frobeniusSq_nonneg _)
  have hneg : nuclearNorm (Y - X) = nuclearNorm (X - Y) := by
    rw [← neg_sub X Y, ← neg_one_smul ℂ (X - Y), nuclearNorm_smul, norm_neg, norm_one, one_mul]
  have h1 := nuclearNorm_add_le Y (X - Y)
  have h2 := nuclearNorm_add_le X (Y - X)
  rw [add_sub_cancel] at h1 h2
  rw [abs_le]
  constructor <;> linarith

omit [Fintype S] [Fintype C] [DecidableEq m] in
/-- **The nuclear norm is continuous.** -/
theorem continuous_nuclearNorm : Continuous (nuclearNorm : Matrix m n ℂ → ℝ) := by
  rw [continuous_iff_continuousAt]
  intro X₀
  have h : Tendsto (fun X : Matrix m n ℂ => √(Fintype.card m * frobeniusSq (X - X₀))) (𝓝 X₀)
      (𝓝 0) := by
    have hc : Continuous fun X : Matrix m n ℂ => √(Fintype.card m * frobeniusSq (X - X₀)) :=
      Real.continuous_sqrt.comp (continuous_const.mul
        (continuous_frobeniusSq.comp (continuous_id.sub continuous_const)))
    have h0 := hc.tendsto X₀
    simpa [frobeniusSq] using h0
  exact tendsto_iff_dist_tendsto_zero.2 (squeeze_zero (fun _ => dist_nonneg)
    (fun X => by rw [Real.dist_eq]; exact abs_nuclearNorm_sub_le X X₀) h)

omit [DecidableEq m] [DecidableEq n] in
/-- **The margin is continuous**: it is the least of finitely many gaps
(`review/flat_spectrum.md`, items XE4 and XE10). -/
theorem continuous_margin [Nonempty S] [Nontrivial C] (A : S → C → Matrix m n ℂ) (y : S → C) :
    Continuous (margin A y) := by
  classical
  have := nonempty_wrong y
  have h : margin A y = fun X => Finset.univ.inf' Finset.univ_nonempty
      fun p : {p : S × C // p.2 ≠ y p.1} => gap A y X p.1.1 p.1.2 :=
    funext fun X => (Finset.inf'_univ_eq_ciInf _).symm
  rw [h]
  exact Continuous.finset_inf'_apply _ fun p _ => continuous_gap A y p.1.1 p.1.2

omit [DecidableEq m] in
/-- **The margin for the nuclear norm is continuous where the nuclear norm is not zero**
(`review/flat_spectrum.md`, item XE18). -/
theorem continuousAt_margin_div_nuclearNorm [Nonempty S] [Nontrivial C]
    (A : S → C → Matrix m n ℂ) (y : S → C) {X : Matrix m n ℂ} (hX : nuclearNorm X ≠ 0) :
    ContinuousAt (fun X => margin A y X / nuclearNorm X) X :=
  (continuous_margin A y).continuousAt.div continuous_nuclearNorm.continuousAt hX

/-- **A sequence whose margin for the nuclear norm tends to the largest converges in direction
to the product of largest margin, when that is unique** (`review/flat_spectrum.md`, item XE10
for an arbitrary sequence): if `μ(X_j) / ‖X_j‖_S1 → μ*` and `Z₀` is the only product of the
unit ball with the margin `μ*`, then `X_j / ‖X_j‖_S1 → Z₀`. -/
theorem tendsto_inv_smul_of_tendsto_margin_div [Nonempty S] [Nontrivial C]
    {X : ℕ → Matrix m n ℂ}
    (hμ : Tendsto (fun j => margin A y (X j) / nuclearNorm (X j)) atTop (𝓝 (maxMargin A y)))
    {Z₀ : Matrix m n ℂ}
    (huniq : ∀ Z : Matrix m n ℂ, nuclearNorm Z ≤ 1 → margin A y Z = maxMargin A y → Z = Z₀) :
    Tendsto (fun j => (nuclearNorm (X j))⁻¹ • X j) atTop (𝓝 Z₀) := by
  have : FirstCountableTopology (Matrix m n ℂ) :=
    inferInstanceAs (FirstCountableTopology (m → n → ℂ))
  refine tendsto_of_subseq_tendsto fun ns hns => ?_
  obtain ⟨Z, φ, hφ, hZ⟩ := exists_subseq_tendsto_inv_smul fun j => X (ns j)
  obtain ⟨h1, h2⟩ := margin_eq_maxMargin_of_tendsto hZ (fun j => nuclearNorm_inv_smul_le _)
    ((hμ.comp (hns.comp hφ.tendsto_atTop)).congr fun j =>
      (margin_inv_smul (A := A) (y := y) _).symm)
  exact ⟨φ, huniq Z h1 h2 ▸ hZ⟩

/-! The witness: one sample of a `1 × 1` product read with the code `±1`, where `μ* = 2` and
the product `1` has `‖1‖_S1 = 1` and the margin `2`. The constant sequence `1` has the largest
margin for its norm, and its direction is `1`. -/

private lemma margin_signProbe_one : margin signProbe (fun _ => true)
    (1 : Matrix (Fin 1) (Fin 1) ℂ) = 2 :=
  le_antisymm ((margin_le_maxMargin nuclearNorm_one_fin_one.le).trans_eq maxMargin_signProbe)
    (le_margin le_gap_signProbe_one)

example := abs_nuclearNorm_sub_le (1 : Matrix (Fin 1) (Fin 1) ℂ) 0

example := continuous_margin signProbe fun _ => true

example := continuousAt_margin_div_nuclearNorm signProbe (fun _ => true)
  (X := (1 : Matrix (Fin 1) (Fin 1) ℂ)) (by rw [nuclearNorm_one_fin_one]; exact one_ne_zero)

example : Tendsto (fun _ : ℕ => (nuclearNorm (1 : Matrix (Fin 1) (Fin 1) ℂ))⁻¹ •
    (1 : Matrix (Fin 1) (Fin 1) ℂ)) atTop (𝓝 1) :=
  tendsto_inv_smul_of_tendsto_margin_div (A := signProbe) (y := fun _ => true)
    (by rw [margin_signProbe_one, nuclearNorm_one_fin_one, maxMargin_signProbe, div_one]
        exact tendsto_const_nhds)
    fun _ => eq_one_of_margin_signProbe

end Transformer.Grokking.FlatSpectrum.Training.Scope
