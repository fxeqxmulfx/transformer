module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DensityBounds.Fourier

/-!
# Feature error and the observable kernel

Source: `review/flat_spectrum.md`, CM22. The entrywise feature error is an
explicit structural hypothesis, not inferred from normalization or flatness.
The bound retains its dimension factor and applies to arbitrary densities.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.DensityBounds

open Finset Matrix DensityExtraction

variable {ι X : Type*} [Fintype ι]

/-- A two-factor perturbation estimate, CM22. -/
theorem product_error (a b c d : ℂ) {δ R : ℝ} (hδ : 0 ≤ δ) (hR : 0 ≤ R)
    (ha : ‖a - c‖ ≤ δ) (hb : ‖b - d‖ ≤ δ) (hc : ‖c‖ ≤ R) (hd : ‖d‖ ≤ R) :
    ‖a * star b - c * star d‖ ≤ δ * (2 * R + δ) := by
  have hbn : ‖b‖ ≤ R + δ := by linarith [norm_sub_norm_le b d]
  have he : a * star b - c * star d = (a - c) * star b + c * star (b - d) := by
    rw [star_sub]
    ring
  rw [he]
  calc
    _ ≤ ‖(a - c) * star b‖ + ‖c * star (b - d)‖ := norm_add_le _ _
    _ = ‖a - c‖ * ‖b‖ + ‖c‖ * ‖b - d‖ := by simp only [norm_mul, norm_star]
    _ ≤ δ * (R + δ) + R * δ := add_le_add
      (mul_le_mul ha hbn (norm_nonneg _) hδ) (mul_le_mul hc hb (norm_nonneg _) hR)
    _ = _ := by ring

/-- Explicit kernel-reading error for a PSD trace-one coefficient matrix,
CM22: `κ = d² δ (2R+δ)`. No unknown operator norm is hidden in the bound. -/
theorem reading_perturbation (B C : Matrix X ι ℂ) {M : Matrix ι ι ℂ}
    (hM : Density M) {δ R : ℝ} (hδ : 0 ≤ δ) (hR : 0 ≤ R)
    (he : ∀ x i, ‖B x i - C x i‖ ≤ δ) (hc : ∀ x i, ‖C x i‖ ≤ R) (x y : X) :
    ‖reading B M x y - reading C M x y‖ ≤ (Fintype.card ι : ℝ)^2 * δ * (2 * R + δ) := by
  rw [reading_eq, reading_eq, ← sum_sub_distrib]
  calc
    _ ≤ ∑ i, ‖(∑ j, M i j * (B x i * star (B y j))) -
        ∑ j, M i j * (C x i * star (C y j))‖ := norm_sum_le _ _
    _ ≤ ∑ i : ι, ∑ j : ι, δ * (2 * R + δ) := by
      apply sum_le_sum
      intro i hi
      rw [← sum_sub_distrib]
      apply norm_sum_le_of_le
      intro j hj
      rw [← mul_sub, norm_mul]
      simpa only [one_mul] using mul_le_mul (density_entry_le_one hM i j)
        (product_error _ _ _ _ hδ hR (he x i) (he y j) (hc x i) (hc y j))
        (norm_nonneg _) (by norm_num : (0 : ℝ) ≤ 1)
    _ = _ := by simp [pow_two, mul_assoc]

/-- Moving from an estimated basis to its structural reference increases
the diagonal tolerance by the explicit kernel error, CM22. -/
theorem relaxed_transfer (B C : Matrix X ι ℂ) {M : Matrix ι ι ℂ} {ε δ R : ℝ}
    (hM : Relaxed B ε M) (hδ : 0 ≤ δ) (hR : 0 ≤ R)
    (he : ∀ x i, ‖B x i - C x i‖ ≤ δ) (hc : ∀ x i, ‖C x i‖ ≤ R) :
    Relaxed C (ε + (Fintype.card ι : ℝ)^2 * δ * (2 * R + δ)) M := by
  refine ⟨hM.1, fun x => ?_⟩
  have hp := reading_perturbation B C hM.1 hδ hR he hc x x
  have ht := norm_add_le (reading C M x x - reading B M x x) (reading B M x x - 1)
  rw [sub_add_sub_cancel, norm_sub_rev (reading C M x x) (reading B M x x)] at ht
  linarith [hM.2 x]

/-- Every exact-reference feasible point remains feasible in the supplied
basis once its declared tolerance covers `κ`, CM22. This includes all pure
projectors and mixtures, without knowing which mode will be selected. -/
theorem reference_admitted (B C : Matrix X ι ℂ) {M : Matrix ι ι ℂ} {ε δ R : ℝ}
    (hM : Feasible C M) (hδ : 0 ≤ δ) (hR : 0 ≤ R)
    (he : ∀ x i, ‖B x i - C x i‖ ≤ δ) (hc : ∀ x i, ‖C x i‖ ≤ R)
    (hbudget : (Fintype.card ι : ℝ)^2 * δ * (2 * R + δ) ≤ ε) : Relaxed B ε M := by
  refine ⟨hM.1, fun x => ?_⟩
  have hp := reading_perturbation B C hM.1 hδ hR he hc x x
  rw [hM.2 x] at hp
  exact hp.trans hbudget

/-- Uniform reading perturbation transfers approximate real-objective
maximality to every exact-reference competitor, CM22, with loss `η+2κ`.
The approximate optimizer is a checked specification, not a convergence claim. -/
theorem objective_transfer (B C : Matrix X ι ℂ) {M : Matrix ι ι ℂ} {ε δ R η : ℝ}
    (hM : Density M) (hδ : 0 ≤ δ) (hR : 0 ≤ R)
    (he : ∀ x i, ‖B x i - C x i‖ ≤ δ) (hc : ∀ x i, ‖C x i‖ ≤ R)
    (hbudget : (Fintype.card ι : ℝ)^2 * δ * (2 * R + δ) ≤ ε) (x y : X)
    (hopt : ∀ N, Relaxed B ε N → (reading B N x y).re ≤ (reading B M x y).re + η)
    {N : Matrix ι ι ℂ} (hN : Feasible C N) :
    (reading C N x y).re ≤ (reading C M x y).re + η +
      2 * ((Fintype.card ι : ℝ)^2 * δ * (2 * R + δ)) := by
  have hn := hopt N (reference_admitted B C hN hδ hR he hc hbudget)
  have hp := (Complex.abs_re_le_norm _).trans (reading_perturbation B C hM hδ hR he hc x y)
  have hq := (Complex.abs_re_le_norm _).trans (reading_perturbation B C hN.1 hδ hR he hc x y)
  rw [Complex.sub_re, abs_le] at hp hq
  linarith [hp.1, hp.2, hq.1, hq.2]

end Transformer.Grokking.FlatSpectrum.CoupledMoments.DensityBounds
