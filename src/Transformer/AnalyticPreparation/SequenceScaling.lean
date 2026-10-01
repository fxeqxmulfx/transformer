/-
# Real analytic preparation: SequenceScaling

Adapted from Bochao Kong's classical-complex-wpt, revision
b4a7273fe5c9752753c52e10494097569089642d:
https://github.com/BochaoKong/classical-complex-wpt
Only the existence proof and its coefficient infrastructure are copied.
The scalar field is real; no complex root or complex differentiation result
is used. Apache-2.0 license: third_party/classical-complex-wpt/LICENSE.
-/

import Transformer.AnalyticPreparation.CoefficientConvergence

open Filter Finset
open scoped BigOperators ENNReal NNReal Topology

noncomputable section
set_option maxHeartbeats 5000000

namespace Transformer.AnalyticPreparation

variable {A I J : Type*} [AddCommMonoid A] [Finset.HasAntidiagonal A]

abbrev OriginSeq := L1Coeff ℕ

/-- The monomial coefficient vector supported at degree `d`. -/
noncomputable def monomialSeq (d : ℕ) : OriginSeq := lp.single 1 d 1

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp] lemma monomialSeq_apply_same (d : ℕ) : monomialSeq d d = 1 := by
  simp [monomialSeq, lp.single_apply]

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp] lemma monomialSeq_apply_ne {d k : ℕ} (h : k ≠ d) : monomialSeq d k = 0 := by
  simp [monomialSeq, lp.single_apply, h]

/-- Diagonal rescaling of an `ℓ¹` sequence by powers of `t ≤ 1`. -/
noncomputable def scaleSeq (t : ℝ≥0) (ht : t ≤ 1) (f : OriginSeq) : OriginSeq :=
  ⟨fun k ↦ (t : ℝ) ^ k * f k, by
    apply memℓp_gen
    have hs : Summable (fun k ↦ ‖(t : ℝ) ^ k * f k‖) :=
      (L1Coeff.summable_norm f).of_nonneg_of_le (fun _ ↦ norm_nonneg _) (fun k ↦ by
        rw [norm_mul, norm_pow, Real.norm_of_nonneg t.coe_nonneg]
        simpa using mul_le_of_le_one_left (norm_nonneg (f k))
          (pow_le_one₀ t.coe_nonneg (by exact_mod_cast ht)))
    simpa using hs⟩

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp] lemma scaleSeq_apply (t : ℝ≥0) (ht : t ≤ 1) (f : OriginSeq) (k : ℕ) :
    scaleSeq t ht f k = (t : ℝ) ^ k * f k := rfl

noncomputable def normalizedScale (t : ℝ≥0) (ht : t ≤ 1)
    (f : OriginSeq) (d : ℕ) : OriginSeq :=
  ((scaleSeq t ht f d)⁻¹) • scaleSeq t ht f

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp] lemma normalizedScale_apply (t : ℝ≥0) (ht : t ≤ 1)
    (f : OriginSeq) (d k : ℕ) :
    normalizedScale t ht f d k =
      (((t : ℝ) ^ d * f d)⁻¹) * ((t : ℝ) ^ k * f k) := rfl

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
lemma normalizedScale_sub_monomial_apply_lt {t : ℝ≥0} (ht : t ≤ 1) (ht0 : 0 < t)
    (f : OriginSeq) (d k : ℕ) (hlow : ∀ j < d, f j = 0) (hfd : f d ≠ 0) :
    ‖(normalizedScale t ht f d - monomialSeq d) k‖ ≤
      ((t : ℝ) / ‖f d‖) * ‖f k‖ := by
  change ‖normalizedScale t ht f d k - monomialSeq d k‖ ≤
    ((t : ℝ) / ‖f d‖) * ‖f k‖
  by_cases hkd : k < d
  · have hne : k ≠ d := ne_of_lt hkd
    simp [normalizedScale_apply, monomialSeq_apply_ne hne, hlow k hkd]
  by_cases hdk : k = d
  · subst k
    have htC : (t : ℝ) ≠ 0 := by exact_mod_cast ht0.ne'
    have hprod : (t : ℝ) ^ d * f d ≠ 0 := mul_ne_zero (pow_ne_zero _ htC) hfd
    have heq : normalizedScale t ht f d d = 1 := by
      rw [normalizedScale_apply, inv_mul_cancel₀ hprod]
    rw [heq, monomialSeq_apply_same, sub_self, norm_zero]
    positivity
  have hdkle : d ≤ k := Nat.le_of_not_gt hkd
  have hdklt : d < k := lt_of_le_of_ne hdkle (Ne.symm hdk)
  have htC : (t : ℝ) ≠ 0 := by exact_mod_cast ht0.ne'
  have htpowd : (t : ℝ) ^ d ≠ 0 := pow_ne_zero _ htC
  rw [normalizedScale_apply, monomialSeq_apply_ne hdk, sub_zero]
  rw [norm_mul, norm_inv, norm_mul, norm_pow, Real.norm_of_nonneg t.coe_nonneg]
  rw [norm_mul, norm_pow, Real.norm_of_nonneg t.coe_nonneg]
  have htRpos : 0 < (t : ℝ) := by exact_mod_cast ht0
  have htRne : (t : ℝ) ≠ 0 := htRpos.ne'
  have hfdnorm : ‖f d‖ ≠ 0 := norm_ne_zero_iff.mpr hfd
  have hpow : (t : ℝ) ^ k = (t : ℝ) ^ (k - d) * (t : ℝ) ^ d := by
    rw [← pow_add, Nat.sub_add_cancel hdkle]
  have heq :
      (((t : ℝ) ^ d * ‖f d‖)⁻¹ * ((t : ℝ) ^ k * ‖f k‖)) =
        (((t : ℝ) ^ (k - d)) / ‖f d‖) * ‖f k‖ := by
    rw [hpow]
    field_simp
  rw [heq]
  gcongr
  apply pow_le_of_le_one t.coe_nonneg (by exact_mod_cast ht)
  omega

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem norm_normalizedScale_sub_monomial_le {t : ℝ≥0} (ht : t ≤ 1) (ht0 : 0 < t)
    (f : OriginSeq) (d : ℕ) (hlow : ∀ j < d, f j = 0) (hfd : f d ≠ 0) :
    ‖normalizedScale t ht f d - monomialSeq d‖ ≤
      ((t : ℝ) / ‖f d‖) * ‖f‖ := by
  rw [L1Coeff.norm_eq_tsum_norm, L1Coeff.norm_eq_tsum_norm]
  have hsright : Summable (fun k ↦ ((t : ℝ) / ‖f d‖) * ‖f k‖) :=
    (L1Coeff.summable_norm f).mul_left _
  calc
    (∑' k, ‖(normalizedScale t ht f d - monomialSeq d) k‖) ≤
        ∑' k, ((t : ℝ) / ‖f d‖) * ‖f k‖ := by
      apply Summable.tsum_le_tsum
      · intro k
        exact normalizedScale_sub_monomial_apply_lt ht ht0 f d k hlow hfd
      · exact L1Coeff.summable_norm _
      · exact hsright
    _ = ((t : ℝ) / ‖f d‖) * ∑' k, ‖f k‖ := by rw [tsum_mul_left]

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem exists_scale_normalized_close_half (f : OriginSeq) (d : ℕ)
    (hlow : ∀ j < d, f j = 0) (hfd : f d ≠ 0) :
    ∃ (t : ℝ≥0) (ht : t ≤ 1), 0 < t ∧ t < 1 ∧
      ‖normalizedScale t ht f d - monomialSeq d‖ < (1 : ℝ) / 2 := by
  let a : ℝ := ‖f d‖
  let b : ℝ := ‖f‖
  have ha : 0 < a := by exact norm_pos_iff.mpr hfd
  have hb : 0 ≤ b := norm_nonneg _
  have hden : 0 < 4 * (b + a) := by positivity
  let tr : ℝ := a / (4 * (b + a))
  have htr0 : 0 < tr := div_pos ha hden
  have htr1 : tr < 1 := by
    rw [div_lt_one hden]
    nlinarith
  let t : ℝ≥0 := ⟨tr, htr0.le⟩
  have ht0 : 0 < t := by exact_mod_cast htr0
  have ht1 : t < 1 := by exact_mod_cast htr1
  refine ⟨t, ht1.le, ht0, ht1, ?_⟩
  calc
    ‖normalizedScale t ht1.le f d - monomialSeq d‖ ≤
        ((t : ℝ) / ‖f d‖) * ‖f‖ :=
      norm_normalizedScale_sub_monomial_le ht1.le ht0 f d hlow hfd
    _ = b / (4 * (b + a)) := by
      change (tr / a) * b = b / (4 * (b + a))
      dsimp only [tr]
      field_simp
    _ < (1 : ℝ) / 2 := by
      rw [div_lt_iff₀ hden]
      nlinarith

/-- Distinguished-variable coefficients at the base origin, weighted by `R^k`. -/
noncomputable def originWeightedCoeffs {n : ℕ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (R : ℝ≥0)
    (hR : (R : ℝ≥0∞) < p.radius) : OriginSeq :=
  ⟨fun k ↦ (R : ℝ) ^ k * lastTaylorCoefficient p k 0, by
    apply memℓp_gen
    have hs : Summable (fun k ↦ ‖(R : ℝ) ^ k * lastTaylorCoefficient p k 0‖) :=
      (p.summable_norm_mul_pow hR).of_nonneg_of_le (fun _ ↦ norm_nonneg _) (fun k ↦ by
        rw [lastTaylorCoefficient_zero, norm_mul, norm_pow, Real.norm_of_nonneg R.coe_nonneg]
        have hc : ‖p k (fun _ ↦ lastDirection n)‖ ≤ ‖p k‖ := by
          simpa [norm_lastDirection] using (p k).le_opNorm (fun _ ↦ lastDirection n)
        calc
          (R : ℝ) ^ k * ‖p k (fun _ ↦ lastDirection n)‖ ≤
              (R : ℝ) ^ k * ‖p k‖ :=
            mul_le_mul_of_nonneg_left hc (pow_nonneg R.coe_nonneg k)
          _ = ‖p k‖ * (R : ℝ) ^ k := mul_comm _ _)
    simpa using hs⟩

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp] lemma originWeightedCoeffs_apply {n : ℕ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (R : ℝ≥0)
    (hR : (R : ℝ≥0∞) < p.radius) (k : ℕ) :
    originWeightedCoeffs p R hR k =
      (R : ℝ) ^ k * lastTaylorCoefficient p k 0 := rfl

end Transformer.AnalyticPreparation
