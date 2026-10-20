module

public import Transformer.Causal.Section3_SingleTokenLift
public import Mathlib.Analysis.Calculus.Deriv.MeanValue

/-!
# Exponential decay of the lifted transverse component

ArXiv:2411.04990v2, §3, `lemma1`, specialized to the orthogonal
leading splitting assumed in §4, `thm2`. The original strict quadratic
inequality supplies a positive gap by compactness. Multiplication of
the transverse squared norm by exp(2ct) then gives a decreasing function.
All decay bounds follow from the actual lifted differential equation.
-/

@[expose] public section

open Real Filter
open scoped Topology

namespace Transformer.Causal

variable {d : ℕ}

/-- The shifted linear equation and a dissipative quadratic bound imply
exponential decay of its squared norm. Source: arXiv:2411.04990v2, §3,
the exponential approach in `lemma1`, with the spectral hypotheses of `thm2`. -/
theorem norm_sq_le_exp_of_shifted_dissipation (V : ParamMatrix d) (lam c : ℝ)
    (z : ℝ → EucSpace d)
    (hz : ∀ t, HasDerivAt z (V (z t) - lam • z t) t)
    (hq : ∀ t, 0 ≤ t → inner ℝ (V (z t)) (z t) ≤ (lam - c) * ‖z t‖ ^ 2) :
    ∀ t, 0 ≤ t → ‖z t‖ ^ 2 ≤ ‖z 0‖ ^ 2 * Real.exp (-(2 * c * t)) := by
  let f : ℝ → ℝ := fun t => Real.exp (2 * c * t) * ‖z t‖ ^ 2
  have hd (t : ℝ) : HasDerivAt f
      (Real.exp (2 * c * t) * (2 * c * ‖z t‖ ^ 2 +
        2 * (inner ℝ (V (z t)) (z t) - lam * ‖z t‖ ^ 2))) t := by
    have h := (((hasDerivAt_id t).const_mul (2 * c)).exp).mul (hz t).norm_sq
    rw [inner_sub_right, real_inner_smul_right, real_inner_self_eq_norm_sq] at h
    have hi : inner ℝ (z t) (V (z t)) = inner ℝ (V (z t)) (z t) := real_inner_comm _ _
    rw [hi] at h
    convert h using 1
    · rfl
    · simp only [id_eq, mul_one]
      ring
  have hanti : AntitoneOn f (Set.Ici (0 : ℝ)) := by
    apply antitoneOn_of_deriv_nonpos (convex_Ici 0)
      (continuous_iff_continuousAt.mpr fun t => (hd t).continuousAt).continuousOn
      (fun t _ => (hd t).differentiableAt.differentiableWithinAt)
    intro t ht
    rw [(hd t).deriv]
    apply mul_nonpos_of_nonneg_of_nonpos (Real.exp_nonneg _)
    have h := hq t (interior_subset ht)
    nlinarith
  intro t ht
  have h := hanti (show (0 : ℝ) ∈ Set.Ici 0 from le_rfl) ht ht
  change Real.exp (2 * c * t) * ‖z t‖ ^ 2 ≤ Real.exp (2 * c * 0) * ‖z 0‖ ^ 2 at h
  simp only [mul_zero, Real.exp_zero, one_mul] at h
  have hm := mul_le_mul_of_nonneg_left h (Real.exp_nonneg (-(2 * c * t)))
  simpa only [← mul_assoc, ← Real.exp_add, neg_add_cancel, Real.exp_zero, one_mul,
    mul_comm (Real.exp (-(2 * c * t))) (‖z 0‖ ^ 2)] using hm

/-- A nonconstant decaying mode realizes the shifted equation and its
quadratic bound. Source: arXiv:2411.04990v2, §3, `lemma1`. -/
example : ∃ z : ℝ → EucSpace 1,
    (∀ t, HasDerivAt z ((0 : ParamMatrix 1) (z t) - (1 : ℝ) • z t) t) ∧
    (∀ t, 0 ≤ t → inner ℝ ((0 : ParamMatrix 1) (z t)) (z t) ≤
      ((1 : ℝ) - 1) * ‖z t‖ ^ 2) ∧ z 0 ≠ 0 := by
  let u : EucSpace 1 := basePoint 0
  let z : ℝ → EucSpace 1 := fun t => Real.exp (-t) • u
  refine ⟨z, ?_, fun t _ => by simp, ?_⟩
  · intro t
    have h := (((hasDerivAt_id t).neg).exp).smul (hasDerivAt_const t u)
    convert! h using 1
    simp only [z, zero_apply, one_smul, zero_sub, mul_neg_one, neg_smul,
      smul_zero, zero_add, Pi.neg_apply, id_eq]
  · simp only [z, neg_zero, Real.exp_zero, one_smul]
    exact norm_ne_zero_iff.mp (by rw [mem_sphere_zero_iff_norm.mp (basePoint 0).2]; norm_num)

/-- The actual lift minus its initial leading component solves the shifted
linear value equation. Invariance is needed only to identify this remainder
with the transverse projection, as in `singleTokenLift_transverse_decay`.
Source: arXiv:2411.04990v2, §§3–4, `lemma1` and `thm2`. -/
theorem hasDerivAt_singleTokenLift_transverse (V : ParamMatrix d)
    (L : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (heig : ∀ v ∈ L, V v = lam • v)
    (x : ℝ → EucSpace d) (hx : singleTokenODE d V x) (t : ℝ) :
    HasDerivAt (fun s => singleTokenLift V lam x s - L.starProjection (x 0))
      (V (singleTokenLift V lam x t - L.starProjection (x 0)) -
        lam • (singleTokenLift V lam x t - L.starProjection (x 0))) t := by
  have h := (hasDerivAt_singleTokenLift V lam x hx t).sub_const (L.starProjection (x 0))
  have hp := heig _ (L.starProjection_apply_mem (x 0))
  convert! h using 1
  rw [map_sub, hp, smul_sub]
  module

/-- The original strict transverse spectral condition gives a single
positive exponential rate for the actual lifted solution.
Source: arXiv:2411.04990v2, §§3–4, `lemma1` and `thm2`. -/
theorem singleTokenLift_transverse_decay (V : ParamMatrix d)
    (L : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (heig : ∀ v ∈ L, V v = lam • v)
    (hinv : ∀ z ∈ Lᗮ, V z ∈ Lᗮ)
    (hdecay : ∀ z ∈ Lᗮ, z ≠ 0 → inner ℝ (V z) z < lam * ‖z‖ ^ 2)
    (x : ℝ → EucSpace d) (hx : singleTokenODE d V x) :
    ∃ c : ℝ, 0 < c ∧ ∀ t, 0 ≤ t →
      ‖singleTokenLift V lam x t - L.starProjection (x 0)‖ ^ 2 ≤
        ‖x 0 - L.starProjection (x 0)‖ ^ 2 * Real.exp (-(2 * c * t)) := by
  obtain ⟨c, hc, hq⟩ := exists_transverse_gap V L lam hdecay
  refine ⟨c, hc, ?_⟩
  have h := norm_sq_le_exp_of_shifted_dissipation V lam c
    (fun t => singleTokenLift V lam x t - L.starProjection (x 0))
    (hasDerivAt_singleTokenLift_transverse V L lam heig x hx) ?_
  · simpa only [singleTokenLift, singleTokenScale, intervalIntegral.integral_same,
      Real.exp_zero, one_smul] using h
  intro t _
  apply hq
  rw [← starProjection_singleTokenLift V L lam heig hinv x hx t]
  exact L.sub_starProjection_mem_orthogonal _

/-- A genuine stationary unit leader on its leading space satisfies every
decay premise, including a positive transverse gap on a nontrivial complement.
Source: arXiv:2411.04990v2, §§3–4, `lemma1` and `thm2`. -/
example (u : SSphere 2) : ∃ (V : ParamMatrix 2) (L : Submodule ℝ (EucSpace 2))
    (x : ℝ → EucSpace 2),
    (∀ v ∈ L, V v = (1 : ℝ) • v) ∧ (∀ z ∈ Lᗮ, V z ∈ Lᗮ) ∧
    (∀ z ∈ Lᗮ, z ≠ 0 → inner ℝ (V z) z < (1 : ℝ) * ‖z‖ ^ 2) ∧
    singleTokenODE 2 V x := by
  let L : Submodule ℝ (EucSpace 2) := ℝ ∙ (u : EucSpace 2)
  have hu : (u : EucSpace 2) ∈ L := Submodule.mem_span_singleton_self _
  have hp := L.starProjection_eq_self_iff.mpr hu
  refine ⟨L.starProjection, L, fun _ => (u : EucSpace 2),
    fun v hv => by simpa using L.starProjection_eq_self_iff.mpr hv,
    ?_, ?_, ?_⟩
  · intro z hz
    rw [L.starProjection_apply_eq_zero_iff.mpr hz]
    exact Lᗮ.zero_mem
  · intro z hz hz0
    rw [L.starProjection_apply_eq_zero_iff.mpr hz, inner_zero_left, one_mul]
    exact sq_pos_of_pos (norm_pos_iff.mpr hz0)
  · intro t
    have hproj : proj 2 (u : EucSpace 2) (u : EucSpace 2) = 0 := by
      simpa only [one_smul] using proj_smul_self (mem_sphere_zero_iff_norm.mp u.2) 1
    simpa only [hp, hproj] using hasDerivAt_const t (u : EucSpace 2)

end Transformer.Causal
