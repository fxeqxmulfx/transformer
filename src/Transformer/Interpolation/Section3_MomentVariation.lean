/-
# Uniform first variations of barycenters and determinant estimates

ArXiv:2411.04551v3, §3.2, `lem: perturbation`, Part 2, from
the original integrated weak equation `eq: cauchy.pb` in §1.

Bounded velocities give uniform bounds for changes of linear moments.
For a continuous autonomous field, the continuous-test time modulus
also gives a first-order expansion whose time radius is chosen before
any weak solution. This remains valid for the actual nonsmooth ReLU
field; differentiability of the vector field is unnecessary.

The final scalar estimates compare the two-dimensional determinant of
longitudinal and transverse means with its first-order term. A nonzero
leading coefficient supplies one tolerance and time radius for all
moment pairs satisfying the proved bounds.
-/

import Transformer.Interpolation.Section1_TestModulus
import Transformer.Interpolation.Section3_RidgeLaw
import Transformer.Interpolation.Clustering

open Real MeasureTheory Filter Set

namespace Transformer.Interpolation

variable {d : ℕ}

/-- Every original linear moment changes at most by its test-vector norm
times the velocity bound and elapsed time.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem norm_cauchyPB_inner_sub_le (θ : TimeParams d) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d θ μ) (B : ℝ)
    (hVF : ∀ (s : ℝ) (x : SSphere d), ‖fullVF d θ (μ s) s x‖ ≤ B)
    (a : EucSpace d) (s t : ℝ) :
    ‖inner ℝ a (barycenter d (μ t)) - inner ℝ a (barycenter d (μ s))‖ ≤ ‖a‖ * B * |t - s| := by
  let φ (x : EucSpace d) := inner ℝ a x
  have hφ : ContDiff ℝ 1 φ := contDiff_const.inner ℝ contDiff_id
  have hD (x : SSphere d) : ‖fderiv ℝ φ (x : EucSpace d)‖ ≤ ‖a‖ := by
    have he : φ = (innerSL ℝ a) := rfl
    rw [he, (innerSL ℝ a).fderiv, innerSL_apply_norm]
  have h := norm_cauchyPB_test_sub_le θ μ hμ φ hφ B ‖a‖ (norm_nonneg _) hVF hD s t
  simpa only [φ, integral_inner (integrable_spherical_coe (μ t)),
    integral_inner (integrable_spherical_coe (μ s)), barycenter] using h

/-- A continuous bounded autonomous field gives a first-order expansion
of every linear moment, with one radius for every actual weak solution.
The remainder is bounded by `εt`; no field derivative is assumed.
Source: arXiv:2411.04551v3, §3.2, the perturbation first variation. -/
theorem exists_uniform_cauchyPB_linear_expansion (θ : TimeParams d)
    (F : EucSpace d → EucSpace d) (hF : Continuous F) (B : ℝ) (hB : 0 ≤ B)
    (hfield : ∀ (ν : Perspective.ProbSphere d) (s : ℝ) (x : SSphere d),
      fullVF d θ ν s x = F x)
    (hbound : ∀ x : SSphere d, ‖F x‖ ≤ B) (a : EucSpace d) (ε : ℝ) (hε : 0 < ε) :
    ∃ δ : ℝ, 0 < δ ∧ ∀ (μ : ℝ → Perspective.ProbSphere d), cauchyPB d θ μ →
      ∀ t : ℝ, 0 < t → t < δ →
        ‖inner ℝ a (barycenter d (μ t)) - inner ℝ a (barycenter d (μ 0)) -
          t * ∫ x : SSphere d, inner ℝ a (F x) ∂(μ 0 : Measure (SSphere d))‖ ≤ ε * t := by
  let f (x : EucSpace d) := inner ℝ a (F x)
  have hf : Continuous f := continuous_const.inner hF
  obtain ⟨δ, hδ, hm⟩ := exists_uniform_cauchyPB_test_radius θ B hB
    (fun ν s x => (hfield ν s x) ▸ hbound x) f hf ε hε
  refine ⟨δ, hδ, fun μ hμ t ht htd => ?_⟩
  let g (s : ℝ) := ∫ x : SSphere d, f x ∂(μ s : Measure (SSphere d))
  let φ (x : EucSpace d) := inner ℝ a x
  have hφ : ContDiff ℝ 1 φ := contDiff_const.inner ℝ contDiff_id
  have heq : (fun s => ∫ x : SSphere d, inner ℝ (gradient φ (x : EucSpace d))
      (fullVF d θ (μ s) s x) ∂(μ s : Measure (SSphere d))) = g := by
    funext s
    apply integral_congr_ae
    exact ae_of_all _ fun x => by
      dsimp only
      rw [gradient_inner_linear, hfield]
  have hweak := hμ φ hφ t
  rw [heq] at hweak
  have hmean (s : ℝ) : (∫ x : SSphere d, φ x ∂(μ s : Measure (SSphere d))) =
      inner ℝ a (barycenter d (μ s)) := integral_inner (integrable_spherical_coe (μ s)) a
  have hind : inner ℝ a (barycenter d (μ t)) - inner ℝ a (barycenter d (μ 0)) - t * g 0 =
      ∫ s in (0 : ℝ)..t, g s - g 0 := by
    rw [intervalIntegral.integral_sub hweak.1 intervalIntegrable_const,
      intervalIntegral.integral_const, sub_zero, smul_eq_mul]
    rw [hmean t, hmean 0] at hweak
    linarith [hweak.2]
  change ‖inner ℝ a (barycenter d (μ t)) - inner ℝ a (barycenter d (μ 0)) - t * g 0‖ ≤ ε * t
  rw [hind]
  have hbd : ∀ s ∈ uIoc (0 : ℝ) t, ‖g s - g 0‖ ≤ ε := by
    intro s hs
    rw [uIoc_of_le ht.le] at hs
    exact (hm μ hμ 0 s (by rw [sub_zero, abs_of_pos hs.1]; exact hs.2.trans_lt htd)).le
  simpa only [sub_zero, abs_of_pos ht] using intervalIntegral.norm_integral_le_of_norm_le_const hbd


/-- Longitudinal drift and transverse first-order errors bound the
actual determinant error. Auxiliary scalar estimate for the non-collinear
barycenters in arXiv:2411.04551v3, §3.2, `lem: perturbation`, Part 2. -/
theorem wedge_error_bound (A B M γ ε t α β x y u v : ℝ)
    (hA : 0 ≤ A) (hM : 0 ≤ M) (ht : 0 ≤ t)
    (hγ : |γ| ≤ 1)
    (hu : |u - t * α| ≤ ε * t) (hv : |v - t * β| ≤ ε * t)
    (hub : |u| ≤ B * t) (hvb : |v| ≤ B * t)
    (hx : |x - γ * A| ≤ M * t) (hy : |y - A| ≤ M * t) :
    |y * u - x * v - A * t * (α - γ * β)| ≤ 2 * A * ε * t + 2 * M * B * t ^ 2 := by
  have hlong : |y * u - x * v - A * (u - γ * v)| ≤ 2 * M * B * t ^ 2 := by
    calc
      _ = |(y - A) * u - (x - γ * A) * v| := congrArg abs (by ring)
      _ ≤ |(y - A) * u| + |(x - γ * A) * v| := abs_sub _ _
      _ = |y - A| * |u| + |x - γ * A| * |v| := by rw [abs_mul, abs_mul]
      _ ≤ (M * t) * (B * t) + (M * t) * (B * t) := by
        exact add_le_add (mul_le_mul hy hub (abs_nonneg _) (mul_nonneg hM ht))
          (mul_le_mul hx hvb (abs_nonneg _) (mul_nonneg hM ht))
      _ = _ := by ring
  have htrans : |u - γ * v - t * (α - γ * β)| ≤ 2 * ε * t := by
    calc
      _ = |(u - t * α) - γ * (v - t * β)| := congrArg abs (by ring)
      _ ≤ |u - t * α| + |γ * (v - t * β)| := abs_sub _ _
      _ = |u - t * α| + |γ| * |v - t * β| := by rw [abs_mul]
      _ ≤ ε * t + 1 * (ε * t) := add_le_add hu
        (mul_le_mul hγ hv (abs_nonneg _) zero_le_one)
      _ = _ := by ring
  calc
    _ = |(y * u - x * v - A * (u - γ * v)) + A * (u - γ * v - t * (α - γ * β))| :=
      congrArg abs (by ring)
    _ ≤ |y * u - x * v - A * (u - γ * v)| + |A * (u - γ * v - t * (α - γ * β))| := abs_add_le _ _
    _ = |y * u - x * v - A * (u - γ * v)| + A * |u - γ * v - t * (α - γ * β)| := by
      rw [abs_mul, abs_of_nonneg hA]
    _ ≤ 2 * M * B * t ^ 2 + A * (2 * ε * t) :=
      add_le_add hlong (mul_le_mul_of_nonneg_left htrans hA)
    _ = _ := by ring

/-- A determinant stays nonzero when its error is smaller than its
nonzero linear term. Auxiliary for arXiv:2411.04551v3, §3.2. -/
theorem wedge_ne_zero_of_error_bound (A c t ε B M D : ℝ) (hA : 0 ≤ A) (ht : 0 < t)
    (herr : |D - A * t * c| ≤ 2 * A * ε * t + 2 * M * B * t ^ 2)
    (hsmall : 2 * A * ε + 2 * M * B * t < A * |c|) : D ≠ 0 := by
  intro hD
  rw [hD, zero_sub, abs_neg, abs_mul, abs_mul, abs_of_nonneg hA, abs_of_pos ht] at herr
  have h := mul_lt_mul_of_pos_right hsmall ht
  nlinarith

/-- A nonzero first-order coefficient supplies tolerances chosen before
all moment values. Auxiliary for arXiv:2411.04551v3, §3.2,
`lem: perturbation`, Part 2. -/
theorem exists_wedge_tolerance (A c B M : ℝ) (hA : 0 < A) (hc : c ≠ 0) (hBM : 0 ≤ M * B) :
    ∃ ε δ : ℝ, 0 < ε ∧ 0 < δ ∧ ∀ t : ℝ, 0 ≤ t → t < δ →
      2 * A * ε + 2 * M * B * t < A * |c| := by
  have hcpos : 0 < |c| := abs_pos.mpr hc
  have hd : 0 < 8 * (M * B + 1) := by positivity
  refine ⟨|c| / 8, A * |c| / (8 * (M * B + 1)), by positivity, by positivity, fun t ht htd => ?_⟩
  have h := (lt_div_iff₀ hd).mp htd
  have hAc : 0 < A * |c| := mul_pos hA hcpos
  nlinarith [mul_nonneg hBM ht]

/-- An actual weak solution with zero neuron output jointly realizes
all weak-equation, continuity, uniform-field, and tolerance premises. -/
example : ∃ (θ : TimeParams 1) (μ : ℝ → Perspective.ProbSphere 1),
    cauchyPB 1 θ μ ∧ (0 : ℝ) ≤ 1 ∧ (0 : ℝ) < 1 ∧
    Continuous (fun _ : EucSpace 1 => (0 : EucSpace 1)) ∧
    (‖(0 : EucSpace 1)‖ ≤ 1) ∧
    ∀ (ρ : Perspective.ProbSphere 1) (s : ℝ) (x : SSphere 1),
      fullVF 1 θ ρ s x = 0 ∧ ‖fullVF 1 θ ρ s x‖ ≤ 1 := by
  let θ : TimeParams 1 := fun _ => reluRidgeParams (0 : Fin 1) 0 0 1
  let μ := ridgeLaw (0 : EucSpace 1) 0 1 (Perspective.diracProb 1 (basePoint 0))
  refine ⟨θ, μ, cauchyPB_ridgeLaw _ _ _ _ _, zero_le_one, zero_lt_one,
    continuous_const, by simp, fun ρ s x => ?_⟩
  have he : fullVF 1 θ ρ s x = 0 := by
    rw [fullVF_reluRidgeParams]
    simp [proj]
  rw [he]
  simp

/-- Nonzero numerical first variation and actual distinct longitudinal
and transverse moment values jointly realize every determinant premise. -/
example :
    let A : ℝ := 1; let B : ℝ := 1; let M : ℝ := 1; let γ : ℝ := 1 / 2
    let ε : ℝ := 1 / 8; let t : ℝ := 1 / 32; let α : ℝ := 1; let β : ℝ := 0
    let x := γ * A; let y := A; let u := t * α; let v := t * β
    0 < A ∧ 0 ≤ M ∧ 0 < t ∧ |γ| ≤ 1 ∧
      |u - t * α| ≤ ε * t ∧ |v - t * β| ≤ ε * t ∧
      |u| ≤ B * t ∧ |v| ≤ B * t ∧ |x - γ * A| ≤ M * t ∧ |y - A| ≤ M * t ∧
      α - γ * β ≠ 0 ∧ 0 ≤ M * B ∧
      |(y * u - x * v) - A * t * (α - γ * β)| ≤ 2 * A * ε * t + 2 * M * B * t ^ 2 ∧
      2 * A * ε + 2 * M * B * t < A * |α - γ * β| := by norm_num

end Transformer.Interpolation
