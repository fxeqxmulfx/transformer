module

public import Transformer.Interpolation.Section3_MomentVariation
public import Transformer.Interpolation.Section3_PerturbationGeometry

/-!
# One common separation time for every original weak solution

ArXiv:2411.04551v3, §3.2, `lem: perturbation`, Part 2.

Initially collinear means have zero transverse component. If the scaled
transverse fluxes differ, the determinant of their longitudinal and
transverse moments has a nonzero first-order term. The preceding
uniform moment estimates control its remainder for every weak solution.

Consequently one positive radius works simultaneously for all original
solution pairs. The conclusion rules out every real collinearity factor.
No uniqueness of weak solutions is presumed in this argument.
-/

@[expose] public section

open Real MeasureTheory Filter Set

namespace Transformer.Interpolation

/-- A nonzero actual transverse first variation gives non-collinearity
at every sufficiently small positive time, with the time radius chosen
before all original weak solution pairs.
Source: arXiv:2411.04551v3, §3.2, `lem: perturbation`, Part 2. -/
theorem exists_uniform_noncolinear_time_of_flux_variation {d : ℕ}
    (θ : TimeParams d) (F : EucSpace d → EucSpace d) (hF : Continuous F)
    (B : ℝ) (hB : 0 ≤ B)
    (hfield : ∀ (ρ : Perspective.ProbSphere d) (s : ℝ) (x : SSphere d), fullVF d θ ρ s x = F x)
    (hbound : ∀ x : SSphere d, ‖F x‖ ≤ B)
    (μ₀ ν₀ : Perspective.ProbSphere d) (γ : ℝ) (hγ : |γ| ≤ 1)
    (hbary : barycenter d μ₀ = γ • barycenter d ν₀)
    (p : EucSpace d) (hp : ‖p‖ = 1) (hpm : inner ℝ p (barycenter d ν₀) = 0)
    (hm : barycenter d ν₀ ≠ 0)
    (hflux : (∫ x : SSphere d, inner ℝ p (F x) ∂(μ₀ : Measure (SSphere d))) ≠
      γ * ∫ x : SSphere d, inner ℝ p (F x) ∂(ν₀ : Measure (SSphere d))) :
    ∃ δ : ℝ, 0 < δ ∧ ∀ t : ℝ, 0 < t → t < δ →
      ∀ (μ ν : ℝ → Perspective.ProbSphere d), μ 0 = μ₀ → cauchyPB d θ μ →
        ν 0 = ν₀ → cauchyPB d θ ν →
        ∀ q : ℝ, barycenter d (μ t) ≠ q • barycenter d (ν t) := by
  let m := barycenter d ν₀
  let A := inner ℝ m m
  have hA : 0 < A := by
    change 0 < inner ℝ (barycenter d ν₀) (barycenter d ν₀)
    exact real_inner_self_pos.mpr hm
  let α := ∫ x : SSphere d, inner ℝ p (F x) ∂(μ₀ : Measure (SSphere d))
  let β := ∫ x : SSphere d, inner ℝ p (F x) ∂(ν₀ : Measure (SSphere d))
  let c := α - γ * β
  have hc : c ≠ 0 := sub_ne_zero.mpr hflux
  let M := ‖m‖ * B
  have hM : 0 ≤ M := mul_nonneg (norm_nonneg _) hB
  obtain ⟨ε, δ₁, hε, hδ₁, hsmall⟩ := exists_wedge_tolerance A c B M hA hc (mul_nonneg hM hB)
  obtain ⟨δ₂, hδ₂, hexpand⟩ := exists_uniform_cauchyPB_linear_expansion θ F hF B hB hfield hbound p ε hε
  refine ⟨min δ₁ δ₂, lt_min hδ₁ hδ₂, fun t ht htd μ ν hμ₀ hμ hν₀ hν q hq => ?_⟩
  have ht₁ : t < δ₁ := (lt_min_iff.mp htd).1
  have ht₂ : t < δ₂ := (lt_min_iff.mp htd).2
  let x := inner ℝ m (barycenter d (μ t))
  let y := inner ℝ m (barycenter d (ν t))
  let u := inner ℝ p (barycenter d (μ t))
  let v := inner ℝ p (barycenter d (ν t))
  have hμp : inner ℝ p (barycenter d μ₀) = 0 := by rw [hbary, real_inner_smul_right, hpm, mul_zero]
  have hμm : inner ℝ m (barycenter d μ₀) = γ * A := by rw [hbary, real_inner_smul_right]
  have hu : |u - t * α| ≤ ε * t := by
    have h := hexpand μ hμ t ht ht₂
    rw [hμ₀, hμp, sub_zero, Real.norm_eq_abs] at h
    exact h
  have hv : |v - t * β| ≤ ε * t := by
    have h := hexpand ν hν t ht ht₂
    rw [hν₀, hpm, sub_zero, Real.norm_eq_abs] at h
    exact h
  have hVFμ (s : ℝ) (x : SSphere d) : ‖fullVF d θ (μ s) s x‖ ≤ B := (hfield (μ s) s x) ▸ hbound x
  have hVFν (s : ℝ) (x : SSphere d) : ‖fullVF d θ (ν s) s x‖ ≤ B := (hfield (ν s) s x) ▸ hbound x
  have hub : |u| ≤ B * t := by
    have h := norm_cauchyPB_inner_sub_le θ μ hμ B hVFμ p 0 t
    rw [hμ₀, hμp, sub_zero, hp, one_mul, sub_zero, abs_of_pos ht, Real.norm_eq_abs] at h
    exact h
  have hvb : |v| ≤ B * t := by
    have h := norm_cauchyPB_inner_sub_le θ ν hν B hVFν p 0 t
    rw [hν₀, hpm, sub_zero, hp, one_mul, sub_zero, abs_of_pos ht, Real.norm_eq_abs] at h
    exact h
  have hx : |x - γ * A| ≤ M * t := by
    have h := norm_cauchyPB_inner_sub_le θ μ hμ B hVFμ m 0 t
    rw [hμ₀, hμm, sub_zero, abs_of_pos ht, Real.norm_eq_abs] at h
    exact h
  have hy : |y - A| ≤ M * t := by
    have h := norm_cauchyPB_inner_sub_le θ ν hν B hVFν m 0 t
    rw [hν₀, sub_zero, abs_of_pos ht, Real.norm_eq_abs] at h
    exact h
  have herr := wedge_error_bound A B M γ ε t α β x y u v hA.le hM ht.le hγ hu hv hub hvb hx hy
  have hne := wedge_ne_zero_of_error_bound A c t ε B M (y * u - x * v) hA.le ht herr (hsmall t ht.le ht₁)
  apply hne
  have hxm : x = q * y := by
    change inner ℝ m (barycenter d (μ t)) = q * inner ℝ m (barycenter d (ν t))
    rw [hq, real_inner_smul_right]
  have hup : u = q * v := by
    change inner ℝ p (barycenter d (μ t)) = q * inner ℝ p (barycenter d (ν t))
    rw [hq, real_inner_smul_right]
  rw [hxm, hup]
  ring

end Transformer.Interpolation

namespace Transformer.Interpolation

/-- The auxiliary allows a zero initial mean ratio. An antipodal mixture
and a Dirac law, with an actual projection neuron, jointly realize every
premise, including the nonzero transverse flux difference. -/
example : ∃ (θ : TimeParams 2) (F : EucSpace 2 → EucSpace 2)
    (μ₀ ν₀ : Perspective.ProbSphere 2) (p : EucSpace 2),
    Continuous F ∧ (0 : ℝ) ≤ 1 ∧
    (∀ (ρ : Perspective.ProbSphere 2) (s : ℝ) (x : SSphere 2), fullVF 2 θ ρ s x = F x) ∧
    (∀ x : SSphere 2, ‖F x‖ ≤ 1) ∧ |(0 : ℝ)| ≤ 1 ∧
    barycenter 2 μ₀ = (0 : ℝ) • barycenter 2 ν₀ ∧ ‖p‖ = 1 ∧
    inner ℝ p (barycenter 2 ν₀) = 0 ∧ barycenter 2 ν₀ ≠ 0 ∧
    (∫ x : SSphere 2, inner ℝ p (F x) ∂(μ₀ : Measure (SSphere 2))) ≠
      (0 : ℝ) * ∫ x : SSphere 2, inner ℝ p (F x) ∂(ν₀ : Measure (SSphere 2)) := by
  let a : SSphere 2 := basePoint 1
  let p : EucSpace 2 := EuclideanSpace.single 1 1
  let θ : TimeParams 2 := fun _ => reluRidgeParams (0 : Fin 2) p 0 1
  let F (x : EucSpace 2) := proj 2 x p
  let μ₀ := halfDirac 2 a (antipode 2 a)
  let ν₀ := Perspective.diracProb 2 a
  have hF : Continuous F := by unfold F proj; fun_prop
  have hp : ‖p‖ = 1 := by simp [p]
  have hν : barycenter 2 ν₀ = (a : EucSpace 2) := by simp [barycenter, ν₀, Perspective.diracProb]
  have hfield (ρ : Perspective.ProbSphere 2) (s : ℝ) (x : SSphere 2) : fullVF 2 θ ρ s x = F x := by
    rw [fullVF_reluRidgeParams]
    simp [F]
  have hbound (x : SSphere 2) : ‖F x‖ ≤ 1 :=
    (norm_proj_le (mem_sphere_zero_iff_norm.mp x.2) p).trans (by rw [hp])
  have hμ : barycenter 2 μ₀ = (0 : ℝ) • barycenter 2 ν₀ := by
    rw [barycenter_halfDirac]
    simp [antipode]
  have hpm : inner ℝ p (barycenter 2 ν₀) = 0 := by
    rw [hν]
    simp [p, a, basePoint, EuclideanSpace.inner_single_left]
  have hne : barycenter 2 ν₀ ≠ 0 := by
    rw [hν]
    intro h
    have hc := congrArg (fun z : EucSpace 2 => z 0) h
    simp [a, basePoint] at hc
  let f (z : SSphere 2) := inner ℝ p (F z)
  have hI (z : SSphere 2) : Integrable f ((2⁻¹ : ENNReal) • Measure.dirac z) :=
    (integrable_dirac (by simp)).smul_measure (by simp)
  have hflux : (∫ x : SSphere 2, f x ∂(μ₀ : Measure (SSphere 2))) = 1 := by
    rw [coe_halfDirac, integral_add_measure (hI a) (hI (antipode 2 a)),
      integral_smul_measure, integral_smul_measure, integral_dirac, integral_dirac]
    norm_num [f, F, p, proj, a, basePoint, antipode, EuclideanSpace.inner_single_left,
      EuclideanSpace.inner_single_right]
  refine ⟨θ, F, μ₀, ν₀, p, hF, zero_le_one, hfield, hbound, by norm_num, hμ, hp, hpm, hne, ?_⟩
  rw [zero_mul]
  change (∫ x : SSphere 2, f x ∂(μ₀ : Measure (SSphere 2))) ≠ 0
  rw [hflux]
  exact one_ne_zero

end Transformer.Interpolation
