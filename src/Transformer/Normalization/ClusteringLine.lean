/-
# Normalization — `thm: convergence` is false on the line

The source states `thm: convergence` and its corollary for every `d`.  At
`d = 1` the tangent space of `𝕊⁰` is `0` (`proj_one_eq_zero`), so every
configuration of directions is a rest point of `eq: NA` (`na_const_one`), and
two tokens of opposite signs stay apart forever (`not_synchronizes_const`):

* `not_clusters_from_uniform_one` — Post-LN from the uniform law on `(𝕊⁰)²`;
* `not_clusters_or_stalls_from_gaussian_one` — Pre-LN from a Gaussian pair,
  whose radii moreover grow at the rate `≥ 1/(2e^β)` (`pre_line_counter`);
* `not_unconditional_synchronization_one` — the corollary, at `n = 2 ≤ e^β`.

`Normalization.Clustering` therefore states all three for `d ≥ 2`.

Source: arXiv:2510.22026v2, §3, `thm: convergence` and its corollary.
-/

import Transformer.Normalization.Clustering
import Transformer.Normalization.Line

open Real MeasureTheory ProbabilityTheory

namespace Transformer
namespace Normalization

/-- On `𝕊⁰ ⊆ ℝ¹` the tangent space is `0`: `Proj_θ v = 0` for every unit `θ`. -/
theorem proj_one_eq_zero {θ : EucSpace 1} (hθ : ‖θ‖ = 1) (v : EucSpace 1) : proj 1 θ v = 0 := by
  have h2 := sq_coord_of_norm_one hθ
  ext i
  rw [Fin.fin_one_eq_zero i]
  simp [proj, EuclideanSpace.inner_eq_star_dotProduct, dotProduct]
  linear_combination (-(v 0)) * h2

/-- The hypothesis of `proj_one_eq_zero` is satisfiable: the base point. -/
example : ‖(basePoint 0 : EucSpace 1)‖ = 1 := mem_sphere_zero_iff_norm.mp (basePoint 0).2

/-- A constant configuration on `𝕊⁰` follows `eq: NA`, whatever the speed factors. -/
theorem na_const_one {n : ℕ} (β : ℝ) (Q K V : ℝ → ParamMatrix 1) (s : ℝ → Idx n → ℝ)
    (Θ : Idx n → EucSpace 1) (hΘ : ∀ j, ‖Θ j‖ = 1) :
    NA 1 n β Q K V s (fun _ => Θ) := by
  intro t j
  rw [proj_one_eq_zero (hΘ j), smul_zero]
  exact hasDerivAt_const t (Θ j)

/-- The hypothesis of `na_const_one` is satisfiable: every token at the base point. -/
example : ∀ j : Idx 2, ‖(fun _ => (basePoint 0 : EucSpace 1)) j‖ = 1 :=
  fun _ => mem_sphere_zero_iff_norm.mp (basePoint 0).2

/-- A constant configuration with two distinct points does not synchronize. -/
theorem not_synchronizes_const {d n : ℕ} (Θ : Idx n → EucSpace d) (i j : Idx n)
    (hij : Θ i ≠ Θ j) : ¬ Synchronizes d n (fun _ => Θ) := by
  rintro ⟨c, hc⟩
  exact hij ((tendsto_nhds_unique tendsto_const_nhds (hc i)).trans
    (tendsto_nhds_unique tendsto_const_nhds (hc j)).symm)

/-- The hypothesis of `not_synchronizes_const` is satisfiable: `±1`. -/
example : (fun i : Idx 2 => ((sph0 (i = 0) : SSphere 1) : EucSpace 1)) 0 ≠
    (fun i : Idx 2 => ((sph0 (i = 0) : SSphere 1) : EucSpace 1)) 1 := fun h =>
  sph0_ne (Subtype.ext (by simpa using h))

/-- **`thm: convergence` fails on `𝕊⁰`: Post-LN from a uniform start.**

`clusters_from_uniform` without its `d ≥ 2`, at `d = 1`, `n = 2`, Post-LN: the
two tokens start at `+1` and `-1` with probability `1/2`, and stay there.
arXiv:2510.22026v2, §3, `thm: convergence`, first half. -/
theorem not_clusters_from_uniform_one :
    ¬ ∀ (β : ℝ) (α : ℝ → ℝ) (τ : ℝ),
      ∀ σ : Measure (SphereTuple 1 2), Perspective.UniformTuple 1 2 σ →
      ∀ᵐ Θ₀ ∂σ, ∀ (θ : ℝ → Idx 2 → EucSpace 1) (r : ℝ → Idx 2 → ℝ),
        θ 0 = tupleCoe Θ₀ →
          SchemeDynamics 1 2 β (idParams 1) (idParams 1) (idParams 1) α τ Scheme.post θ r →
            Synchronizes 1 2 θ := by
  intro h
  set σ := Measure.pi fun _ : Idx 2 => uniformSph0
  have hae := h 0 0 0 σ (uniformTuple_sph0 2)
  set B : Set (SphereTuple 1 2) := Set.univ.pi fun i => {sph0 (i = 0)}
  have hB : σ B ≠ 0 := by
    rw [Measure.pi_pi, Fin.prod_univ_two]
    exact mul_ne_zero (uniform_sph0_pos _ uniformSph0_invariant _)
      (uniform_sph0_pos _ uniformSph0_invariant _)
  refine hB (measure_mono_null ?_ (ae_iff.1 hae))
  intro Θ₀ hΘ₀ hQ
  have hΘ : ∀ i, Θ₀ i = sph0 (i = 0) := fun i => hΘ₀ i (Set.mem_univ i)
  refine not_synchronizes_const (tupleCoe Θ₀) 0 1 ?_
    (hQ (fun _ => tupleCoe Θ₀) (fun _ _ => 0) rfl
      ⟨na_const_one _ _ _ _ _ _ fun j => mem_sphere_zero_iff_norm.mp (Θ₀ j).2,
        fun t _ => hasDerivAt_const t 0⟩)
  intro h01
  apply sph0_ne
  have := hΘ 0; have := hΘ 1
  apply Subtype.ext
  simp_all [tupleCoe]

/-- A positive and a negative token are both nonzero. -/
theorem ne_zero_of_pos_neg {X₀ : Idx 2 → EucSpace 1} (h0 : 0 < X₀ 0 0) (h1 : X₀ 1 0 < 0) :
    ∀ j, X₀ j ≠ 0 := by
  refine Fin.forall_fin_two.2 ⟨fun h => ?_, fun h => ?_⟩
  · rw [h] at h0; simp at h0
  · rw [h] at h1; simp at h1

/-- `2 ≤ e`: the corollary's `n ≤ e^β` at `n = 2`, `β = 1`. -/
theorem two_le_exp_one : ((2 : ℕ) : ℝ) ≤ Real.exp 1 := by
  have := Real.add_one_le_exp 1
  push_cast
  linarith

/-- The hypotheses of `ne_zero_of_pos_neg` and `pre_line_counter` are
satisfiable: `β = 1` and the tokens `±1`. -/
example : (0 : ℝ) < 1 ∧ ((2 : ℕ) : ℝ) ≤ Real.exp 1 ∧
    0 < (fun i : Idx 2 => ((sph0 (i = 0) : SSphere 1) : EucSpace 1)) 0 0 ∧
    (fun i : Idx 2 => ((sph0 (i = 0) : SSphere 1) : EucSpace 1)) 1 0 < 0 :=
  ⟨one_pos, two_le_exp_one, by simp [sph0_coord], by simp [sph0_coord]⟩

/-- From a Gaussian sample with one positive and one negative token on `ℝ¹`,
Pre-LN keeps both directions fixed and the radii growing at a constant rate
`≥ 1/(2e^β)`: the tokens neither synchronize nor stall.
arXiv:2510.22026v2, §3, `thm: convergence`, second half. -/
theorem pre_line_counter (β : ℝ) (hβ : 0 < β) (hn : ((2 : ℕ) : ℝ) ≤ Real.exp β) (α : ℝ → ℝ)
    (τ : ℝ) (X₀ : Idx 2 → EucSpace 1) (h0 : 0 < X₀ 0 0) (h1 : X₀ 1 0 < 0) :
    ∃ (θ : ℝ → Idx 2 → EucSpace 1) (r : ℝ → Idx 2 → ℝ),
      (∀ j, θ 0 j = ‖X₀ j‖⁻¹ • X₀ j) ∧ (∀ j, r 0 j = ‖X₀ j‖) ∧
      SchemeDynamics 1 2 β (idParams 1) (idParams 1) (idParams 1) α τ Scheme.pre θ r ∧
      ¬ Synchronizes 1 2 θ ∧ ¬ RadialStalls 1 2 β τ Scheme.pre θ := by
  have hX := ne_zero_of_pos_neg h0 h1
  set Θ : Idx 2 → EucSpace 1 := fun j => ‖X₀ j‖⁻¹ • X₀ j
  have hΘ : ∀ j, ‖Θ j‖ = 1 := fun j => norm_smul_inv_norm (hX j)
  set k : Idx 2 → ℝ := fun j =>
    radialDerivative 1 2 β (idParams 1) (idParams 1) (idParams 1) (fun _ => Θ) τ .pre 0 j
  have hk : ∀ j, ((2 : ℕ) * Real.exp β)⁻¹ ≤ k j := fun j =>
    radialDerivative_pre_lower_bound 1 2 β hβ (fun _ => Θ) 0 τ (fun l => hΘ l) hn j
  have hrd : ∀ t j,
      radialDerivative 1 2 β (idParams 1) (idParams 1) (idParams 1) (fun _ => Θ) τ .pre t j
        = k j := fun _ _ => rfl
  refine ⟨fun _ => Θ, fun t j => ‖X₀ j‖ + t * k j, fun _ => rfl, fun _ => by simp,
    ⟨na_const_one _ _ _ _ _ _ hΘ, fun t j => ?_⟩, ?_, ?_⟩
  · rw [hrd]
    simpa using ((hasDerivAt_id t).mul_const (k j)).const_add ‖X₀ j‖
  · refine not_synchronizes_const Θ 0 1 fun h => ?_
    have := congrArg (fun x : EucSpace 1 => x 0) h
    simp only [Θ, PiLp.smul_apply, smul_eq_mul] at this
    have a := mul_pos (inv_pos.2 (norm_pos_iff.2 (hX 0))) h0
    have b := mul_neg_of_pos_of_neg (inv_pos.2 (norm_pos_iff.2 (hX 1))) h1
    linarith
  · rintro ⟨⟨j, hj⟩, -⟩
    have hj' : (0 : ℝ) = k j := by rw [hj, funext fun t => hrd t j]; exact Filter.liminf_const _
    have := hk j
    have : (0 : ℝ) < ((2 : ℕ) * Real.exp β)⁻¹ := by positivity
    linarith

/-- **`thm: convergence` fails on `ℝ¹`: Pre-LN from a Gaussian start.**

`clusters_or_stalls_from_gaussian` without its `d ≥ 2`, at `d = 1`, `n = 2`,
Pre-LN, `β = 1`.  arXiv:2510.22026v2, §3, `thm: convergence`, second half. -/
theorem not_clusters_or_stalls_from_gaussian_one :
    ¬ ∀ (β : ℝ) (α : ℝ → ℝ) (τ : ℝ),
      ∀ᵐ X₀ ∂(Measure.pi fun _ : Idx 2 => stdGaussian (EucSpace 1)), (∀ j : Idx 2, X₀ j ≠ 0) →
        ∀ (θ : ℝ → Idx 2 → EucSpace 1) (r : ℝ → Idx 2 → ℝ),
          (∀ j : Idx 2, θ 0 j = ‖X₀ j‖⁻¹ • X₀ j) → (∀ j : Idx 2, r 0 j = ‖X₀ j‖) →
            SchemeDynamics 1 2 β (idParams 1) (idParams 1) (idParams 1) α τ Scheme.pre θ r →
              Synchronizes 1 2 θ ∨ RadialStalls 1 2 β τ Scheme.pre θ := by
  intro h
  refine not_ae_gaussian_pair (fun X₀ h0 h1 hQ => ?_) (h 1 0 0)
  obtain ⟨θ, r, hθ, hr, hdyn, hs, hst⟩ := pre_line_counter 1 one_pos two_le_exp_one 0 0 X₀ h0 h1
  exact (hQ (ne_zero_of_pos_neg h0 h1) θ r hθ hr hdyn).elim hs hst

/-- **The corollary to `thm: convergence` fails on `ℝ¹`.**

`unconditional_synchronization` without its `d ≥ 2`, at `d = 1`, Pre-LN,
`n = 2 ≤ e^β`, `β = 1`.  arXiv:2510.22026v2, §3, the corollary to
`thm: convergence`. -/
theorem not_unconditional_synchronization_one :
    ¬ ∀ (β : ℝ) (α : ℝ → ℝ) (τ : ℝ), 0 < β → ((2 : ℕ) : ℝ) ≤ Real.exp β →
      ∀ᵐ X₀ ∂(Measure.pi fun _ : Idx 2 => stdGaussian (EucSpace 1)), (∀ j : Idx 2, X₀ j ≠ 0) →
        ∀ (θ : ℝ → Idx 2 → EucSpace 1) (r : ℝ → Idx 2 → ℝ),
          (∀ j : Idx 2, θ 0 j = ‖X₀ j‖⁻¹ • X₀ j) → (∀ j : Idx 2, r 0 j = ‖X₀ j‖) →
            SchemeDynamics 1 2 β (idParams 1) (idParams 1) (idParams 1) α τ Scheme.pre θ r →
              Synchronizes 1 2 θ := by
  intro h
  refine not_ae_gaussian_pair (fun X₀ h0 h1 hQ => ?_) (h 1 0 0 one_pos two_le_exp_one)
  obtain ⟨θ, r, hθ, hr, hdyn, hs, -⟩ := pre_line_counter 1 one_pos two_le_exp_one 0 0 X₀ h0 h1
  exact hs (hQ (ne_zero_of_pos_neg h0 h1) θ r hθ hr hdyn)

end Normalization
end Transformer
