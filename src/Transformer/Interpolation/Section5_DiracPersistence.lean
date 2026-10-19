module

public import Transformer.Interpolation.Section5_ConcentrationDefect
public import Mathlib.Analysis.ODE.Gronwall

/-!
# Every zero-temperature weak solution from a Dirac law is stationary

ArXiv:2411.04551v3, §1, `eq: cauchy.pb` and `eq: vf`, and §5,
`prop:generic-discrete` for one token and `β = 0`.

The cluster hypothesis in that proposition quantifies over all weak solutions,
so a stationary particle trajectory alone does not witness it. Here the
actual integrated continuity equation gives a differentiable scalar defect.
Its derivative is bounded by twice its value, and the initial defect is zero.
Grönwall's inequality therefore forces the defect to vanish for every
nonnegative time. The probability law must remain the original Dirac law.

This supplies a complete, nonvacuous model of the original cluster hypothesis
without adding uniqueness or concentration to the definition of a solution.
The theorem concerns arbitrary weak solutions, including non-empirical laws.
-/

@[expose] public section

open Real MeasureTheory

namespace Transformer.Interpolation

variable {d : ℕ}

/-- **The linear concentration moment of every actual weak solution from
a unit Dirac law stays one at nonnegative times.**
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb` and `eq: vf` at `β = 0`;
the one-token case used in §5, `prop:generic-discrete`. -/
theorem cauchyPB_zero_dirac_inner (μ : ℝ → Perspective.ProbSphere d) (a : SSphere d)
    (hμ : cauchyPB d (pureAttentionParams d 0) μ)
    (h0 : (μ 0 : Measure (SSphere d)) = Measure.dirac a) (t : ℝ) (ht : 0 ≤ t) :
    (∫ x : SSphere d, inner (𝕜 := ℝ) (a : EucSpace d) (x : EucSpace d)
      ∂(μ t : Measure (SSphere d))) = 1 := by
  let u (s : ℝ) := 1 - ∫ x : SSphere d,
    inner (𝕜 := ℝ) (a : EucSpace d) (x : EucSpace d) ∂(μ s : Measure (SSphere d))
  let g (s : ℝ) := ∫ x : SSphere d, inner (𝕜 := ℝ) (a : EucSpace d)
    (proj d (x : EucSpace d) (∫ y : SSphere d, (y : EucSpace d)
      ∂(μ s : Measure (SSphere d)))) ∂(μ s : Measure (SSphere d))
  have hc : Continuous u := continuous_const.sub
    (continuous_cauchyPB_inner (pureAttentionParams d 0) μ hμ (a : EucSpace d))
  have hu (s : ℝ) : HasDerivAt u (-g s) s :=
    (hasDerivAt_cauchyPB_zero_inner μ hμ (a : EucSpace d) s).const_sub 1
  have hz : u 0 = 0 := by
    dsimp [u]
    rw [h0, integral_dirac, real_inner_self_eq_norm_sq, mem_sphere_zero_iff_norm.mp a.2]
    norm_num
  have hb (s : ℝ) : ‖-g s‖ ≤ 2 * ‖u s‖ + 0 := by
    have hg : |g s| ≤ 2 * u s := by
      simpa only [g, u, integral_inner (integrable_spherical_coe (μ s))] using
        zero_temperature_flux_abs_le_defect (μ s) a
    calc
      ‖-g s‖ = |g s| := by rw [norm_neg, Real.norm_eq_abs]
      _ ≤ 2 * u s := hg
      _ ≤ 2 * ‖u s‖ + 0 := by
        rw [Real.norm_eq_abs]
        linarith [le_abs_self (u s)]
  have hG := norm_le_gronwallBound_of_norm_deriv_right_le
    (δ := 0) (K := 2) (ε := 0) (a := 0) (b := t) hc.continuousOn
    (fun s _ => (hu s).hasDerivWithinAt) (by rw [hz, norm_zero])
    (fun s _ => hb s) t ⟨ht, le_rfl⟩
  rw [gronwallBound_ε0_δ0] at hG
  have hut : u t = 0 := norm_eq_zero.mp (le_antisymm hG (norm_nonneg _))
  dsimp [u] at hut
  linarith

/-- All hypotheses, including the actual weak equation and initial law,
hold for a stationary nonzero unit Dirac trajectory at time one. -/
example (a : SSphere 3) :
    cauchyPB 3 (pureAttentionParams 3 0) (fun _ => Perspective.diracProb 3 a) ∧
    (Perspective.diracProb 3 a : Measure (SSphere 3)) = Measure.dirac a ∧
    (0 : ℝ) ≤ 1 :=
  ⟨cauchyPB_pureAttention_dirac 0 a, rfl, zero_le_one⟩

/-- **Every actual zero-temperature weak solution with a unit Dirac
initial law remains that Dirac law.** Source: arXiv:2411.04551v3, §1,
`eq: cauchy.pb` and `eq: vf`; §5, `prop:generic-discrete` at `n = 1`.
No uniqueness hypothesis is required: vanishing concentration determines
the entire probability measure. -/
theorem cauchyPB_zero_dirac_eq (μ : ℝ → Perspective.ProbSphere d) (a : SSphere d)
    (hμ : cauchyPB d (pureAttentionParams d 0) μ)
    (h0 : (μ 0 : Measure (SSphere d)) = Measure.dirac a) (t : ℝ) (ht : 0 ≤ t) :
    μ t = Perspective.diracProb d a := by
  apply probSphere_eq_dirac_of_inner_mean_eq_one
  have hi := cauchyPB_zero_dirac_inner μ a hμ h0 t ht
  rwa [integral_inner (integrable_spherical_coe (μ t))] at hi

example (a : SSphere 1) :
    cauchyPB 1 (pureAttentionParams 1 0) (fun _ => Perspective.diracProb 1 a) ∧
    (Perspective.diracProb 1 a : Measure (SSphere 1)) = Measure.dirac a ∧
    (0 : ℝ) ≤ 0 :=
  ⟨cauchyPB_pureAttention_dirac 0 a, rfl, le_rfl⟩

/-- **Every weak solution from a unit Dirac law has the specified cluster
limit at zero temperature.** Source: arXiv:2411.04551v3, §5,
`prop:generic-discrete`, using the equation of §1.
Its nonnegative-time trajectory is constant by the proved persistence. -/
theorem cauchyPB_zero_dirac_tendsto (μ : ℝ → Perspective.ProbSphere d) (a : SSphere d)
    (hμ : cauchyPB d (pureAttentionParams d 0) μ)
    (h0 : (μ 0 : Measure (SSphere d)) = Measure.dirac a) :
    Filter.Tendsto μ Filter.atTop (nhds (Perspective.diracProb d a)) := by
  have heq : (fun _ : ℝ => Perspective.diracProb d a) =ᶠ[Filter.atTop] μ := by
    filter_upwards [Filter.eventually_ge_atTop (0 : ℝ)] with t ht
    exact (cauchyPB_zero_dirac_eq μ a hμ h0 t ht).symm
  exact Filter.Tendsto.congr' heq tendsto_const_nhds

example (a : SSphere 3) :
    cauchyPB 3 (pureAttentionParams 3 0) (fun _ => Perspective.diracProb 3 a) ∧
    (Perspective.diracProb 3 a : Measure (SSphere 3)) = Measure.dirac a :=
  ⟨cauchyPB_pureAttention_dirac 0 a, rfl⟩

/-- The source's exact empirical initial-law condition gives the same
limit for a one-token tuple and every actual weak solution.
Source: arXiv:2411.04551v3, §1, the empirical measure before `eq: vf`,
and §5, `prop:generic-discrete` at `n = 1`, `β = 0`. -/
theorem cauchyPB_zero_one_token_tendsto (X : SphereTuple d 1)
    (μ : ℝ → Perspective.ProbSphere d)
    (h0 : (μ 0 : Measure (SSphere d)) = Perspective.empiricalMeasure d 1 X)
    (hμ : cauchyPB d (pureAttentionParams d 0) μ) :
    Filter.Tendsto μ Filter.atTop (nhds (Perspective.diracProb d (X 0))) := by
  have he : Perspective.empiricalMeasure d 1 X = Measure.dirac (X 0) := by
    rw [← Perspective.coe_empiricalProb one_pos X, Perspective.empiricalProb_one]
    rfl
  exact cauchyPB_zero_dirac_tendsto μ (X 0) hμ (h0.trans he)

/-- The complete empirical initial-law and weak-equation assumptions are
attained by a concrete one-token solution. -/
example (a : SSphere 3) :
    (Perspective.diracProb 3 a : Measure (SSphere 3)) =
      Perspective.empiricalMeasure 3 1 (fun _ => a) ∧
    cauchyPB 3 (pureAttentionParams 3 0) (fun _ => Perspective.diracProb 3 a) := by
  refine ⟨?_, cauchyPB_pureAttention_dirac 0 a⟩
  rw [← Perspective.coe_empiricalProb one_pos, Perspective.empiricalProb_one]

/-- **The entire universal cluster-map condition is attained.** For a
one-token input at zero temperature, the cluster map is evaluation at its
only token, for every initial reference law. This quantifies over every
weak solution, exactly as the original theorem does.
Source: arXiv:2411.04551v3, §5, `prop:generic-discrete`, at `n = 1`, `β = 0`. -/
theorem zero_one_token_clusters (ref : Measure (SphereTuple d 1)) :
    ∀ᵐ X : SphereTuple d 1 ∂ref, ∀ μ : ℝ → Perspective.ProbSphere d,
      (μ 0 : Measure (SSphere d)) = Perspective.empiricalMeasure d 1 X →
      cauchyPB d (pureAttentionParams d 0) μ →
      Filter.Tendsto μ Filter.atTop (nhds (Perspective.diracProb d (X 0))) := by
  apply ae_of_all
  intro X μ h0 hμ
  exact cauchyPB_zero_one_token_tendsto X μ h0 hμ

end Transformer.Interpolation
