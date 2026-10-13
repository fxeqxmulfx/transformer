/-
# The unrestricted slow manifold does not satisfy the required flatness

ArXiv:2410.06833v1, §3.1 (H2), and §3.2, `eq: otto.attention`.
The corollary applies the Otto framework with `δ=exp(-λβ/2)` to
the displayed `𝒩_β⊂𝕋^n`. That set only controls pairs already in
one of the fixed caps. Configurations entirely outside the caps
therefore belong to it without any restriction on their interactions.

At the parameters of `circle_witness`, two such configurations
violate (H2): the consensus at `π/2` and the tuple `(π/3,2π/3,2π/3)`.
Their angular coordinate differences are each `π/6`, so the Euclidean
norm below also equals their product geodesic distance on the torus.
This refutes the unrestricted flatness needed by the source's
application. It does not by itself refute the dynamical corollary.

The corrected corollary explicitly restricts every projection to
the part of the slow manifold contained in the original caps.
No definition of `𝒩_β` or of the energy is changed.
-/

import Transformer.Metastability.Section3_SlowManifold
import Transformer.Metastability.Section3_EnergyAlgebra
import Mathlib.Analysis.Real.Pi.Bounds

open scoped BigOperators
open Real

namespace Transformer.Metastability

/-- **Counterexample to (H2) on the unrestricted `𝒩_β` used by
`eq: otto.attention`.** Source: arXiv:2410.06833v1, §3.1 (H2)
and §3.2, the displayed slow manifold and `eq: otto.attention`.
The numerical parameters and antipodal cap centres are exactly
those of `circle_witness`, which realizes the source's separation,
smallness, and `λ` assumptions for a genuine initial configuration.
Both points below are outside the fixed caps; their actual energies
differ by `(1-exp(-500))/4500`, whereas their distance is less
than two and `δ=exp(-500)<1/100000`. -/
theorem slow_manifold_flatness_counterexample :
    ∃ V W : Idx 3 → ℝ,
      V ∈ slowManifold 3 1000 (1 / 10 ^ 8) 1 2 ![0, π] ∧
      W ∈ slowManifold 3 1000 (1 / 10 ^ 8) 1 2 ![0, π] ∧
      (∀ q : Idx 2, ∀ i : Idx 3,
        Real.cos (V i - (![0, π] : Idx 2 → ℝ) q) < 1 - 2 * (1 / 10 ^ 8 : ℝ)) ∧
      Real.exp (-500) * Real.sqrt (∑ i : Idx 3, (V i - W i) ^ 2) <
        |angularEβ 3 1000 V - angularEβ 3 1000 W| := by
  let V : Idx 3 → ℝ := fun _ => π / 2
  let W : Idx 3 → ℝ := ![π / 3, 2 * π / 3, 2 * π / 3]
  have hhalf : π / 2 - π = -(π / 2) := by ring
  have hthird : π / 3 - π = -(2 * π / 3) := by ring
  have htwothird : 2 * π / 3 - π = -(π / 3) := by ring
  have hcross : π / 3 - 2 * π / 3 = -(π / 3) := by ring
  have hcross' : 2 * π / 3 - π / 3 = π / 3 := by ring
  have hcos2 : Real.cos (2 * π / 3) = -(1 / 2 : ℝ) := by
    rw [show 2 * π / 3 = π - π / 3 by ring, Real.cos_pi_sub, Real.cos_pi_div_three]
  have hvcos (q : Idx 2) (i : Idx 3) :
      Real.cos (V i - (![0, π] : Idx 2 → ℝ) q) = 0 := by
    fin_cases q <;> simp [V, hhalf]
  have hwcos (q : Idx 2) (i : Idx 3) :
      Real.cos (W i - (![0, π] : Idx 2 → ℝ) q) ≤ (1 / 2 : ℝ) := by
    fin_cases q <;> fin_cases i <;>
      norm_num [W, hthird, htwothird, hcos2]
  have hvN : V ∈ slowManifold 3 1000 (1 / 10 ^ 8) 1 2 ![0, π] := by
    intro q i j hi _
    rw [hvcos q i] at hi
    norm_num at hi
  have hwN : W ∈ slowManifold 3 1000 (1 / 10 ^ 8) 1 2 ![0, π] := by
    intro q i j hi _
    have h := hwcos q i
    norm_num at hi
    linarith
  have hEV : angularEβ 3 1000 V = (1 / 2000 : ℝ) := by
    rw [angularEβ_eq_pairPotential 3 (by norm_num) 1000 (by norm_num)]
    norm_num [V, angularPairPotential]
  have hEW : angularEβ 3 1000 W = (5 + 4 * Real.exp (-500)) / 18000 := by
    rw [angularEβ_eq_pairPotential 3 (by norm_num) 1000 (by norm_num)]
    norm_num [W, angularPairPotential, Fin.sum_univ_three, hcross, hcross']
    ring
  have hgap : angularEβ 3 1000 V - angularEβ 3 1000 W =
      (1 - Real.exp (-500)) / 4500 := by
    rw [hEV, hEW]
    ring
  have hsmall : Real.exp (-500) < (1 / 100000 : ℝ) := by
    rw [Real.exp_neg, inv_lt_comm₀ (Real.exp_pos _) (by norm_num)]
    have h := Real.quadratic_le_exp_of_nonneg (show (0 : ℝ) ≤ 500 by norm_num)
    norm_num at h ⊢
    linarith
  have hdist : Real.sqrt (∑ i : Idx 3, (V i - W i) ^ 2) ≤ 2 := by
    have hsq : (∑ i : Idx 3, (V i - W i) ^ 2) = π ^ 2 / 12 := by
      norm_num [Fin.sum_univ_three, V, W]
      ring
    rw [hsq]
    apply Real.sqrt_le_iff.mpr
    refine ⟨by norm_num, ?_⟩
    nlinarith [Real.pi_pos, Real.pi_lt_four]
  refine ⟨V, W, hvN, hwN, fun q i => ?_, ?_⟩
  · rw [hvcos q i]
    norm_num
  · rw [hgap, abs_of_pos (by linarith : 0 < (1 - Real.exp (-500)) / 4500)]
    have h := mul_le_mul_of_nonneg_left hdist (Real.exp_pos (-500)).le
    linarith

/-- **(H2) fails for the exact `δ` asserted in `eq: otto.attention`
on its unrestricted displayed slow manifold.** Source:
arXiv:2410.06833v1, §3.1 (H2), and §3.2, `eq: otto.attention`.
This negative theorem records the omitted-domain obstruction separately
from the corrected positive corollary. -/
theorem not_slow_manifold_energy_flatness :
    ¬ ∀ V W : Idx 3 → ℝ,
      V ∈ slowManifold 3 1000 (1 / 10 ^ 8) 1 2 ![0, π] →
      W ∈ slowManifold 3 1000 (1 / 10 ^ 8) 1 2 ![0, π] →
      |angularEβ 3 1000 V - angularEβ 3 1000 W| ≤
        Real.exp (-500) * Real.sqrt (∑ i : Idx 3, (V i - W i) ^ 2) := by
  intro hflat
  obtain ⟨V, W, hV, hW, -, hgap⟩ := slow_manifold_flatness_counterexample
  exact (not_le_of_gt hgap) (hflat V W hV hW)

/-- **Membership in the displayed `𝒩_β` does not imply cap containment.**
Source: arXiv:2410.06833v1, §3.2, `𝒩_β` in `lem: PL.borjan`.
The same actual point witnesses the failure at all three token indices. -/
theorem not_slow_manifold_implies_cap_containment :
    ¬ ∀ V : Idx 3 → ℝ,
      V ∈ slowManifold 3 1000 (1 / 10 ^ 8) 1 2 ![0, π] →
      ∀ i : Idx 3, ∃ q : Idx 2,
        1 - 2 * (1 / 10 ^ 8 : ℝ) ≤ Real.cos (V i - (![0, π] : Idx 2 → ℝ) q) := by
  intro hcontain
  obtain ⟨V, W, hV, -, houtside, -⟩ := slow_manifold_flatness_counterexample
  obtain ⟨q, hq⟩ := hcontain V hV 0
  exact (not_le_of_gt (houtside q 0)) hq

/-- The constants and cap family of the flatness counterexample are
admissible in the source's corollary. This example includes every
separation, diameter, `δ`, and `λ` condition, together with an actual
initial configuration. The two points refuting (H2) are other points
of its displayed slow manifold for precisely these fixed caps. -/
example : let τ : ℝ := 1 / 10 ^ 8
    let α : ℝ := 1 - 2 * (1 - 2 * τ) ^ 2
    let ω : Idx 2 → ℝ := ![0, π]
    let Θ : Idx 3 → ℝ := ![0, 1 / 10 ^ 4, π]
    (1 : ℝ) < 1000 ∧ 2 ≤ 3 ∧ 0 < τ ∧ τ < 1 / 16 ∧ 2 ≤ 3 ∧
    (∀ i : Idx 3, ∃ q : Idx 2, 1 - τ ≤ Real.cos (Θ i - ω q)) ∧
    (∀ q q' : Idx 2, q ≠ q' → ∀ u v : ℝ,
      1 - 2 * τ ≤ Real.cos (u - ω q) → 1 - 2 * τ ≤ Real.cos (v - ω q') →
        Real.cos (u - v) ≤ α) ∧
    0 < 1 - α - 8 * τ - (1000 : ℝ)⁻¹ * Real.log (2 * (3 : ℝ) ^ 2 / τ) ∧
    (∀ q : Idx 2, ∀ u v : ℝ, |u - ω q| ≤ π → |v - ω q| ≤ π →
      1 - 2 * τ ≤ Real.cos (u - ω q) → 1 - 2 * τ ≤ Real.cos (v - ω q) →
        |u - v| ≤ (1 / 8) * Real.sqrt ((1 - 1 / 2) / (1000 + 1 / 2))) ∧
    8 * (1 + 1000) * Real.exp (-((1 - α) * 1000)) * Real.exp (-(1 / 2 : ℝ)) < 1 / 2 ∧
    (1 / 2 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (1 : ℝ) < min
      (Real.exp ((1 - α - (1000 : ℝ)⁻¹ * Real.log ((1000 - 1) * τ
          / (1000 ^ 2 * (3 : ℝ) ^ 2 * Real.exp 1))) * 1000)
        * (1 - Real.exp (-((1 - α - 8 * τ
            - (1000 : ℝ)⁻¹ * Real.log (2 * (3 : ℝ) ^ 2 / τ)) * 1000))))
      (1 - α - (1000 : ℝ)⁻¹ * Real.log (2 * (3 : ℝ) ^ 2
          / (1 - Real.exp (-((1000 : ℝ)⁻¹ * Real.log (1 / (8 * τ)) * 1000))))
        - Real.exp (-((1000 : ℝ)⁻¹ * Real.log (1 / (8 * τ)) * 1000))) := by
  intro τ α ω Θ
  obtain ⟨-, hα, hγ, hsmall, hδ, hlam, -, -, -, -, hinit⟩ := circle_witness
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    hinit, hα, hγ, hsmall, hδ, by norm_num, one_pos, hlam⟩

end Transformer.Metastability
