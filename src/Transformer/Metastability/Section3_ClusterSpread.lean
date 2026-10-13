/-
# The collapsed configuration's displacement is controlled by the true gradient

ArXiv:2410.06833v1, §3.2, `lem: PL.borjan`, `eq: Ht.second.lb`.
Every coordinate is collapsed to one actual token from its assigned cap.
The uniform force slope bounds each resulting squared gap by the two
squared gradient components. Summing gives a configuration-independent
bound by the full Euclidean gradient-square norm.

The representative map takes values in the original finite token set.
The cap geometry proves the force-gap input, and the finite-sum estimate
then uses only the actual coordinate values of that input.
Chart replacement preserves the original torus configuration's gradient.
No evolution or off-slow-manifold gap is assumed in this static estimate.
-/

import Transformer.Metastability.Section3_ForceCoercivity
import Transformer.Metastability.Section3_SlowManifold

open scoped BigOperators
open Set Real

namespace Transformer.Metastability

/-- A coordinatewise force-gap bound controls the full squared displacement.
This is the finite-sum arithmetic used with the true angular gradients
below. Source: arXiv:2410.06833v1, §3.2, the Euclidean norm in `lem: PL.borjan`. -/
theorem sum_displacement_sq_le_of_force_gaps (n : ℕ) (Θ U g : Idx n → ℝ)
    (r : Idx n → Idx n)
    (hgaps : ∀ i : Idx n, |U i - Θ i| ≤ 4 * (n : ℝ) ^ 2 * |g (r i) - g i|) :
    (∑ i : Idx n, (U i - Θ i) ^ 2) ≤ 64 * (n : ℝ) ^ 5 * ∑ i : Idx n, (g i) ^ 2 := by
  let G := ∑ i : Idx n, (g i) ^ 2
  have hgi (i : Idx n) : (g i) ^ 2 ≤ G :=
    Finset.single_le_sum (fun j _ => sq_nonneg (g j)) (Finset.mem_univ i)
  have hper (i : Idx n) : (U i - Θ i) ^ 2 ≤ 64 * (n : ℝ) ^ 4 * G := by
    have hs := square_gap_le_of_force_bound (n : ℝ) (U i) (Θ i) (g (r i)) (g i) (hgaps i)
    have hpair : (g (r i)) ^ 2 + (g i) ^ 2 ≤ 2 * G := by linarith [hgi (r i), hgi i]
    have hm := mul_le_mul_of_nonneg_left hpair (by positivity : 0 ≤ 32 * (n : ℝ) ^ 4)
    nlinarith
  calc
    _ ≤ ∑ _i : Idx n, 64 * (n : ℝ) ^ 4 * G := Finset.sum_le_sum fun i _ => hper i
    _ = _ := by
      rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
      dsimp only [G]
      ring

/-- The finite-sum force-gap input is met on a nonempty two-coordinate tuple,
with the identity representative map and zero displacement. -/
example : ∀ i : Idx 2, |(i : ℝ) - (i : ℝ)| ≤
    4 * (2 : ℝ) ^ 2 * |(i : ℝ) - (i : ℝ)| := by
  intro _
  norm_num

/-- **Collapsing each assigned cap costs at most `64n⁵‖∇Eβ‖²` in squared displacement.**
The representative is an actual token, and every gradient component is
that of the original energy at the supplied configuration.
Source: arXiv:2410.06833v1, §3.2, the gradient-norm control in `lem: PL.borjan`.
The constant is a coarse bound uniform in the configuration and cap radius. -/
theorem collapsed_spread_le_gradient (n k : ℕ) (hn : 0 < n) (β τ δ α : ℝ)
    (ω : Idx k → ℝ) (p : Idx n → Idx k) (Θ : Idx n → ℝ)
    (hβ : 1 < β) (hτ : 0 < τ) (hτ16 : τ < 1 / 16) (hδ : 0 ≤ δ) (hδ1 : δ < 1)
    (hα : ∀ q q' : Idx k, q ≠ q' → ∀ u v : ℝ,
      1 - 2 * τ ≤ Real.cos (u - ω q) → 1 - 2 * τ ≤ Real.cos (v - ω q') →
        Real.cos (u - v) ≤ α)
    (hγ : 0 < 1 - α - 8 * τ - β⁻¹ * Real.log (2 * (n : ℝ) ^ 2 / τ))
    (hsmall : ∀ q : Idx k, ∀ u v : ℝ, |u - ω q| ≤ π → |v - ω q| ≤ π →
      1 - 2 * τ ≤ Real.cos (u - ω q) → 1 - 2 * τ ≤ Real.cos (v - ω q) →
        |u - v| ≤ (1 / 8) * Real.sqrt ((1 - δ) / (β + 1 / 2)))
    (hcaps : ∀ i : Idx n, 1 - 2 * τ ≤ Real.cos (Θ i - ω (p i)))
    (hchart : ∀ i : Idx n, |Θ i - ω (p i)| ≤ π) :
    (∑ i : Idx n, (collapseTuple p Θ i - Θ i) ^ 2) ≤
      64 * (n : ℝ) ^ 5 * ∑ i : Idx n, (angularGrad n β Θ i) ^ 2 := by
  have hβ0 : 0 < β := by linarith
  have hα1 := alpha_lt_one_of_gamma n hn β τ α hβ0 hτ hτ16 hγ
  have hcover (i : Idx n) : ∃ q : Idx k, 1 - 2 * τ ≤ Real.cos (Θ i - ω q) :=
    ⟨p i, hcaps i⟩
  have hchartAll (q : Idx k) (i : Idx n) (hi : 1 - 2 * τ ≤ Real.cos (Θ i - ω q)) :
      |Θ i - ω q| ≤ π := by
    have hq := cos_cap_unique τ α ω hα1 hα (Θ i) q (p i) hi (hcaps i)
    rw [hq]
    exact hchart i
  have hgap (i : Idx n) : |collapseTuple p Θ i - Θ i| ≤
      4 * (n : ℝ) ^ 2 * |angularGrad n β Θ (capRepresentative p i) - angularGrad n β Θ i| := by
    let j := capRepresentative p i
    have hpj : p j = p i := capRepresentative_same_cap p i
    have hjcap : 1 - 2 * τ ≤ Real.cos (Θ j - ω (p i)) := by simpa only [hpj] using hcaps j
    have hjchart : |Θ j - ω (p i)| ≤ π := by simpa only [hpj] using hchart j
    have hg := angularForce_gap_bound n k β τ δ α ω Θ (p i) i
      hβ hτ hτ16 hδ hδ1 hcover hα hγ (hsmall (p i)) (hchartAll (p i)) (hcaps i)
      (Θ j) (Θ i) hjchart hjcap (hchart i) (hcaps i)
    rw [angularForce_at_token, angularForce_at_token] at hg
    exact hg
  exact sum_displacement_sq_le_of_force_gaps n Θ (collapseTuple p Θ)
    (angularGrad n β Θ) (capRepresentative p) hgap

/-- **The spread bound is invariant under the original angle representatives.**
First rechart each token in its assigned cap, then collapse. No chart
hypothesis is needed on the original real representatives.
Source: arXiv:2410.06833v1, §3.2, the torus configuration in `lem: PL.borjan`. -/
theorem charted_collapsed_spread_le_gradient (n k : ℕ) (hn : 0 < n) (β τ δ α : ℝ)
    (ω : Idx k → ℝ) (p : Idx n → Idx k) (Θ : Idx n → ℝ)
    (hβ : 1 < β) (hτ : 0 < τ) (hτ16 : τ < 1 / 16) (hδ : 0 ≤ δ) (hδ1 : δ < 1)
    (hα : ∀ q q' : Idx k, q ≠ q' → ∀ u v : ℝ,
      1 - 2 * τ ≤ Real.cos (u - ω q) → 1 - 2 * τ ≤ Real.cos (v - ω q') →
        Real.cos (u - v) ≤ α)
    (hγ : 0 < 1 - α - 8 * τ - β⁻¹ * Real.log (2 * (n : ℝ) ^ 2 / τ))
    (hsmall : ∀ q : Idx k, ∀ u v : ℝ, |u - ω q| ≤ π → |v - ω q| ≤ π →
      1 - 2 * τ ≤ Real.cos (u - ω q) → 1 - 2 * τ ≤ Real.cos (v - ω q) →
        |u - v| ≤ (1 / 8) * Real.sqrt ((1 - δ) / (β + 1 / 2)))
    (hcaps : ∀ i : Idx n, 1 - 2 * τ ≤ Real.cos (Θ i - ω (p i))) :
    (∑ i : Idx n, (collapseTuple p (chartTuple ω p Θ) i - chartTuple ω p Θ i) ^ 2) ≤
      64 * (n : ℝ) ^ 5 * ∑ i : Idx n, (angularGrad n β Θ i) ^ 2 := by
  have hown (i : Idx n) : 1 - 2 * τ ≤ Real.cos (chartTuple ω p Θ i - ω (p i)) := by
    rw [chartTuple_cap_cos]
    exact hcaps i
  have h := collapsed_spread_le_gradient n k hn β τ δ α ω p (chartTuple ω p Θ)
    hβ hτ hτ16 hδ hδ1 hα hγ hsmall hown (chartTuple_own_chart ω p Θ)
  simpa only [angularGrad_chartTuple] using h

/-- The full hypotheses of both spread bounds hold for the separated three-token
configuration and its actual two-cap assignment. -/
example : let τ : ℝ := 1 / 10 ^ 8
    let α : ℝ := 1 - 2 * (1 - 2 * τ) ^ 2
    let ω : Idx 2 → ℝ := ![0, π]
    let Θ : Idx 3 → ℝ := ![0, 1 / 10 ^ 4, π]
    let p : Idx 3 → Idx 2 := ![0, 0, 1]
    0 < 3 ∧ (1 : ℝ) < 1000 ∧ 0 < τ ∧ τ < 1 / 16 ∧ (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) < 1 ∧
    (∀ q q' : Idx 2, q ≠ q' → ∀ u v : ℝ,
      1 - 2 * τ ≤ Real.cos (u - ω q) → 1 - 2 * τ ≤ Real.cos (v - ω q') →
        Real.cos (u - v) ≤ α) ∧
    0 < 1 - α - 8 * τ - (1000 : ℝ)⁻¹ * Real.log (2 * (3 : ℝ) ^ 2 / τ) ∧
    (∀ q : Idx 2, ∀ u v : ℝ, |u - ω q| ≤ π → |v - ω q| ≤ π →
      1 - 2 * τ ≤ Real.cos (u - ω q) → 1 - 2 * τ ≤ Real.cos (v - ω q) →
        |u - v| ≤ (1 / 8) * Real.sqrt ((1 - 1 / 2) / (1000 + 1 / 2))) ∧
    (∀ i : Idx 3, 1 - 2 * τ ≤ Real.cos (Θ i - ω (p i))) ∧
    (∀ i : Idx 3, |Θ i - ω (p i)| ≤ π) := by
  intro τ α ω Θ p
  obtain ⟨-, hα, hγ, hsmall, -⟩ := circle_witness
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    hα, hγ, hsmall, ?_, ?_⟩
  · intro i
    fin_cases i
    · norm_num [p, Θ, ω, τ]
    · have h := Real.one_sub_sq_div_two_le_cos (x := 1 / 10 ^ 4)
      norm_num [p, Θ, ω, τ] at h ⊢
      linarith
    · norm_num [p, Θ, ω, τ]
  · intro i
    fin_cases i <;> norm_num [p, Θ, ω] <;> linarith [Real.pi_gt_three]

end Transformer.Metastability
