/-
# A proved, uniform PL inequality for the angular interaction energy

ArXiv:2410.06833v1, §3.2, `lem: PL.borjan`.
The configuration is recharted without changing its true energy or gradient.
Each assigned cap is then collapsed to one of its actual token angles.
The uniform force slope controls the displacement, and the quadratic
bound on the actual energy controls the energy gain.
The resulting coefficient depends only on temperature and token count.

The bound is constructive at the level of configurations: the point of
the slow manifold is the explicit `collapseTuple` of the charted tokens.
The energy cost is proved from the pair interaction, rather than assumed
of that selected point. Its coefficient is fixed before the configuration.
All norm squares are Euclidean coordinate sums.
-/

import Transformer.Metastability.Section3_EnergyTaylor

open scoped BigOperators
open Set Real

namespace Transformer.Metastability

/-- A coarse explicit coefficient for the source's PL inequality.
It is chosen before the cap radius, slow threshold, cap centers, or
configuration. Source: arXiv:2410.06833v1, §3.2, `κ(β,n)` in `lem: PL.borjan`. -/
noncomputable def plCoefficient (n : ℕ) (β : ℝ) : ℝ :=
  1 / (1024 * (n : ℝ) ^ 5 * (1 + β))

/-- The displayed coefficient is strictly positive in the source's range.
Source: arXiv:2410.06833v1, §3.2, the positive `κ(β,n)` of `lem: PL.borjan`. -/
theorem plCoefficient_pos (n : ℕ) (hn : 0 < n) (β : ℝ) (hβ : 0 < β) :
    0 < plCoefficient n β := by
  have hnR : (0 : ℝ) < n := by exact_mod_cast hn
  unfold plCoefficient
  positivity

/-- Its inverse matches the proved uniform energy-gain coefficient.
Source: arXiv:2410.06833v1, §3.2, `1/(2κ(β,n))` in `lem: PL.borjan`. -/
theorem one_div_two_plCoefficient (n : ℕ) (hn : 0 < n) (β : ℝ) (hβ : 0 < β) :
    1 / (2 * plCoefficient n β) = 512 * (n : ℝ) ^ 5 * (1 + β) := by
  have hnR : (0 : ℝ) < n := by exact_mod_cast hn
  unfold plCoefficient
  field_simp
  ring

/-- All coefficient inputs are realized by two tokens at temperature `1`. -/
example : 0 < 2 ∧ (0 : ℝ) < 1 := ⟨by norm_num, one_pos⟩

/-- The source's exponential lower bound on `δ` implies strict positivity,
so it implies the nonnegative input of the proved PL inequality below.
Source: arXiv:2410.06833v1, §3.2, the range of `δ` in `lem: PL.borjan`. -/
theorem delta_pos_of_source_PL_range (β α δ : ℝ) (hβ : 1 < β)
    (hδ : 8 * (1 + β) * Real.exp (-((1 - α) * β)) * Real.exp (-(1 / 2 : ℝ)) < δ) :
    0 < δ := by
  have hp : 0 < 8 * (1 + β) * Real.exp (-((1 - α) * β)) * Real.exp (-(1 / 2 : ℝ)) :=
    mul_pos (mul_pos (mul_pos (by norm_num) (by linarith)) (Real.exp_pos _)) (Real.exp_pos _)
  exact hp.trans hδ

/-- Both source-range inputs are realized by the genuine three-token witness. -/
example : let τ : ℝ := 1 / 10 ^ 8
    let α : ℝ := 1 - 2 * (1 - 2 * τ) ^ 2
    (1 : ℝ) < 1000 ∧
    8 * (1 + (1000 : ℝ)) * Real.exp (-((1 - α) * 1000)) * Real.exp (-(1 / 2 : ℝ)) < 1 / 2 := by
  intro τ α
  obtain ⟨-, -, -, -, hδ, -⟩ := circle_witness
  exact ⟨by norm_num, hδ⟩

/-- **Lemma (lem: PL.borjan): a uniform PL inequality for the true angular energy.**
For `β>1`, `n≥2`, and a separated small-cap configuration `Θ`, there are
`U∈𝒩_β` and `κ(β,n)>0` such that
`Eβ(U)-Eβ(Θ) ≤ (1/(2κ(β,n)))∑ᵢ(∂ᵢEβ(Θ))²`.

Here `κ` is the explicit `plCoefficient n β`, independent even of `δ` and
`α`, and chosen before `Θ`, the caps, and their radius. Every displacement
and gradient in the proof is for the actual exponential interaction.

**What the source assumes and what the proof needs.** The manuscript
requires `k≤n`, a positive `λ` in the range `eq: lambda.1`--`eq: lambda.2`,
and `8(1+β)e^{-(1-α)β}e^{-1/2}<δ<1`. The construction here has identical
angles for every two tokens in a cap. Thus no bound on `k` or `λ` is
needed, and every real `λ` works. The source's lower bound on `δ` implies
`0≤δ`; only that weaker input and `δ<1` enter the uniform cap slope.
These are stronger versions of the printed conclusion, with fewer
hypotheses; the actual inequality and its normalization are retained.

The torus uses real representatives. The diameter condition is asserted
in each cap chart, within `π` of its center. Recharting preserves energy
and gradient, and all caps remain uniquely assigned by `γ>0`.
The squared norm is Euclidean, written as the coordinate-square sum.

Source: arXiv:2410.06833v1, §3.2, `lem: PL.borjan`, `eq: tau.small`,
`eq: alpha.dist.2`, and `d: condition_ineq_2`. -/
theorem PL_borjan (n : ℕ) (β δ α : ℝ) (hβ : 1 < β) (hn : 2 ≤ n) :
    ∃ κ : ℝ, 0 < κ ∧ ∀ (τ lam : ℝ) (k : ℕ) (ω : Idx k → ℝ) (Θ : Idx n → ℝ),
      0 < τ → τ < 1 / 16 →
      (∀ i : Idx n, ∃ q : Idx k, 1 - τ ≤ Real.cos (Θ i - ω q)) →
      (∀ p p' : Idx k, p ≠ p' → ∀ u v : ℝ,
        1 - 2 * τ ≤ Real.cos (u - ω p) → 1 - 2 * τ ≤ Real.cos (v - ω p') →
          Real.cos (u - v) ≤ α) →
      0 < 1 - α - 8 * τ - β⁻¹ * Real.log (2 * (n : ℝ) ^ 2 / τ) →
      (∀ q : Idx k, ∀ u v : ℝ, |u - ω q| ≤ π → |v - ω q| ≤ π →
        1 - 2 * τ ≤ Real.cos (u - ω q) → 1 - 2 * τ ≤ Real.cos (v - ω q) →
          |u - v| ≤ (1 / 8) * Real.sqrt ((1 - δ) / (β + 1 / 2))) →
      0 ≤ δ → δ < 1 →
      ∃ U : Idx n → ℝ, U ∈ slowManifold n β τ lam k ω ∧
        angularEβ n β U - angularEβ n β Θ ≤
          (1 / (2 * κ)) * ∑ i : Idx n, (angularGrad n β Θ i) ^ 2 := by
  have hn0 : 0 < n := by omega
  have hβ0 : 0 < β := by linarith
  refine ⟨plCoefficient n β, plCoefficient_pos n hn0 β hβ0, ?_⟩
  intro τ lam k ω Θ hτ hτ16 hcover hα hγ hsmall hδ hδ1
  let p : Idx n → Idx k := fun i => Classical.choose (hcover i)
  have hp (i : Idx n) : 1 - 2 * τ ≤ Real.cos (Θ i - ω (p i)) := by
    have h := Classical.choose_spec (hcover i)
    dsimp only [p]
    linarith
  have hα1 := alpha_lt_one_of_gamma n hn0 β τ α hβ0 hτ hτ16 hγ
  have hown (i : Idx n) : 1 - 2 * τ ≤ Real.cos (chartTuple ω p Θ i - ω (p i)) := by
    rw [chartTuple_cap_cos]
    exact hp i
  refine ⟨collapseTuple p (chartTuple ω p Θ),
    collapseTuple_mem_slowManifold β τ lam α ω p (chartTuple ω p Θ) hα1 hα hown, ?_⟩
  rw [one_div_two_plCoefficient n hn0 β hβ0]
  exact collapsed_angularEβ_gain_le_gradient n k hn0 β τ δ α ω p Θ
    hβ hτ hτ16 hδ hδ1 hα hγ hsmall hp

/-- Every PL hypothesis is jointly realized by the source-compatible
three-token configuration at `β=1000`, `δ=1/2`, and two antipodal caps.
The slow threshold may be chosen arbitrarily; for example `λ=1`. -/
example : let τ : ℝ := 1 / 10 ^ 8
    let α : ℝ := 1 - 2 * (1 - 2 * τ) ^ 2
    let ω : Idx 2 → ℝ := ![0, π]
    let Θ : Idx 3 → ℝ := ![0, 1 / 10 ^ 4, π]
    (1 : ℝ) < 1000 ∧ 2 ≤ 3 ∧ 0 < τ ∧ τ < 1 / 16 ∧
    (∀ i : Idx 3, ∃ q : Idx 2, 1 - τ ≤ Real.cos (Θ i - ω q)) ∧
    (∀ p p' : Idx 2, p ≠ p' → ∀ u v : ℝ,
      1 - 2 * τ ≤ Real.cos (u - ω p) → 1 - 2 * τ ≤ Real.cos (v - ω p') →
        Real.cos (u - v) ≤ α) ∧
    0 < 1 - α - 8 * τ - (1000 : ℝ)⁻¹ * Real.log (2 * (3 : ℝ) ^ 2 / τ) ∧
    (∀ q : Idx 2, ∀ u v : ℝ, |u - ω q| ≤ π → |v - ω q| ≤ π →
      1 - 2 * τ ≤ Real.cos (u - ω q) → 1 - 2 * τ ≤ Real.cos (v - ω q) →
        |u - v| ≤ (1 / 8) * Real.sqrt ((1 - 1 / 2) / (1000 + 1 / 2))) ∧
    (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) < 1 := by
  intro τ α ω Θ
  obtain ⟨-, hα, hγ, hsmall, -, -, -, -, -, -, hcover⟩ := circle_witness
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, hcover,
    hα, hγ, hsmall, by norm_num, by norm_num⟩

end Transformer.Metastability
