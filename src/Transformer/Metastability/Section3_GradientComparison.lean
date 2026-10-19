module

public import Transformer.Metastability.Section3_ForceMonotonicity
public import Mathlib.Analysis.Complex.ExponentialBounds

/-!
# The ordered-cluster gradient comparison

ArXiv:2410.06833v1, §3.2, `claim: 1`, `eq: Ht.third.lb`.
The actual scalar force is antitone on a separated small cap. Thus every
component is between its endpoint values, and its absolute value is
bounded by the larger endpoint absolute value. This gives the source's
`e/2` comparison without its unsupported final absorption step.
-/

@[expose] public section

open Set Real

namespace Transformer.Metastability

/-- An antitone scalar function has no absolute-value maximum above its
two endpoint values. This is the ordered-interval step in the proof of
arXiv:2410.06833v1, §3.2, `claim: 1`, `eq: Ht.third.lb`. -/
theorem abs_le_max_of_antitone (f : ℝ → ℝ) (a b x : ℝ)
    (hf : AntitoneOn f (Icc a b)) (hx : x ∈ Icc a b) :
    |f x| ≤ max |f a| |f b| := by
  have hab := hx.1.trans hx.2
  have hleft := hf ⟨le_rfl, hab⟩ hx hx.1
  have hright := hf hx ⟨hab, le_rfl⟩ hx.2
  apply abs_le.mpr
  constructor
  · linarith [neg_abs_le (f b), le_max_right |f a| |f b|]
  · exact hleft.trans ((le_abs_self (f a)).trans (le_max_left _ _))

/-- Both interval inputs are realized by `f(t)=-t`, at an interior point. -/
example : AntitoneOn (fun t : ℝ => -t) (Icc 0 1) ∧ (1 / 2 : ℝ) ∈ Icc 0 1 := by
  refine ⟨fun a _ b _ hab => ?_, by norm_num⟩
  change -b ≤ -a
  linarith

/-- **The stronger ordered-cluster gradient comparison.**
For an ordered cluster in a separated `2τ` cap,
`|∂_{θ_l} Eβ| ≤ max(|∂_{θ_first} Eβ|, |∂_{θ_last} Eβ|)`.

The proof gives the stronger bound with factor `1`: every component is
the value of the same antitone force at that token's angle. The source's
`eq: tau.small` controls `τ(1+β)`, and `γ>0` controls the sum of all
cross-cap derivatives, so a within-cap term dominates them.

The source assumes `0<r<n`, `k≤n`, its positive lower bound on `δ`, and
an off-slow-manifold gap with the bounds on `λ`. None of `r<n`, `k≤n`,
the gap, or `λ` is needed for this stronger interval argument. Its bound
on `δ` implies the weaker input `0≤δ` used here. The source's strict
ordering is kept. Chart representatives lie within `π` of their cap
center, and cap membership identifies all tokens in the ordered cluster.
No assertion about a maximum-gradient particle lying off the slow
manifold is used.

Source: arXiv:2410.06833v1, §3.2, `claim: 1`, `eq: Ht.third.lb`. -/
theorem ordered_cap_gradient_le_endpoints (n : ℕ) (β τ δ α : ℝ) (k : ℕ)
    (ω : Idx k → ℝ) (Θ : Idx n → ℝ)
    (hβ : 1 < β) (hτ : 0 < τ) (hτ16 : τ < 1 / 16) (hδ : 0 ≤ δ) (hδ1 : δ < 1)
    (hcaps : ∀ i : Idx n, ∃ p : Idx k, 1 - 2 * τ ≤ Real.cos (Θ i - ω p))
    (hα : ∀ p p' : Idx k, p ≠ p' → ∀ u v : ℝ,
      1 - 2 * τ ≤ Real.cos (u - ω p) → 1 - 2 * τ ≤ Real.cos (v - ω p') →
        Real.cos (u - v) ≤ α)
    (hγ : 0 < 1 - α - 8 * τ - β⁻¹ * Real.log (2 * (n : ℝ) ^ 2 / τ))
    (hsmall : ∀ p : Idx k, ∀ u v : ℝ, |u - ω p| ≤ π → |v - ω p| ≤ π →
      1 - 2 * τ ≤ Real.cos (u - ω p) → 1 - 2 * τ ≤ Real.cos (v - ω p) →
        |u - v| ≤ (1 / 8) * Real.sqrt ((1 - δ) / (β + 1 / 2)))
    (q : Idx k) (r : ℕ) (idx : Fin r → Idx n) (hr : 0 < r)
    (hmono : StrictMono (Θ ∘ idx))
    (hchart : ∀ j : Fin r, |Θ (idx j) - ω q| ≤ π)
    (hmem : ∀ i : Idx n, 1 - 2 * τ ≤ Real.cos (Θ i - ω q) ↔ i ∈ Set.range idx) :
    ∀ l : Fin r, |angularGrad n β Θ (idx l)|
      ≤ max |angularGrad n β Θ (idx ⟨0, hr⟩)|
          |angularGrad n β Θ (idx ⟨r - 1, by omega⟩)| := by
  let first : Fin r := ⟨0, hr⟩
  let last : Fin r := ⟨r - 1, by omega⟩
  have hcmem (j : Fin r) : 1 - 2 * τ ≤ Real.cos (Θ (idx j) - ω q) :=
    (hmem (idx j)).mpr ⟨j, rfl⟩
  have hchartAll (i : Idx n) (hi : 1 - 2 * τ ≤ Real.cos (Θ i - ω q)) : |Θ i - ω q| ≤ π := by
    obtain ⟨j, rfl⟩ := (hmem i).mp hi
    exact hchart j
  have hforce := angularForce_antitoneOn_cap n k β τ δ α ω Θ q (idx first)
    hβ hτ hτ16 hδ hδ1 hcaps hα hγ (hsmall q) hchartAll (hcmem first)
  have hcap := interval_mem_cos_cap τ (ω q) (Θ (idx first)) (Θ (idx last))
    (hchart first) (hchart last) (hcmem first) (hcmem last)
  have hanti : AntitoneOn (angularForce n β Θ) (Icc (Θ (idx first)) (Θ (idx last))) :=
    fun a ha b hb hab => hforce (hcap a ha) (hcap b hb) hab
  intro l
  have hx : Θ (idx l) ∈ Icc (Θ (idx first)) (Θ (idx last)) := by
    constructor
    · exact hmono.monotone (by change (0 : ℕ) ≤ (l : ℕ); omega)
    · exact hmono.monotone (by change (l : ℕ) ≤ r - 1; omega)
  have hbound := abs_le_max_of_antitone (angularForce n β Θ) _ _ _ hanti hx
  simp only [angularForce_at_token] at hbound
  exact hbound

/-- **Claim (claim: 1), equation (eq: Ht.third.lb).**
The manuscript's factor `e/2` follows from the stronger factor `1`
comparison above. The signature keeps precisely the static hypotheses
used there; the original off-manifold and `λ` assumptions are unnecessary.
Source: arXiv:2410.06833v1, §3.2, `claim: 1`, `eq: Ht.third.lb`. -/
theorem claim_one (n : ℕ) (β τ δ α : ℝ) (k : ℕ) (ω : Idx k → ℝ) (Θ : Idx n → ℝ)
    (hβ : 1 < β) (hτ : 0 < τ) (hτ16 : τ < 1 / 16) (hδ : 0 ≤ δ) (hδ1 : δ < 1)
    (hcaps : ∀ i : Idx n, ∃ p : Idx k, 1 - 2 * τ ≤ Real.cos (Θ i - ω p))
    (hα : ∀ p p' : Idx k, p ≠ p' → ∀ u v : ℝ,
      1 - 2 * τ ≤ Real.cos (u - ω p) → 1 - 2 * τ ≤ Real.cos (v - ω p') →
        Real.cos (u - v) ≤ α)
    (hγ : 0 < 1 - α - 8 * τ - β⁻¹ * Real.log (2 * (n : ℝ) ^ 2 / τ))
    (hsmall : ∀ p : Idx k, ∀ u v : ℝ, |u - ω p| ≤ π → |v - ω p| ≤ π →
      1 - 2 * τ ≤ Real.cos (u - ω p) → 1 - 2 * τ ≤ Real.cos (v - ω p) →
        |u - v| ≤ (1 / 8) * Real.sqrt ((1 - δ) / (β + 1 / 2)))
    (q : Idx k) (r : ℕ) (idx : Fin r → Idx n) (hr : 0 < r)
    (hmono : StrictMono (Θ ∘ idx))
    (hchart : ∀ j : Fin r, |Θ (idx j) - ω q| ≤ π)
    (hmem : ∀ i : Idx n, 1 - 2 * τ ≤ Real.cos (Θ i - ω q) ↔ i ∈ Set.range idx) :
    ∀ l : Fin r, |angularGrad n β Θ (idx l)|
      ≤ (Real.exp 1 / 2) * max |angularGrad n β Θ (idx ⟨0, hr⟩)|
          |angularGrad n β Θ (idx ⟨r - 1, by omega⟩)| := by
  intro l
  have hbound := ordered_cap_gradient_le_endpoints n β τ δ α k ω Θ hβ hτ hτ16 hδ hδ1
    hcaps hα hγ hsmall q r idx hr hmono hchart hmem l
  have hfac : 1 ≤ Real.exp 1 / 2 := by linarith [Real.exp_one_gt_two]
  have hmax : 0 ≤ max |angularGrad n β Θ (idx ⟨0, hr⟩)|
      |angularGrad n β Θ (idx ⟨r - 1, by omega⟩)| :=
    (abs_nonneg _).trans (le_max_left _ _)
  exact hbound.trans (by simpa only [one_mul] using mul_le_mul_of_nonneg_right hfac hmax)

/-- All hypotheses of both gradient comparisons hold at the three-token witness.
In particular the cluster has two distinct angles, so the comparison is
realized beyond the singleton case. -/
example : let τ : ℝ := 1 / 10 ^ 8
    let α : ℝ := 1 - 2 * (1 - 2 * τ) ^ 2
    let ω : Idx 2 → ℝ := ![0, π]
    let Θ : Idx 3 → ℝ := ![0, 1 / 10 ^ 4, π]
    let idx : Fin 2 → Idx 3 := ![0, 1]
    (1 : ℝ) < 1000 ∧ 0 < τ ∧ τ < 1 / 16 ∧ (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) < 1 ∧
    (∀ i : Idx 3, ∃ p : Idx 2, 1 - 2 * τ ≤ Real.cos (Θ i - ω p)) ∧
    (∀ p p' : Idx 2, p ≠ p' → ∀ u v : ℝ,
      1 - 2 * τ ≤ Real.cos (u - ω p) → 1 - 2 * τ ≤ Real.cos (v - ω p') →
        Real.cos (u - v) ≤ α) ∧
    0 < 1 - α - 8 * τ - (1000 : ℝ)⁻¹ * Real.log (2 * (3 : ℝ) ^ 2 / τ) ∧
    (∀ p : Idx 2, ∀ u v : ℝ, |u - ω p| ≤ π → |v - ω p| ≤ π →
      1 - 2 * τ ≤ Real.cos (u - ω p) → 1 - 2 * τ ≤ Real.cos (v - ω p) →
        |u - v| ≤ (1 / 8) * Real.sqrt ((1 - 1 / 2) / (1000 + 1 / 2))) ∧
    0 < 2 ∧ StrictMono (Θ ∘ idx) ∧
    (∀ j : Fin 2, |Θ (idx j) - ω 0| ≤ π) ∧
    (∀ i : Idx 3, 1 - 2 * τ ≤ Real.cos (Θ i - ω 0) ↔ i ∈ Set.range idx) := by
  intro τ α ω Θ idx
  obtain ⟨hcaps, hα, hγ, hsmall, -, -, hmono, hchart, hmem, -⟩ := circle_witness
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    hcaps, hα, hγ, hsmall, by norm_num, hmono, hchart, hmem⟩

end Transformer.Metastability
