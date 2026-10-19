module

public import Transformer.Metastability.OttoAttention
public import Transformer.Metastability.Section3_SlowProjection
public import Transformer.Metastability.Section3_SlowFlatnessCounterexample

/-!
# Metastability — the Otto energy comparison with an explicit cap domain

ArXiv:2410.06833v1, §3.2, `eq: otto.attention`, corrected by requiring
the projection to stay in the union of the original caps. The source's
unrestricted displayed slow manifold fails (H2), as proved in
`Section3_SlowFlatnessCounterexample`. The positive theorem below
states the added restriction explicitly and proves a stronger static
energy comparison for the actual normalized angular energy.
-/

@[expose] public section

open scoped BigOperators
open Real

namespace Transformer.Metastability

/-- **The source's second `λ` bound implies the interaction-rate bound.**
Source: arXiv:2410.06833v1, §2, `eq: lambda.3`, used in §3.2,
`lem: PL.borjan` and `eq: otto.attention`. The second exponential
is `8τ`, and the remaining logarithmic term is nonnegative.
Consequently `λ<1-α`, which implies `λ/2≤1-α` for `λ>0`. -/
theorem lambda_half_le_of_source_second_bound (n : ℕ) (hn : 0 < n)
    (β : ℝ) (hβ : 0 < β) (α τ lam : ℝ) (hτ : 0 < τ)
    (hτ16 : τ < 1 / 16) (hlam : 0 < lam)
    (hbound : lam < 1 - α - β⁻¹ * Real.log (2 * (n : ℝ) ^ 2
        / (1 - Real.exp (-(β⁻¹ * Real.log (1 / (8 * τ)) * β))))
      - Real.exp (-(β⁻¹ * Real.log (1 / (8 * τ)) * β))) :
    lam / 2 ≤ 1 - α := by
  have heq : Real.exp (-(β⁻¹ * Real.log (1 / (8 * τ)) * β)) = 8 * τ := by
    have hcancel : β⁻¹ * Real.log (1 / (8 * τ)) * β = Real.log (1 / (8 * τ)) := by
      field_simp
    rw [hcancel, Real.exp_neg, Real.exp_log (by positivity)]
    field_simp
  rw [heq] at hbound
  have hn1 : (1 : ℝ) ≤ n := by exact_mod_cast (show 1 ≤ n by omega)
  have hden : 0 < 1 - 8 * τ := by linarith
  have hfrac : 1 ≤ 2 * (n : ℝ) ^ 2 / (1 - 8 * τ) :=
    (le_div_iff₀ hden).mpr (by nlinarith)
  have hterm := mul_nonneg (inv_nonneg.mpr hβ.le) (Real.log_nonneg hfrac)
  linarith

/-- The entire numerical input of the `λ` comparison is realized by
the source-compatible three-token witness. The minimum bound in that
witness supplies its second component. -/
example : let τ : ℝ := 1 / 10 ^ 8
    let α : ℝ := 1 - 2 * (1 - 2 * τ) ^ 2
    0 < 3 ∧ (0 : ℝ) < 1000 ∧ 0 < τ ∧ τ < 1 / 16 ∧ (0 : ℝ) < 1 ∧
    (1 : ℝ) < 1 - α - (1000 : ℝ)⁻¹ * Real.log (2 * (3 : ℝ) ^ 2
      / (1 - Real.exp (-((1000 : ℝ)⁻¹ * Real.log (1 / (8 * τ)) * 1000))))
      - Real.exp (-((1000 : ℝ)⁻¹ * Real.log (1 / (8 * τ)) * 1000)) := by
  intro τ α
  obtain ⟨-, -, -, -, -, hboundWitness, -⟩ := circle_witness
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, one_pos,
    hboundWitness.trans_le (min_le_right _ _)⟩

/-- **Corrected energy part of Corollary `eq: otto.attention`.**

Source: arXiv:2410.06833v1, §3.1 (H1)--(H2), and §3.2,
`eq: otto.attention`.

The source says that the Otto conclusions hold for the angular flow
and the displayed `𝒩_β⊂𝕋^n`, with `δ=exp(-λβ/2)`. The earlier
formal statement retained only the square-root energy comparison.
That comparison is proved here with a missing domain hypothesis:
**every token of `V(t)` lies in one of the fixed `2τ` caps**.
Membership in the source's `𝒩_β` alone does not imply this, and (H2)
on that unrestricted set is false (`not_slow_manifold_energy_flatness`).
The unrestricted dynamical corollary is not asserted by this theorem.

The energy sign is `𝖤_β(V)-𝖤_β(U)`, since the angular attention flow
ascends `𝖤_β`. The norm in (H1) is Euclidean, written out as a sum
of squares. The original norm term omitted by the earlier formal
statement is not added to this energy-only theorem.

With the cap restriction, the lower half of (H1) alone gives
`sqrt(𝖤_β(V)-𝖤_β(U))≤3 exp(-λβ/2)` at every time. Thus the stated
comparison holds with `C_ε=3`, independent of all parameters and
even of `ε`. The flow equation, initialization, upper half of (H1),
and small-cap diameter assumptions are unnecessary and are omitted.
The source's bound on `λ` implies the weaker rate inequality used
here, by `lambda_half_le_of_source_second_bound`.

This is a correction of the projection domain followed by a stronger
pointwise estimate on that domain. The actual energy, gradient,
and displayed slow-manifold definition are unchanged. -/
theorem otto_attention_in_caps (ε : ℝ) :
    ∃ Cε : ℝ, 0 < Cε ∧
      ∀ (n : ℕ) (α β τ lam : ℝ) (k : ℕ) (ω : Idx k → ℝ),
        2 ≤ n → 0 < β → lam / 2 ≤ 1 - α →
        (∀ q q' : Idx k, q ≠ q' → ∀ u v : ℝ,
          1 - 2 * τ ≤ Real.cos (u - ω q) → 1 - 2 * τ ≤ Real.cos (v - ω q') →
            Real.cos (u - v) ≤ α) →
        ∀ U V : ℝ → Idx n → ℝ,
          (∀ t : ℝ, V t ∈ slowManifold n β τ lam k ω ∧
            (∀ i : Idx n, ∃ q : Idx k, 1 - 2 * τ ≤ Real.cos (V t i - ω q)) ∧
            (1 / 2 : ℝ) * ∑ i : Idx n, (U t i - V t i) ^ 2
              ≤ angularEβ n β (V t) - angularEβ n β (U t)) →
          ∀ t : ℝ,
            Real.sqrt (angularEβ n β (V t) - angularEβ n β (U t)) ≤
              Real.exp (-(1 - ε) * t)
                  * Real.sqrt (angularEβ n β (V 0) - angularEβ n β (U 0))
                + Cε * Real.exp (-(lam * β / 2)) := by
  refine ⟨3, by norm_num, ?_⟩
  intro n α β τ lam k ω hn hβ hrate hα U V hUV t
  obtain ⟨hN, hcaps, hproj⟩ := hUV t
  have h := angularEβ_sqrt_gain_le_at_slow_point n hn k β hβ α τ lam hrate
    ω hα (U t) (V t) hcaps hN hproj
  have hinit := mul_nonneg (Real.exp_pos (-(1 - ε) * t)).le
    (Real.sqrt_nonneg (angularEβ n β (V 0) - angularEβ n β (U 0)))
  linarith

/-- All inputs of the corrected comparison are simultaneously
realized at the source's numerical parameters and two antipodal caps.
Three tokens are placed at the cap centres by `p=(0,0,1)`, and
`U=V` is constant at that actual slow configuration. In particular,
the example includes the full projection and containment hypotheses,
which the earlier configuration-only example did not witness. -/
example : let τ : ℝ := 1 / 10 ^ 8
    let α : ℝ := 1 - 2 * (1 - 2 * τ) ^ 2
    let ω : Idx 2 → ℝ := ![0, π]
    let p : Idx 3 → Idx 2 := ![0, 0, 1]
    let Z : Idx 3 → ℝ := fun i => ω (p i)
    2 ≤ 3 ∧ (0 : ℝ) < 1000 ∧ (1 / 2 : ℝ) ≤ 1 - α ∧
    (∀ q q' : Idx 2, q ≠ q' → ∀ u v : ℝ,
      1 - 2 * τ ≤ Real.cos (u - ω q) → 1 - 2 * τ ≤ Real.cos (v - ω q') →
        Real.cos (u - v) ≤ α) ∧
    ∀ t : ℝ, (fun _ : ℝ => Z) t ∈ slowManifold 3 1000 τ 1 2 ω ∧
      (∀ i : Idx 3, ∃ q : Idx 2, 1 - 2 * τ ≤ Real.cos ((fun _ : ℝ => Z) t i - ω q)) ∧
      (1 / 2 : ℝ) * ∑ i : Idx 3,
        ((fun _ : ℝ => Z) t i - (fun _ : ℝ => Z) t i) ^ 2 ≤
        angularEβ 3 1000 ((fun _ : ℝ => Z) t) - angularEβ 3 1000 ((fun _ : ℝ => Z) t) := by
  intro τ α ω p Z
  obtain ⟨-, hα, -⟩ := circle_witness
  have hown (i : Idx 3) : 1 - 2 * τ ≤ Real.cos (Z i - ω (p i)) := by
    change 1 - 2 * τ ≤ Real.cos (ω (p i) - ω (p i))
    norm_num [τ]
  have hcollapse : collapseTuple p Z = Z := by
    funext i
    change ω (p (capRepresentative p i)) = ω (p i)
    rw [capRepresentative_same_cap p i]
  have hN := collapseTuple_mem_slowManifold 1000 τ 1 α ω p Z
    (by norm_num [α, τ]) hα hown
  rw [hcollapse] at hN
  exact ⟨by norm_num, by norm_num, by norm_num [α, τ], hα,
    fun _ => ⟨hN, fun i => ⟨p i, hown i⟩, by norm_num⟩⟩

end Transformer.Metastability
