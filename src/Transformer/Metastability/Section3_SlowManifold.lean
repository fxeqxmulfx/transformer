module

public import Transformer.Metastability.Section3_AngularCharts
public import Transformer.Metastability.CircleWitness

/-!
# An explicit collapsed configuration in the slow manifold

ArXiv:2410.06833v1, §3.2, `lem: PL.borjan`, `eq: cones_2`, and
`eq: alpha.dist.2`. Positive `γ` makes the cap cosine bound strictly
smaller than one, so cap membership is unique. For each nonempty assigned
cap, choose one actual token index and collapse its assigned tokens to
that token's angle. Every within-cap gap of this configuration is zero.
-/

@[expose] public section

open Set Real

namespace Transformer.Metastability

/-- The source's slow manifold: every two tokens in the same `2τ` cap
have gap at most `exp(-λβ/2)`. Source: arXiv:2410.06833v1, §3.2,
`lem: PL.borjan`, the displayed definition of `𝒩_β`. -/
def slowManifold (n : ℕ) (β τ lam : ℝ) (k : ℕ) (ω : Idx k → ℝ) : Set (Idx n → ℝ) :=
  {θ | ∀ q : Idx k, ∀ i j : Idx n,
    (1 - 2 * τ ≤ Real.cos (θ i - ω q)) →
    (1 - 2 * τ ≤ Real.cos (θ j - ω q)) →
    |θ i - θ j| ≤ Real.exp (-(lam * β / 2))}

/-- The kernel `g` of the source's Hessian formula.
Source: arXiv:2410.06833v1, §3.2, before `eq: Ht.second.lb`. -/
noncomputable def g_OR (β x : ℝ) : ℝ :=
  (Real.cos x - β * Real.sin x ^ 2) * Real.exp (β * (Real.cos x - 1))

/-- **Positive separation forces `α<1`.** Source: arXiv:2410.06833v1,
§3.2, `d: condition_ineq_2`. The logarithmic term is nonnegative here. -/
theorem alpha_lt_one_of_gamma (n : ℕ) (hn : 0 < n) (β τ α : ℝ)
    (hβ : 0 < β) (hτ : 0 < τ) (hτ16 : τ < 1 / 16)
    (hγ : 0 < 1 - α - 8 * τ - β⁻¹ * Real.log (2 * (n : ℝ) ^ 2 / τ)) : α < 1 := by
  have hn1 : (1 : ℝ) ≤ n := by exact_mod_cast (show 1 ≤ n by omega)
  have hA : 1 ≤ 2 * (n : ℝ) ^ 2 / τ := (le_div_iff₀ hτ).mpr (by nlinarith)
  have hlog := Real.log_nonneg hA
  have hterm := mul_nonneg (inv_nonneg.mpr hβ.le) hlog
  linarith

/-- **Separated caps have unique membership.** Source: arXiv:2410.06833v1,
§3.2, `eq: alpha.dist.2`: an intersection would give cosine `1≤α`. -/
theorem cos_cap_unique {k : ℕ} (τ α : ℝ) (ω : Idx k → ℝ) (hα1 : α < 1)
    (hα : ∀ p p' : Idx k, p ≠ p' → ∀ u v : ℝ,
      1 - 2 * τ ≤ Real.cos (u - ω p) → 1 - 2 * τ ≤ Real.cos (v - ω p') →
        Real.cos (u - v) ≤ α) (u : ℝ) (q q' : Idx k)
    (hq : 1 - 2 * τ ≤ Real.cos (u - ω q))
    (hq' : 1 - 2 * τ ≤ Real.cos (u - ω q')) : q = q' := by
  by_contra hne
  have h := hα q q' hne u u hq hq'
  rw [sub_self, Real.cos_zero] at h
  linarith

/-- One actual index from the assigned cap of token `i`. A cap used by a
token is nonempty, witnessed by that same token; no existence assumption
is hidden in this selection. Source: arXiv:2410.06833v1, §3.2,
the construction of a point of `𝒩_β` in `lem: PL.borjan`. -/
noncomputable def capRepresentative {n k : ℕ} (p : Idx n → Idx k) (i : Idx n) : Idx n :=
  Classical.choose (show ∃ j : Idx n, p j = p i from ⟨i, rfl⟩)

/-- The selected index belongs to the assigned cap.
Source: arXiv:2410.06833v1, §3.2, the same slow-manifold construction. -/
theorem capRepresentative_same_cap {n k : ℕ} (p : Idx n → Idx k) (i : Idx n) :
    p (capRepresentative p i) = p i :=
  Classical.choose_spec (show ∃ j : Idx n, p j = p i from ⟨i, rfl⟩)

/-- The representative depends on the assigned cap, so two tokens with
the same assignment choose the same actual index.
Source: arXiv:2410.06833v1, §3.2, the construction in `lem: PL.borjan`. -/
theorem capRepresentative_eq_of_same {n k : ℕ} (p : Idx n → Idx k)
    (i j : Idx n) (h : p i = p j) : capRepresentative p i = capRepresentative p j := by
  unfold capRepresentative
  congr 1
  funext x
  rw [h]

/-- Collapse each assigned cap to the angle of its selected actual token.
Source: arXiv:2410.06833v1, §3.2, the slow manifold in `lem: PL.borjan`. -/
noncomputable def collapseTuple {n k : ℕ} (p : Idx n → Idx k)
    (Θ : Idx n → ℝ) : Idx n → ℝ := fun i => Θ (capRepresentative p i)

/-- Equal cap assignments give identical collapsed angles.
Source: arXiv:2410.06833v1, §3.2, the zero within-cap gaps in `𝒩_β`. -/
theorem collapseTuple_eq_of_same {n k : ℕ} (p : Idx n → Idx k) (Θ : Idx n → ℝ)
    (i j : Idx n) (h : p i = p j) : collapseTuple p Θ i = collapseTuple p Θ j :=
  congrArg Θ (capRepresentative_eq_of_same p i j h)

/-- All assignment-equality inputs are satisfied by a nonempty two-token
cap assignment, with distinct actual token indices. -/
example : (fun _ : Idx 2 => (0 : Idx 1)) 0 = (fun _ : Idx 2 => (0 : Idx 1)) 1 := rfl

/-- Collapsing retains membership in every token's assigned cap.
Source: arXiv:2410.06833v1, §3.2, `eq: cones_2`, `lem: PL.borjan`. -/
theorem collapseTuple_own_cap {n k : ℕ} (τ : ℝ) (ω : Idx k → ℝ)
    (p : Idx n → Idx k) (Θ : Idx n → ℝ)
    (hcaps : ∀ i : Idx n, 1 - 2 * τ ≤ Real.cos (Θ i - ω (p i))) (i : Idx n) :
    1 - 2 * τ ≤ Real.cos (collapseTuple p Θ i - ω (p i)) := by
  have h := hcaps (capRepresentative p i)
  rw [capRepresentative_same_cap p i] at h
  exact h

/-- **The collapsed configuration lies in the source's slow manifold.**
Cap uniqueness and identical angles within each assigned cap give zero
gaps, so this holds for every value of `λ`.
Source: arXiv:2410.06833v1, §3.2, the definition of `𝒩_β` in `lem: PL.borjan`. -/
theorem collapseTuple_mem_slowManifold {n k : ℕ} (β τ lam α : ℝ) (ω : Idx k → ℝ)
    (p : Idx n → Idx k) (Θ : Idx n → ℝ) (hα1 : α < 1)
    (hα : ∀ q q' : Idx k, q ≠ q' → ∀ u v : ℝ,
      1 - 2 * τ ≤ Real.cos (u - ω q) → 1 - 2 * τ ≤ Real.cos (v - ω q') →
        Real.cos (u - v) ≤ α)
    (hcaps : ∀ i : Idx n, 1 - 2 * τ ≤ Real.cos (Θ i - ω (p i))) :
    collapseTuple p Θ ∈ slowManifold n β τ lam k ω := by
  intro q i j hi hj
  have hqi := cos_cap_unique τ α ω hα1 hα (collapseTuple p Θ i) q (p i) hi
    (collapseTuple_own_cap τ ω p Θ hcaps i)
  have hqj := cos_cap_unique τ α ω hα1 hα (collapseTuple p Θ j) q (p j) hj
    (collapseTuple_own_cap τ ω p Θ hcaps j)
  rw [collapseTuple_eq_of_same p Θ i j (hqi.symm.trans hqj), sub_self, abs_zero]
  exact (Real.exp_pos _).le

/-- The separation parameter hypotheses are satisfied by the full
three-token witness used in the source's cap estimates. -/
example : let τ : ℝ := 1 / 10 ^ 8
    let α : ℝ := 1 - 2 * (1 - 2 * τ) ^ 2
    0 < 3 ∧ (0 : ℝ) < 1000 ∧ 0 < τ ∧ τ < 1 / 16 ∧
    0 < 1 - α - 8 * τ - (1000 : ℝ)⁻¹ * Real.log (2 * (3 : ℝ) ^ 2 / τ) := by
  intro τ α
  obtain ⟨-, -, hγ, -⟩ := circle_witness
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, hγ⟩

/-- All cap-uniqueness and collapsed-membership inputs hold with the
two antipodal caps and the source's genuine three-token configuration. -/
example : let τ : ℝ := 1 / 10 ^ 8
    let α : ℝ := 1 - 2 * (1 - 2 * τ) ^ 2
    let ω : Idx 2 → ℝ := ![0, π]
    let Θ : Idx 3 → ℝ := ![0, 1 / 10 ^ 4, π]
    let p : Idx 3 → Idx 2 := ![0, 0, 1]
    α < 1 ∧
    (∀ q q' : Idx 2, q ≠ q' → ∀ u v : ℝ,
      1 - 2 * τ ≤ Real.cos (u - ω q) → 1 - 2 * τ ≤ Real.cos (v - ω q') →
        Real.cos (u - v) ≤ α) ∧
    (∀ i : Idx 3, 1 - 2 * τ ≤ Real.cos (Θ i - ω (p i))) := by
  intro τ α ω Θ p
  obtain ⟨-, hα, -⟩ := circle_witness
  refine ⟨by norm_num [α, τ], hα, fun i => ?_⟩
  fin_cases i
  · norm_num [p, Θ, ω, τ]
  · have h := Real.one_sub_sq_div_two_le_cos (x := 1 / 10 ^ 4)
    norm_num [p, Θ, ω, τ] at h ⊢
    linarith
  · norm_num [p, Θ, ω, τ]

end Transformer.Metastability
