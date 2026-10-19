module

public import Transformer.Metastability.Section3_SlowManifold
public import Transformer.Causal.Interaction

/-!
# Small gradients on the part of the slow manifold inside the caps

ArXiv:2410.06833v1, §3.1 (H2), and §3.2, `lem: PL.borjan` and
`eq: otto.attention`. The displayed definition of `𝒩_β` only controls
pairs lying in the same cap. An additional assertion that every token
lies in one of the fixed caps is therefore used explicitly here.

Pairs in the same cap have small angular distance; pairs in different
caps have exponentially small interactions. Their actual normalized
gradient consequently has Euclidean squared norm at most `exp(-λβ)`.
-/

@[expose] public section

open scoped BigOperators
open Real

namespace Transformer.Metastability

/-- **A within-cluster interaction is bounded by its angular gap.**
Source: arXiv:2410.06833v1, §3.2, the angular gradient and
the within-cap gap in the definition of `𝒩_β`. -/
theorem interaction_abs_le_gap (β : ℝ) (hβ : 0 ≤ β) (x : ℝ) :
    |Causal.h_pot β x| ≤ |x| := by
  have hexp : Real.exp (β * (Real.cos x - 1)) ≤ 1 :=
    Real.exp_le_one_iff.mpr (mul_nonpos_of_nonneg_of_nonpos hβ
      (by linarith [Real.cos_le_one x]))
  rw [Causal.h_pot, abs_mul, abs_of_pos (Real.exp_pos _)]
  calc
    _ ≤ 1 * |Real.sin x| := mul_le_mul_of_nonneg_right hexp (abs_nonneg _)
    _ ≤ |x| := by simpa only [one_mul] using Real.abs_sin_le_abs (x := x)

/-- **A separated cross-cap interaction is exponentially small.**
Source: arXiv:2410.06833v1, §3.2, `eq: alpha.dist.2` and
the cross-cap terms in `lem: PL.borjan`. -/
theorem interaction_abs_le_cross (β α x : ℝ) (hβ : 0 ≤ β)
    (hcos : Real.cos x ≤ α) :
    |Causal.h_pot β x| ≤ Real.exp (-((1 - α) * β)) := by
  rw [Causal.h_pot, abs_mul, abs_of_pos (Real.exp_pos _)]
  calc
    _ ≤ Real.exp (β * (Real.cos x - 1)) * 1 :=
      mul_le_mul_of_nonneg_left (Real.abs_sin_le_one x) (Real.exp_pos _).le
    _ ≤ Real.exp (-((1 - α) * β)) := by
      rw [mul_one]
      exact Real.exp_le_exp.mpr (by nlinarith)

/-- Both scalar interaction bounds have all their inputs at
`β=1`, `α=1/2`, `x=π/3`. -/
example : (0 : ℝ) ≤ 1 ∧ Real.cos (π / 3) ≤ (1 / 2 : ℝ) := by
  rw [Real.cos_pi_div_three]
  norm_num

/-- **Every interaction at an in-cap slow point is at most `exp(-λβ/2)`.**
Source: arXiv:2410.06833v1, §3.2, `𝒩_β` and `eq: otto.attention`.
The containment hypothesis supplies a cap for each of the two tokens;
it does not follow from membership in the displayed `𝒩_β`. -/
theorem interaction_abs_le_at_slow_point (n k : ℕ) (β : ℝ) (hβ : 0 ≤ β)
    (α τ lam : ℝ) (hrate : lam / 2 ≤ 1 - α) (ω : Idx k → ℝ)
    (hα : ∀ q q' : Idx k, q ≠ q' → ∀ u v : ℝ,
      1 - 2 * τ ≤ Real.cos (u - ω q) → 1 - 2 * τ ≤ Real.cos (v - ω q') →
        Real.cos (u - v) ≤ α)
    (V : Idx n → ℝ) (hcaps : ∀ i : Idx n, ∃ q : Idx k,
      1 - 2 * τ ≤ Real.cos (V i - ω q))
    (hN : V ∈ slowManifold n β τ lam k ω) (i j : Idx n) :
    |Causal.h_pot β (V i - V j)| ≤ Real.exp (-(lam * β / 2)) := by
  obtain ⟨q, hq⟩ := hcaps i
  obtain ⟨q', hq'⟩ := hcaps j
  rcases eq_or_ne q q' with heq | hne
  · subst q'
    exact (interaction_abs_le_gap β hβ _).trans (hN q i j hq hq')
  · exact (interaction_abs_le_cross β α _ hβ (hα q q' hne _ _ hq hq')).trans
      (Real.exp_le_exp.mpr (by nlinarith))

/-- **Each actual gradient coordinate is bounded by `exp(-λβ/2)/n`.**
Source: arXiv:2410.06833v1, §3.2, the normalized gradient
and (H2) required by `eq: otto.attention`. -/
theorem angularGrad_abs_le_at_slow_point (n : ℕ) (hn : 0 < n) (k : ℕ)
    (β : ℝ) (hβ : 0 ≤ β) (α τ lam : ℝ) (hrate : lam / 2 ≤ 1 - α)
    (ω : Idx k → ℝ)
    (hα : ∀ q q' : Idx k, q ≠ q' → ∀ u v : ℝ,
      1 - 2 * τ ≤ Real.cos (u - ω q) → 1 - 2 * τ ≤ Real.cos (v - ω q') →
        Real.cos (u - v) ≤ α)
    (V : Idx n → ℝ) (hcaps : ∀ i : Idx n, ∃ q : Idx k,
      1 - 2 * τ ≤ Real.cos (V i - ω q))
    (hN : V ∈ slowManifold n β τ lam k ω) (i : Idx n) :
    |angularGrad n β V i| ≤ Real.exp (-(lam * β / 2)) / (n : ℝ) := by
  have hnR : (0 : ℝ) < n := by exact_mod_cast hn
  have heq : angularGrad n β V i = -(1 / (n : ℝ) ^ 2) *
      ∑ j : Idx n, Causal.h_pot β (V i - V j) := by
    simp only [angularGrad, Causal.h_pot, mul_comm]
    ring
  rw [heq, abs_mul, abs_neg, abs_of_nonneg (by positivity : 0 ≤ 1 / (n : ℝ) ^ 2)]
  calc
    _ ≤ (1 / (n : ℝ) ^ 2) * ∑ j : Idx n, |Causal.h_pot β (V i - V j)| :=
      mul_le_mul_of_nonneg_left (Finset.abs_sum_le_sum_abs _ _) (by positivity)
    _ ≤ (1 / (n : ℝ) ^ 2) * ∑ _j : Idx n, Real.exp (-(lam * β / 2)) := by
      apply mul_le_mul_of_nonneg_left _ (by positivity)
      exact Finset.sum_le_sum fun j _ =>
        interaction_abs_le_at_slow_point n k β hβ α τ lam hrate ω hα V hcaps hN i j
    _ = _ := by
      simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
      field_simp

/-- **The Euclidean squared gradient is at most `exp(-λβ/2)²`.**
Source: arXiv:2410.06833v1, §3.1 (H2), and §3.2,
`eq: otto.attention`, with explicit containment in the fixed caps. -/
theorem angularGrad_sq_sum_le_at_slow_point (n : ℕ) (hn : 0 < n) (k : ℕ)
    (β : ℝ) (hβ : 0 ≤ β) (α τ lam : ℝ) (hrate : lam / 2 ≤ 1 - α)
    (ω : Idx k → ℝ)
    (hα : ∀ q q' : Idx k, q ≠ q' → ∀ u v : ℝ,
      1 - 2 * τ ≤ Real.cos (u - ω q) → 1 - 2 * τ ≤ Real.cos (v - ω q') →
        Real.cos (u - v) ≤ α)
    (V : Idx n → ℝ) (hcaps : ∀ i : Idx n, ∃ q : Idx k,
      1 - 2 * τ ≤ Real.cos (V i - ω q))
    (hN : V ∈ slowManifold n β τ lam k ω) :
    (∑ i : Idx n, (angularGrad n β V i) ^ 2) ≤ Real.exp (-(lam * β / 2)) ^ 2 := by
  have hnR : (0 : ℝ) < n := by exact_mod_cast hn
  have hn1 : (1 : ℝ) ≤ n := by exact_mod_cast (show 1 ≤ n by omega)
  calc
    _ ≤ ∑ _i : Idx n, (Real.exp (-(lam * β / 2)) / (n : ℝ)) ^ 2 := by
      apply Finset.sum_le_sum
      intro i _
      have h := pow_le_pow_left₀ (abs_nonneg _) (angularGrad_abs_le_at_slow_point
        n hn k β hβ α τ lam hrate ω hα V hcaps hN i) 2
      simpa only [sq_abs] using h
    _ = Real.exp (-(lam * β / 2)) ^ 2 / (n : ℝ) := by
      simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
      field_simp
    _ ≤ Real.exp (-(lam * β / 2)) ^ 2 := by
      apply (div_le_iff₀ hnR).mpr
      nlinarith [sq_nonneg (Real.exp (-(lam * β / 2)))]

/-- All inputs of the three in-cap slow-point bounds are realized at
once: two consensus tokens, one cap, `β=1`, `τ=1/100`, `λ=1`, `α=0`.
The fixed cap is nonempty, the slow point is actual, and the separation
condition has no pair of distinct cap indices. -/
example : 0 < 2 ∧ (0 : ℝ) ≤ 1 ∧ (1 / 2 : ℝ) ≤ 1 - 0 ∧
    (∀ q q' : Idx 1, q ≠ q' → ∀ u v : ℝ,
      1 - 2 * (1 / 100 : ℝ) ≤ Real.cos (u - (fun _ => 0 : Idx 1 → ℝ) q) →
      1 - 2 * (1 / 100 : ℝ) ≤ Real.cos (v - (fun _ => 0 : Idx 1 → ℝ) q') →
        Real.cos (u - v) ≤ 0) ∧
    (∀ i : Idx 2, ∃ q : Idx 1,
      1 - 2 * (1 / 100 : ℝ) ≤ Real.cos ((fun _ => 0 : Idx 2 → ℝ) i -
        (fun _ => 0 : Idx 1 → ℝ) q)) ∧
    (fun _ : Idx 2 => (0 : ℝ)) ∈ slowManifold 2 1 (1 / 100) 1 1 (fun _ => 0) := by
  refine ⟨by norm_num, zero_le_one, by norm_num,
    fun q q' hne => (hne (Subsingleton.elim q q')).elim, fun _ => ⟨0, by norm_num⟩, ?_⟩
  intro q i j _ _
  simp only [sub_self, abs_zero]
  exact (Real.exp_pos _).le

end Transformer.Metastability
