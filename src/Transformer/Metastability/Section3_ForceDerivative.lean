/-
# A quantitative derivative bound for the actual angular force

ArXiv:2410.06833v1, §3.2, `claim: 1` and `eq: Ht.third.lb`.
At a fixed configuration every gradient component evaluates the same
scalar force at its token angle. Inside a cap, a single positive
interaction derivative already dominates the sum of all possible
negative derivatives from other caps. Thus the force is antitone on the
cap chart. The source's ordered-cluster comparison follows from this.
The numerical margin is explicit: an internal derivative contributes
at least `1/2`, while the entire possible cross-cap deficit is below `1/4`.
The factor `1/n²` is exactly the normalization of the actual energy gradient.
-/

import Transformer.Metastability.Section3_CapTemperature
import Transformer.Metastability.AngularEnergy
import Mathlib.Analysis.Calculus.Deriv.MeanValue

open scoped BigOperators
open Set Real

namespace Transformer.Metastability

/-- The scalar force whose value at `Θ l` is the actual `l`-th energy
gradient component. Source: arXiv:2410.06833v1, §3.2, the gradient in
`lem: PL.borjan` and the expression compared in `claim: 1`. -/
noncomputable def angularForce (n : ℕ) (β : ℝ) (Θ : Idx n → ℝ) (u : ℝ) : ℝ :=
  -((1 / (n : ℝ) ^ 2) * ∑ m : Idx n, Causal.h_pot β (u - Θ m))

/-- The scalar force agrees exactly with the gradient component.
Source: arXiv:2410.06833v1, §3.2, the derivative preceding `eq: Ht.third.lb`. -/
theorem angularForce_at_token (n : ℕ) (β : ℝ) (Θ : Idx n → ℝ) (l : Idx n) :
    angularForce n β Θ (Θ l) = angularGrad n β Θ l := by
  simp only [angularForce, angularGrad, Causal.h_pot, mul_comm]

/-- Differentiating the genuine force sums the negatives of the
interaction derivatives. Source: arXiv:2410.06833v1, §3.2, `g` and the
Hessian estimates preceding `eq: Ht.second.lb`. -/
theorem hasDerivAt_angularForce (n : ℕ) (β : ℝ) (Θ : Idx n → ℝ) (u : ℝ) :
    HasDerivAt (angularForce n β Θ)
      (-((1 / (n : ℝ) ^ 2) * ∑ m : Idx n, Causal.g_pot β (u - Θ m))) u := by
  have hterm (m : Idx n) : HasDerivAt (fun v => Causal.h_pot β (v - Θ m))
      (Causal.g_pot β (u - Θ m)) u := by
    simpa only [mul_one, Function.comp_def, id_eq] using (Causal.hasDerivAt_h_pot β (u - Θ m)).comp u
      ((hasDerivAt_id u).sub_const (Θ m))
  have hs := HasDerivAt.sum (u := (Finset.univ : Finset (Idx n))) (fun m _ => hterm m)
  convert! (hs.const_mul (1 / (n : ℝ) ^ 2)).neg using 1
  simp only [Finset.sum_fn, Pi.neg_def]
  rfl

/-- **The actual force derivative is at most `-1/(4n²)` on a small cap.**
Only the source's cap geometry, `γ>0`, and `eq: tau.small` are needed;
the off-manifold threshold is unnecessary for this stronger comparison.
All hypotheses are jointly witnessed by the three-token example below.
Source: arXiv:2410.06833v1, §3.2, `claim: 1`, `eq: Ht.third.lb`. -/
theorem angularForce_deriv_le_on_cap (n k : ℕ) (β τ δ α : ℝ)
    (ω : Idx k → ℝ) (Θ : Idx n → ℝ) (q : Idx k) (i₀ : Idx n)
    (hβ : 1 < β) (hτ : 0 < τ) (hτ16 : τ < 1 / 16) (hδ : 0 ≤ δ) (hδ1 : δ < 1)
    (hcaps : ∀ i : Idx n, ∃ p : Idx k, 1 - 2 * τ ≤ Real.cos (Θ i - ω p))
    (hα : ∀ p p' : Idx k, p ≠ p' → ∀ u v : ℝ,
      1 - 2 * τ ≤ Real.cos (u - ω p) → 1 - 2 * τ ≤ Real.cos (v - ω p') →
        Real.cos (u - v) ≤ α)
    (hγ : 0 < 1 - α - 8 * τ - β⁻¹ * Real.log (2 * (n : ℝ) ^ 2 / τ))
    (hsmall : ∀ u v : ℝ, |u - ω q| ≤ π → |v - ω q| ≤ π →
      1 - 2 * τ ≤ Real.cos (u - ω q) → 1 - 2 * τ ≤ Real.cos (v - ω q) →
        |u - v| ≤ (1 / 8) * Real.sqrt ((1 - δ) / (β + 1 / 2)))
    (hchart : ∀ i : Idx n, 1 - 2 * τ ≤ Real.cos (Θ i - ω q) → |Θ i - ω q| ≤ π)
    (hmem₀ : 1 - 2 * τ ≤ Real.cos (Θ i₀ - ω q)) :
    ∀ u : ℝ, |u - ω q| ≤ π → 1 - 2 * τ ≤ Real.cos (u - ω q) →
      deriv (angularForce n β Θ) u ≤ -(1 / (4 * (n : ℝ) ^ 2)) := by
  have hn : 0 < n := by have := i₀.isLt; omega
  have hβ0 : 0 < β := by linarith
  have hwidth := small_cap_temperature_bound β τ δ (ω q) hβ hτ hτ16 hδ hδ1 hsmall
  have hcross := cross_interaction_count_lt n β τ α hn hβ0 hτ hwidth hγ
  let B := (1 + β) * Real.exp (-((1 - α) * β))
  have hB : 0 ≤ B := by dsimp only [B]; positivity
  have hcrossB : (n : ℝ) * B < 1 / 4 := by simpa only [B, mul_assoc] using hcross
  have hinside (u : ℝ) (hu : |u - ω q| ≤ π ∧ 1 - 2 * τ ≤ Real.cos (u - ω q))
      (m : Idx n) (hm : 1 - 2 * τ ≤ Real.cos (Θ m - ω q)) :
      (1 / 2 : ℝ) ≤ Causal.g_pot β (u - Θ m) := by
    have hd := hsmall u (Θ m) hu.1 (hchart m hm) hu.2 hm
    have hden : 0 < β + 1 / 2 := by linarith
    have hratio : 0 ≤ (1 - δ) / (β + 1 / 2) := div_nonneg (by linarith) hden.le
    have hsq := Real.sq_sqrt hratio
    have hd2 : (u - Θ m) ^ 2 ≤ (1 / 64) * ((1 - δ) / (β + 1 / 2)) := by
      nlinarith [sq_abs (u - Θ m), Real.sqrt_nonneg ((1 - δ) / (β + 1 / 2)),
        abs_nonneg (u - Θ m)]
    have hscaled := mul_le_mul_of_nonneg_left hd2 hden.le
    have heq : (β + 1 / 2) * ((1 / 64) * ((1 - δ) / (β + 1 / 2))) = (1 - δ) / 64 := by
      field_simp
    rw [heq] at hscaled
    exact interaction_derivative_ge_half β (u - Θ m) hβ0.le (by linarith)
  have hsum (u : ℝ) (hu : |u - ω q| ≤ π ∧ 1 - 2 * τ ≤ Real.cos (u - ω q)) :
      (1 / 4 : ℝ) ≤ ∑ m : Idx n, Causal.g_pot β (u - Θ m) := by
    have hterm (m : Idx n) : (if m = i₀ then (1 / 2 : ℝ) else 0) - B ≤
        Causal.g_pot β (u - Θ m) := by
      by_cases hm : 1 - 2 * τ ≤ Real.cos (Θ m - ω q)
      · have hin := hinside u hu m hm
        split_ifs <;> linarith
      · have hne : m ≠ i₀ := by intro heq; subst m; exact hm hmem₀
        rw [ite_eq_right hne, zero_sub]
        obtain ⟨p, hp⟩ := hcaps m
        have hpq : q ≠ p := by intro heq; subst p; exact hm hp
        exact interaction_derivative_cross_lower β α (u - Θ m) hβ0.le (hα q p hpq u (Θ m) hu.2 hp)
    have hbound : (1 / 2 : ℝ) - (n : ℝ) * B ≤
        ∑ m : Idx n, Causal.g_pot β (u - Θ m) := by
      calc
        _ = ∑ m : Idx n, ((if m = i₀ then (1 / 2 : ℝ) else 0) - B) := by
          rw [Finset.sum_sub_distrib, Finset.sum_ite_eq' Finset.univ i₀]
          simp only [Finset.mem_univ, ite_true, Finset.sum_const, Finset.card_univ,
            Fintype.card_fin, nsmul_eq_mul]
        _ ≤ _ := Finset.sum_le_sum fun m _ => hterm m
    linarith
  intro u huChart huCap
  rw [(hasDerivAt_angularForce n β Θ u).deriv]
  have hs := hsum u ⟨huChart, huCap⟩
  have hnR : (0 : ℝ) < n := by exact_mod_cast hn
  calc
    _ ≤ -((1 / (n : ℝ) ^ 2) * (1 / 4)) :=
      neg_le_neg (mul_le_mul_of_nonneg_left hs (by positivity))
    _ = _ := by field_simp

/-- The complete derivative-bound hypotheses hold for a cap of the
separated three-token configuration. Its two close particles have actual
angles `0` and `10⁻⁴`, and the remaining particle is antipodal. -/
example : let τ : ℝ := 1 / 10 ^ 8
    let α : ℝ := 1 - 2 * (1 - 2 * τ) ^ 2
    let ω : Idx 2 → ℝ := ![0, π]
    let Θ : Idx 3 → ℝ := ![0, 1 / 10 ^ 4, π]
    (1 : ℝ) < 1000 ∧ 0 < τ ∧ τ < 1 / 16 ∧ (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) < 1 ∧
    (∀ i : Idx 3, ∃ p : Idx 2, 1 - 2 * τ ≤ Real.cos (Θ i - ω p)) ∧
    (∀ p p' : Idx 2, p ≠ p' → ∀ u v : ℝ,
      1 - 2 * τ ≤ Real.cos (u - ω p) → 1 - 2 * τ ≤ Real.cos (v - ω p') →
        Real.cos (u - v) ≤ α) ∧
    0 < 1 - α - 8 * τ - (1000 : ℝ)⁻¹ * Real.log (2 * (3 : ℝ) ^ 2 / τ) ∧
    (∀ u v : ℝ, |u - ω 0| ≤ π → |v - ω 0| ≤ π →
      1 - 2 * τ ≤ Real.cos (u - ω 0) → 1 - 2 * τ ≤ Real.cos (v - ω 0) →
        |u - v| ≤ (1 / 8) * Real.sqrt ((1 - 1 / 2) / (1000 + 1 / 2))) ∧
    (∀ i : Idx 3, 1 - 2 * τ ≤ Real.cos (Θ i - ω 0) → |Θ i - ω 0| ≤ π) ∧
    1 - 2 * τ ≤ Real.cos (Θ 0 - ω 0) := by
  intro τ α ω Θ
  obtain ⟨hcaps, hα, hγ, hsmall, -, -, -, hchart, hmem, -⟩ := circle_witness
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    hcaps, hα, hγ, hsmall 0, ?_, ?_⟩
  · intro i hi
    obtain ⟨j, rfl⟩ := (hmem i).mp hi
    exact hchart j
  · norm_num [Θ, ω, τ]

end Transformer.Metastability
