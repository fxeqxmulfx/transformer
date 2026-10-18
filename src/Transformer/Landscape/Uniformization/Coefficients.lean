/-
# Local uniformization: the product step at a point

Let `F = M(x') · V(x) · P(x', y)` with `P = y^d + Σ_i B_i(x') y^i` centered, `M` and
the `B_i` that do not vanish identically with normal crossings at `w'`, and every
combination `B_i^{d - i'} - c B_{i'}^{d - i}` with normal crossings, so that the
exponents of the `B_i` have comparable multiples. Then `F` has monomial charts at every
`(w', y₀)`: by induction on the degree when `P(w', ·) ≠ y^d`, as a unit times `M` when
`P(w', y₀) ≠ 0`, and by the Newton step otherwise. Auxiliary for Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2.
-/

import Transformer.Landscape.Uniformization.Newton

open Filter Set Polynomial
open scoped BigOperators Topology

namespace Transformer.Landscape.Uniformization

open Transformer.Landscape.AnalyticPreparation (Base preparedPolynomial distinguishedPolynomial
  distinguishedPolynomial_eval)

variable {n : ℕ}

theorem not_eventuallyEq_zero_mul {f g : Base n → ℝ} {w : Base n} (hf : AnalyticAt ℝ f w)
    (hg : AnalyticAt ℝ g w) (hf0 : ¬ f =ᶠ[𝓝 w] 0) (hg0 : ¬ g =ᶠ[𝓝 w] 0) :
    ¬ (fun x => f x * g x) =ᶠ[𝓝 w] 0 := fun h =>
  (eventuallyEq_zero_or_of_mul hf hg h).elim hf0 hg0

theorem not_eventuallyEq_zero_pow {f : Base n → ℝ} {w : Base n} (hf : AnalyticAt ℝ f w)
    (hf0 : ¬ f =ᶠ[𝓝 w] 0) (k : ℕ) : ¬ (fun x => f x ^ k) =ᶠ[𝓝 w] 0 := by
  have h := not_eventuallyEq_zero_prod (Finset.range k) (F := fun _ => f) (fun _ _ => hf)
    fun _ _ => hf0
  simpa only [Finset.prod_const, Finset.card_range] using h

/-- One of `f - g`, `f - 2 g` does not vanish identically when `g` does not. -/
theorem exists_sub_mul_not_eventuallyEq_zero {f g : Base n → ℝ} {w : Base n}
    (hg0 : ¬ g =ᶠ[𝓝 w] 0) : ∃ c : ℝ, c ≠ 0 ∧ ¬ (fun x => f x - c * g x) =ᶠ[𝓝 w] 0 := by
  by_cases h : (fun x => f x - 1 * g x) =ᶠ[𝓝 w] 0
  · refine ⟨2, two_ne_zero, fun h2 => hg0 ?_⟩
    filter_upwards [h, h2] with x h1 h2
    simp only [Pi.zero_apply] at h1 h2 ⊢
    linarith
  · exact ⟨1, one_ne_zero, h⟩

theorem NormalCrossingAt.of_mul_right {F H : Base n → ℝ} {w : Base n} (hF : AnalyticAt ℝ F w)
    (hH : AnalyticAt ℝ H w) (h : NormalCrossingAt (fun x => F x * H x) w) :
    NormalCrossingAt H w :=
  NormalCrossingAt.of_mul hH hF (h.congr (Eventually.of_forall fun x => mul_comm (F x) (H x)))

/-- Normal crossings of a family of coefficients, with zero units off `S`. -/
theorem exists_normalCrossing_family {d : ℕ} {B : Fin d → Base n → ℝ} {w : Base n}
    {S : Finset (Fin d)} (hS : ∀ i ∈ S, NormalCrossingAt (B i) w)
    (hS0 : ∀ i ∉ S, B i =ᶠ[𝓝 w] 0) :
    ∃ (β : Fin d → Base n → ℝ) (α : Fin d → Fin n → ℕ), (∀ i, AnalyticAt ℝ (β i) w) ∧
      (∀ i, ∀ᶠ x in 𝓝 w, B i x = β i x * monomial (α i) (x - w)) ∧
      (∀ i ∈ S, β i w ≠ 0) ∧ ∀ i ∉ S, ∀ x, β i x = 0 := by
  classical
  have hS' : ∀ i ∈ S, ∃ (p : Fin n → ℕ) (u : Base n → ℝ), AnalyticAt ℝ u w ∧ u w ≠ 0 ∧
      ∀ᶠ x in 𝓝 w, B i x = u x * monomial p (x - w) := hS
  choose! α β hβ hβw hB using hS'
  refine ⟨fun i => if i ∈ S then β i else 0, fun i => if i ∈ S then α i else 0,
    fun i => ?_, fun i => ?_, fun i hi => ?_, fun i hi x => ?_⟩
  · by_cases hi : i ∈ S
    · simpa only [hi, ↓reduceIte] using hβ i hi
    · simp only [hi, ↓reduceIte]
      exact analyticAt_const
  · by_cases hi : i ∈ S
    · simpa only [hi, ↓reduceIte] using hB i hi
    · filter_upwards [hS0 i hi] with x hx
      simp only [hi, ↓reduceIte, Pi.zero_apply, zero_mul]
      exact hx
  · simpa only [hi, ↓reduceIte] using hβw i hi
  · simp only [hi, ↓reduceIte, Pi.zero_apply]

/-- If `B₁^{k₁} - c B₂^{k₂}` has normal crossings, the multiples `k₁ • α₁`, `k₂ • α₂` of the
exponents of `B₁`, `B₂` are comparable. -/
theorem le_or_le_of_normalCrossingAt_pow_sub {B₁ B₂ β₁ β₂ : Base n → ℝ} {α₁ α₂ : Fin n → ℕ}
    {w : Base n} {k₁ k₂ : ℕ} {c : ℝ} (hc : c ≠ 0) (hβ₁ : AnalyticAt ℝ β₁ w)
    (hβ₁0 : β₁ w ≠ 0) (hβ₂ : AnalyticAt ℝ β₂ w) (hβ₂0 : β₂ w ≠ 0)
    (hB₁ : ∀ᶠ x in 𝓝 w, B₁ x = β₁ x * monomial α₁ (x - w))
    (hB₂ : ∀ᶠ x in 𝓝 w, B₂ x = β₂ x * monomial α₂ (x - w))
    (h : NormalCrossingAt (fun x => B₁ x ^ k₁ - c * B₂ x ^ k₂) w) :
    k₁ • α₁ ≤ k₂ • α₂ ∨ k₂ • α₂ ≤ k₁ • α₁ := by
  refine le_or_le_of_normalCrossingAt_sub (U := fun x => β₁ x ^ k₁)
    (V := fun x => c * β₂ x ^ k₂) (hβ₁.fun_pow k₁) (pow_ne_zero _ hβ₁0)
    (analyticAt_const.fun_mul (hβ₂.fun_pow k₂)) (mul_ne_zero hc (pow_ne_zero _ hβ₂0))
    (h.congr ?_)
  filter_upwards [hB₁, hB₂] with x h1 h2
  rw [h1, h2, monomial_smul_exp, monomial_smul_exp]
  ring

/-- **The product step at a point.** Auxiliary for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
theorem hasMonomialCharts_product_point {d : ℕ}
    (hcent : ∀ (m : Base n → ℝ) (V : Base (n + 1) → ℝ) (c : Fin d → Base n → ℝ)
      (x₀ : Base (n + 1)), AnalyticAt ℝ m (Fin.init x₀) → ¬ m =ᶠ[𝓝 (Fin.init x₀)] 0 →
      AnalyticAt ℝ V x₀ → V x₀ ≠ 0 → (∀ i, AnalyticAt ℝ (c i) (Fin.init x₀)) →
      (∀ i : Fin d, (i : ℕ) + 1 = d → c i (Fin.init x₀) = 0) →
      distinguishedPolynomial d c (Fin.init x₀) ≠ X ^ d →
      HasMonomialCharts (fun x => m (Fin.init x) *
        (V x * preparedPolynomial d c (Fin.init x, x (Fin.last n)))) x₀)
    {w' : Base n} {y₀ : ℝ} {M : Base n → ℝ} {V : Base (n + 1) → ℝ}
    {B : Fin d → Base n → ℝ} {S : Finset (Fin d)}
    (hM : NormalCrossingAt M w') (hB : ∀ i, AnalyticAt ℝ (B i) w')
    (hBS : ∀ i ∈ S, NormalCrossingAt (B i) w') (hB0 : ∀ i ∉ S, B i =ᶠ[𝓝 w'] 0)
    (hcenter : ∀ i : Fin d, (i : ℕ) + 1 = d → ∀ x', B i x' = 0)
    (hcomp : ∀ i ∈ S, ∀ i' ∈ S, ∃ c : ℝ, c ≠ 0 ∧
      NormalCrossingAt (fun x => B i x ^ (d - i') - c * B i' x ^ (d - i)) w')
    (hV : AnalyticAt ℝ V (Fin.snoc w' y₀)) (hV0 : V (Fin.snoc w' y₀) ≠ 0) :
    HasMonomialCharts (fun x => M (Fin.init x) *
      (V x * preparedPolynomial d B (Fin.init x, x (Fin.last n)))) (Fin.snoc w' y₀) := by
  have hw : (Fin.init (Fin.snoc w' y₀ : Base (n + 1)) : Base n) = w' := Fin.init_snoc _ _
  have hB' : ∀ i, AnalyticAt ℝ (B i) (Fin.init (Fin.snoc w' y₀ : Base (n + 1))) := by
    rw [hw]
    exact hB
  by_cases hQ : distinguishedPolynomial d B w' = X ^ d
  · rcases eq_or_ne y₀ 0 with rfl | hy
    · obtain ⟨γ, μ, hμ, hμ0, hMeq⟩ := hM
      obtain ⟨β, α, hβ, hBeq, hβS, hβ0⟩ := exists_normalCrossing_family hBS hB0
      refine hasMonomialCharts_newton hcent hμ hμ0 hMeq hβ hBeq hβS hβ0 (fun i hi hα0 => ?_)
        (fun i hi hiS => ?_) (fun i hi i' hi' => ?_) hV hV0
      · have h := (hBeq i).self_of_nhds
        have hc := distinguishedPolynomial_coeff d B w' i
        rw [hQ, coeff_X_pow, ite_eq_right (by have := i.isLt; omega)] at hc
        rw [hα0, sub_self, monomial_zero_exp, mul_one, ← hc] at h
        exact hβS i hi h.symm
      · exact (hBS i hiS).not_eventuallyEq_zero (Eventually.of_forall fun x => hcenter i hi x)
      · obtain ⟨c, hc, hNC⟩ := hcomp i hi i' hi'
        exact le_or_le_of_normalCrossingAt_pow_sub hc (hβ i) (hβS i hi) (hβ i') (hβS i' hi')
          (hBeq i) (hBeq i') hNC
    · have hM' : NormalCrossingAt M (Fin.init (Fin.snoc w' y₀ : Base (n + 1))) := by
        rwa [hw]
      refine hasMonomialCharts_of_normalCrossingAt (hM'.comp_init.mul
        (normalCrossingAt_of_ne_zero (hV.fun_mul (analyticAt_preparedPolynomial hB')) ?_))
      rw [hw, Fin.snoc_last, ← distinguishedPolynomial_eval, hQ, eval_pow, eval_X]
      exact mul_ne_zero hV0 (pow_ne_zero _ hy)
  · exact hcent M V B _ (by rw [hw]; exact hM.analyticAt)
      (by rw [hw]; exact hM.not_eventuallyEq_zero) hV hV0 hB'
      (fun i hi => hcenter i hi _) (by rwa [hw])

end Transformer.Landscape.Uniformization
