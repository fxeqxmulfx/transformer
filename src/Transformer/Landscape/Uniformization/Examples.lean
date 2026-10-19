module

public import Transformer.Landscape.Uniformization.Induction

/-!
# Local uniformization: examples for the product step and the induction

The coordinate `x₀` of `ℝ¹` satisfies the hypotheses of the lemmas on coefficients;
the polynomial `y² + x₀²` those of the product step, and `y² - 1` those of the centered
step; `y²` has exact order two in the last variable. The cusp `x₀² - x₁³`, which has no
normal crossings at the origin, has monomial charts there. The induction hypotheses are
discharged by local uniformization itself. Auxiliary examples for Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2.
-/

@[expose] public section

open Filter Set Polynomial
open scoped BigOperators Topology

namespace Transformer.Landscape.Uniformization

open Transformer.Landscape.AnalyticPreparation (Base preparedPolynomial distinguishedPolynomial)

/-- The coefficients `(x₀², 0)` of `P = y² + x₀²`. -/
local notation "Bₑ" => (![fun x' : Base 1 => x' 0 ^ 2, fun _ => 0] : Fin 2 → Base 1 → ℝ)

/-- Products and powers of `x₀` are nonzero germs, so is some `x₀ - c · x₀` with `c ≠ 0`,
`x₀` is a factor of `x₀ · x₀`, and `x₀ - 2 x₀` has comparable exponents.
Appendix D.1 of arXiv:2510.22026v2. -/
example : ¬ (fun x : Base 1 => x 0 * x 0) =ᶠ[𝓝 0] 0 ∧ ¬ (fun x : Base 1 => x 0 ^ 3) =ᶠ[𝓝 0] 0 ∧
    (∃ c : ℝ, c ≠ 0 ∧ ¬ (fun x : Base 1 => x 0 - c * x 0) =ᶠ[𝓝 0] 0) ∧
    NormalCrossingAt (fun x : Base 1 => x 0) 0 ∧
    ((1 • Pi.single 0 1 : Fin 1 → ℕ) ≤ 1 • Pi.single 0 1 ∨
      (1 • Pi.single 0 1 : Fin 1 → ℕ) ≤ 1 • Pi.single 0 1) := by
  have h : NormalCrossingAt (fun x : Base 1 => x 0) 0 :=
    (normalCrossingAt_coord 0 0).congr (Eventually.of_forall fun x => by simp)
  have hB : ∀ᶠ x in 𝓝 (0 : Base 1),
      x 0 = (fun _ => (1 : ℝ)) x * monomial (Pi.single 0 1) (x - 0) :=
    Eventually.of_forall fun x => by simp [monomial_single]
  have hNC : NormalCrossingAt (fun x : Base 1 => x 0 ^ 1 - 2 * x 0 ^ 1) 0 :=
    normalCrossingAt_of_eq_mul_monomial 0 (Pi.single 0 1) analyticAt_const
      (by norm_num : (-1 : ℝ) ≠ 0) (Eventually.of_forall fun x => by simp [monomial_single]; ring)
  exact ⟨not_eventuallyEq_zero_mul h.analyticAt h.analyticAt h.not_eventuallyEq_zero
      h.not_eventuallyEq_zero, not_eventuallyEq_zero_pow h.analyticAt h.not_eventuallyEq_zero 3,
    exists_sub_mul_not_eventuallyEq_zero h.not_eventuallyEq_zero,
    NormalCrossingAt.of_mul_right h.analyticAt h.analyticAt (h.mul h),
    le_or_le_of_normalCrossingAt_pow_sub two_ne_zero analyticAt_const one_ne_zero
      analyticAt_const one_ne_zero hB hB hNC⟩

/-- **The product step** for `1 · (1 · (y² + x₀²))`, at the origin and at every point
`(0, y₀)`; the coefficients `(x₀², 0)` have units `(β i)` and exponents `(α i)`, and
`(x₀²)² - 2 (x₀²)²` has normal crossings. Appendix D.1 of arXiv:2510.22026v2. -/
example : (∃ (β : Fin 2 → Base 1 → ℝ) (α : Fin 2 → Fin 1 → ℕ), (∀ i, AnalyticAt ℝ (β i) 0) ∧
      (∀ i, ∀ᶠ x in 𝓝 0, Bₑ i x = β i x * monomial (α i) (x - 0)) ∧
      (∀ i ∈ ({0} : Finset (Fin 2)), β i 0 ≠ 0) ∧ ∀ i ∉ ({0} : Finset (Fin 2)), ∀ x, β i x = 0) ∧
    (∀ y₀ : ℝ, HasMonomialCharts (fun x : Base 2 => (fun _ : Base 1 => (1 : ℝ)) (Fin.init x) *
      ((fun _ : Base 2 => (1 : ℝ)) x * preparedPolynomial 2 Bₑ (Fin.init x, x (Fin.last 1))))
      (Fin.snoc 0 y₀)) ∧
    HasMonomialCharts (fun x : Base 2 => (fun _ : Base 1 => (1 : ℝ)) (Fin.init x) *
      ((fun _ : Base 2 => (1 : ℝ)) x * preparedPolynomial 2 Bₑ (Fin.init x, x (Fin.last 1))))
      0 := by
  have hA : AnalyticAt ℝ (fun x : Base 1 => x 0) 0 :=
    (ContinuousLinearMap.proj 0 : Base 1 →L[ℝ] ℝ).analyticAt 0
  have hB : ∀ i, AnalyticAt ℝ (Bₑ i) 0 := by
    intro i
    fin_cases i
    · exact hA.fun_pow 2
    · exact analyticAt_const
  have hBS : ∀ i ∈ ({0} : Finset (Fin 2)), NormalCrossingAt (Bₑ i) 0 := by
    intro i hi
    rw [Finset.mem_singleton.1 hi]
    exact normalCrossingAt_of_eq_mul_monomial 0 (fun _ => 2) analyticAt_const one_ne_zero
      (Eventually.of_forall fun x => by simp [monomial])
  have hB0 : ∀ i ∉ ({0} : Finset (Fin 2)), Bₑ i =ᶠ[𝓝 0] 0 := by
    intro i hi
    fin_cases i
    · exact absurd (Finset.mem_singleton_self 0) hi
    · exact Eventually.of_forall fun _ => rfl
  have hcenter : ∀ i : Fin 2, (i : ℕ) + 1 = 2 → ∀ x', Bₑ i x' = 0 := by
    intro i hi x'
    fin_cases i
    · simp at hi
    · rfl
  have hcent : ∀ (m : Base 1 → ℝ) (V : Base 2 → ℝ) (c : Fin 2 → Base 1 → ℝ) (x₀ : Base 2),
      AnalyticAt ℝ m (Fin.init x₀) → ¬ m =ᶠ[𝓝 (Fin.init x₀)] 0 → AnalyticAt ℝ V x₀ →
      V x₀ ≠ 0 → (∀ i, AnalyticAt ℝ (c i) (Fin.init x₀)) →
      (∀ i : Fin 2, (i : ℕ) + 1 = 2 → c i (Fin.init x₀) = 0) →
      distinguishedPolynomial 2 c (Fin.init x₀) ≠ X ^ 2 →
      HasMonomialCharts (fun x => m (Fin.init x) *
        (V x * preparedPolynomial 2 c (Fin.init x, x (Fin.last 1)))) x₀ :=
    fun m V c x₀ hm hm0 hV hV0 hc hcenter hne => hasMonomialCharts_of_centered
      (fun d' _ => hasMonomialCharts_mul_of_order_le
        (fun _ _ => hasMonomialCharts_of_analyticAt) d') hm hm0 hV hV0 hc hcenter hne
  refine ⟨exists_normalCrossing_family hBS hB0, fun y₀ => hasMonomialCharts_product_point hcent
    (normalCrossingAt_of_ne_zero analyticAt_const one_ne_zero) hB hBS hB0 hcenter
    (fun i hi i' hi' => ⟨2, two_ne_zero, ?_⟩) analyticAt_const one_ne_zero,
    hasMonomialCharts_product (fun _ _ => hasMonomialCharts_of_analyticAt) hcent analyticAt_const
      (normalCrossingAt_of_ne_zero analyticAt_const one_ne_zero).not_eventuallyEq_zero
      analyticAt_const one_ne_zero hB hcenter⟩
  rw [Finset.mem_singleton.1 hi, Finset.mem_singleton.1 hi']
  exact normalCrossingAt_of_eq_mul_monomial 0 (fun _ => 4) analyticAt_const
    (by norm_num : (-1 : ℝ) ≠ 0) (Eventually.of_forall fun x => by simp [monomial]; ring)

/-- The coefficients `(x₀, x₀)` vanish at the origin and are centered by a translation;
`y² + x₀²` is analytic, and its slice at the origin has the order of the root `0` of
`y²`. Appendix D.1 of arXiv:2510.22026v2. -/
example : (∃ (s : Base 1 → ℝ) (b : Fin 2 → Base 1 → ℝ), AnalyticAt ℝ s 0 ∧ s 0 = 0 ∧
      (∀ i, AnalyticAt ℝ (b i) 0) ∧ (∀ i, b i 0 = 0) ∧
      (∀ i : Fin 2, (i : ℕ) + 1 = 2 → ∀ x, b i x = 0) ∧
      ∀ x y, preparedPolynomial 2 (fun _ x => x 0) (x, y + s x) =
        preparedPolynomial 2 b (x, y)) ∧
    AnalyticAt ℝ (fun x : Base 2 => preparedPolynomial 2 Bₑ (Fin.init x, x (Fin.last 1))) 0 ∧
    analyticOrderAt (fun y => (fun _ : Base 2 => (1 : ℝ))
        (Fin.snoc (Fin.init (0 : Base 2)) ((0 : Base 2) (Fin.last 1) + y)) *
        preparedPolynomial 2 Bₑ (Fin.init (0 : Base 2), (0 : Base 2) (Fin.last 1) + y)) 0 =
      ((distinguishedPolynomial 2 Bₑ (Fin.init (0 : Base 2))).rootMultiplicity
        ((0 : Base 2) (Fin.last 1)) : ℕ) := by
  have hA : AnalyticAt ℝ (fun x : Base 1 => x 0) 0 :=
    (ContinuousLinearMap.proj 0 : Base 1 →L[ℝ] ℝ).analyticAt 0
  have hB : ∀ i, AnalyticAt ℝ (Bₑ i) (Fin.init (0 : Base 2)) := by
    intro i
    fin_cases i
    · exact hA.fun_pow 2
    · exact analyticAt_const
  exact ⟨exists_centering _ (fun _ => hA) fun _ => rfl, analyticAt_preparedPolynomial hB,
    analyticOrderAt_slice analyticAt_const one_ne_zero⟩

/-- **The centered step** for `1 · (1 · (y² - 1))`: the coefficients `(-1, 0)` are
centered, and `y² - 1 ≠ y²`. Appendix D.1 of arXiv:2510.22026v2. -/
example : HasMonomialCharts (fun x : Base 2 => (fun _ : Base 1 => (1 : ℝ)) (Fin.init x) *
    ((fun _ : Base 2 => (1 : ℝ)) x * preparedPolynomial 2
      (![fun _ => -1, fun _ => 0] : Fin 2 → Base 1 → ℝ) (Fin.init x, x (Fin.last 1)))) 0 := by
  refine hasMonomialCharts_of_centered (fun d' _ => hasMonomialCharts_mul_of_order_le
      (fun _ _ => hasMonomialCharts_of_analyticAt) d') analyticAt_const
    (normalCrossingAt_of_ne_zero analyticAt_const one_ne_zero).not_eventuallyEq_zero
    analyticAt_const one_ne_zero (fun i => ?_) (fun i hi => ?_) fun h => ?_
  · fin_cases i <;> exact analyticAt_const
  · fin_cases i
    · simp at hi
    · rfl
  · have h0 := distinguishedPolynomial_coeff 2
      (![fun _ => -1, fun _ => 0] : Fin 2 → Base 1 → ℝ) (Fin.init (0 : Base 2)) 0
    rw [h, coeff_X_pow] at h0
    simp at h0

/-- **The order step** for `1 · y²`, whose slice `y ↦ y²` has order exactly `2` at the
origin. Appendix D.1 of arXiv:2510.22026v2. -/
example : HasMonomialCharts (fun x : Base 2 => (fun _ : Base 1 => (1 : ℝ)) (Fin.init x) *
    x (Fin.last 1) ^ 2) 0 := by
  have hord : analyticOrderAt (fun y : ℝ => (Fin.snoc (0 : Base 1) y : Base 2) (Fin.last 1) ^ 2)
      0 = (2 : ℕ) := by
    simp only [Fin.snoc_last]
    simpa [Pi.pow_def] using analyticOrderAt_pow (analyticAt_id (𝕜 := ℝ) (z := 0)) 2
  exact hasMonomialCharts_of_order_eq (fun _ _ => hasMonomialCharts_of_analyticAt)
    (fun d' _ => hasMonomialCharts_mul_of_order_le
      (fun _ _ => hasMonomialCharts_of_analyticAt) d') analyticAt_const
    (normalCrossingAt_of_ne_zero analyticAt_const one_ne_zero).not_eventuallyEq_zero
    (((ContinuousLinearMap.proj (Fin.last 1) : Base 2 →L[ℝ] ℝ).analyticAt 0).fun_pow 2) hord

/-- **The induction on the order** for `1 · y²` at `(0, 1)`, where the slice
`y ↦ (1 + y)²` has order `0 ≤ 2`. Appendix D.1 of arXiv:2510.22026v2. -/
example : HasMonomialCharts (fun x : Base 2 => (fun _ : Base 1 => (1 : ℝ)) (Fin.init x) *
    x (Fin.last 1) ^ 2) (Fin.snoc 0 1) := by
  refine hasMonomialCharts_mul_of_order_le (fun _ _ => hasMonomialCharts_of_analyticAt) 2 _ _ _
    analyticAt_const
    (normalCrossingAt_of_ne_zero analyticAt_const one_ne_zero).not_eventuallyEq_zero
    (((ContinuousLinearMap.proj (Fin.last 1) : Base 2 →L[ℝ] ℝ).analyticAt _).fun_pow 2) ?_
  have h0 : analyticOrderAt (fun y : ℝ => (Fin.snoc (Fin.init (Fin.snoc (0 : Base 1) 1 : Base 2))
      ((Fin.snoc (0 : Base 1) 1 : Base 2) (Fin.last 1) + y) : Base 2) (Fin.last 1) ^ 2) 0 = 0 := by
    rw [analyticOrderAt_eq_zero]
    right
    simp only [Fin.snoc_last, add_zero]
    norm_num
  rw [h0]
  exact zero_le

/-- **Local uniformization** for the cusp `x₀² - x₁³` at the origin: it is analytic, and
it does not vanish identically since it is `-t³` on the line `x₀ = 0`. Appendix D.1 of
arXiv:2510.22026v2. -/
example : HasMonomialCharts (fun x : Base 2 => x 0 ^ 2 - x 1 ^ 3) 0 := by
  have hA : ∀ i : Fin 2, AnalyticAt ℝ (fun x : Base 2 => x i) 0 :=
    fun i => (ContinuousLinearMap.proj i : Base 2 →L[ℝ] ℝ).analyticAt 0
  refine hasMonomialCharts_of_analyticAt (((hA 0).fun_pow 2).fun_sub ((hA 1).fun_pow 3))
    fun h => ?_
  have hc : Tendsto (fun t : ℝ => (Fin.snoc (0 : Base 1) t : Base 2)) (𝓝[≠] 0) (𝓝 0) :=
    ((continuous_snoc.comp (continuous_const.prodMk continuous_id)).tendsto' 0 0
      snoc_zero_zero).mono_left nhdsWithin_le_nhds
  obtain ⟨t, ht, ht0⟩ := ((hc.eventually h).and self_mem_nhdsWithin).exists
  change (0 : ℝ) ^ 2 - t ^ 3 = 0 at ht
  exact ht0 (pow_eq_zero_iff (n := 3) (by norm_num) |>.1 (by linarith))

end Transformer.Landscape.Uniformization
