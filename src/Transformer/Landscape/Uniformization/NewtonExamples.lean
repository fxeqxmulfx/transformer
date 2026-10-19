module

public import Transformer.Landscape.Uniformization.Induction

/-!
# Local uniformization: examples for the Newton step

The polynomial `P = y² + x₀²` on `ℝ²`, with coefficients `(x₀², 0)`, units `(1, 0)` and
exponents `(2, 0)`, satisfies the hypotheses of the Newton step at the origin and of the
lemmas on its two kinds of charts. The centered induction hypothesis is discharged by
local uniformization itself. Auxiliary examples for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2.
-/

@[expose] public section

open Filter Set Polynomial
open scoped BigOperators Topology

namespace Transformer.Landscape.Uniformization

open Transformer.Landscape.AnalyticPreparation (Base preparedPolynomial)

/-- The coefficients `(x₀², 0)` of `P = y² + x₀²`. -/
local notation "Bₑ" => (![fun x' : Base 1 => x' 0 ^ 2, fun _ => 0] : Fin 2 → Base 1 → ℝ)
/-- The units `(1, 0)` of the coefficients. -/
local notation "βₑ" => (![fun _ => 1, fun _ => 0] : Fin 2 → Base 1 → ℝ)
/-- The exponents `(2, 0)` of the coefficients. -/
local notation "αₑ" => (![fun _ => 2, 0] : Fin 2 → Fin 1 → ℕ)

/-- The coefficients split as units times monomials, the exponent `2 · (1)` of the
leading term is `1 · (2)`, `t²` splits off `(t¹)²`, and at the origin `t = 0 · 0`.
Appendix D.1 of arXiv:2510.22026v2. -/
example : (∀ t : Base 1, monomial (fun _ => 2) t =
      monomial (fun _ => 1) t ^ 2 * monomial ((fun _ => 2) - 2 • fun _ => 1) t) ∧
    (2 - ((0 : Fin 2) : ℕ)) • ((Nat.factorial 2 / (2 - ((0 : Fin 2) : ℕ))) • αₑ 0) ≤
      Nat.factorial 2 • αₑ 0 ∧
    (∀ Y : ℝ, preparedPolynomial 2 Bₑ
        (ram (0 : Base 1) 1 (fun _ => true) 0, monomial (fun _ => 1) (0 : Base 1) * Y) =
      monomial (fun _ => 1) (0 : Base 1) ^ 2 *
        preparedPolynomial 2 (coeffA 0 1 (fun _ => true) (fun _ => 1) βₑ αₑ) (0, Y)) ∧
    preparedPolynomial 2 Bₑ (ram (0 : Base 1) 1 (fun _ => true) 0, 0) = 0 ^ 2 *
      (1 + ∑ i : Fin 2, βₑ i (ram 0 1 (fun _ => true) 0) *
        monomial (αₑ i) (signVec fun _ => true) * 0 ^ (2 - (i : ℕ)) *
          monomial (1 • αₑ i - (2 - (i : ℕ)) • fun _ => 1) (0 : Base 1)) := by
  have hB : ∀ (i : Fin 2) (x' : Base 1), Bₑ i x' = βₑ i x' * monomial (αₑ i) (x' - 0) := by
    intro i x'
    fin_cases i <;> simp [monomial]
  have hterm : ∀ i : Fin 2, βₑ i (ram 0 1 (fun _ => true) 0) = 0 ∨
      (2 - (i : ℕ)) • (fun _ => 1 : Fin 1 → ℕ) ≤ 1 • αₑ i := by
    intro i
    fin_cases i
    · exact Or.inr fun j => by simp
    · exact Or.inl rfl
  exact ⟨monomial_split fun j => by simp, newton_exponent_le (S := {0})
    (fun i hi => by rw [Finset.mem_singleton.1 hi]) (Finset.mem_singleton_self 0),
    preparedPolynomial_chartA (fun i => hB i _) hterm,
    preparedPolynomial_chartB (fun i => hB i _) hterm (by simp [monomial])⟩

/-- **The Newton step at the origin** for `1 · (1 · (y² + x₀²))`: the only coefficient
that does not vanish is `x₀²`, of exponent `2 ≠ 0`, and the leading coefficient
vanishes. Appendix D.1 of arXiv:2510.22026v2. -/
example : HasMonomialCharts (fun x : Base 2 => (fun _ : Base 1 => (1 : ℝ)) (Fin.init x) *
    ((fun _ : Base 2 => (1 : ℝ)) x * preparedPolynomial 2 Bₑ (Fin.init x, x (Fin.last 1))))
    (Fin.snoc 0 0) := by
  have hB : ∀ (i : Fin 2) (x' : Base 1), Bₑ i x' = βₑ i x' * monomial (αₑ i) (x' - 0) := by
    intro i x'
    fin_cases i <;> simp [monomial]
  refine hasMonomialCharts_newton (M := fun _ => 1) (μ := fun _ => 1) (γ := 0)
    (V := fun _ => 1) (β := βₑ) (α := αₑ) (S := {0})
    (fun m V c x₀ hm hm0 hV hV0 hc hcenter hne => hasMonomialCharts_of_centered
      (fun d' _ => hasMonomialCharts_mul_of_order_le
        (fun _ _ => hasMonomialCharts_of_analyticAt) d') hm hm0 hV hV0 hc hcenter hne)
    analyticAt_const one_ne_zero (Eventually.of_forall fun x' => by simp [monomial_zero_exp])
    (fun i => ?_) (fun i => Eventually.of_forall (hB i)) (fun i hi => ?_) (fun i hi x' => ?_)
    (fun i hi => ?_) (fun i hi hiS => ?_) (fun i hi i' hi' => ?_) analyticAt_const one_ne_zero
  · fin_cases i <;> exact analyticAt_const
  · rw [Finset.mem_singleton.1 hi]
    simp
  · fin_cases i
    · exact absurd (Finset.mem_singleton_self 0) hi
    · rfl
  · rw [Finset.mem_singleton.1 hi]
    exact fun h => two_ne_zero (congrFun h 0)
  · rw [Finset.mem_singleton.1 hiS] at hi
    simp at hi
  · rw [Finset.mem_singleton.1 hi, Finset.mem_singleton.1 hi']
    exact Or.inl le_rfl

/-- In the first Newton chart of `P = y² + x₀²`, for `q = 1` and `t^p = t`, the leading
exponent `2 · (1)` equals `1 · (2)` and the chart through the origin stays in the unit
ball. Appendix D.1 of arXiv:2510.22026v2. -/
example : HasMonomialCharts ((fun x : Base 2 => (fun _ : Base 1 => (1 : ℝ)) (Fin.init x) *
    ((fun _ : Base 2 => (1 : ℝ)) x * preparedPolynomial 2 Bₑ (Fin.init x, x (Fin.last 1)))) ∘
      chartA 0 1 (fun _ => true) (fun _ => 1)) 0 := by
  have hB : ∀ (i : Fin 2) (x' : Base 1), Bₑ i x' = βₑ i x' * monomial (αₑ i) (x' - 0) := by
    intro i x'
    fin_cases i <;> simp [monomial]
  refine hasMonomialCharts_chartA (ε := 1) (M := fun _ => 1) (μ := fun _ => 1) (γ := 0)
    (V := fun _ => 1) (β := βₑ) (α := αₑ) (q := 1) (p := fun _ => 1) (k := 0) (s₀ := 0)
    (fun m V c x₀ hm hm0 hV hV0 hc hcenter hne => hasMonomialCharts_of_centered
      (fun d' _ => hasMonomialCharts_mul_of_order_le
        (fun _ _ => hasMonomialCharts_of_analyticAt) d') hm hm0 hV hV0 hc hcenter hne)
    (fun x' _ => ⟨analyticAt_const, one_ne_zero, by simp [monomial_zero_exp]⟩)
    (fun x' _ i => ⟨?_, hB i x'⟩) (fun i => ?_) (fun i hi x' => ?_) (funext fun j => by simp)
    (fun x' _ => one_ne_zero) (fun x _ => ⟨analyticAt_const, one_ne_zero⟩) (fun _ => true) ?_ ?_
  · fin_cases i <;> exact analyticAt_const
  · fin_cases i
    · exact Or.inr fun j => by simp
    · exact Or.inl fun x' => rfl
  · fin_cases i
    · simp at hi
    · rfl
  · simp [ram, Fin.init]
  · have h0 : chartA (0 : Base 1) 1 (fun _ => true) (fun _ => 1) 0 = Fin.snoc 0 0 := by
      unfold chartA ram
      congr 1
      · funext j
        simp [Fin.init]
      · simp [Fin.init, monomial]
    rw [h0]
    exact Metric.mem_ball_self one_pos

/-- In the second Newton chart `blowTwo 0` of `P = y² + x₀²`, where `t = y · s`, the
bounds `1, 0` of the units and `c₀ = 1/2` of `s` keep `1 + …` away from zero, and the
function has normal crossings at the origin. Appendix D.1 of arXiv:2510.22026v2. -/
example : NormalCrossingAt ((fun x : Base 2 => (fun _ : Base 1 => (1 : ℝ)) (Fin.init x) *
    ((fun _ : Base 2 => (1 : ℝ)) x * preparedPolynomial 2 Bₑ (Fin.init x, x (Fin.last 1)))) ∘
      chartB 0 1 (fun _ => true) (blowTwo 0)) 0 := by
  have hB : ∀ (i : Fin 2) (x' : Base 1), Bₑ i x' = βₑ i x' * monomial (αₑ i) (x' - 0) := by
    intro i x'
    fin_cases i <;> simp [monomial]
  have hE0 : monomialMap (blowTwo (0 : Fin 1)) (0 : Base 2) = 0 := by
    rw [monomialMap_blowTwo, Pi.zero_apply, Pi.zero_apply, mul_zero]
    exact Function.update_eq_self _ _
  have hram : ram (0 : Base 1) 1 (fun _ => true) (Fin.init (0 : Base 2)) = 0 := by
    funext j
    simp [ram, Fin.init]
  refine normalCrossingAt_chartB (ε := 1) (c₀ := 1 / 2) (M := fun _ => 1) (μ := fun _ => 1)
    (γ := 0) (V := fun _ => 1) (β := βₑ) (α := αₑ) (q := 1) (p := fun _ => 1)
    (Cβ := ![1, 0]) (E := blowTwo 0) (s₀ := 0)
    (fun x' _ => ⟨analyticAt_const, one_ne_zero, by simp [monomial_zero_exp]⟩)
    (fun x' _ i => ⟨?_, hB i x'⟩) (fun i => ?_) (fun x' _ i => ?_) (by norm_num) (by norm_num)
    (by simp [Fin.sum_univ_two]) (fun x _ => ⟨analyticAt_const, one_ne_zero⟩) (fun _ => true)
    (monomial_blowTwo one_ne_zero) ?_ (fun j => by simp [hE0, Fin.init]) ?_ ?_
  · fin_cases i <;> exact analyticAt_const
  · fin_cases i
    · exact Or.inr fun j => by simp
    · exact Or.inl fun x' => rfl
  · fin_cases i <;> simp
  · rw [monomial_zero_point _ fun h => by simpa using congrFun h 0]
    norm_num
  · rw [hE0, hram]
    exact Metric.mem_ball_self one_pos
  · rw [chartB, hE0, hram]
    exact Metric.mem_ball_self one_pos

/-- With `q = 1`, `c₀ = 1` and `τ_A = τ_B = 1`, the origin lies in a chart of the first
or the second kind. Appendix D.1 of arXiv:2510.22026v2. -/
example : ∃ Cp, PrincipalCharts 1 (fun _ : Fin 1 => 1) Cp ∧
    ((∃ b, ∃ s ∈ boxA 1 1⁻¹, chartA (0 : Base 1) 1 b (fun _ => 1) s = Fin.snoc 0 0) ∨
      ∃ b, ∃ e ∈ Cp, ∃ s ∈ e.2.2 ∩ monomialMap e.1 ⁻¹' Metric.closedBall 0 1,
        chartB (0 : Base 1) 1 b e.1 s = Fin.snoc 0 0) := by
  obtain ⟨Cp, hCp⟩ := exists_principalCharts (n := 1) one_pos (fun _ => 1)
  exact ⟨Cp, hCp, newton_cover one_ne_zero one_pos le_rfl one_pos le_rfl le_rfl hCp (by simp)⟩

end Transformer.Landscape.Uniformization
