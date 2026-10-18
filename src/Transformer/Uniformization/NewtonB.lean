/-
# Local uniformization: the Newton step, principalized charts

With the data of `NewtonA`, in a chart `s ↦ (ram w' q b t, y)` with `(t, y) = s^E` in
which `t^p = y · r` and `|r| ≤ c₀` (`Principal`), the polynomial is
`P = y^d (1 + e)` with `|e| ≤ Σ_i C_i c₀ ≤ 1/2`, where `|β_i| ≤ C_i`. Hence `F` is a unit
times a monomial in `s`: it has normal crossings at every point of the chart. Auxiliary
for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2.
-/

import Transformer.Uniformization.NewtonA
import Transformer.Uniformization.Principal

open Filter Set Polynomial
open scoped BigOperators Topology

namespace Transformer.Uniformization

open Transformer.AnalyticPreparation (Base preparedPolynomial)

variable {n : ℕ}

/-- The principalized Newton chart `s ↦ (ram w' q b t, y)` with `(t, y) = s^E`. -/
def chartB (w' : Base n) (q : ℕ) (b : Fin n → Bool) (E : Fin (n + 1) → Fin (n + 1) → ℕ)
    (s : Base (n + 1)) : Base (n + 1) :=
  Fin.snoc (ram w' q b (Fin.init (monomialMap E s))) (monomialMap E s (Fin.last n))

theorem analyticAt_chartB (w' : Base n) (q : ℕ) (b : Fin n → Bool)
    (E : Fin (n + 1) → Fin (n + 1) → ℕ) (s : Base (n + 1)) :
    AnalyticAt ℝ (chartB w' q b E) s := by
  have hT : AnalyticAt ℝ (fun s : Base (n + 1) => (Fin.init (monomialMap E s) : Base n)) s :=
    (analyticAt_init _).comp (analyticAt_monomialMap E s)
  exact AnalyticAt.snoc ((analyticAt_ram w' q b _).comp hT)
    ((analyticAt_last _).comp (analyticAt_monomialMap E s))

/-- The correction `e` in `P = y^d (1 + e)` in a principalized Newton chart. -/
def errB (w' : Base n) (q : ℕ) (b : Fin n → Bool) (p : Fin n → ℕ) {d : ℕ}
    (β : Fin d → Base n → ℝ) (α : Fin d → Fin n → ℕ) (ρ : Fin (n + 1) → ℕ)
    (E : Fin (n + 1) → Fin (n + 1) → ℕ) (s : Base (n + 1)) : ℝ :=
  ∑ i : Fin d, β i (ram w' q b (Fin.init (monomialMap E s))) * monomial (α i) (signVec b) *
    monomial ρ s ^ (d - i) * monomial (q • α i - (d - i) • p) (Fin.init (monomialMap E s))

/-- Where `t^p = y · r`, the polynomial is `y^d (1 + e)`. Auxiliary for Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2. -/
theorem preparedPolynomial_chartB {d : ℕ} {w' : Base n} {q : ℕ} {b : Fin n → Bool}
    {p : Fin n → ℕ} {B β : Fin d → Base n → ℝ} {α : Fin d → Fin n → ℕ} {t : Base n} {y r : ℝ}
    (hB : ∀ i, B i (ram w' q b t) = β i (ram w' q b t) * monomial (α i) (ram w' q b t - w'))
    (hterm : ∀ i, β i (ram w' q b t) = 0 ∨ (d - i) • p ≤ q • α i)
    (htp : monomial p t = y * r) :
    preparedPolynomial d B (ram w' q b t, y) = y ^ d * (1 + ∑ i : Fin d, β i (ram w' q b t) *
      monomial (α i) (signVec b) * r ^ (d - i) * monomial (q • α i - (d - i) • p) t) := by
  simp only [preparedPolynomial, mul_add, mul_one, Finset.mul_sum]
  congr 1
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [hB, monomial_ram_sub]
  rcases hterm i with h | h
  · simp [h]
  · have hd : y ^ d = y ^ (d - i) * y ^ (i : ℕ) := by
      rw [← pow_add, Nat.sub_add_cancel i.isLt.le]
    rw [hd, monomial_split h t, htp, mul_pow]
    ring

/-- In a principalized Newton chart the function has normal crossings at every point
whose image lies in the ball where the data are defined. Auxiliary for Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2. -/
theorem normalCrossingAt_chartB {d : ℕ} {w' : Base n} {ε c₀ : ℝ} {M μ : Base n → ℝ}
    {γ : Fin n → ℕ} {V : Base (n + 1) → ℝ} {B β : Fin d → Base n → ℝ} {α : Fin d → Fin n → ℕ}
    {q : ℕ} {p : Fin n → ℕ} {Cβ : Fin d → ℝ}
    (hμ : ∀ x' ∈ Metric.ball w' ε,
      AnalyticAt ℝ μ x' ∧ μ x' ≠ 0 ∧ M x' = μ x' * monomial γ (x' - w'))
    (hβ : ∀ x' ∈ Metric.ball w' ε, ∀ i,
      AnalyticAt ℝ (β i) x' ∧ B i x' = β i x' * monomial (α i) (x' - w'))
    (hterm : ∀ i, (∀ x', β i x' = 0) ∨ (d - i) • p ≤ q • α i)
    (hβC : ∀ x' ∈ Metric.ball w' ε, ∀ i, |β i x'| ≤ Cβ i) (hc₀ : 0 ≤ c₀) (hc₀1 : c₀ ≤ 1)
    (hC : (∑ i, Cβ i) * c₀ ≤ 1 / 2)
    (hV : ∀ x ∈ Metric.ball (Fin.snoc w' 0 : Base (n + 1)) ε, AnalyticAt ℝ V x ∧ V x ≠ 0)
    (b : Fin n → Bool) {E : Fin (n + 1) → Fin (n + 1) → ℕ} {ρ : Fin (n + 1) → ℕ}
    (hE : ∀ s, monomial p (Fin.init (monomialMap E s)) =
      monomialMap E s (Fin.last n) * monomial ρ s)
    {s₀ : Base (n + 1)} (hρ : |monomial ρ s₀| ≤ c₀)
    (ht₀ : ∀ j, |(Fin.init (monomialMap E s₀) : Base n) j| ≤ 1)
    (hs₀ : ram w' q b (Fin.init (monomialMap E s₀)) ∈ Metric.ball w' ε)
    (hs₀' : chartB w' q b E s₀ ∈ Metric.ball (Fin.snoc w' 0 : Base (n + 1)) ε) :
    NormalCrossingAt ((fun x => M (Fin.init x) *
      (V x * preparedPolynomial d B (Fin.init x, x (Fin.last n)))) ∘ chartB w' q b E) s₀ := by
  have hT : AnalyticAt ℝ (fun s : Base (n + 1) => (Fin.init (monomialMap E s) : Base n)) s₀ :=
    (analyticAt_init _).comp (analyticAt_monomialMap E s₀)
  let X : Base (n + 1) → Base n := fun s => ram w' q b (Fin.init (monomialMap E s))
  have hX : AnalyticAt ℝ X s₀ := (analyticAt_ram w' q b _).comp hT
  obtain ⟨hμa, hμ0, -⟩ := hμ _ hs₀
  let u : Base (n + 1) → ℝ := fun s => μ (ram w' q b (Fin.init (monomialMap E s))) *
    monomial γ (signVec b) * V (chartB w' q b E s) * (1 + errB w' q b p β α ρ E s)
  have hu : AnalyticAt ℝ u s₀ := by
    refine (((hμa.comp (f := X) hX).fun_mul analyticAt_const).fun_mul
      ((hV _ hs₀').1.comp (analyticAt_chartB w' q b E s₀))).fun_mul
      (analyticAt_const.fun_add (Finset.analyticAt_fun_sum _ fun i _ => ?_))
    exact ((((hβ _ hs₀ i).1.comp (f := X) hX).fun_mul analyticAt_const).fun_mul
      ((analyticAt_monomial ρ s₀).fun_pow _)).fun_mul ((analyticAt_monomial _ _).comp hT)
  have herr : |errB w' q b p β α ρ E s₀| ≤ 1 / 2 := by
    refine (Finset.abs_sum_le_sum_abs _ _).trans ((Finset.sum_le_sum fun i _ => ?_).trans
      ((Finset.sum_mul _ _ _).symm.le.trans hC))
    rw [abs_mul, abs_mul, abs_mul, abs_pow, abs_monomial_signVec, mul_one]
    have h1 := hβC _ hs₀ i
    have h2 : |monomial ρ s₀| ^ (d - i : ℕ) ≤ c₀ :=
      (pow_le_of_le_one (abs_nonneg _) (hρ.trans hc₀1) (by have := i.isLt; omega)).trans hρ
    have h3 : |monomial (q • α i - (d - i) • p) (Fin.init (monomialMap E s₀))| ≤ 1 := by
      have h := abs_monomial_le (p := q • α i - (d - i) • p) (b := fun _ => 1) ht₀
      rwa [monomial_const, one_pow] at h
    have hC0 : 0 ≤ Cβ i := (abs_nonneg _).trans h1
    calc _ ≤ Cβ i * c₀ * 1 :=
          mul_le_mul (mul_le_mul h1 h2 (by positivity) hC0) h3 (abs_nonneg _)
            (mul_nonneg hC0 hc₀)
      _ = Cβ i * c₀ := mul_one _
  have hu0 : u s₀ ≠ 0 := by
    refine mul_ne_zero (mul_ne_zero (mul_ne_zero hμ0 (monomial_signVec_ne_zero b γ))
      (hV _ hs₀').2) ?_
    have := abs_le.1 herr
    linarith
  refine normalCrossingAt_of_eq_mul_monomial 0 (∑ l, (Fin.snoc (q • γ) d : Fin (n + 1) → ℕ) l • E l)
    hu hu0 ?_
  have hopen : ∀ᶠ s in 𝓝 s₀, ram w' q b (Fin.init (monomialMap E s)) ∈ Metric.ball w' ε :=
    hX.continuousAt.preimage_mem_nhds (Metric.isOpen_ball.mem_nhds hs₀)
  filter_upwards [hopen] with s hs
  obtain ⟨-, -, hM⟩ := hμ _ hs
  simp only [Function.comp, chartB, Fin.init_snoc, Fin.snoc_last, sub_zero]
  rw [hM, preparedPolynomial_chartB (fun i => (hβ _ hs i).2)
    (fun i => (hterm i).imp (fun h => h _) id) (hE s), monomial_ram_sub,
    ← monomial_monomialMap, monomial_snoc]
  simp only [u, errB, chartB]
  ring

/-- The charts of the Newton step cover the ball of radius `τA ^ q` around `(w', 0)`: with
`x' = ram w' q b t`, the first charts cover `|y| ≤ c₀⁻¹ |t^p|` and the principalized ones
the rest. Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem newton_cover {w' : Base n} {q : ℕ} (hq : q ≠ 0) {p : Fin n → ℕ} {c₀ τA τB : ℝ}
    (hc₀ : 0 < c₀) (hc₀1 : c₀ ≤ 1) (hτA : 0 < τA) (hτA1 : τA ≤ 1) (hτAB : τA ≤ τB)
    {Cp : Set ((Fin (n + 1) → Fin (n + 1) → ℕ) × (Fin (n + 1) → ℕ) × Set (Base (n + 1)))}
    (hCp : PrincipalCharts c₀ p Cp) {x : Base (n + 1)}
    (hx : ‖x - (Fin.snoc w' 0 : Base (n + 1))‖ ≤ τA ^ q) :
    (∃ b, ∃ s ∈ boxA τA c₀⁻¹, chartA w' q b p s = x) ∨
      ∃ b, ∃ e ∈ Cp, ∃ s ∈ e.2.2 ∩ monomialMap e.1 ⁻¹' Metric.closedBall 0 τB,
        chartB w' q b e.1 s = x := by
  have hx' : ∀ j, |(Fin.init x : Base n) j - w' j| ≤ τA ^ q := fun j => by
    have h := (norm_le_pi_norm (x - (Fin.snoc w' 0 : Base (n + 1))) j.castSucc).trans hx
    simpa [Fin.init] using h
  have hy : |x (Fin.last n)| ≤ τA := by
    have h := (norm_le_pi_norm (x - (Fin.snoc w' 0 : Base (n + 1))) (Fin.last n)).trans hx
    simp only [Pi.sub_apply, Fin.snoc_last, sub_zero, Real.norm_eq_abs] at h
    exact h.trans (pow_le_of_le_one hτA.le hτA1 hq)
  obtain ⟨b, t, hbt, ht⟩ := exists_ram_eq w' (Fin.init x) hq hτA.le hx'
  by_cases hA : |x (Fin.last n)| ≤ c₀⁻¹ * |monomial p t|
  · left
    have hmul : monomial p t * (x (Fin.last n) / monomial p t) = x (Fin.last n) := by
      rcases eq_or_ne (monomial p t) 0 with h0 | h0
      · rw [h0, abs_zero, mul_zero] at hA
        rw [h0, abs_nonpos_iff.1 hA, zero_mul]
      · exact mul_div_cancel₀ _ h0
    refine ⟨b, Fin.snoc t (x (Fin.last n) / monomial p t), mem_boxA.2 ⟨fun j => ?_, ?_⟩, ?_⟩
    · rw [Fin.init_snoc]
      exact ht j
    · rw [Fin.snoc_last]
      rcases eq_or_ne (monomial p t) 0 with h0 | h0
      · rw [h0, div_zero, abs_zero]
        positivity
      · rw [abs_div, div_le_iff₀ (abs_pos.2 h0)]
        exact hA
    · rw [chartA, Fin.init_snoc, Fin.snoc_last, hbt, hmul, Fin.snoc_init_self]
  · right
    push Not at hA
    have hlt : |monomial p t| < c₀ * |x (Fin.last n)| := by
      have h := mul_lt_mul_of_pos_left hA hc₀
      rwa [← mul_assoc, mul_inv_cancel₀ hc₀.ne', one_mul] at h
    obtain ⟨e, he, s, hs, hes⟩ := hCp.2.2 t (x (Fin.last n)) (fun j => (ht j).trans hτA1)
      (hy.trans (hτA1.trans ((one_le_inv₀ hc₀).2 hc₀1))) hlt
    refine ⟨b, e, he, s, ⟨hs, ?_⟩, ?_⟩
    · rw [Set.mem_preimage, hes, Metric.mem_closedBall, dist_zero_right]
      exact norm_snoc_le ((pi_norm_le_iff_of_nonneg (hτA.le.trans hτAB)).2 fun j =>
        (ht j).trans hτAB) (hy.trans hτAB)
    · rw [chartB, hes, Fin.init_snoc, Fin.snoc_last, hbt, Fin.snoc_init_self]

end Transformer.Uniformization
