/-
# Local uniformization: the Newton step, first chart

Near `(w', 0)` let `F = M(x') · V(x) · P(x', y)` with `M = μ · (x' - w')^γ`,
`P = y^d + Σ_i B_i(x') y^i` and `B_i = β_i · (x' - w')^{α_i}`. If `(d - i) • p ≤ q • α_i`
whenever `β_i ≢ 0`, with equality at an index `k` where `β_k` does not vanish, then in
the chart `(t, Y) ↦ (ram w' q b t, t^p Y)` the function `F` is a function of `t` with
normal crossings times a unit times a centered monic polynomial in `Y` of degree `d`
which is not `Y^d`. Such products have monomial charts by induction on the degree.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2.
-/

import Transformer.Uniformization.Prepared
import Transformer.Uniformization.Ramification
import Mathlib.Algebra.Order.Sub.Prod

open Filter Set Polynomial
open scoped BigOperators Topology

namespace Transformer.Uniformization

open Transformer.AnalyticPreparation (Base preparedPolynomial distinguishedPolynomial)

variable {n : ℕ}

/-- `t^a = (t^p)^j · t^{a - j p}` when `j p ≤ a`. -/
theorem monomial_split {p a : Fin n → ℕ} {j : ℕ} (h : j • p ≤ a) (t : Base n) :
    monomial a t = monomial p t ^ j * monomial (a - j • p) t := by
  conv_lhs => rw [← add_tsub_cancel_of_le h]
  rw [monomial_add_exp, monomial_smul_exp]

/-- The first Newton chart `(t, Y) ↦ (ram w' q b t, t^p Y)`. -/
def chartA (w' : Base n) (q : ℕ) (b : Fin n → Bool) (p : Fin n → ℕ) (s : Base (n + 1)) :
    Base (n + 1) :=
  Fin.snoc (ram w' q b (Fin.init s)) (monomial p (Fin.init s) * s (Fin.last n))

theorem analyticAt_chartA (w' : Base n) (q : ℕ) (b : Fin n → Bool) (p : Fin n → ℕ)
    (s : Base (n + 1)) : AnalyticAt ℝ (chartA w' q b p) s := by
  have h1 : AnalyticAt ℝ (fun s : Base (n + 1) => ram w' q b (Fin.init s)) s :=
    (analyticAt_ram w' q b _).comp (analyticAt_init s)
  have h2 : AnalyticAt ℝ (fun s : Base (n + 1) => monomial p (Fin.init s)) s :=
    (analyticAt_monomial p _).comp (analyticAt_init s)
  exact AnalyticAt.snoc h1 (h2.fun_mul (analyticAt_last s))

/-- The coefficients of the polynomial in the first Newton chart. -/
def coeffA (w' : Base n) (q : ℕ) (b : Fin n → Bool) (p : Fin n → ℕ) {d : ℕ}
    (β : Fin d → Base n → ℝ) (α : Fin d → Fin n → ℕ) (i : Fin d) (t : Base n) : ℝ :=
  β i (ram w' q b t) * monomial (α i) (signVec b) * monomial (q • α i - (d - i) • p) t

/-- In the first Newton chart the polynomial is `(t^p)^d` times the polynomial with
coefficients `coeffA`. Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem preparedPolynomial_chartA {d : ℕ} {w' : Base n} {q : ℕ} {b : Fin n → Bool}
    {p : Fin n → ℕ} {B β : Fin d → Base n → ℝ} {α : Fin d → Fin n → ℕ} {t : Base n}
    (hB : ∀ i, B i (ram w' q b t) = β i (ram w' q b t) * monomial (α i) (ram w' q b t - w'))
    (hterm : ∀ i, β i (ram w' q b t) = 0 ∨ (d - i) • p ≤ q • α i) (Y : ℝ) :
    preparedPolynomial d B (ram w' q b t, monomial p t * Y) =
      monomial p t ^ d * preparedPolynomial d (coeffA w' q b p β α) (t, Y) := by
  simp only [preparedPolynomial, mul_add, Finset.mul_sum, mul_pow]
  congr 1
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [hB, monomial_ram_sub]
  rcases hterm i with h | h
  · simp [coeffA, h]
  · have hd : monomial p t ^ d = monomial p t ^ (d - i) * monomial p t ^ (i : ℕ) := by
      rw [← pow_add, Nat.sub_add_cancel i.isLt.le]
    rw [coeffA, hd, monomial_split h t]
    ring

/-- In the first Newton chart the function has monomial charts at every point whose
image lies in the ball where the data are defined. Auxiliary for Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2. -/
theorem hasMonomialCharts_chartA {d : ℕ}
    (hcent : ∀ (m : Base n → ℝ) (V : Base (n + 1) → ℝ) (c : Fin d → Base n → ℝ)
      (x₀ : Base (n + 1)), AnalyticAt ℝ m (Fin.init x₀) → ¬ m =ᶠ[𝓝 (Fin.init x₀)] 0 →
      AnalyticAt ℝ V x₀ → V x₀ ≠ 0 → (∀ i, AnalyticAt ℝ (c i) (Fin.init x₀)) →
      (∀ i : Fin d, (i : ℕ) + 1 = d → c i (Fin.init x₀) = 0) →
      distinguishedPolynomial d c (Fin.init x₀) ≠ X ^ d →
      HasMonomialCharts (fun x => m (Fin.init x) *
        (V x * preparedPolynomial d c (Fin.init x, x (Fin.last n)))) x₀)
    {w' : Base n} {ε : ℝ} {M μ : Base n → ℝ} {γ : Fin n → ℕ} {V : Base (n + 1) → ℝ}
    {B β : Fin d → Base n → ℝ} {α : Fin d → Fin n → ℕ} {q : ℕ} {p : Fin n → ℕ} {k : Fin d}
    (hμ : ∀ x' ∈ Metric.ball w' ε,
      AnalyticAt ℝ μ x' ∧ μ x' ≠ 0 ∧ M x' = μ x' * monomial γ (x' - w'))
    (hβ : ∀ x' ∈ Metric.ball w' ε, ∀ i,
      AnalyticAt ℝ (β i) x' ∧ B i x' = β i x' * monomial (α i) (x' - w'))
    (hterm : ∀ i, (∀ x', β i x' = 0) ∨ (d - i) • p ≤ q • α i)
    (hcenter : ∀ i : Fin d, (i : ℕ) + 1 = d → ∀ x', β i x' = 0)
    (hk : (d - k) • p = q • α k) (hβk : ∀ x' ∈ Metric.ball w' ε, β k x' ≠ 0)
    (hV : ∀ x ∈ Metric.ball (Fin.snoc w' 0 : Base (n + 1)) ε, AnalyticAt ℝ V x ∧ V x ≠ 0)
    (b : Fin n → Bool) {s₀ : Base (n + 1)} (hs₀ : ram w' q b (Fin.init s₀) ∈ Metric.ball w' ε)
    (hs₀' : chartA w' q b p s₀ ∈ Metric.ball (Fin.snoc w' 0 : Base (n + 1)) ε) :
    HasMonomialCharts ((fun x => M (Fin.init x) *
      (V x * preparedPolynomial d B (Fin.init x, x (Fin.last n)))) ∘ chartA w' q b p) s₀ := by
  obtain ⟨hμa, hμ0, -⟩ := hμ _ hs₀
  let u : Base n → ℝ := fun t => μ (ram w' q b t) * monomial γ (signVec b)
  have hu : AnalyticAt ℝ u (Fin.init s₀) :=
    (hμa.comp (analyticAt_ram w' q b _)).fun_mul analyticAt_const
  have hu0 : u (Fin.init s₀) ≠ 0 := mul_ne_zero hμ0 (monomial_signVec_ne_zero b γ)
  let mA : Base n → ℝ := fun t => u t * monomial (q • γ + d • p) t
  have hmA : NormalCrossingAt mA (Fin.init s₀) :=
    normalCrossingAt_of_eq_mul_monomial 0 _ hu hu0
      (Eventually.of_forall fun t => by rw [sub_zero])
  have hc : ∀ i, AnalyticAt ℝ (coeffA w' q b p β α i) (Fin.init s₀) := fun i =>
    (((hβ _ hs₀ i).1.comp (analyticAt_ram w' q b _)).fun_mul analyticAt_const).fun_mul
      (analyticAt_monomial _ _)
  have hne : distinguishedPolynomial d (coeffA w' q b p β α) (Fin.init s₀) ≠ X ^ d := by
    intro h
    have h1 := distinguishedPolynomial_coeff d (coeffA w' q b p β α) (Fin.init s₀) k
    rw [h, coeff_X_pow, ite_eq_right (ne_of_lt k.isLt)] at h1
    have h2 : coeffA w' q b p β α k (Fin.init s₀) ≠ 0 := by
      rw [coeffA, hk, tsub_self, monomial_zero_exp, mul_one]
      exact mul_ne_zero (hβk _ hs₀) (monomial_signVec_ne_zero b _)
    exact h2 h1.symm
  have hmain := hcent mA (V ∘ chartA w' q b p) (coeffA w' q b p β α) s₀ hmA.analyticAt
    hmA.not_eventuallyEq_zero ((hV _ hs₀').1.comp (analyticAt_chartA w' q b p s₀))
    (hV _ hs₀').2 hc (fun i hi => by rw [coeffA, hcenter i hi, zero_mul, zero_mul]) hne
  refine hmain.congr ?_
  have hopen : ∀ᶠ s in 𝓝 s₀, ram w' q b (Fin.init s) ∈ Metric.ball w' ε :=
    ((analyticAt_ram w' q b _).comp (analyticAt_init s₀)).continuousAt.preimage_mem_nhds
      (Metric.isOpen_ball.mem_nhds hs₀)
  filter_upwards [hopen] with s hs
  obtain ⟨-, -, hM⟩ := hμ _ hs
  simp only [Function.comp, chartA, Fin.init_snoc, Fin.snoc_last]
  rw [hM, preparedPolynomial_chartA (fun i => (hβ _ hs i).2)
    (fun i => (hterm i).imp (fun h => h _) id), monomial_ram_sub]
  simp only [mA, u, monomial_add_exp, monomial_smul_exp]
  ring

/-- The exponent `p = (d! / (d - k)) • α_k` of the Newton step: `(d - i) • p ≤ d! • α_i`
when `α_k` has the least ratio. -/
theorem newton_exponent_le {d : ℕ} {S : Finset (Fin d)} {α : Fin d → Fin n → ℕ} {k : Fin d}
    (hmin : ∀ i ∈ S, (d - i) • α k ≤ (d - k) • α i) {i : Fin d} (hi : i ∈ S) :
    (d - i) • ((d.factorial / (d - k)) • α k) ≤ d.factorial • α i := by
  intro j
  have hdvd : d - k ∣ d.factorial := Nat.dvd_factorial (by have := k.isLt; omega) (by omega)
  have h := hmin i hi j
  simp only [Pi.smul_apply, smul_eq_mul] at h ⊢
  calc (d - i) * (d.factorial / (d - k) * α k j) = d.factorial / (d - k) * ((d - i) * α k j) :=
        by ring
    _ ≤ d.factorial / (d - k) * ((d - k) * α i j) := Nat.mul_le_mul_left _ h
    _ = d.factorial * α i j := by rw [← mul_assoc, Nat.div_mul_cancel hdvd]

theorem newton_exponent_eq {d : ℕ} (α : Fin d → Fin n → ℕ) (k : Fin d) :
    (d - k) • ((d.factorial / (d - k)) • α k) = d.factorial • α k := by
  rw [smul_smul, Nat.mul_div_cancel' (Nat.dvd_factorial (by have := k.isLt; omega) (by omega))]

/-- The domain `|t j| ≤ τ`, `|Y| ≤ R` of the first Newton charts. -/
def boxA (τ R : ℝ) : Set (Base (n + 1)) :=
  univ.pi fun l => Icc (-(Fin.snoc (fun _ => τ) R : Base (n + 1)) l)
    ((Fin.snoc (fun _ => τ) R : Base (n + 1)) l)

theorem mem_boxA {τ R : ℝ} {s : Base (n + 1)} :
    s ∈ boxA τ R ↔ (∀ j, |(Fin.init s : Base n) j| ≤ τ) ∧ |s (Fin.last n)| ≤ R := by
  simp only [boxA, Set.mem_univ_pi, Fin.forall_fin_succ', Fin.snoc_castSucc, Fin.snoc_last,
    Set.mem_Icc, abs_le, Fin.init]

theorem isCompact_boxA (τ R : ℝ) : IsCompact (boxA (n := n) τ R) :=
  isCompact_univ_pi fun _ => isCompact_Icc

end Transformer.Uniformization
