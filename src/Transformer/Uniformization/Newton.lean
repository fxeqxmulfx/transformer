/-
# Local uniformization: the Newton step

At a point `(w', 0)` where the coefficients `B_i = β_i · (x' - w')^{α_i}` of a centered
polynomial all vanish, choose `k` minimizing `|α_i| / (d - i)` among the `i` with
`β_i ≢ 0` (the multiples of the `α_i` are comparable); with `q = d!` and
`p = (q / (d - k)) • α_k`, the first Newton charts (`NewtonA`) cover `|y| ≤ R |t^p|` and
the principalized charts (`NewtonB`) cover `R |t^p| < |y|`, for every sign vector of the
ramified coordinates. Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2.
-/

import Transformer.Uniformization.NewtonB
import Transformer.Uniformization.Comparable

open Filter Set Polynomial
open scoped BigOperators Topology

namespace Transformer.Uniformization

open Transformer.AnalyticPreparation (Base preparedPolynomial distinguishedPolynomial)

variable {n : ℕ}

/-- **The Newton step.** A function of `x'` with normal crossings at `w'` times a unit
times a centered monic polynomial whose coefficients have normal crossings at `w'`,
with comparable multiples `(d - i') • α_i`, `(d - i) • α_{i'}` of their exponents and
all vanishing at `w'`, has monomial charts at `(w', 0)`, given monomial charts for such
products whose polynomial at the point is not `y^d`. Auxiliary for Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2. -/
theorem hasMonomialCharts_newton {d : ℕ}
    (hcent : ∀ (m : Base n → ℝ) (V : Base (n + 1) → ℝ) (c : Fin d → Base n → ℝ)
      (x₀ : Base (n + 1)), AnalyticAt ℝ m (Fin.init x₀) → ¬ m =ᶠ[𝓝 (Fin.init x₀)] 0 →
      AnalyticAt ℝ V x₀ → V x₀ ≠ 0 → (∀ i, AnalyticAt ℝ (c i) (Fin.init x₀)) →
      (∀ i : Fin d, (i : ℕ) + 1 = d → c i (Fin.init x₀) = 0) →
      distinguishedPolynomial d c (Fin.init x₀) ≠ X ^ d →
      HasMonomialCharts (fun x => m (Fin.init x) *
        (V x * preparedPolynomial d c (Fin.init x, x (Fin.last n)))) x₀)
    {w' : Base n} {M μ : Base n → ℝ} {γ : Fin n → ℕ} {V : Base (n + 1) → ℝ}
    {B β : Fin d → Base n → ℝ} {α : Fin d → Fin n → ℕ} {S : Finset (Fin d)}
    (hμ : AnalyticAt ℝ μ w') (hμ0 : μ w' ≠ 0)
    (hM : ∀ᶠ x' in 𝓝 w', M x' = μ x' * monomial γ (x' - w'))
    (hβ : ∀ i, AnalyticAt ℝ (β i) w')
    (hB : ∀ i, ∀ᶠ x' in 𝓝 w', B i x' = β i x' * monomial (α i) (x' - w'))
    (hβS : ∀ i ∈ S, β i w' ≠ 0) (hβ0 : ∀ i ∉ S, ∀ x', β i x' = 0) (hα : ∀ i ∈ S, α i ≠ 0)
    (hcenter : ∀ i : Fin d, (i : ℕ) + 1 = d → i ∉ S)
    (hcomp : ∀ i ∈ S, ∀ i' ∈ S,
      (d - i') • α i ≤ (d - i) • α i' ∨ (d - i) • α i' ≤ (d - i') • α i)
    (hV : AnalyticAt ℝ V (Fin.snoc w' 0)) (hV0 : V (Fin.snoc w' 0) ≠ 0) :
    HasMonomialCharts (fun x => M (Fin.init x) *
      (V x * preparedPolynomial d B (Fin.init x, x (Fin.last n)))) (Fin.snoc w' 0) := by
  classical
  rcases S.eq_empty_or_nonempty with hS | hS
  · have hμ' : AnalyticAt ℝ μ (Fin.init (Fin.snoc w' 0 : Base (n + 1))) := by
      rwa [Fin.init_snoc]
    refine hasMonomialCharts_of_normalCrossingAt ⟨Fin.snoc γ d, fun x => μ (Fin.init x) * V x,
      (hμ'.comp (analyticAt_init _)).fun_mul hV, ?_, ?_⟩
    · show μ (Fin.init (Fin.snoc w' 0 : Base (n + 1))) * V (Fin.snoc w' 0) ≠ 0
      rw [Fin.init_snoc]
      exact mul_ne_zero hμ0 hV0
    filter_upwards [(tendsto_init_snoc w' 0).eventually (hM.and (eventually_all.2 hB))]
      with x ⟨hMx, hBx⟩
    have hP : preparedPolynomial d B (Fin.init x, x (Fin.last n)) = x (Fin.last n) ^ d := by
      simp [preparedPolynomial, hBx, hβ0 _ (by simp [hS])]
    rw [hMx, hP, monomial_snoc, init_sub, Fin.init_snoc, Pi.sub_apply, Fin.snoc_last, sub_zero]
    ring
  obtain ⟨k, hkS, hmin⟩ := exists_min_ratio S hS (fun i : Fin d => d - i)
    (fun i _ => by have := i.isLt; omega) α hcomp
  have hq : d.factorial ≠ 0 := Nat.factorial_ne_zero d
  have hterm : ∀ i, (∀ x', β i x' = 0) ∨
      (d - i) • ((d.factorial / (d - k)) • α k) ≤ d.factorial • α i := fun i => by
    by_cases hi : i ∈ S
    · exact Or.inr (newton_exponent_le hmin hi)
    · exact Or.inl (hβ0 i hi)
  have hp : ∑ j, ((d.factorial / (d - k)) • α k) j ≠ 0 := by
    obtain ⟨j, hj⟩ : ∃ j, α k j ≠ 0 := by
      by_contra h
      push Not at h
      exact hα k hkS (funext h)
    have hdvd : d - k ∣ d.factorial := Nat.dvd_factorial (by have := k.isLt; omega) (by omega)
    have hpos : 0 < d.factorial / (d - k) :=
      Nat.div_pos (Nat.le_of_dvd (Nat.factorial_pos d) hdvd) (by have := k.isLt; omega)
    intro h0
    have h := Finset.sum_eq_zero_iff.1 h0 j (Finset.mem_univ j)
    simp only [Pi.smul_apply, smul_eq_mul] at h
    exact mul_ne_zero hpos.ne' hj h
  -- the ball where all data are defined
  have hev1 : ∀ᶠ x' in 𝓝 w', (AnalyticAt ℝ μ x' ∧ μ x' ≠ 0 ∧
      M x' = μ x' * monomial γ (x' - w')) ∧
      (∀ i, AnalyticAt ℝ (β i) x' ∧ B i x' = β i x' * monomial (α i) (x' - w')) ∧
      β k x' ≠ 0 ∧ ∀ i, |β i x'| ≤ |β i w'| + 1 := by
    refine (hμ.eventually_analyticAt.and ((hμ.continuousAt.eventually_ne hμ0).and hM)).and
      ((eventually_all.2 fun i => (hβ i).eventually_analyticAt.and (hB i)).and
      (((hβ k).continuousAt.eventually_ne (hβS k hkS)).and (eventually_all.2 fun i => ?_)))
    filter_upwards [(hβ i).continuousAt.preimage_mem_nhds
      (Metric.closedBall_mem_nhds (β i w') one_pos)] with x' hx'
    rw [Set.mem_preimage, Metric.mem_closedBall, Real.dist_eq] at hx'
    linarith [abs_sub_abs_le_abs_sub (β i x') (β i w')]
  obtain ⟨ε, hε, hgood, hgoodV⟩ : ∃ ε > 0, (∀ x' ∈ Metric.ball w' ε,
      (AnalyticAt ℝ μ x' ∧ μ x' ≠ 0 ∧ M x' = μ x' * monomial γ (x' - w')) ∧
      (∀ i, AnalyticAt ℝ (β i) x' ∧ B i x' = β i x' * monomial (α i) (x' - w')) ∧
      β k x' ≠ 0 ∧ ∀ i, |β i x'| ≤ |β i w'| + 1) ∧
      ∀ x ∈ Metric.ball (Fin.snoc w' 0 : Base (n + 1)) ε, AnalyticAt ℝ V x ∧ V x ≠ 0 := by
    obtain ⟨ε₁, hε₁, h₁⟩ := Metric.eventually_nhds_iff_ball.1 hev1
    obtain ⟨ε₂, hε₂, h₂⟩ := Metric.eventually_nhds_iff_ball.1
      (hV.eventually_analyticAt.and (hV.continuousAt.eventually_ne hV0))
    exact ⟨min ε₁ ε₂, lt_min hε₁ hε₂,
      fun x' hx' => h₁ x' (Metric.ball_subset_ball (min_le_left _ _) hx'),
      fun x hx => h₂ x (Metric.ball_subset_ball (min_le_right _ _) hx)⟩
  -- the constants
  obtain ⟨c₀, hc₀, hc₀1, hC⟩ : ∃ c₀ : ℝ, 0 < c₀ ∧ c₀ ≤ 1 ∧
      (∑ i, (|β i w'| + 1)) * c₀ ≤ 1 / 2 := by
    have hCs : 0 ≤ ∑ i, (|β i w'| + 1) := Finset.sum_nonneg fun i _ => by positivity
    refine ⟨1 / (2 * ((∑ i, (|β i w'| + 1)) + 1)), by positivity, ?_, ?_⟩
    · rw [div_le_one (by positivity)]
      linarith
    · rw [mul_one_div, div_le_div_iff₀ (by positivity) (by norm_num)]
      linarith
  obtain ⟨τB, hτB, hτB1, hτBε⟩ : ∃ τB : ℝ, 0 < τB ∧ τB ≤ 1 ∧ τB < ε :=
    ⟨min 1 (ε / 2), lt_min one_pos (half_pos hε), min_le_left _ _,
      (min_le_right _ _).trans_lt (half_lt_self hε)⟩
  obtain ⟨τA, hτA, hτAB, hτAR⟩ : ∃ τA : ℝ, 0 < τA ∧ τA ≤ τB ∧ τA * c₀⁻¹ = τB :=
    ⟨τB * c₀, mul_pos hτB hc₀, mul_le_of_le_one_right hτB.le hc₀1,
      mul_inv_cancel_right₀ hc₀.ne' τB⟩
  have hτA1 : τA ≤ 1 := hτAB.trans hτB1
  set p : Fin n → ℕ := (d.factorial / (d - k)) • α k with hpdef
  obtain ⟨Cp, hCpc⟩ := exists_principalCharts hc₀ p
  -- the charts
  let SA : Set ((Base (n + 1) → Base (n + 1)) × Set (Base (n + 1))) :=
    Set.range fun b : Fin n → Bool => (chartA w' d.factorial b p, boxA τA c₀⁻¹)
  let SB : Set ((Base (n + 1) → Base (n + 1)) × Set (Base (n + 1))) :=
    (fun be : (Fin n → Bool) × ((Fin (n + 1) → Fin (n + 1) → ℕ) × (Fin (n + 1) → ℕ) ×
      Set (Base (n + 1))) => (chartB w' d.factorial be.1 be.2.1,
        be.2.2.2 ∩ monomialMap be.2.1 ⁻¹' Metric.closedBall 0 τB)) '' (univ ×ˢ Cp)
  refine hasMonomialCharts_of_cover (SA ∪ SB)
    ((Set.finite_range _).union ((Set.finite_univ.prod hCpc.1).image _)) ?_ ?_
  · rintro c (⟨b, rfl⟩ | ⟨⟨b, e⟩, ⟨-, he⟩, rfl⟩)
    · refine ⟨isCompact_boxA _ _, fun s _ => analyticAt_chartA w' _ b p s, fun s₀ hs₀ => ?_⟩
      obtain ⟨ht, hY⟩ := mem_boxA.1 hs₀
      have hram : ‖ram w' d.factorial b (Fin.init s₀) - w'‖ ≤ τA :=
        (norm_ram_sub_le w' _ b hτA.le ht).trans (pow_le_of_le_one hτA.le hτA1 hq)
      have h1 : ram w' d.factorial b (Fin.init s₀) ∈ Metric.ball w' ε := by
        rw [Metric.mem_ball, dist_eq_norm]
        linarith
      have h2 : chartA w' d.factorial b p s₀ ∈ Metric.ball (Fin.snoc w' 0 : Base (n + 1)) ε := by
        rw [Metric.mem_ball, dist_eq_norm, chartA, snoc_sub_snoc, sub_zero]
        refine lt_of_le_of_lt (norm_snoc_le (hram.trans hτAB) ?_) hτBε
        have hmp : |monomial p (Fin.init s₀)| ≤ τA := by
          have h := abs_monomial_le (p := p) (b := fun _ => τA) ht
          rw [monomial_const] at h
          exact h.trans (pow_le_of_le_one hτA.le hτA1 hp)
        rw [abs_mul, ← hτAR]
        exact mul_le_mul hmp hY (abs_nonneg _) hτA.le
      exact hasMonomialCharts_chartA hcent (fun x' hx' => (hgood x' hx').1)
        (fun x' hx' => (hgood x' hx').2.1) hterm (fun i hi => hβ0 i (hcenter i hi))
        (newton_exponent_eq α k) (fun x' hx' => (hgood x' hx').2.2.1) hgoodV b h1 h2
    · obtain ⟨hK, hE, hρ⟩ := hCpc.2.1 e he
      refine ⟨hK.inter_right (Metric.isClosed_closedBall.preimage (continuous_monomialMap _)),
        fun s _ => analyticAt_chartB w' _ b e.1 s, fun s₀ hs₀ => ?_⟩
      have hnorm : ‖monomialMap e.1 s₀‖ ≤ τB := by
        simpa [Metric.mem_closedBall, dist_zero_right] using hs₀.2
      have ht : ∀ j, |(Fin.init (monomialMap e.1 s₀) : Base n) j| ≤ τB := fun j =>
        (norm_le_pi_norm _ j).trans ((norm_init_le _).trans hnorm)
      have hram : ‖ram w' d.factorial b (Fin.init (monomialMap e.1 s₀)) - w'‖ ≤ τB :=
        (norm_ram_sub_le w' _ b hτB.le ht).trans (pow_le_of_le_one hτB.le hτB1 hq)
      have h1 : ram w' d.factorial b (Fin.init (monomialMap e.1 s₀)) ∈ Metric.ball w' ε := by
        rw [Metric.mem_ball, dist_eq_norm]
        linarith
      have h2 : chartB w' d.factorial b e.1 s₀ ∈
          Metric.ball (Fin.snoc w' 0 : Base (n + 1)) ε := by
        rw [Metric.mem_ball, dist_eq_norm, chartB, snoc_sub_snoc, sub_zero]
        exact lt_of_le_of_lt (norm_snoc_le hram ((abs_last_le _).trans hnorm)) hτBε
      exact hasMonomialCharts_of_normalCrossingAt (normalCrossingAt_chartB
        (fun x' hx' => (hgood x' hx').1) (fun x' hx' => (hgood x' hx').2.1) hterm
        (fun x' hx' => (hgood x' hx').2.2.2) hc₀.le hc₀1 hC hgoodV b hE (hρ s₀ hs₀.1)
        (fun j => (ht j).trans hτB1) h1 h2)
  · refine Filter.mem_of_superset
      (Metric.closedBall_mem_nhds (Fin.snoc w' 0 : Base (n + 1)) (pow_pos hτA d.factorial)) ?_
    intro x hx
    rw [Metric.mem_closedBall, dist_eq_norm] at hx
    rw [Set.mem_iUnion₂]
    rcases newton_cover hq hc₀ hc₀1 hτA hτA1 hτAB hCpc hx with
      ⟨b, s, hs, hsx⟩ | ⟨b, e, he, s, hs, hsx⟩
    · exact ⟨_, Or.inl ⟨b, rfl⟩, s, hs, hsx⟩
    · exact ⟨_, Or.inr ⟨(b, e), ⟨Set.mem_univ _, he⟩, rfl⟩, s, hs, hsx⟩

end Transformer.Uniformization
