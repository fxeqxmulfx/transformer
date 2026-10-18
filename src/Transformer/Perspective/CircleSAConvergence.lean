/-
# Almost every solution of SA on the circle synchronizes

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5, §4 and
Appendix A.

SA in angles is `θ̇ = -G(θ) ∇f(θ)` on `EucSpace n`, with `f = -𝖤_β` and
`G = diag(n²/Z_i)` (`angularSA_iff`).  Each
`Z_i = Σ_k e^{β cos(θ_i - θ_k)}` lies between `n e^{-|β|}` and `n e^{|β|}`
(`circlePartition_mem_Icc`), so `G` lies between the same two constants
(`le_inner_saGain`, `norm_saGain_le`), and every solution descends `f` at
these rates (`isDescentCurve_angularSA`).  `f` is analytic, so it satisfies the
Łojasiewicz inequality at every point
(`exists_isLojasiewiczAt_neg_eucTorusEnergy`), and it is invariant under the
periods `2π e_k`, which span the space.  So every solution has a limit point
up to the periods, and converges, to a critical point of `𝖤_β`
(`exists_tendsto_angularSA`, by `SaddleAvoidance.DescentConvergence`).  For
`β > 0` a critical point is a global maximum, hence synchronized, or a strict
saddle (`strict_saddle`), to which almost no solution converges
(`ae_not_tendsto_angularSA_of_isStrictSaddle`): almost every solution
synchronizes (`ae_tendsto_synchronized_of_angularSA`).

Source: arXiv:2312.10794v5, §4, after `thm: beta.interval`: "since all the
intervening functions and metrics are real-analytic", Łojasiewicz's theorem
makes every trajectory converge, and the center-stable manifold theorem
(`l:nosaddleconv`) keeps almost every one away from the strict saddles; SA is
the gradient flow of §3.4, for another metric (Appendix A, Remark
`r:extensiontoSAcase`).
-/

import Transformer.Perspective.CircleConvergence
import Transformer.SaddleAvoidance.DescentConvergence

open scoped BigOperators Topology
open Real Filter MeasureTheory

namespace Transformer
namespace Perspective

variable (n : ℕ)

/-- **The partition function lies between `n e^{-|β|}` and `n e^{|β|}`**:
each of its `n` terms `e^{β cos(θ_i - θ_k)}` lies between `e^{-|β|}` and
`e^{|β|}`. -/
theorem circlePartition_mem_Icc (β : ℝ) (θ : Angles n) (i : Idx n) :
    n * exp (-|β|) ≤ circlePartition n β θ i ∧ circlePartition n β θ i ≤ n * exp |β| := by
  have hb : ∀ x, |β * cos x| ≤ |β| := fun x => by
    rw [abs_mul]
    exact mul_le_of_le_one_right (abs_nonneg β) (abs_cos_le_one x)
  have h1 := Finset.card_nsmul_le_sum Finset.univ (fun k => exp (β * cos (θ i - θ k)))
    (exp (-|β|)) fun k _ => exp_le_exp.2 (neg_le_of_abs_le (hb _))
  have h2 := Finset.sum_le_card_nsmul Finset.univ (fun k => exp (β * cos (θ i - θ k)))
    (exp |β|) fun k _ => exp_le_exp.2 (le_of_abs_le (hb _))
  simp only [Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] at h1 h2
  exact ⟨h1, h2⟩

/-- **The gain `G = diag(n²/Z_i)` is at least `n e^{-|β|}`**:
`⟪G v, v⟫ = Σ_i (n²/Z_i) v_i² ≥ n e^{-|β|} ‖v‖²`, since `Z_i ≤ n e^{|β|}`. -/
theorem le_inner_saGain (β : ℝ) (ψ v : EucSpace n) :
    n * exp (-|β|) * ‖v‖ ^ 2 ≤ inner ℝ (saGain n β ψ v) v := by
  rw [inner_saGain, EuclideanSpace.norm_sq_eq, Finset.mul_sum]
  refine Finset.sum_le_sum fun i _ => ?_
  have hZ0 := circlePartition_pos n β ψ.ofLp i
  have hl : n * exp (-|β|) ≤ (n : ℝ) ^ 2 / circlePartition n β ψ.ofLp i := by
    rw [le_div_iff₀ hZ0]
    calc n * exp (-|β|) * circlePartition n β ψ.ofLp i ≤ n * exp (-|β|) * (n * exp |β|) :=
          mul_le_mul_of_nonneg_left (circlePartition_mem_Icc n β ψ.ofLp i).2 (by positivity)
      _ = (n : ℝ) ^ 2 := by
          rw [mul_mul_mul_comm, ← exp_add, neg_add_cancel, exp_zero, mul_one, sq]
  calc (n : ℝ) * exp (-|β|) * ‖v.ofLp i‖ ^ 2 = n * exp (-|β|) * (v.ofLp i * v.ofLp i) := by
        rw [Real.norm_eq_abs, sq_abs, sq]
    _ ≤ (n : ℝ) ^ 2 / circlePartition n β ψ.ofLp i * (v.ofLp i * v.ofLp i) :=
        mul_le_mul_of_nonneg_right hl (mul_self_nonneg _)
    _ = _ := by ring

/-- **The gain `G = diag(n²/Z_i)` is at most `n e^{|β|}`**:
`‖G v‖ ≤ n e^{|β|} ‖v‖`, since `Z_i ≥ n e^{-|β|}`. -/
theorem norm_saGain_le (β : ℝ) (ψ v : EucSpace n) :
    ‖saGain n β ψ v‖ ≤ n * exp |β| * ‖v‖ := by
  refine (sq_le_sq₀ (norm_nonneg _) (by positivity)).1 ?_
  rw [mul_pow, EuclideanSpace.norm_sq_eq, EuclideanSpace.norm_sq_eq, Finset.mul_sum]
  refine Finset.sum_le_sum fun i _ => ?_
  have hZ0 := circlePartition_pos n β ψ.ofLp i
  have hl : (n : ℝ) ^ 2 / circlePartition n β ψ.ofLp i ≤ n * exp |β| := by
    rw [div_le_iff₀ hZ0]
    calc (n : ℝ) ^ 2 = n * exp |β| * (n * exp (-|β|)) := by
          rw [mul_mul_mul_comm, ← exp_add, add_neg_cancel, exp_zero, mul_one, sq]
      _ ≤ n * exp |β| * circlePartition n β ψ.ofLp i :=
          mul_le_mul_of_nonneg_left (circlePartition_mem_Icc n β ψ.ofLp i).1 (by positivity)
  rw [saGain_apply, Real.norm_eq_abs, Real.norm_eq_abs, sq_abs, sq_abs]
  calc ((n : ℝ) ^ 2 / circlePartition n β ψ.ofLp i * v.ofLp i) ^ 2
      = ((n : ℝ) ^ 2 / circlePartition n β ψ.ofLp i) ^ 2 * v.ofLp i ^ 2 := mul_pow _ _ _
    _ ≤ (n * exp |β|) ^ 2 * v.ofLp i ^ 2 := mul_le_mul_of_nonneg_right
        (pow_le_pow_left₀ (div_nonneg (sq_nonneg _) hZ0.le) hl 2) (sq_nonneg _)

/-- **SA in angles descends `-𝖤_β`** at the rates `n e^{-|β|}` and
`n e^{|β|}`: it is the flow `θ̇ = -G(θ) ∇f(θ)` (`angularSA_iff`) for a metric
`G` between them. -/
theorem isDescentCurve_angularSA {β : ℝ} (hβ : β ≠ 0) {θ : ℝ → Angles n}
    (hθ : angularSA n β θ) :
    SaddleAvoidance.IsDescentCurve (fun φ => -eucTorusEnergy n β φ) (n * exp (-|β|))
      (n * exp |β|) 0 fun t => WithLp.toLp 2 (θ t) :=
  SaddleAvoidance.isDescentCurve_of_metric (le_inner_saGain n β) (norm_saGain_le n β)
    fun t _ => (angularSA_iff n β hβ θ).1 hθ t

/-- **SA in angles has a solution from every start.** -/
theorem exists_angularSA {β : ℝ} (hβ : β ≠ 0) (θ₀ : Angles n) :
    ∃ θ : ℝ → Angles n, angularSA n β θ ∧ θ 0 = θ₀ := by
  obtain ⟨L, hM, hL⟩ := exists_lipschitzWith_saField n β
  refine ⟨_, angularSA_metricGradientFlow n β hβ (WithLp.toLp 2 θ₀), ?_⟩
  simp [SaddleAvoidance.metricGradientFlow,
    (SaddleAvoidance.isFlowField_neg_metricGradient hL hM).flow_self]

/-- **Every solution of SA in angles converges, to a critical point of
`𝖤_β`**, for `β ≠ 0`: it descends `f = -𝖤_β` (`isDescentCurve_angularSA`) and
has a limit point up to the integer combinations of the periods `2π e_k`,
which span the space and leave `f` invariant; there `f` satisfies the
Łojasiewicz inequality (`exists_isLojasiewiczAt_neg_eucTorusEnergy`). -/
theorem exists_tendsto_angularSA {β : ℝ} (hβ : β ≠ 0) (hn : 0 < n) {θ : ℝ → Angles n}
    (hθ : angularSA n β θ) :
    ∃ θ' : Angles n, IsCriticalTorusEnergy n β θ' ∧ Tendsto θ atTop (𝓝 θ') := by
  have hf : ContDiff ℝ 2 fun φ => -eucTorusEnergy n β φ := (contDiff_eucTorusEnergy n β).neg
  have hx := isDescentCurve_angularSA n hβ hθ
  have hc₁ : (0 : ℝ) < n * exp (-|β|) := mul_pos (Nat.cast_pos.2 hn) (exp_pos _)
  have hΛ : ∀ v ∈ Set.range fun k : Idx n => (2 * π) • EuclideanSpace.single k (1 : ℝ),
      ∀ y, -eucTorusEnergy n β (y + v) = -eucTorusEnergy n β y := by
    rintro _ ⟨k, rfl⟩ y
    simp only [eucTorusEnergy_add_period]
  obtain ⟨z, hz⟩ := SaddleAvoidance.exists_frequently_norm_add_sub_lt_of_span_eq_top
    (span_periods n) fun t => WithLp.toLp 2 (θ t)
  obtain ⟨a, -, ha1, hza⟩ := exists_isLojasiewiczAt_neg_eucTorusEnergy n β z
  obtain ⟨w, hw⟩ := hx.exists_tendsto_of_isLojasiewiczAt (hf.differentiable (by norm_num)) hc₁
    (by positivity) (fun v hv => SaddleAvoidance.periodic_of_mem_span
      (f := fun φ : EucSpace n => -eucTorusEnergy n β φ) hΛ hv) ha1 hza hz
  have hw0 := hx.gradient_eq_zero_of_tendsto (hf.differentiable (by norm_num)) hc₁
    (SaddleAvoidance.contDiff_gradient hf).continuous.continuousAt hw
  exact ⟨w.ofLp, (gradient_neg_eucTorusEnergy_eq_zero_iff n hβ w).1 hw0,
    ((PiLp.continuous_ofLp 2 _).tendsto w).comp hw⟩

/-- **Almost every solution of SA on the circle synchronizes**, for every
`β > 0`: for Lebesgue-almost every initial angle vector `θ₀ ∈ ℝⁿ` there is a
`θ'` with `θ'_i - θ'_j ∈ 2πℤ` to which every solution of SA in angles with
`θ(0) = θ₀` converges.  The solution from `θ₀` is unique
(`toLp_eq_metricGradientFlow`) and converges to a critical point
(`exists_tendsto_angularSA`); almost surely not to a strict saddle
(`ae_not_tendsto_angularSA_of_isStrictSaddle`), so to a global maximum of
`𝖤_β` (`strict_saddle`), which is synchronized
(`synchronized_of_isGlobalMaxTorusEnergy`).

Source: arXiv:2312.10794v5, §4, after `thm: beta.interval`, and Appendix A,
Remark `r:extensiontoSAcase`: the arguments of Appendix A for SA.  The common
limit on `𝕋` is a limit `θ'` in `ℝⁿ` whose coordinates agree modulo `2π`. -/
theorem ae_tendsto_synchronized_of_angularSA {β : ℝ} (hβ : 0 < β) :
    ∀ᵐ θ₀ : Angles n, ∃ θ' : Angles n, (∀ i j, ∃ k : ℤ, θ' i - θ' j = k * (2 * π)) ∧
      ∀ θ : ℝ → Angles n, angularSA n β θ → θ 0 = θ₀ → Tendsto θ atTop (𝓝 θ') := by
  rcases Nat.eq_zero_or_pos n with rfl | hn
  · refine Eventually.of_forall fun θ₀ => ⟨θ₀, fun i => i.elim0, fun θ _ _ => ?_⟩
    rw [show θ = fun _ => θ₀ from funext fun _ => Subsingleton.elim _ _]
    exact tendsto_const_nhds
  filter_upwards [ae_not_tendsto_angularSA_of_isStrictSaddle n β hβ.ne'] with θ₀ hθ₀
  obtain ⟨θs, hsol, h0⟩ := exists_angularSA n hβ.ne' θ₀
  have hall : ∀ θ : ℝ → Angles n, angularSA n β θ → θ 0 = θ₀ → θ = θs := fun θ hθ hθ0 =>
    funext fun t => by
      have h := toLp_eq_metricGradientFlow n β hβ.ne' hθ t
      rw [hθ0, ← h0, ← toLp_eq_metricGradientFlow n β hβ.ne' hsol t] at h
      exact congrArg WithLp.ofLp h
  obtain ⟨θ', hcrit, hlim⟩ := exists_tendsto_angularSA n hβ.ne' hn hsol
  refine ⟨θ', ?_, fun θ hθ hθ0 => by rwa [hall θ hθ hθ0]⟩
  by_contra hs
  exact hθ₀ θs hsol h0 θ' hcrit (strict_saddle n β hβ θ' hcrit fun hmax => hs
    (synchronized_of_isGlobalMaxTorusEnergy n hβ hmax)) hlim

/-- The hypotheses of the theorems above hold for two tokens at `β = 1`: the
consensus `θ_i = c` solves SA in angles. -/
example (c : ℝ) : (1 : ℝ) ≠ 0 ∧ (0 : ℝ) < 1 ∧ 0 < 2 ∧ angularSA 2 1 (fun _ _ => c) :=
  ⟨one_ne_zero, one_pos, two_pos, fun t i => by simpa using hasDerivAt_const t c⟩

end Perspective
end Transformer
