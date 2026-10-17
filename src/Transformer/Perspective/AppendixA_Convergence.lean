/-
# Appendix A — every trajectory of `e:gradfl` converges to a critical point

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*, Appendix A, proof of `p:beta0`.

The proof of `p:beta0` cites Łojasiewicz's theorem for the convergence of every
trajectory of `e:gradfl` to a single critical point of `𝖤_0`.  Here it follows
from the finite length of the trajectories (`exists_norm_sub_le`): a trajectory
is Cauchy, so it converges, and at the limit the gradient vanishes
(`exists_tendsto_isCriticalE0`).  For the gradient is small at arbitrarily late
times (`frequently_tangentSq_lt`): `‖s‖²` grows at rate
`(4/n) Σ_i ‖Proj_{x_i} s‖²` but stays below `n²`.
The solutions of `e:Snonres0` are those of `e:gradfl` at half speed
(`eq_e0Flow_half`), so they converge as well: the source's sentence is
`exists_tendsto_isCriticalE0_of_beta0Dynamics`.
-/

import Transformer.Perspective.AppendixA_FiniteLength

open Filter Set
open scoped Topology

namespace Transformer
namespace Perspective

variable {d n : ℕ}

/-- **The gradient is small at arbitrarily late times.**  Were
`Σ_i ‖Proj_{x_i} s‖² ≥ η` from time `T` on, `‖s‖²` would grow by `(4/n) η L`
over `[T, T + L]` (mean value theorem), past `n²` for `L = n³/η`. -/
theorem frequently_tangentSq_lt (hn : 0 < n) {X : ℝ → SphereTuple d n}
    (hX : E0GradientAscent d n X) {η : ℝ} (hη : 0 < η) :
    ∃ᶠ t in atTop, tangentSq (X t) < η := by
  rw [Filter.frequently_atTop]
  intro T
  by_contra h
  simp only [not_exists, not_and, not_lt] at h
  have hn' : (0 : ℝ) < n := Nat.cast_pos.mpr hn
  set L : ℝ := (n : ℝ) ^ 3 / η with hL
  have hL0 : 0 < L := by positivity
  obtain ⟨c, hc, hslope⟩ := exists_hasDerivAt_eq_slope
    (fun s => ‖∑ j : Idx n, (X s j : EucSpace d)‖ ^ 2)
    (fun s => 4 * (n : ℝ)⁻¹ * tangentSq (X s)) (by linarith : T < T + L)
    (fun s _ => (hasDerivAt_norm_sum_sq hn hX s).continuousAt.continuousWithinAt)
    (fun s _ => hasDerivAt_norm_sum_sq hn hX s)
  rw [add_sub_cancel_left, eq_div_iff hL0.ne'] at hslope
  have hQ1 : ‖∑ j : Idx n, (X (T + L) j : EucSpace d)‖ ^ 2 ≤ (n : ℝ) ^ 2 :=
    pow_le_pow_left₀ (norm_nonneg _) (norm_sum_coe_le _) 2
  have hQ0 : 0 ≤ ‖∑ j : Idx n, (X T j : EucSpace d)‖ ^ 2 := sq_nonneg _
  have hηL : 4 * (n : ℝ)⁻¹ * η * L = 4 * (n : ℝ) ^ 2 := by
    rw [hL]
    field_simp
  have hG : 4 * (n : ℝ)⁻¹ * η * L ≤ 4 * (n : ℝ)⁻¹ * tangentSq (X c) * L := by
    gcongr
    exact h c hc.1.le
  nlinarith

/-- The hypotheses of `frequently_tangentSq_lt` hold for a consensus of three
particles on the circle and `η = 1`. -/
example (x : SSphere 2) : 0 < 3 ∧ E0GradientAscent 2 3 (fun _ => signedTuple x ∅) ∧
    (0 : ℝ) < 1 :=
  ⟨by norm_num, E0GradientAscent_const _ (isCriticalE0_signedTuple x ∅), one_pos⟩

/-- **Every trajectory of `e:gradfl` converges to a critical point of `𝖤_0`.**
The trajectory has finite length (`exists_norm_sub_le`), so it is Cauchy and
converges, and at the limit `Σ_i ‖Proj_{x_i} s‖²`, arbitrarily small at late
times (`frequently_tangentSq_lt`), vanishes. -/
theorem exists_tendsto_isCriticalE0 {X : ℝ → SphereTuple d n}
    (hX : E0GradientAscent d n X) :
    ∃ Z : SphereTuple d n, IsCriticalE0 d n Z ∧
      ∀ i : Idx n, Tendsto (fun t => (X t i : EucSpace d) - (Z i : EucSpace d)) atTop (𝓝 0) := by
  rcases Nat.eq_zero_or_pos n with rfl | hn
  · exact ⟨X 0, fun i => i.elim0, fun i => i.elim0⟩
  obtain ⟨K, T, hKT⟩ := exists_norm_sub_le hn hX
  have hφ : Tendsto (fun t => K * √((⨆ s, ‖∑ j : Idx n, (X s j : EucSpace d)‖ ^ 2)
      - ‖∑ j : Idx n, (X t j : EucSpace d)‖ ^ 2)) atTop (𝓝 0) := by
    have h := ((tendsto_const_nhds (x := ⨆ s, ‖∑ j : Idx n, (X s j : EucSpace d)‖ ^ 2)).sub
      (tendsto_atTop_ciSup (monotone_norm_sum_sq X hX) (bddAbove_norm_sum_sq X))).sqrt.const_mul K
    simpa using h
  have hcauchy : ∀ i : Idx n, CauchySeq fun t => (X t i : EucSpace d) := by
    intro i
    rw [Metric.cauchySeq_iff']
    intro ε hε
    obtain ⟨N, hN⟩ := eventually_atTop.mp
      ((hφ.eventually (gt_mem_nhds hε)).and (eventually_ge_atTop T))
    refine ⟨N, fun t ht => ?_⟩
    rw [dist_eq_norm]
    exact (hKT i N (hN N le_rfl).2 t ht).trans_lt (hN N le_rfl).1
  choose z hz using fun i => cauchySeq_tendsto_of_complete (hcauchy i)
  have hz1 : ∀ i, z i ∈ Metric.sphere (0 : EucSpace d) 1 := fun i =>
    Metric.isClosed_sphere.mem_of_tendsto (hz i) (Eventually.of_forall fun t => (X t i).2)
  set Z : SphereTuple d n := fun i => ⟨z i, hz1 i⟩
  have hXZ : Tendsto X atTop (𝓝 Z) :=
    tendsto_pi_nhds.mpr fun i => tendsto_subtype_rng.mpr (hz i)
  refine ⟨Z, isCriticalE0_of_tangentSq_eq_zero ?_, fun i => tendsto_sub_nhds_zero_iff.mpr (hz i)⟩
  by_contra hne
  have hpos : 0 < tangentSq Z := lt_of_le_of_ne (tangentSq_nonneg Z) (Ne.symm hne)
  obtain ⟨t, h1, h2⟩ := ((frequently_tangentSq_lt hn hX (half_pos hpos)).and_eventually
    (((continuous_tangentSq.tendsto Z).comp hXZ).eventually
      (lt_mem_nhds (half_lt_self hpos)))).exists
  exact lt_asymm h1 h2

/-- The hypothesis of `exists_tendsto_isCriticalE0` holds for a consensus of
three particles on the circle. -/
example (x : SSphere 2) : E0GradientAscent 2 3 (fun _ => signedTuple x ∅) :=
  E0GradientAscent_const _ (isCriticalE0_signedTuple x ∅)

/-- **The flow of `e:gradfl` solves `e:gradfl`:** the `SA` flow at `β = 0` run
at double speed. -/
theorem E0GradientAscent_e0Flow (X₀ : SphereTuple d n) :
    E0GradientAscent d n (e0Flow X₀) := by
  intro t i
  have h := ((saFlow_spec 0 X₀).2 (2 * t) i).scomp t ((hasDerivAt_id' t).const_mul 2)
  refine h.congr_deriv ?_
  rw [SA_zero_drift, beta0Dynamics_smul d n (e0Flow X₀) t i, mul_one]
  rfl

/-- **`e:Snonres0` is `SA` at `β = 0`:** the attention weights are uniform. -/
theorem SA_zero_of_beta0Dynamics {X : ℝ → SphereTuple d n} (hX : beta0Dynamics d n X) :
    SA d n 0 X := fun t i => by
  rw [SA_zero_drift X t i]
  exact hX t i

/-- **The solutions of `e:Snonres0` are the flow of `e:gradfl` at half
speed.** -/
theorem eq_e0Flow_half {X : ℝ → SphereTuple d n} (hX : beta0Dynamics d n X) (t : ℝ) :
    X t = e0Flow (X 0) (t / 2) := by
  rw [e0Flow, mul_div_cancel₀ t two_ne_zero, saFlow_eq_of_SA 0 X (SA_zero_of_beta0Dynamics hX)]

/-- **The solutions of `e:Snonres0` converge with the flow of `e:gradfl`:** if
`e0Flow X₀ t → Z`, every solution of `e:Snonres0` from `X₀` tends to `Z`, being
that flow at half speed (`eq_e0Flow_half`). -/
theorem tendsto_of_beta0Dynamics {X₀ Z : SphereTuple d n}
    (hlim : ∀ i : Idx n,
      Tendsto (fun t => (e0Flow X₀ t i : EucSpace d) - (Z i : EucSpace d)) atTop (𝓝 0))
    {X : ℝ → SphereTuple d n} (hX0 : X 0 = X₀) (hX : beta0Dynamics d n X) (i : Idx n) :
    Tendsto (fun t => (X t i : EucSpace d) - (Z i : EucSpace d)) atTop (𝓝 0) := by
  refine ((hlim i).comp (tendsto_id.atTop_div_const two_pos)).congr fun t => ?_
  simp only [Function.comp_apply, id, eq_e0Flow_half hX t, hX0]

/-- **Every solution of `e:Snonres0` converges to a critical point of `𝖤_0`.**

The source: since `e:Snonres0` is the gradient ascent of the real-analytic
`𝖤_0` on the compact real-analytic manifold `(𝕊^{d-1})^n`, Łojasiewicz's
theorem implies that for any initial condition `X`, the solution `Φ^t(X)`
converges to some critical point `X^*` of `𝖤_0` as `t → +∞`.  Here the
gradient inequality of `𝖤_0` is explicit (`exists_int_mul_abs_sub_le`), and
the convergence is `exists_tendsto_isCriticalE0` for `e:gradfl`, run at half
speed (`tendsto_of_beta0Dynamics`).

Source: arXiv:2312.10794v5, Appendix A (proof of `p:beta0`), second
paragraph. -/
theorem exists_tendsto_isCriticalE0_of_beta0Dynamics {X : ℝ → SphereTuple d n}
    (hX : beta0Dynamics d n X) :
    ∃ Z : SphereTuple d n, IsCriticalE0 d n Z ∧
      ∀ i : Idx n, Tendsto (fun t => (X t i : EucSpace d) - (Z i : EucSpace d)) atTop (𝓝 0) := by
  obtain ⟨Z, hcrit, hlim⟩ := exists_tendsto_isCriticalE0 (E0GradientAscent_e0Flow (X 0))
  exact ⟨Z, hcrit, tendsto_of_beta0Dynamics hlim rfl hX⟩

/-- The hypotheses of `SA_zero_of_beta0Dynamics`, `eq_e0Flow_half`,
`tendsto_of_beta0Dynamics` and `exists_tendsto_isCriticalE0_of_beta0Dynamics`
hold for a single particle at rest, the flow of `e:gradfl` from it standing
still. -/
example : (∀ i : Idx 1, Tendsto (fun t => (e0Flow (fun _ => basePoint 0) t i : EucSpace 1)
      - (basePoint 0 : EucSpace 1)) atTop (𝓝 0)) ∧
    beta0Dynamics 1 1 (fun _ _ => basePoint 0) := by
  have hrest : beta0Dynamics 1 1 (fun _ _ => basePoint 0) := by
    intro t i
    have h0 := proj_smul_self (mem_sphere_zero_iff_norm.mp (basePoint 0).2) 1
    rw [one_smul] at h0
    simpa [h0] using hasDerivAt_const t ((basePoint 0 : SSphere 1) : EucSpace 1)
  refine ⟨fun i => ?_, hrest⟩
  simp [e0Flow, saFlow_consensus (n := 1) 0 one_pos (basePoint 0)]

end Perspective
end Transformer
