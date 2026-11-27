module

public import Transformer.Optimization.AnalyticLift
public import Transformer.Landscape.Lojasiewicz.AnalyticMonomialCharts
public import Mathlib.Analysis.Normed.Operator.Prod

/-! Transfer of the analytic Łojasiewicz gradient inequality. If no
uniform local bound holds, choose violations with exponent tending to
one and coefficient tending to zero. One analytic lift of a subsequence
then rules out those violations. -/

@[expose] public section

open Filter Metric Set Function
open scoped Topology

namespace Transformer.Optimization.HasAnalyticLift

open Transformer.Landscape.Lojasiewicz

variable {X : Type*} [NormedAddCommGroup X] [NormedSpace ℝ X] [FiniteDimensional ℝ X]

/-- The analytic gradient inequality in arbitrary finite-dimensional
normed coordinates. The norm is the operator norm of the derivative. -/
theorem analytic_local_fderiv_inequality {f : X → ℝ} {z : X} (hf : AnalyticAt ℝ f z) :
    ∃ a k : ℝ, 0 < a ∧ a < 1 ∧ 0 < k ∧
      ∀ᶠ x in 𝓝 z, |f x - f z| ^ a ≤ k * ‖fderiv ℝ f x‖ := by
  let E := EucSpace (Module.finrank ℝ X)
  let l : X ≃ₗ[ℝ] E := LinearEquiv.ofFinrankEq (R := ℝ) X E (by
    change Module.finrank ℝ X = Module.finrank ℝ (EuclideanSpace ℝ (Fin (Module.finrank ℝ X)))
    rw [finrank_euclideanSpace_fin])
  let e := l.toContinuousLinearEquiv
  let g : E → ℝ := f ∘ e.symm
  have hg : AnalyticAt ℝ g (e z) :=
    hf.comp_of_eq (e.symm.analyticAt (e z)) (e.symm_apply_apply z)
  obtain ⟨a, k, V, ha, ha₁, hk, hV, hzV, hv⟩ := analytic_gradient_inequality g (e z) hg
  let K := ‖(e.symm : E →L[ℝ] X)‖ + 1
  refine ⟨a, k * K, ha, ha₁, mul_pos hk (by dsimp [K]; positivity), ?_⟩
  filter_upwards [hf.eventually_analyticAt,
    (e.continuous.tendsto z).eventually (hV.mem_nhds hzV)] with x hax hxV
  have h := hv (e x) hxV
  change |f (e.symm (e x)) - f (e.symm (e z))| ^ a ≤ k * ‖gradient g (e x)‖ at h
  rw [e.symm_apply_apply, e.symm_apply_apply] at h
  have hD : HasFDerivAt f (fderiv ℝ f x) (e.symm (e x)) := by
    simpa only [e.symm_apply_apply] using hax.differentiableAt.hasFDerivAt
  have hd := (hD.comp (e x) e.symm.hasFDerivAt).fderiv
  have hgrad : ‖gradient g (e x)‖ ≤ ‖fderiv ℝ f x‖ * K := by
    rw [gradient, LinearIsometryEquiv.norm_map]
    change ‖fderiv ℝ (f ∘ e.symm) (e x)‖ ≤ _
    rw [hd]
    exact (ContinuousLinearMap.opNorm_comp_le _ _).trans
      (mul_le_mul_of_nonneg_left (le_add_of_nonneg_right zero_le_one) (norm_nonneg _))
  calc
    _ ≤ k * (‖fderiv ℝ f x‖ * K) := h.trans (mul_le_mul_of_nonneg_left hgrad hk.le)
    _ = _ := by ring

/-- The local gradient inequality passes from the stationary auxiliary
coordinates back to the underlying function. -/
theorem local_fderiv_inequality {f : X → ℝ} (hf : HasAnalyticLift f) (z : X) :
    ∃ a m : ℝ, 0 ≤ a ∧ a < 1 ∧ 0 < m ∧
      ∀ᶠ x in 𝓝 z, f z < f x → m * (f x - f z) ^ a ≤ ‖fderiv ℝ f x‖ := by
  classical
  by_contra h
  let b (n : ℕ) : ℝ := 1 / ((n : ℝ) + 2)
  let a (n : ℕ) : ℝ := 1 - b n
  have hb (n : ℕ) : 0 < b n := by dsimp [b]; positivity
  have hb₁ (n : ℕ) : b n ≤ 1 := by
    dsimp [b]
    exact (div_le_one (by positivity)).2 (by linarith [Nat.cast_nonneg (α := ℝ) n])
  have ha (n : ℕ) : 0 ≤ a n := sub_nonneg.mpr (hb₁ n)
  have ha₁ (n : ℕ) : a n < 1 := by dsimp [a]; linarith [hb n]
  have hfail (n : ℕ) :
      ∃ x, dist x z < b n ∧ f z < f x ∧ ‖fderiv ℝ f x‖ < b n * (f x - f z) ^ a n := by
    have hn : ¬ ∀ᶠ x in 𝓝 z, f z < f x → b n * (f x - f z) ^ a n ≤ ‖fderiv ℝ f x‖ :=
      fun hh => h ⟨a n, b n, ha n, ha₁ n, hb n, hh⟩
    have hfreq : ∃ᶠ x in 𝓝 z, ¬ (f z < f x → b n * (f x - f z) ^ a n ≤ ‖fderiv ℝ f x‖) :=
      Filter.not_eventually.mp hn
    obtain ⟨x, hx, hball⟩ := (hfreq.and_eventually (ball_mem_nhds z (hb n))).exists
    push Not at hx
    exact ⟨x, hball, hx⟩
  choose x hdist hgt hsmall using hfail
  have hbt : Tendsto b atTop (𝓝 0) := by
    simpa only [b, Nat.cast_add, Nat.cast_ofNat] using
      (tendsto_add_atTop_iff_nat 2).2 (tendsto_one_div_atTop_nhds_zero_nat (𝕜 := ℝ))
  have hxt : Tendsto x atTop (𝓝 z) := by
    apply Metric.tendsto_nhds.mpr
    intro ε hε
    exact (hbt.eventually (gt_mem_nhds hε)).mono fun n hn => (hdist n).trans hn
  obtain ⟨U, q, c, c₀, u, hu, hc, hq, hfit⟩ := hf.lift x z hxt
  let p (n : ℕ) : X × U.carrier := (x (u n), c n)
  have hpt : Tendsto p atTop (𝓝 (z, c₀)) := (hxt.comp hu.tendsto_atTop).prodMk_nhds hc
  have hft : Tendsto (f ∘ x ∘ u) atTop (𝓝 (f z)) :=
    (hf.differentiable z).continuousAt.tendsto.comp (hxt.comp hu.tendsto_atTop)
  have he : q (z, c₀) = f z :=
    tendsto_nhds_unique (hq.continuousAt.tendsto.comp hpt)
      (hft.congr fun n => (hfit n).1.symm)
  obtain ⟨θ, K, hθ, hθ₁, hK, hloc⟩ := analytic_local_fderiv_inequality hq
  have hat : Tendsto a atTop (𝓝 1) := by
    simpa only [sub_zero] using (tendsto_const_nhds (x := (1 : ℝ))).sub hbt
  have hθt := (hat.comp hu.tendsto_atTop).eventually (lt_mem_nhds hθ₁)
  have hbt' := (hbt.comp hu.tendsto_atTop).eventually (gt_mem_nhds (one_div_pos.mpr hK))
  have hdiff : Tendsto (fun n => f (x (u n)) - f z) atTop (𝓝 0) := by
    simpa only [sub_self, Function.comp_apply] using hft.sub_const (f z)
  have hs := hdiff.eventually (gt_mem_nhds zero_lt_one)
  obtain ⟨n, hlocn, hθn, hbn, hsn⟩ :=
    ((hpt.eventually hloc).and (hθt.and (hbt'.and hs))).exists
  let gap := f (x (u n)) - f z
  have hg : 0 < gap := sub_pos.mpr (hgt (u n))
  have hg₁ : gap ≤ 1 := hsn.le
  rw [(hfit n).1, he, abs_of_pos hg] at hlocn
  have hd : ‖fderiv ℝ q (p n)‖ ≤ ‖fderiv ℝ f (x (u n))‖ := by
    rw [(hfit n).2.fderiv]
    refine (ContinuousLinearMap.opNorm_comp_le _ _).trans ?_
    exact (mul_le_mul_of_nonneg_left (ContinuousLinearMap.norm_fst_le ℝ X U.carrier)
      (norm_nonneg _)).trans_eq (mul_one _)
  have hlt := mul_lt_mul_of_pos_left (hsmall (u n)) hK
  have hpow : gap ^ a (u n) ≤ gap ^ θ :=
    Real.rpow_le_rpow_of_exponent_ge hg hg₁ hθn.le
  have hbK : K * b (u n) < 1 := by
    have hh : b (u n) < 1 / K := hbn
    linarith only [(lt_div_iff₀ hK).mp hh, mul_comm (b (u n)) K]
  exact (lt_irrefl (gap ^ θ)) (calc
    gap ^ θ ≤ K * ‖fderiv ℝ q (p n)‖ := hlocn
    _ ≤ K * ‖fderiv ℝ f (x (u n))‖ := mul_le_mul_of_nonneg_left hd hK.le
    _ < K * (b (u n) * gap ^ a (u n)) := hlt
    _ ≤ K * b (u n) * gap ^ θ := by
      rw [mul_assoc]
      exact mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left hpow (hb (u n)).le) hK.le
    _ < gap ^ θ := by
      simpa only [one_mul] using mul_lt_mul_of_pos_right hbK (Real.rpow_pos_of_pos hg θ))

end Transformer.Optimization.HasAnalyticLift
