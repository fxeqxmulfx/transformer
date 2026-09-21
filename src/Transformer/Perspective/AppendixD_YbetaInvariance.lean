/-
# Appendix D — the solutions of `eq: ybeta` and `eq: ybetaUSA` stay in `[0, 1]`

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

The properties of `γ_β` the proof of `e:ybetacloseto1` states in one line —
"`t ↦ γ_β(t)` is increasing and thus `γ_β(t) ≥ 0`, as well as
`γ̇_β(t) ≥ 1/(n e^β)` as long as `γ_β(t) ≤ 1/2`.  Therefore
`γ_β(n e^β / 2) ≥ 1/2`" — proved from the equations with `γ_β(0) = 0`.

Two facts about scalar equations carry them.  An equation
`γ̇ = a(γ) (1 - γ)` with `a` continuous keeps `1 - γ` of one sign, since
`(1 - γ) e^{∫ a(γ)}` is constant (`lt_one_of_deriv_eq_mul_one_sub`).  And an
equation `γ̇ = F(γ)` with `F(0) > 0` cannot cross `0` downwards
(`nonneg_of_deriv_eq`).
-/

import Transformer.Perspective.Section5_HighDCurve
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.MeasureTheory.Integral.IntervalIntegral.FundThmCalculus

open Real

namespace Transformer
namespace Perspective

/-- **An equation `γ̇ = a(γ)(1 - γ)` never reaches `1` from below.**  The
product `(1 - γ(t)) e^{∫_0^t a(γ)}` has derivative zero, so it keeps the value
`1 - γ(0) > 0`.

Source: arXiv:2312.10794v5, Appendix D, proof of `e:ybetacloseto1` (the
invariance of `[0, 1]` it uses). -/
theorem lt_one_of_deriv_eq_mul_one_sub {γ a : ℝ → ℝ} (ha : Continuous a)
    (hγ : ∀ t : ℝ, HasDerivAt γ (a (γ t) * (1 - γ t)) t) (h0 : γ 0 < 1) :
    ∀ t : ℝ, γ t < 1 := by
  have hγc : Continuous γ := continuous_iff_continuousAt.2 fun t => (hγ t).continuousAt
  have hc : Continuous fun u => a (γ u) := ha.comp hγc
  set A : ℝ → ℝ := fun t => ∫ u in (0 : ℝ)..t, a (γ u) with hA
  have hAd : ∀ t : ℝ, HasDerivAt A (a (γ t)) t := fun t =>
    (hc.integral_hasStrictDerivAt 0 t).hasDerivAt
  have hd : ∀ t : ℝ, HasDerivAt (fun s => (1 - γ s) * Real.exp (A s)) 0 t := by
    intro t
    have h := ((hasDerivAt_const t (1 : ℝ)).sub (hγ t)).mul (hAd t).exp
    refine h.congr_deriv ?_
    simp only [Pi.sub_apply]
    ring
  intro t
  have hconst := is_const_of_deriv_eq_zero (fun s => (hd s).differentiableAt)
    (fun s => (hd s).deriv) t 0
  have hA0 : A 0 = 0 := by simp [hA]
  simp only [hA0, Real.exp_zero, mul_one] at hconst
  have hpos := Real.exp_pos (A t)
  nlinarith

/-- The hypotheses of `lt_one_of_deriv_eq_mul_one_sub` are satisfiable: the
constant `0` solves `γ̇ = 0 · (1 - γ)`. -/
example : Continuous (fun _ : ℝ => (0 : ℝ)) ∧
    (∀ t : ℝ, HasDerivAt (fun _ : ℝ => (0 : ℝ)) (0 * (1 - 0)) t) ∧ (0 : ℝ) < 1 :=
  ⟨continuous_const, fun t => by simpa using hasDerivAt_const t (0 : ℝ), one_pos⟩

/-- **An equation `γ̇ = F(γ)` with `F(0) > 0` and `γ(0) = 0` stays nonnegative
forward in time.**  Whenever `-γ` touches `0`, its derivative `-F(0)` is
negative.

Source: arXiv:2312.10794v5, Appendix D, proof of `e:ybetacloseto1`
("`γ_β(t) ≥ 0`"). -/
theorem nonneg_of_deriv_eq {γ F : ℝ → ℝ} (hγ : ∀ t : ℝ, HasDerivAt γ (F (γ t)) t)
    (hF : 0 < F 0) (h0 : γ 0 = 0) : ∀ t : ℝ, 0 ≤ t → 0 ≤ γ t := by
  intro t ht
  have h := image_le_of_deriv_right_lt_deriv_boundary' (a := 0) (b := t)
    (f := fun s => -γ s) (f' := fun s => -F (γ s)) (B := fun _ => (0 : ℝ))
    (B' := fun _ => (0 : ℝ))
    (fun s _ => (hγ s).continuousAt.neg.continuousWithinAt)
    (fun s _ => (hγ s).neg.hasDerivWithinAt) (by simp [h0]) continuousOn_const
    (fun s _ => hasDerivWithinAt_const s _ 0)
    (fun s _ hs => by
      have : γ s = 0 := by linarith
      simp only [this]
      linarith)
    ⟨ht, le_rfl⟩
  linarith

/-- The hypotheses of `nonneg_of_deriv_eq` are satisfiable: `γ = id` solves
`γ̇ = 1`. -/
example : (∀ t : ℝ, HasDerivAt (fun s : ℝ => s) ((fun _ : ℝ => (1 : ℝ)) t) t) ∧
    (0 : ℝ) < 1 ∧ (fun s : ℝ => s) 0 = 0 :=
  ⟨fun t => hasDerivAt_id t, one_pos, rfl⟩

/-- **The solution of `eq: ybeta` stays in `[0, 1)` forward in time.**

Source: arXiv:2312.10794v5, Appendix D, proof of `e:ybetacloseto1`. -/
theorem ybetaODE_SA.mem_Ico {n : ℕ} (hn : 0 < n) {β : ℝ} {γ : ℝ → ℝ}
    (hγ : ybetaODE_SA n β γ) : ∀ t : ℝ, 0 ≤ t → 0 ≤ γ t ∧ γ t < 1 := by
  have hn1 : (1 : ℝ) ≤ (n : ℝ) := by exact_mod_cast hn
  have hD : ∀ g : ℝ, 0 < Real.exp β + ((n : ℝ) - 1) * Real.exp (β * g) := fun g => by
    have := mul_nonneg (by linarith : (0 : ℝ) ≤ (n : ℝ) - 1) (Real.exp_pos (β * g)).le
    linarith [Real.exp_pos β]
  have hlt := lt_one_of_deriv_eq_mul_one_sub
    (a := fun g => 2 * Real.exp (β * g) * (((n : ℝ) - 1) * g + 1)
      / (Real.exp β + ((n : ℝ) - 1) * Real.exp (β * g)))
    (by fun_prop (disch := exact fun g => (hD g).ne'))
    (fun t => (hγ.2 t).congr_deriv (by ring)) (by rw [hγ.1]; norm_num)
  have hnn := nonneg_of_deriv_eq (γ := γ)
    (F := fun g => 2 * Real.exp (β * g) * (1 - g) * (((n : ℝ) - 1) * g + 1)
      / (Real.exp β + ((n : ℝ) - 1) * Real.exp (β * g)))
    hγ.2 (by simpa using hD 0) hγ.1
  exact fun t ht => ⟨hnn t ht, hlt t⟩

/-- **The solution of `eq: ybetaUSA` stays in `[0, 1)` forward in time.**

Source: arXiv:2312.10794v5, Appendix D, `rem: usa.d` ("the same arguments
readily apply"). -/
theorem ybetaODE_USA.mem_Ico {n : ℕ} (hn : 0 < n) {β : ℝ} {γ : ℝ → ℝ}
    (hγ : ybetaODE_USA n β γ) : ∀ t : ℝ, 0 ≤ t → 0 ≤ γ t ∧ γ t < 1 := by
  have hnR : (0 : ℝ) < (n : ℝ) := by exact_mod_cast hn
  have hlt := lt_one_of_deriv_eq_mul_one_sub
    (a := fun g => (2 / (n : ℝ)) * Real.exp (β * g) * (((n : ℝ) - 1) * g + 1))
    (by fun_prop) (fun t => (hγ.2 t).congr_deriv (by ring)) (by rw [hγ.1]; norm_num)
  have hnn := nonneg_of_deriv_eq (γ := γ)
    (F := fun g => (2 / (n : ℝ)) * Real.exp (β * g) * (1 - g) * (((n : ℝ) - 1) * g + 1))
    hγ.2 (by simpa using hnR) hγ.1
  exact fun t ht => ⟨hnn t ht, hlt t⟩

/-- **The solution of `eq: ybeta` is nondecreasing forward in time**: on
`[0, ∞)` it stays in `[0, 1]`, where the right-hand side is nonnegative.

Source: arXiv:2312.10794v5, Appendix D, proof of `e:ybetacloseto1`
("`t ↦ γ_β(t)` is increasing"). -/
theorem ybetaODE_SA.monotoneOn {n : ℕ} (hn : 0 < n) {β : ℝ} {γ : ℝ → ℝ}
    (hγ : ybetaODE_SA n β γ) : MonotoneOn γ (Set.Ici 0) := by
  have hn1 : (1 : ℝ) ≤ (n : ℝ) := by exact_mod_cast hn
  refine monotoneOn_of_deriv_nonneg (convex_Ici 0)
    (fun s _ => (hγ.2 s).continuousAt.continuousWithinAt)
    (fun s _ => (hγ.2 s).differentiableAt.differentiableWithinAt) ?_
  intro s hs
  rw [interior_Ici] at hs
  obtain ⟨hg0, hg1⟩ := hγ.mem_Ico hn s (Set.mem_Ioi.mp hs).le
  rw [(hγ.2 s).deriv]
  have hD : (0 : ℝ) ≤ Real.exp β + ((n : ℝ) - 1) * Real.exp (β * γ s) := by
    have := mul_nonneg (by linarith : (0 : ℝ) ≤ (n : ℝ) - 1) (Real.exp_pos (β * γ s)).le
    linarith [Real.exp_pos β]
  refine div_nonneg ?_ hD
  exact mul_nonneg (mul_nonneg (by positivity) (by linarith)) (by nlinarith)

/-- **The half-way time.**  `γ̇_β ≥ 1/(n e^β)` as long as `γ_β ≤ 1/2`, hence
`γ_β(n e^β / 2) ≥ 1/2`.

Source: arXiv:2312.10794v5, Appendix D, proof of `e:ybetacloseto1`. -/
theorem ybetaODE_SA.half_le {n : ℕ} (hn : 0 < n) {β : ℝ} (hβ : 0 ≤ β) {γ : ℝ → ℝ}
    (hγ : ybetaODE_SA n β γ) : (1 / 2 : ℝ) ≤ γ ((n : ℝ) * Real.exp β / 2) := by
  have hn1 : (1 : ℝ) ≤ (n : ℝ) := by exact_mod_cast hn
  have hEb : (0 : ℝ) < Real.exp β := Real.exp_pos β
  set T0 : ℝ := (n : ℝ) * Real.exp β / 2 with hT0
  have hT0nn : (0 : ℝ) ≤ T0 := by positivity
  set c : ℝ := ((n : ℝ) * Real.exp β)⁻¹ with hc
  have hcpos : 0 < c := by positivity
  by_contra hlt
  push Not at hlt
  have hbelow : ∀ s ∈ Set.Icc 0 T0, γ s < 1 / 2 := fun s hs =>
    lt_of_le_of_lt (hγ.monotoneOn hn hs.1 (Set.mem_Ici.mpr hT0nn) hs.2) hlt
  have hφd : ∀ s : ℝ, HasDerivAt (fun s => γ s - c * s)
      (2 * Real.exp (β * γ s) * (1 - γ s) * (((n : ℝ) - 1) * γ s + 1)
        / (Real.exp β + ((n : ℝ) - 1) * Real.exp (β * γ s)) - c) s := fun s =>
    (hγ.2 s).sub (by simpa using (hasDerivAt_id s).const_mul c)
  have hφ : MonotoneOn (fun s => γ s - c * s) (Set.Icc 0 T0) := by
    refine monotoneOn_of_deriv_nonneg (convex_Icc 0 T0)
      (fun s _ => (hφd s).continuousAt.continuousWithinAt)
      (fun s _ => (hφd s).differentiableAt.differentiableWithinAt) ?_
    intro s hs
    rw [interior_Icc] at hs
    rw [(hφd s).deriv]
    obtain ⟨hg0, -⟩ := hγ.mem_Ico hn s hs.1.le
    have hgh := hbelow s ⟨hs.1.le, hs.2.le⟩
    have hE1 : (1 : ℝ) ≤ Real.exp (β * γ s) := Real.one_le_exp (mul_nonneg hβ hg0)
    have hEle : Real.exp (β * γ s) ≤ Real.exp β :=
      Real.exp_le_exp.mpr (by nlinarith)
    have hD : (0 : ℝ) < Real.exp β + ((n : ℝ) - 1) * Real.exp (β * γ s) := by
      have := mul_nonneg (by linarith : (0 : ℝ) ≤ (n : ℝ) - 1) (Real.exp_pos (β * γ s)).le
      linarith
    have hN : (1 : ℝ) ≤ 2 * Real.exp (β * γ s) * (1 - γ s) * (((n : ℝ) - 1) * γ s + 1) := by
      have h1 : (1 : ℝ) ≤ ((n : ℝ) - 1) * γ s + 1 := by nlinarith
      have h2 : (1 : ℝ) ≤ 2 * (1 - γ s) := by linarith
      have h3 : (1 : ℝ) ≤ 2 * (1 - γ s) * (((n : ℝ) - 1) * γ s + 1) := by nlinarith
      nlinarith
    have hDle : Real.exp β + ((n : ℝ) - 1) * Real.exp (β * γ s) ≤ (n : ℝ) * Real.exp β := by
      nlinarith [mul_le_mul_of_nonneg_left hEle (by linarith : (0 : ℝ) ≤ (n : ℝ) - 1)]
    have hrate : c ≤ 2 * Real.exp (β * γ s) * (1 - γ s) * (((n : ℝ) - 1) * γ s + 1)
        / (Real.exp β + ((n : ℝ) - 1) * Real.exp (β * γ s)) := by
      rw [hc, ← one_div, div_le_div_iff₀ (by positivity) hD]
      nlinarith [mul_le_mul_of_nonneg_right hN (by positivity : (0 : ℝ) ≤ (n : ℝ) * Real.exp β)]
    linarith
  have h := hφ ⟨le_rfl, hT0nn⟩ ⟨hT0nn, le_rfl⟩ hT0nn
  simp only [mul_zero, sub_zero, hγ.1] at h
  have hcT : c * T0 = 1 / 2 := by
    rw [hc, hT0]; field_simp
  linarith

/-- The hypotheses of `ybetaODE_SA.mem_Ico`, `ybetaODE_SA.monotoneOn` and
`ybetaODE_SA.half_le` are satisfiable: `tanh` solves `eq: ybeta` at `n = 2`,
`β = 0`. -/
example : 0 < 2 ∧ (0 : ℝ) ≤ 0 ∧ ybetaODE_SA 2 0 Real.tanh :=
  ⟨two_pos, le_rfl, ybetaODE_SA_two_zero⟩

/-- The hypotheses of `ybetaODE_USA.mem_Ico` are satisfiable: `tanh` solves
`eq: ybetaUSA` at `n = 2`, `β = 0`. -/
example : 0 < 2 ∧ ybetaODE_USA 2 0 Real.tanh := ⟨two_pos, ybetaODE_USA_two_zero⟩

end Perspective
end Transformer
