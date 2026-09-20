/-
# The scalar collapse ODE

  `u̇ = u (1 - u) e^{β (u - 1)}`

is the Cauchy problem `lem: eminem` of

  Geshkovski, Koubbi, Polyanskiy, Rigollet,
  *Dynamic metastability in the self-attention model*, arXiv:2410.06833v1, §2

is about: `u` is the smallest inner product inside a cap, and the lemma
bounds the time `1 - u` needs to fall below `e^{-cβ}`.  This file proves that
bound; `Metastability.eminem` is its `sInf` reading.

The proof takes the source's bound apart at `s = 1/β`, where `s = 1 - u`,
and replaces the integral `∫ e^{βs} / (s (1-s)) ds` — which has no elementary
antiderivative — by two auxiliary functions whose derivative is bounded by a
constant:

* `g(t) = e^{β(1-u(t))} + u(0) t` has `g' = u(0) - β u (1-u) ≤ 0` as long as
  `s ≥ 1/β`, the two exponentials cancelling exactly.  If `s` stayed above
  `1/β` on `[0, B₁]`, `B₁ = e^{β s₀} / u₀`, then `g B₁ ≤ g 0` would read
  `e^{β(1-u(B₁))} ≤ 0`.  So `s` drops below `1/β` at some `t⋆ ≤ B₁`.
* `h(t) = log (1 - u(t)) + ((β-1)/(βe)) t` has `h' ≤ 0` once `s ≤ 1/β`, and
  `((β-1)/(βe)) · B₂ = cβ` exactly for `B₂ = β² c e/(β-1)`, so
  `log s(t⋆ + B₂) ≤ log s(t⋆) - cβ ≤ -cβ`.

The two terms of the source's bound are `B₁` and `B₂`, and the constants
match with nothing to spare.

Everything runs by contradiction from "`1 - u > e^{-cβ}` on `[0, B]`", which
is what makes the qualitative half free: it gives `u < 1` there outright,
`0 < u` follows from one crossing argument, and monotonicity of `u` from the
sign of its derivative.  No uniqueness theorem for the ODE is used.
-/

import Mathlib.Analysis.Calculus.Deriv.MeanValue
import Mathlib.Analysis.SpecialFunctions.Log.Deriv

open Real Set

namespace Transformer
namespace Metastability

/-- **The collapse time of `u̇ = u(1-u)e^{β(u-1)}`.**

For `β > 1`, `c > 0` and a solution with `u(0) > 0`, the time at which
`1 - u` first drops to `e^{-cβ}` is at most

  `e^{β(1-u(0))}/u(0) + β² c e/(β-1)`.

This is the content of `lem: eminem`, in the form that exhibits the time
rather than taking an infimum; `Metastability.eminem` is the infimum reading.

Source: arXiv:2410.06833v1, §2, `lem: eminem`. -/
theorem exists_collapse_time
    (β : ℝ) (hβ : 1 < β) (c : ℝ) (hc : 0 < c) (u : ℝ → ℝ) (hu0 : 0 < u 0)
    (hu_ode : ∀ t : ℝ, HasDerivAt u (u t * (1 - u t) * Real.exp (β * (u t - 1))) t) :
    ∃ t : ℝ, 0 ≤ t ∧
      t ≤ Real.exp (β * (1 - u 0)) / u 0 + β ^ 2 * c * Real.exp 1 / (β - 1) ∧
      1 - u t ≤ Real.exp (-(c * β)) := by
  have hβ0 : (0 : ℝ) < β := lt_trans one_pos hβ
  have hβ1 : (0 : ℝ) < β - 1 := by linarith
  have hdiff : Differentiable ℝ u := fun t => (hu_ode t).differentiableAt
  have hderiv_nonneg : ∀ x : ℝ, 0 < u x → u x < 1 → 0 ≤ deriv u x := by
    intro x hx1 hx2
    rw [(hu_ode x).deriv]
    exact le_of_lt (mul_pos (mul_pos hx1 (by linarith)) (Real.exp_pos _))
  set B₁ : ℝ := Real.exp (β * (1 - u 0)) / u 0 with hB₁def
  set B₂ : ℝ := β ^ 2 * c * Real.exp 1 / (β - 1) with hB₂def
  have hB₁pos : 0 < B₁ := div_pos (Real.exp_pos _) hu0
  have hB₂pos : 0 < B₂ :=
    div_pos (mul_pos (mul_pos (pow_pos hβ0 2) hc) (Real.exp_pos 1)) hβ1
  by_contra hcon
  push Not at hcon
  set B : ℝ := B₁ + B₂ with hBdef
  have hBpos : 0 < B := by rw [hBdef]; linarith
  -- On `[0, B]` the contradiction hypothesis gives `u < 1` outright.
  have hlt1 : ∀ t ∈ Icc (0 : ℝ) B, u t < 1 := by
    intro t ht
    have h := hcon t ht.1 ht.2
    have := Real.exp_pos (-(c * β))
    linarith
  -- and `u > 0` by a crossing argument: `u` cannot reach `0` while it is
  -- nondecreasing on everything to the left of its first zero.
  have hpos : ∀ t ∈ Icc (0 : ℝ) B, 0 < u t := by
    by_contra hbad
    push Not at hbad
    obtain ⟨t₁, ht₁mem, ht₁⟩ := hbad
    set S : Set ℝ := {t : ℝ | t ∈ Icc (0 : ℝ) B ∧ u t ≤ 0} with hSdef
    have hSclosed : IsClosed S := by
      have hS : S = Icc (0 : ℝ) B ∩ u ⁻¹' Iic 0 := by
        ext t; simp [hSdef, Set.mem_preimage]
      rw [hS]
      exact isClosed_Icc.inter (isClosed_Iic.preimage hdiff.continuous)
    have hSne : S.Nonempty := ⟨t₁, ht₁mem, ht₁⟩
    have hSbdd : BddBelow S := ⟨0, fun t ht => ht.1.1⟩
    have ht₀S : sInf S ∈ S := hSclosed.csInf_mem hSne hSbdd
    set t₀ : ℝ := sInf S with ht₀def
    have ht₀pos : 0 < t₀ := by
      rcases ht₀S.1.1.lt_or_eq with h | h
      · exact h
      · rw [← h] at ht₀S; exact absurd ht₀S.2 (not_le.mpr hu0)
    have hmono0 : MonotoneOn u (Icc 0 t₀) := by
      refine monotoneOn_of_deriv_nonneg (convex_Icc 0 t₀) hdiff.continuous.continuousOn
        hdiff.differentiableOn ?_
      intro x hx
      rw [interior_Icc] at hx
      have hxmem : x ∈ Icc (0 : ℝ) B := ⟨hx.1.le, hx.2.le.trans ht₀S.1.2⟩
      have hxpos : 0 < u x := by
        by_contra hx0
        push Not at hx0
        exact absurd (csInf_le hSbdd (show x ∈ S from ⟨hxmem, hx0⟩)) (not_le.mpr hx.2)
      exact hderiv_nonneg x hxpos (hlt1 x hxmem)
    have := hmono0 (left_mem_Icc.mpr ht₀pos.le) (right_mem_Icc.mpr ht₀pos.le) ht₀pos.le
    linarith [ht₀S.2]
  have hmono : MonotoneOn u (Icc 0 B) := by
    refine monotoneOn_of_deriv_nonneg (convex_Icc 0 B) hdiff.continuous.continuousOn
      hdiff.differentiableOn ?_
    intro x hx
    rw [interior_Icc] at hx
    have hxmem : x ∈ Icc (0 : ℝ) B := ⟨hx.1.le, hx.2.le⟩
    exact hderiv_nonneg x (hpos x hxmem) (hlt1 x hxmem)
  have h0mem : (0 : ℝ) ∈ Icc (0 : ℝ) B := left_mem_Icc.mpr hBpos.le
  -- **First phase.**  `1 - u` drops below `1/β` before time `B₁`.
  have hstar : ∃ ts ∈ Icc (0 : ℝ) B₁, 1 - u ts < 1 / β := by
    by_contra hbad
    push Not at hbad
    have hsub : Icc (0 : ℝ) B₁ ⊆ Icc (0 : ℝ) B :=
      Icc_subset_Icc le_rfl (by rw [hBdef]; linarith)
    have hgd : ∀ t : ℝ, HasDerivAt (fun s => Real.exp (β * (1 - u s)) + u 0 * s)
        (Real.exp (β * (1 - u t)) * (β * -(u t * (1 - u t) * Real.exp (β * (u t - 1))))
          + u 0 * 1) t := fun t =>
      ((((hu_ode t).const_sub 1).const_mul β).exp).add ((hasDerivAt_id' (x := t)).const_mul (u 0))
    have hganti : AntitoneOn (fun s => Real.exp (β * (1 - u s)) + u 0 * s) (Icc 0 B₁) := by
      refine antitoneOn_of_deriv_nonpos (convex_Icc 0 B₁)
        (fun t _ => (hgd t).continuousAt.continuousWithinAt)
        (fun t _ => (hgd t).differentiableAt.differentiableWithinAt) ?_
      intro x hx
      rw [interior_Icc] at hx
      have hxmem₁ : x ∈ Icc (0 : ℝ) B₁ := ⟨hx.1.le, hx.2.le⟩
      have hxmem : x ∈ Icc (0 : ℝ) B := hsub hxmem₁
      have hxpos : 0 < u x := hpos x hxmem
      have hxmono : u 0 ≤ u x := hmono h0mem hxmem hx.1.le
      have hxβ : 1 / β ≤ 1 - u x := hbad x hxmem₁
      rw [(hgd x).deriv]
      have hEE : Real.exp (β * (1 - u x)) * Real.exp (β * (u x - 1)) = 1 := by
        rw [← Real.exp_add]
        have hz : β * (1 - u x) + β * (u x - 1) = 0 := by ring
        rw [hz, Real.exp_zero]
      have hkey : Real.exp (β * (1 - u x)) * (β * -(u x * (1 - u x) * Real.exp (β * (u x - 1))))
          = -(β * (1 - u x) * u x) := by
        linear_combination (-(β * (1 - u x) * u x)) * hEE
      rw [hkey]
      have h1 : 1 ≤ β * (1 - u x) := by
        rw [div_le_iff₀ hβ0] at hxβ
        nlinarith
      nlinarith [mul_le_mul_of_nonneg_right h1 hxpos.le]
    have hle := hganti (left_mem_Icc.mpr hB₁pos.le) (right_mem_Icc.mpr hB₁pos.le) hB₁pos.le
    dsimp only at hle
    have hgB : u 0 * B₁ = Real.exp (β * (1 - u 0)) := by
      rw [hB₁def]; field_simp
    linarith [Real.exp_pos (β * (1 - u B₁))]
  obtain ⟨ts, hts, htslt⟩ := hstar
  -- **Second phase.**  From there, `log (1 - u)` falls at rate `(β-1)/(βe)`.
  have htsB : ts + B₂ ≤ B := by rw [hBdef]; linarith [hts.2]
  have hsub2 : Icc ts (ts + B₂) ⊆ Icc (0 : ℝ) B :=
    Icc_subset_Icc hts.1 htsB
  have htsmem : ts ∈ Icc (0 : ℝ) B := hsub2 (left_mem_Icc.mpr (by linarith))
  have hlogd : ∀ t ∈ Icc (0 : ℝ) B,
      HasDerivAt (fun s => Real.log (1 - u s) + ((β - 1) / (β * Real.exp 1)) * s)
        (-(u t * (1 - u t) * Real.exp (β * (u t - 1))) / (1 - u t)
          + (β - 1) / (β * Real.exp 1) * 1) t := by
    intro t ht
    have hne : (1 : ℝ) - u t ≠ 0 := sub_ne_zero.mpr (hlt1 t ht).ne'
    exact (((hu_ode t).const_sub 1).log hne).add
      ((hasDerivAt_id' (x := t)).const_mul ((β - 1) / (β * Real.exp 1)))
  have hanti : AntitoneOn (fun s => Real.log (1 - u s) + ((β - 1) / (β * Real.exp 1)) * s)
      (Icc ts (ts + B₂)) := by
    refine antitoneOn_of_deriv_nonpos (convex_Icc _ _)
      (fun t ht => (hlogd t (hsub2 ht)).continuousAt.continuousWithinAt) (fun t ht => ?_) ?_
    · rw [interior_Icc] at ht
      exact (hlogd t (hsub2 (Ioo_subset_Icc_self ht))).differentiableAt.differentiableWithinAt
    · intro x hx
      rw [interior_Icc] at hx
      have hxmem : x ∈ Icc (0 : ℝ) B := hsub2 (Ioo_subset_Icc_self hx)
      have hxpos : 0 < u x := hpos x hxmem
      have hx1 : u x < 1 := hlt1 x hxmem
      have hxts : u ts ≤ u x := hmono htsmem hxmem hx.1.le
      rw [(hlogd x hxmem).deriv]
      have hne : (1 : ℝ) - u x ≠ 0 := sub_ne_zero.mpr hx1.ne'
      have hsimp : -(u x * (1 - u x) * Real.exp (β * (u x - 1))) / (1 - u x)
          = -(u x * Real.exp (β * (u x - 1))) := by
        field_simp
      rw [hsimp]
      have hβs : β * (1 - u x) < 1 := by
        rw [lt_div_iff₀ hβ0] at htslt
        nlinarith
      have hexp : (Real.exp 1)⁻¹ ≤ Real.exp (β * (u x - 1)) := by
        rw [← Real.exp_neg]
        exact Real.exp_le_exp.mpr (by nlinarith)
      have hlow : (β - 1) / β ≤ u x := by
        rw [div_le_iff₀ hβ0]
        nlinarith
      have hdivp : (β - 1) / (β * Real.exp 1) = (β - 1) / β * (Real.exp 1)⁻¹ := by
        field_simp
      have hmul : (β - 1) / β * (Real.exp 1)⁻¹ ≤ u x * Real.exp (β * (u x - 1)) :=
        mul_le_mul hlow hexp (by positivity) hxpos.le
      rw [hdivp]
      linarith
  have hlast := hanti (left_mem_Icc.mpr (by linarith)) (right_mem_Icc.mpr (by linarith))
    (by linarith)
  dsimp only at hlast
  have hkB : (β - 1) / (β * Real.exp 1) * B₂ = c * β := by
    rw [hB₂def]
    field_simp
  have hts1 : u ts < 1 := hlt1 ts htsmem
  have htsp : 0 < u ts := hpos ts htsmem
  have hlog_ts : Real.log (1 - u ts) ≤ 0 :=
    Real.log_nonpos (by linarith) (by linarith)
  have hmem2 : ts + B₂ ∈ Icc (0 : ℝ) B := hsub2 (right_mem_Icc.mpr (by linarith))
  have hpos2 : 0 < 1 - u (ts + B₂) := by linarith [hlt1 (ts + B₂) hmem2]
  have hfinal : Real.log (1 - u (ts + B₂)) ≤ -(c * β) := by nlinarith [hlast, hkB, hlog_ts]
  have hconcl : 1 - u (ts + B₂) ≤ Real.exp (-(c * β)) := by
    have hiff := Real.log_le_log_iff hpos2 (Real.exp_pos (-(c * β)))
    rw [Real.log_exp] at hiff
    exact hiff.mp hfinal
  exact absurd hconcl (not_le.mpr (hcon (ts + B₂) hmem2.1 htsB))

/-- The hypotheses of `exists_collapse_time` are satisfiable: `β = 2`, `c = 1`
and the constant solution `u ≡ 1`, for which `u(1-u)e^{β(u-1)} = 0`. -/
example : (1 : ℝ) < 2 ∧ (0 : ℝ) < 1 ∧ 0 < (fun _ : ℝ => (1 : ℝ)) 0 ∧
    ∀ t : ℝ, HasDerivAt (fun _ : ℝ => (1 : ℝ))
      ((fun _ : ℝ => (1 : ℝ)) t * (1 - (fun _ : ℝ => (1 : ℝ)) t) *
        Real.exp (2 * ((fun _ : ℝ => (1 : ℝ)) t - 1))) t := by
  refine ⟨by norm_num, by norm_num, by norm_num, fun t => ?_⟩
  simpa using hasDerivAt_const t (1 : ℝ)

end Metastability
end Transformer
