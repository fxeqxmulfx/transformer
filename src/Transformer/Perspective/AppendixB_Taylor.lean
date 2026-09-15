/-
# Appendix B — the block Hessian of `𝖤_β` on the circle

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

The `d = 2` half of Appendix B:

* the second derivative of `𝖤_β` along the block direction `1_𝒮`,
* `eq: taylor2` — non-positivity of that block Hessian,
* `eq: taylor3` — the equivalent inequality for the kernel `g_β`.

The kernel `g_β` and the angle `τ_β^*` are in
`Perspective.AppendixB_BetaInterval`, the energy `𝖤_β` on `𝕋^n` is
`Perspective.torusEnergy`.
-/

import Transformer.Perspective.AppendixB_BetaInterval
import Transformer.Perspective.Section6_Circle

open scoped BigOperators
open Real

namespace Transformer
namespace Perspective

variable (n : ℕ)

/-- The direction `1_𝒮 ∈ ℝ^n` that rotates the tokens indexed by `𝒮` at unit
speed and holds the others fixed.  Appendix B perturbs a critical point along
exactly this direction. -/
def blockDir (𝒮 : Finset (Idx n)) : Angles n := fun i => if i ∈ 𝒮 then 1 else 0

/-- `SecondDerivTorusEnergy n β θ v c`: along the straight line `s ↦ θ + s v`
the energy `𝖤_β` is differentiable, and its derivative is differentiable at
`s = 0` with value `c`.

At a critical point `c` is the Hessian quadratic form `∇²𝖤_β(θ)[v, v]`, which
is how Appendix B uses it. -/
def SecondDerivTorusEnergy (β : ℝ) (θ v : Angles n) (c : ℝ) : Prop :=
  ∃ f' : ℝ → ℝ,
    (∀ s : ℝ,
      HasDerivAt (fun u : ℝ => torusEnergy n β (fun i => θ i + u * v i)) (f' s) s) ∧
    HasDerivAt f' c 0

/-- *Non-positive Hessian at `θ`*: every straight line through `θ` sees a
non-positive second derivative of `𝖤_β`.  This is the hypothesis Appendix B
places on a critical point that is not a strict saddle.

Source: arXiv:2312.10794v5, Appendix B. -/
def TorusHessianNonPos (β : ℝ) (θ : Angles n) : Prop :=
  ∀ (v : Angles n) (c : ℝ), SecondDerivTorusEnergy n β θ v c → c ≤ 0

/-- The interaction kernel along a line: `s ↦ e^{β cos(u + s c)}`. -/
theorem hasDerivAt_expCosLine (β u c s : ℝ) :
    HasDerivAt (fun x : ℝ => Real.exp (β * Real.cos (u + x * c)))
      (Real.exp (β * Real.cos (u + s * c)) * (-β * Real.sin (u + s * c)) * c) s := by
  have hline : HasDerivAt (fun x : ℝ => u + x * c) c s := by
    simpa using ((hasDerivAt_id s).mul_const c).const_add u
  have h := (HasDerivAt.const_mul β hline.cos).exp
  convert h using 1
  ring

/-- Differentiating the kernel a second time at `s = 0`:

  `d²/ds² e^{β cos(u + s c)} |_{s = 0} = c² e^{β cos u} (β² sin² u - β cos u)`,

and the bracket is `-β g_β(u)`. -/
theorem hasDerivAt_expCosLine_deriv (β u c : ℝ) :
    HasDerivAt
      (fun x : ℝ =>
        Real.exp (β * Real.cos (u + x * c)) * (-β * Real.sin (u + x * c)) * c)
      (c ^ 2 * (-β * g_β_2d β u)) 0 := by
  have hline : HasDerivAt (fun x : ℝ => u + x * c) c 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).mul_const c).const_add u
  have hA := (HasDerivAt.const_mul β hline.cos).exp
  have hB := HasDerivAt.const_mul (-β) hline.sin
  have h := (hA.mul hB).mul_const c
  simp only [zero_mul, add_zero] at h
  convert h using 1
  simp only [g_β_2d]
  ring

/-- `g_β` is even, so the pair `(i, j)` and the pair `(j, i)` contribute the
same thing to every block sum below. -/
theorem g_β_2d_symm (β x y : ℝ) : g_β_2d β (x - y) = g_β_2d β (y - x) := by
  have hc : Real.cos (x - y) = Real.cos (y - x) := by
    rw [show x - y = -(y - x) by ring]; exact Real.cos_neg _
  have hs : Real.sin (x - y) ^ 2 = Real.sin (y - x) ^ 2 := by
    rw [show x - y = -(y - x) by ring, Real.sin_neg]; exact neg_sq _
  simp only [g_β_2d, hc, hs]

/-- **The block Hessian of `𝖤_β` on the torus.**

Rotating the tokens of `𝒮` together and leaving the rest fixed changes only
the interactions across the cut `𝒮 | 𝒮^c`, and

  `d²/ds² 𝖤_β(θ + s 1_𝒮) |_{s = 0}
      = - n^{-2} Σ_{i ∈ 𝒮} Σ_{j ∈ 𝒮^c} g_β(θ_i - θ_j)`,

which is the identity behind `eq: taylor2` ⇒ `eq: taylor3`.

Source: arXiv:2312.10794v5, Appendix B, `eq: taylor2`, `eq: taylor3`. -/
theorem secondDeriv_torusEnergy_block
    (β : ℝ) (hβ : β ≠ 0) (hn : (n : ℝ) ≠ 0) (θ : Angles n) (𝒮 : Finset (Idx n)) :
    SecondDerivTorusEnergy n β θ (blockDir n 𝒮)
      (-((n : ℝ) ^ 2)⁻¹ * ∑ i ∈ 𝒮, ∑ j ∈ 𝒮ᶜ, g_β_2d β (θ i - θ j)) := by
  set χ := blockDir n 𝒮 with hχ
  have hfun : (fun u : ℝ => torusEnergy n β (fun i => θ i + u * χ i))
      = fun u : ℝ => (1 / (2 * β * (n : ℝ) ^ 2)) *
          ∑ a : Idx n, ∑ b : Idx n,
            Real.exp (β * Real.cos ((θ a - θ b) + u * (χ a - χ b))) := by
    funext u
    simp only [torusEnergy]
    congr 1
    refine Finset.sum_congr rfl fun a _ => Finset.sum_congr rfl fun b _ => ?_
    ring_nf
  refine ⟨fun s => (1 / (2 * β * (n : ℝ) ^ 2)) *
      ∑ a : Idx n, ∑ b : Idx n,
        Real.exp (β * Real.cos ((θ a - θ b) + s * (χ a - χ b))) *
          (-β * Real.sin ((θ a - θ b) + s * (χ a - χ b))) * (χ a - χ b), ?_, ?_⟩
  · intro s
    rw [hfun]
    exact HasDerivAt.const_mul _
      (HasDerivAt.fun_sum fun a _ => HasDerivAt.fun_sum fun b _ =>
        hasDerivAt_expCosLine β (θ a - θ b) (χ a - χ b) s)
  · have hstep := HasDerivAt.const_mul (1 / (2 * β * (n : ℝ) ^ 2))
      (HasDerivAt.fun_sum fun a (_ : a ∈ Finset.univ) =>
        HasDerivAt.fun_sum fun b (_ : b ∈ Finset.univ) =>
          hasDerivAt_expCosLine_deriv β (θ a - θ b) (χ a - χ b))
    refine hstep.congr_deriv ?_
    have hrowMem : ∀ a ∈ 𝒮,
        ∑ b : Idx n, (χ a - χ b) ^ 2 * (-β * g_β_2d β (θ a - θ b))
          = ∑ b ∈ 𝒮ᶜ, (-β * g_β_2d β (θ a - θ b)) := by
      intro a ha
      rw [← Finset.sum_add_sum_compl 𝒮
        (fun b => (χ a - χ b) ^ 2 * (-β * g_β_2d β (θ a - θ b)))]
      have h1 : ∑ b ∈ 𝒮, (χ a - χ b) ^ 2 * (-β * g_β_2d β (θ a - θ b)) = 0 :=
        Finset.sum_eq_zero fun b hb => by simp [hχ, blockDir, ha, hb]
      rw [h1, zero_add]
      exact Finset.sum_congr rfl fun b hb => by
        simp [hχ, blockDir, ha, Finset.mem_compl.mp hb]
    have hrowNot : ∀ a ∉ 𝒮,
        ∑ b : Idx n, (χ a - χ b) ^ 2 * (-β * g_β_2d β (θ a - θ b))
          = ∑ b ∈ 𝒮, (-β * g_β_2d β (θ a - θ b)) := by
      intro a ha
      rw [← Finset.sum_add_sum_compl 𝒮
        (fun b => (χ a - χ b) ^ 2 * (-β * g_β_2d β (θ a - θ b)))]
      have h2 : ∑ b ∈ 𝒮ᶜ, (χ a - χ b) ^ 2 * (-β * g_β_2d β (θ a - θ b)) = 0 :=
        Finset.sum_eq_zero fun b hb => by
          simp [hχ, blockDir, ha, Finset.mem_compl.mp hb]
      rw [h2, add_zero]
      exact Finset.sum_congr rfl fun b hb => by simp [hχ, blockDir, ha, hb]
    have hsplit : ∑ a : Idx n, ∑ b : Idx n,
          (χ a - χ b) ^ 2 * (-β * g_β_2d β (θ a - θ b))
        = ∑ a ∈ 𝒮, ∑ b ∈ 𝒮ᶜ, (-β * g_β_2d β (θ a - θ b))
          + ∑ a ∈ 𝒮ᶜ, ∑ b ∈ 𝒮, (-β * g_β_2d β (θ a - θ b)) := by
      rw [← Finset.sum_add_sum_compl 𝒮
        (fun a => ∑ b : Idx n, (χ a - χ b) ^ 2 * (-β * g_β_2d β (θ a - θ b)))]
      congr 1
      · exact Finset.sum_congr rfl fun a ha => hrowMem a ha
      · exact Finset.sum_congr rfl fun a ha => hrowNot a (Finset.mem_compl.mp ha)
    have hswap : ∑ a ∈ 𝒮ᶜ, ∑ b ∈ 𝒮, (-β * g_β_2d β (θ a - θ b))
        = ∑ i ∈ 𝒮, ∑ j ∈ 𝒮ᶜ, (-β * g_β_2d β (θ i - θ j)) := by
      rw [Finset.sum_comm]
      exact Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => by
        rw [g_β_2d_symm]
    have hpull : ∑ i ∈ 𝒮, ∑ j ∈ 𝒮ᶜ, (-β * g_β_2d β (θ i - θ j))
        = -β * ∑ i ∈ 𝒮, ∑ j ∈ 𝒮ᶜ, g_β_2d β (θ i - θ j) := by
      rw [Finset.mul_sum]
      exact Finset.sum_congr rfl fun i _ => (Finset.mul_sum _ _ _).symm
    rw [hsplit, hswap, hpull]
    have hn2 : (0 : ℝ) < (n : ℝ) ^ 2 :=
      pow_pos ((Nat.cast_nonneg n).lt_of_ne (Ne.symm hn)) 2
    field_simp
    ring

/-- **Equation (eq: taylor2).** *The block Hessian is non-positive.*

At a critical point of `𝖤_β` whose Hessian is non-positive,

  `Σ_{i ∈ 𝒮} Σ_{j ∈ 𝒮} ∂_{θ_i} ∂_{θ_j} 𝖤_β(θ) ≤ 0`

for every `𝒮 ⊂ [n]`; the left-hand side is the second derivative along `1_𝒮`
computed in `secondDeriv_torusEnergy_block`.

Source: arXiv:2312.10794v5, Appendix B, `eq: taylor2`. -/
theorem taylor2_inequality
    (β : ℝ) (hβ : β ≠ 0) (hn : (n : ℝ) ≠ 0) (θ : Angles n) (𝒮 : Finset (Idx n))
    (h_hess : TorusHessianNonPos n β θ) :
    -((n : ℝ) ^ 2)⁻¹ * ∑ i ∈ 𝒮, ∑ j ∈ 𝒮ᶜ, g_β_2d β (θ i - θ j) ≤ 0 :=
  h_hess _ _ (secondDeriv_torusEnergy_block n β hβ hn θ 𝒮)

/-- **Equation (eq: taylor3).** *Sub-block inequality.*

  `Σ_{i ∈ 𝒮} Σ_{j ∈ 𝒮^c} g_β(θ_i - θ_j) ≥ 0`,

which is `eq: taylor2` divided by `-n^{-2}`.

Source: arXiv:2312.10794v5, Appendix B, `eq: taylor3`. -/
theorem taylor3_inequality
    (β : ℝ) (hβ : β ≠ 0) (hn : (n : ℝ) ≠ 0) (θ : Angles n) (𝒮 : Finset (Idx n))
    (h_hess : TorusHessianNonPos n β θ) :
    0 ≤ ∑ i ∈ 𝒮, ∑ j ∈ 𝒮ᶜ, g_β_2d β (θ i - θ j) := by
  have h := taylor2_inequality n β hβ hn θ 𝒮 h_hess
  have hn2 : (0 : ℝ) < (n : ℝ) ^ 2 :=
    pow_pos ((Nat.cast_nonneg n).lt_of_ne (Ne.symm hn)) 2
  nlinarith [h, inv_pos.mpr hn2]

/-! ### The hypotheses are satisfiable -/

/-- A single token interacts only with itself, so `𝖤_β` is constant on `𝕋^1`
and its Hessian vanishes: the hypotheses of `taylor2_inequality` and
`taylor3_inequality` are satisfiable. -/
example : (1 : ℝ) ≠ 0 ∧ ((1 : ℕ) : ℝ) ≠ 0 ∧ TorusHessianNonPos 1 1 (fun _ => 0) := by
  refine ⟨one_ne_zero, by norm_num, ?_⟩
  rintro v c ⟨f', hf', hc⟩
  have hconst : (fun u : ℝ => torusEnergy 1 1 (fun i => (0 : ℝ) + u * v i))
      = fun _ : ℝ => torusEnergy 1 1 (fun _ : Idx 1 => (0 : ℝ)) := by
    funext u
    simp [torusEnergy]
  have hf0 : f' = fun _ : ℝ => (0 : ℝ) := by
    funext s
    exact (hf' s).unique (by rw [hconst]; exact hasDerivAt_const s _)
  rw [hf0] at hc
  have hc0 : c = 0 := hc.unique (hasDerivAt_const (0 : ℝ) (0 : ℝ))
  simp [hc0]

end Perspective
end Transformer
