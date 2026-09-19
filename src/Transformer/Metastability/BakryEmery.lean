/-
# Metastability — the Bakry–Émery lemma (§3.1 of 2410.06833v1)

`Lemma lem: bakry-emery` with `ineq: Almost Hessian`: along the ascending
gradient flow `Ẋ = ∇𝖤(X)`, a Hessian bound `⟨∇𝖤, Hess 𝖤 ∇𝖤⟩ ≤ -c‖∇𝖤‖²`
forces the total rise of `𝖤` to be at most `(2c)⁻¹‖∇𝖤(u)‖²`.

The proof is the paper's: Grönwall on `‖∇𝖤(X(t))‖²`, whose derivative is
twice the Hessian quadratic form, gives `‖∇𝖤(X(t))‖² ≤ ‖∇𝖤(u)‖² e^{-2ct}`;
then `d/dt 𝖤(X(t)) = ‖∇𝖤(X(t))‖²` is integrated by observing that
`t ↦ 𝖤(X(t)) + (2c)⁻¹‖∇𝖤(u)‖² e^{-2ct}` is antitone.

Source: arXiv:2410.06833v1, §3.1, `lem: bakry-emery`, `ineq: Almost Hessian`.
-/

import Mathlib.Analysis.Calculus.Gradient.Basic
import Mathlib.Analysis.Calculus.Deriv.MeanValue
import Mathlib.Analysis.Calculus.Deriv.Pow
import Mathlib.Analysis.SpecialFunctions.ExpDeriv

open Real

namespace Transformer
namespace Metastability

/-- **Lemma (lem: bakry-emery), ineq: Almost Hessian.**

Let `X` solve the ascending gradient flow `Ẋ(t) = ∇𝖤(X(t))` with `X(0) = u`,
`X(T) = v`, and suppose the Hessian bound

  `⟨∇𝖤(X(t)), Hess 𝖤(X(t)) ∇𝖤(X(t))⟩ ≤ -c ‖∇𝖤(X(t))‖²`   for `t ∈ [0, T]`.

Then `𝖤(v) - 𝖤(u) ≤ (2c)⁻¹ ‖∇𝖤(u)‖²`.

*Deviation from the earlier Lean statement.*  This lemma used to be stated
with `gradNorm` and `gradHess` two arbitrary functions `M → ℝ`, unrelated to
`E` and to each other, and with no flow equation on `X`; in that form it is
false — see `not_forall_bakry_emery`.  What is restored here is exactly what
the paper assumes: `gradE` is the gradient of `E` (`hgrad`), `X` follows it
(`hflow`), and `gradHess` is the derivative data of `‖∇𝖤(X(·))‖²` (`hsq`),
which is the Hessian quadratic form `⟨∇𝖤, Hess 𝖤 ∇𝖤⟩` whenever `E` is twice
differentiable.  The Riemannian Hessian itself is still not formalized: `hsq`
is how the second-order information enters, and it is a hypothesis, not a
definition.

Source: arXiv:2410.06833v1, §3.1, `lem: bakry-emery`. -/
theorem bakry_emery
    {M : Type*} [NormedAddCommGroup M] [InnerProductSpace ℝ M]
    (E : M → ℝ) (gradE : M → M) (gradHess : M → ℝ)
    (u v : M) (c T : ℝ) (hc : 0 < c) (hT : 0 < T)
    (X : ℝ → M) (hX0 : X 0 = u) (hXT : X T = v)
    (hgrad : ∀ x : M, HasFDerivAt E (innerSL ℝ (gradE x)) x)
    (hflow : ∀ t : ℝ, HasDerivAt X (gradE (X t)) t)
    (hsq : ∀ t ∈ Set.Icc (0 : ℝ) T,
      HasDerivAt (fun s => ‖gradE (X s)‖ ^ 2) (2 * gradHess (X t)) t)
    (hHess : ∀ t ∈ Set.Icc (0 : ℝ) T,
      gradHess (X t) ≤ -(c * ‖gradE (X t)‖ ^ 2)) :
    E v - E u ≤ (1 / (2 * c)) * ‖gradE u‖ ^ 2 := by
  -- `E` rises along the flow at rate `‖∇E‖²`
  have hEX : ∀ t : ℝ, HasDerivAt (fun s => E (X s)) (‖gradE (X t)‖ ^ 2) t := by
    intro t
    have h := (hgrad (X t)).comp_hasDerivAt t (hflow t)
    rw [show (innerSL ℝ (gradE (X t))) (gradE (X t)) = ‖gradE (X t)‖ ^ 2 from
      real_inner_self_eq_norm_sq _] at h
    exact h
  -- Grönwall: `‖∇E(X t)‖² e^{2ct}` is antitone
  set N : ℝ → ℝ := fun s => ‖gradE (X s)‖ ^ 2 with hN
  have hφ : ∀ t ∈ Set.Icc (0 : ℝ) T,
      HasDerivAt (fun s => N s * Real.exp (2 * c * s))
        (2 * gradHess (X t) * Real.exp (2 * c * t)
          + N t * (Real.exp (2 * c * t) * (2 * c))) t := by
    intro t ht
    have he : HasDerivAt (fun s : ℝ => 2 * c * s) (2 * c) t := by
      simpa using (hasDerivAt_id t).const_mul (2 * c)
    exact (hsq t ht).mul he.exp
  have hφanti : AntitoneOn (fun s => N s * Real.exp (2 * c * s)) (Set.Icc 0 T) := by
    refine antitoneOn_of_deriv_nonpos (convex_Icc 0 T)
      (fun t ht => ((hφ t ht).continuousAt).continuousWithinAt) ?_ ?_
    · intro t ht
      rw [interior_Icc] at ht
      exact ((hφ t (Set.mem_Icc_of_Ioo ht)).differentiableAt).differentiableWithinAt
    · intro t ht
      rw [interior_Icc] at ht
      rw [(hφ t (Set.mem_Icc_of_Ioo ht)).deriv]
      have h1 := hHess t (Set.mem_Icc_of_Ioo ht)
      have h2 : (0 : ℝ) < Real.exp (2 * c * t) := Real.exp_pos _
      nlinarith
  have hgron : ∀ t ∈ Set.Icc (0 : ℝ) T, N t ≤ N 0 * Real.exp (-(2 * c * t)) := by
    intro t ht
    have h := hφanti (Set.left_mem_Icc.2 hT.le) ht ht.1
    simp only [mul_zero, Real.exp_zero, mul_one] at h
    have h2 : (0 : ℝ) < Real.exp (2 * c * t) := Real.exp_pos _
    rw [Real.exp_neg, ← div_eq_mul_inv, le_div_iff₀ h2]
    exact h
  -- integrate: `E(X t) + N(0) e^{-2ct}/(2c)` is antitone
  set K : ℝ := N 0 / (2 * c) with hK
  have hψ : ∀ t ∈ Set.Icc (0 : ℝ) T,
      HasDerivAt (fun s => E (X s) + K * Real.exp (-(2 * c) * s))
        (N t + K * (Real.exp (-(2 * c) * t) * (-(2 * c)))) t := by
    intro t _
    have he : HasDerivAt (fun s : ℝ => -(2 * c) * s) (-(2 * c)) t := by
      simpa using (hasDerivAt_id t).const_mul (-(2 * c))
    exact (hEX t).add (he.exp.const_mul K)
  have hψanti : AntitoneOn (fun s => E (X s) + K * Real.exp (-(2 * c) * s))
      (Set.Icc 0 T) := by
    refine antitoneOn_of_deriv_nonpos (convex_Icc 0 T)
      (fun t ht => ((hψ t ht).continuousAt).continuousWithinAt) ?_ ?_
    · intro t ht
      rw [interior_Icc] at ht
      exact ((hψ t (Set.mem_Icc_of_Ioo ht)).differentiableAt).differentiableWithinAt
    · intro t ht
      rw [interior_Icc] at ht
      rw [(hψ t (Set.mem_Icc_of_Ioo ht)).deriv]
      have h1 := hgron t (Set.mem_Icc_of_Ioo ht)
      have h2 : -(2 * c) * t = -(2 * c * t) := by ring
      rw [h2, hK]
      field_simp
      linarith
  have hfin := hψanti (Set.left_mem_Icc.2 hT.le) (Set.right_mem_Icc.2 hT.le) hT.le
  simp only [mul_zero, Real.exp_zero, mul_one, hX0, hXT] at hfin
  have hpos : (0 : ℝ) < Real.exp (-(2 * c) * T) := Real.exp_pos _
  have hK0 : 0 ≤ K := by
    simp only [hK, hN]; positivity
  have hle : E v - E u ≤ K - K * Real.exp (-(2 * c) * T) := by linarith
  have hKle : K - K * Real.exp (-(2 * c) * T) ≤ K := by nlinarith
  have hKeq : K = (1 / (2 * c)) * ‖gradE u‖ ^ 2 := by
    simp only [hK, hN, hX0]; ring
  linarith

/-- The hypotheses of `bakry_emery` are satisfiable, and not only by a
vanishing gradient: on `M = ℝ` take `𝖤(y) = -y²/2`, whose gradient is
`∇𝖤(y) = -y` and whose ascending gradient flow through `u = 1` is
`X(t) = e^{-t}`.  Then `‖∇𝖤(X(t))‖² = e^{-2t}` has derivative `-2e^{-2t}`,
so the Hessian quadratic form is `gradHess(x) = -x²`, and
`ineq: Almost Hessian` holds at `c = 1` with equality.  The gradient at `u`
is `-1 ≠ 0`, so the conclusion is not the trivial `0 ≤ 0`. -/
example :
    (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
      Real.exp (-(0 : ℝ)) = 1 ∧ Real.exp (-(1 : ℝ)) = Real.exp (-1) ∧
      (∀ x : ℝ, HasFDerivAt (fun y : ℝ => -y ^ 2 / 2) (innerSL ℝ (-x)) x) ∧
      (∀ t : ℝ, HasDerivAt (fun s : ℝ => Real.exp (-s)) (-Real.exp (-t)) t) ∧
      (∀ t ∈ Set.Icc (0 : ℝ) 1,
        HasDerivAt (fun s : ℝ => ‖-Real.exp (-s)‖ ^ 2) (2 * -(Real.exp (-t) ^ 2)) t) ∧
      (∀ t ∈ Set.Icc (0 : ℝ) 1,
        -(Real.exp (-t) ^ 2) ≤ -(1 * ‖-Real.exp (-t)‖ ^ 2)) ∧
      (-(1 : ℝ) ≠ 0) := by
  have hnormsq : ∀ s : ℝ, (fun s : ℝ => ‖-Real.exp (-s)‖ ^ 2) s = Real.exp (-2 * s) := by
    intro s
    show ‖-Real.exp (-s)‖ ^ 2 = Real.exp (-2 * s)
    rw [norm_neg, Real.norm_eq_abs, abs_of_pos (Real.exp_pos _),
      show (-2 : ℝ) * s = -s + -s by ring, Real.exp_add, sq]
  refine ⟨one_pos, one_pos, by simp, rfl, ?_, ?_, ?_, ?_, by norm_num⟩
  · intro x
    have hd : HasDerivAt (fun y : ℝ => -y ^ 2 / 2) (-x) x := by
      have h := ((hasDerivAt_pow 2 x).neg).div_const 2
      norm_num at h
      convert h using 1
      ring
    have he : innerSL ℝ (-x) = ContinuousLinearMap.smulRight (1 : ℝ →L[ℝ] ℝ) (-x) :=
      ContinuousLinearMap.ext fun y => by simp [mul_comm]
    rw [he]
    exact hd.hasFDerivAt
  · intro t
    simpa using ((hasDerivAt_neg t).exp)
  · intro t _
    rw [funext hnormsq]
    have he : HasDerivAt (fun s : ℝ => -2 * s) (-2 : ℝ) t := by
      simpa using (hasDerivAt_id t).const_mul (-2 : ℝ)
    have hd := he.exp
    convert hd using 1
    rw [show (-2 : ℝ) * t = -t + -t by ring, Real.exp_add]
    ring
  · intro t _
    rw [norm_neg, Real.norm_eq_abs, abs_of_pos (Real.exp_pos _), one_mul]

/-- **The lemma is false with `gradNorm` and `gradHess` left free.**

The statement `bakry_emery` used to carry — `E`, `gradNorm`, `gradHess`
arbitrary, no flow equation — is refuted by `𝖤 = id`, `gradNorm ≡ 0`,
`gradHess ≡ -1`, `X = id`, `u = 0`, `v = 1`, `c = T = 1`: the Hessian
hypothesis reads `-1 ≤ 0` and the conclusion reads `1 ≤ 0`.  This is why
`bakry_emery` above ties `gradE`, `gradHess` and `X` to `E`.

Source: arXiv:2410.06833v1, §3.1, `lem: bakry-emery`. -/
theorem not_forall_bakry_emery :
    ¬ ∀ (E gradNorm gradHess : ℝ → ℝ) (u v c T : ℝ), 0 < c → 0 < T →
        ∀ X : ℝ → ℝ, X 0 = u → X T = v →
          (∀ t : ℝ, 0 ≤ t → t ≤ T → gradHess (X t) ≤ -(c * gradNorm (X t) ^ 2)) →
          E v - E u ≤ (1 / (2 * c)) * gradNorm u ^ 2 := by
  intro h
  have := h (fun x => x) (fun _ => 0) (fun _ => -1) 0 1 1 1 one_pos one_pos
    (fun x => x) rfl rfl (fun _ _ _ => by norm_num)
  norm_num at this

end Metastability
end Transformer
