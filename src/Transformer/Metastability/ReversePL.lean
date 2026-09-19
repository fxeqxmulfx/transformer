/-
# Metastability — §3.3, acceleration of the gradient between metastable states

arXiv:2410.06833v1, `sec: acceleration`.

Where the Otto–Reznikoff framework of §3.1 bounds the energy gap by the
gradient and so forces decay, §3.3 runs the comparison the other way: along
the ascent flow `Ẋ = ∇𝖤(X)`, a Hessian bounded *below* on the gradient
direction — `eq: hessian.lb.reverse.pl` — makes `‖∇𝖤(X(t))‖²` grow
exponentially.  That is the whole of the paragraph: a chain rule, an
inequality, and Grönwall.

`reversePL` is the inequality; `reverse_PL_acceleration` is the Grönwall step
that integrates it.  The chain rule is a hypothesis, for the reason its
docstring records, and `not_forall_reverse_PL_acceleration` shows that it has
to be.
-/

import Transformer.Basic
import Mathlib.Analysis.Calculus.Deriv.MeanValue
import Mathlib.Analysis.SpecialFunctions.ExpDeriv

open Real

namespace Transformer
namespace Metastability

/-- **Equation (eq: hessian.lb.reverse.pl).**

  `⟨Hess 𝖤(X(t)) ∇𝖤(X(t)), ∇𝖤(X(t))⟩ ≥ c ‖∇𝖤(X(t))‖²`  for `t ∈ [0, T]`,

with `T = T_u` the exit time from the accelerating manifold `𝒜`.  The two
scalar fields are the paper's `H` and `‖∇𝖤‖`: the Riemannian structure of
§3.1 is not formalized, so they are carried abstractly.

Source: arXiv:2410.06833v1, §3.3, `eq: hessian.lb.reverse.pl`. -/
def reversePL
    {M : Type*} [NormedAddCommGroup M]
    (gradHess : M → ℝ) (gradNorm : M → ℝ)
    (c T : ℝ) (X : ℝ → M) : Prop :=
  ∀ t : ℝ, 0 ≤ t → t ≤ T →
    c * (gradNorm (X t))^2 ≤ gradHess (X t)

/-- **The acceleration of §3.3.**  Under (eq: hessian.lb.reverse.pl),

  `‖∇𝖤(X(t))‖² ≥ e^{2ct} ‖∇𝖤(X(0))‖²`   for `t ∈ [0, T]`.

**What the source says and what is changed here.**  The survey's derivation is
two lines: the chain rule along the ascent flow `Ẋ = ∇𝖤(X)`,

  `d/dt ‖∇𝖤(X(t))‖² = 2 ⟨∇𝖤(X(t)), Hess 𝖤(X(t)) ∇𝖤(X(t))⟩`,

followed by Grönwall.  `gradNorm` and `gradHess` are abstract fields here —
the Hessian is not constructed, and nothing ties either of them to a
derivative of the other — so that identity cannot be derived and is carried
as the hypothesis `hchain`, in the same way the flow itself is carried.  It is
not a technicality: without it the statement is false, which is
`not_forall_reverse_PL_acceleration` below.

The conclusion is stated on `[0, T]` rather than on `[0, T_u]`; the exit time
enters only through the range on which the hypotheses are assumed.  The sign
of `c` is not used either — Grönwall gives the estimate as written for every
`c`, and `c > 0` is what makes it an acceleration rather than a decay.

Source: arXiv:2410.06833v1, §3.3, `eq: hessian.lb.reverse.pl`. -/
theorem reverse_PL_acceleration
    {M : Type*} [NormedAddCommGroup M]
    (gradHess gradNorm : M → ℝ)
    (X : ℝ → M) (c T : ℝ) (hT : 0 < T)
    (hchain : ∀ t : ℝ, 0 ≤ t → t ≤ T →
      HasDerivAt (fun s : ℝ => (gradNorm (X s))^2) (2 * gradHess (X t)) t)
    (hrev : reversePL gradHess gradNorm c T X) :
    ∀ t : ℝ, 0 ≤ t → t ≤ T →
      (gradNorm (X 0))^2 * Real.exp (2 * c * t)
        ≤ (gradNorm (X t))^2 := by
  -- `r ↦ ‖∇𝖤(X(r))‖² e^{-2cr}` has nonnegative derivative on `[0, T]`.
  have hFd : ∀ r : ℝ, 0 ≤ r → r ≤ T →
      ∃ D : ℝ, HasDerivAt (fun s : ℝ => (gradNorm (X s))^2 * Real.exp (-(2 * c * s))) D r
        ∧ 0 ≤ D := by
    intro r h0 h1
    have hexp : HasDerivAt (fun s : ℝ => Real.exp (-(2 * c * s)))
        (Real.exp (-(2 * c * r)) * -(2 * c)) r := by
      simpa using (((hasDerivAt_id r).const_mul (2 * c)).neg).exp
    refine ⟨2 * gradHess (X r) * Real.exp (-(2 * c * r))
      + (gradNorm (X r))^2 * (Real.exp (-(2 * c * r)) * -(2 * c)),
      (hchain r h0 h1).mul hexp, ?_⟩
    have hpos : (0 : ℝ) < Real.exp (-(2 * c * r)) := Real.exp_pos _
    nlinarith [hrev r h0 h1]
  have hmono : MonotoneOn (fun s : ℝ => (gradNorm (X s))^2 * Real.exp (-(2 * c * s)))
      (Set.Icc 0 T) := by
    refine monotoneOn_of_deriv_nonneg (convex_Icc 0 T) ?_ ?_ ?_
    · intro s hs
      obtain ⟨D, hD, -⟩ := hFd s hs.1 hs.2
      exact hD.continuousAt.continuousWithinAt
    · intro s hs
      rw [interior_Icc] at hs
      obtain ⟨D, hD, -⟩ := hFd s hs.1.le hs.2.le
      exact hD.differentiableAt.differentiableWithinAt
    · intro s hs
      rw [interior_Icc] at hs
      obtain ⟨D, hD, hDnn⟩ := hFd s hs.1.le hs.2.le
      rw [hD.deriv]
      exact hDnn
  intro t ht0 htT
  have hkey := hmono ⟨le_rfl, hT.le⟩ ⟨ht0, htT⟩ ht0
  simp only [mul_zero, neg_zero, Real.exp_zero, mul_one] at hkey
  refine le_of_mul_le_mul_right ?_ (Real.exp_pos (-(2 * c * t)))
  calc (gradNorm (X 0))^2 * Real.exp (2 * c * t) * Real.exp (-(2 * c * t))
      = (gradNorm (X 0))^2 := by
        rw [mul_assoc, ← Real.exp_add]; simp
    _ ≤ (gradNorm (X t))^2 * Real.exp (-(2 * c * t)) := hkey

/-- The hypotheses of `reverse_PL_acceleration` are satisfiable, and sharply:
on `M = ℝ` with `X(t) = t`, `‖∇𝖤‖(x) = e^x` and `H(x) = e^{2x}`, the chain
rule holds and `eq: hessian.lb.reverse.pl` holds with equality at `c = 1` — as
does the conclusion, `e^{2t} ≥ e^{2t}`. -/
example :
    (0 : ℝ) < 1 ∧
      (∀ t : ℝ, 0 ≤ t → t ≤ 1 →
        HasDerivAt (fun s : ℝ => (Real.exp s)^2) (2 * Real.exp (2 * t)) t) ∧
      reversePL (fun x : ℝ => Real.exp (2 * x)) (fun x : ℝ => Real.exp x) 1 1
        (fun r : ℝ => r) := by
  refine ⟨one_pos, fun t _ _ => ?_, fun t _ _ => ?_⟩
  · refine ((Real.hasDerivAt_exp t).pow 2).congr_deriv ?_
    rw [two_mul t, Real.exp_add]
    push_cast
    ring
  · refine le_of_eq ?_
    show (1 : ℝ) * Real.exp t ^ 2 = Real.exp (2 * t)
    rw [two_mul t, Real.exp_add]
    ring

/-- **The chain rule is not optional.**  With `gradNorm` and `gradHess` free,
(eq: hessian.lb.reverse.pl) alone says nothing about how `‖∇𝖤(X(t))‖²` moves,
and the conclusion fails: take both fields constant equal to `1` and
`c = 1/2`, where the inequality reads `1/2 ≤ 1` at every time while the
conclusion demands `e^{t} ≤ 1`.

This is why `reverse_PL_acceleration` carries `hchain`.

Source: arXiv:2410.06833v1, §3.3, `eq: hessian.lb.reverse.pl`. -/
theorem not_forall_reverse_PL_acceleration :
    ¬ ∀ (gradHess gradNorm : ℝ → ℝ) (X : ℝ → ℝ) (c T : ℝ), 0 < c → 0 < T →
        reversePL gradHess gradNorm c T X →
        ∀ t : ℝ, 0 ≤ t → t ≤ T →
          (gradNorm (X 0))^2 * Real.exp (2 * c * t) ≤ (gradNorm (X t))^2 := by
  intro h
  have hbad := h (fun _ => 1) (fun _ => 1) (fun r => r) (1/2) 1
    (by norm_num) one_pos (fun _ _ _ => by norm_num) 1 zero_le_one le_rfl
  norm_num at hbad

end Metastability
end Transformer
