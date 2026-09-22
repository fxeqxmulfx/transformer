/-
# Measure-to-measure interpolation — Exponential settling near the attractor

`eq: Hartman.Grobman` of §4 of arXiv:2411.04551v3 asserts that, while the
parameters are frozen, the perceptron-only flow on the sphere settles onto its
attractor `ω_+` at an exponential rate,

  `d_g(x(t), ω_+) ≤ K e^{-λ t}`,   `t ≥ 0`.

It is proved, for the flow `eq: neural.ode.separation` of Step 2 and every
start in `𝒮_+`, as `Interpolation.Hartman_Grobman` in
`Interpolation.HartmanGrobman`.  This file holds what that proof and the rest
of §4 share: the perceptron field `perceptronField`, and the Gronwall estimate
`norm_sub_le_of_contraction` — a field pointing back towards `w` at a linear
rate brings the path to `w` at that rate — whose hypothesis
`Interpolation.Hartman_Grobman` verifies.

The form the statement once had — for *every* path `x : ℝ → 𝕊^{d-1}` and
*every* `ω_+` — is false, and `not_forall_Hartman_Grobman` refutes it: a path
standing still at the antipode of `ω_+` keeps the distance `2`, which no
`K e^{-λ t}` can dominate for all `t`.

Source: arXiv:2411.04551v3, §4, `eq: Hartman.Grobman`.
-/

import Transformer.Interpolation.Basic
import Transformer.Interpolation.NeuralODE
import Mathlib.Analysis.InnerProductSpace.Calculus
import Mathlib.Analysis.Calculus.Deriv.MeanValue
import Mathlib.Analysis.SpecialFunctions.ExpDeriv

open scoped BigOperators
open Real

namespace Transformer
namespace Interpolation

variable (d : ℕ)

/-- The **perceptron velocity field** at frozen parameters: the right-hand
side of `eq: neural.ode.sphere`,

  `F(z) = Proj_z 𝐖 (𝐔 z + b)_+`.

Source: arXiv:2411.04551v3, §4, `eq: neural.ode.sphere`. -/
noncomputable def perceptronField (W U : ParamMatrix d) (b z : EucSpace d) :
    EucSpace d :=
  proj d z
    ((W)
      (EuclideanSpace.equiv _ ℝ |>.symm
        (fun k => max ((EuclideanSpace.equiv _ ℝ ((U) z + b)) k) 0)))

/-- `eq: neural.ode.sphere` is the integrated perceptron field, particle by
particle: the two read the same. -/
theorem neuralODESphere_iff_perceptronField
    (M : ℕ) (W U : ℝ → ParamMatrix d) (b : ℝ → EucSpace d)
    (x : ℝ → Idx M → EucSpace d) :
    neuralODESphere d M W U b x ↔
      ∀ t : ℝ, ∀ i : Idx M,
        IntervalIntegrable (fun s => perceptronField d (W s) (U s) (b s) (x s i))
            MeasureTheory.volume 0 t ∧
          x t i = x 0 i + ∫ s in (0 : ℝ)..t, perceptronField d (W s) (U s) (b s) (x s i) :=
  Iff.rfl

/-- **Gronwall's lemma in the form the settling estimate needs.**

If the velocity of a path points back towards `w` at a linear rate — the inner
product `⟨ẋ(t), x(t) - w⟩` is at most `-λ ‖x(t) - w‖²` — then the distance to
`w` decays at the rate `λ`.

The proof is the antitonicity of `t ↦ ‖x(t) - w‖² e^{2λt}`: its derivative is
`(2⟨ẋ, x - w⟩ + 2λ‖x - w‖²) e^{2λt} ≤ 0`. -/
theorem norm_sub_le_of_contraction
    (x v : ℝ → EucSpace d) (w : EucSpace d) (lam : ℝ)
    (hx : ∀ t : ℝ, 0 ≤ t → HasDerivAt x (v t) t)
    (hcontr : ∀ t : ℝ, 0 ≤ t →
      inner (𝕜 := ℝ) (v t) (x t - w) ≤ -(lam * ‖x t - w‖ ^ 2)) :
    ∀ t : ℝ, 0 ≤ t → ‖x t - w‖ ≤ ‖x 0 - w‖ * Real.exp (-(lam * t)) := by
  have hN : ∀ s : ℝ, 0 ≤ s →
      HasDerivAt (fun r => ‖x r - w‖ ^ 2)
        (2 * inner (𝕜 := ℝ) (v s) (x s - w)) s := by
    intro s hs
    have hd : HasDerivAt (fun r => x r - w) (v s) s := (hx s hs).sub_const w
    have h := HasDerivAt.inner (𝕜 := ℝ) hd hd
    have heq : (fun r => ‖x r - w‖ ^ 2)
        = fun r => (inner (𝕜 := ℝ) (x r - w) (x r - w) : ℝ) := by
      funext r
      rw [real_inner_self_eq_norm_sq]
    rw [heq]
    refine h.congr_deriv ?_
    rw [real_inner_comm (x s - w) (v s)]
    ring
  have hφ : ∀ s : ℝ, 0 ≤ s →
      HasDerivAt (fun r => ‖x r - w‖ ^ 2 * Real.exp (2 * lam * r))
        ((2 * inner (𝕜 := ℝ) (v s) (x s - w)) * Real.exp (2 * lam * s)
          + ‖x s - w‖ ^ 2 * (Real.exp (2 * lam * s) * (2 * lam))) s := by
    intro s hs
    have hlin : HasDerivAt (fun r : ℝ => 2 * lam * r) (2 * lam) s := by
      simpa using (hasDerivAt_id s).const_mul (2 * lam)
    exact (hN s hs).mul hlin.exp
  have hanti : AntitoneOn (fun r => ‖x r - w‖ ^ 2 * Real.exp (2 * lam * r))
      (Set.Ici (0 : ℝ)) := by
    refine antitoneOn_of_deriv_nonpos (convex_Ici 0) ?_ ?_ ?_
    · exact fun s hs => ((hφ s hs).differentiableAt).continuousAt.continuousWithinAt
    · intro s hs
      rw [interior_Ici] at hs
      exact (hφ s (le_of_lt hs)).differentiableAt.differentiableWithinAt
    · intro s hs
      rw [interior_Ici] at hs
      rw [(hφ s (le_of_lt hs)).deriv]
      have hc := hcontr s (le_of_lt hs)
      have hexp : 0 < Real.exp (2 * lam * s) := Real.exp_pos _
      nlinarith
  intro t ht
  have hle := hanti (Set.mem_Ici.mpr le_rfl) (Set.mem_Ici.mpr ht) ht
  simp only [mul_zero, Real.exp_zero, mul_one] at hle
  have hexp : 0 < Real.exp (2 * lam * t) := Real.exp_pos _
  have hsq : ‖x t - w‖ ^ 2 ≤ (‖x 0 - w‖ * Real.exp (-(lam * t))) ^ 2 := by
    have hdiv : ‖x t - w‖ ^ 2 ≤ ‖x 0 - w‖ ^ 2 / Real.exp (2 * lam * t) :=
      (le_div_iff₀ hexp).mpr hle
    have h2 : Real.exp (-(lam * t)) ^ 2 = (Real.exp (2 * lam * t))⁻¹ := by
      rw [← Real.exp_neg, sq, ← Real.exp_add]
      ring_nf
    calc ‖x t - w‖ ^ 2 ≤ ‖x 0 - w‖ ^ 2 / Real.exp (2 * lam * t) := hdiv
      _ = (‖x 0 - w‖ * Real.exp (-(lam * t))) ^ 2 := by
          rw [mul_pow, h2, div_eq_mul_inv]
  have h0 : 0 ≤ ‖x 0 - w‖ * Real.exp (-(lam * t)) :=
    mul_nonneg (norm_nonneg _) (Real.exp_pos _).le
  nlinarith [norm_nonneg (x t - w)]

/-- The hypotheses of `norm_sub_le_of_contraction` are satisfiable, and not
only by a path standing still: `x(t) = e^{-t} e_1` runs into `w = 0` at the
rate `λ = 1`, with equality in the contraction bound at every time. -/
example :
    (∀ t : ℝ, 0 ≤ t →
        HasDerivAt (fun s : ℝ => Real.exp (-s) • ((basePoint 0 : SSphere 1) : EucSpace 1))
          (-(Real.exp (-t) • ((basePoint 0 : SSphere 1) : EucSpace 1))) t) ∧
      (∀ t : ℝ, 0 ≤ t →
        inner (𝕜 := ℝ) (-(Real.exp (-t) • ((basePoint 0 : SSphere 1) : EucSpace 1)))
            (Real.exp (-t) • ((basePoint 0 : SSphere 1) : EucSpace 1) - 0)
          ≤ -((1 : ℝ) * ‖Real.exp (-t) • ((basePoint 0 : SSphere 1) : EucSpace 1) - 0‖ ^ 2)) := by
  have he : ‖((basePoint 0 : SSphere 1) : EucSpace 1)‖ = 1 :=
    mem_sphere_zero_iff_norm.mp (basePoint 0).2
  constructor
  · intro t _
    have h : HasDerivAt (fun s : ℝ => Real.exp (-s)) (Real.exp (-t) * (-1)) t :=
      ((hasDerivAt_id t).neg).exp
    have h2 := h.smul_const ((basePoint 0 : SSphere 1) : EucSpace 1)
    refine h2.congr_deriv ?_
    rw [mul_neg, mul_one, neg_smul]
  · intro t _
    rw [sub_zero, inner_neg_left, real_inner_smul_left, real_inner_smul_right,
      real_inner_self_eq_norm_sq, he, norm_smul, he]
    simp [sq]

/-- **The form `eq: Hartman.Grobman` had before is false.**

Read for *every* path `x : ℝ → 𝕊^{d-1}` and every `ω_+`, with no relation
between the two, the estimate fails at once: the constant path sitting at the
antipode of `ω_+` keeps the distance `2` forever, while `K e^{-λt}` with
`λ > 0` drops below `2` — at `t = K/λ` it is already `K e^{-K} ≤ K/(K+1) < 1`.

Source: arXiv:2411.04551v3, §4, `eq: Hartman.Grobman` (the claim this
refutes is the unrestricted reading of it, not the paper's). -/
theorem not_forall_Hartman_Grobman :
    ¬ ∀ (D : ℕ) (_γ ω_plus : SSphere D) (ε : ℝ), 0 < ε → ∀ x : ℝ → SSphere D,
        ∃ K lam : ℝ, 1 ≤ K ∧ 0 < lam ∧
          ∀ t : ℝ, 0 ≤ t →
            ‖((x t : EucSpace D)) - ((ω_plus : EucSpace D))‖
              ≤ K * Real.exp (-(lam * t)) := by
  intro h
  obtain ⟨K, lam, hK, hlam, hbound⟩ :=
    h 1 (basePoint 0) (basePoint 0) 1 one_pos (fun _ => antipode 1 (basePoint 0))
  have he : ‖((basePoint 0 : SSphere 1) : EucSpace 1)‖ = 1 :=
    mem_sphere_zero_iff_norm.mp (basePoint 0).2
  have hdist : ‖((antipode 1 (basePoint 0) : SSphere 1) : EucSpace 1)
      - ((basePoint 0 : SSphere 1) : EucSpace 1)‖ = 2 := by
    have hsub : ((antipode 1 (basePoint 0) : SSphere 1) : EucSpace 1)
        - ((basePoint 0 : SSphere 1) : EucSpace 1)
        = (-2 : ℝ) • ((basePoint 0 : SSphere 1) : EucSpace 1) := by
      show -((basePoint 0 : SSphere 1) : EucSpace 1)
        - ((basePoint 0 : SSphere 1) : EucSpace 1) = _
      module
    rw [hsub, norm_smul, Real.norm_eq_abs, he]
    norm_num
  have hb := hbound (K / lam) (div_nonneg (by linarith) hlam.le)
  rw [hdist, show lam * (K / lam) = K by field_simp] at hb
  have hEF : Real.exp (-K) * Real.exp K = 1 := by
    rw [← Real.exp_add]
    simp
  have hF : K + 1 ≤ Real.exp K := Real.add_one_le_exp K
  have hmul := mul_le_mul_of_nonneg_right hb (Real.exp_pos K).le
  nlinarith [Real.exp_pos K, Real.exp_pos (-K)]

end Interpolation
end Transformer
