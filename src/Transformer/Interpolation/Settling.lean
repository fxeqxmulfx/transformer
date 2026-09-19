/-
# Measure-to-measure interpolation — Exponential settling near the attractor

`eq: Hartman.Grobman` of §4 of arXiv:2411.04551v3: while the parameters are
frozen, the perceptron-only flow on the sphere settles onto its attractor
`ω_+` at an exponential rate,

  `d_g(x(t), ω_+) ≤ K e^{-λ t}`,   `t ≥ 0`,

with `λ > 0` and `K ≥ 1` depending on `x_0`, `ε` and `γ` alone.

**What the source says and what is changed here.**  The paper reads the rate
off the Hartman-Grobman linearization at a hyperbolic sink of the frozen
field.  The linearization is not formalized; its consequence is, and it is
carried as an explicit hypothesis — that along the trajectory the field points
back towards `ω_+` at a linear rate, `⟨F(x(t)), x(t) - ω_+⟩ ≤ -λ ‖x(t) -
ω_+‖²`.  Granted that, the exponential bound is Gronwall's, and it is proved
below.  Nothing else of the statement is weakened: the conclusion is the
paper's, with `K = max(1, ‖x_0 - ω_+‖)` exhibited.

The form the statement had before — for *every* path `x : ℝ → 𝕊^{d-1}` and
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
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp

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

/-- `eq: neural.ode.sphere` is the perceptron field, particle by particle:
the two read the same. -/
theorem neuralODESphere_iff_perceptronField
    (M : ℕ) (W U : ℝ → ParamMatrix d) (b : ℝ → EucSpace d)
    (x : ℝ → Idx M → EucSpace d) :
    neuralODESphere d M W U b x ↔
      ∀ t : ℝ, ∀ i : Idx M,
        HasDerivAt (fun s => x s i) (perceptronField d (W t) (U t) (b t) (x t i)) t :=
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

/-- **Equation (eq: Hartman.Grobman).** *Exponential settling near the
attractor.*

With the parameters frozen, a trajectory of the perceptron-only flow that the
field pushes back towards `ω_+` at the linear rate `λ` satisfies

  `‖x(t) - ω_+‖ ≤ K e^{-λ t}`   for `t ≥ 0`,   `K = max(1, ‖x(0) - ω_+‖) ≥ 1`.

The linear rate is the hypothesis `hcontr`; the paper obtains it from the
Hartman-Grobman linearization at the hyperbolic sink `ω_+`, which is not
formalized here and is therefore carried as an explicit hypothesis rather
than used silently.  See the header for what this changes.

Source: arXiv:2411.04551v3, §4, `eq: Hartman.Grobman`. -/
theorem Hartman_Grobman
    (W U : ParamMatrix d) (b : EucSpace d)
    (x : ℝ → EucSpace d) (ω_plus : EucSpace d) (lam : ℝ) (hlam : 0 < lam)
    (hflow : ∀ t : ℝ, 0 ≤ t → HasDerivAt x (perceptronField d W U b (x t)) t)
    (hcontr : ∀ t : ℝ, 0 ≤ t →
      inner (𝕜 := ℝ) (perceptronField d W U b (x t)) (x t - ω_plus)
        ≤ -(lam * ‖x t - ω_plus‖ ^ 2)) :
    ∃ K : ℝ, 1 ≤ K ∧ 0 < lam ∧
      ∀ t : ℝ, 0 ≤ t → ‖x t - ω_plus‖ ≤ K * Real.exp (-(lam * t)) := by
  refine ⟨max 1 ‖x 0 - ω_plus‖, le_max_left _ _, hlam, fun t ht => ?_⟩
  refine le_trans
    (norm_sub_le_of_contraction d x (fun s => perceptronField d W U b (x s)) ω_plus lam
      hflow hcontr t ht) ?_
  gcongr
  exact le_max_right _ _

/-- The hypotheses of `Hartman_Grobman` are satisfiable, and by a trajectory
that genuinely moves.  In `ℝ^1` with `𝐔 = 0`, `b = e_1` and `𝐖 = Id` the
perceptron field is `F(s e_1) = (1 - s²) e_1`, whose solution started at the
origin is `x(t) = tanh(t) e_1`; it settles onto `ω_+ = e_1`, and does so at
the rate `λ = 1` because `⟨F(x(t)), x(t) - e_1⟩ = -(1 - T)²(1 + T)` with
`T = tanh t ≥ 0`. -/
example :
    ∃ (W U : ParamMatrix 1) (b ω_plus : EucSpace 1) (x : ℝ → EucSpace 1) (lam : ℝ),
      0 < lam ∧ x 0 ≠ ω_plus ∧
      (∀ t : ℝ, 0 ≤ t → HasDerivAt x (perceptronField 1 W U b (x t)) t) ∧
      (∀ t : ℝ, 0 ≤ t →
        inner (𝕜 := ℝ) (perceptronField 1 W U b (x t)) (x t - ω_plus)
          ≤ -(lam * ‖x t - ω_plus‖ ^ 2)) := by
  have he : ‖((basePoint 0 : SSphere 1) : EucSpace 1)‖ = 1 :=
    mem_sphere_zero_iff_norm.mp (basePoint 0).2
  have htanh : ∀ t : ℝ, HasDerivAt Real.tanh (1 - Real.tanh t ^ 2) t := by
    intro t
    have hcosh : Real.cosh t ≠ 0 := ne_of_gt (Real.cosh_pos t)
    have hfun : Real.tanh = fun s : ℝ => Real.sinh s / Real.cosh s :=
      funext Real.tanh_eq_sinh_div_cosh
    have h : HasDerivAt Real.tanh
        ((Real.cosh t * Real.cosh t - Real.sinh t * Real.sinh t) / Real.cosh t ^ 2) t := by
      rw [hfun]
      exact (Real.hasDerivAt_sinh t).div (Real.hasDerivAt_cosh t) hcosh
    refine h.congr_deriv ?_
    rw [Real.tanh_eq_sinh_div_cosh]
    field_simp
  have hfield : ∀ T : ℝ,
      perceptronField 1 (ContinuousLinearMap.id ℝ (EucSpace 1)) 0
          ((basePoint 0 : SSphere 1) : EucSpace 1)
          (T • ((basePoint 0 : SSphere 1) : EucSpace 1))
        = (1 - T ^ 2) • ((basePoint 0 : SSphere 1) : EucSpace 1) := by
    intro T
    have hrelu : (EuclideanSpace.equiv (Fin 1) ℝ).symm
        (fun k => max ((EuclideanSpace.equiv (Fin 1) ℝ
          (((0 : ParamMatrix 1)) (T • ((basePoint 0 : SSphere 1) : EucSpace 1))
            + ((basePoint 0 : SSphere 1) : EucSpace 1))) k) 0)
          = ((basePoint 0 : SSphere 1) : EucSpace 1) := by
      ext k
      fin_cases k
      simp [basePoint]
    rw [perceptronField, hrelu, ContinuousLinearMap.coe_id', id_eq, proj,
      real_inner_smul_left, real_inner_self_eq_norm_sq, he]
    module
  refine ⟨ContinuousLinearMap.id ℝ (EucSpace 1), 0,
    ((basePoint 0 : SSphere 1) : EucSpace 1), ((basePoint 0 : SSphere 1) : EucSpace 1),
    fun t => Real.tanh t • ((basePoint 0 : SSphere 1) : EucSpace 1), 1,
    one_pos, ?_, ?_, ?_⟩
  · show Real.tanh 0 • ((basePoint 0 : SSphere 1) : EucSpace 1)
        ≠ ((basePoint 0 : SSphere 1) : EucSpace 1)
    rw [Real.tanh_zero, zero_smul]
    intro h
    rw [← h, norm_zero] at he
    norm_num at he
  · intro t _
    rw [hfield]
    exact (htanh t).smul_const _
  · intro t ht
    have hT0 : 0 ≤ Real.tanh t := by
      rw [Real.tanh_eq_sinh_div_cosh]
      exact div_nonneg (Real.sinh_nonneg_iff.mpr ht) (Real.cosh_pos t).le
    have hT1 : Real.tanh t < 1 := Real.tanh_lt_one t
    have hsub : Real.tanh t • ((basePoint 0 : SSphere 1) : EucSpace 1)
        - ((basePoint 0 : SSphere 1) : EucSpace 1)
        = (Real.tanh t - 1) • ((basePoint 0 : SSphere 1) : EucSpace 1) := by
      module
    rw [hfield, hsub, real_inner_smul_left, real_inner_smul_right,
      real_inner_self_eq_norm_sq, he, norm_smul, Real.norm_eq_abs, he]
    have habs : |Real.tanh t - 1| = 1 - Real.tanh t := by
      rw [abs_of_nonpos (by linarith)]
      ring
    rw [habs]
    nlinarith [sq_nonneg (1 - Real.tanh t)]

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
