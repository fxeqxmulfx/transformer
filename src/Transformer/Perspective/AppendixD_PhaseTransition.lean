/-
# Appendix D — the stability estimates behind `thm: phase.transition.curve`

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

The first half of Appendix D of the survey:

* `eq: almost.ortho.vec`  — almost-orthogonality of a uniform sample (Lévy),
* `e:ineqfirstpart`       — the first half of `eq: upto-t`,
* `eq: d.large`           — the definition of `d⋆(n, β)`.

`eq: stability.4ortho` and the `e:shortdist` it specializes to are in
`Perspective.AppendixD_Stability`.

The decay of `1 - γ_β(t)`, `e:ybetacloseto1`, is in
`Perspective.AppendixD_Ybeta`.  The lower bound on the smallest coordinate
`α(t)` is in `Perspective.AppendixD_Alpha`, and what the survey builds on it
in `Perspective.AppendixD_Assembly`.
-/

import Transformer.Basic
import Transformer.Perspective.AppendixD_Stability
import Transformer.Perspective.Section1_IPS
import Transformer.Perspective.Section5_HighD
import Transformer.Perspective.Section5_HighDCurve

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Perspective

variable (d n : ℕ)

/-- **Equation (eq: almost.ortho.vec).** *Almost-orthogonality of a uniform
sample (Lévy's concentration of measure).*

For `n` i.i.d. uniform points on `𝕊^{d-1}` with `2 ≤ n ≤ d` there are, with
probability at least `1 - 2 n² d^{-1/64}`, pairwise orthogonal points
`y_1, …, y_n ∈ 𝕊^{d-1}` with `‖x_i(0) - y_i‖ ≤ √(log d / d)`.

Not proved here: neither the concentration inequality on the sphere nor the
Gram–Schmidt construction of the `y_i` from it is formalized.  As in §4, the
uniform law is quantified over as `UniformTuple`, which pins it down uniquely.

Source: arXiv:2312.10794v5, Appendix D, `eq: almost.ortho.vec`. -/
theorem almost_orthogonal (hn : 2 ≤ n) (hnd : n ≤ d) :
    ∀ P : Measure (SphereTuple d n), UniformTuple d n P →
      1 - 2 * (n : ℝ) ^ 2 * (d : ℝ) ^ (-(1 / 64 : ℝ)) ≤
        (P { X₀ : SphereTuple d n |
              ∃ Y : SphereTuple d n,
                (∀ i j : Idx n, i ≠ j →
                  inner (𝕜 := ℝ) ((Y i : EucSpace d)) ((Y j : EucSpace d)) = 0) ∧
                ∀ i : Idx n,
                  ‖(X₀ i : EucSpace d) - (Y i : EucSpace d)‖
                    ≤ Real.sqrt (Real.log d / d) }).toReal := by
  sorry

/-- The hypotheses of `almost_orthogonal` are satisfiable: `d = n = 2`. -/
example : 2 ≤ 2 ∧ 2 ≤ 2 := ⟨le_rfl, le_rfl⟩


/-- **Equation (e:ineqfirstpart).** *First half of `eq: upto-t`.*

If `y` is a solution whose particles stay pairwise at the common angle `γ_β`
(that is `⟨y_i(t), y_j(t)⟩ = γ_β(t)` for `i ≠ j`, which is `thm: orthogonal`),
and `x` starts within `√(log d / d)` of it, then

  `|⟨x_i(t), x_j(t)⟩ - γ_β(t)| ≤ 2 c(β)^{n t} √(log d / d)`.

The proof is `e:shortdist` followed by Cauchy–Schwarz on
`⟨x_i, x_j⟩ - ⟨y_i, y_j⟩ = ⟨x_i - y_i, x_j⟩ + ⟨y_i, x_j - y_j⟩`, both particles
being unit vectors.  It carries `stability_orthogonal`'s added hypothesis
`0 ≤ β`.

Source: arXiv:2312.10794v5, Appendix D, `e:ineqfirstpart`. -/
theorem ineq_first_part
    (β : ℝ) (hβ : 0 ≤ β) (X Y : ℝ → SphereTuple d n) (γ : ℝ → ℝ)
    (hX : SA d n β X) (hY : SA d n β Y)
    (hM : ∀ j : Idx n,
      ‖(X 0 j : EucSpace d) - (Y 0 j : EucSpace d)‖ ≤ Real.sqrt (Real.log d / d))
    (hγ : ∀ t : ℝ, 0 ≤ t → ∀ i j : Idx n, i ≠ j →
      inner (𝕜 := ℝ) ((Y t i : EucSpace d)) ((Y t j : EucSpace d)) = γ t) :
    ∀ t : ℝ, 0 ≤ t → ∀ i j : Idx n, i ≠ j →
      |inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((X t j : EucSpace d)) - γ t|
        ≤ 2 * cBeta β ^ ((n : ℝ) * t) * Real.sqrt (Real.log d / d) := by
  intro t ht i j hij
  have hi := shortdist_bound d n β hβ X Y hX hY hM t ht i
  have hj := shortdist_bound d n β hβ X Y hX hY hM t ht j
  have hxj : ‖(X t j : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp (X t j).2
  have hyi : ‖(Y t i : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp (Y t i).2
  have key :
      inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((X t j : EucSpace d)) - γ t
        = inner (𝕜 := ℝ)
              ((X t i : EucSpace d) - (Y t i : EucSpace d)) ((X t j : EucSpace d))
          + inner (𝕜 := ℝ)
              ((Y t i : EucSpace d)) ((X t j : EucSpace d) - (Y t j : EucSpace d)) := by
    rw [← hγ t ht i j hij, inner_sub_left, inner_sub_right]
    ring
  rw [key]
  calc
    |inner (𝕜 := ℝ)
          ((X t i : EucSpace d) - (Y t i : EucSpace d)) ((X t j : EucSpace d))
        + inner (𝕜 := ℝ)
          ((Y t i : EucSpace d)) ((X t j : EucSpace d) - (Y t j : EucSpace d))|
      ≤ |inner (𝕜 := ℝ)
            ((X t i : EucSpace d) - (Y t i : EucSpace d)) ((X t j : EucSpace d))|
        + |inner (𝕜 := ℝ)
            ((Y t i : EucSpace d)) ((X t j : EucSpace d) - (Y t j : EucSpace d))| :=
        abs_add_le _ _
    _ ≤ ‖(X t i : EucSpace d) - (Y t i : EucSpace d)‖ * ‖(X t j : EucSpace d)‖
        + ‖(Y t i : EucSpace d)‖ * ‖(X t j : EucSpace d) - (Y t j : EucSpace d)‖ :=
        add_le_add (abs_real_inner_le_norm _ _) (abs_real_inner_le_norm _ _)
    _ = ‖(X t i : EucSpace d) - (Y t i : EucSpace d)‖
        + ‖(X t j : EucSpace d) - (Y t j : EucSpace d)‖ := by
        rw [hxj, hyi, mul_one, one_mul]
    _ ≤ cBeta β ^ ((n : ℝ) * t) * Real.sqrt (Real.log d / d)
        + cBeta β ^ ((n : ℝ) * t) * Real.sqrt (Real.log d / d) := add_le_add hi hj
    _ = 2 * cBeta β ^ ((n : ℝ) * t) * Real.sqrt (Real.log d / d) := by ring

/-- The hypotheses of `ineq_first_part` are satisfiable: at `d = n = 1` there
is no pair `i ≠ j`, so `hγ` is vacuous, and the consensus solution is at
distance `0` from itself. -/
example :
    (0 : ℝ) ≤ 0 ∧ SA 1 1 0 (fun _ _ => basePoint 0) ∧
      ‖((basePoint 0 : SSphere 1) : EucSpace 1) -
        ((basePoint 0 : SSphere 1) : EucSpace 1)‖
        ≤ Real.sqrt (Real.log 1 / 1) ∧
      ∀ t : ℝ, 0 ≤ t → ∀ i j : Idx 1, i ≠ j →
        inner (𝕜 := ℝ) ((basePoint 0 : EucSpace 1)) ((basePoint 0 : EucSpace 1))
          = (0 : ℝ) :=
  ⟨le_rfl, SA_const_consensus 1 1 one_pos 0 (basePoint 0), by simp,
    fun _ _ i j hij => absurd (Subsingleton.elim i j) hij⟩

/-- `d / log d` grows beyond every bound: for each real `K` there is a
dimension past which `K ≤ d / log d`.

The elementary route: `log d = 2 log √d ≤ 2 (√d - 1) < 2 √d`, so
`d / log d ≥ √d / 2`, which passes `K` as soon as `d ≥ 4 K²`.

Source: arXiv:2312.10794v5, Appendix D, `eq: d.large` — the growth that makes
the threshold `d⋆(n, β)` exist. -/
theorem exists_le_div_log (K : ℝ) :
    ∃ D : ℕ, ∀ d : ℕ, D ≤ d → K ≤ (d : ℝ) / Real.log d := by
  set K' : ℝ := max K 0 with hK'def
  have hKK' : K ≤ K' := le_max_left _ _
  have hK'0 : 0 ≤ K' := le_max_right _ _
  refine ⟨max 2 ⌈4 * K' ^ 2⌉₊, fun d hd => ?_⟩
  have hd2 : (2 : ℝ) ≤ (d : ℝ) := by exact_mod_cast le_trans (le_max_left _ _) hd
  have hd0 : (0 : ℝ) ≤ (d : ℝ) := by linarith
  have hdK : 4 * K' ^ 2 ≤ (d : ℝ) :=
    le_trans (Nat.le_ceil _) (by exact_mod_cast le_trans (le_max_right 2 _) hd)
  have hsq : Real.sqrt d * Real.sqrt d = (d : ℝ) := Real.mul_self_sqrt hd0
  have hsqrt0 : 0 < Real.sqrt d := Real.sqrt_pos.mpr (by linarith)
  have h2K : 2 * K' ≤ Real.sqrt d := Real.le_sqrt_of_sq_le (by nlinarith)
  have hlogpos : 0 < Real.log d := Real.log_pos (by linarith)
  have hlog : Real.log d ≤ 2 * Real.sqrt d := by
    have h := Real.log_le_sub_one_of_pos hsqrt0
    rw [Real.log_sqrt hd0] at h
    linarith
  rw [le_div_iff₀ hlogpos]
  nlinarith [mul_le_mul_of_nonneg_left hlog hK'0,
    mul_le_mul_of_nonneg_right h2K hsqrt0.le,
    mul_le_mul_of_nonneg_right hKK' hlogpos.le]

/-- **Equation (eq: d.large).** *The threshold `d⋆(n, β)`.*

There is a dimension past which

  `16 c(β)² / γ_β(1/n)² ≤ d / log d`,

which is what makes the error term `√(log d / d)` of `e:ineqfirstpart` small
compared with `γ_β(1/n)` at time `1/n`.

The survey derives the threshold from `γ_β(1/n) > 0`, but that hypothesis is
not needed: `d / log d` passes *every* real bound, and when `γ_β(1/n) = 0` the
left-hand side is `0` under Lean's division convention.  So the ODE plays no
role here and the hypothesis `ybetaODE_SA n β γ` is dropped.

Source: arXiv:2312.10794v5, Appendix D, `eq: d.large`. -/
theorem d_star_definition (β : ℝ) (γ : ℝ → ℝ) :
    ∃ d_star : ℕ, ∀ d : ℕ, d_star ≤ d →
      16 * (cBeta β)^2 / (γ ((n : ℝ)⁻¹))^2
        ≤ (d : ℝ) / Real.log d :=
  exists_le_div_log _

end Perspective
end Transformer
