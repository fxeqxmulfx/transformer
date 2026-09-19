/-
# Appendix D — the stability estimates behind `thm: phase.transition.curve`

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

The first half of Appendix D of the survey:

* `eq: stability.4ortho`  — the Grönwall stability estimate for `SA`,
* `eq: almost.ortho.vec`  — almost-orthogonality of a uniform sample (Lévy),
* `e:shortdist`           — the distance between `x_i` and its orthogonal
                            approximation `y_i`,
* `e:ineqfirstpart`       — the first half of `eq: upto-t`,
* `e:ybetacloseto1`       — the decay of `1 - γ_β(t)`,
* `eq: d.large`           — the definition of `d⋆(n, β)`.

The lower bound on the smallest coordinate `α(t)` is in
`Perspective.AppendixD_Alpha`, and what the survey builds on it in
`Perspective.AppendixD_Assembly`.
-/

import Transformer.Basic
import Transformer.Perspective.Section1_IPS
import Transformer.Perspective.Section5_HighD
import Transformer.Perspective.Section5_HighDCurve

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Perspective

variable (d n : ℕ)

/-- The Lipschitz constant `c(β) = e^{10 max(1, β)}` of the flow, as it appears
throughout Appendix D.

Source: arXiv:2312.10794v5, Appendix D, `eq: lip.3`. -/
noncomputable def cBeta (β : ℝ) : ℝ := Real.exp (10 * max 1 β)

/-- `c(β) ≥ 1`: the bound it multiplies never shrinks. -/
theorem one_le_cBeta (β : ℝ) : 1 ≤ cBeta β := by
  have : (0 : ℝ) ≤ 10 * max 1 β := by
    have : (1 : ℝ) ≤ max 1 β := le_max_left _ _
    linarith
  simp [cBeta, Real.one_le_exp this]

/-- **Equation (eq: stability.4ortho).** *Grönwall stability of the flow.*

Any two solutions of `SA` with the same `β` separate at most at the rate
`c(β)^{n t}`: if every pair of initial particles is within `M`, then

  `‖x_i(t) - y_i(t)‖ ≤ c(β)^{n t} · M`   for all `t ≥ 0` and all `i`.

The survey writes this with `max_j` on both sides; taking an arbitrary upper
bound `M` of the initial distances says the same thing and avoids carrying a
nonemptiness proof of `[n]` inside the statement.

Not proved here: the Grönwall argument rests on the Lipschitz bounds
`eq: lip.1`–`eq: lip.3` for the right-hand side of `SA`.

Source: arXiv:2312.10794v5, Appendix D, `eq: stability.4ortho`. -/
theorem stability_orthogonal
    (β : ℝ) (X Y : ℝ → SphereTuple d n) (M : ℝ)
    (hX : SA d n β X) (hY : SA d n β Y)
    (hM : ∀ j : Idx n, ‖(X 0 j : EucSpace d) - (Y 0 j : EucSpace d)‖ ≤ M) :
    ∀ t : ℝ, 0 ≤ t → ∀ i : Idx n,
      ‖(X t i : EucSpace d) - (Y t i : EucSpace d)‖ ≤ cBeta β ^ ((n : ℝ) * t) * M := by
  sorry

/-- The hypotheses of `stability_orthogonal` are satisfiable: one consensus
solution compared with itself, at initial distance `M = 0`. -/
example :
    SA 1 1 0 (fun _ _ => basePoint 0) ∧
      ‖((basePoint 0 : SSphere 1) : EucSpace 1) -
        ((basePoint 0 : SSphere 1) : EucSpace 1)‖ ≤ (0 : ℝ) :=
  ⟨SA_const_consensus 1 1 one_pos 0 (basePoint 0), by simp⟩

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

/-- **Equation (e:shortdist).**

  `‖x_i(t) - y_i(t)‖ ≤ c(β)^{n t} √(log d / d)`,

where `y` is the solution started from the orthogonal approximation supplied by
`eq: almost.ortho.vec`.  It is `eq: stability.4ortho` at `M = √(log d / d)`.

`eq: stability.4ortho` is `stability_orthogonal`, which is not proved, so it is
carried here as a hypothesis rather than used: what this theorem asserts is the
specialization, and that is what it proves.

Source: arXiv:2312.10794v5, Appendix D, `e:shortdist`. -/
theorem shortdist_bound
    (β : ℝ) (X Y : ℝ → SphereTuple d n)
    (hX : SA d n β X) (hY : SA d n β Y)
    (hM : ∀ j : Idx n,
      ‖(X 0 j : EucSpace d) - (Y 0 j : EucSpace d)‖ ≤ Real.sqrt (Real.log d / d)) :
    (∀ M : ℝ, SA d n β X → SA d n β Y →
        (∀ j : Idx n, ‖(X 0 j : EucSpace d) - (Y 0 j : EucSpace d)‖ ≤ M) →
        ∀ t : ℝ, 0 ≤ t → ∀ i : Idx n,
          ‖(X t i : EucSpace d) - (Y t i : EucSpace d)‖
            ≤ cBeta β ^ ((n : ℝ) * t) * M) →
    ∀ t : ℝ, 0 ≤ t → ∀ i : Idx n,
      ‖(X t i : EucSpace d) - (Y t i : EucSpace d)‖
        ≤ cBeta β ^ ((n : ℝ) * t) * Real.sqrt (Real.log d / d) :=
  fun hstab => hstab _ hX hY hM

/-- The hypotheses of `shortdist_bound` are satisfiable: at `d = 1` the bound
`√(log d / d)` is `0`, and a consensus solution is at distance `0` from
itself. -/
example :
    SA 1 1 0 (fun _ _ => basePoint 0) ∧
      ‖((basePoint 0 : SSphere 1) : EucSpace 1) -
        ((basePoint 0 : SSphere 1) : EucSpace 1)‖
        ≤ Real.sqrt (Real.log 1 / 1) :=
  ⟨SA_const_consensus 1 1 one_pos 0 (basePoint 0), by simp⟩

/-- **Equation (e:ineqfirstpart).** *First half of `eq: upto-t`.*

If `y` is a solution whose particles stay pairwise at the common angle `γ_β`
(that is `⟨y_i(t), y_j(t)⟩ = γ_β(t)` for `i ≠ j`, which is `thm: orthogonal`),
and `x` starts within `√(log d / d)` of it, then

  `|⟨x_i(t), x_j(t)⟩ - γ_β(t)| ≤ 2 c(β)^{n t} √(log d / d)`.

The proof is `e:shortdist` followed by Cauchy–Schwarz on
`⟨x_i, x_j⟩ - ⟨y_i, y_j⟩ = ⟨x_i - y_i, x_j⟩ + ⟨y_i, x_j - y_j⟩`, both particles
being unit vectors.  Only the Cauchy–Schwarz half is proved here;
`eq: stability.4ortho`, which `e:shortdist` specializes, is a hypothesis, as in
`shortdist_bound`.

Source: arXiv:2312.10794v5, Appendix D, `e:ineqfirstpart`. -/
theorem ineq_first_part
    (β : ℝ) (X Y : ℝ → SphereTuple d n) (γ : ℝ → ℝ)
    (hX : SA d n β X) (hY : SA d n β Y)
    (hM : ∀ j : Idx n,
      ‖(X 0 j : EucSpace d) - (Y 0 j : EucSpace d)‖ ≤ Real.sqrt (Real.log d / d))
    (hγ : ∀ t : ℝ, 0 ≤ t → ∀ i j : Idx n, i ≠ j →
      inner (𝕜 := ℝ) ((Y t i : EucSpace d)) ((Y t j : EucSpace d)) = γ t) :
    (∀ M : ℝ, SA d n β X → SA d n β Y →
        (∀ j : Idx n, ‖(X 0 j : EucSpace d) - (Y 0 j : EucSpace d)‖ ≤ M) →
        ∀ t : ℝ, 0 ≤ t → ∀ i : Idx n,
          ‖(X t i : EucSpace d) - (Y t i : EucSpace d)‖
            ≤ cBeta β ^ ((n : ℝ) * t) * M) →
    ∀ t : ℝ, 0 ≤ t → ∀ i j : Idx n, i ≠ j →
      |inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((X t j : EucSpace d)) - γ t|
        ≤ 2 * cBeta β ^ ((n : ℝ) * t) * Real.sqrt (Real.log d / d) := by
  intro hstab t ht i j hij
  have hi := shortdist_bound d n β X Y hX hY hM hstab t ht i
  have hj := shortdist_bound d n β X Y hX hY hM hstab t ht j
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
    SA 1 1 0 (fun _ _ => basePoint 0) ∧
      ‖((basePoint 0 : SSphere 1) : EucSpace 1) -
        ((basePoint 0 : SSphere 1) : EucSpace 1)‖
        ≤ Real.sqrt (Real.log 1 / 1) ∧
      ∀ t : ℝ, 0 ≤ t → ∀ i j : Idx 1, i ≠ j →
        inner (𝕜 := ℝ) ((basePoint 0 : EucSpace 1)) ((basePoint 0 : EucSpace 1))
          = (0 : ℝ) :=
  ⟨SA_const_consensus 1 1 one_pos 0 (basePoint 0), by simp,
    fun _ _ i j hij => absurd (Subsingleton.elim i j) hij⟩

/-- **Equation (e:ybetacloseto1).**

  `1 - γ_β(t) ≤ (1/2) exp( n² e^β / (2(n + e^{β/2})) - n t / (n + e^{β/2}) )`.

Not proved here: it is the Grönwall estimate for `eq: ybeta` itself, using
`e^{βγ} ≤ e^β` and `(n-1)γ + 1 ≥ 1` on `[0, 1]`.

Source: arXiv:2312.10794v5, Appendix D, `e:ybetacloseto1`. -/
theorem ybeta_close_to_1
    (β : ℝ) (γ : ℝ → ℝ) (hγ : ybetaODE_SA n β γ) :
    ∀ t : ℝ, 0 ≤ t →
      1 - γ t
        ≤ (1/2 : ℝ) * Real.exp
            ((n : ℝ)^2 * Real.exp β
                / (2 * ((n : ℝ) + Real.exp (β / 2)))
              - ((n : ℝ) * t) / ((n : ℝ) + Real.exp (β / 2))) := by
  sorry

/-- The hypothesis `ybetaODE_SA n β γ` of `ybeta_close_to_1` is satisfiable:
`ybetaODE_SA_one_zero`. -/
example : ybetaODE_SA 1 0 (fun t => 1 - Real.exp (-2 * t)) := ybetaODE_SA_one_zero

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
