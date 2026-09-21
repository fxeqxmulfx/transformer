/-
# Normalization — Initial and terminal velocities (§4.2–4.3 of 2510.22026v2)

* `Theorem thm: initial-velocity` — at a uniform initialization the attention
  vector is small, `‖A_j(0)‖ ≤ C(√(log n / n) + log n / d)` for every token at
  once with probability `1 - n^{-c}`;
* `Theorem thm: preln-slow (ii)` — from a local-cone initialization the
  intra-cluster variance decays at the per-scheme rate
  `d/dt Var(t) = -Θ(Var(t) / scale(t))`.

Part (i) of `thm: preln-slow`, the radial growth `r_k(t) ≥ (1 - δ) t`, is
proved in `Normalization.Velocities`; the bound `‖A_j‖ ≤ 1` that holds at every
configuration is there too.

Neither statement here is proved.  The first is almost-sure with respect to the
uniform measure on `(𝕊^{d-1})^{⊗ n}`, which is pinned down by
`Perspective.UniformTuple`; read over an arbitrary measure the statement is
false, and `not_forall_initial_velocity_small` proves it.  The second is
an asymptotic two-sided bound: the `Θ(·)` is spelled out as a pair of
constants `0 < c ≤ C` sandwiching the derivative, which is what `Θ` means, and
neither is proved here.
-/

import Transformer.Basic
import Transformer.Normalization.Basic
import Transformer.Normalization.Velocities
import Transformer.Perspective.Section3_SmallBeta
import Mathlib.MeasureTheory.Constructions.BorelSpace.Basic
import Mathlib.Analysis.Normed.Lp.MeasurableSpace
import Mathlib.Analysis.Complex.ExponentialBounds

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Normalization

variable (d n : ℕ)

/-- **Theorem (thm: initial-velocity).** *The attention vector is small at a
uniform initialization.*

For `β = 1` and `max{‖Q^⊤ K‖_op, ‖V‖_op} ≤ 1`, with `θ_j(0)` drawn i.i.d.
uniform on `𝕊^{d-1}` and `e^{√d} ≥ n log n ≥ d`, there are absolute constants
`c, C > 0` such that with probability `1 - n^{-c}`, simultaneously for all
`j ∈ [n]`,

  `‖A_j(0)‖ ≤ C (√(log n / n) + log n / d)`.

`‖Q^⊤ K‖_op ≤ 1` is stated on the bilinear form it is the norm of,
`|⟨Q x, K y⟩| ≤ ‖x‖ ‖y‖`, so that no adjoint has to be formed.

**What the source says and what is changed here.**  The reference measure is
`Perspective.UniformTuple d n`, the `n`-fold product of a rotation-invariant
Borel probability measure on the sphere — which is exactly the paper's "drawn
i.i.d. uniform on `𝕊^{d-1}`", and pins the measure down uniquely, so
quantifying over every such `σ` is no weaker than naming one.  Carried as a
free parameter, as it was, the statement is false: `σ = 0` makes the
right-hand side `0` while the left-hand side `1 - n^{-c}` is positive for
`c > 0`.  That is `not_forall_initial_velocity_small`.

The constants are chosen before `d` and `n`, as "absolute constants" says.
Chosen after them, as they were, the statement is empty: `‖A_j‖ ≤ 1` at every
configuration (`norm_attentionVec_le_one`), and a `C` depending on `d, n` makes
the bound hold everywhere as soon as `n ≥ 2`.  The source's probability is
`1 - n^{-C}`, with the `c` it introduces left unused; it is read as
`1 - n^{-c}`, the reading in which both constants are used.

Not proved here.

Source: arXiv:2510.22026v2, §4.2, `thm: initial-velocity`. -/
theorem initial_velocity_small :
    ∃ c C : ℝ, 0 < c ∧ 0 < C ∧
      ∀ (d n : ℕ) (σ : Measure (SphereTuple d n)), Perspective.UniformTuple d n σ →
      ∀ Q K V : ParamMatrix d,
        (∀ x y : EucSpace d, |inner (𝕜 := ℝ) (Q x) (K y)| ≤ ‖x‖ * ‖y‖) →
        ‖V‖ ≤ 1 →
        (n : ℝ) * Real.log n ≤ Real.exp (Real.sqrt d) →
        (d : ℝ) ≤ (n : ℝ) * Real.log n →
        1 - ENNReal.ofReal ((n : ℝ) ^ (-c))
          ≤ σ { Θ : SphereTuple d n | ∀ j : Idx n,
              ‖attentionVec d n 1 Q K V (tupleCoe Θ) j‖
                ≤ C * (Real.sqrt (Real.log n / n) + Real.log n / d) } := by
  sorry

/-- The hypotheses `initial_velocity_small` carries under its quantifiers are
satisfiable: `Q = K = V = I_d` meet the two operator bounds by Cauchy–Schwarz,
and `n = 2`, `d = 1` meet `n log n ≤ e^{√d}` and `d ≤ n log n`, since
`1 ≤ 2 log 2 ≈ 1.386 ≤ e`. -/
example :
    (∀ x y : EucSpace 1,
        |inner (𝕜 := ℝ) (ContinuousLinearMap.id ℝ (EucSpace 1) x)
          (ContinuousLinearMap.id ℝ (EucSpace 1) y)| ≤ ‖x‖ * ‖y‖) ∧
      ‖ContinuousLinearMap.id ℝ (EucSpace 1)‖ ≤ 1 ∧
      ((2 : ℕ) : ℝ) * Real.log ((2 : ℕ) : ℝ) ≤ Real.exp (Real.sqrt ((1 : ℕ) : ℝ)) ∧
      ((1 : ℕ) : ℝ) ≤ ((2 : ℕ) : ℝ) * Real.log ((2 : ℕ) : ℝ) := by
  refine ⟨fun x y => abs_real_inner_le_norm x y, ContinuousLinearMap.norm_id_le, ?_, ?_⟩
  · have h1 : Real.log 2 < 0.6931471808 := Real.log_two_lt_d9
    have h2 : (2.7182818283 : ℝ) < Real.exp 1 := Real.exp_one_gt_d9
    push_cast
    rw [Real.sqrt_one]
    nlinarith
  · have h1 : (0.6931471803 : ℝ) < Real.log 2 := Real.log_two_gt_d9
    push_cast
    nlinarith

/-- **`thm: initial-velocity` is false over an arbitrary reference measure.**

The theorem is a probability bound, and a probability bound says nothing
unless the measure is a probability measure: with `σ` free, `σ = 0` sends the
right-hand side to `0`, while the left-hand side `1 - n^{-c}` is positive for
every `c > 0` as soon as `n ≥ 2`.  At `d = 1`, `n = 2` the two size
conditions `n log n ≤ e^{√d}` and `d ≤ n log n` hold and `Q = K = V = I_1`
meet the operator bounds, so the hypotheses are all met and the conclusion
fails.

This is why `initial_velocity_small` reads the bound against
`Perspective.UniformTuple`, which is the paper's i.i.d. uniform law and is
unique.

Source: arXiv:2510.22026v2, §4.2, `thm: initial-velocity`. -/
theorem not_forall_initial_velocity_small :
    ¬ ∀ (d n : ℕ) (σ : Measure (SphereTuple d n)),
        ∃ c C : ℝ, 0 < c ∧ 0 < C ∧
          ∀ Q K V : ParamMatrix d,
            (∀ x y : EucSpace d, |inner (𝕜 := ℝ) (Q x) (K y)| ≤ ‖x‖ * ‖y‖) →
            ‖V‖ ≤ 1 →
            (n : ℝ) * Real.log n ≤ Real.exp (Real.sqrt d) →
            (d : ℝ) ≤ (n : ℝ) * Real.log n →
            1 - ENNReal.ofReal ((n : ℝ) ^ (-c))
              ≤ σ { Θ : SphereTuple d n | ∀ j : Idx n,
                  ‖attentionVec d n 1 Q K V (tupleCoe Θ) j‖
                    ≤ C * (Real.sqrt (Real.log n / n) + Real.log n / d) } := by
  intro h
  obtain ⟨c, C, hc, -, hkey⟩ := h 1 2 0
  have hsize : ((2 : ℕ) : ℝ) * Real.log ((2 : ℕ) : ℝ)
      ≤ Real.exp (Real.sqrt ((1 : ℕ) : ℝ)) := by
    have h1 : Real.log 2 < 0.6931471808 := Real.log_two_lt_d9
    have h2 : (2.7182818283 : ℝ) < Real.exp 1 := Real.exp_one_gt_d9
    push_cast
    rw [Real.sqrt_one]
    nlinarith
  have hdim : ((1 : ℕ) : ℝ) ≤ ((2 : ℕ) : ℝ) * Real.log ((2 : ℕ) : ℝ) := by
    have h1 : (0.6931471803 : ℝ) < Real.log 2 := Real.log_two_gt_d9
    push_cast
    nlinarith
  have hbad := hkey (ContinuousLinearMap.id ℝ (EucSpace 1))
    (ContinuousLinearMap.id ℝ (EucSpace 1)) (ContinuousLinearMap.id ℝ (EucSpace 1))
    (fun x y => abs_real_inner_le_norm x y) ContinuousLinearMap.norm_id_le
    hsize hdim
  rw [Measure.coe_zero, Pi.zero_apply, nonpos_iff_eq_zero, tsub_eq_zero_iff_le,
    ENNReal.one_le_ofReal] at hbad
  have : ((2 : ℕ) : ℝ) ^ (-c) < 1 :=
    Real.rpow_lt_one_of_one_lt_of_neg (by norm_num) (neg_neg_iff_pos.mpr hc)
  linarith

/-- The time scale by which `thm: preln-slow (ii)` divides the intra-cluster
variance, one row per scheme: `1` for Post-LN, `t` for Pre-LN, Mix-LN and
Peri-LN, the step size `α_t` for nGPT, and `√t` for CoD.

Source: arXiv:2510.22026v2, §4.3, the display of `thm: preln-slow (ii)`. -/
noncomputable def varScale (α : ℝ → ℝ) (scheme : Scheme) (t : ℝ) : ℝ :=
  match scheme with
  | .post => 1
  | .pre  => t
  | .mix  => t
  | .peri => t
  | .nGPT => α t
  | .CoD  => Real.sqrt t

/-- **Theorem (thm: preln-slow), (ii).** *Speed of cluster collapse.*

For `V = I_d` and arbitrary `Q, K` with `‖Q^⊤ K‖_op ≤ 1`, started in a local
cone `⟨θ_j(0), θ_k(0)⟩ ≥ 1 - δ` with `δ < 1 / (100 n² β²)`, the intra-cluster
variance obeys

  `d/dt Var(t) = -Θ(Var(t) / scale(t))`,

with `scale` the per-scheme time scale of `varScale`: Post-LN clusters
exponentially, CoD at `√t`, and Pre-LN, Mix-LN, Peri-LN polynomially, while
nGPT controls the rate through `α_t`.

`Θ(·)` is written out as a pair of absolute constants `0 < c ≤ C` that
sandwich the derivative from both sides for all positive times; this is the
whole content of the statement, the paper giving no explicit constants.
Part (i) is `radialDerivative_pre_ge_of_localCone`.

Not proved here.

Source: arXiv:2510.22026v2, §4.3, `thm: preln-slow` (ii). -/
theorem clustering_rate (β δ : ℝ) (α : ℝ → ℝ) (τ : ℝ) (scheme : Scheme)
    (hδ : δ < 1 / (100 * (n : ℝ) ^ 2 * β ^ 2)) :
    ∀ Q K : ℝ → ParamMatrix d,
      (∀ t : ℝ, ∀ x y : EucSpace d,
        |inner (𝕜 := ℝ) (Q t x) (K t y)| ≤ ‖x‖ * ‖y‖) →
      ∃ c C : ℝ, 0 < c ∧ c ≤ C ∧
        ∀ (θ : ℝ → Idx n → EucSpace d) (r : ℝ → Idx n → ℝ),
          (∀ j k : Idx n, 1 - δ ≤ inner (𝕜 := ℝ) (θ 0 j) (θ 0 k)) →
          SchemeDynamics d n β Q K (idParams d) α τ scheme θ r →
            ∀ t : ℝ, 0 < t → ∃ v : ℝ,
              HasDerivAt (intraClusterVar d n θ) v t ∧
              -(C * intraClusterVar d n θ t / varScale α scheme t) ≤ v ∧
              v ≤ -(c * intraClusterVar d n θ t / varScale α scheme t) := by
  sorry

/-- The hypotheses of `clustering_rate` are satisfiable: one token, `β = 1` and
`δ = 0`, since `0 < 1 / 100`; and `Q = K = I_d` meet the operator bound at every
time by Cauchy–Schwarz. -/
example :
    (0 : ℝ) < 1 / (100 * ((1 : ℕ) : ℝ) ^ 2 * (1 : ℝ) ^ 2) ∧
      ∀ _t : ℝ, ∀ x y : EucSpace 1,
        |inner (𝕜 := ℝ) (ContinuousLinearMap.id ℝ (EucSpace 1) x)
          (ContinuousLinearMap.id ℝ (EucSpace 1) y)| ≤ ‖x‖ * ‖y‖ := by
  refine ⟨by norm_num, fun _ x y => abs_real_inner_le_norm x y⟩

end Normalization
end Transformer
