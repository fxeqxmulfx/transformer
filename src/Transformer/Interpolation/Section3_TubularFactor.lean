/-
# The retained-fraction correction for the tubular mass lemma

ArXiv:2411.04551v3, Appendix, `lem: tubular.mass.movement`.

The source allows every positive error and prints the real power
`(1 - ε)^K`. For error three and two steps this is four. The explicit
three-ball chain below satisfies all source geometry hypotheses, and its
middle ball covers the sphere. Initial unit mass therefore lies in the
union, whereas no terminal probability can put four units in the last
ball. This refutes that factor even after the initial law is chosen first.

The corrected fraction is `ofReal (1 - ε)` before its power. It agrees
with the printed factor for errors at most one, lies between zero and
one for nonnegative errors, and vanishes for errors at least one when
there is a positive number of steps.
With zero steps the power is one, as required by the identity base case.

This correction is separate from the source's quantifier-order failure
proved in `BallTransportFalse`: the counterexample here applies even to
controls allowed to depend on the initial probability.
-/

import Transformer.Interpolation.BallTransportFalse
import Transformer.Interpolation.Section3_NullRepeller

open Real MeasureTheory Set
open scoped ENNReal

namespace Transformer.Interpolation

variable {d : ℕ}

/-- On the source's meaningful error range, clamping before or after
the power gives the same fraction. Auxiliary for arXiv:2411.04551v3,
Appendix, `lem: tubular.mass.movement`. -/
theorem tubular_retention_of_error_le_one (ε : ℝ) (hε : ε ≤ 1) (K : ℕ) :
    ENNReal.ofReal ((1 - ε) ^ K) = ENNReal.ofReal (1 - ε) ^ K :=
  ENNReal.ofReal_pow (by linarith) K

/-- A positive-step chain with error at least one has corrected retained
fraction zero. Auxiliary for arXiv:2411.04551v3, Appendix,
`lem: tubular.mass.movement`, the corrected error range. -/
theorem tubular_retention_of_one_le_error (ε : ℝ) (hε : 1 ≤ ε) (K : ℕ) (hK : 0 < K) :
    ENNReal.ofReal (1 - ε) ^ K = 0 := by
  rw [ENNReal.ofReal_eq_zero.mpr (by linarith), zero_pow (by omega : K ≠ 0)]

/-- The corrected retained power never exceeds total mass for a
nonnegative error. Auxiliary for arXiv:2411.04551v3, Appendix,
`lem: tubular.mass.movement`, its corrected mass factor. -/
theorem tubular_retention_le_one (ε : ℝ) (hε : 0 ≤ ε) (K : ℕ) :
    ENNReal.ofReal (1 - ε) ^ K ≤ 1 := by
  exact pow_le_one₀ (by positivity) (ENNReal.ofReal_le_one.mpr (by linarith))

/-- The printed and corrected factors differ at the explicit source
counterexample. Source: arXiv:2411.04551v3, Appendix,
`lem: tubular.mass.movement`, with error three and two steps. -/
theorem tubular_retention_at_three :
    ENNReal.ofReal (((1 : ℝ) - 3) ^ (2 : ℕ)) = 4 ∧
      ENNReal.ofReal ((1 : ℝ) - 3) ^ (2 : ℕ) = 0 := by
  norm_num

/-- Increasing the requested error decreases the retained mass fraction.
Auxiliary for arXiv:2411.04551v3, Appendix,
`lem: tubular.mass.movement`, the corrected error dependence. -/
theorem tubular_retention_antitone_error (ε η : ℝ) (hεη : ε ≤ η) (K : ℕ) :
    ENNReal.ofReal (1 - η) ^ K ≤ ENNReal.ofReal (1 - ε) ^ K := by
  apply pow_le_pow_left₀ (by positivity)
  exact ENNReal.ofReal_le_ofReal (by linarith)

/-- A chain with no transport step retains its entire original ball mass.
Auxiliary for arXiv:2411.04551v3, Appendix, the base case of
`lem: tubular.mass.movement`. -/
theorem tubular_retention_zero_steps (ε : ℝ) :
    ENNReal.ofReal (1 - ε) ^ (0 : ℕ) = 1 := pow_zero _

/-- Two antipodal half-radius balls are disjoint on the unit sphere.
Auxiliary for the counterexample to arXiv:2411.04551v3, Appendix,
`lem: tubular.mass.movement` with unrestricted error. -/
theorem disjoint_antipodal_half_balls (p : SSphere d) :
    Disjoint (Metric.ball p (1 / 2 : ℝ)) (Metric.ball (antipode d p) (1 / 2 : ℝ)) := by
  apply disjoint_left.mpr
  intro x hx hy
  have htri := dist_triangle (antipode d p) x p
  rw [dist_antipode, dist_comm (antipode d p) x] at htri
  have hx' : dist x p < 1 / 2 := hx
  have hy' : dist x (antipode d p) < 1 / 2 := hy
  linarith

/-- The source's factor in `lem: tubular.mass.movement` is false for
`ε = 3`, `K = 2`, even with initial-data-dependent controls. The explicit
ball chain has both adjacent overlaps and disjoint end balls; every
terminal probability violates the printed bound. Source:
arXiv:2411.04551v3, Appendix, `lem: tubular.mass.movement`. -/
theorem exists_tubular_unclamped_retention_counterexample :
    ∃ (z : ℕ → SSphere 2) (R : ℕ → ℝ) (ν₀ : Perspective.ProbSphere 2),
      (∀ k : ℕ, 1 ≤ k → k ≤ 2 →
        (Metric.ball (z k) (R k) ∩ Metric.ball (z (k - 1)) (R (k - 1))).Nonempty) ∧
      (∀ k k' : ℕ, k ≤ 2 → k' ≤ 2 → k + 2 ≤ k' →
        Disjoint (Metric.ball (z k) (R k)) (Metric.ball (z k') (R k'))) ∧
      (0 : ℝ) < 3 ∧
      ∀ ν : Perspective.ProbSphere 2,
        ¬ ENNReal.ofReal (((1 : ℝ) - 3) ^ (2 : ℕ)) *
          (ν₀ : Measure (SSphere 2)) (⋃ k ∈ Iic 2, Metric.ball (z k) (R k)) ≤
          (ν : Measure (SSphere 2)) (Metric.ball (z 2) (R 2)) := by
  let p : SSphere 2 := basePoint 1
  let z : ℕ → SSphere 2 := fun k => if k = 2 then antipode 2 p else p
  let R : ℕ → ℝ := fun k => if k = 1 then 3 else 1 / 2
  have h1 : Metric.ball (z 1) (R 1) = univ := by
    exact spherical_ball_eq_univ_of_two_lt p 3 (by norm_num)
  have hu : (⋃ k ∈ Iic 2, Metric.ball (z k) (R k)) = univ := by
    apply eq_univ_of_forall
    intro x
    exact mem_iUnion.mpr ⟨1, mem_iUnion.mpr ⟨by norm_num, by rw [h1]; exact mem_univ x⟩⟩
  refine ⟨z, R, Perspective.diracProb 2 p, ?_, ?_, by norm_num, ?_⟩
  · intro k hk hk2
    have he : k = 1 ∨ k = 2 := by omega
    rcases he with rfl | rfl
    · refine ⟨p, ?_, ?_⟩
      · rw [h1]
        exact mem_univ p
      · simp [z, R]
    · refine ⟨antipode 2 p, ?_, ?_⟩
      · simp [z, R]
      · rw [show 2 - 1 = (1 : ℕ) by norm_num, h1]
        exact mem_univ _
  · intro k k' hk hk' hkk'
    have hk0 : k = 0 := by omega
    have hk2 : k' = 2 := by omega
    subst k
    subst k'
    exact disjoint_antipodal_half_balls p
  · intro ν hbound
    rw [hu, measure_univ, mul_one] at hbound
    have hmass : (ν : Measure (SSphere 2)) (Metric.ball (z 2) (R 2)) ≤ 1 := by
      calc
        _ ≤ (ν : Measure (SSphere 2)) univ := measure_mono (subset_univ _)
        _ = 1 := measure_univ
    have hbad := hbound.trans hmass
    norm_num at hbad

/-- Positive errors on both sides of one, a positive chain length and a
genuine unit-sphere point realize every comparison and geometry premise. -/
example : (1 / 2 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 / 2 ∧ (1 : ℝ) ≤ 3 ∧
    (1 / 2 : ℝ) ≤ 3 ∧
    0 < (2 : ℕ) ∧ ∃ p : SSphere 2, ‖(p : EucSpace 2)‖ = 1 := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, basePoint 1, ?_⟩
  exact mem_sphere_zero_iff_norm.mp (basePoint 1).2

end Transformer.Interpolation
