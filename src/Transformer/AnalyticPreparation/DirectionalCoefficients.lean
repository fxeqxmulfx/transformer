/-
# Real analytic preparation: DirectionalCoefficients

Adapted from Bochao Kong's classical-complex-wpt, revision
b4a7273fe5c9752753c52e10494097569089642d:
https://github.com/BochaoKong/classical-complex-wpt
Only the existence proof and its coefficient infrastructure are copied.
The scalar field is real; no complex root or complex differentiation result
is used. Apache-2.0 license: third_party/classical-complex-wpt/LICENSE.
-/

import Transformer.AnalyticPreparation.SequenceDivision
import Mathlib.Analysis.Analytic.ChangeOrigin
import Mathlib.Analysis.Analytic.Uniqueness

open Filter Finset
open scoped BigOperators ENNReal NNReal Topology

noncomputable section
set_option maxHeartbeats 5000000

namespace Transformer.AnalyticPreparation

variable {A I J : Type*} [AddCommMonoid A] [Finset.HasAntidiagonal A]

/-- Continuous-linear inclusion of the distinguished real axis. -/
noncomputable def lastAxis (n : ℕ) : ℝ →L[ℝ] Ambient n :=
  ContinuousLinearMap.inr ℝ (Base n) ℝ

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp]
theorem lastAxis_apply (n : ℕ) (w : ℝ) : lastAxis n w = (0, w) := rfl

/-- The unit vector in the distinguished real direction. -/
def lastDirection (n : ℕ) : Ambient n := (0, 1)

/-- The `k`-th (factorial-normalized) distinguished-variable Taylor coefficient at `(z, 0)`. -/
noncomputable def lastTaylorCoefficient {n : ℕ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (k : ℕ) (z : Base n) : ℝ :=
  p.changeOrigin (z, 0) k (fun _ ↦ lastDirection n)

/-- At the base origin, changing origin leaves the diagonal distinguished coefficient unchanged.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem lastTaylorCoefficient_zero {n k : ℕ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) :
    lastTaylorCoefficient p k 0 = p k (fun _ ↦ lastDirection n) := by
  classical
  unfold lastTaylorCoefficient FormalMultilinearSeries.changeOrigin
  rw [FormalMultilinearSeries.sum, tsum_eq_single 0]
  · simp only [FormalMultilinearSeries.changeOriginSeries]
    rw [Finset.sum_eq_single ⟨∅, by simp⟩]
    · exact FormalMultilinearSeries.changeOriginSeriesTerm_apply p k 0 ∅ (by simp)
        (0 : Ambient n) (lastDirection n)
    · intro b _ hb
      exact (hb (Subtype.ext (Finset.card_eq_zero.mp b.property))).elim
    · simp
  · intro b hb
    cases b with
    | zero => exact (hb rfl).elim
    | succ b =>
      apply ContinuousMultilinearMap.map_coord_zero _ (0 : Fin (b + 1))
      rfl

/--
The moving coefficient at the base origin is the usual factorial-normalized
iterated derivative of the distinguished slice.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem factorial_smul_lastTaylorCoefficient_zero {n k : ℕ} {f : Ambient n → ℝ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ)
    (hp : HasFPowerSeriesAt f p 0) :
    k.factorial • lastTaylorCoefficient p k 0 =
      iteratedDeriv k (lastSlice f) 0 := by
  have hpa0 : HasFPowerSeriesAt (fun w : ℝ ↦ f ((0 : Base n), w))
      (p.compContinuousLinearMap (lastAxis n)) 0 := by
    simpa [lastAxis, Function.comp_def] using
      hp.compContinuousLinearMap (u := lastAxis n) (x := (0 : ℝ))
  have hpa : HasFPowerSeriesAt (lastSlice f)
      (p.compContinuousLinearMap (lastAxis n)) 0 := by
    change HasFPowerSeriesAt (fun w : ℝ ↦ f ((0 : Base n), w))
      (p.compContinuousLinearMap (lastAxis n)) 0
    exact hpa0
  have hc := hpa.analyticAt.hasFPowerSeriesAt
  have heq := hpa.eq_formalMultilinearSeries hc
  have hcoeff := congrArg
    (fun q : FormalMultilinearSeries ℝ ℝ ℝ ↦ q k (fun _ ↦ (1 : ℝ))) heq
  have hcoeff' : p k (fun _ ↦ lastDirection n) =
      iteratedDeriv k (lastSlice f) 0 / k.factorial := by
    simpa [lastAxis, lastDirection, FormalMultilinearSeries.compContinuousLinearMap,
      FormalMultilinearSeries.ofScalars] using hcoeff
  rw [lastTaylorCoefficient_zero, hcoeff', nsmul_eq_mul]
  have hfact : (k.factorial : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr k.factorial_ne_zero
  exact mul_div_cancel₀ _ hfact

/-- Vanishing of a directional Taylor coefficient is equivalent to vanishing of the derivative.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem lastTaylorCoefficient_zero_iff_iteratedDeriv_zero {n k : ℕ} {f : Ambient n → ℝ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ)
    (hp : HasFPowerSeriesAt f p 0) :
    lastTaylorCoefficient p k 0 = 0 ↔ iteratedDeriv k (lastSlice f) 0 = 0 := by
  rw [← factorial_smul_lastTaylorCoefficient_zero p hp]
  simp [nsmul_eq_mul, Nat.factorial_ne_zero]

/-- The public exact-order condition is exactly the first-nonzero-coefficient condition.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem exactOrderInLastVariable_iff_lastTaylorCoefficients {n d : ℕ} {f : Ambient n → ℝ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ)
    (hp : HasFPowerSeriesAt f p 0) :
    ExactOrderInLastVariable f d ↔
      (∀ k < d, lastTaylorCoefficient p k 0 = 0) ∧
        lastTaylorCoefficient p d 0 ≠ 0 := by
  constructor
  · rintro ⟨hlow, htop⟩
    refine ⟨fun k hk ↦
      (lastTaylorCoefficient_zero_iff_iteratedDeriv_zero p hp).mpr (hlow k hk), ?_⟩
    intro hzero
    exact htop ((lastTaylorCoefficient_zero_iff_iteratedDeriv_zero p hp).mp hzero)
  · rintro ⟨hlow, htop⟩
    refine ⟨fun k hk ↦
      (lastTaylorCoefficient_zero_iff_iteratedDeriv_zero p hp).mp (hlow k hk), ?_⟩
    intro hzero
    exact htop ((lastTaylorCoefficient_zero_iff_iteratedDeriv_zero p hp).mpr hzero)

end Transformer.AnalyticPreparation
