module

public import Transformer.Clusters.Section6_LagrangianField
public import Transformer.Clusters.Section6_Stability
public import Transformer.Landscape.GlobalFlow.Existence

/-!
# The emergence of clusters in self-attention dynamics — the continuity
  equation

`c:wellposedtransformers` of arXiv:2305.05465v6, §6: global well-posedness of
`e:conteq`, the mean-field form of the dynamics, with a `W_2` stability
estimate.

**What the source says and what is carried here.**

* `e:conteq` and its solutions (`d:solconti`) are `IsContEqSolution`, in
  `Section6_WeakSolution`, where the way they are carried is explained.

* `c:wellposedtransformers` is split in two: existence and uniqueness, and the
  stability estimate `e:w2estimate`, which quantifies over pairs of solutions.
  The source asks `T > 0` in `e:w2estimate`; the estimate holds for every `T`.

**The proof.**  Existence takes another route than the source's frozen-field
scheme and Arzelà–Ascoli. The solution is `μ_t = (Y_t)_# μ₀` for the solution
`Y` of the Lagrangian ODE `Ẏ_t = F(Y_t)` of `Section6_LagrangianField`,
started from the retraction of `ℝ^d` onto `B̄(0, R)`, which is the identity on
the support of `μ₀`. By Grönwall, `‖Y_t‖ ≤ R e^{‖V‖ |t|}`, which bounds the
supports; each trajectory `t ↦ Y_t x` solves `ẏ = 𝒳[μ_t](y)`, so `μ` satisfies
the weak formulation of `d:solconti` (`ContinuityEq.weak_of_map`).
Uniqueness is `IsContEqSolution.unique`. For `e:w2estimate`, uniqueness puts
every solution from a measure carried by `B̄(0, R)` in `B̄(0, R e^{‖V‖ T})` on
`[0, T]`, where `exists_W2_le` applies with constants depending only on `R`
and `T`.

Source: arXiv:2305.05465v6, `e:conteq`, `d:solconti`,
`c:wellposedtransformers`, `e:w2estimate`.
-/

@[expose] public section

open Set Metric Filter Topology MeasureTheory
open scoped BoundedContinuousFunction

namespace Transformer
namespace Clusters

open Landscape.GlobalFlow

variable {d : ℕ}

/-- **Existence (c:wellposedtransformers)**, with the growth of the support:
from a probability measure `μ₀` carried by `B̄(0, R)`, `e:conteq` has a
solution with `μ_t` carried by `B̄(0, R e^{‖V‖ |t|})` for every `t`. -/
theorem exists_isContEqSolution (Q K V : ParamMatrix d) (μ₀ : Measure (EucSpace d))
    [IsProbabilityMeasure μ₀] {R : ℝ} (hR : 0 < R) (hsupp : IsCarriedBy μ₀ R) :
    ∃ μ : ℝ → Measure (EucSpace d), IsContEqSolution Q K V μ₀ μ ∧
      ∀ t, IsCarriedBy (μ t) (R * Real.exp (‖V‖ * |t|)) := by
  set Y₀ : EucSpace d →ᵇ EucSpace d := BoundedContinuousFunction.ofNormedAddCommGroup
    (ballRetract R) (lipschitzWith_ballRetract hR).continuous R (norm_ballRetract_le hR)
  obtain ⟨γ, hγ0, hγ⟩ := exists_global_of_lipschitzOnWith
    (lipschitzOnWith_lagrangianField Q K V μ₀) (norm_nonneg V) (norm_lagrangianField_le Q K V μ₀) Y₀
  have hcar : ∀ t, IsCarriedBy (μ₀.map (γ t)) (R * Real.exp (‖V‖ * |t|)) := fun t => by
    refine isCarriedBy_map μ₀ (γ t) ?_
    have := norm_le_of_linearGrowth (norm_nonneg V) (norm_lagrangianField_le Q K V μ₀)
      (T := |t|) (fun s _ => (hγ s).hasDerivWithinAt) t ⟨neg_abs_le t, le_abs_self t⟩
    rw [hγ0] at this
    exact this.trans (mul_le_mul_of_nonneg_right
      (BoundedContinuousFunction.norm_ofNormedAddCommGroup_le _ hR.le _) (Real.exp_pos _).le)
  have hloc : ∀ t₀, ∀ᶠ t in 𝓝 t₀,
      IsCarriedBy (μ₀.map (γ t)) (R * Real.exp (‖V‖ * (|t₀| + 1))) := fun t₀ => by
    filter_upwards [ball_mem_nhds t₀ one_pos] with t ht
    refine (hcar t).mono ?_
    have : |t| ≤ |t₀| + 1 := by
      have := abs_sub_abs_le_abs_sub t t₀
      rw [mem_ball, Real.dist_eq] at ht
      linarith
    gcongr
  -- each trajectory follows the field of the measure it transports
  have hY : ∀ t x, HasDerivAt (fun t => γ t x)
      (attentionKernel Q K V (μ₀.map (γ t)) (γ t x)) t := fun t x => by
    have := ((BoundedContinuousFunction.evalCLM ℝ (β := EucSpace d) x).hasFDerivAt
      (x := γ t)).comp_hasDerivAt t (hγ t)
    simpa [Function.comp_def, lagrangianField_apply] using this
  have hcont : ∀ g : EucSpace d → ℝ, Continuous g → HasCompactSupport g →
      Continuous fun t => ∫ x, g x ∂μ₀.map (γ t) := fun g hg hgc => by
    obtain ⟨C₀, hC₀⟩ := hg.bounded_above_of_compact_support hgc
    have e : (fun t => ∫ x, g x ∂μ₀.map (γ t)) = fun t => ∫ x, g (γ t x) ∂μ₀ := funext fun t =>
      integral_map (γ t).continuous.aemeasurable hg.aestronglyMeasurable
    rw [e]
    exact continuous_of_dominated (bound := fun _ => C₀)
      (fun t => (hg.comp (γ t).continuous).aestronglyMeasurable)
      (fun t => Eventually.of_forall fun x => hC₀ _) (integrable_const _)
      (Eventually.of_forall fun x =>
        hg.comp (continuous_iff_continuousAt.2 fun t => (hY t x).continuousAt))
  have h0 : μ₀.map (γ 0) = μ₀ := by
    rw [hγ0]
    conv_rhs => rw [← Measure.map_id (μ := μ₀)]
    refine Measure.map_congr ?_
    filter_upwards [ae_mem_closedBall hsupp] with x hx
    exact ballRetract_of_norm_le hR (mem_closedBall_zero_iff.1 hx)
  refine ⟨fun t => μ₀.map (γ t), ⟨fun t => inferInstance, fun a b => ?_,
    fun g hg hgc => hcont g hg.continuous hgc, h0, fun g hg hgc t => ?_⟩, hcar⟩
  · refine ⟨R * Real.exp (‖V‖ * max |a| |b|), by positivity, fun t ht => (hcar t).mono ?_⟩
    have := abs_le_max_abs_abs ht.1 ht.2
    gcongr
  · have h := ContinuityEq.weak_of_map (w := fun t x => attentionKernel Q K V (μ₀.map (γ t)) x)
      (continuous_attentionKernel_curve Q K V (fun t₀ => ⟨_, hloc t₀⟩)
        fun g hg hgc => hcont g hg.continuous hgc)
      (fun t₀ => ⟨‖V‖ * (R * Real.exp (‖V‖ * (|t₀| + 1))), (hloc t₀).mono fun t ht x =>
        attentionKernel_norm_le Q K V _ (by positivity) _ inferInstance ht x⟩)
      (Y := fun t => γ t) (fun t => (γ t).continuous) hY μ₀
      (hg.of_le (by exact_mod_cast le_top)) hgc t
    rw [h0] at h
    simpa only [inner_gradient] using h

/-- **Proposition (c:wellposedtransformers), existence and uniqueness.**  Every
compactly supported probability measure is the initial datum of exactly one
solution of `e:conteq`: `exists_isContEqSolution` and
`IsContEqSolution.unique`.

Source: arXiv:2305.05465v6, `c:wellposedtransformers`. -/
theorem wellposed_contEq (Q K V : ParamMatrix d) (μ₀ : Measure (EucSpace d))
    (hμ₀ : IsProbabilityMeasure μ₀) (R : ℝ) (hR : 0 < R) (hsupp : IsCarriedBy μ₀ R) :
    ∃! μ : ℝ → Measure (EucSpace d), IsContEqSolution Q K V μ₀ μ := by
  obtain ⟨μ, hμ, -⟩ := exists_isContEqSolution Q K V μ₀ hR hsupp
  exact ⟨μ, hμ, fun ν hν => hν.unique hμ⟩

/-- The hypotheses of `exists_isContEqSolution` and `wellposed_contEq` are
satisfiable, and the solution they assert exists in that instance: the Dirac
mass at the origin. -/
example (Q K V : ParamMatrix d) :
    IsProbabilityMeasure (Measure.dirac (0 : EucSpace d)) ∧ (0 : ℝ) < 1 ∧
      IsCarriedBy (Measure.dirac (0 : EucSpace d)) 1 ∧
      IsContEqSolution Q K V (Measure.dirac (0 : EucSpace d))
        (fun _ => Measure.dirac (0 : EucSpace d)) :=
  ⟨inferInstance, one_pos, isCarriedBy_dirac 1 zero_le_one, isContEqSolution_dirac Q K V⟩

/-- **Estimate (e:w2estimate).**  Solutions of `e:conteq` starting in a fixed
ball depend on their initial datum with an exponential `W_2` modulus on every
bounded time interval.

The source states it for `R > 0` and `T > 0`. The hypothesis `T > 0` is
dropped: the estimate holds for every `T`, trivially for `T ≤ 0`, where at
most `t = 0` is left.

Every solution from a measure carried by `B̄(0, R)` is the one of
`exists_isContEqSolution` (`IsContEqSolution.unique`), so it is carried by
`B̄(0, R e^{‖V‖ T})` on `[0, T]`; `exists_W2_le` on that ball gives `C`.

Source: arXiv:2305.05465v6, `c:wellposedtransformers`, `e:w2estimate`. -/
theorem contEq_w2_stability (Q K V : ParamMatrix d) (R T : ℝ) (hR : 0 < R) :
    ∃ C : ℝ, 0 < C ∧ ∀ μ₀ ν₀ : Measure (EucSpace d), IsProbabilityMeasure μ₀ →
      IsProbabilityMeasure ν₀ → IsCarriedBy μ₀ R → IsCarriedBy ν₀ R →
        ∀ μ ν : ℝ → Measure (EucSpace d), IsContEqSolution Q K V μ₀ μ →
          IsContEqSolution Q K V ν₀ ν → ∀ t ∈ Set.Icc (0 : ℝ) T,
            Wasserstein.W2 (μ t) (ν t) ≤ Real.exp (C * t) * Wasserstein.W2 μ₀ ν₀ := by
  obtain ⟨C, hC, hW⟩ := exists_W2_le Q K V (R := R * Real.exp (‖V‖ * T)) (by positivity) T
  have hcar : ∀ μ₀ : Measure (EucSpace d), IsProbabilityMeasure μ₀ → IsCarriedBy μ₀ R →
      ∀ μ, IsContEqSolution Q K V μ₀ μ →
        ∀ t ∈ Icc 0 T, IsCarriedBy (μ t) (R * Real.exp (‖V‖ * T)) := by
    intro μ₀ hμ₀ hμR μ hμ t ht
    obtain ⟨μ', hμ', hμ'R⟩ := exists_isContEqSolution Q K V μ₀ hR hμR
    rw [hμ.unique hμ']
    refine (hμ'R t).mono ?_
    rw [abs_of_nonneg ht.1]
    gcongr
    exact ht.2
  exact ⟨C, hC, fun μ₀ ν₀ hμ₀ hν₀ hμR hνR μ ν hμ hν =>
    hW μ₀ ν₀ μ ν hμ hν (hcar μ₀ hμ₀ hμR μ hμ) (hcar ν₀ hν₀ hνR ν hν)⟩

/-- The hypotheses of `contEq_w2_stability` are satisfiable: the Dirac mass at
the origin, against itself. -/
example (Q K V : ParamMatrix d) :
    (0 : ℝ) < 1 ∧ IsProbabilityMeasure (Measure.dirac (0 : EucSpace d)) ∧
      IsCarriedBy (Measure.dirac (0 : EucSpace d)) 1 ∧
      IsContEqSolution Q K V (Measure.dirac (0 : EucSpace d))
        (fun _ => Measure.dirac (0 : EucSpace d)) :=
  ⟨one_pos, inferInstance, isCarriedBy_dirac 1 zero_le_one, isContEqSolution_dirac Q K V⟩

end Clusters
end Transformer
