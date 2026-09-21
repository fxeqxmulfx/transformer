/-
# The 2-Wasserstein distance — a lower bound

`Transformer.Wasserstein` bounds `W_2` from above by exhibiting a coupling.
The bound the other way is the one a refutation needs, and the one Markov-type
arguments run on: if `μ` puts mass `m` on a set `A` lying at distance at least
`δ` from where `ν` lives, then every coupling has to carry that mass across
the gap, and

  `m δ² ≤ W_2(μ, ν)²`.

As for `norm_integral_sub_le_W2`, both measures are carried by a bounded set,
which is what makes every transport cost the integral of a bounded function.

Sources: arXiv:2411.04551v3, `cl: W.to.ball`, `lem: mass.concentration.Q1`.
-/

import Transformer.Wasserstein.Basic

open scoped ENNReal
open Real MeasureTheory

namespace Transformer
namespace Wasserstein

variable {X : Type*} [MeasurableSpace X] [PseudoMetricSpace X]
  [OpensMeasurableSpace X] [SecondCountableTopology X]

/-- **Mass across a gap bounds `W_2` from below.**  If `ν` lives on `B`, and
every point of `A` is at distance at least `δ` from every point of `B`, then
`μ(A) δ² ≤ W_2(μ, ν)²`. -/
theorem measureReal_mul_sq_le_W2_sq (μ ν : Measure X) [IsProbabilityMeasure μ]
    [IsProbabilityMeasure ν] {s : Set X} (hs : MeasurableSet s)
    (hsb : Bornology.IsBounded s) (hμ : μ sᶜ = 0) (hν : ν sᶜ = 0)
    {A B : Set X} (hA : MeasurableSet A) (hB : MeasurableSet B) (hνB : ν Bᶜ = 0)
    {δ : ℝ} (hAB : ∀ a ∈ A, ∀ b ∈ B, δ ≤ dist a b) (hδ : 0 ≤ δ) :
    μ.real A * δ ^ 2 ≤ W2 μ ν ^ 2 := by
  have hc : ∀ c ∈ transportCosts μ ν, μ.real A * δ ^ 2 ≤ c := by
    rintro c ⟨γ, hγ, rfl⟩
    have hγ1 : IsProbabilityMeasure γ := ⟨by
      have := Measure.map_apply measurable_fst MeasurableSet.univ (μ := γ)
      rw [hγ.1] at this; simpa using this.symm⟩
    have hae : ∀ᵐ p ∂γ, p.1 ∈ s ∧ p.2 ∈ s :=
      (ae_fst_mem hγ hs hμ).and (ae_snd_mem hγ hs hν)
    have hd2 : Integrable (fun p : X × X => dist p.1 p.2 ^ 2) γ := by
      refine Integrable.of_bound (by fun_prop) (Metric.diam s ^ 2) ?_
      filter_upwards [hae] with p hp
      rw [Real.norm_of_nonneg (by positivity)]
      exact pow_le_pow_left₀ dist_nonneg (Metric.dist_le_diam_of_mem hsb hp.1 hp.2) 2
    have hAx : MeasurableSet (A ×ˢ (Set.univ : Set X)) := hA.prod MeasurableSet.univ
    have hmass : γ.real (A ×ˢ (Set.univ : Set X)) = μ.real A := by
      rw [Set.prod_univ, measureReal_def, ← Measure.map_apply measurable_fst hA, hγ.1,
        measureReal_def]
    have hind : Integrable ((A ×ˢ (Set.univ : Set X)).indicator fun _ => δ ^ 2) γ :=
      (integrable_const _).indicator hAx
    calc μ.real A * δ ^ 2 = ∫ p, (A ×ˢ (Set.univ : Set X)).indicator (fun _ => δ ^ 2) p ∂γ := by
          rw [integral_indicator hAx, setIntegral_const, smul_eq_mul, hmass]
      _ ≤ ∫ p, dist p.1 p.2 ^ 2 ∂γ := by
          refine integral_mono_ae hind hd2 ?_
          filter_upwards [ae_snd_mem hγ hB hνB] with p hp
          by_cases hpA : p.1 ∈ A
          · rw [Set.indicator_of_mem (Set.mk_mem_prod hpA (Set.mem_univ _))]
            exact pow_le_pow_left₀ hδ (hAB _ hpA _ hp) 2
          · rw [Set.indicator_of_notMem (fun h => hpA h.1)]
            positivity
  obtain ⟨c0, hc0⟩ := transportCosts_nonempty μ ν
  have hinf : μ.real A * δ ^ 2 ≤ sInf (transportCosts μ ν) := le_csInf ⟨c0, hc0⟩ hc
  rw [W2, Real.sq_sqrt ((by positivity : (0 : ℝ) ≤ μ.real A * δ ^ 2).trans hinf)]
  exact hinf

/-- The hypotheses of `measureReal_mul_sq_le_W2_sq` are satisfiable: two
Dirac masses at `0` and `1` of `ℝ`, carried by `[-1, 1]`, with `A = {1}`,
`B = {0}` and `δ = 1`. -/
example : MeasurableSet (Metric.closedBall (0 : ℝ) 1) ∧
    Bornology.IsBounded (Metric.closedBall (0 : ℝ) 1) ∧
    Measure.dirac (1 : ℝ) (Metric.closedBall (0 : ℝ) 1)ᶜ = 0 ∧
    Measure.dirac (0 : ℝ) (Metric.closedBall (0 : ℝ) 1)ᶜ = 0 ∧
    MeasurableSet ({1} : Set ℝ) ∧ MeasurableSet ({0} : Set ℝ) ∧
    Measure.dirac (0 : ℝ) ({0} : Set ℝ)ᶜ = 0 ∧
    (∀ a ∈ ({1} : Set ℝ), ∀ b ∈ ({0} : Set ℝ), (1 : ℝ) ≤ dist a b) ∧ (0 : ℝ) ≤ 1 := by
  refine ⟨Metric.isClosed_closedBall.measurableSet, Metric.isBounded_closedBall, ?_, ?_,
    measurableSet_singleton _, measurableSet_singleton _, ?_, ?_, zero_le_one⟩
  · simp [Measure.dirac_apply' _ Metric.isClosed_closedBall.measurableSet.compl]
  · simp [Measure.dirac_apply' _ Metric.isClosed_closedBall.measurableSet.compl]
  · simp [Measure.dirac_apply' _ (measurableSet_singleton _).compl]
  · rintro a rfl b rfl
    simp

end Wasserstein
end Transformer
