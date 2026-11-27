module

public import Transformer.Grokking.FiniteRecovery.ArgandEnergyRecovery
public import Transformer.Grokking.FiniteRecovery.ArgandRecoveryObstruction

/-! Excessive coupled decay forces zero logits with arbitrarily high
probability. The same rate and decay apply to all draws in the event. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open Filter Metric Set MeasureTheory Transformer.Optimization
open scoped Topology ENNReal

variable {A : Type*} [Fintype A] {H d layers : ℕ}

private theorem zero_limit_of_large_decay (realScoreInit : Bool) (repeats : ℕ)
    (batch : List (LabeledPosition A)) {R : ℝ} (hR : 0 ≤ R) :
    ∃ lam₀ > 0, ∀ lam, lam₀ ≤ lam → ∃ η₀ > 0, ∀ η, 0 < η → η ≤ η₀ →
      ∀ draw : ComplexParameter A H d layers → ℂ,
        ‖(WithLp.toLp 2 (sgdInitialWeights realScoreInit draw) : WeightSpace A H d layers)‖ ≤ R →
        Tendsto (fun t => (fullGradientMomentumRun realScoreInit repeats batch lam
          (fun _ => η) draw t).weights) atTop (𝓝 0) := by
  let E := WeightSpace A H d layers
  let data := fullGradientObjective (H := H) (d := d) (layers := layers) repeats batch 0
  have hD := (contDiff_fullGradientObjective (H := H) (d := d) (layers := layers)
    repeats batch 0).continuous
  obtain ⟨best, _, hbest⟩ := (isCompact_closedBall (0 : E) R).exists_isMaxOn
    ⟨0, mem_closedBall_self hR⟩ hD.continuousOn
  obtain ⟨L, hL⟩ := (locallyLipschitz_fullGradient (H := H) (d := d) (layers := layers)
    repeats batch 0).locallyLipschitzOn.exists_lipschitzOnWith_of_compact
      (isCompact_closedBall (0 : E) (R + 1))
  have hg₀ : gradient data 0 = 0 := by
    rw [fullGradientObjective_gradient]
    have hz := batchGradient_eq_zero_of_zeroEmbeddings repeats batch (0 : Weights A H d layers)
      (fun _ _ _ => rfl)
    simp only [WithLp.ofLp_zero, hz, WithLp.toLp_zero, zero_smul, add_zero]
  let lam₀ := max (L : ℝ) (2 * data best) + 1
  have hL₀ : (L : ℝ) < lam₀ := (le_max_left _ _).trans_lt (lt_add_one _)
  have hb₀ : 2 * data best < lam₀ := (le_max_right _ _).trans_lt (lt_add_one _)
  refine ⟨lam₀, L.coe_nonneg.trans_lt hL₀, fun lam hlarge => ?_⟩
  have hlam : 0 < lam := (L.coe_nonneg.trans_lt hL₀).trans_le hlarge
  let f := fullGradientObjective (H := H) (d := d) (layers := layers) repeats batch lam
  let cap := (lam / 2) * (R + 1) ^ 2
  obtain ⟨η₀, hη₀, hrun⟩ := fullGradientMomentumRun_converges_below
    (H := H) (d := d) (layers := layers) realScoreInit repeats batch hlam (fun _ => 0) 0 cap
  refine ⟨η₀, hη₀, fun η hη hηle draw hdraw => ?_⟩
  let x : E := WithLp.toLp 2 (sgdInitialWeights realScoreInit draw)
  have hx : x ∈ closedBall 0 R := mem_closedBall_zero_iff.mpr hdraw
  have hdata : data x ≤ data best := hbest hx
  have hdef : f x = data x + lam / 2 * ‖x‖ ^ 2 := by simp [f, data, fullGradientObjective]
  have hcap : f x ≤ cap := by
    rw [hdef]
    have hsq := mul_le_mul_of_nonneg_left (sq_le_sq' (by linarith [norm_nonneg x]) hdraw) (half_pos hlam).le
    dsimp [cap]
    nlinarith only [hsq, hdata, hb₀, hlarge, mul_nonneg hlam.le hR]
  have henergy : fullMomentumEnergy f η x 0 ≤ cap := by simpa [fullMomentumEnergy] using hcap
  obtain ⟨star, hs, hstat, hcost⟩ := hrun η hη hηle draw henergy
  have hstar : ‖star‖ ≤ R + 1 := by
    have h := (fullGradientObjective_coercive repeats batch lam star).trans hcost
    have hsq := (mul_le_mul_iff_right₀ (half_pos hlam)).mp h
    nlinarith only [hsq, norm_nonneg star, hR]
  have hG : gradient data star + lam • star = 0 := by
    simpa only [data, fullGradientObjective_gradient, zero_smul, add_zero] using hstat
  have hbound := hL.norm_sub_le (mem_closedBall_zero_iff.mpr hstar)
    (mem_closedBall_self (by linarith : 0 ≤ R + 1))
  change ‖gradient data star - gradient data 0‖ ≤ (L : ℝ) * ‖star - 0‖ at hbound
  rw [hg₀, sub_zero, sub_zero, eq_neg_of_add_eq_zero_left hG, norm_neg, norm_smul,
    Real.norm_of_nonneg hlam.le] at hbound
  have hzero : star = 0 := by
    by_contra hne
    exact (not_lt_of_ge hbound) (mul_lt_mul_of_pos_right (hL₀.trans_le hlarge) (norm_pos_iff.mpr hne))
  subst star
  let coords := PiLp.continuousLinearEquiv 2 ℝ (fun _ : Parameter A H d layers => ℝ)
  have hout := (coords.continuous.tendsto 0).comp hs
  change Tendsto (fun t => (fullGradientMomentumRun realScoreInit repeats batch lam
    (fun j => if j < 0 then (0 : ℝ) else η) draw t).weights) atTop (𝓝 (coords 0)) at hout
  simpa only [Nat.not_lt_zero, ite_false, map_zero] using hout

/-- For any probability threshold below one, a fixed large enough decay
makes the complete weights converge to zero above that probability.
This includes every fixed observation mask of every finite task. -/
theorem prob_zero_limit_of_large_decay (mu : Measure (ComplexParameter A H d layers → ℂ))
    [IsProbabilityMeasure mu] (realScoreInit : Bool) (repeats : ℕ)
    (batch : List (LabeledPosition A)) {q : ℝ≥0∞} (hq : q < 1) :
    ∃ S : Set (ComplexParameter A H d layers → ℂ), MeasurableSet S ∧ q < mu S ∧
      ∃ lam₀ > 0, ∀ lam, lam₀ ≤ lam → ∃ η₀ > 0, ∀ η, 0 < η → η ≤ η₀ → ∀ draw ∈ S,
        Tendsto (fun t => (fullGradientMomentumRun realScoreInit repeats batch lam
          (fun _ => η) draw t).weights) atTop (𝓝 0) := by
  obtain ⟨n,S,hS,hp,hbound⟩ := exists_initial_norm_bound mu realScoreInit hq
  obtain ⟨l,hl,h⟩ := zero_limit_of_large_decay (H := H) (d := d) (layers := layers)
    realScoreInit repeats batch (R := (n : ℝ)) (Nat.cast_nonneg _)
  refine ⟨S,hS,hp,l,hl,fun lam hlam => ?_⟩
  obtain ⟨a,ha,hrun⟩ := h lam hlam
  exact ⟨a,ha,fun η hη hb draw hd => hrun η hη hb draw (hbound draw hd)⟩

example (mu : Measure (ComplexParameter (ZMod 2) 4 16 1 → ℂ)) [IsProbabilityMeasure mu] :=
  let e := wordEnumeration 2 4
  prob_zero_limit_of_large_decay mu false 1
    (observedLastPositions (T := 7) (modularSum e)
      (inputsAtCells (enumeratedCut e) (FlatSpectrum.staircase 15)))
    (q := ENNReal.ofReal (99 / 100)) (by norm_num [← ENNReal.ofReal_one, ENNReal.ofReal_lt_ofReal_iff])

end Transformer.Grokking.FiniteRecovery.Argand
