module

public import Transformer.Grokking.FiniteRecovery.ArgandFullGradientConvergence
public import Mathlib.MeasureTheory.Measure.OpenPos

/-! A uniform recovery certificate from a strict objective sublevel.
Continuity of the actual finite prefix gives a positive-probability
event for initialization laws with full support. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open Filter MeasureTheory Metric Set Transformer.AMSGradW Transformer.Optimization
open scoped Topology

variable {A : Type*} {H d layers : ℕ}

/-- Initialization is continuous in the raw complex tensor draws. -/
theorem continuous_sgdInitialWeights (realScoreInit : Bool) :
    Continuous (sgdInitialWeights (A := A) (H := H) (d := d) (layers := layers) realScoreInit) := by
  apply continuous_pi
  rintro (⟨p, b⟩ | i)
  · cases p <;> cases b <;>
      first
      | exact continuous_const
      | exact Complex.continuous_re.comp (continuous_apply _)
      | exact Complex.continuous_im.comp (continuous_apply _)
  · exact continuous_const

variable [Fintype A]

/-- For any law on the native draws, one deterministic bound on the norm
of the initialized weights holds on a measurable event of mass above q. -/
theorem exists_initial_norm_bound (mu : Measure (ComplexParameter A H d layers → ℂ))
    [IsProbabilityMeasure mu] (realScoreInit : Bool) {q : ENNReal} (hq : q < 1) :
    ∃ n : ℕ, ∃ U : Set (ComplexParameter A H d layers → ℂ), MeasurableSet U ∧ q < mu U ∧
      ∀ draw ∈ U, ‖(WithLp.toLp 2 (sgdInitialWeights realScoreInit draw) : WeightSpace A H d layers)‖ ≤ n := by
  let coords := PiLp.continuousLinearEquiv 2 ℝ (fun _ : Parameter A H d layers => ℝ)
  let radius (draw : ComplexParameter A H d layers → ℂ) := ‖coords.symm (sgdInitialWeights realScoreInit draw)‖
  let S (n : ℕ) := {draw | radius draw ≤ n}
  have hmono : Monotone S := by
    intro i j hij draw hd
    exact le_trans (show radius draw ≤ (i : ℝ) from hd) (Nat.cast_le.mpr hij)
  have hall : ⋃ n, S n = univ := by
    apply eq_univ_of_forall
    intro draw
    obtain ⟨n, hn⟩ := exists_nat_ge (radius draw)
    exact mem_iUnion.mpr ⟨n, hn⟩
  have ht := tendsto_measure_iUnion_atTop (μ := mu) hmono
  rw [hall, measure_univ] at ht
  obtain ⟨n, hn⟩ := ((tendsto_order.mp ht).1 q hq).exists
  have hc : Continuous radius := (coords.symm.continuous.comp (continuous_sgdInitialWeights realScoreInit)).norm
  exact ⟨n,S n,(isClosed_le hc continuous_const).measurableSet,hn,fun _ h => h⟩

/-- Each finite full-gradient update is continuous in the tensor draws. -/
theorem continuous_prefix_weights_velocity (realScoreInit : Bool) (repeats : ℕ)
    (batch : List (LabeledPosition A)) (lam : ℝ) (rate : ℕ → ℝ) (n : ℕ) :
    let fit := fullGradientMomentumRun (H := H) (d := d) (layers := layers)
      realScoreInit repeats batch lam rate
    Continuous (fun draw => (fit draw n).weights) ∧ Continuous (fun draw => (fit draw n).velocity) := by
  dsimp only
  let W := Weights A H d layers
  let C := PiLp.continuousLinearEquiv 2 ℝ (fun _ : Parameter A H d layers => ℝ)
  let f := fullGradientObjective (H := H) (d := d) (layers := layers) repeats batch lam
  have hgrad : Continuous (fun w : W => WithLp.ofLp (gradient f (WithLp.toLp 2 w))) :=
    C.continuous.comp ((locallyLipschitz_fullGradient repeats batch lam).continuous.comp C.symm.continuous)
  have hclip : Continuous (Kernite.clip : W → W) := by
    refine continuous_pi fun i => (continuous_apply i).div
      ((C.symm.continuous.norm).max continuous_const) ?_
    intro w
    exact ne_of_gt (zero_lt_one.trans_le (le_max_right _ _))
  induction n with
  | zero => exact ⟨continuous_sgdInitialWeights realScoreInit, continuous_const⟩
  | succ n ih =>
    obtain ⟨hx, hv⟩ := ih
    have hs := (hv.const_smul (9 / 10 : ℝ)).add ((hclip.comp hgrad).comp hx)
    exact ⟨hx.sub (hs.const_smul (rate n)), hs⟩

/-- If a finite prefix reaches a strict objective sublevel, the capped
limit theorem applies to a full-support event of draws. A single
positive constant learning rate bound serves the entire event. -/
theorem fullGradientMomentumRun_posMeasure_of_sublevel
    (mu : Measure (ComplexParameter A H d layers → ℂ)) [mu.IsOpenPosMeasure]
    (realScoreInit : Bool) (repeats : ℕ)
    (batch : List (LabeledPosition A)) {lam : ℝ} (hlam : 0 < lam)
    (prefixRate : ℕ → ℝ) (N : ℕ) (cap : ℝ)
    (start : ComplexParameter A H d layers → ℂ)
    (hstart : fullGradientObjective repeats batch lam
      (WithLp.toLp 2 (fullGradientMomentumRun realScoreInit repeats batch lam prefixRate start N).weights) <
        cap)
    (good : Weights A H d layers → Prop)
    (hgood : ∀ star : WeightSpace A H d layers,
      fullGradientObjective repeats batch lam star ≤ cap → good (WithLp.ofLp star)) :
    ∃ U : Set (ComplexParameter A H d layers → ℂ), IsOpen U ∧ start ∈ U ∧ 0 < mu U ∧
      ∃ η₀ > 0, ∀ η, 0 < η → η ≤ η₀ → ∀ draw ∈ U,
        ∃ star : Weights A H d layers,
          Tendsto (fun t => (fullGradientMomentumRun realScoreInit repeats batch lam
            (fun j => if j < N then prefixRate j else η) draw t).weights) atTop (𝓝 star) ∧
          good star := by
  let C := PiLp.continuousLinearEquiv 2 ℝ (fun _ : Parameter A H d layers => ℝ)
  let f := fullGradientObjective (H := H) (d := d) (layers := layers) repeats batch lam
  let prefixFit := fullGradientMomentumRun (H := H) (d := d) (layers := layers)
    realScoreInit repeats batch lam prefixRate
  let x draw := WithLp.toLp 2 (prefixFit draw N).weights
  let v draw := WithLp.toLp 2 (prefixFit draw N).velocity
  let energy η draw := fullMomentumEnergy f η (x draw) (v draw)
  have hη : Continuous (fun η : ℝ => energy η start) := by
    unfold energy fullMomentumEnergy
    fun_prop
  have hpos : ∃ a > 0, energy a start < cap := by
    have hzero : energy 0 start < cap := by simpa [energy, fullMomentumEnergy, f, x, prefixFit] using hstart
    obtain ⟨δ, hδ, hball⟩ := Metric.isOpen_iff.mp ((isOpen_Iio (a := cap)).preimage hη) 0 hzero
    refine ⟨δ / 2, half_pos hδ, hball ?_⟩
    have h : dist (δ / 2) (0 : ℝ) < δ := by
      rw [Real.dist_eq, sub_zero, abs_of_pos (half_pos hδ)]
      linarith
    exact h
  obtain ⟨a, ha, hastart⟩ := hpos
  obtain ⟨hx, hv⟩ := continuous_prefix_weights_velocity (H := H) (d := d) (layers := layers)
    realScoreInit repeats batch lam prefixRate N
  have hx' : Continuous x := C.symm.continuous.comp hx
  have hv' : Continuous v := C.symm.continuous.comp hv
  have hf : Continuous f := (contDiff_fullGradientObjective repeats batch lam).continuous
  have hg : Continuous (gradient f) := (locallyLipschitz_fullGradient repeats batch lam).continuous
  have hc : Continuous (energy a) := by
    unfold energy fullMomentumEnergy clipScale
    exact (hf.comp hx').add
      ((continuous_const.mul (((hg.comp hx').norm).max continuous_const)).mul (hv'.norm.pow 2))
  let U := {draw | energy a draw < cap}
  have hU : IsOpen U := isOpen_Iio.preimage hc
  have hprob : 0 < mu U := hU.measure_pos mu ⟨start, hastart⟩
  obtain ⟨η₀, hη₀, hfit⟩ := fullGradientMomentumRun_converges_below
    (H := H) (d := d) (layers := layers) realScoreInit repeats batch hlam prefixRate N cap
  refine ⟨U, hU, hastart, hprob, min a η₀, lt_min ha hη₀, ?_⟩
  intro η hηpos hηle draw hdraw
  let rate t := if t < N then prefixRate t else η
  let fit := fullGradientMomentumRun realScoreInit repeats batch lam rate draw
  have hprefix : fit N = prefixFit draw N := fullGradientMomentumRun_eq_of_prefix
    realScoreInit repeats batch lam rate prefixRate draw N (by intro t ht; simp [rate, ht])
  have henergy : fullMomentumEnergy f η (WithLp.toLp 2 (fit N).weights)
      (WithLp.toLp 2 (fit N).velocity) ≤ cap := by
    rw [hprefix]
    exact (fullMomentumEnergy_mono_rate f (x draw) (v draw) (hηle.trans (min_le_left _ _))).trans
      hdraw.le
  obtain ⟨star, hstar, _, hcost⟩ := hfit η hηpos (hηle.trans (min_le_right _ _)) draw henergy
  exact ⟨WithLp.ofLp star, (C.continuous.tendsto star).comp hstar, hgood star hcost⟩

end Transformer.Grokking.FiniteRecovery.Argand
