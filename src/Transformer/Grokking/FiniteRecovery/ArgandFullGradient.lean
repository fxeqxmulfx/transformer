module

public import Transformer.Grokking.FiniteRecovery.ArgandAccumulation
public import Transformer.Grokking.FiniteRecovery.ArgandMomentum
public import Transformer.Optimization.FullMomentumConvergence

/-! The native Argand weights and clipped Polyak velocity on one fixed
finite objective. Accumulation contributes the mean data gradient and
the quadratic penalty is applied once, before clipping. -/

@[expose] public section

open Filter Metric Set
open scoped Topology

namespace Transformer.Grokking.FiniteRecovery.Argand

open Transformer.AMSGradW Transformer.Optimization

variable {A : Type*} [Fintype A] {H d layers : ℕ}

/-- The original momentum rule starts with zero velocity and trains every
coordinate. Here the same complete gradient is used at each optimizer
step, evaluated at that step's weights. -/
noncomputable def fullGradientMomentumRun (realScoreInit : Bool) (repeats : ℕ)
    (batch : List (LabeledPosition A)) (lam : ℝ) (rate : ℕ → ℝ)
    (draw : ComplexParameter A H d layers → ℂ) : ℕ → ClippedMomentum.State (Parameter A H d layers) :=
  ClippedMomentum.run rate (fun _ w =>
    WithLp.ofLp (gradient (fullGradientObjective repeats batch lam) (WithLp.toLp 2 w)))
    (sgdInitialWeights realScoreInit draw)

private theorem fullGradientMomentumRun_velocity (realScoreInit : Bool) (repeats : ℕ)
    (batch : List (LabeledPosition A)) (lam : ℝ) (rate : ℕ → ℝ)
    (draw : ComplexParameter A H d layers → ℂ) (t : ℕ) :
    let run := fullGradientMomentumRun realScoreInit repeats batch lam rate draw
    let x := WithLp.toLp 2 (run t).weights
    let g := gradient (fullGradientObjective repeats batch lam) x
    (WithLp.toLp 2 (run (t + 1)).velocity : WeightSpace A H d layers) =
      (9 / 10 : ℝ) • WithLp.toLp 2 (run t).velocity + (clipScale g)⁻¹ • g := by
  let f := fullGradientObjective (H := H) (d := d) (layers := layers) repeats batch lam
  let run := fullGradientMomentumRun realScoreInit repeats batch lam rate draw
  let x : WeightSpace A H d layers := WithLp.toLp 2 (run t).weights
  let v : WeightSpace A H d layers := WithLp.toLp 2 (run t).velocity
  ext i
  change (9 / 10 : ℝ) * WithLp.ofLp v i + WithLp.ofLp (gradient f x) i / clipScale (gradient f x) = _
  simp only [PiLp.add_apply, PiLp.smul_apply, smul_eq_mul]
  apply congrArg ((9 / 10 : ℝ) * WithLp.ofLp v i + ·)
  exact div_eq_inv_mul _ _

private theorem fullGradientMomentumRun_position (realScoreInit : Bool) (repeats : ℕ)
    (batch : List (LabeledPosition A)) (lam : ℝ) (rate : ℕ → ℝ)
    (draw : ComplexParameter A H d layers → ℂ) (t : ℕ) :
    let run := fullGradientMomentumRun realScoreInit repeats batch lam rate draw
    (WithLp.toLp 2 (run (t + 1)).weights : WeightSpace A H d layers) =
      WithLp.toLp 2 (run t).weights - rate t • WithLp.toLp 2 (run (t + 1)).velocity := by
  ext i
  rfl

private theorem fullGradientMomentumRun_bufferBound (realScoreInit : Bool) (repeats : ℕ)
    (batch : List (LabeledPosition A)) (lam : ℝ) (rate : ℕ → ℝ)
    (draw : ComplexParameter A H d layers → ℂ) (t : ℕ) :
    ‖(WithLp.toLp 2 (fullGradientMomentumRun realScoreInit repeats batch lam rate draw t).velocity :
      WeightSpace A H d layers)‖ ≤ 10 := by
  induction t with
  | zero => simp [fullGradientMomentumRun, ClippedMomentum.run]
  | succ t ih =>
    rw [fullGradientMomentumRun_velocity]
    exact norm_fullMomentum_velocity_le _ _ ih

/-- An optimizer state depends only on the rates used before its step. -/
theorem fullGradientMomentumRun_eq_of_prefix (realScoreInit : Bool) (repeats : ℕ)
    (batch : List (LabeledPosition A)) (lam : ℝ) (r s : ℕ → ℝ)
    (draw : ComplexParameter A H d layers → ℂ) (n : ℕ) (h : ∀ t < n, r t = s t) :
    fullGradientMomentumRun realScoreInit repeats batch lam r draw n =
      fullGradientMomentumRun realScoreInit repeats batch lam s draw n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    have ht := h n (Nat.lt_succ_self n)
    have hh := ih (fun t ht => h t (ht.trans (Nat.lt_succ_self _)))
    simp only [fullGradientMomentumRun] at hh ⊢
    simp only [ClippedMomentum.run, hh, ht]

/-- A bound on the energy at the end of the prefix gives a tail rate
bound shared by all draws meeting the same energy cap. -/
theorem fullGradientMomentumRun_limits_below (realScoreInit : Bool) (repeats : ℕ)
    (batch : List (LabeledPosition A)) {lam : ℝ} (hlam : 0 < lam)
    (prefixRate : ℕ → ℝ) (N : ℕ) (cap : ℝ) :
    ∃ η₀ > 0, ∀ η, 0 < η → η ≤ η₀ → ∀ draw : ComplexParameter A H d layers → ℂ,
      let f := fullGradientObjective repeats batch lam
      let run := fullGradientMomentumRun realScoreInit repeats batch lam
        (fun t => if t < N then prefixRate t else η) draw
      let x := fun t => WithLp.toLp 2 (run t).weights
      let v := fun t => WithLp.toLp 2 (run t).velocity
      fullMomentumEnergy f η (x N) (v N) ≤ cap →
        ∃ (star : WeightSpace A H d layers) (u : ℕ → ℕ),
          StrictMono u ∧ Tendsto (x ∘ u) atTop (𝓝 star) ∧ gradient f star = 0 ∧
            Tendsto (fun t => f (x t)) atTop (𝓝 (f star)) ∧ Tendsto v atTop (𝓝 0) ∧
            Tendsto (fun t => gradient f (x t)) atTop (𝓝 0) ∧ f star ≤ cap ∧
            (MomentumEnergyInequality f η star → Tendsto x atTop (𝓝 star)) := by
  let E := WeightSpace A H d layers
  let f := fullGradientObjective (H := H) (d := d) (layers := layers) repeats batch lam
  have hf : ContDiff ℝ 1 f := contDiff_fullGradientObjective repeats batch lam
  have hLip : LocallyLipschitz (gradient f) := locallyLipschitz_fullGradient repeats batch lam
  have hc (x : E) : (lam / 2) * ‖x‖ ^ 2 ≤ f x := fullGradientObjective_coercive repeats batch lam x
  have hnonneg (x : E) : 0 ≤ f x := (mul_nonneg (half_pos hlam).le (sq_nonneg _)).trans (hc x)
  obtain ⟨R, _, η₀, hη₀, hb⟩ := exists_pos_fullMomentum_descent_below
    (hf.differentiable one_ne_zero) hLip (half_pos hlam) hc cap
  refine ⟨η₀, hη₀, fun η hη hη' draw hcap => ?_⟩
  let rate t := if t < N then prefixRate t else η
  let fit := fullGradientMomentumRun realScoreInit repeats batch lam rate draw
  let x (t : ℕ) : E := WithLp.toLp 2 (fit t).weights
  let v (t : ℕ) : E := WithLp.toLp 2 (fit t).velocity
  let xTail t := x (t + N)
  let vTail t := v (t + N)
  have hv (t : ℕ) : vTail (t + 1) = (9 / 10 : ℝ) • vTail t +
      (clipScale (gradient f (xTail t)))⁻¹ • gradient f (xTail t) := by
    simpa only [xTail, vTail, Nat.add_right_comm t 1 N] using
      fullGradientMomentumRun_velocity realScoreInit repeats batch lam rate draw (t + N)
  have hx (t : ℕ) : xTail (t + 1) = xTail t - η • vTail (t + 1) := by
    have hrate : rate (t + N) = η := by simp [rate, show ¬t + N < N by omega]
    simpa only [xTail, vTail, Nat.add_right_comm t 1 N, hrate] using
      fullGradientMomentumRun_position realScoreInit repeats batch lam rate draw (t + N)
  have hcap₀ : fullMomentumEnergy f η (xTail 0) (vTail 0) ≤ cap := by
    simpa only [xTail, vTail, Nat.zero_add] using hcap
  have hv₀ : ‖vTail 0‖ ≤ 10 := by
    simpa only [vTail, v, Nat.zero_add] using
      fullGradientMomentumRun_bufferBound realScoreInit repeats batch lam rate draw N
  obtain ⟨hR, hstep⟩ := hb η hη hη' xTail vTail hcap₀ hv₀ hv hx
  obtain ⟨value, hvalue, hvel, hgrad, _⟩ := fullMomentum_limits hnonneg hη xTail vTail hv hstep
  obtain ⟨star, _, u, hu, hlim⟩ := (isCompact_closedBall (0 : E) R).tendsto_subseq
    (fun t => mem_closedBall_zero_iff.mpr (hR t))
  have hstationary := tendsto_nhds_unique ((hLip.continuous.tendsto star).comp hlim)
    (hgrad.comp hu.tendsto_atTop)
  have hvalue' := tendsto_nhds_unique (hvalue.comp hu.tendsto_atTop) ((hf.continuous.tendsto star).comp hlim)
  subst value
  have hcapAt (t : ℕ) : fullMomentumEnergy f η (xTail t) (vTail t) ≤ cap := by
    induction t with
    | zero => exact hcap₀
    | succ t ih => exact (le_add_of_nonneg_right (by positivity)).trans ((hstep t).trans ih)
  have hcapLimit : f star ≤ cap := le_of_tendsto hvalue
    (Eventually.of_forall fun t => (le_fullMomentumEnergy f hη.le _ _).trans (hcapAt t))
  refine ⟨star, fun t => u t + N, fun i j h => Nat.add_lt_add_right (hu h) N, hlim,
    hstationary, (tendsto_add_atTop_iff_nat N).1 hvalue, (tendsto_add_atTop_iff_nat N).1 hvel,
    (tendsto_add_atTop_iff_nat N).1 hgrad, hcapLimit, ?_⟩
  intro hKL
  exact (tendsto_add_atTop_iff_nat N).1
    (fullMomentum_tendsto_of_energy_inequality hf.continuous hnonneg hη xTail vTail hv hx hstep hu hlim hKL)

/-- After any finite prefix, a small constant tail gives vanishing
velocity and full gradient. The Łojasiewicz inequality at a stationary
cluster point then gives a limit of the entire trajectory. -/
theorem fullGradientMomentumRun_limits (realScoreInit : Bool) (repeats : ℕ)
    (batch : List (LabeledPosition A)) {lam : ℝ} (hlam : 0 < lam)
    (draw : ComplexParameter A H d layers → ℂ) (prefixRate : ℕ → ℝ) (N : ℕ) :
    ∃ η₀ > 0, ∀ η, 0 < η → η ≤ η₀ →
      let f := fullGradientObjective repeats batch lam
      let run := fullGradientMomentumRun realScoreInit repeats batch lam
        (fun t => if t < N then prefixRate t else η) draw
      let x := fun t => WithLp.toLp 2 (run t).weights
      let v := fun t => WithLp.toLp 2 (run t).velocity
      ∃ (star : WeightSpace A H d layers) (u : ℕ → ℕ),
        StrictMono u ∧ Tendsto (x ∘ u) atTop (𝓝 star) ∧ gradient f star = 0 ∧
          Tendsto (fun t => f (x t)) atTop (𝓝 (f star)) ∧ Tendsto v atTop (𝓝 0) ∧
          Tendsto (fun t => gradient f (x t)) atTop (𝓝 0) ∧
          (MomentumEnergyInequality f η star → Tendsto x atTop (𝓝 star)) := by
  let f := fullGradientObjective (H := H) (d := d) (layers := layers) repeats batch lam
  let init := fullGradientMomentumRun realScoreInit repeats batch lam prefixRate draw N
  let cap := fullMomentumEnergy f 1 (WithLp.toLp 2 init.weights) (WithLp.toLp 2 init.velocity)
  obtain ⟨η₀, hη₀, hb⟩ := fullGradientMomentumRun_limits_below
    (H := H) (d := d) (layers := layers) realScoreInit repeats batch hlam prefixRate N cap
  refine ⟨min η₀ 1, lt_min hη₀ zero_lt_one, fun η hη hη' => ?_⟩
  let rate t := if t < N then prefixRate t else η
  let run := fullGradientMomentumRun realScoreInit repeats batch lam rate draw
  have hinit : run N = init := fullGradientMomentumRun_eq_of_prefix
    realScoreInit repeats batch lam rate prefixRate draw N (by intro t ht; simp [rate, ht])
  have hcap : fullMomentumEnergy f η (WithLp.toLp 2 (run N).weights)
      (WithLp.toLp 2 (run N).velocity) ≤ cap := by
    rw [hinit]
    exact fullMomentumEnergy_mono_rate f _ _ (hη'.trans (min_le_right _ _))
  obtain ⟨star, u, hu, hsub, hstat, hvalue, hvel, hgrad, _, hlim⟩ :=
    hb η hη (hη'.trans (min_le_left _ _)) draw hcap
  exact ⟨star, u, hu, hsub, hstat, hvalue, hvel, hgrad, hlim⟩

example (batch : List (LabeledPosition (Fin 16))) (draw : ComplexParameter (Fin 16) 4 16 2 → ℂ) :=
  fullGradientMomentumRun_limits false 2 batch (show 0 < (1 / 10 : ℝ) by norm_num)
    draw (wsdRate 20000) 19999

end Transformer.Grokking.FiniteRecovery.Argand
