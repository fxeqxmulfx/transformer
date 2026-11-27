module

public import Transformer.Grokking.FiniteRecovery.TransitionForecast
public import Transformer.AMSGradW.Convergence
public import Mathlib.Analysis.Calculus.MeanValue

/-! A convergent training example for the compiled Argand transition.
The labels give the transition table, and the deterministic AMSGradW
run trains a shared positive logit margin with cross-entropy. The
one-dimensional parameterization controls the tied embedding scale and
compensates the gates and input projections as in ArgandTransition. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open Filter
open scoped Topology InnerProductSpace
open Transformer.AMSGradW

noncomputable section

def marginDerivative (competitors : ℕ) (t : ℝ) : ℝ :=
  -(competitors : ℝ) / (Real.exp t + competitors)

private theorem marginLoss_hasDerivAt (competitors : ℕ) (t : ℝ) :
    HasDerivAt (marginLoss competitors) (marginDerivative competitors t) t := by
  have hd := (((Real.hasDerivAt_exp t).add_const (competitors : ℝ)).log (by positivity)).sub
    (hasDerivAt_id t)
  convert hd using 1
  · funext x
    rfl
  · unfold marginDerivative
    field_simp
    ring

private theorem marginDerivative_hasDerivAt (competitors : ℕ) (t : ℝ) :
    HasDerivAt (marginDerivative competitors)
      (competitors * Real.exp t / (Real.exp t + competitors) ^ 2) t := by
  convert (hasDerivAt_const t (-(competitors : ℝ))).div
    ((Real.hasDerivAt_exp t).add_const (competitors : ℝ)) (by positivity) using 1
  · funext x
    simp [marginDerivative, add_comm]
  · ring

private theorem marginDerivative_lipschitz (competitors : ℕ) (s t : ℝ) :
    ‖marginDerivative competitors s - marginDerivative competitors t‖ ≤ ‖s - t‖ := by
  have hb (x : ℝ) : ‖deriv (marginDerivative competitors) x‖ ≤ 1 := by
    rw [(marginDerivative_hasDerivAt competitors x).deriv, Real.norm_of_nonneg (by positivity)]
    apply (div_le_iff₀ (show 0 < (Real.exp x + (competitors : ℝ)) ^ 2 by positivity)).2
    nlinarith [mul_nonneg (Real.exp_pos x).le (Nat.cast_nonneg competitors : (0 : ℝ) ≤ competitors),
      sq_nonneg (Real.exp x), sq_nonneg (competitors : ℝ)]
  simpa only [one_mul] using Convex.norm_image_sub_le_of_norm_deriv_le
    (fun x _ => (marginDerivative_hasDerivAt competitors x).differentiableAt)
    (fun x _ => hb x) (convex_univ : Convex ℝ (Set.univ : Set ℝ)) (Set.mem_univ t) (Set.mem_univ s)

/-- The margin is the scalar parameter plus one. Along the run the
parameter stays nonnegative. -/
def marginObjective (competitors : ℕ) (x : TrainingSpace 1) : ℝ :=
  marginLoss competitors (x 0 + 1)

private theorem marginObjective_hasGradientAt (competitors : ℕ) (x : TrainingSpace 1) :
    HasGradientAt (marginObjective competitors)
      (EuclideanSpace.single 0 (marginDerivative competitors (x 0 + 1))) x := by
  let coord : TrainingSpace 1 →L[ℝ] ℝ := PiLp.proj 2 (fun _ : Fin 1 => ℝ) 0
  have hf := (marginLoss_hasDerivAt competitors (x 0 + 1)).hasFDerivAt.comp x
    ((coord.hasFDerivAt (x := x)).add_const 1)
  have he : InnerProductSpace.toDual ℝ (TrainingSpace 1)
      (EuclideanSpace.single 0 (marginDerivative competitors (x 0 + 1))) =
      (ContinuousLinearMap.toSpanSingleton ℝ (marginDerivative competitors (x 0 + 1))).comp coord := by
    ext y
    simp [coord, InnerProductSpace.toDual_apply_apply, EuclideanSpace.inner_single_left,
      ContinuousLinearMap.toSpanSingleton_apply, mul_comm]
  apply hasGradientAt_iff_hasFDerivAt.mpr
  rw [he]
  exact hf

theorem marginObjective_gradient (competitors : ℕ) (x : TrainingSpace 1) :
    gradient (marginObjective competitors) x 0 = marginDerivative competitors (x 0 + 1) := by
  rw [(marginObjective_hasGradientAt competitors x).gradient]
  simp

private theorem marginObjective_lipschitz (competitors : ℕ) :
    LipschitzGradient (marginObjective competitors) 1 := by
  intro x y
  rw [one_mul]
  refine (pi_norm_le_iff_of_nonneg (norm_nonneg _)).2 fun i => ?_
  have hi : i = 0 := Fin.eq_zero i
  subst i
  change ‖gradient (marginObjective competitors) (WithLp.toLp 2 x) 0 -
    gradient (marginObjective competitors) (WithLp.toLp 2 y) 0‖ ≤ ‖x - y‖
  rw [marginObjective_gradient, marginObjective_gradient]
  have h := marginDerivative_lipschitz competitors (x 0 + 1) (y 0 + 1)
  rw [add_sub_add_right_eq_sub] at h
  exact h.trans (norm_le_pi_norm (x - y) 0)

/-- Ordinary deterministic AMSGradW on the scalar objective. The chosen
constants meet the decay-dominated hypotheses of AMSGradW.Convergence. -/
def marginRun (competitors : ℕ) : ℕ → TrainingState 1 :=
  trainingRun (1 / 4) 1 2 (9 / 10) (98 / 100) (marginObjective competitors) 0

private theorem marginRun_nonneg (competitors : ℕ) (step : ℕ) :
    0 ≤ (marginRun competitors step).position 0 ∧ (marginRun competitors step).momentum 0 ≤ 0 := by
  induction step with
  | zero => simp [marginRun, trainingRun]
  | succ step ih =>
    let state := marginRun competitors step
    let next := trainingStep (1 / 4) 1 2 (9 / 10) (98 / 100) (marginObjective competitors) state
    change 0 ≤ next.position 0 ∧ next.momentum 0 ≤ 0
    have hstep := trainingStep_coordinates (1 / 4) 1 2 (9 / 10) (98 / 100)
      (marginObjective competitors) state 0
    have hg : gradient (marginObjective competitors) state.position 0 ≤ 0 := by
      rw [marginObjective_gradient]
      unfold marginDerivative
      exact div_nonpos_of_nonpos_of_nonneg (neg_nonpos.mpr (Nat.cast_nonneg _)) (by positivity)
    have hm : next.momentum 0 ≤ 0 := by
      change (trainingStep (1 / 4) 1 2 (9 / 10) (98 / 100)
        (marginObjective competitors) state).momentum 0 ≤ 0
      rw [hstep.1]
      linarith [ih.2]
    refine ⟨?_, hm⟩
    change 0 ≤ (trainingStep (1 / 4) 1 2 (9 / 10) (98 / 100)
      (marginObjective competitors) state).position 0
    rw [hstep.2]
    change 0 ≤ (1 - 1 / 4 * 2) * state.position 0 -
      1 / 4 * next.momentum 0 / Transformer.AMSGrad.trainingDenominator 1 next 0
    have hd := denominator_lower 1 next 0
    have hdiv := div_nonpos_of_nonpos_of_nonneg (mul_nonpos_of_nonneg_of_nonpos
      (show (0 : ℝ) ≤ 1 / 4 by norm_num) hm) (zero_le_one.trans hd)
    have hp : 0 ≤ state.position 0 := ih.1
    linarith

/-- The trained margin and its cross-entropy converge, with margin at
least one throughout the run. -/
theorem marginRun_converges (competitors : ℕ) :
    ∃ margin : ℝ, 1 ≤ margin ∧
      (∀ step, (1 : ℝ) ≤ (marginRun competitors step).position 0 + 1) ∧
      Tendsto (fun step => (marginRun competitors step).position 0 + 1) atTop (𝓝 margin) ∧
      Tendsto (fun step => marginObjective competitors (marginRun competitors step).position)
        atTop (𝓝 (marginLoss competitors margin)) := by
  have hf : Differentiable ℝ (marginObjective competitors) :=
    fun x => (marginObjective_hasGradientAt competitors x).differentiableAt
  obtain ⟨D, star, hD, hpos, hmoment, hmetric, hloss, hbalance⟩ :=
    trainingRun_convergence (1 / 4) 1 2 (9 / 10) (98 / 100) 1 (marginObjective competitors)
      0 hf (marginObjective_lipschitz competitors) (by norm_num) (by norm_num) (by norm_num)
      (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (by norm_num)
  have ht : Tendsto (fun step => (marginRun competitors step).position 0) atTop (𝓝 (star 0)) :=
    ((PiLp.proj 2 (𝕜 := ℝ) (fun _ : Fin 1 => ℝ) 0).continuous.tendsto star).comp hpos
  have hn : 0 ≤ star 0 := le_of_tendsto_of_tendsto' tendsto_const_nhds ht (fun step =>
    (marginRun_nonneg competitors step).1)
  refine ⟨star 0 + 1, by linarith, fun step => ?_, ht.add_const 1, hloss⟩
  linarith [(marginRun_nonneg competitors step).1]

/-- Compile the next-token law from the observed graph and use the trained
margin for the ordinary Argand readout at this step. -/
def trainedTransition {H d T : ℕ} (part : Finset (Word (Fin (H * d)) (T + 1) × Fin (H * d)))
    (step : ℕ) : Model (Fin (H * d)) H d 1 :=
  transitionModel (transitionFromData part) ((marginRun (H * d - 1) step).position 0 + 1)

/-- The actual training and held-out losses of the compiled ordinary Argand
converge to the same finite value. The labels can come from any histories
covering the possible current states. Greedy forecasts are exact throughout
because the recovered transition is always scored with positive margin. -/
theorem trainedTransition_converges {H d T : ℕ} (hc : 0 < H * d)
    (next : Fin (H * d) → Fin (H * d)) (inputs : Finset (Word (Fin (H * d)) (T + 1)))
    (hcover : ∀ a, ∃ word ∈ inputs, word (Fin.last T) = a) :
    let part := observe inputs (transitionTask next)
    ∃ margin : ℝ, 1 ≤ margin ∧
      (∀ step, (trainedTransition part step).realizes 1 (transitionTask (T := T) next)) ∧
      Tendsto (fun step => (trainedTransition part step).observedLoss 1 part)
        atTop (𝓝 (marginLoss (H * d - 1) margin)) ∧
      ∀ word : Word (Fin (H * d)) (T + 1), Tendsto (fun step => (trainedTransition part step).labelLoss 1 word
          (transitionTask next word)) atTop (𝓝 (marginLoss (H * d - 1) margin)) := by
  dsimp only
  have hdata := transitionFromData_eq_of_covers next inputs hcover
  obtain ⟨word, hw, -⟩ := hcover ⟨0, hc⟩
  obtain ⟨margin, hmargin, hrange, -, hlim⟩ := marginRun_converges (H * d - 1)
  have hp (step : ℕ) : 0 < (marginRun (H * d - 1) step).position 0 + 1 :=
    zero_lt_one.trans_le (hrange step)
  refine ⟨margin, hmargin, ?_, ?_, ?_⟩
  · intro step
    rw [trainedTransition, hdata]
    exact transitionModel_realizes hc next (hp step)
  · have he (step : ℕ) :
        (trainedTransition (observe inputs (transitionTask next)) step).observedLoss 1
          (observe inputs (transitionTask next)) =
          marginObjective (H * d - 1) (marginRun (H * d - 1) step).position := by
      rw [trainedTransition, hdata, transitionModel_observedLoss hc next (hp step) inputs ⟨word, hw⟩]
      rfl
    simpa only [he] using hlim
  · intro w
    have he (step : ℕ) :
        (trainedTransition (observe inputs (transitionTask next)) step).labelLoss 1 w
          (transitionTask next w) =
          marginObjective (H * d - 1) (marginRun (H * d - 1) step).position := by
      rw [trainedTransition, hdata, transitionModel_labelLoss hc next (hp step)]
      rfl
    simpa only [he] using hlim

example (next : Fin 256 → Fin 256) :=
  trainedTransition_converges (H := 4) (d := 64) (T := 31) (by decide) next
    (transitionInputs (Fin 256) 31) fun a =>
      ⟨fun _ => a, Finset.mem_map.mpr ⟨a, Finset.mem_univ _, rfl⟩, rfl⟩

end

end Transformer.Grokking.FiniteRecovery.Argand
