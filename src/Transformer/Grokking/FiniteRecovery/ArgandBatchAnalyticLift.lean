module

public import Transformer.Grokking.FiniteRecovery.ArgandLayerAnalyticLift
public import Transformer.Grokking.FiniteRecovery.ArgandAccumulation
public import Mathlib.Analysis.SpecialFunctions.Log.Deriv

/-! The complete real-coordinate batch loss of Argand is C¹. In
particular, vanishing minibatch gradients at a finite limit give actual
stationarity, even at zero activations. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open Transformer.PhasorV2 Transformer.Optimization Transformer.AMSGradW
open scoped InnerProductSpace

variable {A : Type*} [Fintype A] {H d layers T : ℕ}

private theorem analyticLift_hidden_fromWeights (repeats : ℕ) (word : Word A T) (i : Fin T) :
    HasAnalyticLift (fun w : Weights A H d layers => (Model.fromWeights w).hidden repeats word i) := by
  let W := Weights A H d layers
  let ε : ℝ := 1 / 1000000
  let scale := 1 / Real.sqrt (2 * (layers : ℝ) * repeats)
  let x (w : W) (j : Fin T) := argandRMS ε ((Model.fromWeights w).embedding (word j))
  let slide (w : W) (z : Word (CSpace (H * d)) T) (l : Fin layers) :=
    ((Model.fromWeights w).layer l).forwardWith ε scale (causalGeometry (Model.fromWeights w).alpha) z
  let pass (w : W) (z : Word (CSpace (H * d)) T) := (List.finRange layers).foldl (slide w) z
  have hrms := analyticLift_argandRMS (c := H * d) (by norm_num : 0 < ε)
  have hx (j : Fin T) : HasAnalyticLift (fun w => x w j) :=
    hrms.comp (analyticLift_embedding_fromWeights (word j))
  have hpass (ls : List (Fin layers)) (z : W → Word (CSpace (H * d)) T)
      (hz : ∀ j, HasAnalyticLift (fun w => z w j)) (j : Fin T) :
      HasAnalyticLift (fun w => ls.foldl (slide w) (z w) j) := by
    induction ls generalizing z with
    | nil => exact hz j
    | cons l ls ih =>
      simp only [List.foldl_cons]
      exact ih (fun w => slide w (z w) l) (analyticLift_forward_fromWeights l scale z hz)
  have hrepeat (n : ℕ) (j : Fin T) : HasAnalyticLift (fun w => ((pass w)^[n]) (x w) j) := by
    induction n generalizing j with
    | zero => exact hx j
    | succ n ih =>
      simp only [Function.iterate_succ_apply']
      exact hpass _ _ ih j
  simp only [Model.hidden, Model.hiddenWith, List.ofFn_eq_map, List.foldl_map]
  change HasAnalyticLift (fun w : W => argandRMS ε ((pass w)^[repeats] (x w) i))
  exact hrms.comp (hrepeat repeats i)

theorem analyticLift_logits_fromWeights (repeats : ℕ) (word : Word A T) (i : Fin T) (a : A) :
    HasAnalyticLift (fun w : Weights A H d layers => (Model.fromWeights w).logits repeats word i a) :=
  ((HasAnalyticLift.clm Complex.reCLM).comp ((analyticLift_embedding_fromWeights a).inner ℂ
    (analyticLift_hidden_fromWeights repeats word i))).div_const 2

theorem analyticLift_labelLoss_fromWeights (repeats : ℕ) (word : Word A T) (i : Fin T) (a : A) :
    HasAnalyticLift (fun w : Weights A H d layers =>
      -Real.log ((Model.fromWeights w).probability repeats word i a)) := by
  have hp (t : A) := (analyticLift_logits_fromWeights (H := H) (d := d) (layers := layers) repeats word i t).exp
  have hsum (w : Weights A H d layers) : 0 < ∑ t, Real.exp ((Model.fromWeights w).logits repeats word i t) :=
    Finset.sum_pos' (fun _ _ => (Real.exp_pos _).le) ⟨a, Finset.mem_univ _, Real.exp_pos _⟩
  have hprob : HasAnalyticLift (fun w : Weights A H d layers => (Model.fromWeights w).probability repeats word i a) :=
    (hp a).div (HasAnalyticLift.sum fun t _ => hp t) (fun w => (hsum w).ne')
  exact (hprob.log fun w => (Model.fromWeights w).probability_pos repeats word i a).neg

theorem analyticLift_batchLoss (repeats : ℕ) (batch : List (LabeledPosition A)) :
    HasAnalyticLift (batchLoss (H := H) (d := d) (layers := layers) repeats batch) := by
  unfold batchLoss
  apply HasAnalyticLift.div_const
  induction batch with
  | nil => exact (HasAnalyticLift.const _)
  | cons p batch ih =>
    simp only [List.map_cons, List.sum_cons]
    exact (analyticLift_labelLoss_fromWeights repeats p.2.1 p.2.2.1 p.2.2.2).add ih



/-- Squared Euclidean norms are real analytic, even at zero. -/
theorem analyticAt_sq_norm {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    (z : E) : AnalyticAt ℝ (fun x : E => ‖x‖ ^ 2) z := by
  have heq : (fun x : E => Inner.inner ℝ x x) = (fun x => ‖x‖ ^ 2) :=
    funext real_inner_self_eq_norm_sq
  rw [← heq]
  exact ((innerSL ℝ : E →L[ℝ] E →L[ℝ] ℝ).analyticAt_bilinear (z, z)).comp
    (f := fun x : E => (x, x)) (analyticAt_id.prod analyticAt_id)

theorem analyticLift_fullGradientObjective (repeats : ℕ) (batch : List (LabeledPosition A))
    (lam : ℝ) :
    HasAnalyticLift (fullGradientObjective (H := H) (d := d) (layers := layers) repeats batch lam) := by
  let E := WeightSpace A H d layers
  let coords := PiLp.continuousLinearEquiv 2 ℝ (fun _ : Parameter A H d layers => ℝ)
  have hd : HasAnalyticLift (fun x : E => batchLoss repeats batch (WithLp.ofLp x)) :=
    (analyticLift_batchLoss repeats batch).comp (HasAnalyticLift.clm coords.toContinuousLinearMap)
  have hnorm : HasAnalyticLift (fun x : E => ‖x‖ ^ 2) :=
    HasAnalyticLift.of_analytic analyticAt_sq_norm
  exact hd.add ((HasAnalyticLift.const (lam / 2)).mul hnorm)

end Transformer.Grokking.FiniteRecovery.Argand
