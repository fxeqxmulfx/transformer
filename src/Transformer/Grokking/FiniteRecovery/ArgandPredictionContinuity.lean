module

public import Transformer.Grokking.FiniteRecovery.ArgandForwardContinuity

/-! The repeated shared layer stack and tied softmax retain continuity
in the full parameter vector. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open Transformer.PhasorV2
open scoped InnerProductSpace

variable {A : Type*} {H d layers T : ℕ}

private theorem continuous_hidden_fromWeights (repeats : ℕ) (word : Word A T) (i : Fin T) :
    Continuous (fun w : Weights A H d layers => (Model.fromWeights w).hidden repeats word i) := by
  let W := Weights A H d layers
  let ε : ℝ := 1 / 1000000
  let scale := 1 / Real.sqrt (2 * (layers : ℝ) * repeats)
  let x (w : W) (j : Fin T) := argandRMS ε ((Model.fromWeights w).embedding (word j))
  let slide (w : W) (z : Word (CSpace (H * d)) T) (l : Fin layers) :=
    ((Model.fromWeights w).layer l).forwardWith ε scale (causalGeometry (Model.fromWeights w).alpha) z
  let pass (w : W) (z : Word (CSpace (H * d)) T) := (List.finRange layers).foldl (slide w) z
  have hx (j : Fin T) : Continuous (fun w => x w j) :=
    continuous_argandRMS (by norm_num : 0 < ε) (continuous_embedding_fromWeights (word j))
  have hpass (ls : List (Fin layers)) (z : W → Word (CSpace (H * d)) T)
      (hz : ∀ j, Continuous (fun w => z w j)) (j : Fin T) :
      Continuous (fun w => ls.foldl (slide w) (z w) j) := by
    induction ls generalizing z with
    | nil => exact hz j
    | cons l ls ih =>
      simp only [List.foldl_cons]
      exact ih (fun w => slide w (z w) l) (continuous_forward_fromWeights l scale z hz)
  have hrepeat (n : ℕ) (j : Fin T) : Continuous (fun w => ((pass w)^[n]) (x w) j) := by
    induction n generalizing j with
    | zero => exact hx j
    | succ n ih =>
      simp only [Function.iterate_succ_apply']
      exact hpass _ _ ih j
  simp only [Model.hidden, Model.hiddenWith, List.ofFn_eq_map, List.foldl_map]
  change Continuous (fun w : W => argandRMS ε ((pass w)^[repeats] (x w) i))
  exact continuous_argandRMS (by norm_num : 0 < ε) (hrepeat repeats i)

theorem continuous_logits_fromWeights (repeats : ℕ) (word : Word A T) (i : Fin T) (a : A) :
    Continuous (fun w : Weights A H d layers => (Model.fromWeights w).logits repeats word i a) := by
  exact (Complex.continuous_re.comp ((continuous_embedding_fromWeights a).inner (𝕜 := ℂ)
    (continuous_hidden_fromWeights repeats word i))).div_const 2

theorem continuous_labelLoss_fromWeights [Fintype A] (repeats : ℕ) (word : Word A T) (i : Fin T) (a : A) :
    Continuous (fun w : Weights A H d layers =>
      -Real.log ((Model.fromWeights w).probability repeats word i a)) := by
  have hp (t : A) := (continuous_logits_fromWeights (H := H) (d := d) (layers := layers) repeats word i t).rexp
  have hsum (w : Weights A H d layers) : 0 < ∑ t, Real.exp ((Model.fromWeights w).logits repeats word i t) :=
    Finset.sum_pos' (fun _ _ => (Real.exp_pos _).le) ⟨a, Finset.mem_univ _, Real.exp_pos _⟩
  have hprob : Continuous (fun w : Weights A H d layers => (Model.fromWeights w).probability repeats word i a) :=
    (hp a).div (continuous_finsetSum _ fun t _ => hp t) (fun w => (hsum w).ne')
  exact (hprob.log fun w => ((Model.fromWeights w).probability_pos repeats word i a).ne').neg

example := continuous_labelLoss_fromWeights (H := 4) (d := 16) (layers := 2) 2
  (fun _ : Fin 32 => (0 : Fin 256)) 0 1

end Transformer.Grokking.FiniteRecovery.Argand
