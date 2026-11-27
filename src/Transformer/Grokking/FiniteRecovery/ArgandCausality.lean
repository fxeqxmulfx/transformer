module

public import Transformer.Grokking.FiniteRecovery.Argand

/-! Prefix information in the full skip/write Argand. At position zero
the only legal key is the query itself, through every layer and pass. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open Transformer.PhasorV2

variable {A : Type*} {T H d layers : ℕ}

private theorem attention_first (projection : Attention (H * d) H d) (eps : ℝ)
    (multiplier : ℂ) (alpha : Fin d → ℝ) (x : Word (CSpace (H * d)) (T + 1)) :
    attention projection eps multiplier alpha x 0 =
      projection.WO (fun h m => (1 + Complex.I) * projection.WV (x 0) h m) := by
  have hmask (z : Word ℝ (T + 1)) (j : Fin (T + 1)) :
      maskedSoftmax ((causalGeometry alpha).visible 0) z j = if j = 0 then 1 else 0 := by
    have hp : (Finset.univ.filter fun k : Fin (T + 1) => k = 0) = {0} := by
      ext k
      simp
    simp only [maskedSoftmax, causalGeometry, decide_eq_true_eq, Fin.le_zero_iff, hp,
      Finset.sum_singleton]
    split_ifs with h
    · subst j
      exact div_self (Real.exp_ne_zero _)
    · rfl
  unfold attention attentionWith
  dsimp only
  apply congrArg projection.WO
  funext h m
  simp only [hmask]
  rw [Finset.sum_eq_single (0 : Fin (T + 1))]
  · have h : (⟨1, 1⟩ : ℂ) = 1 + Complex.I := Complex.ext (by norm_num) (by norm_num)
    simp [h]
  · intro j _ hj
    have hz : (⟨0, 0⟩ : ℂ) = 0 := Complex.ext rfl rfl
    simp [hj, hz]
  · simp

private theorem layer_first_congr (layer : Layer H d) (eps scale : ℝ) (alpha : Fin d → ℝ)
    {x y : Word (CSpace (H * d)) (T + 1)} (h : x 0 = y 0) :
    layer.forward eps scale alpha x 0 = layer.forward eps scale alpha y 0 := by
  have ha : sublayer layer.attentionGates eps scale
      (attention layer.attentionProjections eps layer.scoreMultiplier alpha) x 0 =
      sublayer layer.attentionGates eps scale
      (attention layer.attentionProjections eps layer.scoreMultiplier alpha) y 0 := by
    simp only [sublayer, attention_first, h]
  simpa only [Layer.forward, Layer.forwardWith, sublayer, attention] using
    congrArg (fun z : CSpace (H * d) =>
        (1 + layer.feedForwardGates.skip (argandRMS eps z)) • z +
          ((scale : ℂ) * (1 + layer.feedForwardGates.write (argandRMS eps z))) •
            argandRMS eps (layer.feedForward.WF
              (layer.feedForward.Wg (argandRMS eps z) * layer.feedForward.Wu (argandRMS eps z)))) ha

/-- At the first query the complete model only sees the first token.
This holds for all weights, including nonzero attention and FFN weights. -/
theorem Model.logits_first_eq_of_token_eq (model : Model A H d layers) (repeats : ℕ)
    {u v : Word A (T + 1)} (h : u 0 = v 0) (label : A) :
    model.logits repeats u 0 label = model.logits repeats v 0 label := by
  let slide (z : Word (CSpace (H * d)) (T + 1)) (l : Layer H d) :=
    l.forward model.eps (1 / Real.sqrt (2 * (layers : ℝ) * repeats)) model.alpha z
  have hfold (ls : List (Layer H d)) {x y : Word (CSpace (H * d)) (T + 1)}
      (hxy : x 0 = y 0) : ls.foldl slide x 0 = ls.foldl slide y 0 := by
    induction ls generalizing x y with
    | nil => exact hxy
    | cons l ls ih => exact ih (layer_first_congr _ _ _ _ hxy)
  let x := fun j => argandRMS model.eps (model.embedding (u j))
  let y := fun j => argandRMS model.eps (model.embedding (v j))
  let pass := (List.ofFn model.layer).foldl slide
  have hpass (n : ℕ) : (pass^[n]) x 0 = (pass^[n]) y 0 := by
    induction n with
    | zero => simp only [Function.iterate_zero_apply, x, y, h]
    | succ n ih =>
      simp only [Function.iterate_succ_apply']
      exact hfold _ ih
  change model.readout (argandRMS model.eps ((pass^[repeats]) x 0)) label =
    model.readout (argandRMS model.eps ((pass^[repeats]) y 0)) label
  rw [hpass]

/-- With no layer passes, the final readout depends only on the last token. -/
theorem Model.logits_no_passes_eq_of_token_eq (model : Model A H d layers)
    {u v : Word A T} {i : Fin T} (h : u i = v i) (label : A) :
    model.logits 0 u i label = model.logits 0 v i label := by
  simp [Model.logits, Model.hidden, Model.hiddenWith, h]

example (model : Model Bool 4 16 2) (v : Word Bool 32) :=
  model.logits_first_eq_of_token_eq 2
    (u := fun _ : Fin 32 => v 0) (v := v) rfl true

end Transformer.Grokking.FiniteRecovery.Argand
