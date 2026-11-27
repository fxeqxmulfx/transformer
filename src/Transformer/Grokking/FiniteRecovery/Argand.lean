module

public import Transformer.Grokking.FiniteRecovery.ArgandAttention

/-! The single-stream Argand forward pass from
experiments/phasor_value_transport/ARGAND.md, including skip and write gates,
split complex attention, shared layer repeats, and the tied token readout. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open Transformer.PhasorV2
open scoped BigOperators ComplexConjugate InnerProductSpace

variable {A P : Type*} {T c H d layers : ℕ}

/-- The learned linear forms in the skip and write gates of one sublayer. -/
structure Gates (c : ℕ) where
  skip : CSpace c →ₗ[ℂ] ℂ
  write : CSpace c →ₗ[ℂ] ℂ

/-- Both gates read the normalized input. The skip scales the original residual,
and the write scales the normalized branch output. -/
noncomputable def sublayer (gates : Gates c) (eps scale : ℝ)
    (branch : (P → CSpace c) → P → CSpace c)
    (x : P → CSpace c) : P → CSpace c :=
  let z := fun i => argandRMS eps (x i)
  fun i => (1 + gates.skip (z i)) • x i +
    ((scale : ℂ) * (1 + gates.write (z i))) • argandRMS eps (branch z i)

/-- Zero gate weights give the ordinary residual at initialization. -/
theorem sublayer_zero_gates (eps scale : ℝ)
    (branch : Word (CSpace c) T → Word (CSpace c) T) (x : Word (CSpace c) T) (i : Fin T) :
    sublayer ⟨0, 0⟩ eps scale branch x i =
      x i + (scale : ℂ) • argandRMS eps (branch (fun j => argandRMS eps (x j)) i) := by
  simp [sublayer]

/-- A single Argand layer shares a complex score multiplier across its heads. -/
structure Layer (H d : ℕ) where
  attentionProjections : Attention (H * d) H d
  feedForward : FFN (H * d) (4 * (H * d))
  scoreMultiplier : ℂ
  attentionGates : Gates (H * d)
  feedForwardGates : Gates (H * d)

/-- The attention sublayer is followed by the product feed-forward sublayer. -/
noncomputable def Layer.forwardWith [Fintype P] (layer : Layer H d) (eps scale : ℝ)
    (geometry : AttentionGeometry P d) (x : P → CSpace (H * d)) : P → CSpace (H * d) :=
  let afterAttention := sublayer layer.attentionGates eps scale
    (attentionWith geometry layer.attentionProjections eps layer.scoreMultiplier) x
  let f := layer.feedForward
  sublayer layer.feedForwardGates eps scale
    (fun z i => f.WF (f.Wg (z i) * f.Wu (z i))) afterAttention

noncomputable def Layer.forward (layer : Layer H d) (eps scale : ℝ) (alpha : Fin d → ℝ)
    (x : Word (CSpace (H * d)) T) : Word (CSpace (H * d)) T :=
  layer.forwardWith eps scale (causalGeometry alpha) x

/-- The same embedding table also supplies the output classifier. -/
structure Model (A : Type*) (H d layers : ℕ) where
  embedding : A → CSpace (H * d)
  layer : Fin layers → Layer H d
  alpha : Fin d → ℝ
  eps : ℝ

/-- Normalize the input embeddings, repeat the ordered layer stack with shared
weights, and normalize the final single residual stream. -/
noncomputable def Model.hiddenWith [Fintype P] (model : Model A H d layers) (repeats : ℕ)
    (geometry : AttentionGeometry P d) (tokens : P → A) : P → CSpace (H * d) :=
  let scale := 1 / Real.sqrt (2 * (layers : ℝ) * repeats)
  let pass := fun z : P → CSpace (H * d) =>
    (List.ofFn model.layer).foldl (fun x layer => layer.forwardWith model.eps scale geometry x) z
  let initial := fun i => argandRMS model.eps (model.embedding (tokens i))
  fun i => argandRMS model.eps ((pass^[repeats]) initial i)

noncomputable def Model.hidden (model : Model A H d layers) (repeats : ℕ)
    (tokens : Word A T) : Word (CSpace (H * d)) T :=
  model.hiddenWith repeats (causalGeometry model.alpha) tokens

/-- The tied output head takes half the real complex inner product. -/
noncomputable def Model.readout (model : Model A H d layers) (state : CSpace (H * d)) : A → ℝ :=
  fun token => (⟪model.embedding token, state⟫_ℂ).re / 2

noncomputable def Model.logits (model : Model A H d layers) (repeats : ℕ)
    (tokens : Word A T) (i : Fin T) : A → ℝ :=
  model.readout (model.hidden repeats tokens i)

noncomputable def Model.logitsWith [Fintype P] (model : Model A H d layers) (repeats : ℕ)
    (geometry : AttentionGeometry P d) (tokens : P → A) (i : P) : A → ℝ :=
  model.readout (model.hiddenWith repeats geometry tokens i)

/-- The distribution over the next token from the complete forward pass. -/
noncomputable def Model.probability [Fintype A] (model : Model A H d layers) (repeats : ℕ)
    (tokens : Word A T) (i : Fin T) (token : A) : ℝ :=
  let logits := model.logits repeats tokens i
  Real.exp (logits token) / ∑ a, Real.exp (logits a)

/-- Each possible output has positive probability under softmax. -/
theorem Model.probability_pos [Fintype A] (model : Model A H d layers) (repeats : ℕ)
    (tokens : Word A T) (i : Fin T) (token : A) :
    0 < model.probability repeats tokens i token := by
  unfold Model.probability
  apply div_pos (Real.exp_pos _)
  exact Finset.sum_pos' (fun _ _ => (Real.exp_pos _).le)
    ⟨token, Finset.mem_univ _, Real.exp_pos _⟩

/-- The complete Argand readout is a probability distribution. -/
theorem Model.sum_probability [Fintype A] [Nonempty A] (model : Model A H d layers)
    (repeats : ℕ) (tokens : Word A T) (i : Fin T) :
    ∑ token, model.probability repeats tokens i token = 1 := by
  unfold Model.probability
  rw [← Finset.sum_div]
  apply div_self
  obtain ⟨token⟩ := ‹Nonempty A›
  exact ne_of_gt (Finset.sum_pos' (fun _ _ => (Real.exp_pos _).le)
    ⟨token, Finset.mem_univ _, Real.exp_pos _⟩)

/-- The recovery loss supervises the next token after the last word position. -/
noncomputable def Model.labelLoss [Fintype A] (model : Model A H d layers) (repeats : ℕ)
    (word : Word A (T + 1)) (label : A) : ℝ :=
  -Real.log (model.probability repeats word (Fin.last T) label)

/-- Mean last-token cross-entropy on the observed graph. -/
noncomputable def Model.observedLoss [Fintype A] (model : Model A H d layers) (repeats : ℕ)
    (part : Finset (Word A (T + 1) × A)) : ℝ :=
  (∑ pair ∈ part, model.labelLoss repeats pair.1 pair.2) / part.card

/-- An individual loss is nonnegative because its softmax probability is at most one. -/
theorem Model.labelLoss_nonneg [Fintype A] [Nonempty A] (model : Model A H d layers)
    (repeats : ℕ) (word : Word A (T + 1)) (label : A) :
    0 ≤ model.labelLoss repeats word label := by
  unfold Model.labelLoss
  apply neg_nonneg.mpr
  apply Real.log_nonpos (model.probability_pos repeats word (Fin.last T) label).le
  calc
    model.probability repeats word (Fin.last T) label ≤
        ∑ a, model.probability repeats word (Fin.last T) a :=
      Finset.single_le_sum (fun a _ => (model.probability_pos repeats word (Fin.last T) a).le)
        (Finset.mem_univ label)
    _ = 1 := model.sum_probability repeats word (Fin.last T)

/-- On a fixed finite sample, the total observed loss controls each observation. -/
theorem Model.labelLoss_le_card_mul_observedLoss [Fintype A] [Nonempty A]
    (model : Model A H d layers) (repeats : ℕ) (part : Finset (Word A (T + 1) × A))
    {pair : Word A (T + 1) × A} (hp : pair ∈ part) :
    model.labelLoss repeats pair.1 pair.2 ≤ (part.card : ℝ) * model.observedLoss repeats part := by
  unfold Model.observedLoss
  have hcard : (part.card : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (Finset.card_pos.mpr ⟨pair, hp⟩).ne'
  rw [mul_div_cancel₀ _ hcard]
  exact Finset.single_le_sum (fun q _ => model.labelLoss_nonneg repeats q.1 q.2) hp

example (model : Model (Fin 2) 4 16 1) (word : Word (Fin 2) 2) :
    ∑ token, model.probability 1 word 1 token = 1 :=
  model.sum_probability 1 word 1

example (model : Model (Fin 2) 4 16 1) (word : Word (Fin 2) 2) :=
  model.labelLoss_le_card_mul_observedLoss 1 {(word, 0)} (pair := (word, 0)) (by simp)

end Transformer.Grokking.FiniteRecovery.Argand
