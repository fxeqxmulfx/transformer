import Transformer.GPTMini.Convex.Structured.TensorObservations

/-!
# Full learned inference and likelihood follow actual tensor observations

Source: the genuine state/pointer heads at 5bdff3c, complete tensor
probability/training at 815556c and TensorObservations' actual field
identities. Each chronological transition matrix and every all-pair
query/key/value/bias input agrees on observation-equivalent rows.

The actual compact state propagation, channel means, mixed output,
computed complete NLL and whole normalized joint probability therefore
agree, even when residual code coordinates and RMS multipliers differ.
No semantic state, desired distribution or successful prediction is a
premise. This transfers a derived low-level residual preservation law
to complete computations; arbitrary learned head-global weights remain
unchanged and free. Full stack induction is a subsequent obligation.
-/

namespace Transformer.GPTMini.Convex.Structured

open scoped BigOperators Classical
noncomputable section

variable {C d T : ℕ}

/-- Actual chronological state/value expectations depend only on recovered head observations.
Source: equality of every true transition matrix inside the physical List.ofFn chronological fold. -/
theorem tensorStateMean_fieldsSame (hwidth : 64 ≤ d) (ψ : TensorHeadParameters C)
    (x y : Fin T → EucSpace d) (hfields : ∀ position, tensorFieldsSame hwidth (x position) (y position))
    (group : Fin 5) (code : Fin 4 → ℝ) :
    tensorStateMean hwidth ψ x group code = tensorStateMean hwidth ψ y group code := by
  have hstep : (fun current position => markovAdvance (tensorTransition hwidth (x position)) current) =
      (fun current position => markovAdvance (tensorTransition hwidth (y position)) current) := by
    funext current position
    rw [tensorTransition_fieldsSame hwidth _ _ (hfields position)]
  unfold tensorStateMean markovValueMean markovRun markovPropagate
  rw [List.ofFn_eq_map, List.ofFn_eq_map, List.foldl_map, List.foldl_map, hstep]

example : (64 : ℕ) ≤ 64 ∧ (∀ p : Fin 3, tensorFieldsSame (d := 64) (by omega)
    ((fun _ : Fin 3 => (0 : EucSpace 64)) p) 0) :=
  ⟨by omega, fun _ => tensorFieldsSame_refl _ _⟩

/-- Every actually inferred signed-axis state value is unchanged on equivalent head observations.
Source: both genuine small-channel expectations and the actual even/odd decoder coordinate layout. -/
theorem tensorStateCoordinates_fieldsSame (hwidth : 64 ≤ d) (ψ : TensorHeadParameters C)
    (x y : Fin T → EucSpace d) (hfields : ∀ position, tensorFieldsSame hwidth (x position) (y position)) :
    tensorStateCoordinates hwidth ψ x = tensorStateCoordinates hwidth ψ y := by
  funext axis
  simp only [tensorStateCoordinates, tensorStateMean_fieldsSame hwidth ψ x y hfields]

example : (64 : ℕ) ≤ 128 ∧ (∀ p : Fin 3, tensorFieldsSame (d := 128) (by omega)
    ((fun _ : Fin 3 => (0 : EucSpace 128)) p) 0) :=
  ⟨by omega, fun _ => tensorFieldsSame_refl _ _⟩

/-- Actual all-pair pointer channel means agree at every unchanged physical query.
Source: all recovered Q/K/value arrays and the complete freely learned positional pair bias. -/
theorem tensorPointerCoordinates_fieldsSame (hwidth : 64 ≤ d) (ψ : TensorHeadParameters C)
    (x y : Fin T → EucSpace d) (hcap : T ≤ C) (query : Fin T)
    (hfields : ∀ position, tensorFieldsSame hwidth (x position) (y position)) :
    tensorPointerCoordinates hwidth ψ x hcap query = tensorPointerCoordinates hwidth ψ y hcap query := by
  unfold tensorPointerCoordinates
  simp_rw [tensorQuery_fieldsSame hwidth _ _ (hfields query),
    tensorKey_fieldsSame hwidth _ _ (hfields _), tensorValue_fieldsSame hwidth _ _ (hfields _)]
  rw [tensorBindingBias_fieldsSame hwidth ψ x y hcap hfields]

example : (64 : ℕ) ≤ 64 ∧ (3 : ℕ) ≤ 64 ∧ (∀ p : Fin 3, tensorFieldsSame (d := 64) (by omega)
    ((fun _ : Fin 3 => (0 : EucSpace 64)) p) 0) :=
  ⟨by omega, by omega, fun _ => tensorFieldsSame_refl _ _⟩

/-- The actual full mixture's ten-coordinate output depends only on true preserved input observations.
Source: both genuine compact branches and the same freely learned global mixture weights. -/
theorem tensorMixedCoordinates_fieldsSame (hwidth : 64 ≤ d) (ψ : TensorHeadParameters C)
    (x y : Fin T → EucSpace d) (hcap : T ≤ C) (query : Fin T)
    (hfields : ∀ position, tensorFieldsSame hwidth (x position) (y position)) :
    tensorMixedCoordinates hwidth ψ x hcap query = tensorMixedCoordinates hwidth ψ y hcap query := by
  funext axis
  simp only [tensorMixedCoordinates, tensorStateCoordinates_fieldsSame hwidth ψ x y hfields,
    tensorPointerCoordinates_fieldsSame hwidth ψ x y hcap query hfields]

example : (64 : ℕ) ≤ 128 ∧ (3 : ℕ) ≤ 64 ∧ (∀ p : Fin 3, tensorFieldsSame (d := 128) (by omega)
    ((fun _ : Fin 3 => (0 : EucSpace 128)) p) 0) :=
  ⟨by omega, by omega, fun _ => tensorFieldsSame_refl _ _⟩

/-- Actual computed complete state/path/channel likelihood is identical on preserved head observations.
Source: every genuine observed transition row, unchanged initial/value logits and fixed external path labels. -/
theorem tensorStateNLL_fieldsSame (hwidth : 64 ≤ d) (ψ : TensorHeadParameters C)
    (x y : Fin T → EucSpace d) (hfields : ∀ position, tensorFieldsSame hwidth (x position) (y position))
    (observed : MarkovConfiguration (Fin 6) (Fin 5) (Fin 4) T) :
    tensorStateNLL hwidth ψ x observed = tensorStateNLL hwidth ψ y observed := by
  unfold tensorStateNLL
  simp_rw [tensorTransition_fieldsSame hwidth _ _ (hfields _)]

example : (64 : ℕ) ≤ 64 ∧ (∀ p : Fin 3, tensorFieldsSame (d := 64) (by omega)
    ((fun _ : Fin 3 => (0 : EucSpace 64)) p) 0) :=
  ⟨by omega, fun _ => tensorFieldsSame_refl _ _⟩

/-- The true contracted all-pair normalizer and observed energy yield identical likelihoods after preserved observations.
Source: actual Q/K/value and learned position reads in the computed tensorPointerNLL. -/
theorem tensorPointerNLL_fieldsSame (hwidth : 64 ≤ d) (ψ : TensorHeadParameters C)
    (x y : Fin T → EucSpace d) (hcap : T ≤ C) (query : Fin T)
    (hfields : ∀ position, tensorFieldsSame hwidth (x position) (y position))
    (observed : SharedPointerConfiguration (Fin T × Fin T)) :
    tensorPointerNLL hwidth ψ x hcap query observed = tensorPointerNLL hwidth ψ y hcap query observed := by
  unfold tensorPointerNLL
  simp_rw [tensorQuery_fieldsSame hwidth _ _ (hfields query),
    tensorKey_fieldsSame hwidth _ _ (hfields _), tensorValue_fieldsSame hwidth _ _ (hfields _)]
  rw [tensorBindingBias_fieldsSame hwidth ψ x y hcap hfields]

example : (64 : ℕ) ≤ 128 ∧ (3 : ℕ) ≤ 64 ∧ (∀ p : Fin 3, tensorFieldsSame (d := 128) (by omega)
    ((fun _ : Fin 3 => (0 : EucSpace 128)) p) 0) :=
  ⟨by omega, by omega, fun _ => tensorFieldsSame_refl _ _⟩

/-- The entire actual complete training computation remains unchanged by intermediate code-only residual updates.
Source: both true branch likelihood identities, retaining the learned mixture row and all observed labels. -/
theorem tensorMixedNLL_fieldsSame (hwidth : 64 ≤ d) (ψ : TensorHeadParameters C)
    (x y : Fin T → EucSpace d) (hcap : T ≤ C) (query : Fin T)
    (hfields : ∀ position, tensorFieldsSame hwidth (x position) (y position)) (observed : TensorConfiguration T) :
    tensorMixedNLL hwidth ψ x hcap query observed = tensorMixedNLL hwidth ψ y hcap query observed := by
  cases observed with
  | inl state => simp only [tensorMixedNLL, tensorStateNLL_fieldsSame hwidth ψ x y hfields]
  | inr route => simp only [tensorMixedNLL, tensorPointerNLL_fieldsSame hwidth ψ x y hcap query hfields]

example : (64 : ℕ) ≤ 64 ∧ (3 : ℕ) ≤ 64 ∧ (∀ p : Fin 3, tensorFieldsSame (d := 64) (by omega)
    ((fun _ : Fin 3 => (0 : EucSpace 64)) p) 0) :=
  ⟨by omega, by omega, fun _ => tensorFieldsSame_refl _ _⟩

/-- The actual full tensor joint probability is unchanged together with inference and computed training.
Source: every physical state-path probability or all-pair Gibbs factor reads only the proved identical real observations. -/
theorem tensorMixedProbability_fieldsSame (hwidth : 64 ≤ d) (ψ : TensorHeadParameters C)
    (x y : Fin T → EucSpace d) (hcap : T ≤ C) (query : Fin T)
    (hfields : ∀ position, tensorFieldsSame hwidth (x position) (y position)) :
    tensorMixedProbability hwidth ψ x hcap query = tensorMixedProbability hwidth ψ y hcap query := by
  funext observed
  cases observed with
  | inl state =>
      unfold tensorMixedProbability tensorStateProbability
      simp_rw [tensorTransition_fieldsSame hwidth _ _ (hfields _)]
  | inr route =>
      unfold tensorMixedProbability tensorPointerProbability
      simp_rw [tensorQuery_fieldsSame hwidth _ _ (hfields query),
        tensorKey_fieldsSame hwidth _ _ (hfields _), tensorValue_fieldsSame hwidth _ _ (hfields _)]
      rw [tensorBindingBias_fieldsSame hwidth ψ x y hcap hfields]

example : (64 : ℕ) ≤ 128 ∧ (3 : ℕ) ≤ 64 ∧ (∀ p : Fin 3, tensorFieldsSame (d := 128) (by omega)
    ((fun _ : Fin 3 => (0 : EucSpace 128)) p) 0) :=
  ⟨by omega, by omega, fun _ => tensorFieldsSame_refl _ _⟩

end
end Transformer.GPTMini.Convex.Structured
