import Transformer.GPTMini.Sparsemax.ObservedMemoryPreferences
import Transformer.GPTMini.Sparsemax.IncidentNearestMemory

/-!
# Causal text-derived geometry selection

Derived memory architecture before arXiv:1602.02068v2, Eq. (1). Masked
observed prefixes supply both adjacent Hamming distances and fixed data
feature energies for the additional geometry criterion. Hidden prototype
continuations cannot affect the reference, its unique selected parameters
or the actual joint prediction pipeline, including the input code table.

Distinct registered observations retain arbitrary vector-output fitting
with the selected geometry and one global common value table. Structural
bounds supply actual embeddings and an inverse, not a desired-attention
factorization assumed as input. No text-target regularity or unseen coverage
is inferred from these statements; the earlier conditional bounds remain.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

/-- Build the data geometry reference from visible prefix records only.
Source: the causal observation criterion preceding arXiv:1602.02068v2, Eq. (1). -/
def prefixObservedMemoryReference {T V N Q K : ℕ}
    (prototypes : Fin (N + 1) → Fin T → Fin V) (rows : Fin (N + 1) → Fin T)
    (queryFeatures : (Fin T → Option (Fin V)) → Fin Q → ℝ)
    (keyFeatures : (Fin T → Option (Fin V)) → Fin K → ℝ) (scale : ℝ) : LocalMemoryParameters N :=
  observedMemoryReference prefixSignatureDistance
    (fun j => causalPrefixSignature (prototypes j) (rows j)) queryFeatures keyFeatures scale

/-- Positive scale gives positive preferences from the actual nonnegative Hamming distances.
Source: the causal data-derived criterion for arXiv:1602.02068v2, Eq. (1). -/
theorem prefixObservedMemoryReference_edge_pos {T V N Q K : ℕ}
    (prototypes : Fin (N + 1) → Fin T → Fin V) (rows : Fin (N + 1) → Fin T)
    (queryFeatures : (Fin T → Option (Fin V)) → Fin Q → ℝ)
    (keyFeatures : (Fin T → Option (Fin V)) → Fin K → ℝ) (scale : ℝ)
    (hs : 0 < scale) (e : Fin N) :
    0 < (prefixObservedMemoryReference prototypes rows queryFeatures keyFeatures scale).1 e :=
  observedMemoryReference_edge_pos _ _ queryFeatures keyFeatures scale hs e
    (prefixSignatureDistance_nonneg _ _)

/-- Two actual different text observations inhabit positive-scale preferences. -/
example : 0 < (prefixObservedMemoryReference (fun j : Fin 2 => fun _ : Fin 1 => j) (fun _ => 0)
    (fun x => fun _ : Fin 1 => if x 0 = some 1 then (1 : ℝ) else 0)
    (fun x => fun _ : Fin 1 => if x 0 = some 1 then (1 / 2 : ℝ) else 0) (1 / 8)).1 0 :=
  prefixObservedMemoryReference_edge_pos _ _ _ _ _ (by norm_num) 0

/-- Changing prototype tokens beyond their observed prefixes leaves the complete reference unchanged.
Source: masked observation signatures preceding arXiv:1602.02068v2, Eq. (1). -/
theorem prefixObservedMemoryReference_causal {T V N Q K : ℕ}
    (prototypes other : Fin (N + 1) → Fin T → Fin V) (rows : Fin (N + 1) → Fin T)
    (queryFeatures : (Fin T → Option (Fin V)) → Fin Q → ℝ)
    (keyFeatures : (Fin T → Option (Fin V)) → Fin K → ℝ) (scale : ℝ)
    (h : ∀ r j, j ≤ rows r → prototypes r j = other r j) :
    prefixObservedMemoryReference prototypes rows queryFeatures keyFeatures scale =
      prefixObservedMemoryReference other rows queryFeatures keyFeatures scale := by
  have he : (fun r => causalPrefixSignature (prototypes r) (rows r)) =
      (fun r => causalPrefixSignature (other r) (rows r)) := by
    funext r
    exact (causalPrefixSignature_eq_iff _ _ _ _).2 ⟨rfl, h r⟩
  unfold prefixObservedMemoryReference
  rw [he]

/-- An actual change of hidden continuation inhabits full-reference causality. -/
example : prefixObservedMemoryReference (fun r : Fin 2 => fun _ : Fin 2 => r) (fun _ => 0)
    (fun x => fun j : Fin 2 => if x j = some 1 then (1 : ℝ) else 0)
    (fun x => fun j : Fin 2 => if x j = some 1 then (1 / 2 : ℝ) else 0) (1 / 8) =
    prefixObservedMemoryReference (fun r : Fin 2 => fun j : Fin 2 => if j = 0 then r else 1)
      (fun _ => 0) (fun x => fun j : Fin 2 => if x j = some 1 then (1 : ℝ) else 0)
      (fun x => fun j : Fin 2 => if x j = some 1 then (1 / 2 : ℝ) else 0) (1 / 8) := by
  apply prefixObservedMemoryReference_causal
  intro r j hj
  fin_cases j
  · rfl
  · norm_num at hj

/-- The uniquely selected compact geometry also ignores hidden prototype continuations.
Source: proved reference causality and constrained minimization for arXiv:1602.02068v2, Eq. (1). -/
theorem prefixObservedMemorySelected_causal {T V N Q K : ℕ}
    (prototypes other : Fin (N + 1) → Fin T → Fin V) (rows : Fin (N + 1) → Fin T)
    (queryFeatures : (Fin T → Option (Fin V)) → Fin Q → ℝ)
    (keyFeatures : (Fin T → Option (Fin V)) → Fin K → ℝ) (scale cap floor : ℝ)
    (hc : 1 ≤ cap) (hf : floor ≤ 1)
    (h : ∀ r j, j ≤ rows r → prototypes r j = other r j) :
    incidentMemorySelectedParameters cap floor
      (prefixObservedMemoryReference prototypes rows queryFeatures keyFeatures scale) hc hf =
    incidentMemorySelectedParameters cap floor
      (prefixObservedMemoryReference other rows queryFeatures keyFeatures scale) hc hf := by
  rw [prefixObservedMemoryReference_causal prototypes other rows queryFeatures keyFeatures scale h]

/-- The same hidden continuation change and valid domain bounds inhabit selection causality. -/
example : incidentMemorySelectedParameters 4 (3 / 4)
    (prefixObservedMemoryReference (fun r : Fin 2 => fun _ : Fin 2 => r) (fun _ => 0)
      (fun x => fun j : Fin 2 => if x j = some 1 then (1 : ℝ) else 0)
      (fun x => fun j : Fin 2 => if x j = some 1 then (1 / 2 : ℝ) else 0) (1 / 8))
      (by norm_num) (by norm_num) =
    incidentMemorySelectedParameters 4 (3 / 4)
      (prefixObservedMemoryReference (fun r : Fin 2 => fun j : Fin 2 => if j = 0 then r else 1)
        (fun _ => 0) (fun x => fun j : Fin 2 => if x j = some 1 then (1 : ℝ) else 0)
        (fun x => fun j : Fin 2 => if x j = some 1 then (1 / 2 : ℝ) else 0) (1 / 8))
        (by norm_num) (by norm_num) := by
  apply prefixObservedMemorySelected_causal
  intro r j hj
  fin_cases j
  · rfl
  · norm_num at hj

/-- Both the selected geometry and actual nearest code pipeline ignore hidden prototype tokens.
Source: the data-derived memory before arXiv:1602.02068v2, Eq. (1), with common original values. -/
theorem prefixObservedMemoryForward_prototypes_causal {R T V N D Q K : ℕ}
    (queries : Fin R → Fin T → Fin V) (queryRows : Fin R → Fin T)
    (prototypes other : Fin (N + 1) → Fin T → Fin V) (rows : Fin (N + 1) → Fin T)
    (queryFeatures : (Fin T → Option (Fin V)) → Fin Q → ℝ)
    (keyFeatures : (Fin T → Option (Fin V)) → Fin K → ℝ) (scale cap floor : ℝ)
    (Z : Matrix (Fin (N + 1)) (Fin D) ℝ) (hc : 1 ≤ cap) (hf : floor ≤ 1)
    (h : ∀ r j, j ≤ rows r → prototypes r j = other r j) :
    localJointMemoryForward (prefixNearestCodes queries queryRows prototypes rows)
      (incidentMemorySelectedParameters cap floor
        (prefixObservedMemoryReference prototypes rows queryFeatures keyFeatures scale) hc hf, Z) =
    localJointMemoryForward (prefixNearestCodes queries queryRows other rows)
      (incidentMemorySelectedParameters cap floor
        (prefixObservedMemoryReference other rows queryFeatures keyFeatures scale) hc hf, Z) := by
  rw [prefixNearestCodes_prototypes_causal queries queryRows prototypes other rows h,
    prefixObservedMemorySelected_causal prototypes other rows queryFeatures keyFeatures scale cap floor hc hf h]

/-- A real hidden continuation change and arbitrary shared outputs inhabit pipeline causality. -/
example (Z : Matrix (Fin 2) (Fin 1) ℝ) :
    localJointMemoryForward (prefixNearestCodes (fun _ : Fin 1 => fun _ : Fin 2 => (0 : Fin 2))
      (fun _ => 0) (fun r : Fin 2 => fun _ : Fin 2 => r) (fun _ => 0))
      (incidentMemorySelectedParameters 4 (3 / 4)
        (prefixObservedMemoryReference (fun r : Fin 2 => fun _ : Fin 2 => r) (fun _ => 0)
          (fun x => fun j : Fin 2 => if x j = some 1 then (1 : ℝ) else 0)
          (fun x => fun j : Fin 2 => if x j = some 1 then (1 / 2 : ℝ) else 0) (1 / 8))
          (by norm_num) (by norm_num), Z) =
    localJointMemoryForward (prefixNearestCodes (fun _ : Fin 1 => fun _ : Fin 2 => (0 : Fin 2))
      (fun _ => 0) (fun r : Fin 2 => fun j : Fin 2 => if j = 0 then r else 1) (fun _ => 0))
      (incidentMemorySelectedParameters 4 (3 / 4)
        (prefixObservedMemoryReference (fun r : Fin 2 => fun j : Fin 2 => if j = 0 then r else 1)
          (fun _ => 0) (fun x => fun j : Fin 2 => if x j = some 1 then (1 : ℝ) else 0)
          (fun x => fun j : Fin 2 => if x j = some 1 then (1 / 2 : ℝ) else 0) (1 / 8))
          (by norm_num) (by norm_num), Z) := by
  apply prefixObservedMemoryForward_prototypes_causal
  intro r j hj
  fin_cases j
  · rfl
  · norm_num at hj

/-- The selected geometry fits arbitrary registered causal targets with one common value table.
Source: nearest prototype registration and the actual arXiv:1602.02068v2, Eq. (1) inverse chart. -/
theorem prefixObservedMemoryForward_registered {T V N D Q K : ℕ}
    (prototypes : Fin (N + 1) → Fin T → Fin V) (rows : Fin (N + 1) → Fin T)
    (queryFeatures : (Fin T → Option (Fin V)) → Fin Q → ℝ)
    (keyFeatures : (Fin T → Option (Fin V)) → Fin K → ℝ) (scale cap floor : ℝ)
    (Y : Matrix (Fin (N + 1)) (Fin D) ℝ) (hc : 1 ≤ cap) (hf : 1 / 2 < floor) (hf1 : floor ≤ 1)
    (hs : Function.Injective (fun j => causalPrefixSignature (prototypes j) (rows j))) :
    localJointMemoryForward (prefixNearestCodes prototypes rows prototypes rows)
      (incidentMemorySelectedParameters cap floor
        (prefixObservedMemoryReference prototypes rows queryFeatures keyFeatures scale) hc hf1, Y) = Y :=
  incidentPrefixNearestForward_registered cap floor _ prototypes rows Y hf
    (incidentMemorySelectedParameters_mem cap floor _ hc hf1) hs

/-- Distinct visible tokens, nonconstant features and nonconstant targets inhabit selected fitting. -/
example : localJointMemoryForward
    (prefixNearestCodes (fun r : Fin 2 => fun _ : Fin 1 => r) (fun _ => 0)
      (fun r : Fin 2 => fun _ : Fin 1 => r) (fun _ => 0))
    (incidentMemorySelectedParameters 4 (3 / 4)
      (prefixObservedMemoryReference (fun r : Fin 2 => fun _ : Fin 1 => r) (fun _ => 0)
        (fun x => fun _ : Fin 1 => if x 0 = some 1 then (1 : ℝ) else 0)
        (fun x => fun _ : Fin 1 => if x 0 = some 1 then (1 / 2 : ℝ) else 0) (1 / 8))
        (by norm_num) (by norm_num), Matrix.of (fun j : Fin 2 => fun _ : Fin 1 => (j.val : ℝ))) =
      Matrix.of (fun j : Fin 2 => fun _ : Fin 1 => (j.val : ℝ)) := by
  apply prefixObservedMemoryForward_registered _ _ _ _ _ _ _ _ (by norm_num) (by norm_num) (by norm_num)
  intro i j h
  exact ((causalPrefixSignature_eq_iff _ _ _ _).1 h).2 0 (by norm_num)

end Transformer.GPTMini.Sparsemax
