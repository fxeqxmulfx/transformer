import Transformer.GPTMini.Sparsemax.CodeMemoryOutputs
import Mathlib.Data.Finset.Max

/-!
# Sparse data codes from nearest registered observations

Derived memory input architecture for arXiv:1602.02068v2, Eq. (1).
A finite nonempty prototype table and a fixed data distance select one
nearest observation. Its one-hot code mixes learned query embeddings;
actual attention remains the selected row of the learned memory Gram.

The encoder has no output targets and no positive constant kernel term.
Exact prototype registration follows from a separating nonnegative data
distance, rather than a full-rank feature premise. Distinct registered
observations therefore retain arbitrary-target interpolation. Unseen
queries use a genuine nearest-prototype prediction; regularity assumptions
on their targets will be stated separately, rather than inferred from fit.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open scoped BigOperators

/-- A nearest prototype exists because the parameter dictionary is finite and nonempty.
Source: the derived sparse input encoder before arXiv:1602.02068v2, Eq. (1).
Ties are resolved by one fixed choice independent of output targets. -/
def nearestPrototype {Key : Type*} {N : ℕ} (distance : Key → Key → ℝ)
    (prototypes : Fin (N + 1) → Key) (query : Key) : Fin (N + 1) :=
  Classical.choose (Finset.exists_min_image Finset.univ
    (fun j => distance query (prototypes j)) ⟨0, Finset.mem_univ 0⟩)

/-- The actual selected distance is no greater than any registered alternative.
Source: the finite minimum in the derived arXiv:1602.02068v2, Eq. (1) encoder. -/
theorem nearestPrototype_min {Key : Type*} {N : ℕ} (distance : Key → Key → ℝ)
    (prototypes : Fin (N + 1) → Key) (query : Key) (j : Fin (N + 1)) :
    distance query (prototypes (nearestPrototype distance prototypes query)) ≤
      distance query (prototypes j) := by
  exact (Classical.choose_spec (Finset.exists_min_image Finset.univ
    (fun k => distance query (prototypes k)) ⟨0, Finset.mem_univ 0⟩)).2 j
    (Finset.mem_univ j)

/-- A separating nonnegative distance recovers every distinct registered observation.
Source: the data-only nearest encoder for arXiv:1602.02068v2, Eq. (1).
All distance and distinctness premises concern observations, not desired outputs. -/
theorem nearestPrototype_self {Key : Type*} {N : ℕ} (distance : Key → Key → ℝ)
    (prototypes : Fin (N + 1) → Key) (hn : ∀ x y, 0 ≤ distance x y)
    (hz : ∀ x, distance x x = 0) (hs : ∀ x y, distance x y = 0 → x = y)
    (hi : Function.Injective prototypes) (j : Fin (N + 1)) :
    nearestPrototype distance prototypes (prototypes j) = j := by
  have hm := nearestPrototype_min distance prototypes (prototypes j) j
  rw [hz] at hm
  have he := le_antisymm hm (hn _ _)
  exact hi (hs _ _ he).symm

/-- Two distinct observations and their discrete distance inhabit every registration premise. -/
example : nearestPrototype (fun x y : Fin 2 => if x = y then (0 : ℝ) else 1)
    (fun j : Fin 2 => j) 1 = 1 := by
  apply nearestPrototype_self
  · intro x y
    split_ifs <;> norm_num
  · intro x
    simp
  · intro x y h
    by_contra he
    simp [he] at h
  · intro x y h
    exact h

/-- One nearest registered observation gives a sparse probability data code.
Source: the derived categorical input before arXiv:1602.02068v2, Eq. (1). -/
def nearestPrototypeCodes {Key : Type*} {R N : ℕ} (distance : Key → Key → ℝ)
    (prototypes : Fin (N + 1) → Key) (queries : Fin R → Key) :
    Matrix (Fin R) (Fin (N + 1)) ℝ :=
  oneHotContextCodes (fun r => nearestPrototype distance prototypes (queries r))

/-- Every query, including an unseen one, supplies a valid probability mixture.
Source: categorical normalization in the derived arXiv:1602.02068v2, Eq. (1) encoder. -/
theorem nearestPrototypeCodes_mem {Key : Type*} {R N : ℕ} (distance : Key → Key → ℝ)
    (prototypes : Fin (N + 1) → Key) (queries : Fin R → Key) :
    nearestPrototypeCodes distance prototypes queries ∈ contextCodeDomain R N :=
  oneHotContextCodes_mem _

/-- All slots other than the selected observation have exactly zero input-code mass.
Source: the sparse categorical input to arXiv:1602.02068v2, Eq. (1). -/
theorem nearestPrototypeCodes_zero {Key : Type*} {R N : ℕ} (distance : Key → Key → ℝ)
    (prototypes : Fin (N + 1) → Key) (queries : Fin R → Key) (r : Fin R)
    (j : Fin (N + 1)) (hj : nearestPrototype distance prototypes (queries r) ≠ j) :
    nearestPrototypeCodes distance prototypes queries r j = 0 := by
  change (if nearestPrototype distance prototypes (queries r) = j then (1 : ℝ) else 0) = 0
  exact ite_eq_right hj

/-- A genuinely unselected slot inhabits the sparse-code premise. -/
example : nearestPrototypeCodes (fun x y : Fin 2 => if x = y then (0 : ℝ) else 1)
    (fun j : Fin 2 => j) (fun _ : Fin 1 => (0 : Fin 2)) 0 1 = 0 := by
  apply nearestPrototypeCodes_zero
  have hm := nearestPrototype_min (fun x y : Fin 2 => if x = y then (0 : ℝ) else 1)
    (fun j : Fin 2 => j) 0 0
  intro h
  rw [h] at hm
  norm_num at hm

/-- Prototype training codes are identity, with no data-kernel inverse or conditioning cost.
Source: proved nearest registration before arXiv:1602.02068v2, Eq. (1). -/
theorem nearestPrototypeCodes_identity {Key : Type*} {N : ℕ} (distance : Key → Key → ℝ)
    (prototypes : Fin (N + 1) → Key) (hn : ∀ x y, 0 ≤ distance x y)
    (hz : ∀ x, distance x x = 0) (hs : ∀ x y, distance x y = 0 → x = y)
    (hi : Function.Injective prototypes) : nearestPrototypeCodes distance prototypes prototypes = 1 := by
  ext r j
  change (if nearestPrototype distance prototypes (prototypes r) = j then (1 : ℝ) else 0) =
    (1 : Matrix (Fin (N + 1)) (Fin (N + 1)) ℝ) r j
  rw [nearestPrototype_self distance prototypes hn hz hs hi, Matrix.one_apply]

/-- Concrete distinct prototypes satisfy every identity-code premise. -/
example : nearestPrototypeCodes (fun x y : Fin 2 => if x = y then (0 : ℝ) else 1)
    (fun j : Fin 2 => j) (fun j : Fin 2 => j) = 1 := by
  apply nearestPrototypeCodes_identity
  · intro x y
    split_ifs <;> norm_num
  · intro x
    simp
  · intro x y h
    by_contra he
    simp [he] at h
  · intro x y h
    exact h

/-- Actual context attention selects a learned memory row, retaining its trainable support.
Source: variational arXiv:1602.02068v2, Eq. (1), on the structural memory domain. -/
theorem contextNearestAttention_row {Key : Type*} {R N : ℕ} (cap floor : ℝ)
    (G : EmbeddingGram (N + 1)) (distance : Key → Key → ℝ)
    (prototypes : Fin (N + 1) → Key) (queries : Fin R → Key)
    (hG : G ∈ memoryGramDomain N cap floor) (r : Fin R) :
    contextMemoryAttention G (nearestPrototypeCodes distance prototypes queries) r =
      memoryGramAttention G (nearestPrototype distance prototypes (queries r)) := by
  rw [contextMemoryAttention_eq_product cap floor G _ hG (nearestPrototypeCodes_mem _ _ _)]
  exact congrFun (oneHotContextCodes_mul _ (memoryGramAttention G)) r

/-- Real learned embeddings and a nonempty data table inhabit actual row selection. -/
example : contextMemoryAttention (memoryIdentityGram 1)
    (nearestPrototypeCodes (fun x y : Fin 2 => if x = y then (0 : ℝ) else 1)
      (fun j : Fin 2 => j) (fun _ : Fin 1 => (0 : Fin 2))) 0 =
    memoryGramAttention (memoryIdentityGram 1)
      (nearestPrototype (fun x y : Fin 2 => if x = y then (0 : ℝ) else 1)
        (fun j : Fin 2 => j) 0) :=
  contextNearestAttention_row 1 (3 / 4) _ _ _ _
    (memoryIdentityGram_mem _ _ _ (by norm_num) (by norm_num)) _

end Transformer.GPTMini.Sparsemax
