import Transformer.GPTMini.Sparsemax.CodeMemoryOutputs
import Mathlib.Data.Fintype.EquivFin
import Mathlib.Data.Fintype.BigOperators

/-!
# Complete causal observation codes for a finite context window

Derived input architecture before sparsemax arXiv:1602.02068v2, Eq. (1).
The data encoder records each visible token and masks every future token
with `none`. A finite enumeration turns this complete observation into a
categorical dictionary key. No output label or teacher route enters it.

Equality of keys is exactly equality of the causal observation: the query
position and every token through that position. Thus distinct visible
prefixes never collide, while changes strictly in the future are ignored.
The encoder retains order and repetitions lost by a frequency code.

The construction enumerates all option-valued position signatures, including
unused ones. Its dictionary has `(V + 1)^T` slots for vocabulary size V and
window length T. This deliberately expensive bound proves expressivity;
it does not establish a compact transformer or generalization beyond T.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

/-- An observed finite prefix: visible tokens are present, future slots absent.
Source: the derived causal data encoder before sparsemax Eq. (1). -/
def causalPrefixSignature {T V : ℕ} (tokens : Fin T → Fin V) (i : Fin T) :
    Fin T → Option (Fin V) := fun j => if j ≤ i then some (tokens j) else none

/-- Complete observations agree exactly when position and visible tokens agree.
Source: the derived encoder, including the mask as part of the observation.
Context: derived memory architecture for arXiv:1602.02068v2, Eq. (1). -/
theorem causalPrefixSignature_eq_iff {T V : ℕ} (tokens other : Fin T → Fin V)
    (i j : Fin T) : causalPrefixSignature tokens i = causalPrefixSignature other j ↔
      i = j ∧ ∀ k, k ≤ i → tokens k = other k := by
  constructor
  · intro h
    have hij : i ≤ j := by
      by_contra hn
      have hk := congrFun h i
      simp [causalPrefixSignature, hn] at hk
    have hji : j ≤ i := by
      by_contra hn
      have hk := congrFun h j
      simp [causalPrefixSignature, hn] at hk
    have he := le_antisymm hij hji
    refine ⟨he, ?_⟩
    intro k hk
    subst j
    have hv := congrFun h k
    simp only [causalPrefixSignature, hk, ite_true] at hv
    exact Option.some_injective (Fin V) hv
  · rintro ⟨rfl, h⟩
    funext k
    by_cases hk : k ≤ i
    · simp only [causalPrefixSignature, hk, ite_true, h k hk]
    · simp only [causalPrefixSignature, hk, ite_false]

/-- The number of memory slots minus one, matching the nonempty memory API.
Source: all finite option-valued data signatures, independent of target data. -/
def prefixMemoryN (T V : ℕ) : ℕ := Fintype.card (Fin T → Option (Fin V)) - 1

/-- Even the empty window has a nonempty finite signature dictionary.
Source: the all-masked signature witnesses positivity before subtraction.
Context: derived memory architecture for arXiv:1602.02068v2, Eq. (1). -/
theorem prefixMemoryN_card (T V : ℕ) :
    prefixMemoryN T V + 1 = Fintype.card (Fin T → Option (Fin V)) := by
  have hn : 0 < Fintype.card (Fin T → Option (Fin V)) := Fintype.card_pos
  unfold prefixMemoryN
  omega

/-- The explicit universality construction pays an exponential dictionary cost.
Source: the complete option-valued position signatures, not a width approximation.
Context: derived memory architecture for arXiv:1602.02068v2, Eq. (1). -/
theorem prefixMemoryN_size (T V : ℕ) : prefixMemoryN T V + 1 = (V + 1) ^ T := by
  rw [prefixMemoryN_card, Fintype.card_fun, Fintype.card_option, Fintype.card_fin,
    Fintype.card_fin]

/-- One target-independent enumeration of all complete data signatures.
Source: the finite causal encoder; unused signature slots are retained. -/
def prefixSignatureEquiv (T V : ℕ) :
    (Fin T → Option (Fin V)) ≃ Fin (prefixMemoryN T V + 1) :=
  Fintype.equivFinOfCardEq (prefixMemoryN_card T V).symm

/-- A categorical key determined by the visible prefix alone.
Source: the derived finite enumeration preceding learned memory attention. -/
def causalPrefixKey {T V : ℕ} (tokens : Fin T → Fin V) (i : Fin T) :
    Fin (prefixMemoryN T V + 1) := prefixSignatureEquiv T V (causalPrefixSignature tokens i)

/-- Categorical keys distinguish exactly the observed causal prefixes.
Source: the proved mask-and-token characterization and the finite enumeration.
Context: derived memory architecture for arXiv:1602.02068v2, Eq. (1). -/
theorem causalPrefixKey_eq_iff {T V : ℕ} (tokens other : Fin T → Fin V) (i j : Fin T) :
    causalPrefixKey tokens i = causalPrefixKey other j ↔
      i = j ∧ ∀ k, k ≤ i → tokens k = other k := by
  unfold causalPrefixKey
  rw [(prefixSignatureEquiv T V).apply_eq_iff_eq, causalPrefixSignature_eq_iff]

/-- Complete causal signatures supplied as probability codes to learned memory.
Source: the new data encoder; the learned Gram still determines actual routes. -/
def contextFullPrefixCodes {R T V : ℕ} (tokens : Fin R → Fin T → Fin V)
    (rows : Fin R → Fin T) : Matrix (Fin R) (Fin (prefixMemoryN T V + 1)) ℝ :=
  oneHotContextCodes (fun r => causalPrefixKey (tokens r) (rows r))

/-- Arbitrary repetitions and context positions give valid memory inputs.
Source: the complete causal encoder, with no restriction on token patterns.
Context: derived memory architecture for arXiv:1602.02068v2, Eq. (1). -/
theorem contextFullPrefixCodes_mem {R T V : ℕ} (tokens : Fin R → Fin T → Fin V)
    (rows : Fin R → Fin T) : contextFullPrefixCodes tokens rows ∈
      contextCodeDomain R (prefixMemoryN T V) := oneHotContextCodes_mem _

/-- Future changes cannot change the complete causal probability code.
Source: the explicit mask in the derived data encoder.
Context: derived memory architecture for arXiv:1602.02068v2, Eq. (1). -/
theorem contextFullPrefixCodes_eq_of_visible {R T V : ℕ}
    (tokens other : Fin R → Fin T → Fin V) (rows : Fin R → Fin T)
    (h : ∀ r k, k ≤ rows r → tokens r k = other r k) :
    contextFullPrefixCodes tokens rows = contextFullPrefixCodes other rows := by
  have he : (fun r => causalPrefixKey (tokens r) (rows r)) =
      (fun r => causalPrefixKey (other r) (rows r)) := by
    funext r
    exact (causalPrefixKey_eq_iff _ _ _ _).mpr ⟨rfl, h r⟩
  unfold contextFullPrefixCodes
  rw [he]

/-- Different future tokens inhabit the complete-code causality premise. -/
example : contextFullPrefixCodes (fun _ : Fin 1 => fun _ : Fin 2 => (0 : Fin 2))
    (fun _ => 0) = contextFullPrefixCodes
      (fun _ : Fin 1 => fun k : Fin 2 => if k = 0 then 0 else (1 : Fin 2)) (fun _ => 0) := by
  apply contextFullPrefixCodes_eq_of_visible
  intro r k hk
  fin_cases k
  · rfl
  · norm_num at hk

/-- Order changes are retained even when token frequencies are the same.
Source: the complete visible-prefix signature, unlike token-count aggregation.
Context: derived memory architecture for arXiv:1602.02068v2, Eq. (1). -/
theorem causalPrefixKey_distinguishes_order :
    causalPrefixKey (fun k : Fin 2 => k) 1 ≠
      causalPrefixKey (fun k : Fin 2 => if k = 0 then 1 else 0) 1 := by
  intro h
  have hk := ((causalPrefixKey_eq_iff _ _ _ _).mp h).2 0 (by norm_num)
  norm_num at hk

/-- The mask preserves query position even if every visible token is repeated.
Source: the complete causal signature, whose first absent position marks its length.
Context: derived memory architecture for arXiv:1602.02068v2, Eq. (1). -/
theorem causalPrefixKey_ne_of_positions {T V : ℕ} (tokens other : Fin T → Fin V)
    (i j : Fin T) (hij : i ≠ j) : causalPrefixKey tokens i ≠ causalPrefixKey other j := by
  intro h
  exact hij ((causalPrefixKey_eq_iff _ _ _ _).mp h).1

/-- Repeated observations at different query positions satisfy the premise. -/
example : causalPrefixKey (fun _ : Fin 2 => (0 : Fin 1)) 0 ≠
    causalPrefixKey (fun _ : Fin 2 => (0 : Fin 1)) 1 :=
  causalPrefixKey_ne_of_positions _ _ _ _ (by decide)

end Transformer.GPTMini.Sparsemax
