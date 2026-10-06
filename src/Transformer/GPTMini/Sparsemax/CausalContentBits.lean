import Transformer.GPTMini.Sparsemax.CubeContentBits
import Mathlib.Data.Nat.Bitwise

/-!
# Lossless causal content bits with a fixed token code

New data encoder before sparsemax arXiv:1602.02068v2, Eq. (1).
Each position contributes a presence bit and the bits of its observed
token. Future positions contribute zero bits. Any fixed injective token
code is allowed, so this representation preserves visible order, repeated
tokens and prefix length without a stored dictionary of prototypes.

The code and bit identities are fixed architecture inputs, not teacher
attention labels or an independently trained nonlinear encoder. Learned
content interactions are supplied by the coefficient chart in the next
modules. A vocabulary can use a binary code rather than one-hot bits;
the causal representation uses T*(B+1) coordinates for B token-code bits.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

/-- Presence and encoded token bits at actual visible positions; hidden continuation is absent.
Source: the new causal content input before sparsemax Eq. (1). -/
def causalContentBits {T V B : ℕ} (encode : Fin V → Fin B → Bool)
    (tokens : Fin T → Fin V) (row : Fin T) : CubeContentState (Fin T × Option (Fin B))
  | (j, none) => decide (j ≤ row)
  | (j, some k) => if j ≤ row then encode (tokens j) k else false

/-- Future token changes cannot change the actual content query bits.
Source: explicit observation masking before sparsemax Eq. (1). -/
theorem causalContentBits_causal {T V B : ℕ} (encode : Fin V → Fin B → Bool)
    (tokens other : Fin T → Fin V) (row : Fin T)
    (h : ∀ j, j ≤ row → tokens j = other j) :
    causalContentBits encode tokens row = causalContentBits encode other row := by
  funext p
  rcases p with ⟨j, k⟩
  cases k with
  | none => rfl
  | some k =>
    by_cases hj : j ≤ row
    · simp only [causalContentBits, hj, ite_true, h j hj]
    · simp only [causalContentBits, hj, ite_false]

/-- A changed future binary token inhabits every causal-encoding premise. -/
example : causalContentBits (fun v : Fin 2 => fun _ : Fin 1 => decide (v = 1))
    (fun _ : Fin 2 => 0) 0 =
    causalContentBits (fun v : Fin 2 => fun _ : Fin 1 => decide (v = 1))
      (fun j : Fin 2 => if j = 0 then 0 else 1) 0 := by
  apply causalContentBits_causal
  intro j hj
  have he : j = 0 := by omega
  subst j
  rfl

/-- With an injective fixed token code, content equality is exactly equality of the visible prefix.
Source: lossless data-derived content representation before sparsemax Eq. (1). -/
theorem causalContentBits_eq_iff {T V B : ℕ} (encode : Fin V → Fin B → Bool)
    (hencode : Function.Injective encode) (tokens other : Fin T → Fin V) (i j : Fin T) :
    causalContentBits encode tokens i = causalContentBits encode other j ↔
      i = j ∧ ∀ k, k ≤ i → tokens k = other k := by
  constructor
  · intro h
    have hij : i ≤ j := by
      by_contra hn
      have hb := congrFun h (i, none)
      simp [causalContentBits, hn] at hb
    have hji : j ≤ i := by
      by_contra hn
      have hb := congrFun h (j, none)
      simp [causalContentBits, hn] at hb
    have he := le_antisymm hij hji
    refine ⟨he, ?_⟩
    subst j
    intro k hk
    apply hencode
    funext b
    have hb := congrFun h (k, some b)
    simpa only [causalContentBits, hk, ite_true] using hb
  · rintro ⟨rfl, h⟩
    exact causalContentBits_causal encode tokens other i h

/-- A one-bit code for a two-token vocabulary is injective.
Source: a genuine binary-code witness before sparsemax Eq. (1). -/
theorem binaryContentCode_injective :
    Function.Injective (fun v : Fin 2 => fun _ : Fin 1 => decide (v = 1)) := by
  intro v w h
  have hb := congrFun h 0
  fin_cases v <;> fin_cases w <;> first | rfl | norm_num at hb

/-- Canonical binary token bits are computed directly from the token number.
Source: compressed data encoding before sparsemax Eq. (1); no prototype enumeration is used. -/
def fixedBinaryContentCode (V B : ℕ) (v : Fin V) (b : Fin B) : Bool := v.val.testBit b.val

/-- Enough binary digits preserve every vocabulary identity, including non-power-of-two vocabularies.
Source: the explicit content code before sparsemax Eq. (1), with its necessary capacity hypothesis. -/
theorem fixedBinaryContentCode_injective (V B : ℕ) (hcapacity : V ≤ 2 ^ B) :
    Function.Injective (fixedBinaryContentCode V B) := by
  intro v w h
  apply Fin.ext
  apply Nat.eq_of_testBit_eq
  intro i
  by_cases hi : i < B
  · exact congrFun h ⟨i, hi⟩
  · have hp : 2 ^ B ≤ 2 ^ i := Nat.pow_le_pow_right (by decide : 0 < 2) (by omega)
    have hv : v.val < 2 ^ i := lt_of_lt_of_le v.isLt (le_trans hcapacity hp)
    have hw : w.val < 2 ^ i := lt_of_lt_of_le w.isLt (le_trans hcapacity hp)
    rw [Nat.testBit_eq_false_of_lt hv, Nat.testBit_eq_false_of_lt hw]

/-- The Basis recall vocabulary fits into ten actual binary digits. -/
example : Function.Injective (fixedBinaryContentCode 548 10) :=
  fixedBinaryContentCode_injective _ _ (by norm_num)

/-- Actual binary codes inhabit all lossless-encoding premises. -/
example : causalContentBits (fun v : Fin 2 => fun _ : Fin 1 => decide (v = 1))
    (fun _ : Fin 2 => 0) 0 =
    causalContentBits (fun v : Fin 2 => fun _ : Fin 1 => decide (v = 1))
      (fun j : Fin 2 => if j = 0 then 0 else 1) 0 ↔
    (0 : Fin 2) = 0 ∧ ∀ k : Fin 2, k ≤ 0 → (0 : Fin 2) = (if k = 0 then 0 else 1) :=
  causalContentBits_eq_iff _ binaryContentCode_injective _ _ _ _

/-- Swapping observed binary tokens changes content even though the token counts agree.
Source: position-preserving content input before sparsemax Eq. (1). -/
theorem causalContentBits_distinguishes_order :
    causalContentBits (fun v : Fin 2 => fun _ : Fin 1 => decide (v = 1))
      (fun j : Fin 2 => j) 1 ≠
    causalContentBits (fun v : Fin 2 => fun _ : Fin 1 => decide (v = 1))
      (fun j : Fin 2 => if j = 0 then 1 else 0) 1 := by
  intro h
  have hv := (causalContentBits_eq_iff _ binaryContentCode_injective _ _ _ _).mp h
  have hb := hv.2 0 (by norm_num)
  norm_num at hb

/-- Presence bits distinguish prefix lengths even when every token-code bit is zero.
Source: the observation mask retained in the content input before sparsemax Eq. (1). -/
theorem causalContentBits_distinguishes_length :
    causalContentBits (fun _ : Fin 1 => fun _ : Fin 1 => false) (fun _ : Fin 2 => 0) 0 ≠
    causalContentBits (fun _ : Fin 1 => fun _ : Fin 1 => false) (fun _ : Fin 2 => 0) 1 := by
  intro h
  have hb := congrFun h (1, none)
  norm_num [causalContentBits] at hb

/-- Generated causal state size depends on token-code width and context length, not training rows.
Source: explicit coordinate count before sparsemax Eq. (1). -/
theorem causalContentBits_coordinate_count (T B : ℕ) :
    Fintype.card (Fin T × Option (Fin B)) = T * (B + 1) := by
  rw [Fintype.card_prod, Fintype.card_option, Fintype.card_fin, Fintype.card_fin]

/-- Sixteen binary positions need thirty-two content/presence bits and no prototype search. -/
example : Fintype.card (Fin 16 × Option (Fin 1)) = 32 :=
  causalContentBits_coordinate_count 16 1

/-- A visible binary token retains its content bit at its actual position. -/
example : causalContentBits (fun v : Fin 2 => fun _ : Fin 1 => decide (v = 1))
    (fun j : Fin 2 => j) 1 (1, some 0) = true := by
  norm_num [causalContentBits]

/-- The same token hidden after the query contributes no content bit. -/
example : causalContentBits (fun v : Fin 2 => fun _ : Fin 1 => decide (v = 1))
    (fun j : Fin 2 => j) 0 (1, some 0) = false := by
  norm_num [causalContentBits]

end Transformer.GPTMini.Sparsemax
