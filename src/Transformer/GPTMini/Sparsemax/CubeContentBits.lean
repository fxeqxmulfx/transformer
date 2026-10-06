import Transformer.GPTMini.Sparsemax.CausalPrefixKeys
import Mathlib.Algebra.BigOperators.Ring.Finset

/-!
# Generated bit states for content-sensitive learned memory

New architecture before arXiv:1602.02068v2, Eq. (1). Memory slots are
virtual bit strings rather than stored observed-prefix records. A query
is its actual causal content code. Its possible destinations are itself
and the strings differing in exactly one bit; these can be generated
without enumerating the exponentially large virtual dictionary.

The next modules use products of selected bit signs as generated value
features. This replaces the former two affine positional features by
arbitrary selected content interactions. The finite enumeration below
only connects this construction to the existing variational sparsemax;
it is not an implementation requirement to materialize every state.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open scoped BigOperators

variable {ι : Type*} [DecidableEq ι]

/-- A content state has one Boolean per fixed observation coordinate.
Source: new virtual memory inputs before sparsemax Eq. (1). -/
abbrev CubeContentState (ι : Type*) := ι → Bool

/-- Bounded real bit signs generate content products without a stored feature matrix.
Source: new invariant-feature architecture before sparsemax Eq. (1). -/
def cubeContentSign (b : Bool) : ℝ := if b then -1 else 1

/-- Negating one bit reverses its generated sign.
Source: the content-flip construction before sparsemax Eq. (1). -/
theorem cubeContentSign_not (b : Bool) : cubeContentSign (!b) = -cubeContentSign b := by
  cases b <;> norm_num [cubeContentSign]

/-- Each generated sign has squared magnitude one.
Source: bounded content features before sparsemax Eq. (1). -/
theorem cubeContentSign_sq (b : Bool) : cubeContentSign b ^ 2 = 1 := by
  cases b <;> norm_num [cubeContentSign]

/-- Generated bit coordinates stay in the fixed interval [-1,1].
Source: bounded physical content inputs before sparsemax Eq. (1). -/
theorem cubeContentSign_bounds (b : Bool) : -1 ≤ cubeContentSign b ∧ cubeContentSign b ≤ 1 := by
  cases b <;> norm_num [cubeContentSign]

/-- A real content sign distinguishes both actual observed Boolean values.
Source: the lossless coordinate map before sparsemax Eq. (1). -/
theorem cubeContentSign_values : cubeContentSign false = 1 ∧ cubeContentSign true = -1 := by
  constructor <;> norm_num [cubeContentSign]

/-- Sign equality identifies the actual observed bit.
Source: the lossless bit feature map before sparsemax Eq. (1). -/
theorem cubeContentSign_injective : Function.Injective cubeContentSign := by
  intro x y h
  cases x <;> cases y <;> first | rfl | norm_num [cubeContentSign] at h

/-- Generate one virtual neighbor by negating a single coordinate.
Source: local content memory before arXiv:1602.02068v2, Eq. (1). -/
def cubeContentFlip (i : ι) (x : CubeContentState ι) : CubeContentState ι :=
  fun j => if j = i then !(x j) else x j

/-- A neighbor differs from its query at exactly the selected bit.
Source: the generated local mask before sparsemax Eq. (1). -/
theorem cubeContentFlip_same (i : ι) (x : CubeContentState ι) :
    cubeContentFlip i x i = !(x i) := by
  simp only [cubeContentFlip, ite_true]

/-- Every other content coordinate survives the neighbor operation.
Source: the generated local mask before sparsemax Eq. (1). -/
theorem cubeContentFlip_other (i j : ι) (x : CubeContentState ι) (h : j ≠ i) :
    cubeContentFlip i x j = x j := by
  simp only [cubeContentFlip, h, ite_false]

/-- Distinct actual coordinates inhabit the unchanged-bit premise. -/
example : cubeContentFlip (0 : Fin 2) (fun _ => false) 1 = false :=
  cubeContentFlip_other _ _ _ (by decide)

/-- Flipping a content coordinate twice recovers every original state.
Source: the involutive local connection before sparsemax Eq. (1). -/
theorem cubeContentFlip_involutive (i : ι) : Function.Involutive (cubeContentFlip i) := by
  intro x
  funext j
  by_cases h : j = i
  · subst j
    rw [cubeContentFlip_same, cubeContentFlip_same]
    cases x i <;> rfl
  · rw [cubeContentFlip_other _ _ _ h, cubeContentFlip_other _ _ _ h]

/-- Each generated content neighbor is different from its original query.
Source: no self-loops among the local flip destinations before sparsemax Eq. (1). -/
theorem cubeContentFlip_ne (i : ι) (x : CubeContentState ι) : cubeContentFlip i x ≠ x := by
  intro h
  have hb := congrFun h i
  rw [cubeContentFlip_same] at hb
  cases x i <;> simp at hb

/-- Different selected bits give different actual local destinations.
Source: the sparse virtual content mask before sparsemax Eq. (1). -/
theorem cubeContentFlip_index_injective (x : CubeContentState ι) :
    Function.Injective (fun i => cubeContentFlip i x) := by
  intro i j h
  by_contra hij
  have hb := congrFun h i
  change cubeContentFlip i x i = cubeContentFlip j x i at hb
  rw [cubeContentFlip_same, cubeContentFlip_other _ _ _ hij] at hb
  cases x i <;> simp at hb

variable [Fintype ι]

/-- A nonempty virtual dictionary connects the bit architecture to finite sparsemax rows.
Source: enumeration only for the variational Eq. (1) interface. -/
def cubeContentMemoryN (ι : Type*) [Fintype ι] [DecidableEq ι] : ℕ := Fintype.card (CubeContentState ι) - 1

/-- The all-zero bit string makes the virtual dictionary nonempty at every input dimension.
Source: the generated memory before sparsemax Eq. (1). -/
theorem cubeContentMemoryN_card :
    cubeContentMemoryN ι + 1 = Fintype.card (CubeContentState ι) := by
  have h : 0 < Fintype.card (CubeContentState ι) := Fintype.card_pos
  unfold cubeContentMemoryN
  omega

/-- There are exponentially many virtual states, without exponentially many stored rows.
Source: the finite bit-state construction before sparsemax Eq. (1). -/
theorem cubeContentMemoryN_size : cubeContentMemoryN ι + 1 = 2 ^ Fintype.card ι := by
  rw [cubeContentMemoryN_card, Fintype.card_fun, Fintype.card_bool]

/-- A target-independent bijection for the existing finite variational attention operator.
Source: the virtual content dictionary before sparsemax Eq. (1). -/
def cubeContentIndex : CubeContentState ι ≃ Fin (cubeContentMemoryN ι + 1) :=
  Fintype.equivFinOfCardEq cubeContentMemoryN_card.symm

/-- Actual query and one generated neighbor already differ at one bit. -/
example : cubeContentFlip (0 : Fin 1) (fun _ => false) ≠ (fun _ => false) :=
  cubeContentFlip_ne _ _

/-- Two input bits give four virtual states, rather than four supplied prototypes. -/
example : cubeContentMemoryN (Fin 2) + 1 = 4 := by
  rw [cubeContentMemoryN_size, Fintype.card_fin]
  norm_num

/-- The empty set of content coordinates still has one virtual state. -/
example : cubeContentMemoryN (Fin 0) + 1 = 1 := by
  rw [cubeContentMemoryN_size, Fintype.card_fin]
  norm_num

end Transformer.GPTMini.Sparsemax
