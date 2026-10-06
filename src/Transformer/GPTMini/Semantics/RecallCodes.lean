import Transformer.GPTMini.Semantics.AdjacencyRouting

/-!
# Compact four-pair codes for the raw recall vocabulary

Source: MQAR's 256 keys/256 values at cbafbe9 and the original small
GPTMini's sixteen-dimensional heads at f11b6e2. This new explicit token
code uses four base-four digits, each represented by one signed axis of
a two-dimensional pair. Every symbol uses eight coordinates and has
norm two. Distinct symbols differ in at least one digit and have a
strict unrotated inner-product gap. No table of prefixes or paired
key/value semantic oracle is part of the encoding.

Raw embedding/QKV realization, the effect of original RoPE and the
latest-write/tied-readout proofs remain separate construction steps.
-/

namespace Transformer.GPTMini.Semantics

open scoped BigOperators

/-- Four actual base-four digits encode the 256 distinct symbol IDs.
Source: the new compact code for the existing finite MQAR alphabet. -/
def recallDigit (symbol : Fin 256) (place : Fin 4) : Fin 4 :=
  ⟨symbol.val / 4 ^ place.val % 4, Nat.mod_lt _ (by decide)⟩

/-- First coordinate of a signed-axis quarter-turn code.
Source: the new four-state token representation, independent of context. -/
def quarterX (digit : Fin 4) : ℝ :=
  if digit.val = 0 then 1 else if digit.val = 2 then -1 else 0

/-- Second coordinate of the same signed-axis quarter-turn code.
Source: the simultaneous pair (quarterX,quarterY) of each base-four digit. -/
def quarterY (digit : Fin 4) : ℝ :=
  if digit.val = 1 then 1 else if digit.val = 3 then -1 else 0

/-- Eight coordinates retain four independent categorical digits.
Source: this explicit compact code, using no learned or precomputed task-answer feature. -/
noncomputable def recallCode (digits : Fin 4 → Fin 4) : EucSpace 8 :=
  (EuclideanSpace.equiv (Fin 8) ℝ).symm fun i =>
    if i.val % 2 = 0 then quarterX (digits ⟨i.val / 2, by have hi := i.isLt; omega⟩)
    else quarterY (digits ⟨i.val / 2, by have hi := i.isLt; omega⟩)

/-- Each vocabulary symbol receives its compact eight-coordinate code.
Source: the actual 256-symbol recall alphabet, before any adjacent-token processing. -/
noncomputable def recallSymbolCode (symbol : Fin 256) : EucSpace 8 := recallCode (recallDigit symbol)

/-- Every single digit pair has exact squared norm one.
Source: the four explicit signed-axis assignments, including both negative axes. -/
theorem quarter_unit (digit : Fin 4) : quarterX digit ^ 2 + quarterY digit ^ 2 = 1 := by
  fin_cases digit <;> norm_num [quarterX, quarterY]

/-- Both actual signed-axis coordinates remain bounded by one.
Source: the explicit digit table; these bounds later control the table gate's true FFN inputs. -/
theorem quarter_bounds (digit : Fin 4) : |quarterX digit| ≤ 1 ∧ |quarterY digit| ≤ 1 := by
  fin_cases digit <;> norm_num [quarterX, quarterY]

/-- Four base-four digits retain the entire bounded symbol, with no collisions.
Source: elementary radix reconstruction on IDs zero through 255. -/
theorem recallDigit_injective : Function.Injective recallDigit := by
  intro a b h
  have h0 := congrArg (fun digits : Fin 4 → Fin 4 => (digits 0).val) h
  have h1 := congrArg (fun digits : Fin 4 → Fin 4 => (digits 1).val) h
  have h2 := congrArg (fun digits : Fin 4 → Fin 4 => (digits 2).val) h
  have h3 := congrArg (fun digits : Fin 4 → Fin 4 => (digits 3).val) h
  norm_num [recallDigit] at h0 h1 h2 h3
  apply Fin.ext
  have ha := a.isLt
  have hb := b.isLt
  omega

/-- Actual coordinates of the compact representation are the two explicit digit coordinates.
Source: the ordinary Euclidean-space coordinate map, not a semantic encoder premise. -/
theorem recallCode_coordinate (digits : Fin 4 → Fin 4) (i : Fin 8) :
    recallCode digits i =
      if i.val % 2 = 0 then quarterX (digits ⟨i.val / 2, by have hi := i.isLt; omega⟩)
      else quarterY (digits ⟨i.val / 2, by have hi := i.isLt; omega⟩) := by
  unfold recallCode
  rfl

/-- Four pairs give exact squared norm four, independent of the token ID.
Source: the actual eight-coordinate vector and the proved unit digit pairs. -/
theorem recallCode_norm_sq (digits : Fin 4 → Fin 4) : ‖recallCode digits‖ ^ 2 = 4 := by
  rw [EuclideanSpace.norm_sq_eq, Fin.sum_univ_eight]
  simp only [recallCode_coordinate, Real.norm_eq_abs, sq_abs]
  norm_num
  nlinarith [quarter_unit (digits 0), quarter_unit (digits 1),
    quarter_unit (digits 2), quarter_unit (digits 3)]

/-- Every code has exact norm two, allowing a common embedding/RMS coefficient.
Source: the proved squared norm and nonnegativity of the actual Euclidean norm. -/
theorem recallCode_norm (digits : Fin 4 → Fin 4) : ‖recallCode digits‖ = 2 := by
  nlinarith [recallCode_norm_sq digits, norm_nonneg (recallCode digits)]

/-- Identical digit axes have score one; every differing axis has score at most zero.
Source: all sixteen actual signed-axis comparisons, before original RoPE is applied. -/
theorem quarter_inner (a b : Fin 4) :
    quarterX a * quarterX b + quarterY a * quarterY b ≤ if a = b then 1 else 0 := by
  fin_cases a <;> fin_cases b <;> norm_num [quarterX, quarterY]

/-- The full eight-coordinate inner product is the sum of the four actual pair inner products.
Source: Euclidean inner product and the code's four-pair layout. -/
theorem recallCode_inner (a b : Fin 4 → Fin 4) :
    inner (𝕜 := ℝ) (recallCode a) (recallCode b) =
      ∑ p : Fin 4, (quarterX (a p) * quarterX (b p) + quarterY (a p) * quarterY (b p)) := by
  rw [PiLp.inner_apply, Fin.sum_univ_eight, Fin.sum_univ_four]
  simp only [RCLike.inner_apply, conj_trivial, recallCode_coordinate]
  norm_num
  ring

/-- Distinct four-digit codes have a uniform unrotated score gap of at least one.
Source: a differing signed-axis pair contributes at most zero, while the other three contribute at most one. -/
theorem recallCode_inner_gap (a b : Fin 4 → Fin 4) (hne : a ≠ b) :
    inner (𝕜 := ℝ) (recallCode a) (recallCode b) ≤ 3 := by
  classical
  obtain ⟨p, hp⟩ : ∃ p, a p ≠ b p := by
    by_contra hn
    apply hne
    funext p
    by_contra h
    exact hn ⟨p, h⟩
  rw [recallCode_inner]
  have hs : ∑ r : Fin 4, (quarterX (a r) * quarterX (b r) + quarterY (a r) * quarterY (b r)) ≤
      ∑ r : Fin 4, if r = p then (0 : ℝ) else 1 := by
    apply Finset.sum_le_sum
    intro r hr
    by_cases he : r = p
    · subst r
      simpa only [ite_true, ite_eq_right hp] using quarter_inner (a p) (b p)
    · rw [ite_eq_right he]
      have hh := quarter_inner (a r) (b r)
      split_ifs at hh <;> linarith
  have hc : (∑ r : Fin 4, if r = p then (0 : ℝ) else 1) = 3 := by
    fin_cases p <;> norm_num [Fin.sum_univ_four]
  exact hs.trans (by rw [hc])

example : (fun _ : Fin 4 => (0 : Fin 4)) ≠ (fun _ : Fin 4 => (1 : Fin 4)) := by
  intro h
  have he := congrArg (fun digits : Fin 4 → Fin 4 => digits 0) h
  contradiction

/-- The eight-coordinate code of every one of the 256 symbols is distinct.
Source: the proved digit injection and geometric score gap, not a list of all possible input prefixes. -/
theorem recallSymbolCode_injective : Function.Injective recallSymbolCode := by
  intro a b h
  by_contra hne
  have hd : recallDigit a ≠ recallDigit b := fun he => hne (recallDigit_injective he)
  have hg := recallCode_inner_gap _ _ hd
  change inner (𝕜 := ℝ) (recallSymbolCode a) (recallSymbolCode b) ≤ 3 at hg
  rw [h, real_inner_self_eq_norm_sq, recallSymbolCode, recallCode_norm_sq] at hg
  norm_num at hg

end Transformer.GPTMini.Semantics
