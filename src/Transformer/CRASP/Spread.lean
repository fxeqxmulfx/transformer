/-
# The string map of the reduction from `TL[◁#]^pos`

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), Appendix F, `lem:tlclpos_reduction`: the map
`f(w₁ ⋯ wₙ) = e^r w₁ e^{r−1} w₂ e^{r−1} ⋯ wₙ e^{r−1}`.

For `r ≥ 1` the string `f(w)` falls into `n + 1` blocks of `r` positions:
block `0` is `e^r`, and block `i ≥ 1` is `wᵢ e^{r−1}`.  Position `r·i + ρ`,
for `1 ≤ ρ ≤ r`, is the `ρ`-th position of block `i`; `getElem?_spread`
reads a symbol off this layout.

A `TL[◁#]^pos` formula at a position `j` looks only at positions up to `j`,
so its value there survives extending the string on the right
(`FormP.sat_append`).  In particular every formula sees the first block of
`f(w)` as the string `e^r`, whatever `w` is (`FormP.sat_spread_of_le`).
-/

import Mathlib.Data.List.ReduceOption
import Transformer.CRASP.Positional

namespace Transformer
namespace CRASP

universe u

variable {σ : Type u}

/-- The string map `f` of `lem:tlclpos_reduction`:
`w₁ ⋯ wₙ ↦ e^r w₁ e^{r−1} w₂ e^{r−1} ⋯ wₙ e^{r−1}`, with the neutral letter
`e` written `none`. -/
def spread (r : ℕ) (w : List σ) : List (Option σ) :=
  List.replicate r none ++ w.flatMap fun a => some a :: List.replicate (r - 1) none

@[simp] theorem spread_nil (r : ℕ) : spread (σ := σ) r [] = List.replicate r none := by
  simp [spread]

/-- `spread` only inserts neutral letters, so deleting them gives the string
back: `spread r` is a section of `List.reduceOption`. -/
@[simp] theorem reduceOption_spread (r : ℕ) (w : List σ) : (spread r w).reduceOption = w := by
  rw [spread, List.reduceOption_append, List.reduceOption_replicate_none, List.nil_append]
  induction w with
  | nil => simp
  | cons a l ih =>
      rw [List.flatMap_cons, List.reduceOption_append, ih, List.reduceOption_cons_of_some,
        List.reduceOption_replicate_none]
      rfl

/-- `f(w)` has `n + 1` blocks of `r` positions. -/
theorem length_spread {r : ℕ} (hr : 1 ≤ r) (w : List σ) :
    (spread r w).length = r * (w.length + 1) := by
  induction w with
  | nil => simp
  | cons a l ih =>
      simp only [spread, List.length_append, List.length_replicate, List.flatMap_cons,
        List.length_cons] at ih ⊢
      rw [Nat.mul_succ, ← ih]
      omega

/-- The blocks after the first: position `t < r` of block `j` carries `w[j]`
when `t = 0`, and `e` otherwise. -/
theorem getElem?_flatMap_block {r : ℕ} (hr : 1 ≤ r) (w : List σ) (j t : ℕ) (ht : t < r) :
    (w.flatMap fun a => some a :: List.replicate (r - 1) none)[r * j + t]? =
      w[j]?.map fun a => if t = 0 then some a else none := by
  induction w generalizing j with
  | nil => simp
  | cons a l ih =>
      have hlen : (some a :: List.replicate (r - 1) none).length = r := by simp; omega
      rw [List.flatMap_cons]
      cases j with
      | zero =>
          rw [Nat.mul_zero, Nat.zero_add, List.getElem?_append_left (by omega)]
          cases t with
          | zero => simp
          | succ t => simp [List.getElem?_replicate]; omega
      | succ j =>
          rw [List.getElem?_append_right (by rw [hlen, Nat.mul_succ]; omega), hlen,
            show r * (j + 1) + t - r = r * j + t by rw [Nat.mul_succ]; omega, ih j]
          rfl

/-- **Reading `f(w)` by blocks.**  Position `t < r` of block `i` carries `wᵢ`
when `i ≥ 1` and `t = 0`, and `e` otherwise; past the last block there is
nothing. -/
theorem getElem?_spread {r : ℕ} (hr : 1 ≤ r) (w : List σ) (i t : ℕ) (ht : t < r) :
    (spread r w)[r * i + t]? =
      if i = 0 then some none else w[i - 1]?.map fun a => if t = 0 then some a else none := by
  rw [spread]
  cases i with
  | zero =>
      rw [Nat.mul_zero, Nat.zero_add, List.getElem?_append_left (by simpa using ht)]
      simp [ht]
  | succ i =>
      rw [List.getElem?_append_right (by simp; rw [Nat.mul_succ]; omega), List.length_replicate,
        show r * (i + 1) + t - r = r * i + t by rw [Nat.mul_succ]; omega,
        getElem?_flatMap_block hr w i t ht]
      rfl

/-- The hypotheses of `length_spread`, `getElem?_flatMap_block` and
`getElem?_spread` are satisfiable: `r = 1`, `t = 0`. -/
example : 1 ≤ 1 ∧ 0 < 1 := ⟨le_rfl, Nat.one_pos⟩

variable [DecidableEq σ]

mutual

/-- **A formula does not look ahead.**  At a position `1 ≤ j ≤ |u|` of
`u ++ v`, a `TL[◁#]^pos` formula has its value on `u` (Appendix F: every
operator of `TL[◁#]^pos` looks at the current position or earlier ones). -/
theorem FormP.sat_append (u v : List σ) :
    ∀ (φ : FormP σ) (j : ℕ), 1 ≤ j → j ≤ u.length → φ.sat (u ++ v) j = φ.sat u j
  | .sym a, j, _, h₂ => by rw [FormP.sat, FormP.sat, List.getElem?_append_left (by omega)]
  | .mod _ _, _, _, _ => rfl
  | .prev φ, j, _, h₂ => by
      rw [FormP.sat, FormP.sat]
      by_cases hj : 1 < j
      · rw [φ.sat_append u v (j - 1) (by omega) (by omega)]
      · simp [hj]
  | .lt t₁ t₂, j, _, h₂ => by
      rw [FormP.sat, FormP.sat, t₁.val_append u v j h₂, t₂.val_append u v j h₂]
  | .neg φ, j, h₁, h₂ => by rw [FormP.sat, FormP.sat, φ.sat_append u v j h₁ h₂]
  | .and φ₁ φ₂, j, h₁, h₂ => by
      rw [FormP.sat, FormP.sat, φ₁.sat_append u v j h₁ h₂, φ₂.sat_append u v j h₁ h₂]

/-- A term does not look ahead either: at `j ≤ |u|` it has its value on `u`. -/
theorem TermP.val_append (u v : List σ) :
    ∀ (t : TermP σ) (j : ℕ), j ≤ u.length → t.val (u ++ v) j = t.val u j
  | .countL φ, j, h => by
      rw [TermP.val, TermP.val, List.filter_congr fun j' hj' =>
        φ.sat_append u v j' (List.mem_range'_1.mp hj').1
          (by have := (List.mem_range'_1.mp hj').2; omega)]
  | .add t₁ t₂, j, h => by rw [TermP.val, TermP.val, t₁.val_append u v j h, t₂.val_append u v j h]
  | .one, _, _ => rfl

end

/-- The hypotheses of `FormP.sat_append` are satisfiable: `j = 1` on a
one-letter string. -/
example (a : σ) : 1 ≤ 1 ∧ 1 ≤ [a].length := ⟨le_rfl, le_rfl⟩

/-- Every formula sees the first block of `f(w)` as the string `e^r`. -/
theorem FormP.sat_spread_of_le (r : ℕ) (w : List σ) (φ : FormP (Option σ)) {j : ℕ}
    (h₁ : 1 ≤ j) (h₂ : j ≤ r) : φ.sat (spread r w) j = φ.sat (List.replicate r none) j := by
  rw [spread, FormP.sat_append _ _ φ j h₁ (by rw [List.length_replicate]; exact h₂)]

/-- The hypotheses of `FormP.sat_spread_of_le` are satisfiable: `j = r = 1`. -/
example : 1 ≤ 1 ∧ 1 ≤ 1 := ⟨le_rfl, le_rfl⟩

end CRASP
end Transformer
