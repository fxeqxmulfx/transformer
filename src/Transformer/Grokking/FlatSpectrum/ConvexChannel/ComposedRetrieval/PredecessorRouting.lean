module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.ComposedRetrieval.SignedProfile

/-!
# Content matching conditioned on the preceding object

Source: `review/convex/moments.md`, CM120. Both normalized content reads
use the same learned phase table for current and preceding objects.
A four-weight identity describes which occurrence of a repeated symbol
can be selected as a function of the query's predecessor.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.ComposedRetrieval

open Perspective
open scoped BigOperators
variable {O : Type*} {d n : ℕ}

/-- CM120. An absent predecessor contributes zero to the comparison. Two
present predecessors use the actual mean inner product of learned keys. -/
noncomputable def predecessorKernel (p : Parameters O d) : Option O → Option O → ℝ
  | some a, some b => average (fun f => Real.cos (p.key a f - p.key b f))
  | _, _ => 0

/-- CM120. The query and candidate each supply their own predecessor
from the visible prefix. The source-position bias remains in `keyScore`. -/
noncomputable def predecessorScore (p : Parameters O d) (t : ℝ) (x : Fin n → O)
    (q : O) (a : Option O) (j : Fin n) : ℝ :=
  keyScore p x q j + t * predecessorKernel p a (previous x j)

/-- CM120. Difference of the two predecessor preferences across two
query contexts. It depends only on the learned keys and visible objects. -/
noncomputable def predecessorContrast (p : Parameters O d) (a b u v : Option O) : ℝ :=
  predecessorKernel p a u - predecessorKernel p a v +
    (predecessorKernel p b v - predecessorKernel p b u)

/-- CM120. Normalization cancels from relative probabilities inside a row. -/
private theorem weight_odds (u : Fin n → ℝ) (j k : Fin n) :
    softmaxWeight u j = Real.exp (u j - u k) * softmaxWeight u k := by
  change Real.exp (u j) / _ = Real.exp (u j - u k) * (Real.exp (u k) / _)
  rw [Real.exp_sub]
  field_simp

/-- CM120. For two occurrences of the same raw object, changing query
context changes their odds by exactly the learned predecessor contrast.
At zero the product of opposite row entries is the same on both sides. -/
theorem repeated_key_cross_odds (p : Parameters O d) (t : ℝ) (x : Fin n → O)
    (q v : O) (a b : Option O) (j k : Fin n) (same : x j = x k) :
    softmaxWeight (predecessorScore p t x q a) j * softmaxWeight (predecessorScore p t x v b) k =
    Real.exp (t * predecessorContrast p a b (previous x j) (previous x k)) *
      (softmaxWeight (predecessorScore p t x q a) k * softmaxWeight (predecessorScore p t x v b) j) := by
  let u := predecessorScore p t x q a
  let w := predecessorScore p t x v b
  have gap : u j - u k + (w k - w j) = t * predecessorContrast p a b (previous x j) (previous x k) := by
    simp only [u, w, predecessorScore, predecessorContrast, keyScore, same]
    ring
  change softmaxWeight u j * softmaxWeight w k = _ * (softmaxWeight u k * softmaxWeight w j)
  calc
    _ = (Real.exp (u j - u k) * softmaxWeight u k) *
        (Real.exp (w k - w j) * softmaxWeight w j) :=
      congrArg₂ (· * ·) (weight_odds u j k) (weight_odds w k j)
    _ = (Real.exp (u j - u k) * Real.exp (w k - w j)) *
        (softmaxWeight u k * softmaxWeight w j) := by ring
    _ = _ := by rw [← Real.exp_add, gap]

/-- CM120. Two observed normalized rows identify the shared coefficient
at fixed keys if their repeated-source contrast is nonzero. -/
theorem predecessor_coefficient_identified (p : Parameters O d) (t s : ℝ) (x : Fin n → O)
    (q v : O) (a b : Option O) (j k : Fin n) (same : x j = x k)
    (sep : predecessorContrast p a b (previous x j) (previous x k) ≠ 0)
    (obs₁ : softmaxWeight (predecessorScore p t x q a) = softmaxWeight (predecessorScore p s x q a))
    (obs₂ : softmaxWeight (predecessorScore p t x v b) = softmaxWeight (predecessorScore p s x v b)) :
    t = s := by
  have ht := repeated_key_cross_odds p t x q v a b j k same
  have hs := repeated_key_cross_odds p s x q v a b j k same
  rw [obs₁, obs₂] at ht
  have pos (u : Fin n → ℝ) (i : Fin n) : 0 < softmaxWeight u i :=
    div_pos (Real.exp_pos _) (Landscape.Neural.sum_exp_pos u i)
  exact mul_right_cancel₀ sep (Real.exp_injective (mul_right_cancel₀
    (mul_pos (pos _ _) (pos _ _)).ne' (ht.symm.trans hs)))

/-- CM120. In a legal memory the two inner queries at positions one
and three have the same current object and distinct predecessors. With
antipodal learned keys their cross ratio is the exponential of four
times the trained coefficient. -/
theorem antipodal_predecessor_odds (t : ℝ) :
    let p := productParameters bitAngle bitAngle
    let x : Fin 4 → Bool := ![false,true,true,true]
    let w := fun i : Fin 4 => softmaxWeight (predecessorScore p t x (x i) (previous x i))
    w 1 1 * w 3 3 = Real.exp (4 * t) * (w 1 3 * w 3 1) := by
  dsimp only
  let x : Fin 4 → Bool := ![false,true,true,true]
  have h := repeated_key_cross_odds (productParameters bitAngle bitAngle) t x
    (x 1) (x 3) (previous x 1) (previous x 3) 1 3 (by decide)
  have c : predecessorContrast (productParameters bitAngle bitAngle)
      (previous x 1) (previous x 3) (previous x 1) (previous x 3) = 4 := by
    norm_num [x, previous, predecessorContrast, predecessorKernel, productParameters, average,
      bitAngle, Fin.sum_univ_one]
  rw [c, mul_comm t] at h
  exact h

example : let p := productParameters bitAngle bitAngle
    let x : Fin 4 → Bool := ![false,true,true,true]
    softmaxWeight (predecessorScore p 0 x true (some false)) 1 *
      softmaxWeight (predecessorScore p 0 x true (some true)) 3 =
    softmaxWeight (predecessorScore p 0 x true (some false)) 3 *
      softmaxWeight (predecessorScore p 0 x true (some true)) 1 := by
  simpa using repeated_key_cross_odds (productParameters bitAngle bitAngle) 0
    (![false,true,true,true] : Fin 4 → Bool) true true (some false) (some true) 1 3 (by decide)

example (t s : ℝ) :
    let w := fun u a => softmaxWeight (predecessorScore (productParameters bitAngle bitAngle)
      u (![false,true,true,true] : Fin 4 → Bool) true (some a))
    (w t false = w s false ∧ w t true = w s true) ↔ t = s := by
  constructor
  · rintro ⟨left, right⟩
    exact predecessor_coefficient_identified (productParameters bitAngle bitAngle) t s
      (![false,true,true,true] : Fin 4 → Bool) true true (some false) (some true) 1 3 (by decide)
      (by norm_num [previous, predecessorContrast, predecessorKernel, productParameters, average,
        bitAngle, Fin.sum_univ_one]) left right
  · rintro rfl
    exact ⟨rfl, rfl⟩

end Transformer.Grokking.FlatSpectrum.ConvexChannel.ComposedRetrieval
