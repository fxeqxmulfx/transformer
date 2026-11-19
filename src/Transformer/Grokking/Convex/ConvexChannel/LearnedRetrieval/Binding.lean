module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedRetrieval.Concentration
public import Transformer.Grokking.FlatSpectrum.Training.Margin

/-!
# Actual output margins from shared learned retrieval

Source: `review/convex/moments.md`, CM111. A good path contributes the learned
decoder separation. All other paths remain in the calculation and have margin
at least minus two. The identified object is the object at the linked position;
no theorem asserts that every task's correct next object has this relation.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedRetrieval

open scoped BigOperators
open FlatSpectrum

/-- A normalized complete path distribution retains all adverse paths,
CM111; concentration does not discard their scores. -/
theorem weighted_gap {I : Type*} [Fintype I] (w v : I → ℝ) (i : I) (m : ℝ)
    (hw : ∀ j, 0 ≤ w j) (hs : ∑ j, w j = 1) (hv : ∀ j, -2 ≤ v j) (hi : m ≤ v i) :
    m - (m + 2) * (1 - w i) ≤ ∑ j, w j * v j := by
  have hsum : w i * (v i + 2) ≤ ∑ j, w j * (v j + 2) :=
    Finset.single_le_sum (fun j _ => mul_nonneg (hw j) (show 0 ≤ v j + 2 by linarith [hv j]))
      (Finset.mem_univ i)
  have he : ∑ j, w j * (v j + 2) = (∑ j, w j * v j) + 2 := by
    simp_rw [mul_add]
    rw [Finset.sum_add_distrib, ← Finset.sum_mul, hs, one_mul]
  rw [he] at hsum
  have hm := mul_le_mul_of_nonneg_left hi (hw i)
  nlinarith

variable {O E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℂ E]

/-- Content and relative-score gaps imply an actual complete-class output
margin for the linked object, CM111. The decoder hypothesis concerns learned
vectors at that object, not the unknown model prediction. -/
theorem logit_margin {n : ℕ} (p : Parameters O E) (hp : Unit p)
    (x : Fin n → O) (q c : O) (i k : Fin n) (g h m : ℝ) (hm : 0 ≤ m)
    (hg : ∀ j, j ≠ i → g ≤ keyScores p x q i - keyScores p x q j)
    (hh : ∀ j, j ≠ k → h ≤ relativeScores p i k - relativeScores p i j)
    (hc : m ≤ kernel p (x k) (x k) - kernel p c (x k)) :
    m - (m + 2) * (n * (Real.exp (-g) + Real.exp (-h))) ≤
      logit p x q (x k) - logit p x q c := by
  have hv : ∀ jk : Fin n × Fin n,
      -2 ≤ kernel p (x k) (x jk.2) - kernel p c (x jk.2) := by
    intro jk
    have ht := abs_le.mp (kernel_bound p hp (x k) (x jk.2))
    have hr := abs_le.mp (kernel_bound p hp c (x jk.2))
    linarith
  have hw := weighted_gap (pathMass p x q)
    (fun jk => kernel p (x k) (x jk.2) - kernel p c (x jk.2)) (i, k) m
    (pathMass_nonneg p x q) (sum_pathMass (Nat.zero_lt_of_lt i.isLt) p x q) hv hc
  have he : (∑ jk, pathMass p x q jk *
      (kernel p (x k) (x jk.2) - kernel p c (x jk.2))) =
      logit p x q (x k) - logit p x q c := by
    simp only [mul_sub, Finset.sum_sub_distrib, logit]
  rw [he] at hw
  have herr := mul_le_mul_of_nonneg_left (path_error p x q i k g h hg hh)
    (show 0 ≤ m + 2 by linarith)
  linarith

/-- The original capacity scale multiplies the actual retrieval margin,
CM111, without changing the object space. -/
theorem scaled_margin {n : ℕ} (p : Parameters O E) (hp : Unit p)
    (x : Fin n → O) (q c : O) (i k : Fin n) (g h m B : ℝ) (hm : 0 ≤ m) (hB : 0 ≤ B)
    (hg : ∀ j, j ≠ i → g ≤ keyScores p x q i - keyScores p x q j)
    (hh : ∀ j, j ≠ k → h ≤ relativeScores p i k - relativeScores p i j)
    (hc : m ≤ kernel p (x k) (x k) - kernel p c (x k)) :
    B * (m - (m + 2) * (n * (Real.exp (-g) + Real.exp (-h)))) ≤
      B * logit p x q (x k) - B * logit p x q c := by
  simpa only [mul_sub] using mul_le_mul_of_nonneg_left
    (logit_margin p hp x q c i k g h m hm hg hh hc) hB

/-- A positive derived margin identifies the linked object against every
original class, CM111. This is a conditional retrieval guarantee. -/
theorem correct_object {n : ℕ} (p : Parameters O E) (hp : Unit p)
    (x : Fin n → O) (q : O) (i k : Fin n) (g h m B : ℝ) (hm : 0 ≤ m) (hB : 0 ≤ B)
    (hg : ∀ j, j ≠ i → g ≤ keyScores p x q i - keyScores p x q j)
    (hh : ∀ j, j ≠ k → h ≤ relativeScores p i k - relativeScores p i j)
    (hc : ∀ c, c ≠ x k → m ≤ kernel p (x k) (x k) - kernel p c (x k))
    (hpositive : 0 < B * (m - (m + 2) * (n * (Real.exp (-g) + Real.exp (-h))))) :
    ∀ c, c ≠ x k → B * logit p x q c < B * logit p x q (x k) := by
  intro c hneq
  have hgap := scaled_margin p hp x q c i k g h m B hm hB hg hh (hc c hneq)
  linarith

/-- The same derived margin bounds ordinary complete-class CE, CM111.
No two-value-restricted loss or new task metric is substituted. -/
theorem crossEntropy_bound [Fintype O] {n : ℕ} (p : Parameters O E) (hp : Unit p)
    (x : Fin n → O) (q : O) (i k : Fin n) (g h m B : ℝ) (hm : 0 ≤ m) (hB : 0 ≤ B)
    (hg : ∀ j, j ≠ i → g ≤ keyScores p x q i - keyScores p x q j)
    (hh : ∀ j, j ≠ k → h ≤ relativeScores p i k - relativeScores p i j)
    (hc : ∀ c, c ≠ x k → m ≤ kernel p (x k) (x k) - kernel p c (x k)) :
    Landscape.Neural.crossEntropy (fun c => B * logit p x q c) (x k) ≤
      Real.log (1 + (Fintype.card O - 1) * Real.exp
        (-(B * (m - (m + 2) * (n * (Real.exp (-g) + Real.exp (-h))))))) :=
  (Training.crossEntropy_le_of_le_gap (fun c hneq =>
    scaled_margin p hp x q c i k g h m B hm hB hg hh (hc c hneq))).1

example : (2 : ℝ) - (2 + 2) * (1 - 1 / 2) ≤
    ∑ j : Fin 2, (1 / 2 : ℝ) * (if j = 0 then 2 else -2) :=
  weighted_gap (fun _ : Fin 2 => 1 / 2) (fun j => if j = 0 then 2 else -2) 0 2
    (by intro j; norm_num) (by norm_num [Fin.sum_univ_two])
    (by intro j; split <;> norm_num) (by norm_num)

end Transformer.Grokking.Convex.ConvexChannel.LearnedRetrieval
