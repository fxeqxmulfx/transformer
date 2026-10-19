module

public import Transformer.Basic
public import Mathlib.Analysis.InnerProductSpace.Projection.Basic
public import Mathlib.Basic.Real.Sign

/-!
# Appendix A — the scalar comparison for causal convergence

Source: arXiv:2411.04990v2, Appendix A, `lemma:scalar`,
in the proof of Theorem `thm1`.
The manuscript compares three original inner products of unit
vectors `x,y` and an arbitrary vector `z`:

  `⟨x,y⟩ ⟨y,z⟩ ≤ |⟨x,y⟩| |⟨y,z⟩| ≤ |⟨x,y⟩| ⟨x,z⟩ ≤ ⟨x,z⟩`.

The absolute comparison hypothesis makes the last inner product
nonnegative. The final inequality uses the actual unit norms of
both original vectors and the Cauchy–Schwarz bound.
The dimension is arbitrary; there is no assumption on the norm of
the original comparison vector `z`.

The equality statement retains both alternatives in the source.
When `⟨x,z⟩=0`, the absolute comparison forces `⟨y,z⟩=0` too.
When `⟨x,z⟩>0`, all three comparisons must be equalities,
forcing the original absolute inner products to agree and the
original first inner product to have the sign of `⟨y,z⟩`.
The proof treats both nonzero signs and the zero case separately.

These already proved declarations and their original hypothesis
witnesses are preserved from `Causal.MainTheorem`, with exactly
the same hypotheses, constants and conclusions. This separate
module follows the manuscript's Appendix A and keeps its scalar
comparison independent of the remaining §4 convergence assertions.
-/

@[expose] public section

open Real

namespace Transformer
namespace Causal

/-- **Lemma (lemma:scalar).** *A scalar inequality for unit vectors.*

For `‖x‖ = ‖y‖ = 1` and `|⟨y, z⟩| ≤ ⟨x, z⟩`,

  `⟨x, z⟩ ≥ ⟨x, y⟩ ⟨y, z⟩`.

The chain is `⟨x,y⟩⟨y,z⟩ ≤ |⟨x,y⟩| |⟨y,z⟩| ≤ |⟨x,y⟩| ⟨x,z⟩ ≤ ⟨x,z⟩`, the last
step because `|⟨x,y⟩| ≤ 1` for unit vectors and `⟨x,z⟩ ≥ 0` by hypothesis.

Source: arXiv:2411.04990v2, §A, `lemma:scalar`. -/
theorem inner_mul_le_inner_of_abs_le
    {d : ℕ} (x y z : EucSpace d) (hx : ‖x‖ = 1) (hy : ‖y‖ = 1)
    (h : |inner (𝕜 := ℝ) y z| ≤ inner (𝕜 := ℝ) x z) :
    inner (𝕜 := ℝ) x y * inner (𝕜 := ℝ) y z ≤ inner (𝕜 := ℝ) x z := by
  have hxz : 0 ≤ inner (𝕜 := ℝ) x z := le_trans (abs_nonneg _) h
  have hxy : |inner (𝕜 := ℝ) x y| ≤ 1 := by
    simpa [hx, hy] using abs_real_inner_le_norm x y
  calc inner (𝕜 := ℝ) x y * inner (𝕜 := ℝ) y z
      ≤ |inner (𝕜 := ℝ) x y| * |inner (𝕜 := ℝ) y z| := by
        rw [← abs_mul]; exact le_abs_self _
    _ ≤ |inner (𝕜 := ℝ) x y| * inner (𝕜 := ℝ) x z :=
        mul_le_mul_of_nonneg_left h (abs_nonneg _)
    _ ≤ 1 * inner (𝕜 := ℝ) x z := mul_le_mul_of_nonneg_right hxy hxz
    _ = inner (𝕜 := ℝ) x z := one_mul _

/-- The hypotheses of `inner_mul_le_inner_of_abs_le` are satisfiable: two
copies of the first standard basis vector, and `z = 0`. -/
example :
    ‖EuclideanSpace.single (0 : Fin 1) (1 : ℝ)‖ = 1 ∧
      |inner (𝕜 := ℝ) (EuclideanSpace.single (0 : Fin 1) (1 : ℝ))
          (0 : EucSpace 1)|
        ≤ inner (𝕜 := ℝ) (EuclideanSpace.single (0 : Fin 1) (1 : ℝ))
            (0 : EucSpace 1) := by
  refine ⟨by simp, ?_⟩
  simp

/-- **Lemma (lemma:scalar), the equality cases.**

Equality `⟨x, z⟩ = ⟨x, y⟩ ⟨y, z⟩` holds if and only if either `⟨x, z⟩ = 0`, or
`|⟨y, z⟩| = ⟨x, z⟩` together with `⟨x, y⟩ = sign⟨y, z⟩`.

The three inequalities of `inner_mul_le_inner_of_abs_le` are `⟨x,y⟩⟨y,z⟩ ≤
|⟨x,y⟩||⟨y,z⟩| ≤ |⟨x,y⟩|⟨x,z⟩ ≤ ⟨x,z⟩`, and equality throughout forces
`|⟨y,z⟩| = ⟨x,z⟩` and — once `⟨x,z⟩ > 0` makes `⟨y,z⟩` nonzero — `|⟨x,y⟩| = 1`
with the sign of `⟨y,z⟩`.

Source: arXiv:2411.04990v2, §A, `lemma:scalar`. -/
theorem inner_mul_eq_inner_iff
    {d : ℕ} (x y z : EucSpace d) (hx : ‖x‖ = 1) (hy : ‖y‖ = 1)
    (h : |inner (𝕜 := ℝ) y z| ≤ inner (𝕜 := ℝ) x z) :
    inner (𝕜 := ℝ) x y * inner (𝕜 := ℝ) y z = inner (𝕜 := ℝ) x z ↔
      inner (𝕜 := ℝ) x z = 0 ∨
        (|inner (𝕜 := ℝ) y z| = inner (𝕜 := ℝ) x z ∧
          inner (𝕜 := ℝ) x y = Real.sign (inner (𝕜 := ℝ) y z)) := by
  set a : ℝ := inner (𝕜 := ℝ) x y with ha
  set b : ℝ := inner (𝕜 := ℝ) y z with hb
  set c : ℝ := inner (𝕜 := ℝ) x z with hc
  have hab : |a| ≤ 1 := by
    rw [ha]; simpa [hx, hy] using abs_real_inner_le_norm x y
  have hb0 : 0 ≤ |b| := abs_nonneg b
  have hc0 : 0 ≤ c := hb0.trans h
  constructor
  · intro heq
    rcases eq_or_lt_of_le hc0 with h0 | hpos
    · exact Or.inl h0.symm
    · refine Or.inr ?_
      have hbne : b ≠ 0 := by
        intro hbz
        rw [hbz, mul_zero] at heq
        exact absurd heq.symm hpos.ne'
      have hbabs : 0 < |b| := abs_pos.mpr hbne
      have habs : |a| * |b| = c := by rw [← abs_mul, heq, abs_of_nonneg hc0]
      have hbc : |b| = c := le_antisymm h (by nlinarith)
      rw [hbc] at habs
      have ha1 : |a| = 1 := mul_right_cancel₀ hpos.ne' (by rw [one_mul]; exact habs)
      refine ⟨hbc, ?_⟩
      rcases (abs_eq (by norm_num : (0 : ℝ) ≤ 1)).mp ha1 with h1 | h1
      · have hbpos : 0 < b := by
          rcases lt_or_gt_of_ne hbne with hneg | hpos'
          · rw [h1, one_mul] at heq
            rw [← heq, abs_of_neg hneg] at hbc
            linarith
          · exact hpos'
        rw [h1, Real.sign_of_pos hbpos]
      · have hbneg : b < 0 := by
          rcases lt_or_gt_of_ne hbne with hneg | hpos'
          · exact hneg
          · rw [h1] at heq
            rw [abs_of_pos hpos'] at hbc
            linarith
        rw [h1, Real.sign_of_neg hbneg]
  · rintro (h0 | ⟨h1, h2⟩)
    · have hbz : b = 0 := abs_eq_zero.mp (le_antisymm (h0 ▸ h) hb0)
      rw [hbz, mul_zero, h0]
    · rcases lt_trichotomy b 0 with hneg | hzero | hpos'
      · rw [h2, Real.sign_of_neg hneg, ← h1, abs_of_neg hneg]; ring
      · rw [h2, hzero, ← h1, hzero]; simp
      · rw [h2, Real.sign_of_pos hpos', ← h1, abs_of_pos hpos']; ring

/-- The hypotheses of `inner_mul_eq_inner_iff` are satisfiable away from the
degenerate case as well: three copies of the first standard basis vector make
`⟨x,z⟩ = 1` rather than `0`, so the second disjunct is the live one. -/
example :
    ‖EuclideanSpace.single (0 : Fin 1) (1 : ℝ)‖ = 1 ∧
      |inner (𝕜 := ℝ) (EuclideanSpace.single (0 : Fin 1) (1 : ℝ))
          (EuclideanSpace.single (0 : Fin 1) (1 : ℝ))|
        ≤ inner (𝕜 := ℝ) (EuclideanSpace.single (0 : Fin 1) (1 : ℝ))
            (EuclideanSpace.single (0 : Fin 1) (1 : ℝ)) := by
  refine ⟨by simp, ?_⟩
  simp

end Causal
end Transformer
