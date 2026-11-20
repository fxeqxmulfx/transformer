module

public import Transformer.Grokking.FlatSpectrum.Kernels
public import Mathlib.LinearAlgebra.Matrix.Hadamard
public import Mathlib.LinearAlgebra.Matrix.Rank

/-!
# Rank across a cut is at most multiplicative under entrywise products

Source: `review/flat_spectrum.md`, *Functions of n variables*, item NV7. A product layer
multiplies two functions of the input entrywise. Across a cut, each function is a matrix, and
their product is the entrywise (Hadamard) product of the two matrices, so its rank is at most the
product of the two ranks (`rank_hadamard_le`): the columns of `A ⊙ B` are entrywise products of
columns of `A` and of `B`, and lie in the span of the products of a basis of each column space. A
bilinear interaction through width `d` has rank at most `d` across its cut (NV2), whereas a tree
of products can multiply ranks.

The readout of a phase: `Re(ψ(f) conj ψ(c))` is `1` at `c = f` and less than `1` at every other
`c` (`re_stdAddChar_mul_star_self`, `re_stdAddChar_mul_star_lt`), so a stream that holds `ψ(f)`
exactly answers `f` by the largest logit.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Finset Matrix

/-- **Rank is at most multiplicative under entrywise products**:
`rank (A ⊙ B) ≤ rank A · rank B`. -/
theorem rank_hadamard_le {K m n : Type*} [Field K] [Fintype m] [Fintype n]
    (A B : Matrix m n K) : (A ⊙ B).rank ≤ A.rank * B.rank := by
  classical
  rw [rank_eq_finrank_span_cols, rank_eq_finrank_span_cols A, rank_eq_finrank_span_cols B]
  set S := Submodule.span K (Set.range A.col)
  set T := Submodule.span K (Set.range B.col)
  let bS := Module.finBasis K S
  let bT := Module.finBasis K T
  let f : Fin (Module.finrank K S) × Fin (Module.finrank K T) → m → K :=
    fun st => (bS st.1 : m → K) * (bT st.2 : m → K)
  have hexp {V : Submodule K (m → K)} (b : Module.Basis (Fin (Module.finrank K V)) K V)
      {v : m → K} (hv : v ∈ V) : v = ∑ s, b.repr ⟨v, hv⟩ s • (b s : m → K) := by
    have h := congrArg Subtype.val (b.sum_repr ⟨v, hv⟩)
    simp only [Submodule.coe_sum, Submodule.coe_smul] at h
    exact h.symm
  have hcol (j : n) : (A ⊙ B).col j ∈ Submodule.span K (Set.range f) := by
    have hA : A.col j ∈ S := Submodule.subset_span ⟨j, rfl⟩
    have hB : B.col j ∈ T := Submodule.subset_span ⟨j, rfl⟩
    have : (A ⊙ B).col j = A.col j * B.col j := by
      funext i
      simp [Matrix.col, hadamard_apply]
    rw [this, hexp bS hA, hexp bT hB, sum_mul_sum]
    refine Submodule.sum_mem _ fun s _ => Submodule.sum_mem _ fun t _ => ?_
    rw [smul_mul_smul_comm]
    exact Submodule.smul_mem _ _ (Submodule.subset_span ⟨(s, t), rfl⟩)
  calc Module.finrank K (Submodule.span K (Set.range (A ⊙ B).col))
      ≤ Module.finrank K (Submodule.span K (Set.range f)) :=
        Submodule.finrank_mono (Submodule.span_le.2 (Set.range_subset_iff.2 hcol))
    _ ≤ Fintype.card (Fin (Module.finrank K S) × Fin (Module.finrank K T)) :=
        finrank_range_le_card f
    _ = Module.finrank K S * Module.finrank K T := by simp

/-- The identity is its own entrywise square: `2 ≤ 2 · 2`. -/
example := rank_hadamard_le (1 : Matrix (Fin 2) (Fin 2) ℚ) 1

variable {p : ℕ} [NeZero p]

/-- The readout of a phase at the right answer: `Re(ψ(f) conj ψ(f)) = 1`. -/
lemma re_stdAddChar_mul_star_self (f : ZMod p) :
    (ZMod.stdAddChar f * star (ZMod.stdAddChar f)).re = 1 := by
  rw [mul_comm, star_stdAddChar_mul, sub_self, AddChar.map_zero_eq_one, Complex.one_re]

/-- **The readout of a phase is largest at the right answer**: `Re(ψ(f) conj ψ(c)) < 1` for
`c ≠ f`. -/
theorem re_stdAddChar_mul_star_lt {f c : ZMod p} (h : c ≠ f) :
    (ZMod.stdAddChar f * star (ZMod.stdAddChar c)).re < 1 := by
  rw [mul_comm, star_stdAddChar_mul]
  set w := ZMod.stdAddChar (f - c)
  have hw : ‖w‖ = 1 := by simp [w]
  have hne : w ≠ 1 := fun h1 => h (sub_eq_zero.1 (ZMod.injective_stdAddChar
    (h1.trans (AddChar.map_zero_eq_one _).symm))).symm
  have hle : w.re ≤ 1 := hw ▸ Complex.re_le_norm w
  refine lt_of_le_of_ne hle fun hre => hne ?_
  have hsq : w.re * w.re + w.im * w.im = 1 := by
    rw [← Complex.normSq_apply, Complex.normSq_eq_norm_sq, hw, one_pow]
  have him : w.im = 0 := by
    rw [hre] at hsq
    nlinarith [mul_self_nonneg w.im]
  exact Complex.ext (by rw [hre, Complex.one_re]) (by rw [him, Complex.one_im])

example := @re_stdAddChar_mul_star_lt 5 _ 1 0 (by decide)

end Transformer.Grokking.FlatSpectrum
