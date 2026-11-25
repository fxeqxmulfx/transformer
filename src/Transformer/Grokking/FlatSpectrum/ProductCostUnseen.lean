module

public import Transformer.Grokking.FlatSpectrum.ProductCostFlat

/-!
# The cheapest interpolant ignores what training never reads

Source: `review/flat_spectrum.md`, *Loops*, item LP4. Mask a matrix to a block `S × T` of its
rows and columns (`blockMask`). The masked matrix is cheaper in NV10's price, strictly unless the
mask changes nothing (`productCost_blockMask_lt`). Take a cheapest factorization `G diag(o) Wᴴ`
of the matrix and zero the rows of `G` outside `S` and those of `W` outside `T`: this writes the
masked matrix, and the weight drops by the squared norms of the rows removed. A nonzero entry
outside the block has a nonzero row of `G` or of `W` behind it, so the drop is strict.

Masked to the rows and the columns that a training set `Ω` reads, an interpolant of `Ω` still
interpolates. So an interpolant that is not zero outside them is beaten in price by one that is
(`exists_productCost_lt`), and one of least price vanishes outside them
(`eq_zero_of_isMinOn_productCost`): on an operand value that no training pair contains it
answers `0`, whatever the task. For a kernel with no zero entry, such as the cubic kernel, it is
wrong on every entry there (`ne_of_isMinOn_productCost`). This is the negative analog of
Theorem A for coordinates that the training set never reads.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix

variable {m n : Type*}

/-- The matrices that agree with `K` on `Ω`. -/
def interpolants (K : Matrix m n ℂ) (Ω : Set (m × n)) : Set (Matrix m n ℂ) :=
  {X | ∀ ij ∈ Ω, X ij.1 ij.2 = K ij.1 ij.2}

variable [DecidableEq m] [DecidableEq n]

/-- The matrix `X` masked to the block `S × T`: zero outside it. -/
def blockMask (S : Finset m) (T : Finset n) (X : Matrix m n ℂ) : Matrix m n ℂ :=
  fun a b => if a ∈ S ∧ b ∈ T then X a b else 0

/-- The rows of `G` outside `S` set to zero. -/
def rowMask {k : Type*} (S : Finset m) (G : Matrix m k ℂ) : Matrix m k ℂ :=
  fun a u => if a ∈ S then G a u else 0

/-- Masking the rows of both factors masks the matrix they write. -/
lemma rowMask_mul {k : Type*} [Fintype k] [DecidableEq k] (S : Finset m) (T : Finset n)
    (G : Matrix m k ℂ) (o : k → ℂ) (W : Matrix n k ℂ) :
    rowMask S G * diagonal o * (rowMask T W)ᴴ = blockMask S T (G * diagonal o * Wᴴ) := by
  ext a b
  by_cases ha : a ∈ S <;> by_cases hb : b ∈ T <;>
    simp [rowMask, blockMask, Matrix.mul_apply, ha, hb]

lemma sq_norm_rowMask_le {k : Type*} (S : Finset m) (G : Matrix m k ℂ) (a : m) (u : k) :
    ‖rowMask S G a u‖ ^ 2 ≤ ‖G a u‖ ^ 2 := by
  gcongr
  simp only [rowMask]
  split_ifs <;> simp

variable [Fintype m] [Fintype n]

lemma frobeniusSq_rowMask_le {k : Type*} [Fintype k] (S : Finset m) (G : Matrix m k ℂ) :
    frobeniusSq (rowMask S G) ≤ frobeniusSq G :=
  Finset.sum_le_sum fun a _ => Finset.sum_le_sum fun u _ => sq_norm_rowMask_le S G a u

lemma frobeniusSq_rowMask_lt {k : Type*} [Fintype k] (S : Finset m) (G : Matrix m k ℂ) {a : m}
    (ha : a ∉ S) {u : k} (hu : G a u ≠ 0) : frobeniusSq (rowMask S G) < frobeniusSq G := by
  refine Finset.sum_lt_sum (fun a' _ => Finset.sum_le_sum fun u' _ => sq_norm_rowMask_le S G a' u')
    ⟨a, Finset.mem_univ _, Finset.sum_lt_sum (fun u' _ => sq_norm_rowMask_le S G a u')
      ⟨u, Finset.mem_univ _, ?_⟩⟩
  rw [show rowMask S G a u = 0 by simp [rowMask, ha], norm_zero, zero_pow two_ne_zero]
  exact pow_pos (norm_pos_iff.2 hu) 2

/-- **Masking is cheaper**: masked to a block `S × T`, a matrix costs strictly less in NV10's
price, unless the mask changes nothing. -/
theorem productCost_blockMask_lt {X : Matrix m n ℂ} (S : Finset m) (T : Finset n)
    (h : blockMask S T X ≠ X) : productCost (blockMask S T X) < productCost X := by
  obtain ⟨r, G, o, W, hX, hc⟩ := (productCost_isLeast X).1
  obtain ⟨a, b, hab⟩ : ∃ a b, blockMask S T X a b ≠ X a b := by
    by_contra hall
    simp only [not_exists, ne_eq, not_not] at hall
    exact h (Matrix.ext hall)
  have hout : ¬(a ∈ S ∧ b ∈ T) := fun hin => hab (by simp [blockMask, hin])
  have hXab : X a b ≠ 0 := fun h0 => hab (by simp [blockMask, hout, h0])
  obtain ⟨u, hu⟩ : ∃ u, G a u * o u * star (W b u) ≠ 0 := by
    by_contra hall
    simp only [not_exists, ne_eq, not_not] at hall
    refine hXab ?_
    rw [hX, Matrix.mul_apply]
    exact Finset.sum_eq_zero fun u _ => by
      rw [Matrix.mul_diagonal, Matrix.conjTranspose_apply]
      exact hall u
  have hG : G a u ≠ 0 := fun h0 => hu (by rw [h0, zero_mul, zero_mul])
  have hW : W b u ≠ 0 := fun h0 => hu (by rw [h0, star_zero, mul_zero])
  have hlt : frobeniusSq (rowMask S G) + frobeniusSq (rowMask T W) <
      frobeniusSq G + frobeniusSq W := by
    by_cases ha : a ∈ S
    · exact add_lt_add_of_le_of_lt (frobeniusSq_rowMask_le S G)
        (frobeniusSq_rowMask_lt T W (fun hb => hout ⟨ha, hb⟩) hW)
    · exact add_lt_add_of_lt_of_le (frobeniusSq_rowMask_lt S G ha hG) (frobeniusSq_rowMask_le T W)
  calc productCost (blockMask S T X)
      ≤ (frobeniusSq (rowMask S G) + ∑ u, ‖o u‖ ^ 2 + frobeniusSq (rowMask T W)) / 2 := by
        rw [hX, ← rowMask_mul]
        exact productCost_le _ _ _
    _ < (frobeniusSq G + ∑ u, ‖o u‖ ^ 2 + frobeniusSq W) / 2 := by linarith
    _ = productCost X := hc.symm

/-- **An interpolant that reads an unseen row or column is beaten in price**: masked to the rows
and the columns that `Ω` reads, it still interpolates, answers `0` there, and costs less. -/
theorem exists_productCost_lt {K X : Matrix m n ℂ} {Ω : Set (m × n)} (hX : X ∈ interpolants K Ω)
    {a : m} {b : n} (hab : (∀ b', (a, b') ∉ Ω) ∨ ∀ a', (a', b) ∉ Ω) (h : X a b ≠ 0) :
    ∃ X' ∈ interpolants K Ω, X' a b = 0 ∧ productCost X' < productCost X := by
  classical
  let S : Finset m := Finset.univ.filter fun a' => ∃ b', (a', b') ∈ Ω
  let T : Finset n := Finset.univ.filter fun b' => ∃ a', (a', b') ∈ Ω
  have hout : ¬(a ∈ S ∧ b ∈ T) := by
    rintro ⟨ha, hb⟩
    rcases hab with hab | hab
    · obtain ⟨b', hb'⟩ := (Finset.mem_filter.1 ha).2
      exact hab b' hb'
    · obtain ⟨a', ha'⟩ := (Finset.mem_filter.1 hb).2
      exact hab a' ha'
  refine ⟨blockMask S T X, fun ij hij => ?_, by simp [blockMask, hout],
    productCost_blockMask_lt S T fun he => h ?_⟩
  · have hin : ij.1 ∈ S ∧ ij.2 ∈ T :=
      ⟨Finset.mem_filter.2 ⟨Finset.mem_univ _, ij.2, hij⟩,
        Finset.mem_filter.2 ⟨Finset.mem_univ _, ij.1, hij⟩⟩
    simp [blockMask, hin, hX ij hij]
  · rw [← he]
    simp [blockMask, hout]

/-- **The cheapest interpolant ignores what training never reads**: an interpolant of least
price vanishes on every row and every column that no pair of `Ω` reads. -/
theorem eq_zero_of_isMinOn_productCost {K X : Matrix m n ℂ} {Ω : Set (m × n)}
    (hX : X ∈ interpolants K Ω) (hmin : IsMinOn productCost (interpolants K Ω) X) {a : m} {b : n}
    (hab : (∀ b', (a, b') ∉ Ω) ∨ ∀ a', (a', b) ∉ Ω) : X a b = 0 := by
  by_contra h
  obtain ⟨X', hX', -, hlt⟩ := exists_productCost_lt hX hab h
  exact absurd hlt (not_lt.2 (isMinOn_iff.1 hmin X' hX'))

/-- For a kernel with no zero entry, an interpolant of least price is wrong on every entry of a
row or a column that no pair of `Ω` reads. -/
theorem ne_of_isMinOn_productCost {K X : Matrix m n ℂ} (hK : ∀ a b, K a b ≠ 0)
    {Ω : Set (m × n)} (hX : X ∈ interpolants K Ω)
    (hmin : IsMinOn productCost (interpolants K Ω) X) {a : m} {b : n}
    (hab : (∀ b', (a, b') ∉ Ω) ∨ ∀ a', (a', b) ∉ Ω) : X a b ≠ K a b := by
  rw [eq_zero_of_isMinOn_productCost hX hmin hab]
  exact (hK a b).symm

/-- Trained on the pairs with `a ≠ 2`, an interpolant of least price of the cubic kernel at
`p = 3` is wrong on the whole row `a = 2`. -/
example {X : Matrix (ZMod 3) (ZMod 3) ℂ} (hX : X ∈ interpolants (cubicKernel 3) {ij | ij.1 ≠ 2})
    (hmin : IsMinOn productCost (interpolants (cubicKernel 3) {ij | ij.1 ≠ 2}) X) (b : ZMod 3) :
    X 2 b ≠ cubicKernel 3 2 b :=
  ne_of_isMinOn_productCost (cubicKernel_apply_ne_zero 3) hX hmin (Or.inl fun _ h => h rfl)

end Transformer.Grokking.FlatSpectrum
