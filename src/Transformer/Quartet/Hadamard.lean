/-
# The randomized Hadamard transform and its orthogonality

arXiv:2601.22813v2, "Quartet II: Accurate LLM Pre-Training in NVFP4 by
Improved Unbiased Gradient Estimation" (ICML 2026), §3.2.

The rotation of §3.2 is the Walsh–Hadamard matrix with random signs, written
on the two-level index of a tensor: `2^k` groups of `16` entries, so
`hadamard` is the tensor product of the transform on the group index and the
transform on the entry index.  The rotation group is the whole tensor, which
the paper allows — "any multiple of the quantization group size 16 is
valid".

Everything here follows from the orthogonality of the Walsh characters
(`Transformer.Quartet.Walsh`): the matrix is symmetric (`hadamard_symm`), its
rows are orthonormal (`sum_hadamard_mul`), it is an involution
(`sum_hadamard_sum_hadamard`), the randomized transform is invertible
(`rhtInv_rht`), and it preserves inner products (`sum_rht_mul_rht`) — the
last being why §5 can say that the outputs of the NVFP4 GEMMs "need no
further processing, as the rotations cancel out along the inner GEMM
dimensions".
-/

import Transformer.Quartet.Walsh
import Mathlib.Algebra.BigOperators.Field
import Mathlib.Analysis.Real.Sqrt

namespace Transformer
namespace Quartet

variable {k : ℕ}

/-- The Walsh–Hadamard matrix on the `2^{k+4}` entries of a tensor laid out as
`2^k` groups of `16`: the sign is `(-1)` to the number of binary digits the two
indices share, over `√(2^{k+4})` (§3.2, the `RHT`). -/
noncomputable def hadamard (k : ℕ) (i i' : Fin (2 ^ k)) (j j' : Fin 16) : ℝ :=
  (-1) ^ (((Finset.range k).filter fun b => i.val.testBit b && i'.val.testBit b).card +
      ((Finset.range 4).filter fun b => j.val.testBit b && j'.val.testBit b).card) /
    Real.sqrt (2 ^ (k + 4))

/-- The randomized Hadamard transform `RHT(x, ω)` (§3.2): flip the sign of
every entry according to the seed `ε`, then apply the Hadamard matrix. -/
noncomputable def rht (k : ℕ) (ε : Fin (2 ^ k) → Fin 16 → Bool) (x : Fin (2 ^ k) → Fin 16 → ℝ)
    (i : Fin (2 ^ k)) (j : Fin 16) : ℝ :=
  ∑ i', ∑ j', hadamard k i i' j j' * (if ε i' j' then -x i' j' else x i' j')

/-- Its inverse `RHT⁻¹(·, ω)`: the Hadamard matrix is its own inverse, so only
the signs have to be undone afterwards. -/
noncomputable def rhtInv (k : ℕ) (ε : Fin (2 ^ k) → Fin 16 → Bool)
    (y : Fin (2 ^ k) → Fin 16 → ℝ) (i : Fin (2 ^ k)) (j : Fin 16) : ℝ :=
  (if ε i j then -1 else 1) * ∑ i', ∑ j', hadamard k i i' j j' * y i' j'

/-- The Hadamard matrix is the product of the Walsh character on the group
index and the Walsh character on the entry index. -/
theorem hadamard_eq (i i' : Fin (2 ^ k)) (j j' : Fin 16) :
    hadamard k i i' j j' =
      walsh k i.val i'.val * walsh 4 j.val j'.val / Real.sqrt (2 ^ (k + 4)) := by
  unfold hadamard walsh
  rw [pow_add]

/-- And so it is symmetric. -/
theorem hadamard_symm (i i' : Fin (2 ^ k)) (j j' : Fin 16) :
    hadamard k i i' j j' = hadamard k i' i j' j := by
  rw [hadamard_eq, hadamard_eq, walsh_comm k i.val i'.val, walsh_comm 4 j.val j'.val]

/-- **The rows of the Hadamard matrix are orthonormal**: this is
`sum_walsh_mul_fin` on the two index levels at once, the `√(2^{k+4})` of §3.2
normalizing the `2^{k+4}` the two orthogonality sums produce. -/
theorem sum_hadamard_mul (i i'' : Fin (2 ^ k)) (j j'' : Fin 16) :
    ∑ p : Fin (2 ^ k) × Fin 16, hadamard k i p.1 j p.2 * hadamard k p.1 i'' p.2 j'' =
      if (i, j) = (i'', j'') then (1 : ℝ) else 0 := by
  have hentry : ∀ p : Fin (2 ^ k) × Fin 16,
      hadamard k i p.1 j p.2 * hadamard k p.1 i'' p.2 j'' =
        (walsh k i.val p.1.val * walsh k i''.val p.1.val) *
          (walsh 4 j.val p.2.val * walsh 4 j''.val p.2.val) / (2 ^ (k + 4) : ℝ) := by
    intro p
    rw [hadamard_eq, hadamard_eq, walsh_comm k p.1.val i''.val, walsh_comm 4 p.2.val j''.val,
      div_mul_div_comm, Real.mul_self_sqrt (by positivity)]
    ring
  rw [Finset.sum_congr rfl fun p _ => hentry p, ← Finset.sum_div]
  simp only [Fintype.sum_prod_type]
  rw [← Finset.sum_mul_sum, sum_walsh_mul_fin rfl i i'',
    sum_walsh_mul_fin (show (16 : ℕ) = 2 ^ 4 by norm_num) j j'']
  have hpos : (0 : ℝ) < 2 ^ k := by positivity
  rcases eq_or_ne i i'' with hi | hi
  · rcases eq_or_ne j j'' with hj | hj
    · rw [ite_eq_left hi, ite_eq_left hj, ite_eq_left (by rw [hi, hj] : (i, j) = (i'', j'')),
        pow_add]
      push_cast
      field_simp
      norm_num
    · rw [ite_eq_right hj, ite_eq_right fun h => hj (congrArg Prod.snd h)]
      simp
  · rw [ite_eq_right hi, ite_eq_right fun h => hi (congrArg Prod.fst h)]
    simp

/-- **The Hadamard matrix is an involution on tensors**: applying it twice
returns the tensor. -/
theorem sum_hadamard_sum_hadamard (y : Fin (2 ^ k) → Fin 16 → ℝ) (i : Fin (2 ^ k)) (j : Fin 16) :
    ∑ i' : Fin (2 ^ k), ∑ j' : Fin 16, hadamard k i i' j j' *
        (∑ i'' : Fin (2 ^ k), ∑ j'' : Fin 16, hadamard k i' i'' j' j'' * y i'' j'') = y i j := by
  calc ∑ i' : Fin (2 ^ k), ∑ j' : Fin 16, hadamard k i i' j j' *
        (∑ i'' : Fin (2 ^ k), ∑ j'' : Fin 16, hadamard k i' i'' j' j'' * y i'' j'')
      = ∑ p : Fin (2 ^ k) × Fin 16, ∑ q : Fin (2 ^ k) × Fin 16,
          hadamard k i p.1 j p.2 * hadamard k p.1 q.1 p.2 q.2 * y q.1 q.2 := by
        rw [Fintype.sum_prod_type]
        refine Finset.sum_congr rfl fun i' _ => Finset.sum_congr rfl fun j' _ => ?_
        rw [Fintype.sum_prod_type, Finset.mul_sum]
        exact Finset.sum_congr rfl fun i'' _ =>
          (Finset.mul_sum _ _ _).trans (Finset.sum_congr rfl fun j'' _ => (mul_assoc _ _ _).symm)
    _ = ∑ q : Fin (2 ^ k) × Fin 16, ∑ p : Fin (2 ^ k) × Fin 16,
          hadamard k i p.1 j p.2 * hadamard k p.1 q.1 p.2 q.2 * y q.1 q.2 := Finset.sum_comm
    _ = ∑ q : Fin (2 ^ k) × Fin 16,
          (∑ p : Fin (2 ^ k) × Fin 16, hadamard k i p.1 j p.2 * hadamard k p.1 q.1 p.2 q.2) *
            y q.1 q.2 :=
        Finset.sum_congr rfl fun q _ => (Finset.sum_mul _ _ _).symm
    _ = ∑ q : Fin (2 ^ k) × Fin 16, (if (i, j) = q then (1 : ℝ) else 0) * y q.1 q.2 :=
        Finset.sum_congr rfl fun q _ => by rw [sum_hadamard_mul i q.1 j q.2]
    _ = y i j := by simp

/-- **The rotation is invertible**, which is what lets §3.2 speak of
unbiasedness "in rotated space". -/
theorem rhtInv_rht (ε : Fin (2 ^ k) → Fin 16 → Bool) (x : Fin (2 ^ k) → Fin 16 → ℝ)
    (i : Fin (2 ^ k)) (j : Fin 16) : rhtInv k ε (rht k ε x) i j = x i j := by
  unfold rhtInv rht
  rw [sum_hadamard_sum_hadamard (fun a b => if ε a b then -x a b else x a b) i j]
  cases ε i j <;> simp

/-- **The Hadamard matrix preserves inner products**: this is `sum_hadamard_mul`
summed against two tensors, the Parseval identity of the `RHT`. -/
theorem sum_hadamard_mul_sum_hadamard_mul (u v : Fin (2 ^ k) → Fin 16 → ℝ) :
    ∑ i : Fin (2 ^ k), ∑ j : Fin 16,
        (∑ i' : Fin (2 ^ k), ∑ j' : Fin 16, hadamard k i i' j j' * u i' j') *
          (∑ i' : Fin (2 ^ k), ∑ j' : Fin 16, hadamard k i i' j j' * v i' j') =
      ∑ i : Fin (2 ^ k), ∑ j : Fin 16, u i j * v i j := by
  have hsum : ∀ f : Fin (2 ^ k) → Fin 16 → ℝ,
      ∑ i : Fin (2 ^ k), ∑ j : Fin 16, f i j = ∑ p : Fin (2 ^ k) × Fin 16, f p.1 p.2 :=
    fun f => (Fintype.sum_prod_type fun p : Fin (2 ^ k) × Fin 16 => f p.1 p.2).symm
  calc ∑ i : Fin (2 ^ k), ∑ j : Fin 16,
        (∑ i' : Fin (2 ^ k), ∑ j' : Fin 16, hadamard k i i' j j' * u i' j') *
          (∑ i' : Fin (2 ^ k), ∑ j' : Fin 16, hadamard k i i' j j' * v i' j')
      = ∑ i : Fin (2 ^ k), ∑ j : Fin 16, ∑ q : Fin (2 ^ k) × Fin 16,
          ∑ r : Fin (2 ^ k) × Fin 16,
            hadamard k i q.1 j q.2 * u q.1 q.2 * (hadamard k i r.1 j r.2 * v r.1 r.2) :=
        Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => by
          rw [hsum fun i' j' => hadamard k i i' j j' * u i' j',
            hsum fun i' j' => hadamard k i i' j j' * v i' j', Finset.sum_mul_sum]
    _ = ∑ p : Fin (2 ^ k) × Fin 16, ∑ q : Fin (2 ^ k) × Fin 16, ∑ r : Fin (2 ^ k) × Fin 16,
          hadamard k p.1 q.1 p.2 q.2 * u q.1 q.2 * (hadamard k p.1 r.1 p.2 r.2 * v r.1 r.2) :=
        hsum fun i j => ∑ q : Fin (2 ^ k) × Fin 16, ∑ r : Fin (2 ^ k) × Fin 16,
          hadamard k i q.1 j q.2 * u q.1 q.2 * (hadamard k i r.1 j r.2 * v r.1 r.2)
    _ = ∑ q : Fin (2 ^ k) × Fin 16, ∑ r : Fin (2 ^ k) × Fin 16, ∑ p : Fin (2 ^ k) × Fin 16,
          hadamard k p.1 q.1 p.2 q.2 * u q.1 q.2 * (hadamard k p.1 r.1 p.2 r.2 * v r.1 r.2) := by
        rw [Finset.sum_comm]
        exact Finset.sum_congr rfl fun _ _ => Finset.sum_comm
    _ = ∑ q : Fin (2 ^ k) × Fin 16, u q.1 q.2 * v q.1 q.2 := by
        refine Finset.sum_congr rfl fun q _ => ?_
        have hr : ∀ r : Fin (2 ^ k) × Fin 16,
            (∑ p : Fin (2 ^ k) × Fin 16, hadamard k p.1 q.1 p.2 q.2 * u q.1 q.2 *
                (hadamard k p.1 r.1 p.2 r.2 * v r.1 r.2))
              = u q.1 q.2 * v r.1 r.2 * (if (q.1, q.2) = (r.1, r.2) then (1 : ℝ) else 0) := by
          intro r
          rw [← sum_hadamard_mul q.1 r.1 q.2 r.2, Finset.mul_sum]
          exact Finset.sum_congr rfl fun p _ => by
            rw [hadamard_symm q.1 p.1 q.2 p.2]; ring
        rw [Finset.sum_congr rfl fun r _ => hr r]
        simp
    _ = ∑ i : Fin (2 ^ k), ∑ j : Fin 16, u i j * v i j :=
        (hsum fun i j => u i j * v i j).symm

/-- **The rotation cancels along the inner dimension of a product**: rotating
both factors with the same seed leaves every inner product where it was.  This
is why §5 can say that the outputs of the NVFP4 GEMMs "need no further
processing, as the rotations cancel out along the inner GEMM dimensions". -/
theorem sum_rht_mul_rht (ε : Fin (2 ^ k) → Fin 16 → Bool) (x y : Fin (2 ^ k) → Fin 16 → ℝ) :
    ∑ i, ∑ j, rht k ε x i j * rht k ε y i j = ∑ i, ∑ j, x i j * y i j := by
  refine (sum_hadamard_mul_sum_hadamard_mul (k := k)
    (fun a b => if ε a b then -x a b else x a b)
    (fun a b => if ε a b then -y a b else y a b)).trans ?_
  refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
  cases ε i j <;> simp

end Quartet
end Transformer
