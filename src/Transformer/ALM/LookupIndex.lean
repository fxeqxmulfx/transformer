/-
# An exact lookup index, and what one query already decides

`Transformer.ALM.OrthVectors` shows that a single exact lookup decides a whole
Orthogonal Vectors instance for one left-hand vector.  To turn that into a
statement about *data structures* we need a name for the thing that answers
lookups.  This file gives one: an `NNIndex` is anything that returns an exact
argmax of the paraboloid score, together with a declared build and query cost.

Nothing here models computation.  `ans` is data and `ans_isGreatest` is its
specification; the costs are abstract reals, constrained only by whatever the
user of the structure proves about them.  What *is* proved here is the part
that is pure geometry: the procedure that queries the index once per left-hand
vector and inspects only the returned key decides Orthogonal Vectors
correctly (`NNIndex.reduce_iff`).

Source of the reduction: V. Vassilevska Williams, *On some fine-grained
questions in algorithms and complexity*, ICM 2018, §3.
-/

import Transformer.ALM.OrthVectors

namespace Transformer
namespace ALM

/-- An exact lookup index: for every finite family of keys and every query it
names a key of maximal score, and it declares what building and querying
cost.  The costs are abstract; only `ans_isGreatest` has content on its own. -/
structure NNIndex where
  /-- The key the index returns for a given family and query. -/
  ans : ∀ {m n : ℕ} [Nonempty (Fin n)], (Fin n → EucSpace m) → EucSpace m → Fin n
  /-- And it really is an argmax: this is exactness, not approximation. -/
  ans_isGreatest : ∀ {m n : ℕ} [Nonempty (Fin n)] (K : Fin n → EucSpace m)
    (q : EucSpace m) (j : Fin n), score q (K j) ≤ score q (K (ans K q))
  /-- Cost of building the index over `n` keys in dimension `m`. -/
  build : ℕ → ℕ → ℝ
  /-- Cost of one query against `n` keys in dimension `m`. -/
  query : ℕ → ℕ → ℝ

namespace NNIndex

variable (I : NNIndex) {d n : ℕ} [Nonempty (Fin n)]

/-- **The reduction as a procedure.**  Embed the right-hand vectors as keys,
then for each left-hand vector ask the index for its nearest key and test that
one key for orthogonality.  The remaining `n - 1` keys are never examined. -/
def reduce (A B : Fin n → BVec d) : Prop :=
  ∃ i, Orth (A i) (B (I.ans (fun j => kvec (B j)) (qvec (A i))))

/-- **The procedure is correct.**  Testing only the returned key loses
nothing: it is orthogonal to the query exactly when some key is.

This is the whole geometric content of the hardness argument.  Everything
downstream is cost accounting. -/
theorem reduce_iff (A B : Fin n → BVec d) :
    I.reduce A B ↔ ∃ i j, Orth (A i) (B j) := by
  constructor
  · rintro ⟨i, hi⟩
    exact ⟨i, _, hi⟩
  · rintro ⟨i, j, hij⟩
    refine ⟨i, ?_⟩
    exact (argmax_decides_ov (A i) B _
      (fun j => I.ans_isGreatest (fun j => kvec (B j)) (qvec (A i)) j)).mpr ⟨j, hij⟩

end NNIndex

/-! ### Brute force is an index

Scanning all keys satisfies the specification, with query cost `n · m`.  It
witnesses that `NNIndex` is inhabited, so nothing proved about indices is
vacuous.
-/

/-- The key of maximal score, chosen by exhaustive comparison. -/
noncomputable def bfAns {m n : ℕ} [Nonempty (Fin n)] (K : Fin n → EucSpace m)
    (q : EucSpace m) : Fin n :=
  (Finset.exists_max_image Finset.univ (fun j => score q (K j))
    Finset.univ_nonempty).choose

lemma bfAns_isGreatest {m n : ℕ} [Nonempty (Fin n)] (K : Fin n → EucSpace m)
    (q : EucSpace m) (j : Fin n) : score q (K j) ≤ score q (K (bfAns K q)) :=
  (Finset.exists_max_image Finset.univ (fun j => score q (K j))
    Finset.univ_nonempty).choose_spec.2 j (Finset.mem_univ j)

/-- Exhaustive scan, as an index: no preprocessing, linear query. -/
noncomputable def bruteForce : NNIndex where
  ans := fun K q => bfAns K q
  ans_isGreatest := fun K q j => bfAns_isGreatest K q j
  build := fun _ _ => 0
  query := fun n m => (n : ℝ) * (m : ℝ)

/-- The hypotheses of `NNIndex.reduce_iff` are satisfiable: an index exists,
and on a one-key instance the procedure answers `True`. -/
example : bruteForce.reduce (fun _ : Fin 1 => fun _ : Fin 1 => true)
    (fun _ : Fin 1 => fun _ : Fin 1 => false) := by
  rw [NNIndex.reduce_iff]
  exact ⟨0, 0, by simp [Orth, ip, bit]⟩

end ALM
end Transformer
