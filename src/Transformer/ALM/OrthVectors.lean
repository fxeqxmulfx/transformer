/-
# Orthogonal Vectors reduce to one lookup

The reason exact nearest-neighbour search is believed to be hard in high
dimension is not a theorem but a reduction: Orthogonal Vectors reduces to it,
and Orthogonal Vectors inherits a quadratic lower bound from the Strong
Exponential Time Hypothesis.

* Impagliazzo, Paturi, *On the complexity of k-SAT*, JCSS 62 (2001) — SETH.
* R. Williams, *A new algorithm for optimal 2-constraint satisfaction and its
  implications*, Theoret. Comput. Sci. 348 (2005) — SETH implies that
  Orthogonal Vectors needs `n^{2-o(1)}` time.
* V. Vassilevska Williams, *On some fine-grained questions in algorithms and
  complexity*, ICM 2018, §3 — the Orthogonal Vectors conjecture and its
  consequences for nearest-neighbour search.

This file formalizes the geometric half of that chain, which needs no model
of computation: an explicit embedding of a `0/1` instance into the paraboloid
scores of `Transformer.ALM.Defs`, under which the **single** key returned by
an exact lookup already decides the whole instance.  The cost accounting that
turns this into a lower bound is `Transformer.ALM.Hardness`.

The embedding is the standard one: a key `b` is sent to `(1 - b, b)`, whose
complement block equalizes all key norms and reverses the direction of
optimization, and a query `a` to `(a, 0)`.
-/

import Transformer.ALM.Basic

open scoped BigOperators

namespace Transformer
namespace ALM

variable {d : ℕ}

/-! ### `0/1` vectors -/

/-- A `0/1` vector of length `d`, the input format of Orthogonal Vectors. -/
abbrev BVec (d : ℕ) : Type := Fin d → Bool

/-- The real coordinate of a bit: `1` for `true`, `0` for `false`. -/
def bit (x : Bool) : ℝ := if x then 1 else 0

lemma bit_nonneg (x : Bool) : 0 ≤ bit x := by
  unfold bit; split <;> norm_num

lemma bit_sq (x : Bool) : bit x ^ 2 = bit x := by
  unfold bit; split <;> norm_num

lemma one_sub_bit_sq (x : Bool) : (1 - bit x) ^ 2 = 1 - bit x := by
  unfold bit; split <;> norm_num

/-- The inner product of two `0/1` vectors, as a real number. -/
def ip (a b : BVec d) : ℝ := ∑ i, bit (a i) * bit (b i)

/-- The Hamming weight of a `0/1` vector. -/
def wt (a : BVec d) : ℝ := ∑ i, bit (a i)

/-- The Orthogonal Vectors relation: no coordinate carries a `1` in both. -/
def Orth (a b : BVec d) : Prop := ip a b = 0

/-- Inner products of `0/1` vectors are nonnegative, with `0` the extreme
value — this is the entire reason the reduction points the right way. -/
lemma ip_nonneg (a b : BVec d) : 0 ≤ ip a b :=
  Finset.sum_nonneg fun _ _ => mul_nonneg (bit_nonneg _) (bit_nonneg _)

/-! ### The embedding -/

/-- The query embedding `a ↦ (a, 0)`. -/
def qvec (a : BVec d) : EucSpace (d + d) :=
  WithLp.toLp 2 (Fin.append (fun i => bit (a i)) (fun _ => 0))

/-- The key embedding `b ↦ (1 - b, b)`.  The complement block gives every key
the same norm, so the score depends on `b` only through `⟪a, 1 - b⟫`, and
maximizing that is minimizing `⟪a, b⟫`. -/
def kvec (b : BVec d) : EucSpace (d + d) :=
  WithLp.toLp 2 (Fin.append (fun i => 1 - bit (b i)) (fun i => bit (b i)))

/-- The embedded inner product decodes `wt a - ip a b`. -/
lemma inner_kvec_qvec (a b : BVec d) :
    inner (𝕜 := ℝ) (kvec b) (qvec a) = wt a - ip a b := by
  simp only [kvec, qvec, PiLp.inner_apply, RCLike.inner_apply, conj_trivial,
    Fin.sum_univ_add, Fin.append_left, Fin.append_right]
  rw [wt, ip, ← Finset.sum_sub_distrib]
  simp only [zero_mul, Finset.sum_const_zero, add_zero]
  exact Finset.sum_congr rfl fun i _ => by ring

/-- Every embedded key has squared norm `d`: this is what the complement
block buys. -/
lemma norm_sq_kvec (b : BVec d) : ‖kvec b‖ ^ 2 = (d : ℝ) := by
  rw [norm_sq_eq_sum]
  simp only [kvec, WithLp.ofLp_toLp, Fin.sum_univ_add, Fin.append_left,
    Fin.append_right, one_sub_bit_sq, bit_sq]
  rw [← Finset.sum_add_distrib]
  simp

/-! ### One lookup decides the instance -/

/-- **The score decodes the inner product.**  Under the embedding the
paraboloid score of `Transformer.ALM.Defs` is an exact affine function of
`ip a b`, with a negative coefficient. -/
theorem score_qvec_kvec (a b : BVec d) :
    score (qvec a) (kvec b) = 2 * wt a - (d : ℝ) - 2 * ip a b := by
  unfold score
  rw [inner_kvec_qvec, norm_sq_kvec]
  ring

/-- `2 * wt a - d` is an upper bound on the score of every key. -/
theorem score_qvec_le (a b : BVec d) :
    score (qvec a) (kvec b) ≤ 2 * wt a - (d : ℝ) := by
  rw [score_qvec_kvec]
  have := ip_nonneg a b
  linarith

/-- The bound is attained exactly at the orthogonal keys. -/
theorem score_qvec_eq_iff_orth (a b : BVec d) :
    score (qvec a) (kvec b) = 2 * wt a - (d : ℝ) ↔ Orth a b := by
  rw [score_qvec_kvec, Orth]
  constructor <;> intro h <;> linarith

/-- **One lookup decides Orthogonal Vectors.**  If `i₀` maximizes the ALM
score among the embedded keys — that is, if it is what an exact
nearest-neighbour index returns — then the single key it returns is
orthogonal to the query exactly when *some* key is.

So a data structure answering exact lookups answers Orthogonal Vectors with
one query per left-hand vector, and no examination of the remaining keys. -/
theorem argmax_decides_ov {n : ℕ} (a : BVec d) (B : Fin n → BVec d) (i₀ : Fin n)
    (hmax : ∀ j, score (qvec a) (kvec (B j)) ≤ score (qvec a) (kvec (B i₀))) :
    Orth a (B i₀) ↔ ∃ j, Orth a (B j) := by
  refine ⟨fun h => ⟨i₀, h⟩, ?_⟩
  rintro ⟨j, hj⟩
  rw [← score_qvec_eq_iff_orth] at hj ⊢
  have h₁ := hmax j
  have h₂ := score_qvec_le a (B i₀)
  linarith

/-! ### Both outcomes are reachable -/

/-- A positive instance: a single key, orthogonal to the query.  With one key
the maximality hypothesis of `argmax_decides_ov` is vacuous, so this witnesses
that its hypotheses are satisfiable together with `Orth`. -/
example : ∃ (a : BVec 1) (B : Fin 1 → BVec 1) (i₀ : Fin 1),
    (∀ j, score (qvec a) (kvec (B j)) ≤ score (qvec a) (kvec (B i₀))) ∧
      Orth a (B i₀) := by
  refine ⟨fun _ => true, fun _ _ => false, 0, fun j => ?_, ?_⟩
  · rw [Subsingleton.elim j 0]
  · simp [Orth, ip, bit]

/-- A negative instance: the same shape, but the key meets the query. -/
example : ∃ (a : BVec 1) (B : Fin 1 → BVec 1) (i₀ : Fin 1),
    (∀ j, score (qvec a) (kvec (B j)) ≤ score (qvec a) (kvec (B i₀))) ∧
      ¬ Orth a (B i₀) := by
  refine ⟨fun _ => true, fun _ _ => true, 0, fun j => ?_, ?_⟩
  · rw [Subsingleton.elim j 0]
  · simp [Orth, ip, bit]

end ALM
end Transformer
