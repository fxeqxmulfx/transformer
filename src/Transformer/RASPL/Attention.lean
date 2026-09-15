/-
# Compiling a RASP-L aggregation into an attention layer

Zhou et al. — arXiv:2310.16028v1, "What Algorithms can Transformers Learn?",
Appendix C ("RASP-L Aggregations").

The appendix compiles `select`/`aggr` into one attention layer: the predicate
becomes a pre-softmax matrix through one-hot encodings, and the `max`
reduction — which is not an average, and so not what a softmax computes —
becomes a `mean` aggregation against a shifted matrix.

Three statements:

* `sum_oneHot_mul` is the encoding, "explicitly, this is by one-hot
  encoding", which realizes `M_{i,j} = P(q_i, k_j)` as an inner product of
  two `{0,1}`-valued embeddings;
* `Constructable.add` is the appendix's Lemma ("Constructability"), "which is
  trivial to prove": constructable pre-softmax matrices are closed under
  linear combinations;
* `argmax_shift` is what the lemma is for — that against `Z + 2|Σ|·M`, the
  0-temperature softmax selects the largest *selected* value, which is the
  `max` reduction.
-/

import Transformer.Basic

namespace Transformer
namespace RASPL

variable {V : Type*} [Fintype V] [DecidableEq V]

/-- **The one-hot encoding.**  Against the one-hot encoding of the query and
the predicate row of the key, the inner product is the predicate itself:
`⟨Q_{i,·}, K_{j,·}⟩ = P(q_i, k_j)`. -/
theorem sum_oneHot_mul (P : V → V → Bool) (a b : V) :
    (∑ m : V, (if a = m then (1 : ℝ) else 0) * (if P m b then (1 : ℝ) else 0))
      = if P a b then (1 : ℝ) else 0 := by
  rw [Finset.sum_eq_single a]
  · rw [ite_eq_left rfl, one_mul]
  · intro m _ hm
    rw [ite_eq_right (fun h : a = m => hm h.symm), zero_mul]
  · intro h
    exact absurd (Finset.mem_univ a) h

variable {T d : ℕ}

/-- A pre-softmax matrix is *constructable* from the embeddings `X` when some
pair of key and query projections produces it at the attention layer
(Appendix C: "there exist transformer weights which produce `M` at the
attention-layer (pre-softmax)"). -/
def Constructable (X : Fin T → Fin d → ℝ) (M : Fin T → Fin T → ℝ) : Prop :=
  ∃ (e : ℕ) (Q K : Fin e → Fin d → ℝ), ∀ i j : Fin T,
    M i j = ∑ a : Fin e, (∑ b, Q a b * X i b) * (∑ b, K a b * X j b)

/-- **The Constructability Lemma.**  Any linear combination of constructable
matrices is constructable: stack the two projections and scale the query
side. -/
theorem Constructable.add {X : Fin T → Fin d → ℝ} {M₁ M₂ : Fin T → Fin T → ℝ}
    (h₁ : Constructable X M₁) (h₂ : Constructable X M₂) (α β : ℝ) :
    Constructable X (fun i j => α * M₁ i j + β * M₂ i j) := by
  obtain ⟨e₁, Q₁, K₁, hQK₁⟩ := h₁
  obtain ⟨e₂, Q₂, K₂, hQK₂⟩ := h₂
  refine ⟨e₁ + e₂, Fin.addCases (fun a b => α * Q₁ a b) (fun a b => β * Q₂ a b),
    Fin.addCases (fun a b => K₁ a b) (fun a b => K₂ a b), fun i j => ?_⟩
  rw [Fin.sum_univ_add]
  simp only [Fin.addCases_left, Fin.addCases_right]
  rw [hQK₁ i j, hQK₂ i j, Finset.mul_sum, Finset.mul_sum]
  refine congrArg₂ (· + ·) (Finset.sum_congr rfl fun a _ => ?_)
    (Finset.sum_congr rfl fun a _ => ?_)
  · have h : (∑ x, α * Q₁ a x * X i x) = α * ∑ x, Q₁ a x * X i x := by
      rw [Finset.mul_sum]
      exact Finset.sum_congr rfl fun x _ => mul_assoc _ _ _
    rw [h, mul_assoc]
  · have h : (∑ x, β * Q₂ a x * X i x) = β * ∑ x, Q₂ a x * X i x := by
      rw [Finset.mul_sum]
      exact Finset.sum_congr rfl fun x _ => mul_assoc _ _ _
    rw [h, mul_assoc]

/-- The value matrix `Z_{i,j} = v_j` is constructable as soon as the
embedding carries the value at one coordinate and a constant at another —
"the matrix `Z` is clearly constructable in the sense of the Lemma above". -/
theorem constructable_value {X : Fin T → Fin d → ℝ} {v : Fin T → ℝ} {b₀ b₁ : Fin d}
    (hv : ∀ j, X j b₀ = v j) (hone : ∀ i, X i b₁ = 1) :
    Constructable X (fun _ j => v j) := by
  refine ⟨1, fun _ b => if b = b₁ then 1 else 0, fun _ b => if b = b₀ then 1 else 0,
    fun i j => ?_⟩
  rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, one_smul,
    Finset.sum_eq_single b₁, Finset.sum_eq_single b₀]
  · rw [ite_eq_left rfl, ite_eq_left rfl, one_mul, one_mul, hv j, hone i, one_mul]
  · intro b _ hb
    rw [ite_eq_right hb, zero_mul]
  · intro h
    exact absurd (Finset.mem_univ b₀) h
  · intro b _ hb
    rw [ite_eq_right hb, zero_mul]
  · intro h
    exact absurd (Finset.mem_univ b₁) h

/-- **The max reduction.**  Adding `2|𝒱|` times the `{0,1}` predicate to the
values moves every selected position above every unselected one, so the
maximizer of the sum is the largest *selected* value: `argmax_m (v_m +
2|𝒱|P(q_i,k_m)) = argmax_{m : P(q_i,k_m) = 1} v_m`. -/
theorem argmax_shift {m : ℕ} (v : Fin m → ℕ) (P : Fin m → Bool) (C : ℕ)
    (hv : ∀ a, v a < C) (hne : ∃ a, P a = true) {j : Fin m}
    (hj : ∀ a, v a + 2 * C * (if P a then 1 else 0)
      ≤ v j + 2 * C * (if P j then 1 else 0)) :
    P j = true ∧ ∀ a, P a = true → v a ≤ v j := by
  obtain ⟨a₀, ha₀⟩ := hne
  have hPj : P j = true := by
    by_contra hc
    have hcf : P j = false := by
      cases hPj' : P j
      · rfl
      · exact absurd hPj' hc
    have := hj a₀
    rw [ite_eq_left ha₀, hcf, ite_eq_right Bool.false_ne_true] at this
    have h1 := hv a₀
    have h2 := hv j
    omega
  refine ⟨hPj, fun a ha => ?_⟩
  have := hj a
  rw [ite_eq_left ha, ite_eq_left hPj] at this
  omega

/-- The hypotheses are satisfiable: the zero matrix is constructable from any
embedding, with no attention dimensions at all, and a shifted maximum does
occur. -/
example (X : Fin T → Fin d → ℝ) : Constructable X (fun _ _ => 0) :=
  ⟨0, fun a => absurd a.isLt (by omega), fun a => absurd a.isLt (by omega), fun _ _ => by simp⟩

example : (![0, 1] : Fin 2 → ℕ) 1 + 2 * 2 * (if (![true, false] : Fin 2 → Bool) 1 then 1 else 0)
    ≤ (![0, 1] : Fin 2 → ℕ) 0 + 2 * 2 * (if (![true, false] : Fin 2 → Bool) 0 then 1 else 0) := by
  decide

end RASPL
end Transformer
