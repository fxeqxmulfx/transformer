import Transformer.GPTMini.Sparsemax.PairedRecallCodes

/-!
# Unit score gaps from compact bounded integer key geometry

New genuine recall-head certificate following arXiv:2211.11052v1, §3.1.
Distinct equal-norm integer vectors have a positive squared separation.
Expanding that distance bounds their integer inner product strictly below
the common squared norm. Integer arithmetic turns strictness into a unit
gap, sufficient for sparsemax Eq. (1) and §2.2 of arXiv:1602.02068v2.

The explicit 256-key construction has four coordinates bounded by four
and common squared norm thirty. Real physical scores retain all these
bounds. This supplies a width-eight contextual-head capacity witness;
the numerical optimizer is not required or proved to discover it.
-/

namespace Transformer.GPTMini.Sparsemax

open scoped BigOperators

/-- Distinct integer vectors of the same norm have at least a unit inner-product gap.
Source: the algebraic certificate of the new compact key construction after §3.1. -/
theorem integer_equal_norm_gap {H : ℕ} (q k : Fin H → ℤ) (N : ℤ)
    (hq : ∑ d, q d ^ 2 = N) (hk : ∑ d, k d ^ 2 = N) (hne : q ≠ k) :
    ∑ d, q d * k d ≤ N - 1 := by
  have hd : ∃ d, q d ≠ k d := by
    by_contra hn
    push Not at hn
    exact hne (funext hn)
  obtain ⟨d, hd⟩ := hd
  have hs : 0 < ∑ j, (q j - k j) ^ 2 := by
    apply Finset.sum_pos' (fun j hj => sq_nonneg _)
    refine ⟨d, Finset.mem_univ _, sq_pos_of_ne_zero ?_⟩
    intro he
    apply hd
    linarith
  have he : (∑ j, (q j - k j) ^ 2) = 2 * N - 2 * ∑ j, q j * k j := by
    calc
      _ = ∑ j, (q j ^ 2 + k j ^ 2 - 2 * (q j * k j)) := by
        apply Finset.sum_congr rfl
        intro j hj
        ring
      _ = 2 * N - 2 * ∑ j, q j * k j := by
        rw [Finset.sum_sub_distrib, Finset.sum_add_distrib, ← Finset.mul_sum, hq, hk]
        ring
  rw [he] at hs
  omega

/-- Two different explicit vectors satisfy every norm and inequality premise. -/
example : (∑ d : Fin 2, (![(1 : ℤ), 0] d) * (![0, (1 : ℤ)] d)) ≤ 0 := by
  apply integer_equal_norm_gap (N := 1)
  · norm_num [Fin.sum_univ_two]
  · norm_num [Fin.sum_univ_two]
  · intro he
    have hc := congrFun he 0
    norm_num at hc

/-- Different keys have integer score at most twenty-nine in the actual compact code table.
Source: the norm/injectivity certificate for the new recall construction after §3.1. -/
theorem pairedRecallCode_dot_gap (i j : Fin 256) (hij : i ≠ j) :
    ∑ d, pairedRecallCode i d * pairedRecallCode j d ≤ 29 := by
  apply integer_equal_norm_gap _ _ 30 (pairedRecallCode_norm i) (pairedRecallCode_norm j)
  intro he
  exact hij (pairedRecallCode_injective he)

/-- Two actual different key identities inhabit the compact dot-gap premise. -/
example : ∑ d, pairedRecallCode 0 d * pairedRecallCode 1 d ≤ 29 :=
  pairedRecallCode_dot_gap _ _ (by decide)

/-- In real physical Q/K arithmetic, each identical key has score exactly thirty.
Source: the original learned dot product of §3.1 using the explicit compact witness. -/
theorem pairedRecallCode_real_self (i : Fin 256) :
    (∑ d, (pairedRecallCode i d : ℝ) * (pairedRecallCode i d : ℝ)) = 30 := by
  have hn := pairedRecallCode_norm i
  have he : (∑ d, pairedRecallCode i d * pairedRecallCode i d) = 30 := by
    simpa only [pow_two] using hn
  exact_mod_cast he

/-- Real physical scores of different keys retain the unit sparsemax score gap.
Source: the compact integer construction before sparsemax Eq. (1). -/
theorem pairedRecallCode_real_gap (i j : Fin 256) (hij : i ≠ j) :
    (∑ d, (pairedRecallCode i d : ℝ) * (pairedRecallCode j d : ℝ)) ≤ 29 := by
  have hg := pairedRecallCode_dot_gap i j hij
  exact_mod_cast hg

/-- A real nontrivial query/key score satisfies the different-key premise. -/
example : (∑ d, (pairedRecallCode 0 d : ℝ) * (pairedRecallCode 1 d : ℝ)) ≤ 29 :=
  pairedRecallCode_real_gap _ _ (by decide)

/-- The complete real physical margin is at least one for any two different key identities.
Source: the compact witness geometry needed by sparsemax §2.2's support certificate. -/
theorem pairedRecallCode_real_margin (i j : Fin 256) (hij : i ≠ j) :
    1 ≤ (∑ d, (pairedRecallCode i d : ℝ) * (pairedRecallCode i d : ℝ)) -
      ∑ d, (pairedRecallCode i d : ℝ) * (pairedRecallCode j d : ℝ) := by
  rw [pairedRecallCode_real_self]
  have hg := pairedRecallCode_real_gap i j hij
  linarith

/-- The first actual two different key codes satisfy the real margin premise. -/
example : 1 ≤ (∑ d, (pairedRecallCode 0 d : ℝ) * (pairedRecallCode 0 d : ℝ)) -
    ∑ d, (pairedRecallCode 0 d : ℝ) * (pairedRecallCode 1 d : ℝ) :=
  pairedRecallCode_real_margin _ _ (by decide)

/-- Integer coordinate bounds survive unchanged in the freely learned real Q/K table.
Source: the numerical cap-four witness following §3.1 and Appendix A.4. -/
theorem pairedRecallCode_real_bounds (i : Fin 256) (d : Fin 4) :
    -4 ≤ (pairedRecallCode i d : ℝ) ∧ (pairedRecallCode i d : ℝ) ≤ 4 := by
  have hb := pairedRecallCode_bounds i d
  exact_mod_cast hb

/-- None of the 256 eligible integer query embeddings is the zero vector.
Source: the common norm thirty of the explicit compact §3.1 construction. -/
theorem pairedRecallCode_ne_zero (i : Fin 256) : pairedRecallCode i ≠ 0 := by
  intro he
  have hn := pairedRecallCode_norm i
  rw [he] at hn
  norm_num at hn

/-- The real physical query embedding is nonzero as well, independently of the selected key.
Source: genuine real dot products of §3.1 and the common score thirty. -/
theorem pairedRecallCode_real_ne_zero (i : Fin 256) : (fun d => (pairedRecallCode i d : ℝ)) ≠ 0 := by
  intro he
  have hs := pairedRecallCode_real_self i
  have hz (d : Fin 4) : (pairedRecallCode i d : ℝ) = 0 := congrFun he d
  simp only [hz, zero_mul, Finset.sum_const_zero] at hs
  norm_num at hs

/-- The entire real physical code has squared norm thirty, with no normalization layer.
Source: the compact integer witness in the original §3.1 dot-product geometry. -/
theorem pairedRecallCode_real_norm (i : Fin 256) : ∑ d, (pairedRecallCode i d : ℝ) ^ 2 = 30 := by
  simpa only [pow_two] using pairedRecallCode_real_self i

/-- Different real key embeddings have squared distance at least two inside the cap-four box.
Source: the unit inner-product gap of the new compact §3.1 construction. -/
theorem pairedRecallCode_distance (i j : Fin 256) (hij : i ≠ j) :
    (2 : ℝ) ≤ ∑ d, ((pairedRecallCode i d : ℝ) - (pairedRecallCode j d : ℝ)) ^ 2 := by
  have he : (∑ d, ((pairedRecallCode i d : ℝ) - (pairedRecallCode j d : ℝ)) ^ 2) =
      60 - 2 * ∑ d, (pairedRecallCode i d : ℝ) * (pairedRecallCode j d : ℝ) := by
    calc
      _ = ∑ d, ((pairedRecallCode i d : ℝ) ^ 2 + (pairedRecallCode j d : ℝ) ^ 2 -
          2 * ((pairedRecallCode i d : ℝ) * (pairedRecallCode j d : ℝ))) := by
        apply Finset.sum_congr rfl
        intro d hd
        ring
      _ = _ := by
        rw [Finset.sum_sub_distrib, Finset.sum_add_distrib, ← Finset.mul_sum,
          pairedRecallCode_real_norm, pairedRecallCode_real_norm]
        ring
  rw [he]
  have hg := pairedRecallCode_real_gap i j hij
  linarith

/-- Two actual key identities satisfy the real separation premise. -/
example : (2 : ℝ) ≤ ∑ d, ((pairedRecallCode 0 d : ℝ) - (pairedRecallCode 1 d : ℝ)) ^ 2 :=
  pairedRecallCode_distance _ _ (by decide)

end Transformer.GPTMini.Sparsemax
