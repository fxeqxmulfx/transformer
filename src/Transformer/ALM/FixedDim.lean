/-
# Why the barrier needs a growing dimension

`Transformer.ALM.Hardness` states the Orthogonal Vectors conjecture at
dimension `c · log n` and remarks that it is silent at any fixed dimension.
That remark is not a hedge: at fixed dimension the conjecture is *false*, and
provably so, for an elementary reason.

A vector of dimension `d` is one of only `2^d` things.  So an instance of `n`
vectors in dimension `d` contains at most `min n (2^d)` distinct vectors
(`card_distinct_le`), and its answer depends on nothing else
(`exists_orth_iff_image`).  Bucket the vectors by value in one pass and scan
the distinct ones: cost `n·d + min(n, 2^d)² · d`.  At dimension `Θ(log n)`
that is quadratic and refutes nothing.  At any *fixed* `d₀` it is linear in
`n`, and the conjecture falls.

`dedupeModel` charges exactly that.  `dedupeModel_OVHard` and
`dedupeModel_not_OVHardFixed` hold simultaneously, so
`OVHard_needs_growing_dimension` is a theorem: the `c · log n` in the
hypothesis is a necessity, not an artefact of how it was written.  This is
the standard observation behind the dimension in the conjecture — see
V. Vassilevska Williams, *On some fine-grained questions in algorithms and
complexity*, ICM 2018, §3, where the conjecture is stated at
`d = ω(log n)` for exactly this reason.

The consequence for the lookup machine: the planar hull of
`transformer_vm/attention/hull2d_cht.h` runs at `m = 1` and is therefore
outside the barrier's reach by a theorem, not by an unproved remark.
-/

import Transformer.ALM.Hardness

namespace Transformer
namespace ALM

/-! ### The dimension caps the number of distinct vectors -/

/-- An instance of `n` vectors in dimension `d` contains at most
`min n (2^d)` distinct vectors: it cannot have more entries than indices, nor
more than there are vectors of that dimension. -/
lemma card_distinct_le {d n : ℕ} (A : Fin n → BVec d) :
    (Finset.image A Finset.univ).card ≤ min n (2 ^ d) := by
  refine le_min ?_ ?_
  · calc (Finset.image A Finset.univ).card
        ≤ (Finset.univ : Finset (Fin n)).card := Finset.card_image_le
      _ = n := by simp
  · calc (Finset.image A Finset.univ).card
        ≤ (Finset.univ : Finset (BVec d)).card := Finset.card_le_univ _
      _ = 2 ^ d := by simp

/-- **Deduplication is sound.**  The answer depends only on which vectors
occur, not on how often, so an algorithm may discard repeats. -/
theorem exists_orth_iff_image {d n : ℕ} (A B : Fin n → BVec d) :
    (∃ i j, Orth (A i) (B j)) ↔
      ∃ u ∈ Finset.image A Finset.univ, ∃ v ∈ Finset.image B Finset.univ, Orth u v := by
  constructor
  · rintro ⟨i, j, h⟩
    exact ⟨A i, Finset.mem_image_of_mem A (Finset.mem_univ i),
      B j, Finset.mem_image_of_mem B (Finset.mem_univ j), h⟩
  · rintro ⟨u, hu, v, hv, h⟩
    obtain ⟨i, -, rfl⟩ := Finset.mem_image.mp hu
    obtain ⟨j, -, rfl⟩ := Finset.mem_image.mp hv
    exact ⟨i, j, h⟩

/-! ### The conjecture at a fixed dimension -/

/-- The Orthogonal Vectors conjecture with the dimension held constant.  Same
statement as `CostModel.OVHard` except that the instances have dimension `d₀`
for a fixed `d₀`, rather than one growing with `n`. -/
def CostModel.OVHardFixed (M : CostModel) (d₀ : ℕ) : Prop :=
  ∀ ε : ℝ, 0 < ε → ∀ a : M.Alg, M.Solves a → ∀ N : ℕ,
    ∃ n, N ≤ n ∧ (n : ℝ) ^ (2 - ε) ≤ M.cost a n d₀

/-- The definition is not self-contradictory: the quadratic model of
`Transformer.ALM.Hardness` satisfies it at any nonzero dimension, since its
only algorithm is charged `2n²d` there. -/
example : naiveModel.OVHardFixed 1 := by
  intro ε hε _ _ N
  refine ⟨max N 1, le_max_left _ _, ?_⟩
  have hnR : (1 : ℝ) ≤ ((max N 1 : ℕ) : ℝ) := by
    exact_mod_cast le_max_right N 1
  have hpow : ((max N 1 : ℕ) : ℝ) ^ (2 - ε) ≤ ((max N 1 : ℕ) : ℝ) ^ 2 := by
    calc ((max N 1 : ℕ) : ℝ) ^ (2 - ε)
        ≤ ((max N 1 : ℕ) : ℝ) ^ (2 : ℝ) :=
          Real.rpow_le_rpow_of_exponent_le hnR (by linarith)
      _ = ((max N 1 : ℕ) : ℝ) ^ 2 := by
          rw [show (2 : ℝ) = ((2 : ℕ) : ℝ) by norm_num, Real.rpow_natCast]
  show ((max N 1 : ℕ) : ℝ) ^ (2 - ε) ≤ 2 * ((max N 1 : ℕ) : ℝ) ^ 2 * ((1 : ℕ) : ℝ)
  rw [Nat.cast_one, mul_one]
  nlinarith [hpow, sq_nonneg ((max N 1 : ℕ) : ℝ)]

/-! ### The bucketing model -/

/-- What bucketing costs: one pass over the input to group equal vectors,
then an exhaustive scan over the distinct ones, whose number `card_distinct_le`
bounds by `min n (2^d)`. -/
noncomputable def dedupeCost (n d : ℕ) : ℝ :=
  (n : ℝ) * (d : ℝ) + ((min n (2 ^ d) : ℕ) : ℝ) ^ 2 * (d : ℝ)

/-- A model with one algorithm, the bucketing scan, charged `dedupeCost`. -/
noncomputable def dedupeModel : CostModel where
  Alg := Unit
  decides := fun _ => fun A B => ∃ i j, Orth (A i) (B j)
  cost := fun _ n d => dedupeCost n d

lemma dedupeModel_solves : dedupeModel.Solves () := fun _ _ => Iff.rfl

/-- **Bucketing does not beat the conjecture at logarithmic dimension.**  At
`d = 2 log₂ n` there are already at least `n` possible vectors, so no
deduplication happens and the scan is the quadratic one. -/
theorem dedupeModel_OVHard : dedupeModel.OVHard := by
  intro ε hε
  refine ⟨2, fun _ _ N => ⟨max N 2, le_max_left _ _, ?_⟩⟩
  set n := max N 2 with hndef
  have hn2 : 2 ≤ n := le_max_right _ _
  have hlog : 1 ≤ Nat.log 2 n := Nat.log_pos (by norm_num) hn2
  have hmin : min n (2 ^ (2 * Nat.log 2 n)) = n := by
    refine min_eq_left (le_of_lt (lt_of_lt_of_le (Nat.lt_pow_succ_log_self (b := 2) (by norm_num) n) ?_))
    exact Nat.pow_le_pow_right (by norm_num) (by omega)
  have hnR : (1 : ℝ) ≤ (n : ℝ) := by
    have : 1 ≤ n := by omega
    exact_mod_cast this
  have hdR : (1 : ℝ) ≤ ((2 * Nat.log 2 n : ℕ) : ℝ) := by
    have : 1 ≤ 2 * Nat.log 2 n := by omega
    exact_mod_cast this
  have hpow : (n : ℝ) ^ (2 - ε) ≤ (n : ℝ) ^ 2 := by
    calc (n : ℝ) ^ (2 - ε) ≤ (n : ℝ) ^ (2 : ℝ) :=
          Real.rpow_le_rpow_of_exponent_le hnR (by linarith)
      _ = (n : ℝ) ^ 2 := by
          rw [show (2 : ℝ) = ((2 : ℕ) : ℝ) by norm_num, Real.rpow_natCast]
  show (n : ℝ) ^ (2 - ε) ≤ dedupeCost n (2 * Nat.log 2 n)
  rw [dedupeCost, hmin]
  nlinarith [hpow, sq_nonneg (n : ℝ)]

/-- **But at a fixed dimension it wins outright.**  For every constant `d₀`
the bucketing cost is linear in `n`, so it is eventually below `n^{3/2}` and
`OVHardFixed d₀` fails.  The witness `ε = 1/2` is any exponent above `1`. -/
theorem dedupeModel_not_OVHardFixed (d₀ : ℕ) : ¬ dedupeModel.OVHardFixed d₀ := by
  intro h
  set C : ℝ := (d₀ : ℝ) + (4 : ℝ) ^ d₀ * (d₀ : ℝ) with hCdef
  have hC0 : 0 ≤ C := by positivity
  obtain ⟨n, hn, hcost⟩ :=
    h (1 / 2) (by norm_num) () dedupeModel_solves (⌈C ^ 2⌉₊ + 1)
  have hnN : ⌈C ^ 2⌉₊ < n := by omega
  have hnR : (0 : ℝ) < (n : ℝ) := by
    have : 0 < n := by omega
    exact_mod_cast this
  have hn1 : (1 : ℝ) ≤ (n : ℝ) := by
    have : 1 ≤ n := by omega
    exact_mod_cast this
  -- `n` exceeds `C²`, so its square root exceeds `C`.
  have hgt : C ^ 2 < (n : ℝ) := by
    have h1 : C ^ 2 ≤ (⌈C ^ 2⌉₊ : ℝ) := Nat.le_ceil _
    have h2 : ((⌈C ^ 2⌉₊ : ℕ) : ℝ) < (n : ℝ) := by exact_mod_cast hnN
    linarith
  have hroot : C < (n : ℝ) ^ ((1 : ℝ) / 2) := by
    rw [← Real.sqrt_eq_rpow]
    exact (Real.lt_sqrt hC0).mpr hgt
  -- The bucketing cost is at most `C · n`.
  have hbound : dedupeCost n d₀ ≤ C * (n : ℝ) := by
    have hm : ((min n (2 ^ d₀) : ℕ) : ℝ) ≤ (2 : ℝ) ^ d₀ := by
      exact_mod_cast min_le_right n (2 ^ d₀)
    have hm0 : (0 : ℝ) ≤ ((min n (2 ^ d₀) : ℕ) : ℝ) := Nat.cast_nonneg _
    have hsq : ((min n (2 ^ d₀) : ℕ) : ℝ) ^ 2 ≤ (4 : ℝ) ^ d₀ := by
      calc ((min n (2 ^ d₀) : ℕ) : ℝ) ^ 2 ≤ ((2 : ℝ) ^ d₀) ^ 2 := by
            exact pow_le_pow_left₀ hm0 hm 2
        _ = (4 : ℝ) ^ d₀ := by
            rw [← pow_mul, mul_comm, pow_mul]; norm_num
    have hd0 : (0 : ℝ) ≤ (d₀ : ℝ) := Nat.cast_nonneg _
    have step1 : ((min n (2 ^ d₀) : ℕ) : ℝ) ^ 2 * (d₀ : ℝ) ≤ (4 : ℝ) ^ d₀ * (d₀ : ℝ) :=
      mul_le_mul_of_nonneg_right hsq hd0
    have h4 : (0 : ℝ) ≤ (4 : ℝ) ^ d₀ * (d₀ : ℝ) := by positivity
    have step2 : (4 : ℝ) ^ d₀ * (d₀ : ℝ) ≤ (4 : ℝ) ^ d₀ * (d₀ : ℝ) * (n : ℝ) :=
      le_mul_of_one_le_right h4 hn1
    rw [dedupeCost, hCdef]
    nlinarith [step1, step2]
  -- And `C · n < n^{3/2}`, contradicting the lower bound.
  have hsplit : (n : ℝ) ^ (2 - (1 : ℝ) / 2) = (n : ℝ) ^ ((1 : ℝ) / 2) * (n : ℝ) := by
    rw [show (2 : ℝ) - (1 : ℝ) / 2 = (1 : ℝ) / 2 + 1 by norm_num,
      Real.rpow_add hnR, Real.rpow_one]
  have hstrict : C * (n : ℝ) < (n : ℝ) ^ ((1 : ℝ) / 2) * (n : ℝ) :=
    mul_lt_mul_of_pos_right hroot hnR
  rw [show dedupeModel.cost () n d₀ = dedupeCost n d₀ from rfl, hsplit] at hcost
  linarith

/-! ### The consequence -/

/-- **The growing dimension is necessary.**  There is a cost model in which
Orthogonal Vectors is hard at dimension `Θ(log n)` yet easy at every fixed
dimension.  So no restatement of `CostModel.OVHard` can drop the `c · log n`
and stay true, and the barrier of `Transformer.ALM.Hardness` genuinely says
nothing about a planar index. -/
theorem OVHard_needs_growing_dimension :
    ∃ M : CostModel, M.OVHard ∧ ∀ d₀ : ℕ, ¬ M.OVHardFixed d₀ :=
  ⟨dedupeModel, dedupeModel_OVHard, dedupeModel_not_OVHardFixed⟩

end ALM
end Transformer
