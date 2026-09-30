/-
# DASH — the probabilistic benefit of independent starting vectors

arXiv:2602.02016v2, §3.5. This is an explicit finite uniform sampling model:
all functions from pool positions to candidate indices are equally likely.
Sampling with replacement makes the draws independent. It does not model
Gaussian starts, finite-precision effects, or a fixed iteration safety bound.
-/

import Transformer.DASH.Section3_MultiPowerConvergence
import Mathlib.Data.Fintype.Pi

noncomputable section

namespace Transformer.DASH

variable {m : ℕ}

/-- Fraction of independent uniform pools whose every member is in the bad
candidate set. For PI, bad candidates miss the relevant maximal eigenspace.
Source: arXiv:2602.02016v2, §3.5, reducing the likelihood of unsuccessful starts. -/
def poolMissFraction (bad : Finset (Fin m)) (N : ℕ) : ℝ :=
  (Fintype.card {draw : Fin N → Fin m // ∀ i, draw i ∈ bad} : ℝ) /
    Fintype.card (Fin N → Fin m)

/-- Exact enumeration of the bad pools: the choices at different pool
positions are independent. Source: arXiv:2602.02016v2, §3.5, multi-PI starts. -/
theorem poolMissFraction_eq_pow (bad : Finset (Fin m)) (N : ℕ) :
    poolMissFraction bad N = ((bad.card : ℝ) / m) ^ N := by
  have hcard : Fintype.card {draw : Fin N → Fin m // ∀ i, draw i ∈ bad} = bad.card ^ N := by
    simpa using Fintype.card_congr
      (Equiv.subtypePiEquivPi :
        {draw : Fin N → Fin m // ∀ i, draw i ∈ bad} ≃ (Fin N → {j // j ∈ bad}))
  simp only [poolMissFraction, hcard, Fintype.card_fun, Fintype.card_fin,
    Nat.cast_pow, div_pow]

/-- The single-draw bad fraction is a probability for a nonempty candidate
population. Source: arXiv:2602.02016v2, §3.5, independent uniform starting vectors. -/
theorem pool_bad_fraction_bounds (bad : Finset (Fin m)) (hm : 0 < m) :
    0 ≤ (bad.card : ℝ) / m ∧ (bad.card : ℝ) / m ≤ 1 := by
  have hm' : (0 : ℝ) < m := by exact_mod_cast hm
  have hcardNat : bad.card ≤ m := by simpa using Finset.card_le_univ bad
  have hcard : (bad.card : ℝ) ≤ m := by exact_mod_cast hcardNat
  exact ⟨div_nonneg (by positivity) hm'.le, (div_le_one hm').mpr hcard⟩

/-- Nonempty finite candidate populations exist, arXiv:2602.02016v2, §3.5. -/
example : 0 < (2 : ℕ) := by norm_num

/-- Increasing pool size cannot increase the chance that all starts are bad.
This is a probability statement under independent uniform sampling, not a
deterministic safety guarantee. Source: arXiv:2602.02016v2, §3.5. -/
theorem poolMissFraction_antitone (bad : Finset (Fin m)) (hm : 0 < m)
    (N K : ℕ) (hNK : N ≤ K) : poolMissFraction bad K ≤ poolMissFraction bad N := by
  rw [poolMissFraction_eq_pow, poolMissFraction_eq_pow]
  obtain ⟨hp, hp'⟩ := pool_bad_fraction_bounds bad hm
  exact pow_le_pow_of_le_one hp hp' hNK

/-- Pool-order hypotheses are satisfiable, arXiv:2602.02016v2, §3.5. -/
example : 0 < (2 : ℕ) ∧ (16 : ℕ) ≤ 32 := by norm_num

/-- A population with at least one good candidate gives a bad fraction
strictly below one. Source: arXiv:2602.02016v2, §3.5, successful starting vectors. -/
theorem pool_bad_fraction_lt_one (bad : Finset (Fin m)) (hm : 0 < m)
    (hgood : ∃ j : Fin m, j ∉ bad) : (bad.card : ℝ) / m < 1 := by
  obtain ⟨j, hj⟩ := hgood
  have hstrict : bad ⊂ Finset.univ := by
    apply Finset.ssubset_iff_subset_ne.2
    refine ⟨Finset.subset_univ _, ?_⟩
    intro heq
    exact hj (heq.symm ▸ Finset.mem_univ j)
  have hcardNat : bad.card < m := by simpa using Finset.card_lt_card hstrict
  have hcard : (bad.card : ℝ) < m := by exact_mod_cast hcardNat
  exact (div_lt_one (by exact_mod_cast hm)).mpr hcard

/-- A population containing one bad and one good candidate satisfies the
strict probability assumptions. Source: arXiv:2602.02016v2, §3.5. -/
example : 0 < (2 : ℕ) ∧ ∃ j : Fin 2, j ∉ ({0} : Finset (Fin 2)) := by
  refine ⟨by norm_num, 1, ?_⟩
  norm_num

/-- If a good start has positive sampling probability, the chance of drawing
only bad starts tends to zero as pool size grows. The source supplies no
distribution-independent numeric probability for its pools of 16 or 32.
Source: arXiv:2602.02016v2, §3.5, the probabilistic motivation of multi-PI. -/
theorem poolMissFraction_tendsto_zero (bad : Finset (Fin m)) (hm : 0 < m)
    (hgood : ∃ j : Fin m, j ∉ bad) :
    Filter.Tendsto (poolMissFraction bad) Filter.atTop (nhds 0) := by
  have hpow := tendsto_pow_atTop_nhds_zero_of_lt_one
    (pool_bad_fraction_bounds bad hm).1 (pool_bad_fraction_lt_one bad hm hgood)
  simpa only [show poolMissFraction bad = (fun N => ((bad.card : ℝ) / m) ^ N) from
    funext (poolMissFraction_eq_pow bad)] using hpow

/-- Vanishing-failure assumptions are satisfiable, arXiv:2602.02016v2, §3.5. -/
example : 0 < (2 : ℕ) ∧ ∃ j : Fin 2, j ∉ ({0} : Finset (Fin 2)) := by
  refine ⟨by norm_num, 1, ?_⟩
  norm_num

/-- With half the candidates bad, the failure fraction is exactly `2^(-N)`.
This illustrates the conditional claim and does not infer that the paper's
actual start distribution has this bad fraction.
Source: arXiv:2602.02016v2, §3.5, the pool-size tradeoff. -/
theorem half_bad_pool_fraction (N : ℕ) :
    poolMissFraction ({0} : Finset (Fin 2)) N = (1 / 2 : ℝ) ^ N := by
  rw [poolMissFraction_eq_pow]
  norm_num

end Transformer.DASH
