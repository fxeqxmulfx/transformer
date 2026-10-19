module

public import Transformer.Homogenized.Section7_SimplexDriftBias

/-!
# Column constraints on the actual simplex attention moments

ArXiv:2604.01978v1, §7, eq:f_g_def_selfcontained and
eq:Phi_def_selfcontained.
The row normalization of attention forces the additional inequality
f(γ)+(n-1)g(γ)≥1 whenever its self and distinct-row moments are
independent of the indices. The proof is finite Cauchy–Schwarz
on the actual column sums, followed by their actual expectation.

Combined with the source's [0,1] bounds, this gives
0≤b(γ)≤2(1-γ) throughout the feasible simplex interval.
No sign, regularity or invariance of the scalar ODE is assumed.
These estimates will keep its trajectory within the domain where
the reference simplex has been explicitly constructed.
-/

@[expose] public section

open scoped BigOperators
open MeasureTheory Set

namespace Transformer.Homogenized

/-- Row normalization controls the sum of all actual row correlations.
Source: arXiv:2604.01978v1, §7, the probabilities in eq:pi_simplex_def_selfcontained. -/
theorem row_total_correlation_ge_card {n : ℕ} (hn : 0 < n) (P : Idx n → Idx n → ℝ)
    (hrows : ∀ i, ∑ k, P i k = 1) :
    (n : ℝ) ≤ ∑ i, ∑ j, ∑ k, P i k * P j k := by
  have hnR : (0 : ℝ) < n := by exact_mod_cast hn
  have hsum : (∑ k, ∑ i, P i k) = (n : ℝ) := by
    rw [Finset.sum_comm]
    simp [hrows]
  have hGram : (∑ i, ∑ j, ∑ k, P i k * P j k) = ∑ k, (∑ i, P i k) ^ 2 := by
    calc
      _ = ∑ i, ∑ k, ∑ j, P i k * P j k := by
        apply Finset.sum_congr rfl
        intro i hi
        rw [Finset.sum_comm]
      _ = ∑ k, ∑ i, ∑ j, P i k * P j k := by rw [Finset.sum_comm]
      _ = _ := by
        apply Finset.sum_congr rfl
        intro k hk
        rw [pow_two, Finset.sum_mul]
        simp only [Finset.mul_sum]
  have hcs := Finset.sum_mul_sq_le_sq_mul_sq Finset.univ (fun k => ∑ i, P i k) (fun _ => (1 : ℝ))
  simp only [mul_one, one_pow, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
    nsmul_eq_mul, hsum] at hcs
  rw [hGram]
  nlinarith

/-- The source's actual self and distinct-row attention expectations obey
F+nG≥1 for n+1 tokens. Source: arXiv:2604.01978v1, §7,
eq:f_g_def_selfcontained. No Gaussian shape or token norm is needed. -/
theorem attention_moments_column_bound {d n : ℕ} (β : ℝ)
    (ρ : Measure (HeadParam d)) [IsProbabilityMeasure ρ] (X : Idx (n + 1) → EucSpace d)
    (F G : ℝ)
    (hF : ∀ i, (∫ θ, ∑ k, attnProb β θ.2 X i k ^ 2 ∂ρ) = F)
    (hG : ∀ i j, i ≠ j → (∫ θ, ∑ k, attnProb β θ.2 X i k * attnProb β θ.2 X j k ∂ρ) = G) :
    1 ≤ F + (n : ℝ) * G := by
  have hInt (i : Idx (n + 1)) : Integrable (fun θ =>
      ∑ j, ∑ k, attnProb β θ.2 X i k * attnProb β θ.2 X j k) ρ :=
    integrable_finsetSum _ fun j hj => integrable_sum_attnProb_product β ρ X i j
  have hIntAll : Integrable (fun θ =>
      ∑ i, ∑ j, ∑ k, attnProb β θ.2 X i k * attnProb β θ.2 X j k) ρ :=
    integrable_finsetSum _ fun i hi => hInt i
  have hmoment (i j : Idx (n + 1)) :
      (∫ θ, ∑ k, attnProb β θ.2 X i k * attnProb β θ.2 X j k ∂ρ) =
        if i = j then F else G := by
    split_ifs with hij
    · subst j
      simpa only [pow_two] using hF i
    · exact hG i j hij
  have hrow (i : Idx (n + 1)) : (∑ j, if i = j then F else G) = F + (n : ℝ) * G := by
    calc
      _ = ∑ j : Idx (n + 1), ((if i = j then F - G else 0) + G) := by
        apply Finset.sum_congr rfl
        intro j hj
        split_ifs <;> ring
      _ = _ := by
        simp only [Finset.sum_add_distrib, Finset.sum_ite_eq, Finset.mem_univ, ite_true,
          Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul,
          Nat.cast_add, Nat.cast_one]
        ring
  have htotal := integral_mono (integrable_const ((n + 1 : ℕ) : ℝ)) hIntAll
    (fun θ => row_total_correlation_ge_card (Nat.succ_pos n) (attnProb β θ.2 X) (sum_attnProb β θ.2 X))
  rw [integral_finsetSum _ (fun i hi => hInt i)] at htotal
  simp_rw [integral_finsetSum _ (fun j hj => integrable_sum_attnProb_product β ρ X _ j),
    hmoment, hrow] at htotal
  simp only [integral_const, probReal_univ, one_smul, Finset.sum_const, Finset.card_univ,
    Fintype.card_fin, nsmul_eq_mul, Nat.cast_add, Nat.cast_one] at htotal
  have hnR : (0 : ℝ) < (n : ℝ) + 1 := by positivity
  nlinarith

/-- The source's scalar drift factors through the distance from consensus.
Source: arXiv:2604.01978v1, §7, eq:Phi_def_selfcontained. -/
theorem simplexDrift_eq_factor (f g : ℝ → ℝ) (γ : ℝ) :
    simplexDrift f g γ = (1 - γ) * (γ * (1 - f γ) + g γ) := by
  unfold simplexDrift
  ring

/-- The actual moment constraints give 0≤b(γ)≤2(1-γ) on the feasible
interval. Source: arXiv:2604.01978v1, §7, eq:f_g_def_selfcontained
and eq:Phi_def_selfcontained. These are derived scalar estimates,
not assumptions on the ODE trajectory. -/
theorem simplexDrift_mem_Icc {n : ℕ} (hn : 2 ≤ n) (f g : ℝ → ℝ) {γ : ℝ}
    (hγ : γ ∈ Icc (-(1 / ((n : ℝ) - 1))) 1)
    (hf : f γ ∈ Icc (0 : ℝ) 1) (hg : g γ ∈ Icc (0 : ℝ) 1)
    (hcolumn : 1 ≤ f γ + ((n : ℝ) - 1) * g γ) :
    simplexDrift f g γ ∈ Icc (0 : ℝ) (2 * (1 - γ)) := by
  have hnR : (2 : ℝ) ≤ n := by exact_mod_cast hn
  have hm : 0 < (n : ℝ) - 1 := by linarith
  have hdiv : (1 - f γ) / ((n : ℝ) - 1) ≤ g γ := (div_le_iff₀ hm).2 (by linarith)
  have hprod : 0 ≤ (γ + 1 / ((n : ℝ) - 1)) * (1 - f γ) :=
    mul_nonneg (by linarith [hγ.1]) (by linarith [hf.2])
  have hlow : 0 ≤ γ * (1 - f γ) + g γ := by
    calc
      _ ≤ (γ + 1 / ((n : ℝ) - 1)) * (1 - f γ) +
          (g γ - (1 - f γ) / ((n : ℝ) - 1)) := add_nonneg hprod (sub_nonneg.mpr hdiv)
      _ = _ := by ring
  have hhigh : γ * (1 - f γ) + g γ ≤ 2 := by
    have hmul := mul_le_mul_of_nonneg_right hγ.2 (sub_nonneg.mpr hf.2)
    linarith [hf.1, hg.2]
  rw [simplexDrift_eq_factor]
  exact ⟨mul_nonneg (sub_nonneg.mpr hγ.2) hlow,
    (mul_le_mul_of_nonneg_left hhigh (sub_nonneg.mpr hγ.2)).trans_eq (by ring)⟩

/-- Uniform actual attention on two orthonormal tokens satisfies all moment
premises with F=G=1/2. Its feasible scalar drift is positive at γ=1/2. -/
example : ∃ X : Idx 2 → EucSpace 2,
    0 < (2 : ℕ) ∧ 2 ≤ (2 : ℕ) ∧
    IsProbabilityMeasure (Measure.dirac (0 : HeadParam 2)) ∧
    (1 / 2 : ℝ) ∈ Icc (0 : ℝ) 1 ∧
    (∀ i, ∑ k, attnProb 1 (0 : Matrix (Fin 2) (Fin 2) ℝ) X i k = 1) ∧
    (∀ i, (∫ θ, ∑ k, attnProb 1 θ.2 X i k ^ 2 ∂(Measure.dirac (0 : HeadParam 2))) = (1 / 2 : ℝ)) ∧
    (∀ i j, i ≠ j → (∫ θ, ∑ k, attnProb 1 θ.2 X i k * attnProb 1 θ.2 X j k
      ∂(Measure.dirac (0 : HeadParam 2))) = (1 / 2 : ℝ)) ∧
    (1 / 2 : ℝ) ∈ Icc (-(1 / ((2 : ℝ) - 1))) 1 ∧
    1 ≤ (1 / 2 : ℝ) + ((2 : ℝ) - 1) * (1 / 2) ∧
    simplexDrift (fun _ => 1 / 2) (fun _ => 1 / 2) (1 / 2) = (3 / 8 : ℝ) := by
  let X : Idx 2 → EucSpace 2 := fun k => EuclideanSpace.single k 1
  refine ⟨X, by norm_num, le_rfl, inferInstance, by norm_num,
    sum_attnProb 1 0 X, ?_, ?_, by norm_num, by norm_num, ?_⟩
  · intro i
    norm_num [attnProb, attnWeight, qkMap, Fin.sum_univ_two]
  · intro i j hij
    norm_num [attnProb, attnWeight, qkMap, Fin.sum_univ_two]
  · norm_num [simplexDrift]

end Transformer.Homogenized
