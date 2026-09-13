/-
# Fourier analysis on the boolean hypercube

Zhou, Bradley, Littwin, Razin, Saremi, Susskind, Bengio, Nakkiran —
arXiv:2310.16028v1, "What Algorithms can Transformers Learn?  A Study in
Length Generalization" (ICLR 2024).

§4 and Appendix E of that paper compare a trained transformer against the
*minimum-degree interpolator* of its training set, and both of its lemmas
about that interpolator are statements about the *degree profile* of a
boolean function — "the tuple of its Fourier weights at each level, with the
natural total ordering which refines the standard polynomial degree"
(footnote to §4.1, following Abbe, Boix-Adserà, Misiakiewicz).

Mathlib has Fourier analysis on the circle but not on the hypercube, so this
file builds the little that is needed: the characters `χ_S`, their
orthogonality, the coefficients `coeff f S`, and the expansion of a function
in them.  The degree profile itself is `RASPL.Degree`.

The cube is `{±1}^n` throughout, "as is standard in boolean function
analysis" (Appendix E), and a point of it is recorded by its sign bits:
`false ↦ +1` and `true ↦ -1`.
-/

import Transformer.Basic

namespace Transformer
namespace RASPL

variable {n : ℕ}

/-- A point of the boolean hypercube `{±1}^n`, recorded by its sign bits. -/
abbrev Cube (n : ℕ) : Type := Fin n → Bool

/-- The `±1` value of a sign bit: `false ↦ 1`, `true ↦ -1`. -/
def bitSign (b : Bool) : ℝ := if b then -1 else 1

@[simp] lemma bitSign_false : bitSign false = 1 := rfl

@[simp] lemma bitSign_true : bitSign true = -1 := rfl

@[simp] lemma bitSign_mul_self (b : Bool) : bitSign b * bitSign b = 1 := by
  cases b <;> norm_num [bitSign]

lemma one_add_bitSign_mul (a b : Bool) :
    1 + bitSign a * bitSign b = if a = b then 2 else 0 := by
  cases a <;> cases b <;> norm_num [bitSign]

/-- The character `χ_S (x) = ∏_{i ∈ S} x_i`, the multilinear monomial indexed
by `S`. -/
def chi (S : Finset (Fin n)) (x : Cube n) : ℝ := ∏ i ∈ S, bitSign (x i)

@[simp] lemma chi_empty (x : Cube n) : chi ∅ x = 1 := by simp [chi]

/-- Every character takes the values `±1`, so it never vanishes. -/
lemma chi_mul_self (S : Finset (Fin n)) (x : Cube n) : chi S x * chi S x = 1 := by
  simp [chi, ← Finset.prod_mul_distrib]

/-- Characters multiply by symmetric difference: `χ_S · χ_T = χ_{S Δ T}`. -/
lemma chi_mul_chi (S T : Finset (Fin n)) (x : Cube n) :
    chi S x * chi T x = chi (symmDiff S T) x := by
  have hS : chi S x = (∏ i ∈ S \ T, bitSign (x i)) * ∏ i ∈ S ∩ T, bitSign (x i) := by
    rw [chi, ← Finset.prod_union (Finset.disjoint_sdiff_inter S T), Finset.sdiff_union_inter]
  have hT : chi T x = (∏ i ∈ T \ S, bitSign (x i)) * ∏ i ∈ T ∩ S, bitSign (x i) := by
    rw [chi, ← Finset.prod_union (Finset.disjoint_sdiff_inter T S), Finset.sdiff_union_inter]
  have hsq : (∏ i ∈ S ∩ T, bitSign (x i)) * ∏ i ∈ T ∩ S, bitSign (x i) = 1 := by
    rw [Finset.inter_comm T S, ← Finset.prod_mul_distrib]
    simp
  have hdisj : Disjoint (S \ T) (T \ S) :=
    Finset.disjoint_left.2 fun a ha ha' => (Finset.mem_sdiff.1 ha').2 (Finset.mem_sdiff.1 ha).1
  have hd : chi (symmDiff S T) x
      = (∏ i ∈ S \ T, bitSign (x i)) * ∏ i ∈ T \ S, bitSign (x i) := by
    rw [chi, show symmDiff S T = (S \ T) ∪ (T \ S) from Finset.sup_eq_union,
      Finset.prod_union hdisj]
  rw [hS, hT, hd, show ∀ a b c d : ℝ, a * b * (c * d) = a * c * (b * d) from fun _ _ _ _ => by ring,
    hsq, mul_one]

/-- Summing a character over the whole cube: every character but the empty
one has mean zero. -/
lemma sum_chi (S : Finset (Fin n)) :
    ∑ x : Cube n, chi S x = if S = ∅ then (2 : ℝ) ^ n else 0 := by
  have key : ∀ x : Cube n, chi S x = ∏ i : Fin n, (if i ∈ S then bitSign (x i) else 1) := by
    intro x
    rw [chi, Finset.prod_ite_mem, Finset.univ_inter]
  have hprod : ∑ x : Cube n, chi S x = ∏ i : Fin n, ∑ b : Bool, (if i ∈ S then bitSign b else 1) := by
    rw [Finset.prod_univ_sum]
    rw [Fintype.piFinset_univ]
    exact Finset.sum_congr rfl fun x _ => key x
  have hfac : ∀ i : Fin n, (∑ b : Bool, (if i ∈ S then bitSign b else 1))
      = if i ∈ S then (0 : ℝ) else 2 := by
    intro i
    by_cases h : i ∈ S <;> simp [h]
  rw [hprod, Finset.prod_congr rfl fun i _ => hfac i]
  by_cases h : S = ∅
  · simp [h]
  · obtain ⟨i, hi⟩ := Finset.nonempty_iff_ne_empty.2 h
    rw [if_neg h, Finset.prod_eq_zero (Finset.mem_univ i) (by simp [hi])]

/-- And the dual orthogonality, summing over the characters instead: the
`2^n` characters separate the points of the cube. -/
lemma sum_chi_mul_chi (x y : Cube n) :
    ∑ S : Finset (Fin n), chi S x * chi S y = if x = y then (2 : ℝ) ^ n else 0 := by
  have key : ∀ S : Finset (Fin n),
      chi S x * chi S y = ∏ i ∈ S, (bitSign (x i) * bitSign (y i)) := by
    intro S
    rw [chi, chi, ← Finset.prod_mul_distrib]
  have hsum : ∑ S : Finset (Fin n), chi S x * chi S y
      = ∏ i : Fin n, (bitSign (x i) * bitSign (y i) + 1) := by
    rw [Finset.prod_add]
    simp only [Finset.prod_const_one, mul_one, Finset.powerset_univ]
    exact Finset.sum_congr rfl fun S _ => key S
  have hfac : ∀ i : Fin n, bitSign (x i) * bitSign (y i) + 1
      = if x i = y i then (2 : ℝ) else 0 := by
    intro i
    rw [add_comm]
    exact one_add_bitSign_mul _ _
  rw [hsum, Finset.prod_congr rfl fun i _ => hfac i]
  by_cases h : x = y
  · simp [h]
  · obtain ⟨i, hi⟩ := Function.ne_iff.1 h
    rw [if_neg h, Finset.prod_eq_zero (Finset.mem_univ i) (by simp [hi])]

/-- The Fourier coefficient `f̂(S)`. -/
noncomputable def coeff (f : Cube n → ℝ) (S : Finset (Fin n)) : ℝ :=
  (∑ x : Cube n, f x * chi S x) / 2 ^ n

/-- **Every function on the cube is its own Fourier expansion**: the
multilinear representation used in the proof sketch of Appendix E. -/
theorem sum_coeff_mul_chi (f : Cube n → ℝ) (x : Cube n) :
    ∑ S : Finset (Fin n), coeff f S * chi S x = f x := by
  have h2 : ((2 : ℝ) ^ n) ≠ 0 := by positivity
  calc ∑ S : Finset (Fin n), coeff f S * chi S x
      = (∑ S : Finset (Fin n), ∑ y : Cube n, f y * (chi S y * chi S x)) / 2 ^ n := by
        rw [Finset.sum_div]
        refine Finset.sum_congr rfl fun S _ => ?_
        rw [coeff, div_mul_eq_mul_div, Finset.sum_mul]
        exact congrArg (· / 2 ^ n) (Finset.sum_congr rfl fun y _ => by ring)
    _ = (∑ y : Cube n, f y * ∑ S : Finset (Fin n), chi S y * chi S x) / 2 ^ n := by
        rw [Finset.sum_comm]
        exact congrArg (· / 2 ^ n) (Finset.sum_congr rfl fun y _ => (Finset.mul_sum _ _ _).symm)
    _ = f x := by
        simp only [sum_chi_mul_chi, mul_ite, mul_zero, Finset.sum_ite_eq', Finset.mem_univ,
          if_pos]
        field_simp

/-- The coefficients of a character are the indicator of that character. -/
@[simp] lemma coeff_chi (S T : Finset (Fin n)) :
    coeff (chi T) S = if S = T then 1 else 0 := by
  have h2 : ((2 : ℝ) ^ n) ≠ 0 := by positivity
  rw [coeff]
  simp only [chi_mul_chi, sum_chi]
  by_cases h : S = T
  · subst h
    simp [h2]
  · rw [if_neg (fun hst : symmDiff T S = ∅ => h (symmDiff_eq_bot.1 hst).symm), if_neg h,
      zero_div]

lemma coeff_add (f g : Cube n → ℝ) (S : Finset (Fin n)) :
    coeff (fun x => f x + g x) S = coeff f S + coeff g S := by
  simp only [coeff, ← add_div, ← Finset.sum_add_distrib, add_mul]

lemma coeff_smul (c : ℝ) (f : Cube n → ℝ) (S : Finset (Fin n)) :
    coeff (fun x => c * f x) S = c * coeff f S := by
  simp only [coeff, ← mul_div_assoc, ← Finset.mul_sum, mul_assoc]

lemma coeff_sum {ι : Type*} (s : Finset ι) (f : ι → Cube n → ℝ) (S : Finset (Fin n)) :
    coeff (fun x => ∑ a ∈ s, f a x) S = ∑ a ∈ s, coeff (f a) S := by
  classical
  induction s using Finset.induction with
  | empty => simp [coeff]
  | insert a s ha ih =>
      simp only [Finset.sum_insert ha]
      rw [coeff_add, ih]

/-- A function is determined by its coefficients. -/
theorem eq_of_coeff_eq {f g : Cube n → ℝ} (h : ∀ S, coeff f S = coeff g S) : f = g := by
  funext x
  rw [← sum_coeff_mul_chi f x, ← sum_coeff_mul_chi g x]
  exact Finset.sum_congr rfl fun S _ => by rw [h S]


end RASPL
end Transformer
