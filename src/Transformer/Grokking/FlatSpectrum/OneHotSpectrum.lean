module

public import Transformer.Grokking.FlatSpectrum.OneHotFill

/-!
# Spectra of the one-hot kernels `[f = c]` with one answer per row

Source: `review/flat_spectrum.md`, Torch check: "Other one-hots 0.56p–0.67p". If row `a` of
`K = [f = c]` holds at most one 1, distinct columns have disjoint supports, so `K* K` is diagonal
with the column counts `N_b = #{a | f(a, b) = c}`. Its characteristic polynomial is
`∏_b (X − N_b)`, so the singular values of `K` are the `√N_b`: `‖K‖_S1 = ∑_b √N_b`,
`‖K‖²_S2 = ∑_b N_b` and `r_eff K = (∑_b √N_b)² / ∑_b N_b`.

For `f = a³ + a b` over a prime `p` and `c ≠ 0`, row `0` is empty and row `a ≠ 0` has its 1 at
`b = a⁻¹ (c − a³)`. So `∑_b N_b = p − 1` and `r_eff = (∑_b √N_b)² / (p − 1)`, where `N_b` is the
number of roots of `X³ + b X − c`. The measured range `0.56p–0.67p` over `c` is not formalized.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset Polynomial

variable {n : Type*} [Fintype n] [DecidableEq n]

/-- If `A.charpoly = ∏ (X − d_i)` for real `d`, the eigenvalues of `A` are the `d_i`:
`∑ g(λ_i) = ∑ g(d_i)` for every `g`. -/
lemma sum_comp_eigenvalues_of_charpoly {A : Matrix n n ℂ} (hA : A.IsHermitian) {d : n → ℝ}
    (h : A.charpoly = ∏ i, (X - C ((d i : ℝ) : ℂ))) (g : ℝ → ℝ) :
    ∑ i, g (hA.eigenvalues i) = ∑ i, g (d i) := by
  have h1 : A.charpoly.roots = Multiset.map (fun i => ((d i : ℝ) : ℂ)) univ.val := by
    rw [h, roots_prod _ _ (prod_ne_zero_iff.mpr fun i _ => X_sub_C_ne_zero _)]
    simp only [roots_X_sub_C, Multiset.bind_singleton]
  have h2 : Multiset.map hA.eigenvalues univ.val = Multiset.map d univ.val := by
    have h3 := congrArg (Multiset.map Complex.re)
      (hA.roots_charpoly_eq_eigenvalues.symm.trans h1)
    simpa [Multiset.map_map, Function.comp_def] using h3
  calc ∑ i, g (hA.eigenvalues i) = (Multiset.map g (Multiset.map hA.eigenvalues univ.val)).sum := by
        rw [Multiset.map_map]; rfl
    _ = ∑ i, g (d i) := by rw [h2, Multiset.map_map]; rfl

example := sum_comp_eigenvalues_of_charpoly (isHermitian_zero (n := Fin 2) (α := ℂ))
  (d := fun _ => 0) (by simp [charpoly_zero]) id

variable {G : Type*} [Fintype G] [DecidableEq G]

/-- With at most one answer `c` per row, `K* K` is diagonal with the column counts:
`(K* K)(b, b) = #{a | f(a, b) = c}`. -/
theorem conjTranspose_mul_classKernel (f : G → G → G) (c : G)
    (huniq : ∀ a b b', f a b = c → f a b' = c → b' = b) :
    (classKernel f c)ᴴ * classKernel f c = diagonal fun b => ((#{a | f a b = c} : ℕ) : ℂ) := by
  ext b b'
  simp only [mul_apply, conjTranspose_apply, classKernel, taskKernel, of_apply, diagonal_apply]
  rcases eq_or_ne b b' with rfl | hbb
  · rw [ite_eq_left rfl, ← sum_boole]
    refine sum_congr rfl fun a _ => ?_
    split_ifs <;> simp
  · rw [ite_eq_right hbb]
    refine sum_eq_zero fun a _ => ?_
    by_cases h1 : f a b = c
    · have h2 : ¬f a b' = c := fun h2 => hbb (huniq a b b' h1 h2).symm
      simp [h2]
    · simp [h1]

example := conjTranspose_mul_classKernel (· + · : ZMod 3 → ZMod 3 → ZMod 3) 0 (by decide)

/-- With at most one answer `c` per row, the singular values of `[f = c]` are the `√N_b`:
`‖K‖_S1 = ∑_b √N_b`. -/
theorem nuclearNorm_classKernel (f : G → G → G) (c : G)
    (huniq : ∀ a b b', f a b = c → f a b' = c → b' = b) :
    nuclearNorm (classKernel f c) = ∑ b, √(#{a | f a b = c} : ℝ) := by
  have h := sum_comp_eigenvalues_of_charpoly (isHermitian_conjTranspose_mul_self (classKernel f c))
    (d := fun b => (#{a | f a b = c} : ℝ)) (by
      rw [conjTranspose_mul_classKernel f c huniq, charpoly_diagonal]
      simp only [Complex.ofReal_natCast]) Real.sqrt
  simpa only [nuclearNorm, singularValues] using h

example := nuclearNorm_classKernel (· + · : ZMod 3 → ZMod 3 → ZMod 3) 0 (by decide)

omit [DecidableEq G] in
/-- `‖[f = c]‖²_S2 = ∑_b N_b`. -/
theorem frobeniusSq_classKernel [DecidableEq G] (f : G → G → G) (c : G) :
    frobeniusSq (classKernel f c) = ∑ b, (#{a | f a b = c} : ℝ) := by
  rw [frobeniusSq, sum_comm]
  refine sum_congr rfl fun b _ => ?_
  simp only [classKernel, taskKernel, of_apply, apply_ite (‖·‖ ^ 2), norm_one, one_pow, norm_zero,
    ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow, sum_boole]

/-- With at most one answer `c` per row, `r_eff [f = c] = (∑_b √N_b)² / ∑_b N_b`. -/
theorem effectiveRank_classKernel (f : G → G → G) (c : G)
    (huniq : ∀ a b b', f a b = c → f a b' = c → b' = b) :
    effectiveRank (classKernel f c) =
      (∑ b, √(#{a | f a b = c} : ℝ)) ^ 2 / ∑ b, (#{a | f a b = c} : ℝ) := by
  rw [effectiveRank, nuclearNorm_classKernel f c huniq, frobeniusSq_classKernel, div_pow,
    Real.sq_sqrt (sum_nonneg fun b _ => Nat.cast_nonneg _)]

example := effectiveRank_classKernel (· + · : ZMod 3 → ZMod 3 → ZMod 3) 0 (by decide)

variable {p : ℕ} [Fact p.Prime]

/-- For `c ≠ 0`, every row of `a³ + a b` takes the value `c` at most once. -/
lemma cube_add_mul_unique_of_ne {c : ZMod p} (hc : c ≠ 0) (a b b' : ZMod p)
    (hab : a ^ 3 + a * b = c) (hb' : a ^ 3 + a * b' = c) : b' = b := by
  have ha : a ≠ 0 := by
    rintro rfl
    exact hc (by simpa using hab.symm)
  exact cube_add_mul_unique ha hab hb'

example := @cube_add_mul_unique_of_ne 5 ⟨Nat.prime_five⟩ 4 (by decide) 1 3 3 (by decide)
  (by decide)

/-- `‖[a³ + a b = c]‖²_S2 = p − 1` for `c ≠ 0`: row `0` is empty and every other row holds one
1. -/
theorem frobeniusSq_classKernel_cube_add_mul {c : ZMod p} (hc : c ≠ 0) :
    frobeniusSq (classKernel (fun a b => a ^ 3 + a * b) c) = p - 1 := by
  have hrow (a : ZMod p) : ∑ b, ‖classKernel (fun a b => a ^ 3 + a * b) c a b‖ ^ 2 =
      1 - if a = 0 then 1 else 0 := by
    simp only [classKernel, taskKernel, of_apply, apply_ite (‖·‖ ^ 2), norm_one, one_pow,
      norm_zero, ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow, sum_boole]
    rcases eq_or_ne a 0 with rfl | ha
    · simp [Ne.symm hc]
    · simp only [cube_add_mul_eq_iff ha, filter_eq', mem_univ, ite_true, card_singleton,
        Nat.cast_one, ite_eq_right ha, sub_zero]
  rw [frobeniusSq, sum_congr rfl fun a _ => hrow a, sum_sub_distrib, sum_ite_eq']
  simp [ZMod.card]

example := @frobeniusSq_classKernel_cube_add_mul 5 ⟨Nat.prime_five⟩ 1 (by decide)

/-- **The other one-hot readouts**, `review/flat_spectrum.md`, Torch check: for `c ≠ 0`,
`r_eff [a³ + a b = c] = (∑_b √N_b)² / (p − 1)` with `N_b = #{a | a³ + a b = c}`. -/
theorem effectiveRank_classKernel_cube_add_mul {c : ZMod p} (hc : c ≠ 0) :
    effectiveRank (classKernel (fun a b => a ^ 3 + a * b) c) =
      (∑ b, √(#{a | a ^ 3 + a * b = c} : ℝ)) ^ 2 / (p - 1) := by
  have hp : (1 : ℝ) ≤ p := by exact_mod_cast (Fact.out : p.Prime).one_le
  rw [effectiveRank, nuclearNorm_classKernel _ c (cube_add_mul_unique_of_ne hc),
    frobeniusSq_classKernel_cube_add_mul hc, div_pow, Real.sq_sqrt (by linarith)]

example := @effectiveRank_classKernel_cube_add_mul 5 ⟨Nat.prime_five⟩ 1 (by decide)

end Transformer.Grokking.FlatSpectrum
