module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.ModularCoding.Basic

/-!
# Checked recombination of modular categorical constraints

Source: `review/flat_spectrum.md`, CM27. An actual Bezout witness permits
recombining component solutions without assuming a prime latent domain.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.ModularCoding

variable {I J : Type*}

/-- Combine the actual integer representatives with checked Bezout coefficients,
CM27. Their validity is a theorem hypothesis, not part of this definition. -/
def recombine (m n u v : ℤ) (x y : J → ℤ) (j : J) : ℤ :=
  v * n * x j + u * m * y j

/-- The constructed representative retains both component residues, CM27. -/
theorem recombine_residues {m n u v : ℤ} (h : u * m + v * n = 1)
    (x y : J → ℤ) (j : J) :
    m ∣ recombine m n u v x y j - x j ∧
      n ∣ recombine m n u v x y j - y j := by
  constructor
  · refine ⟨u * (y j - x j), ?_⟩
    unfold recombine
    linear_combination x j * h
  · refine ⟨v * (x j - y j), ?_⟩
    unfold recombine
    linear_combination y j * h

/-- Component agreement implies product agreement under the same checked
Bezout condition, CM27. No claim is made for arbitrary noncoprime moduli. -/
theorem product_dvd {m n u v a : ℤ} (h : u * m + v * n = 1)
    (hm : m ∣ a) (hn : n ∣ a) : m * n ∣ a := by
  obtain ⟨k, hk⟩ := hm
  obtain ⟨l, hl⟩ := hn
  refine ⟨v * k + u * l, ?_⟩
  linear_combination -a * h + (v * n) * hk + (u * m) * hl

variable [Fintype J]

/-- Actual matrix sums distribute over the constructed representative, CM27. -/
theorem action_recombine (A : I → J → ℤ) (m n u v : ℤ) (x y : J → ℤ) (i : I) :
    action A (recombine m n u v x y) i =
      v * n * action A x i + u * m * action A y i := by
  simp only [action, recombine, mul_add, Finset.sum_add_distrib, Finset.mul_sum]
  congr 1 <;> apply Finset.sum_congr rfl <;> intro j _ <;> ring

/-- **Exact component recombination**, CM27. Every original integer equation
holds at the product modulus when both components fit and Bezout is checked. -/
theorem fits_recombine {A : I → J → ℤ} {b : I → ℤ} {m n u v : ℤ} {x y : J → ℤ}
    (h : u * m + v * n = 1) (hx : Fits m A b x) (hy : Fits n A b y) :
    Fits (m * n) A b (recombine m n u v x y) := by
  intro i
  obtain ⟨k, hk⟩ := hx i
  obtain ⟨l, hl⟩ := hy i
  refine ⟨v * k + u * l, ?_⟩
  rw [action_recombine]
  linear_combination -(b i) * h + (v * n) * hk + (u * m) * hl

example := recombine_residues (m := 4) (n := 3) (u := 1) (v := -1)
  (by norm_num) (![1, 2] : Fin 2 → ℤ) ![2, 2] 0

example := product_dvd (m := 4) (n := 3) (u := 1) (v := -1) (a := 12)
  (by norm_num) (by norm_num) (by norm_num)

example := action_recombine (![![1, 2], ![3, 4]] : Fin 2 → Fin 2 → ℤ)
  4 3 1 (-1) ![1, 2] ![2, 2] 1

example := fits_recombine
  (A := (![![1, 2], ![3, 4]] : Fin 2 → Fin 2 → ℤ)) (b := ![9, 23])
  (m := 4) (n := 3) (u := 1) (v := -1) (x := ![1, 2]) (y := ![2, 2])
  (by norm_num)
  (by norm_num [Fits, action, Fin.sum_univ_two, Fin.forall_fin_two])
  (by norm_num [Fits, action, Fin.sum_univ_two, Fin.forall_fin_two])

/-- CM27's missing-coprimality obstruction: agreement modulo four and six
does not imply agreement modulo their product. -/
theorem noncoprime_product_counterexample :
    (4 : ℤ) ∣ 12 ∧ (6 : ℤ) ∣ 12 ∧ ¬ (4 * 6 : ℤ) ∣ 12 := by
  norm_num

end Transformer.Grokking.FlatSpectrum.CoupledMoments.ModularCoding
