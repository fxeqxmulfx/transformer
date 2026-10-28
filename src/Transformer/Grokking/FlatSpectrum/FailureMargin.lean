module

public import Transformer.Grokking.FlatSpectrum.BlockCertificate

/-!
# The failure margin

Source: `review/flat_spectrum.md`, *The failure margin* (added). Without a permutation graph in
`Ω`, Theorem E.3 gives a completion of `J` shorter than `N`; this measures by how much. Hall's
theorem gives rows `R` meeting fewer than `#R` columns, and shrinking the other columns to `C`
with `k + l = N + 1` keeps `R × C` outside `Ω` (`exists_block_of_not_perm`). The best witness
`J − t 1_R 1_Cᵀ` equals `J` on `Ω` and has `‖X‖_S1 = √(k (N − l)) + √(l (N − k))`
(`exists_nuclearNorm_eq_of_block`, `exists_nuclearNorm_eq_of_not_perm`). By Cauchy–Schwarz
(`sqrt_mul_add_sqrt_mul_le`) this is at most `√((k + l)(2N − k − l)) = √(N² − (k + l − N)²)`
(`sqrt_add_sqrt_le_sqrt_sq_sub`), so at most `√(N² − 1)` (`sqrt_add_sqrt_le_sqrt_sq_sub_one`),
and `√(N² − 1) < N − 1/(2N)` (`sqrt_sq_sub_one_lt`): M misses `J` by more than `1/(2N)`
(`exists_nuclearNorm_le_sqrt_of_not_perm`). `MarginSharp` shows each value is attained.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset

variable {n : Type*} [Fintype n] [DecidableEq n]

/-- Cauchy–Schwarz for two terms: `√(a b) + √(c d) ≤ √((a + c)(b + d))` for `a, b, c, d ≥ 0`. -/
lemma sqrt_mul_add_sqrt_mul_le {a b c d : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c)
    (hd : 0 ≤ d) : √(a * b) + √(c * d) ≤ √((a + c) * (b + d)) := by
  rw [Real.le_sqrt (by positivity) (by positivity)]
  have h1 := Real.sq_sqrt (mul_nonneg ha hb)
  have h2 := Real.sq_sqrt (mul_nonneg hc hd)
  have h3 := Real.sq_sqrt (mul_nonneg ha hd)
  have h4 := Real.sq_sqrt (mul_nonneg hb hc)
  have h5 : √(a * b) * √(c * d) = √(a * d) * √(b * c) := by
    rw [← Real.sqrt_mul (mul_nonneg ha hb), ← Real.sqrt_mul (mul_nonneg ha hd)]
    congr 1
    ring
  nlinarith [sq_nonneg (√(a * d) - √(b * c))]

example := sqrt_mul_add_sqrt_mul_le (a := 1) (b := 1) (c := 1) (d := 1) zero_le_one zero_le_one
  zero_le_one zero_le_one

/-- **The witness bound**: for `0 ≤ k, l ≤ N`,
`√(k (N − l)) + √(l (N − k)) ≤ √(N² − (k + l − N)²)`. -/
lemma sqrt_add_sqrt_le_sqrt_sq_sub {N k l : ℝ} (hk : 0 ≤ k) (hl : 0 ≤ l) (hkN : k ≤ N)
    (hlN : l ≤ N) : √(k * (N - l)) + √(l * (N - k)) ≤ √(N ^ 2 - (k + l - N) ^ 2) := by
  refine (sqrt_mul_add_sqrt_mul_le hk (sub_nonneg.mpr hlN) hl (sub_nonneg.mpr hkN)).trans_eq ?_
  congr 1
  ring

example := sqrt_add_sqrt_le_sqrt_sq_sub (N := 1) (k := 1) (l := 1) zero_le_one zero_le_one le_rfl
  le_rfl

/-- With `k + l ≥ N + 1`, the witness bound is at most `√(N² − 1)`. -/
lemma sqrt_add_sqrt_le_sqrt_sq_sub_one {N k l : ℝ} (hk : 0 ≤ k) (hl : 0 ≤ l) (hkN : k ≤ N)
    (hlN : l ≤ N) (h : N + 1 ≤ k + l) : √(k * (N - l)) + √(l * (N - k)) ≤ √(N ^ 2 - 1) :=
  (sqrt_add_sqrt_le_sqrt_sq_sub hk hl hkN hlN).trans <| Real.sqrt_le_sqrt <| by
    nlinarith [mul_nonneg (by linarith : (0 : ℝ) ≤ k + l - N - 1)
      (by linarith : (0 : ℝ) ≤ k + l - N + 1)]

example := sqrt_add_sqrt_le_sqrt_sq_sub_one (N := 3) (k := 2) (l := 2) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num)

/-- `√(N² − 1) < N − 1/(2N)` for `N ≥ 1`. -/
lemma sqrt_sq_sub_one_lt {N : ℝ} (hN : 1 ≤ N) : √(N ^ 2 - 1) < N - 1 / (2 * N) := by
  have hN0 : N ≠ 0 := by positivity
  have hpos : 0 < N - 1 / (2 * N) := by
    have := one_div_le_one_div_of_le (by norm_num : (0 : ℝ) < 2) (by linarith : 2 ≤ 2 * N)
    linarith
  rw [Real.sqrt_lt' hpos]
  have e : (N - 1 / (2 * N)) ^ 2 = N ^ 2 - 1 + 1 / (4 * N ^ 2) := by
    field_simp
    ring
  have : 0 < 1 / (4 * N ^ 2) := by positivity
  linarith

example := sqrt_sq_sub_one_lt (N := 1) le_rfl

/-- **Hall's block, made tight**: if `Ω` contains no permutation graph, it misses a block
`R × C` with `#R + #C = N + 1`. -/
theorem exists_block_of_not_perm {Ω : Set (n × n)}
    (hΩ : ¬ ∃ σ : Equiv.Perm n, ∀ i, (i, σ i) ∈ Ω) :
    ∃ R C : Finset n, (∀ i ∈ R, ∀ j ∈ C, (i, j) ∉ Ω) ∧ #R + #C = Fintype.card n + 1 := by
  classical
  set t : n → Finset n := fun i => univ.filter fun j => (i, j) ∈ Ω
  obtain ⟨R, hR⟩ : ∃ R : Finset n, #(R.biUnion t) < #R := by
    by_contra h
    obtain ⟨f, hf, hft⟩ := (all_card_le_biUnion_card_iff_exists_injective t).mp
      fun R => not_lt.mp fun hlt => h ⟨R, hlt⟩
    exact hΩ ⟨Equiv.ofBijective f (Finite.injective_iff_bijective.mp hf),
      fun i => by simpa [t] using hft i⟩
  have hRN := card_le_univ R
  have hD : Fintype.card n + 1 - #R ≤ #(R.biUnion t)ᶜ := by
    rw [card_compl]
    omega
  obtain ⟨C, hCD, hC⟩ := exists_subset_card_eq hD
  exact ⟨R, C, fun i hi j hj hij => mem_compl.mp (hCD hj)
    (mem_biUnion.mpr ⟨i, hi, by simpa [t] using hij⟩), by omega⟩

example := exists_block_of_not_perm (n := Fin 1) (Ω := ∅) fun ⟨_, h⟩ => h 0

/-- **The exact witness of a missed block**, `review/flat_spectrum.md`, *The failure margin*: if
`Ω` misses `R × C` with `N < #R + #C`, the best witness `J − t 1_R 1_Cᵀ` equals `J` on `Ω` and
has `‖X‖_S1 = √(k (N − l)) + √(l (N − k))`. -/
theorem exists_nuclearNorm_eq_of_block {Ω : Set (n × n)} {R C : Finset n}
    (hRC : ∀ i ∈ R, ∀ j ∈ C, (i, j) ∉ Ω) (hN : Fintype.card n < #R + #C) :
    ∃ X : Matrix n n ℂ, (∀ ij ∈ Ω, X ij.1 ij.2 = onesMatrix n ij.1 ij.2) ∧
      nuclearNorm X = √(#R * (Fintype.card n - #C)) + √(#C * (Fintype.card n - #R)) := by
  have hR : R.Nonempty := card_pos.mp (by have := card_le_univ C; omega)
  have hC : C.Nonempty := card_pos.mp (by have := card_le_univ R; omega)
  have hN' : (Fintype.card n : ℝ) < #R + #C := by exact_mod_cast hN
  refine ⟨onesSubBlock R C
    (1 - √(#R * #C * (Fintype.card n - #R) * (Fintype.card n - #C)) / (#R * #C)),
    fun ij hij => ?_, nuclearNorm_onesSubBlock_best hR hC ?_⟩
  · have h : ¬(ij.1 ∈ R ∧ ij.2 ∈ C) := fun h => hRC _ h.1 _ h.2 hij
    simp only [onesSubBlock, onesMatrix, of_apply, ite_eq_right h]
  · nlinarith [mul_le_mul_of_nonneg_right hN'.le (Nat.cast_nonneg _ : (0 : ℝ) ≤ Fintype.card n)]

example := exists_nuclearNorm_eq_of_block (n := Fin 3) (Ω := {ij | ij.1 = 0 ∨ ij.2 = 0})
  (R := {1, 2}) (C := {1, 2}) (by decide) (by decide)

/-- **The failure margin, exactly**, `review/flat_spectrum.md`, *The failure margin*: if `Ω`
contains no permutation graph, it misses a block `R × C` with `#R + #C = N + 1`, and some `X`
equal to `J` on `Ω` has `‖X‖_S1 = √(k (N − l)) + √(l (N − k))` for `k = #R`, `l = #C`. -/
theorem exists_nuclearNorm_eq_of_not_perm {Ω : Set (n × n)}
    (hΩ : ¬ ∃ σ : Equiv.Perm n, ∀ i, (i, σ i) ∈ Ω) :
    ∃ R C : Finset n, (∀ i ∈ R, ∀ j ∈ C, (i, j) ∉ Ω) ∧ #R + #C = Fintype.card n + 1 ∧
      ∃ X : Matrix n n ℂ, (∀ ij ∈ Ω, X ij.1 ij.2 = onesMatrix n ij.1 ij.2) ∧
        nuclearNorm X = √(#R * (Fintype.card n - #C)) + √(#C * (Fintype.card n - #R)) := by
  obtain ⟨R, C, hRC, hcard⟩ := exists_block_of_not_perm hΩ
  exact ⟨R, C, hRC, hcard, exists_nuclearNorm_eq_of_block hRC (by omega)⟩

example := exists_nuclearNorm_eq_of_not_perm (n := Fin 1) (Ω := ∅) fun ⟨_, h⟩ => h 0

/-- **The failure margin**, `review/flat_spectrum.md`, *The failure margin*: if `Ω` contains no
permutation graph, some `X` equal to `J` on `Ω` has `‖X‖_S1 ≤ √(N² − 1) < N − 1/(2N)`. -/
theorem exists_nuclearNorm_le_sqrt_of_not_perm {Ω : Set (n × n)}
    (hΩ : ¬ ∃ σ : Equiv.Perm n, ∀ i, (i, σ i) ∈ Ω) :
    ∃ X : Matrix n n ℂ, (∀ ij ∈ Ω, X ij.1 ij.2 = onesMatrix n ij.1 ij.2) ∧
      nuclearNorm X ≤ √((Fintype.card n : ℝ) ^ 2 - 1) ∧
      nuclearNorm X < Fintype.card n - 1 / (2 * Fintype.card n) := by
  obtain ⟨R, C, -, hcard, X, hX, hnorm⟩ := exists_nuclearNorm_eq_of_not_perm hΩ
  have hk := card_le_univ R
  have hl := card_le_univ C
  have hle : nuclearNorm X ≤ √((Fintype.card n : ℝ) ^ 2 - 1) := by
    rw [hnorm]
    exact sqrt_add_sqrt_le_sqrt_sq_sub_one (Nat.cast_nonneg _) (Nat.cast_nonneg _)
      (by exact_mod_cast hk) (by exact_mod_cast hl) (by exact_mod_cast hcard.ge)
  refine ⟨X, hX, hle, hle.trans_lt (sqrt_sq_sub_one_lt ?_)⟩
  exact_mod_cast (show 1 ≤ Fintype.card n by omega)

example := exists_nuclearNorm_le_sqrt_of_not_perm (n := Fin 1) (Ω := ∅) fun ⟨_, h⟩ => h 0

end Transformer.Grokking.FlatSpectrum
