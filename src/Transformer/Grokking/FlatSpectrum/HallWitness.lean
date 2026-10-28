module

public import Mathlib.Combinatorics.Hall.Basic
public import Transformer.Grokking.FlatSpectrum.PermutationRecovery

/-!
# Without a permutation the least nuclear norm misses `J`

Source: `review/flat_spectrum.md`, *Exact recovery of a rank-one kernel*, Theorem E (added).
Let `Ω` miss the block `R × C`, with `k = #R`, `l = #C` and `k + l > N`. The matrix
`X = J − s² 1_R 1_Cᵀ` (`blockWitness_apply`) equals `J` on `Ω`. It is `U Vᴴ` with
`U = [1 − s² 1_R, s 1_R]` and `V = [1, s 1_{Cᶜ}]`, so Lemma 1 gives
`‖X‖_S1 ≤ N − s² (k + l − N − k s²)/2` (`nuclearNorm_blockWitness_le`). This is below `N` for
`s = 1/(k + 1)` (`exists_nuclearNorm_lt_of_block`). If `Ω` contains no permutation graph,
Hall's theorem (`Finset.all_card_le_biUnion_card_iff_exists_injective`) gives rows `R` with
fewer than `#R` neighbours `N(R)`, and `C = N(R)ᶜ` is such a block
(`exists_nuclearNorm_lt_of_not_perm`). So `J` is not a minimum
(`not_isMinOn_onesMatrix_of_not_perm`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset

variable {n : Type*} [Fintype n] [DecidableEq n]

/-- The left factor `[1 − s² 1_R, s 1_R]` of the block witness. -/
noncomputable def blockFactorL (R : Finset n) (s : ℝ) : Matrix n (Fin 2) ℂ :=
  of fun i k => ((if i ∈ R then ![1 - s ^ 2, s] k else ![1, 0] k : ℝ) : ℂ)

/-- The right factor `[1, s 1_{Cᶜ}]` of the block witness. -/
noncomputable def blockFactorR (C : Finset n) (s : ℝ) : Matrix n (Fin 2) ℂ :=
  of fun j k => ((if j ∈ C then ![1, 0] k else ![1, s] k : ℝ) : ℂ)

omit [Fintype n] in
/-- The block witness is `J − s² 1_R 1_Cᵀ`. -/
lemma blockWitness_apply {m : Type*} [DecidableEq m] (R : Finset m) (C : Finset n) (s : ℝ)
    (i : m) (j : n) :
    (blockFactorL R s * (blockFactorR C s)ᴴ) i j =
      if i ∈ R ∧ j ∈ C then 1 - (s : ℂ) ^ 2 else 1 := by
  by_cases hi : i ∈ R <;> by_cases hj : j ∈ C <;>
    simp [mul_apply, blockFactorL, blockFactorR, Fin.sum_univ_two, hi, hj, sq]

/-- `∑_i (a if i ∈ R else b) = #R a + (N − #R) b`. -/
lemma sum_ite_mem_univ (R : Finset n) (a b : ℝ) :
    ∑ i, (if i ∈ R then a else b) = #R * a + (Fintype.card n - #R) * b := by
  have h1 (i : n) (hi : i ∈ R) : (if i ∈ R then a else b) = a := ite_eq_left hi
  have h2 (i : n) (hi : i ∈ Rᶜ) : (if i ∈ R then a else b) = b := ite_eq_right (mem_compl.mp hi)
  rw [← sum_add_sum_compl R, sum_congr rfl h1, sum_congr rfl h2, sum_const, sum_const,
    nsmul_eq_mul, nsmul_eq_mul, card_compl, Nat.cast_sub (card_le_univ R)]

/-- `‖[1 − s² 1_R, s 1_R]‖² = N − #R s² + #R s⁴`. -/
lemma frobeniusSq_blockFactorL (R : Finset n) (s : ℝ) :
    frobeniusSq (blockFactorL R s) = Fintype.card n - #R * s ^ 2 + #R * s ^ 4 := by
  have h (i : n) : ∑ k, ‖blockFactorL R s i k‖ ^ 2 =
      if i ∈ R then (1 - s ^ 2) ^ 2 + s ^ 2 else 1 := by
    by_cases hi : i ∈ R <;>
      simp [blockFactorL, hi, Fin.sum_univ_two, Complex.norm_real, sq_abs]
  rw [frobeniusSq, sum_congr rfl fun i _ => h i, sum_ite_mem_univ]
  ring

/-- `‖[1, s 1_{Cᶜ}]‖² = N + (N − #C) s²`. -/
lemma frobeniusSq_blockFactorR (C : Finset n) (s : ℝ) :
    frobeniusSq (blockFactorR C s) = Fintype.card n + (Fintype.card n - #C) * s ^ 2 := by
  have h (j : n) : ∑ k, ‖blockFactorR C s j k‖ ^ 2 = if j ∈ C then 1 else 1 + s ^ 2 := by
    by_cases hj : j ∈ C <;>
      simp [blockFactorR, hj, Fin.sum_univ_two, Complex.norm_real, sq_abs]
  rw [frobeniusSq, sum_congr rfl fun j _ => h j, sum_ite_mem_univ]
  ring

/-- **The block witness is short**: `‖J − s² 1_R 1_Cᵀ‖_S1 ≤ N − s² (#R + #C − N − #R s²)/2`. -/
theorem nuclearNorm_blockWitness_le (R C : Finset n) (s : ℝ) :
    nuclearNorm (blockFactorL R s * (blockFactorR C s)ᴴ) ≤
      Fintype.card n - s ^ 2 * (#R + #C - Fintype.card n - #R * s ^ 2) / 2 := by
  have h := nuclearNorm_le_of_eq_mul (blockFactorL R s) (blockFactorR C s) rfl
  rw [frobeniusSq_blockFactorL, frobeniusSq_blockFactorR] at h
  exact h.trans_eq (by ring)

/-- **A large missed block beats `J`**, `review/flat_spectrum.md`, Theorem E: if `Ω` misses
`R × C` with `N < #R + #C`, some `X` equal to `J` on `Ω` has `‖X‖_S1 < N`. -/
theorem exists_nuclearNorm_lt_of_block {Ω : Set (n × n)} {R C : Finset n}
    (hRC : ∀ i ∈ R, ∀ j ∈ C, (i, j) ∉ Ω) (hN : Fintype.card n < #R + #C) :
    ∃ X : Matrix n n ℂ, (∀ ij ∈ Ω, X ij.1 ij.2 = onesMatrix n ij.1 ij.2) ∧
      nuclearNorm X < Fintype.card n := by
  set s : ℝ := 1 / (#R + 1)
  refine ⟨blockFactorL R s * (blockFactorR C s)ᴴ, fun ij hij => ?_, ?_⟩
  · rw [blockWitness_apply, ite_eq_right fun h => hRC _ h.1 _ h.2 hij]
    rfl
  have hk : (0 : ℝ) ≤ #R := Nat.cast_nonneg _
  have hN' : (Fintype.card n : ℝ) + 1 ≤ #R + #C := by exact_mod_cast hN
  have hks : (#R : ℝ) * s ^ 2 < 1 := by
    rw [div_pow, one_pow, mul_one_div, div_lt_one (by positivity)]
    nlinarith
  have hpos : 0 < s ^ 2 * (#R + #C - Fintype.card n - #R * s ^ 2) :=
    mul_pos (by positivity) (by linarith)
  exact (nuclearNorm_blockWitness_le R C s).trans_lt (by linarith)

example := exists_nuclearNorm_lt_of_block (n := Fin 3) (Ω := {ij | ij.1 = 0 ∨ ij.2 = 0})
  (R := {1, 2}) (C := {1, 2}) (by decide) (by decide)

/-- **Without a permutation graph `J` is beaten**, `review/flat_spectrum.md`, Theorem E: if
`Ω` contains the graph of no permutation, some `X` equal to `J` on `Ω` has `‖X‖_S1 < N`. -/
theorem exists_nuclearNorm_lt_of_not_perm {Ω : Set (n × n)}
    (hΩ : ¬ ∃ σ : Equiv.Perm n, ∀ i, (i, σ i) ∈ Ω) :
    ∃ X : Matrix n n ℂ, (∀ ij ∈ Ω, X ij.1 ij.2 = onesMatrix n ij.1 ij.2) ∧
      nuclearNorm X < Fintype.card n := by
  classical
  set t : n → Finset n := fun i => univ.filter fun j => (i, j) ∈ Ω
  obtain ⟨R, hR⟩ : ∃ R : Finset n, #(R.biUnion t) < #R := by
    by_contra h
    obtain ⟨f, hf, hft⟩ := (all_card_le_biUnion_card_iff_exists_injective t).mp
      fun R => not_lt.mp fun hlt => h ⟨R, hlt⟩
    exact hΩ ⟨Equiv.ofBijective f (Finite.injective_iff_bijective.mp hf),
      fun i => by simpa [t] using hft i⟩
  refine exists_nuclearNorm_lt_of_block (R := R) (C := (R.biUnion t)ᶜ)
    (fun i hi j hj hij => mem_compl.mp hj (mem_biUnion.mpr ⟨i, hi, by simpa [t] using hij⟩)) ?_
  have := card_le_univ (R.biUnion t)
  rw [card_compl]
  omega

example := exists_nuclearNorm_lt_of_not_perm (n := Fin 1) (Ω := ∅) fun ⟨_, h⟩ => h 0

/-- **Without a permutation graph `J` is not a minimum**, `review/flat_spectrum.md`,
Theorem E: if `Ω` contains the graph of no permutation, `J` is not a minimum of the nuclear
norm among the matrices equal to `J` on `Ω`. -/
theorem not_isMinOn_onesMatrix_of_not_perm {Ω : Set (n × n)}
    (hΩ : ¬ ∃ σ : Equiv.Perm n, ∀ i, (i, σ i) ∈ Ω) :
    ¬ IsMinOn nuclearNorm {X' | ∀ ij ∈ Ω, X' ij.1 ij.2 = onesMatrix n ij.1 ij.2}
      (onesMatrix n) := fun hmin => by
  obtain ⟨X, hX, hlt⟩ := exists_nuclearNorm_lt_of_not_perm hΩ
  have h := isMinOn_iff.mp hmin X hX
  rw [nuclearNorm_onesMatrix] at h
  linarith

example := not_isMinOn_onesMatrix_of_not_perm (n := Fin 1) (Ω := ∅) fun ⟨_, h⟩ => h 0

end Transformer.Grokking.FlatSpectrum
