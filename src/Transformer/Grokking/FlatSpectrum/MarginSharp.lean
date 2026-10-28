module

public import Transformer.Grokking.FlatSpectrum.FailureMargin

/-!
# The failure margin is attained

Source: `review/flat_spectrum.md`, *The failure margin* (added). A block `R × C` with `k + l > N`
leaves no permutation graph in its complement, since a permutation would map `R` into `Cᶜ`
(`not_exists_perm_of_block`), and by `isLeast_nuclearNorm_compl_block` the least norm of the
completions of `J` from the complement is the witness's `√(k (N − l)) + √(l (N − k))`
(`isLeast_nuclearNorm_compl_block_card`, `exists_compl_block_isLeast`). With
`exists_nuclearNorm_eq_of_not_perm`: every pattern without a permutation graph has a completion
of norm `√(k (N − l)) + √(l (N − k))` for some `k + l = N + 1`, and each of these values is the
least completion norm of some pattern without a permutation graph. For odd `N`, `√(N² − 1)` is
reached at `k = l = (N + 1)/2`, so it is the largest (`isLeast_nuclearNorm_compl_half_block`,
`failureMargin_odd`). For `N = 4` the largest is `√2 + √6`, at `k = 2`, `l = 3`, between
`3.8637` and `3.8638` (`sqrt_two_add_sqrt_six_mem`, `failureMargin_four`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset

variable {n : Type*} [Fintype n] [DecidableEq n]

/-- **A large missed block leaves no permutation graph**: if `Ω` misses `R × C` with
`N < #R + #C`, no permutation `σ` has `(i, σ i) ∈ Ω` for every `i`. -/
theorem not_exists_perm_of_block {Ω : Set (n × n)} {R C : Finset n}
    (hRC : ∀ i ∈ R, ∀ j ∈ C, (i, j) ∉ Ω) (hN : Fintype.card n < #R + #C) :
    ¬ ∃ σ : Equiv.Perm n, ∀ i, (i, σ i) ∈ Ω := by
  rintro ⟨σ, hσ⟩
  have h : #R ≤ #Cᶜ := card_le_card_of_injOn σ
    (fun i hi => mem_coe.mpr (mem_compl.mpr fun hc => hRC i (mem_coe.mp hi) (σ i) hc (hσ i)))
    σ.injective.injOn
  rw [card_compl] at h
  have := card_le_univ C
  omega

example := not_exists_perm_of_block (n := Fin 2) (Ω := ∅) (R := {0}) (C := univ)
  (fun _ _ _ _ => Set.notMem_empty _) (by simp)

/-- **Each margin is attained**, `review/flat_spectrum.md`, *The failure margin*: for
`N < #R + #C`, the complement of `R × C` contains no permutation graph, and the least nuclear
norm of the matrices equal to `J` on it is `√(k (N − l)) + √(l (N − k))`. -/
theorem isLeast_nuclearNorm_compl_block_card {R C : Finset n}
    (h : Fintype.card n < #R + #C) :
    (¬ ∃ σ : Equiv.Perm n, ∀ i, (i, σ i) ∈ {ij : n × n | ¬(ij.1 ∈ R ∧ ij.2 ∈ C)}) ∧
      IsLeast (nuclearNorm '' {X | ∀ ij ∈ {ij : n × n | ¬(ij.1 ∈ R ∧ ij.2 ∈ C)},
        X ij.1 ij.2 = onesMatrix n ij.1 ij.2})
        (√(#R * (Fintype.card n - #C)) + √(#C * (Fintype.card n - #R))) := by
  refine ⟨not_exists_perm_of_block (fun i hi j hj hij => by
    simp only [Set.mem_ofPred_eq] at hij
    exact hij ⟨hi, hj⟩) h, ?_⟩
  have hR : R.Nonempty := card_pos.mp (by have := card_le_univ C; omega)
  have hC : C.Nonempty := card_pos.mp (by have := card_le_univ R; omega)
  have hN' : (Fintype.card n : ℝ) < #R + #C := by exact_mod_cast h
  have hS : {X : Matrix n n ℂ | ∀ ij ∈ {ij : n × n | ¬(ij.1 ∈ R ∧ ij.2 ∈ C)},
      X ij.1 ij.2 = onesMatrix n ij.1 ij.2} = {X | ∀ i j, ¬(i ∈ R ∧ j ∈ C) → X i j = 1} := by
    ext X
    simp only [Set.mem_ofPred_eq, Prod.forall, onesMatrix, of_apply]
  rw [hS]
  exact isLeast_nuclearNorm_compl_block hR hC
    (by nlinarith [mul_le_mul_of_nonneg_right hN'.le (Nat.cast_nonneg _ : (0 : ℝ) ≤ Fintype.card n)])

example := isLeast_nuclearNorm_compl_block_card (n := Fin 3) (R := {1, 2}) (C := {1, 2})
  (by decide)

/-- **For odd `N` the bound `√(N² − 1)` is attained**, `review/flat_spectrum.md`, *The failure
margin*: if `#R = #C = (N + 1)/2`, the complement of `R × C` contains no permutation graph, and
the least nuclear norm of the matrices equal to `J` on it is `√(N² − 1)`. -/
theorem isLeast_nuclearNorm_compl_half_block {R C : Finset n}
    (hR : 2 * #R = Fintype.card n + 1) (hC : 2 * #C = Fintype.card n + 1) :
    (¬ ∃ σ : Equiv.Perm n, ∀ i, (i, σ i) ∈ {ij : n × n | ¬(ij.1 ∈ R ∧ ij.2 ∈ C)}) ∧
      IsLeast (nuclearNorm '' {X | ∀ ij ∈ {ij : n × n | ¬(ij.1 ∈ R ∧ ij.2 ∈ C)},
        X ij.1 ij.2 = onesMatrix n ij.1 ij.2}) (√((Fintype.card n : ℝ) ^ 2 - 1)) := by
  have h := isLeast_nuclearNorm_compl_block_card (R := R) (C := C) (by omega)
  have hk : (#R : ℝ) = (Fintype.card n + 1) / 2 := by
    have : (2 * #R : ℝ) = Fintype.card n + 1 := by exact_mod_cast hR
    linarith
  have hl : (#C : ℝ) = (Fintype.card n + 1) / 2 := by
    have : (2 * #C : ℝ) = Fintype.card n + 1 := by exact_mod_cast hC
    linarith
  have e : √(#R * (Fintype.card n - #C)) + √(#C * (Fintype.card n - #R)) =
      √((Fintype.card n : ℝ) ^ 2 - 1) := by
    rw [hk, hl, ← two_mul, show (Fintype.card n : ℝ) ^ 2 - 1 =
      2 ^ 2 * ((Fintype.card n + 1) / 2 * (Fintype.card n - (Fintype.card n + 1) / 2)) by ring,
      Real.sqrt_mul (by norm_num : (0 : ℝ) ≤ 2 ^ 2), Real.sqrt_sq (by norm_num : (0 : ℝ) ≤ 2)]
  rwa [e] at h

example := isLeast_nuclearNorm_compl_half_block (n := Fin 3) (R := {0, 1}) (C := {0, 1})
  (by decide) (by decide)

/-- **Every margin occurs**: for `k, l ≤ N` with `N < k + l`, some `R`, `C` of sizes `k` and
`l` leave no permutation graph in the complement of `R × C`, and the least nuclear norm of the
matrices equal to `J` on it is `√(k (N − l)) + √(l (N − k))`. -/
theorem exists_compl_block_isLeast {k l : ℕ} (hk : k ≤ Fintype.card n)
    (hl : l ≤ Fintype.card n) (h : Fintype.card n < k + l) :
    ∃ R C : Finset n, #R = k ∧ #C = l ∧
      (¬ ∃ σ : Equiv.Perm n, ∀ i, (i, σ i) ∈ {ij : n × n | ¬(ij.1 ∈ R ∧ ij.2 ∈ C)}) ∧
      IsLeast (nuclearNorm '' {X | ∀ ij ∈ {ij : n × n | ¬(ij.1 ∈ R ∧ ij.2 ∈ C)},
        X ij.1 ij.2 = onesMatrix n ij.1 ij.2})
        (√(k * (Fintype.card n - l)) + √(l * (Fintype.card n - k))) := by
  obtain ⟨R, -, hR⟩ := exists_subset_card_eq (show k ≤ #(univ : Finset n) by rwa [card_univ])
  obtain ⟨C, -, hC⟩ := exists_subset_card_eq (show l ≤ #(univ : Finset n) by rwa [card_univ])
  have h' := isLeast_nuclearNorm_compl_block_card (R := R) (C := C) (by omega)
  rw [hR, hC] at h'
  exact ⟨R, C, hR, hC, h'⟩

example := exists_compl_block_isLeast (n := Fin 3) (k := 2) (l := 2) (by simp) (by simp)
  (by simp)

/-- **The worst failure for odd `N`**, `review/flat_spectrum.md`, *The failure margin*: for odd
`N`, every `Ω` without a permutation graph has a completion of `J` of norm at most `√(N² − 1)`,
and some `Ω` without a permutation graph has least completion norm exactly `√(N² − 1)`. -/
theorem failureMargin_odd (hN : Odd (Fintype.card n)) :
    (∀ Ω : Set (n × n), (¬ ∃ σ : Equiv.Perm n, ∀ i, (i, σ i) ∈ Ω) →
      ∃ X : Matrix n n ℂ, (∀ ij ∈ Ω, X ij.1 ij.2 = onesMatrix n ij.1 ij.2) ∧
        nuclearNorm X ≤ √((Fintype.card n : ℝ) ^ 2 - 1)) ∧
    ∃ Ω : Set (n × n), (¬ ∃ σ : Equiv.Perm n, ∀ i, (i, σ i) ∈ Ω) ∧
      IsLeast (nuclearNorm '' {X | ∀ ij ∈ Ω, X ij.1 ij.2 = onesMatrix n ij.1 ij.2})
        (√((Fintype.card n : ℝ) ^ 2 - 1)) := by
  refine ⟨fun Ω hΩ => ?_, ?_⟩
  · obtain ⟨X, hX, hle, -⟩ := exists_nuclearNorm_le_sqrt_of_not_perm hΩ
    exact ⟨X, hX, hle⟩
  · obtain ⟨m, hm⟩ := hN
    obtain ⟨R, -, hR⟩ :=
      exists_subset_card_eq (show m + 1 ≤ #(univ : Finset n) by rw [card_univ]; omega)
    exact ⟨_, isLeast_nuclearNorm_compl_half_block (R := R) (C := R) (by omega) (by omega)⟩

example := failureMargin_odd (n := Fin 3) (by decide)

/-- `√2 + √6` lies between `3.8637` and `3.8638`. -/
lemma sqrt_two_add_sqrt_six_mem : 3.8637 < √2 + √6 ∧ √2 + √6 < 3.8638 := by
  have h2 : (1.414213 : ℝ) < √2 := (Real.lt_sqrt (by norm_num)).mpr (by norm_num)
  have h6 : (2.449489 : ℝ) < √6 := (Real.lt_sqrt (by norm_num)).mpr (by norm_num)
  have h2' : √2 < (1.414214 : ℝ) := (Real.sqrt_lt' (by norm_num)).mpr (by norm_num)
  have h6' : √6 < (2.44949 : ℝ) := (Real.sqrt_lt' (by norm_num)).mpr (by norm_num)
  constructor <;> linarith

/-- **The worst failure for `N = 4`**, `review/flat_spectrum.md`, *The failure margin*: for
`N = 4`, every `Ω` without a permutation graph has a completion of `J` of norm at most
`√2 + √6`, and some `Ω` without a permutation graph, the complement of a `2 × 3` block, has
least completion norm exactly `√2 + √6`. -/
theorem failureMargin_four (hN : Fintype.card n = 4) :
    (∀ Ω : Set (n × n), (¬ ∃ σ : Equiv.Perm n, ∀ i, (i, σ i) ∈ Ω) →
      ∃ X : Matrix n n ℂ, (∀ ij ∈ Ω, X ij.1 ij.2 = onesMatrix n ij.1 ij.2) ∧
        nuclearNorm X ≤ √2 + √6) ∧
    ∃ Ω : Set (n × n), (¬ ∃ σ : Equiv.Perm n, ∀ i, (i, σ i) ∈ Ω) ∧
      IsLeast (nuclearNorm '' {X | ∀ ij ∈ Ω, X ij.1 ij.2 = onesMatrix n ij.1 ij.2})
        (√2 + √6) := by
  have h12 : √12 < √2 + √6 := by
    have : √12 < 3.5 := (Real.sqrt_lt' (by norm_num)).mpr (by norm_num)
    linarith [sqrt_two_add_sqrt_six_mem.1]
  refine ⟨fun Ω hΩ => ?_, ?_⟩
  · obtain ⟨R, C, -, hcard, X, hX, hnorm⟩ := exists_nuclearNorm_eq_of_not_perm hΩ
    refine ⟨X, hX, ?_⟩
    have hk := card_le_univ R
    have hl := card_le_univ C
    rw [hnorm, hN]
    rw [hN] at hcard hk hl
    obtain h | h | h | h : #R = 1 ∨ #R = 2 ∨ #R = 3 ∨ #R = 4 := by omega
    all_goals (rw [show #C = 5 - #R by omega, h]; norm_num)
    all_goals linarith
  · obtain ⟨R, C, -, -, h⟩ := exists_compl_block_isLeast (n := n) (k := 2) (l := 3) (by omega)
      (by omega) (by omega)
    have e : √(((2 : ℕ) : ℝ) * ((Fintype.card n : ℝ) - ((3 : ℕ) : ℝ))) +
        √(((3 : ℕ) : ℝ) * ((Fintype.card n : ℝ) - ((2 : ℕ) : ℝ))) = √2 + √6 := by
      rw [hN]
      norm_num
    rw [e] at h
    exact ⟨_, h⟩

example := failureMargin_four (n := Fin 4) (by simp)

end Transformer.Grokking.FlatSpectrum
