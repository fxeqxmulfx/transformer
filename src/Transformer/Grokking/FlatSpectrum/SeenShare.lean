module

public import Transformer.Grokking.FlatSpectrum.DualCertificate
public import Transformer.Grokking.FlatSpectrum.Factorization
public import Transformer.Grokking.FlatSpectrum.CutRange
public import Transformer.Grokking.FlatSpectrum.FlatKernel

/-!
# The share of cells a minimum needs

Source: `review/flat_spectrum.md`, *Rank above one*, item 5.5. A minimum of M has a certificate `Y`:
a contraction vanishing off `Ω` with `tr(Yᴴ K) = ‖K‖_S1` (item 5.1). A contraction has
`‖Y‖²_S2 ≤ min(m, n)` (`IsContraction.frobeniusSq_le_min`), and Cauchy–Schwarz on `Ω` gives
`‖K‖²_S1 ≤ min(m, n) ‖K on Ω‖²_S2` (`nuclearNorm_sq_le_of_isMinOn`). For a table of phases this is
`r_eff / min(m, n) ≤ |Ω| / (m n)` (`share_ge_of_isMinOn`), so a flat table of phases is a minimum
only when every cell is seen (`eq_univ_of_isFlat_of_isMinOn`). Nothing here depends on the code
that made `K`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset
open scoped ComplexOrder

variable {m n : Type*} [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n]

omit [DecidableEq m] in
/-- The identity of size `n` has `‖1‖²_S2 = n`. -/
lemma frobeniusSq_one : frobeniusSq (1 : Matrix n n ℂ) = Fintype.card n := by
  have h (i : n) : ∑ j, ‖(1 : Matrix n n ℂ) i j‖ ^ 2 = 1 := by
    rw [Finset.sum_eq_single i (fun j _ hj => by simp [Matrix.one_apply_ne' hj]) (by simp)]
    simp
  simp [frobeniusSq, h]

/-- A contraction has `‖Y‖²_S2 ≤ min(m, n)`. -/
lemma IsContraction.frobeniusSq_le_min {Y : Matrix m n ℂ} (hY : IsContraction Y) :
    frobeniusSq Y ≤ (min (Fintype.card m) (Fintype.card n) : ℕ) := by
  have hn : frobeniusSq Y ≤ Fintype.card n := by
    simpa [frobeniusSq_one] using hY.frobeniusSq_mul_le (1 : Matrix n n ℂ)
  have hm : frobeniusSq Y ≤ Fintype.card m := by
    have := hY.conjTranspose.frobeniusSq_mul_le (1 : Matrix m m ℂ)
    simpa [frobeniusSq_one, frobeniusSq_conjTranspose] using this
  rw [Nat.cast_min]
  exact le_min hm hn

example : frobeniusSq (1 : Matrix (Fin 2) (Fin 2) ℂ) ≤ (min 2 2 : ℕ) := by
  simpa using (isContraction_one (n := Fin 2)).frobeniusSq_le_min

omit [DecidableEq m] in
/-- With every cell seen, `K` is the only matrix of the constraint set. -/
lemma isMinOn_univ (K : Matrix m n ℂ) :
    IsMinOn nuclearNorm {X | ∀ ij ∈ (Set.univ : Set (m × n)), X ij.1 ij.2 = K ij.1 ij.2} K := by
  refine isMinOn_iff.2 fun X hX => ?_
  have : X = K := by
    ext i j
    exact hX (i, j) trivial
  rw [this]

/-- **A minimum pays for its nuclear norm with seen cells** (`review/flat_spectrum.md`, *Rank above
one*, item 5.5): `‖K‖²_S1 ≤ min(m, n) ‖K on Ω‖²_S2`. -/
theorem nuclearNorm_sq_le_of_isMinOn {K : Matrix m n ℂ} {Ω : Set (m × n)}
    [DecidablePred (· ∈ Ω)]
    (hK : IsMinOn nuclearNorm {X | ∀ ij ∈ Ω, X ij.1 ij.2 = K ij.1 ij.2} K) :
    nuclearNorm K ^ 2 ≤ (min (Fintype.card m) (Fintype.card n) : ℕ) *
      ∑ i, ∑ j, if (i, j) ∈ Ω then ‖K i j‖ ^ 2 else 0 := by
  obtain ⟨Y, hYΩ, hY, hYK⟩ := exists_isContraction_of_isMinOn hK
  set R : Matrix m n ℂ := Matrix.of fun i j => if (i, j) ∈ Ω then K i j else 0 with hR
  have htr : (Yᴴ * K).trace = ∑ i, ∑ j, star (Y i j) * R i j := by
    rw [trace_conjTranspose_mul]
    refine sum_congr rfl fun i _ => sum_congr rfl fun j _ => ?_
    by_cases hij : (i, j) ∈ Ω
    · simp [hR, hij]
    · simp [hYΩ i j hij]
  have hcs := norm_sum_star_mul_le (univ : Finset (m × n)) (fun x => Y x.1 x.2)
    fun x => R x.1 x.2
  simp only [Finset.univ_product_univ.symm, Finset.sum_product] at hcs
  rw [← htr, hYK, Complex.norm_real, Real.norm_of_nonneg (nuclearNorm_nonneg K)] at hcs
  have hY0 : 0 ≤ frobeniusSq Y := sum_nonneg fun _ _ => sum_nonneg fun _ _ => sq_nonneg _
  have hR0 : 0 ≤ frobeniusSq R := sum_nonneg fun _ _ => sum_nonneg fun _ _ => sq_nonneg _
  have hRe : frobeniusSq R = ∑ i, ∑ j, if (i, j) ∈ Ω then ‖K i j‖ ^ 2 else 0 := by
    refine sum_congr rfl fun i _ => sum_congr rfl fun j _ => ?_
    by_cases hij : (i, j) ∈ Ω <;> simp [hR, hij]
  calc nuclearNorm K ^ 2 ≤ (√(frobeniusSq Y) * √(frobeniusSq R)) ^ 2 :=
        pow_le_pow_left₀ (nuclearNorm_nonneg K) hcs 2
    _ = frobeniusSq Y * frobeniusSq R := by
        rw [mul_pow, Real.sq_sqrt hY0, Real.sq_sqrt hR0]
    _ ≤ _ := by
        rw [← hRe]
        exact mul_le_mul_of_nonneg_right hY.frobeniusSq_le_min hR0

example : nuclearNorm (1 : Matrix (Fin 2) (Fin 2) ℂ) ^ 2 ≤ (min (Fintype.card (Fin 2))
    (Fintype.card (Fin 2)) : ℕ) * ∑ i, ∑ j, if (i, j) ∈ (Set.univ : Set (Fin 2 × Fin 2)) then
      ‖(1 : Matrix (Fin 2) (Fin 2) ℂ) i j‖ ^ 2 else 0 :=
  nuclearNorm_sq_le_of_isMinOn (isMinOn_univ _)

/-- **A minimum of a table of phases has `‖K‖²_S1 ≤ min(m, n) |Ω|`**
(`review/flat_spectrum.md`, *Rank above one*, item 5.5). -/
theorem nuclearNorm_sq_le_card_of_isMinOn {K : Matrix m n ℂ} {Ω : Finset (m × n)}
    (hunit : ∀ i j, ‖K i j‖ = 1)
    (hK : IsMinOn nuclearNorm {X | ∀ ij ∈ (Ω : Set (m × n)), X ij.1 ij.2 = K ij.1 ij.2} K) :
    nuclearNorm K ^ 2 ≤ (min (Fintype.card m) (Fintype.card n) : ℕ) * Ω.card := by
  classical
  have h := nuclearNorm_sq_le_of_isMinOn (Ω := (Ω : Set (m × n))) hK
  have hΩ : (∑ i, ∑ j, if (i, j) ∈ (Ω : Set (m × n)) then ‖K i j‖ ^ 2 else 0 : ℝ) = Ω.card := by
    simp only [hunit, one_pow, Finset.mem_coe]
    rw [← Fintype.sum_prod_type' (f := fun i j => if (i, j) ∈ Ω then (1 : ℝ) else 0)]
    simp
  rwa [hΩ] at h

/-- The `1 × 1` identity from its one cell. -/
example : nuclearNorm (1 : Matrix (Fin 1) (Fin 1) ℂ) ^ 2 ≤
    (min (Fintype.card (Fin 1)) (Fintype.card (Fin 1)) : ℕ) *
      (univ : Finset (Fin 1 × Fin 1)).card :=
  nuclearNorm_sq_le_card_of_isMinOn
    (fun i j => by rw [Subsingleton.elim i j, Matrix.one_apply_eq, norm_one])
    (by simpa using isMinOn_univ (1 : Matrix (Fin 1) (Fin 1) ℂ))

/-- **The share of cells a table of phases needs** (`review/flat_spectrum.md`, *Rank above one*,
item 5.5): `r_eff / min(m, n) ≤ |Ω| / (m n)`. -/
theorem share_ge_of_isMinOn [Nonempty m] [Nonempty n] {K : Matrix m n ℂ} {Ω : Finset (m × n)}
    (hunit : ∀ i j, ‖K i j‖ = 1)
    (hK : IsMinOn nuclearNorm {X | ∀ ij ∈ (Ω : Set (m × n)), X ij.1 ij.2 = K ij.1 ij.2} K) :
    effectiveRank K / (min (Fintype.card m) (Fintype.card n) : ℕ) ≤
      (Ω.card : ℝ) / (Fintype.card m * Fintype.card n) := by
  have h := nuclearNorm_sq_le_card_of_isMinOn hunit hK
  have hF : frobeniusSq K = Fintype.card m * Fintype.card n := by
    simp [frobeniusSq, hunit]
  have hm : (0 : ℝ) < Fintype.card m := by exact_mod_cast Fintype.card_pos
  have hn : (0 : ℝ) < Fintype.card n := by exact_mod_cast Fintype.card_pos
  have hmn : (0 : ℝ) < Fintype.card m * Fintype.card n := mul_pos hm hn
  have hmin : (0 : ℝ) < (min (Fintype.card m) (Fintype.card n) : ℕ) := by
    exact_mod_cast lt_min Fintype.card_pos Fintype.card_pos
  rw [effectiveRank, div_pow, Real.sq_sqrt (by rw [hF]; exact hmn.le), hF, div_div,
    div_le_div_iff₀ (mul_pos hmn hmin) hmn]
  have := mul_le_mul_of_nonneg_right h hmn.le
  linarith

/-- The `1 × 1` identity from its one cell. -/
example : effectiveRank (1 : Matrix (Fin 1) (Fin 1) ℂ) /
    (min (Fintype.card (Fin 1)) (Fintype.card (Fin 1)) : ℕ) ≤
      ((univ : Finset (Fin 1 × Fin 1)).card : ℝ) / (Fintype.card (Fin 1) * Fintype.card (Fin 1)) :=
  share_ge_of_isMinOn (fun i j => by rw [Subsingleton.elim i j, Matrix.one_apply_eq, norm_one])
    (by simpa using isMinOn_univ (1 : Matrix (Fin 1) (Fin 1) ℂ))

/-- **A flat table of phases is a minimum only when every cell is seen**
(`review/flat_spectrum.md`, *Rank above one*, item 5.5). -/
theorem eq_univ_of_isFlat_of_isMinOn [Nonempty m] [Nonempty n] {K : Matrix m n ℂ} {c : ℝ}
    (hc : 0 < c) (hflat : IsFlat K c) {Ω : Finset (m × n)} (hunit : ∀ i j, ‖K i j‖ = 1)
    (hK : IsMinOn nuclearNorm {X | ∀ ij ∈ (Ω : Set (m × n)), X ij.1 ij.2 = K ij.1 ij.2} K) :
    Ω = univ := by
  have h := share_ge_of_isMinOn hunit hK
  have hm : (0 : ℝ) < Fintype.card m := by exact_mod_cast Fintype.card_pos
  have hn : (0 : ℝ) < Fintype.card n := by exact_mod_cast Fintype.card_pos
  have hmn : (0 : ℝ) < Fintype.card m * Fintype.card n := mul_pos hm hn
  have hmin : (0 : ℝ) < (min (Fintype.card m) (Fintype.card n) : ℕ) := by
    exact_mod_cast lt_min Fintype.card_pos Fintype.card_pos
  rw [hflat.effectiveRank_eq hc, div_self hmin.ne', le_div_iff₀ hmn, one_mul] at h
  refine Finset.eq_univ_of_card Ω (le_antisymm (Finset.card_le_univ Ω) ?_)
  rw [Fintype.card_prod]
  exact_mod_cast h

/-- The `1 × 1` identity is flat, and its one cell is all of the table. -/
example : (univ : Finset (Fin 1 × Fin 1)) = univ :=
  eq_univ_of_isFlat_of_isMinOn (K := 1) (c := 1) one_pos (Or.inl (by simp))
    (fun i j => by rw [Subsingleton.elim i j, Matrix.one_apply_eq, norm_one])
    (by simpa using isMinOn_univ (1 : Matrix (Fin 1) (Fin 1) ℂ))

end Transformer.Grokking.FlatSpectrum
