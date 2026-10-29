module

public import Transformer.Grokking.FlatSpectrum.GrowthCertificate
public import Transformer.Grokking.FlatSpectrum.ThreeCells

/-!
# Recovery without linear growth

Source: `review/flat_spectrum.md`, *Rank above one* (added), items 6.2–6.4. Linear growth,
`‖K‖_S1 + c ‖H‖_S1 ≤ ‖K + H‖_S1` with `c > 0` for every `H` vanishing on `Ω`, makes `K` the only
minimum (`isMinOn_iff_of_exists_pos_le_nuclearNorm_add`), and by item 6.1
(`exists_pos_le_nuclearNorm_add_iff`) it holds exactly when item 1.2's hypotheses do. At full
column rank every `H` is `(H V) Vᴴ`, and at full row rank it is `U (Hᴴ U)ᴴ`, so one hidden cell
rules linear growth out (`not_exists_pos_le_nuclearNorm_add_of_full_column_rank`,
`not_exists_pos_le_nuclearNorm_add_of_full_row_rank`). From the three cells of a `2 × 2` table
other than `(0, 1)`, M recovers `!![2, 1; 1, 1]`, and `‖K + h E₀₁‖_S1 = √(9 + h²)` for real
`h ≤ 1` (`nuclearNorm_add_smul_single_threeCells`): the growth is quadratic, and no `c > 0` gives
linear growth (`not_exists_pos_le_nuclearNorm_add_threeCells`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset
open scoped ComplexOrder

variable {m n : Type*} [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n]

omit [DecidableEq m] in
/-- **Linear growth makes `K` the only minimum**: if some `c > 0` has
`‖K‖_S1 + c ‖H‖_S1 ≤ ‖K + H‖_S1` for every `H` vanishing on `Ω`, a matrix `X` equal to `K` on
`Ω` is a minimum of the nuclear norm among those matrices exactly when `X = K`. -/
theorem isMinOn_iff_of_exists_pos_le_nuclearNorm_add {K : Matrix m n ℂ} {Ω : Set (m × n)}
    {c : ℝ} (hc : 0 < c) (hgrow : ∀ H : Matrix m n ℂ, (∀ ij ∈ Ω, H ij.1 ij.2 = 0) →
      nuclearNorm K + c * nuclearNorm H ≤ nuclearNorm (K + H))
    {X : Matrix m n ℂ} (hX : ∀ ij ∈ Ω, X ij.1 ij.2 = K ij.1 ij.2) :
    IsMinOn nuclearNorm {X' | ∀ ij ∈ Ω, X' ij.1 ij.2 = K ij.1 ij.2} X ↔ X = K := by
  have hlow (X' : Matrix m n ℂ) (hX' : ∀ ij ∈ Ω, X' ij.1 ij.2 = K ij.1 ij.2) :
      nuclearNorm K + c * nuclearNorm (X' - K) ≤ nuclearNorm X' := by
    have h := hgrow (X' - K) fun ij hij => by rw [Matrix.sub_apply, hX' ij hij, sub_self]
    rwa [add_sub_cancel] at h
  rw [isMinOn_iff]
  constructor
  · intro hmin
    have h1 := (hlow X hX).trans (hmin K fun _ _ => rfl)
    have h0 := nuclearNorm_nonneg (X - K)
    exact sub_eq_zero.mp (eq_zero_of_nuclearNorm_eq_zero (le_antisymm (by nlinarith) h0))
  · intro hXK X' hX'
    rw [hXK]
    linarith [hlow X' hX', mul_nonneg hc.le (nuclearNorm_nonneg (X' - K))]

/-- On `Ω = univ` only `H = 0` vanishes on `Ω`, so every `c` gives linear growth. -/
example := (isMinOn_iff_of_exists_pos_le_nuclearNorm_add (K := (1 : Matrix (Fin 2) (Fin 2) ℂ))
  (Ω := Set.univ) one_pos (fun H hH => by
    obtain rfl : H = 0 := by
      ext i j
      exact hH (i, j) trivial
    simp [nuclearNorm_zero]) (X := 1) fun _ _ => rfl).mpr rfl

omit [Fintype m] [Fintype n] in
/-- The matrix unit at a cell outside `Ω` vanishes on `Ω`. -/
lemma single_apply_of_notMem {Ω : Set (m × n)} {i : m} {j : n} (hij : (i, j) ∉ Ω) :
    ∀ ij ∈ Ω, single i j (1 : ℂ) ij.1 ij.2 = 0 := by
  rintro ⟨i', j'⟩ hij'
  show single i j (1 : ℂ) i' j' = 0
  rw [single_apply_of_ne]
  rintro ⟨rfl, rfl⟩
  exact hij hij'

example := single_apply_of_notMem (Ω := {ij : Fin 2 × Fin 2 | ij ≠ (0, 1)}) (i := 0) (j := 1)
  (by simp)

/-- **At full column rank one hidden cell rules linear growth out**: with `K = U diag(s) Vᴴ`,
`s > 0` and `V` square, if a cell `(i, j)` is not in `Ω`, no `c > 0` has
`‖K‖_S1 + c ‖H‖_S1 ≤ ‖K + H‖_S1` for every `H` vanishing on `Ω`. -/
theorem not_exists_pos_le_nuclearNorm_add_of_full_column_rank {U : Matrix m n ℂ}
    {V : Matrix n n ℂ} (hU : Uᴴ * U = 1) (hV : Vᴴ * V = 1) {s : n → ℝ} (hs : ∀ k, 0 < s k)
    {K : Matrix m n ℂ} (hK : K = U * diagonal (fun k => (s k : ℂ)) * Vᴴ) {Ω : Set (m × n)}
    {i : m} {j : n} (hij : (i, j) ∉ Ω) :
    ¬ ∃ c > 0, ∀ H : Matrix m n ℂ, (∀ ij ∈ Ω, H ij.1 ij.2 = 0) →
      nuclearNorm K + c * nuclearNorm H ≤ nuclearNorm (K + H) := by
  rintro ⟨c, hc, hgrow⟩
  have hE : U * (0 : Matrix n n ℂ)ᴴ + single i j (1 : ℂ) * V * Vᴴ = single i j 1 := by
    rw [conjTranspose_zero, Matrix.mul_zero, zero_add, Matrix.mul_assoc, mul_eq_one_comm.1 hV,
      Matrix.mul_one]
  have h := eq_zero_of_exists_pos_le_nuclearNorm_add hU hV hs hK hc hgrow 0
    (single i j 1 * V) (hE ▸ single_apply_of_notMem hij)
  rw [hE] at h
  simpa using congrFun (congrFun h i) j

example := not_exists_pos_le_nuclearNorm_add_of_full_column_rank (U := 1) (V := 1)
  (s := fun _ : Fin 2 => 1) (K := 1) (Ω := {ij : Fin 2 × Fin 2 | ij ≠ (0, 1)}) (i := 0) (j := 1)
  (by simp) (by simp) (fun _ => one_pos) (by simp) (by simp)

/-- **At full row rank one hidden cell rules linear growth out**: with `K = U diag(s) Vᴴ`,
`s > 0` and `U` square, if a cell `(i, j)` is not in `Ω`, no `c > 0` has
`‖K‖_S1 + c ‖H‖_S1 ≤ ‖K + H‖_S1` for every `H` vanishing on `Ω`. -/
theorem not_exists_pos_le_nuclearNorm_add_of_full_row_rank {U : Matrix m m ℂ}
    {V : Matrix n m ℂ} (hU : Uᴴ * U = 1) (hV : Vᴴ * V = 1) {s : m → ℝ} (hs : ∀ k, 0 < s k)
    {K : Matrix m n ℂ} (hK : K = U * diagonal (fun k => (s k : ℂ)) * Vᴴ) {Ω : Set (m × n)}
    {i : m} {j : n} (hij : (i, j) ∉ Ω) :
    ¬ ∃ c > 0, ∀ H : Matrix m n ℂ, (∀ ij ∈ Ω, H ij.1 ij.2 = 0) →
      nuclearNorm K + c * nuclearNorm H ≤ nuclearNorm (K + H) := by
  rintro ⟨c, hc, hgrow⟩
  have hE : U * ((single i j (1 : ℂ))ᴴ * U)ᴴ + (0 : Matrix m m ℂ) * Vᴴ = single i j 1 := by
    rw [conjTranspose_mul, conjTranspose_conjTranspose, Matrix.zero_mul, add_zero,
      ← Matrix.mul_assoc, mul_eq_one_comm.1 hU, Matrix.one_mul]
  have h := eq_zero_of_exists_pos_le_nuclearNorm_add hU hV hs hK hc hgrow
    ((single i j 1)ᴴ * U) 0 (hE ▸ single_apply_of_notMem hij)
  rw [hE] at h
  simpa using congrFun (congrFun h i) j

example := not_exists_pos_le_nuclearNorm_add_of_full_row_rank (U := 1) (V := 1)
  (s := fun _ : Fin 2 => 1) (K := 1) (Ω := {ij : Fin 2 × Fin 2 | ij ≠ (0, 1)}) (i := 0) (j := 1)
  (by simp) (by simp) (fun _ => one_pos) (by simp) (by simp)

/-- **Quadratic growth at `!![2, 1; 1, 1]`**: `‖K + h E₀₁‖_S1 = √(9 + h²)` for real `h ≤ 1`,
`E₀₁` the matrix unit at `(0, 1)`. -/
theorem nuclearNorm_add_smul_single_threeCells {h : ℝ} (hh : h ≤ 1) :
    nuclearNorm (!![(2 : ℂ), 1; 1, 1] + (h : ℂ) • single 0 1 1) = √(9 + h ^ 2) := by
  obtain ⟨X, hXdef⟩ : ∃ X : Matrix (Fin 2) (Fin 2) ℂ,
      X = !![(2 : ℂ), 1; 1, 1] + (h : ℂ) • single 0 1 1 := ⟨_, rfl⟩
  rw [← hXdef]
  have hX : ∀ ij ∈ {ij : Fin 2 × Fin 2 | ij ≠ (0, 1)},
      X ij.1 ij.2 = !![(2 : ℂ), 1; 1, 1] ij.1 ij.2 :=
    fun ij hij => by
      rw [hXdef, Matrix.add_apply, Matrix.smul_apply,
        single_apply_of_notMem (Ω := {ij : Fin 2 × Fin 2 | ij ≠ (0, 1)}) (i := 0) (j := 1)
          (by simp) ij hij, smul_zero, add_zero]
  have e01 : X 0 1 = ((1 + h : ℝ) : ℂ) := by
    rw [hXdef]
    simp [add_comm]
  have e00 : !![(2 : ℂ), 1; 1, 1] 0 0 = 2 := rfl
  have e10 : !![(2 : ℂ), 1; 1, 1] 1 0 = 1 := rfl
  have e11 : !![(2 : ℂ), 1; 1, 1] 1 1 = 1 := rfl
  have edet : (2 : ℂ) * 1 - ((1 + h : ℝ) : ℂ) * 1 = ((1 - h : ℝ) : ℂ) := by
    push_cast
    ring
  rw [nuclearNorm_threeCells hX, e01, e00, e10, e11, edet, Complex.norm_real, Complex.norm_real,
    Real.norm_of_nonneg (sub_nonneg.mpr hh), Real.norm_eq_abs, sq_abs, norm_one]
  congr 1
  rw [show ‖(2 : ℂ)‖ = 2 by norm_num]
  ring

example := nuclearNorm_add_smul_single_threeCells (h := 1 / 2) (by norm_num)

/-- **Recovered without linear growth**: M recovers `!![2, 1; 1, 1]` from the three cells other
than `(0, 1)` (the `example` after `forall_isMinOn_threeCells_iff`), yet no `c > 0` has
`‖K‖_S1 + c ‖H‖_S1 ≤ ‖K + H‖_S1` for every `H` vanishing on them. -/
theorem not_exists_pos_le_nuclearNorm_add_threeCells :
    ¬ ∃ c > 0, ∀ H : Matrix (Fin 2) (Fin 2) ℂ,
      (∀ ij ∈ {ij : Fin 2 × Fin 2 | ij ≠ (0, 1)}, H ij.1 ij.2 = 0) →
        nuclearNorm !![(2 : ℂ), 1; 1, 1] + c * nuclearNorm H ≤
          nuclearNorm (!![(2 : ℂ), 1; 1, 1] + H) := by
  rintro ⟨c, hc, hgrow⟩
  obtain ⟨t, ht0, ht1, htc⟩ : ∃ t : ℝ, 0 < t ∧ t ≤ 1 ∧ t ≤ c :=
    ⟨min 1 c, lt_min one_pos hc, min_le_left _ _, min_le_right _ _⟩
  have hK : nuclearNorm !![(2 : ℂ), 1; 1, 1] = 3 := by
    have h0 := nuclearNorm_add_smul_single_threeCells (h := 0) zero_le_one
    rwa [Complex.ofReal_zero, zero_smul, add_zero, show (9 : ℝ) + 0 ^ 2 = 3 ^ 2 by norm_num,
      Real.sqrt_sq (by norm_num)] at h0
  have hH : t ≤ nuclearNorm ((t : ℂ) • single (0 : Fin 2) (1 : Fin 2) (1 : ℂ)) := by
    have h1 := norm_apply_le_nuclearNorm ((t : ℂ) • single (0 : Fin 2) (1 : Fin 2) (1 : ℂ)) 0 1
    rwa [Matrix.smul_apply, single_apply_same, smul_eq_mul, mul_one, Complex.norm_real,
      Real.norm_of_nonneg ht0.le] at h1
  have hg := hgrow ((t : ℂ) • single 0 1 1) fun ij hij => by
    rw [Matrix.smul_apply,
      single_apply_of_notMem (Ω := {ij : Fin 2 × Fin 2 | ij ≠ (0, 1)}) (i := 0) (j := 1)
        (by simp) ij hij, smul_zero]
  rw [hK, nuclearNorm_add_smul_single_threeCells ht1] at hg
  have h3 : 3 + c * t ≤ √(9 + t ^ 2) := by linarith [mul_le_mul_of_nonneg_left hH hc.le]
  have h2 := (Real.le_sqrt (by positivity) (by positivity)).mp h3
  nlinarith [mul_pos hc ht0, mul_le_mul_of_nonneg_left htc ht0.le]

end Transformer.Grokking.FlatSpectrum
