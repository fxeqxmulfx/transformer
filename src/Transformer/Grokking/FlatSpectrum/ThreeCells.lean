module

public import Transformer.Grokking.FlatSpectrum.Blocks

/-!
# A `2 × 2` table from three cells

Source: `review/flat_spectrum.md`, *Rank above one* (added), items 3 and 4. M sees the cells
`(0, 0)`, `(1, 0)` and `(1, 1)` of a `2 × 2` table `K` and fills `(0, 1)`. Since
`‖X‖²_S1 = ‖X‖²_S2 + 2 |det X|` (`nuclearNorm_fin_two`), the fill `x` minimizes
`|x|² + 2 |c − x b|`, with `b = k₁₀` and `c = k₀₀ k₁₁` (`nuclearNorm_threeCells`). At
`x* = conj(b) c / max(|b|², |c|)` (`threeCellFill`; `x* = 0` when `b = c = 0`) the number
`g = c / max(|b|², |c|)` has `|g| ≤ 1`, `Re(conj(g) (c − x* b)) = |c − x* b|` and `x* = g conj(b)`,
so the cost exceeds its value at `x*` by at least `|x − x*|²` (`sq_norm_threeCellFill_add_le`). So M
has exactly one minimum, the table with `x*` at `(0, 1)` (`isMinOn_threeCells_iff`), and recovers
`K` exactly when `k₀₁ = x*` (`forall_isMinOn_threeCells_iff`). `ThreeCellsRank` reads this by the
rank of `K`. At rank two the condition of item 1.2 fails on these cells: every `U` with `U Uᴴ K = K`
has `U Uᴴ = 1`, so the matrix unit at `(0, 1)` is a nonzero `U Aᴴ + B Vᴴ` vanishing on them
(`exists_ne_zero_threeCells_of_det_ne_zero`); yet M recovers `!![2, 1; 1, 1]`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix

/-- M's fill of the cell `(0, 1)` of a `2 × 2` table from `b = k₁₀` and `c = k₀₀ k₁₁`
(`isMinOn_threeCells_iff`): `conj(b) c / max(|b|², |c|)`, and `0` when `b = c = 0`. -/
noncomputable def threeCellFill (b c : ℂ) : ℂ :=
  starRingEnd ℂ b * c / ((max (‖b‖ ^ 2) ‖c‖ : ℝ) : ℂ)

/-- `c − x* b = c (1 − |b|² / max(|b|², |c|))` at M's fill `x*`. -/
lemma sub_threeCellFill_mul (b c : ℂ) :
    c - threeCellFill b c * b = c * ((1 - ‖b‖ ^ 2 / max (‖b‖ ^ 2) ‖c‖ : ℝ) : ℂ) := by
  rw [threeCellFill, div_mul_eq_mul_div, mul_right_comm, Complex.conj_mul']
  push_cast
  ring

/-- If `|g| ≤ 1`, `Re(conj(g) (c − y b)) = |c − y b|` and `y = g conj(b)`, then
`|x|² + 2 |c − x b|` exceeds its value at `y` by at least `|x − y|²`. -/
lemma sq_norm_add_le_of_subgradient {b c y g : ℂ} (hg : ‖g‖ ≤ 1)
    (hgz : (starRingEnd ℂ g * (c - y * b)).re = ‖c - y * b‖) (hy : y = g * starRingEnd ℂ b)
    (x : ℂ) : ‖y‖ ^ 2 + 2 * ‖c - y * b‖ + ‖x - y‖ ^ 2 ≤ ‖x‖ ^ 2 + 2 * ‖c - x * b‖ := by
  have h1 : (starRingEnd ℂ g * (c - x * b)).re ≤ ‖c - x * b‖ :=
    calc _ ≤ ‖starRingEnd ℂ g * (c - x * b)‖ := Complex.re_le_norm _
      _ = ‖g‖ * ‖c - x * b‖ := by rw [norm_mul, Complex.norm_conj]
      _ ≤ ‖c - x * b‖ := mul_le_of_le_one_left (norm_nonneg _) hg
  have h2 : (starRingEnd ℂ g * (c - x * b)).re =
      ‖c - y * b‖ - (starRingEnd ℂ g * b * (x - y)).re := by
    rw [← hgz, ← Complex.sub_re]
    congr 1
    ring
  have hc : y * starRingEnd ℂ (x - y) = starRingEnd ℂ (starRingEnd ℂ g * b * (x - y)) := by
    rw [hy]
    simp only [map_mul, Complex.conj_conj]
  have h3 : ‖x‖ ^ 2 = ‖y‖ ^ 2 + ‖x - y‖ ^ 2 + 2 * (starRingEnd ℂ g * b * (x - y)).re := by
    rw [Complex.sq_norm, Complex.sq_norm, Complex.sq_norm, ← Complex.conj_re, ← hc,
      ← Complex.normSq_add, add_sub_cancel]
  linarith

example := sq_norm_add_le_of_subgradient (b := 1) (c := 2) (y := 1) (g := 1) (by simp)
  (by norm_num) (by simp) 0

/-- `|x|² + 2 |c − x b|` exceeds its value at `x* = conj(b) c / max(|b|², |c|)` by at least
`|x − x*|²`. -/
theorem sq_norm_threeCellFill_add_le (b c x : ℂ) :
    ‖threeCellFill b c‖ ^ 2 + 2 * ‖c - threeCellFill b c * b‖ + ‖x - threeCellFill b c‖ ^ 2 ≤
      ‖x‖ ^ 2 + 2 * ‖c - x * b‖ := by
  obtain ⟨μ, hμ⟩ : ∃ μ : ℝ, max (‖b‖ ^ 2) ‖c‖ = μ := ⟨_, rfl⟩
  have hbμ : ‖b‖ ^ 2 ≤ μ := hμ ▸ le_max_left _ _
  have hcμ : ‖c‖ ≤ μ := hμ ▸ le_max_right _ _
  have hch : μ = ‖b‖ ^ 2 ∨ μ = ‖c‖ := hμ ▸ max_choice _ _
  have hμ0 : 0 ≤ μ := (norm_nonneg c).trans hcμ
  have hy : threeCellFill b c = c / (μ : ℂ) * starRingEnd ℂ b := by
    rw [threeCellFill, hμ]
    ring
  have hl : 0 ≤ 1 - ‖b‖ ^ 2 / μ := sub_nonneg.mpr (div_le_one_of_le₀ hbμ hμ0)
  have hyb := sub_threeCellFill_mul b c
  rw [hμ] at hyb
  refine sq_norm_add_le_of_subgradient (g := c / (μ : ℂ)) ?_ ?_ hy x
  · rw [norm_div, Complex.norm_of_nonneg hμ0]
    exact div_le_one_of_le₀ hcμ hμ0
  · have hz : starRingEnd ℂ (c / (μ : ℂ)) * (c * ((1 - ‖b‖ ^ 2 / μ : ℝ) : ℂ)) =
        ((‖c‖ ^ 2 / μ * (1 - ‖b‖ ^ 2 / μ) : ℝ) : ℂ) := by
      rw [map_div₀, Complex.conj_ofReal, div_mul_eq_mul_div, ← mul_assoc, Complex.conj_mul']
      push_cast
      ring
    rw [hyb, hz, Complex.ofReal_re, norm_mul, Complex.norm_of_nonneg hl]
    rcases eq_or_ne μ 0 with h0 | h0
    · have hc : ‖c‖ = 0 := le_antisymm (h0 ▸ hcμ) (norm_nonneg _)
      rw [hc, h0]
      simp
    · rcases hch with h | h
      · rw [← h, div_self h0, sub_self, mul_zero, mul_zero]
      · rw [h, sq ‖c‖, mul_div_assoc, div_self (h ▸ h0), mul_one]

/-- On the matrices equal to `K` off `(0, 1)`, with `x = X 0 1`,
`‖X‖_S1 = √(|k₀₀|² + |k₁₀|² + |k₁₁|² + |x|² + 2 |k₀₀ k₁₁ − x k₁₀|)` (`nuclearNorm_fin_two`). -/
lemma nuclearNorm_threeCells {K X : Matrix (Fin 2) (Fin 2) ℂ}
    (hX : ∀ ij ∈ {ij : Fin 2 × Fin 2 | ij ≠ (0, 1)}, X ij.1 ij.2 = K ij.1 ij.2) :
    nuclearNorm X = √(‖K 0 0‖ ^ 2 + ‖K 1 0‖ ^ 2 + ‖K 1 1‖ ^ 2 +
      (‖X 0 1‖ ^ 2 + 2 * ‖K 0 0 * K 1 1 - X 0 1 * K 1 0‖)) := by
  have h00 : X 0 0 = K 0 0 := hX (0, 0) (by simp)
  have h10 : X 1 0 = K 1 0 := hX (1, 0) (by simp)
  have h11 : X 1 1 = K 1 1 := hX (1, 1) (by simp)
  rw [nuclearNorm_fin_two, frobeniusSq, Fin.sum_univ_two, Fin.sum_univ_two, Fin.sum_univ_two,
    det_fin_two, h00, h10, h11]
  congr 1
  ring

example := nuclearNorm_threeCells (K := 1) (X := 1) fun _ _ => rfl

/-- The table `K` with `x` in place of `k₀₁` equals `K` off `(0, 1)`. -/
lemma threeCells_completion (K : Matrix (Fin 2) (Fin 2) ℂ) (x : ℂ) :
    ∀ ij ∈ {ij : Fin 2 × Fin 2 | ij ≠ (0, 1)},
      (!![K 0 0, x; K 1 0, K 1 1] : Matrix (Fin 2) (Fin 2) ℂ) ij.1 ij.2 = K ij.1 ij.2 := by
  rintro ⟨i, j⟩ hij
  fin_cases i <;> fin_cases j <;> simp_all

/-- **M on a `2 × 2` table from three cells**: among the matrices equal to `K` off `(0, 1)`,
the nuclear norm is least exactly at the one whose entry `(0, 1)` is
`conj(k₁₀) k₀₀ k₁₁ / max(|k₁₀|², |k₀₀ k₁₁|)`, read as `0` when `k₁₀ = k₀₀ k₁₁ = 0`. -/
theorem isMinOn_threeCells_iff {K X : Matrix (Fin 2) (Fin 2) ℂ}
    (hX : ∀ ij ∈ {ij : Fin 2 × Fin 2 | ij ≠ (0, 1)}, X ij.1 ij.2 = K ij.1 ij.2) :
    IsMinOn nuclearNorm
        {X' | ∀ ij ∈ {ij : Fin 2 × Fin 2 | ij ≠ (0, 1)}, X' ij.1 ij.2 = K ij.1 ij.2} X ↔
      X 0 1 = threeCellFill (K 1 0) (K 0 0 * K 1 1) := by
  have hcost := sq_norm_threeCellFill_add_le (K 1 0) (K 0 0 * K 1 1)
  set y := threeCellFill (K 1 0) (K 0 0 * K 1 1)
  have hy : (!![K 0 0, y; K 1 0, K 1 1] : Matrix (Fin 2) (Fin 2) ℂ) 0 1 = y := by simp
  rw [isMinOn_iff]
  constructor
  · intro hmin
    have h := hmin _ (threeCells_completion K y)
    rw [nuclearNorm_threeCells hX, nuclearNorm_threeCells (threeCells_completion K y), hy,
      Real.sqrt_le_sqrt_iff (by positivity)] at h
    have hsq : ‖X 0 1 - y‖ ^ 2 = 0 := le_antisymm (by linarith [hcost (X 0 1)]) (sq_nonneg _)
    exact sub_eq_zero.mp (norm_eq_zero.mp ((pow_eq_zero_iff two_ne_zero).mp hsq))
  · intro hXy X' hX'
    rw [nuclearNorm_threeCells hX, nuclearNorm_threeCells hX', hXy]
    exact Real.sqrt_le_sqrt (by linarith [hcost (X' 0 1), sq_nonneg ‖X' 0 1 - y‖])

/-- M's unique minimum from the three cells of `!![2, 1; 1, −1]` is `!![2, −1; 1, −1]`. -/
example := (isMinOn_threeCells_iff (K := !![(2 : ℂ), 1; 1, -1]) (X := !![2, -1; 1, -1])
  (by rintro ⟨i, j⟩ hij; fin_cases i <;> fin_cases j <;> simp_all)).mpr
  (by norm_num [threeCellFill])

/-- **When M recovers a `2 × 2` table from three cells**: from the cells other than `(0, 1)`,
M recovers `K` exactly when `k₀₁ = conj(k₁₀) k₀₀ k₁₁ / max(|k₁₀|², |k₀₀ k₁₁|)`, read as `0`
when `k₁₀ = k₀₀ k₁₁ = 0`. -/
theorem forall_isMinOn_threeCells_iff {K : Matrix (Fin 2) (Fin 2) ℂ} :
    (∀ X ∈ {X' : Matrix (Fin 2) (Fin 2) ℂ |
        ∀ ij ∈ {ij : Fin 2 × Fin 2 | ij ≠ (0, 1)}, X' ij.1 ij.2 = K ij.1 ij.2},
      (IsMinOn nuclearNorm
        {X' | ∀ ij ∈ {ij : Fin 2 × Fin 2 | ij ≠ (0, 1)}, X' ij.1 ij.2 = K ij.1 ij.2} X ↔
        X = K)) ↔ K 0 1 = threeCellFill (K 1 0) (K 0 0 * K 1 1) := by
  constructor
  · intro h
    have hc := threeCells_completion K (threeCellFill (K 1 0) (K 0 0 * K 1 1))
    have e := congrFun (congrFun ((h _ hc).mp ((isMinOn_threeCells_iff hc).mpr (by simp))) 0) 1
    simpa using e.symm
  · intro hK X hX
    rw [isMinOn_threeCells_iff hX, ← hK]
    refine ⟨fun h => ?_, fun h => h ▸ rfl⟩
    ext i j
    fin_cases i <;> fin_cases j
    exacts [hX (0, 0) (by simp), h, hX (1, 0) (by simp), hX (1, 1) (by simp)]

/-- M recovers `!![2, 1; 1, 1]` from its three cells other than `(0, 1)`. -/
example := (forall_isMinOn_threeCells_iff (K := !![(2 : ℂ), 1; 1, 1])).mpr
  (by norm_num [threeCellFill])

/-- **Item 1.2's condition fails at rank two**: if `det K ≠ 0`, every `U` with `U Uᴴ K = K`, as
item 1.2 asks, has `U Uᴴ = 1`, so `U (Uᴴ E)`, with `E` the matrix unit at `(0, 1)`, is a nonzero
`U Aᴴ + B Vᴴ` vanishing on the three cells other than `(0, 1)`. Yet M recovers `!![2, 1; 1, 1]`
from them (the example above): the condition is sufficient, not necessary. -/
theorem exists_ne_zero_threeCells_of_det_ne_zero {r : Type*} [Fintype r]
    {K : Matrix (Fin 2) (Fin 2) ℂ} (hK : K.det ≠ 0) {U V : Matrix (Fin 2) r ℂ}
    (hKU : U * Uᴴ * K = K) :
    ∃ A B : Matrix (Fin 2) r ℂ, (∀ ij ∈ {ij : Fin 2 × Fin 2 | ij ≠ (0, 1)},
      (U * Aᴴ + B * Vᴴ) ij.1 ij.2 = 0) ∧ U * Aᴴ + B * Vᴴ ≠ 0 := by
  have hUU : U * Uᴴ = 1 := by
    rw [← Matrix.mul_one (U * Uᴴ), ← K.mul_nonsing_inv hK.isUnit, ← Matrix.mul_assoc, hKU]
  refine ⟨(!![0, 1; 0, 0] : Matrix (Fin 2) (Fin 2) ℂ)ᴴ * U, 0, ?_, ?_⟩ <;>
    rw [conjTranspose_mul, conjTranspose_conjTranspose, ← Matrix.mul_assoc, hUU,
      Matrix.one_mul, Matrix.zero_mul, add_zero]
  · rintro ⟨i, j⟩ hij
    fin_cases i <;> fin_cases j <;> simp_all
  · intro h
    have h01 := congrFun (congrFun h 0) 1
    simp at h01

/-- `!![2, 1; 1, 1]`, recovered from the three cells, with `U = V = 1`. -/
example := exists_ne_zero_threeCells_of_det_ne_zero (K := !![(2 : ℂ), 1; 1, 1])
  (U := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (V := 1) (by norm_num) (by simp)

end Transformer.Grokking.FlatSpectrum
