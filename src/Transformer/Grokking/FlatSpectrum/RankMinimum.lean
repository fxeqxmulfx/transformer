module

public import Transformer.Grokking.FlatSpectrum.DualCertificate
public import Transformer.Grokking.FlatSpectrum.RankCertificate

/-!
# Theorem A at every rank

Source: `review/flat_spectrum.md`, *Rank above one* (added), items 5.2 to 5.4. Let
`K = U diag(s) Vᴴ` with orthonormal columns in `U` and `V` and `s > 0`. A contraction `Y` has
`tr(Yᴴ K) = ‖K‖_S1` exactly when `Y = U Vᴴ + W` with `Uᴴ W = 0`, `W V = 0` and `W` a contraction
(`isContraction_trace_eq_iff`). If: `U Vᴴ + W` is a contraction
(`isContraction_mul_conjTranspose_add`) and `tr(Wᴴ K) = 0`. Only if: for `0 < t ≤ min s`,
`K − t U Vᴴ = U diag(s − t) Vᴴ` has norm `‖K‖_S1 − t r`, so `Re tr(Yᴴ U Vᴴ) ≥ r`, and
`‖U Vᴴ‖_S1 = r` gives equality. Then `‖Y V − U‖²_S2 = tr(Vᴴ Yᴴ Y V) − 2 Re tr(Uᴴ Y V) + r ≤ 0`,
so `Y V = U` (`mul_eq_of_isContraction`), and likewise `Yᴴ U = V`, as `Yᴴ` is a contraction.
With `W = Y − U Vᴴ`, `1 − Wᴴ W = (1 − Yᴴ Y) + V Vᴴ`. With `isMinOn_iff_exists_isContraction`:
`K` is a minimum of M exactly when such a `W` makes `U Vᴴ + W` vanish off `Ω`
(`isMinOn_iff_exists_certificate`). If `V` is square, `W V = 0` forces `W = 0`, and the condition
is that `U Vᴴ` vanishes off `Ω` (`isMinOn_iff_of_full_rank`). For invertible `K`, `U Vᴴ` is the
polar factor of Theorem A (`isMinOn_nuclearNorm_iff`); that identification is a remark, not
proved here.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset
open scoped ComplexOrder

variable {m n r : Type*} [Fintype m] [Fintype n] [Fintype r] [DecidableEq m] [DecidableEq n]
  [DecidableEq r]

omit [DecidableEq m] in
/-- A contraction `Y` with `Re tr(Uᴴ Y V) = r` maps `V` to `U`, for `U`, `V` with `r`
orthonormal columns: `‖Y V − U‖²_S2 = tr(Vᴴ Yᴴ Y V) − 2 Re tr(Uᴴ Y V) + r ≤ 0`. -/
lemma mul_eq_of_isContraction {Y : Matrix m n ℂ} (hY : IsContraction Y) {U : Matrix m r ℂ}
    {V : Matrix n r ℂ} (hU : Uᴴ * U = 1) (hV : Vᴴ * V = 1)
    (h : (Uᴴ * (Y * V)).trace.re = Fintype.card r) : Y * V = U := by
  have h0 := Complex.nonneg_iff.1 (posSemidef_conjTranspose_mul_self (Y * V - U)).trace_nonneg
  have h1 := (Complex.nonneg_iff.1 (hY.conjTranspose_mul_mul_same V).trace_nonneg).1
  have hstar : (Vᴴ * (Yᴴ * U)).trace = star (Uᴴ * (Y * V)).trace := by
    rw [← trace_conjTranspose, conjTranspose_mul, conjTranspose_mul, conjTranspose_conjTranspose,
      Matrix.mul_assoc]
  have e : ((Y * V - U)ᴴ * (Y * V - U)).trace = (Fintype.card r : ℂ) + Fintype.card r -
      (Vᴴ * (1 - Yᴴ * Y) * V).trace - (Vᴴ * (Yᴴ * U)).trace - (Uᴴ * (Y * V)).trace := by
    simp only [conjTranspose_sub, conjTranspose_mul, Matrix.sub_mul, Matrix.mul_sub,
      Matrix.mul_one, Matrix.mul_assoc, hU, hV, trace_sub, trace_one]
    ring
  have hre := congrArg Complex.re e
  rw [hstar] at hre
  simp only [Complex.sub_re, Complex.add_re, Complex.natCast_re, Complex.star_def,
    Complex.conj_re, h] at hre
  have htr : ((Y * V - U)ᴴ * (Y * V - U)).trace = 0 :=
    Complex.ext (by rw [Complex.zero_re]; linarith) (by rw [← h0.2, Complex.zero_im])
  exact sub_eq_zero.1 (trace_conjTranspose_mul_self_eq_zero_iff.1 htr)

example := mul_eq_of_isContraction (isContraction_one (n := Fin 2))
  (U := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (V := 1) (by simp) (by simp) (by simp)

/-- **The contractions attaining the nuclear norm**, the subdifferential of `‖·‖_S1` at
`K = U diag(s) Vᴴ`, `s > 0`: a contraction `Y` has `tr(Yᴴ K) = ‖K‖_S1` exactly when
`Y = U Vᴴ + W` with `Uᴴ W = 0`, `W V = 0` and `W` a contraction. -/
theorem isContraction_trace_eq_iff {U : Matrix m r ℂ} {V : Matrix n r ℂ} (hU : Uᴴ * U = 1)
    (hV : Vᴴ * V = 1) {s : r → ℝ} (hs : ∀ k, 0 < s k) {K : Matrix m n ℂ}
    (hK : K = U * diagonal (fun k => (s k : ℂ)) * Vᴴ) {Y : Matrix m n ℂ} :
    IsContraction Y ∧ (Yᴴ * K).trace = nuclearNorm K ↔
      ∃ W : Matrix m n ℂ, Y = U * Vᴴ + W ∧ Uᴴ * W = 0 ∧ W * V = 0 ∧ IsContraction W := by
  obtain ⟨hKU, hKtr, hKn⟩ := svd_hypotheses hU hV fun k => (hs k).le
  rw [← hK] at hKU hKtr hKn
  constructor
  · rintro ⟨hY, hYK⟩
    -- `Re tr(Yᴴ U Vᴴ) = r`, from `K − t U Vᴴ = U diag(s − t) Vᴴ` for `0 < t ≤ s`
    obtain ⟨t, ht, hts⟩ : ∃ t : ℝ, 0 < t ∧ ∀ k, t ≤ s k := by
      cases isEmpty_or_nonempty r
      · exact ⟨1, one_pos, fun k => isEmptyElim k⟩
      · obtain ⟨k₀, hk₀⟩ := Finite.exists_min s
        exact ⟨s k₀, hs k₀, hk₀⟩
    have hn1 := (svd_hypotheses hU hV (s := fun _ => 1) fun _ => zero_le_one).2.2
    have hnt := (svd_hypotheses hU hV (s := fun k => s k - t) fun k => sub_nonneg.2 (hts k)).2.2
    have hUV : U * diagonal (fun _ => ((1 : ℝ) : ℂ)) * Vᴴ = U * Vᴴ := by
      rw [Complex.ofReal_one, diagonal_one, Matrix.mul_one]
    have hKt : U * diagonal (fun k => ((s k - t : ℝ) : ℂ)) * Vᴴ = K - (t : ℂ) • (U * Vᴴ) := by
      have hd : diagonal (fun k => ((s k - t : ℝ) : ℂ)) =
          diagonal (fun k => (s k : ℂ)) - (t : ℂ) • 1 := by
        rw [← diagonal_one, ← diagonal_smul, diagonal_sub]
        congr 1
        funext k
        simp
      rw [hK, hd, Matrix.mul_sub, Matrix.sub_mul, Matrix.mul_smul, Matrix.smul_mul,
        Matrix.mul_one]
    rw [hUV, sum_const, card_univ, nsmul_eq_mul, mul_one] at hn1
    rw [hKt, sum_sub_distrib, sum_const, card_univ, nsmul_eq_mul, ← hKn] at hnt
    have e1 := (Complex.re_le_norm _).trans (norm_trace_le_nuclearNorm hY (U * Vᴴ))
    have e2 := (Complex.re_le_norm _).trans
      (norm_trace_le_nuclearNorm hY (K - (t : ℂ) • (U * Vᴴ)))
    rw [hn1] at e1
    rw [hnt, Matrix.mul_sub, Matrix.mul_smul, trace_sub, trace_smul, smul_eq_mul,
      Complex.sub_re, Complex.re_ofReal_mul, hYK, Complex.ofReal_re] at e2
    have hD : (Yᴴ * (U * Vᴴ)).trace.re = Fintype.card r :=
      le_antisymm e1 (le_of_mul_le_mul_left (by linarith) ht)
    have hc1 : (Uᴴ * (Y * V)).trace.re = Fintype.card r := by
      have h1 : (Yᴴ * (U * Vᴴ)).trace = star (Uᴴ * (Y * V)).trace := by
        rw [← trace_conjTranspose, conjTranspose_mul, conjTranspose_mul,
          conjTranspose_conjTranspose, ← Matrix.mul_assoc, Matrix.trace_mul_comm (Yᴴ * U),
          Matrix.mul_assoc]
      rw [← hD, h1, Complex.star_def, Complex.conj_re]
    have hYV : Y * V = U := mul_eq_of_isContraction hY hU hV hc1
    have hYU : Yᴴ * U = V := mul_eq_of_isContraction hY.conjTranspose hV hU (by
      rw [show Vᴴ * (Yᴴ * U) = (Uᴴ * (Y * V))ᴴ by
          simp only [conjTranspose_mul, conjTranspose_conjTranspose, Matrix.mul_assoc],
        trace_conjTranspose, Complex.star_def, Complex.conj_re, hc1])
    have hUY : Uᴴ * Y = Vᴴ := by
      simpa only [conjTranspose_mul, conjTranspose_conjTranspose] using congrArg conjTranspose hYU
    refine ⟨Y - U * Vᴴ, by abel, ?_, ?_, ?_⟩
    · rw [Matrix.mul_sub, hUY, ← Matrix.mul_assoc, hU, Matrix.one_mul, sub_self]
    · rw [Matrix.sub_mul, hYV, Matrix.mul_assoc, hV, Matrix.mul_one, sub_self]
    · -- `1 − Wᴴ W = (1 − Yᴴ Y) + V Vᴴ`
      have h1 : Yᴴ * (U * Vᴴ) = V * Vᴴ := by rw [← Matrix.mul_assoc, hYU]
      have h2 : Uᴴ * (U * Vᴴ) = Vᴴ := by rw [← Matrix.mul_assoc, hU, Matrix.one_mul]
      have e : 1 - (Y - U * Vᴴ)ᴴ * (Y - U * Vᴴ) = (1 - Yᴴ * Y) + V * Vᴴ := by
        simp only [conjTranspose_sub, conjTranspose_mul, conjTranspose_conjTranspose,
          Matrix.sub_mul, Matrix.mul_sub, Matrix.mul_assoc, h1, h2, hUY]
        abel
      rw [IsContraction, e]
      exact hY.add (posSemidef_self_mul_conjTranspose V)
  · rintro ⟨W, rfl, hWU, hWV, hW⟩
    have hWP : (1 - U * Uᴴ) * W * (1 - V * Vᴴ) = W := by
      have e1 : U * Uᴴ * W = 0 := by rw [Matrix.mul_assoc, hWU, Matrix.mul_zero]
      have e2 : W * (V * Vᴴ) = 0 := by rw [← Matrix.mul_assoc, hWV, Matrix.zero_mul]
      rw [Matrix.sub_mul, Matrix.one_mul, e1, sub_zero, Matrix.mul_sub, Matrix.mul_one, e2,
        sub_zero]
    have hWU' : Wᴴ * U = 0 := by
      simpa only [conjTranspose_mul, conjTranspose_conjTranspose, conjTranspose_zero] using
        congrArg conjTranspose hWU
    have hWK : Wᴴ * K = 0 := by
      rw [← hKU, Matrix.mul_assoc U, ← Matrix.mul_assoc Wᴴ, hWU', Matrix.zero_mul]
    refine ⟨?_, ?_⟩
    · have h := isContraction_mul_conjTranspose_add hU hV hW
      rwa [hWP] at h
    · rw [conjTranspose_add, Matrix.add_mul, trace_add, hKtr, hWK, trace_zero, add_zero]

example := (isContraction_trace_eq_iff (U := (1 : Matrix (Fin 2) (Fin 2) ℂ))
  (V := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (by simp) (by simp) (s := ![1, 2])
  (fun k => by fin_cases k <;> norm_num) rfl (Y := 1)).2
  ⟨0, by simp, by simp, by simp, isContraction_zero⟩

/-- **Theorem A at every rank**: `K = U diag(s) Vᴴ`, `s > 0`, is a minimum of the nuclear norm
among the matrices equal to it on `Ω` exactly when some contraction `W` with `Uᴴ W = 0` and
`W V = 0` makes `U Vᴴ + W` vanish off `Ω`. -/
theorem isMinOn_iff_exists_certificate {U : Matrix m r ℂ} {V : Matrix n r ℂ} (hU : Uᴴ * U = 1)
    (hV : Vᴴ * V = 1) {s : r → ℝ} (hs : ∀ k, 0 < s k) {K : Matrix m n ℂ}
    (hK : K = U * diagonal (fun k => (s k : ℂ)) * Vᴴ) {Ω : Set (m × n)} :
    IsMinOn nuclearNorm {X | ∀ ij ∈ Ω, X ij.1 ij.2 = K ij.1 ij.2} K ↔
      ∃ W : Matrix m n ℂ, Uᴴ * W = 0 ∧ W * V = 0 ∧ IsContraction W ∧
        ∀ i j, (i, j) ∉ Ω → (U * Vᴴ + W) i j = 0 := by
  rw [isMinOn_iff_exists_isContraction]
  constructor
  · rintro ⟨Y, hYΩ, hY⟩
    obtain ⟨W, rfl, hWU, hWV, hW⟩ := (isContraction_trace_eq_iff hU hV hs hK).1 hY
    exact ⟨W, hWU, hWV, hW, hYΩ⟩
  · rintro ⟨W, hWU, hWV, hW, hΩ⟩
    exact ⟨U * Vᴴ + W, hΩ, (isContraction_trace_eq_iff hU hV hs hK).2 ⟨W, rfl, hWU, hWV, hW⟩⟩

/-- `diag(1, 2)` is a minimum of M from its diagonal cells, with `W = 0`. -/
example := (isMinOn_iff_exists_certificate (U := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (V := 1)
  (by simp) (by simp) (s := ![1, 2]) (fun k => by fin_cases k <;> norm_num) rfl
  (Ω := {ij | ij.1 = ij.2})).2
  ⟨0, by simp, by simp, isContraction_zero, fun i j hij => by simpa [one_apply] using hij⟩

omit [Fintype r] [DecidableEq r] in
/-- **Theorem A at full column rank**: if `K = U diag(s) Vᴴ` with `s > 0` and `V` square, `K` is a
minimum of the nuclear norm among the matrices equal to it on `Ω` exactly when `U Vᴴ` vanishes
off `Ω`: `W V = 0` forces `W = 0`. -/
theorem isMinOn_iff_of_full_rank {U : Matrix m n ℂ} {V : Matrix n n ℂ} (hU : Uᴴ * U = 1)
    (hV : Vᴴ * V = 1) {s : n → ℝ} (hs : ∀ k, 0 < s k) {K : Matrix m n ℂ}
    (hK : K = U * diagonal (fun k => (s k : ℂ)) * Vᴴ) {Ω : Set (m × n)} :
    IsMinOn nuclearNorm {X | ∀ ij ∈ Ω, X ij.1 ij.2 = K ij.1 ij.2} K ↔
      ∀ i j, (i, j) ∉ Ω → (U * Vᴴ) i j = 0 := by
  rw [isMinOn_iff_exists_certificate hU hV hs hK]
  have hV' : V * Vᴴ = 1 := mul_eq_one_comm.1 hV
  constructor
  · rintro ⟨W, -, hWV, -, hΩ⟩ i j hij
    have hW : W = 0 := by
      rw [← Matrix.mul_one W, ← hV', ← Matrix.mul_assoc, hWV, Matrix.zero_mul]
    simpa [hW] using hΩ i j hij
  · intro hΩ
    exact ⟨0, Matrix.mul_zero _, Matrix.zero_mul _, isContraction_zero,
      fun i j hij => by simpa using hΩ i j hij⟩

example := (isMinOn_iff_of_full_rank (U := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (V := 1)
  (by simp) (by simp) (s := ![1, 2]) (fun k => by fin_cases k <;> norm_num) rfl
  (Ω := {ij | ij.1 = ij.2})).2 fun i j hij => by simpa [one_apply] using hij

end Transformer.Grokking.FlatSpectrum
