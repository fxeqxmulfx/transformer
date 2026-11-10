module

public import Transformer.Grokking.FlatSpectrum.OneCell
public import Transformer.Grokking.FlatSpectrum.OnesCertificate
public import Transformer.Grokking.FlatSpectrum.ProjectionKink

/-!
# An unseen row is zero in every minimum

Source: `review/flat_spectrum.md`, *The wrap of a step is a comparison* (added), item CW8.
Let `P` be an orthogonal projection on the rows. For contractions `Z`, `Z'` and real `c`, `s`
with `c² + s² ≤ 1` the matrix `c P Z + s (1 − P) Z'` is a contraction
(`IsContraction.smul_proj_add`), so the two blocks of rows of `X` are bounded in `ℓ²`
(`nuclearNorm_proj_sq_add_sq_le`): `‖P X‖_S1² + ‖(1 − P) X‖_S1² ≤ ‖X‖_S1²`. Hence removing a
block of rows that is not zero lowers the nuclear norm (`nuclearNorm_proj_lt`).

A row with no cell in `Ω` can be removed without leaving the matrices equal to `K` on `Ω`, so
it is zero in every minimum of the nuclear norm among them (`row_eq_zero_of_isMinOn`), and so
is a column with no cell in `Ω` (`col_eq_zero_of_isMinOn`). A token that never occurs in
training is such a row or column of every kernel: the learner that minimizes the nuclear norm
answers `0` there, which decodes to no value.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix
open scoped ComplexOrder

variable {m n : Type*} [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n]

/-- **Two weighted blocks of rows**: for an orthogonal projection `P`, contractions `Z`, `Z'`
and `c² + s² ≤ 1`, the matrix `c P Z + s (1 − P) Z'` is a contraction. -/
theorem IsContraction.smul_proj_add {P : Matrix m m ℂ} (hP : Pᴴ = P) (hPP : P * P = P)
    {Z Z' : Matrix m n ℂ} (hZ : IsContraction Z) (hZ' : IsContraction Z') {c s : ℝ}
    (hcs : c ^ 2 + s ^ 2 ≤ 1) :
    IsContraction ((c : ℂ) • (P * Z) + (s : ℂ) • ((1 - P) * Z')) := by
  have hP' : (1 - P)ᴴ = 1 - P := by rw [conjTranspose_sub, conjTranspose_one, hP]
  have hPP' := one_sub_mul_one_sub hPP
  have h01 : P * (1 - P) = 0 := by rw [Matrix.mul_sub, Matrix.mul_one, hPP, sub_self]
  have h10 : (1 - P) * P = 0 := by rw [Matrix.sub_mul, Matrix.one_mul, hPP, sub_self]
  have e1 : Zᴴ * P * (P * Z) = Zᴴ * P * Z := by
    rw [show Zᴴ * P * (P * Z) = Zᴴ * (P * P) * Z by simp only [Matrix.mul_assoc], hPP]
  have e2 : Z'ᴴ * (1 - P) * ((1 - P) * Z') = Z'ᴴ * (1 - P) * Z' := by
    rw [show Z'ᴴ * (1 - P) * ((1 - P) * Z') = Z'ᴴ * ((1 - P) * (1 - P)) * Z' by
      simp only [Matrix.mul_assoc], hPP']
  have e3 : Zᴴ * P * ((1 - P) * Z') = 0 := by
    rw [show Zᴴ * P * ((1 - P) * Z') = Zᴴ * (P * (1 - P)) * Z' by simp only [Matrix.mul_assoc],
      h01, Matrix.mul_zero, Matrix.zero_mul]
  have e4 : Z'ᴴ * (1 - P) * (P * Z) = 0 := by
    rw [show Z'ᴴ * (1 - P) * (P * Z) = Z'ᴴ * ((1 - P) * P) * Z by simp only [Matrix.mul_assoc],
      h10, Matrix.mul_zero, Matrix.zero_mul]
  have h : 1 - ((c : ℂ) • (P * Z) + (s : ℂ) • ((1 - P) * Z'))ᴴ *
      ((c : ℂ) • (P * Z) + (s : ℂ) • ((1 - P) * Z')) =
      ((c ^ 2 : ℝ) : ℂ) • (1 - Zᴴ * P * Z) + ((s ^ 2 : ℝ) : ℂ) • (1 - Z'ᴴ * (1 - P) * Z') +
        ((1 - (c ^ 2 + s ^ 2) : ℝ) : ℂ) • (1 : Matrix n n ℂ) := by
    rw [conjTranspose_add, conjTranspose_smul, conjTranspose_smul, conjTranspose_mul,
      conjTranspose_mul, hP, hP', Complex.star_def, Complex.conj_ofReal, Complex.conj_ofReal]
    simp only [Matrix.add_mul, Matrix.mul_add, Matrix.smul_mul, Matrix.mul_smul, e1, e2, e3, e4,
      smul_zero, add_zero, zero_add, smul_smul]
    push_cast
    module
  rw [IsContraction, h]
  exact (((hZ.posSemidef_one_sub hP hPP).smul (Complex.zero_le_real.mpr (sq_nonneg c))).add
    ((hZ'.posSemidef_one_sub hP' hPP').smul (Complex.zero_le_real.mpr (sq_nonneg s)))).add
    (PosSemidef.one.smul (Complex.zero_le_real.mpr (sub_nonneg.mpr hcs)))

example := (isContraction_one (n := Fin 2)).smul_proj_add (P := 1) (by simp) (by simp)
  isContraction_one (c := 1) (s := 0) (by norm_num)

/-- **Two blocks of rows in `ℓ²`**, `review/flat_spectrum.md`, item CW8: for an orthogonal
projection `P`, `‖P X‖_S1² + ‖(1 − P) X‖_S1² ≤ ‖X‖_S1²`. -/
theorem nuclearNorm_proj_sq_add_sq_le {P : Matrix m m ℂ} (hP : Pᴴ = P) (hPP : P * P = P)
    (X : Matrix m n ℂ) :
    nuclearNorm (P * X) ^ 2 + nuclearNorm ((1 - P) * X) ^ 2 ≤ nuclearNorm X ^ 2 := by
  obtain ⟨Z, hZ, hZa⟩ := exists_isContraction_trace_eq (P * X)
  obtain ⟨Z', hZ', hZb⟩ := exists_isContraction_trace_eq ((1 - P) * X)
  have hP' : (1 - P)ᴴ = 1 - P := by rw [conjTranspose_sub, conjTranspose_one, hP]
  set a := nuclearNorm (P * X)
  set b := nuclearNorm ((1 - P) * X)
  have ha : 0 ≤ a := nuclearNorm_nonneg _
  have hb : 0 ≤ b := nuclearNorm_nonneg _
  have key : ∀ c s : ℝ, c ^ 2 + s ^ 2 ≤ 1 → c * a + s * b ≤ nuclearNorm X := fun c s hcs => by
    have h1 := (Complex.re_le_norm _).trans
      (norm_trace_le_nuclearNorm (hZ.smul_proj_add hP hPP hZ' hcs) X)
    have h2 : (((c : ℂ) • (P * Z) + (s : ℂ) • ((1 - P) * Z'))ᴴ * X).trace =
        ((c * a + s * b : ℝ) : ℂ) := by
      rw [conjTranspose_add, conjTranspose_smul, conjTranspose_smul, conjTranspose_mul,
        conjTranspose_mul, hP, hP', Complex.star_def, Complex.conj_ofReal, Complex.conj_ofReal,
        Matrix.add_mul, Matrix.smul_mul, Matrix.smul_mul, Matrix.mul_assoc, Matrix.mul_assoc,
        trace_add, trace_smul, trace_smul, hZa, hZb]
      push_cast
      rfl
    rwa [h2, Complex.ofReal_re] at h1
  by_cases hr : a ^ 2 + b ^ 2 = 0
  · rw [hr]
    positivity
  · have hr0 : 0 < a ^ 2 + b ^ 2 := lt_of_le_of_ne (by positivity) (Ne.symm hr)
    have hs : 0 < √(a ^ 2 + b ^ 2) := Real.sqrt_pos.mpr hr0
    have hsq := Real.sq_sqrt hr0.le
    have h := key (a / √(a ^ 2 + b ^ 2)) (b / √(a ^ 2 + b ^ 2))
      (by rw [div_pow, div_pow, ← add_div, hsq, div_self hr])
    have h3 : a / √(a ^ 2 + b ^ 2) * a + b / √(a ^ 2 + b ^ 2) * b = √(a ^ 2 + b ^ 2) := by
      field_simp
      nlinarith
    rw [h3] at h
    nlinarith

example (X : Matrix (Fin 2) (Fin 2) ℂ) :=
  nuclearNorm_proj_sq_add_sq_le (P := 1) (by simp) (by simp) X

/-- Removing a block of rows that is not zero lowers the nuclear norm. -/
theorem nuclearNorm_proj_lt {P : Matrix m m ℂ} (hP : Pᴴ = P) (hPP : P * P = P)
    {X : Matrix m n ℂ} (hX : (1 - P) * X ≠ 0) : nuclearNorm (P * X) < nuclearNorm X := by
  have h := nuclearNorm_proj_sq_add_sq_le hP hPP X
  have hb : 0 < nuclearNorm ((1 - P) * X) :=
    lt_of_le_of_ne (nuclearNorm_nonneg _) fun h0 => hX (eq_zero_of_nuclearNorm_eq_zero h0.symm)
  nlinarith [nuclearNorm_nonneg (P * X), nuclearNorm_nonneg X]

example : nuclearNorm ((0 : Matrix (Fin 2) (Fin 2) ℂ) * (1 : Matrix (Fin 2) (Fin 2) ℂ)) <
    nuclearNorm (1 : Matrix (Fin 2) (Fin 2) ℂ) :=
  nuclearNorm_proj_lt (P := 0) (by simp) (by simp) (by simp)

/-- **An unseen row is zero in every minimum**, `review/flat_spectrum.md`, item CW8: if no
cell of row `i` lies in `Ω`, every minimum of the nuclear norm among the matrices equal to `K`
on `Ω` has a zero row `i`. -/
theorem row_eq_zero_of_isMinOn {K X : Matrix m n ℂ} {Ω : Set (m × n)} {i : m}
    (hi : ∀ j, (i, j) ∉ Ω) (hX : ∀ kl ∈ Ω, X kl.1 kl.2 = K kl.1 kl.2)
    (hmin : IsMinOn nuclearNorm {Y | ∀ kl ∈ Ω, Y kl.1 kl.2 = K kl.1 kl.2} X) (j : n) :
    X i j = 0 := by
  set E : Matrix m m ℂ := single i i 1 with hEdef
  have hE : Eᴴ = E := by rw [hEdef, conjTranspose_single, star_one]
  have hEE : E * E = E := by rw [hEdef, single_mul_single_same, mul_one]
  have hP : (1 - E)ᴴ = 1 - E := by rw [conjTranspose_sub, conjTranspose_one, hE]
  have hPP := one_sub_mul_one_sub hEE
  have hmem : ∀ kl ∈ Ω, ((1 - E) * X : Matrix m n ℂ) kl.1 kl.2 = K kl.1 kl.2 := fun kl hkl => by
    have hne : kl.1 ≠ i := fun h => hi kl.2 (by rw [← h]; exact hkl)
    rw [Matrix.sub_mul, Matrix.one_mul, Matrix.sub_apply, hEdef,
      single_mul_apply_of_ne _ _ _ _ _ hne, sub_zero, hX kl hkl]
  by_contra hij
  have hne : (1 - (1 - E)) * X ≠ 0 := fun h0 => hij (by
    have h1 := congrFun (congrFun h0 i) j
    rwa [sub_sub_cancel, hEdef, single_mul_apply_same, one_mul] at h1)
  exact absurd (isMinOn_iff.mp hmin _ hmem) (not_le.mpr (nuclearNorm_proj_lt hP hPP hne))

/-- **An unseen column is zero in every minimum**, `review/flat_spectrum.md`, item CW8. -/
theorem col_eq_zero_of_isMinOn {K X : Matrix m n ℂ} {Ω : Set (m × n)} {j : n}
    (hj : ∀ i, (i, j) ∉ Ω) (hX : ∀ kl ∈ Ω, X kl.1 kl.2 = K kl.1 kl.2)
    (hmin : IsMinOn nuclearNorm {Y | ∀ kl ∈ Ω, Y kl.1 kl.2 = K kl.1 kl.2} X) (i : m) :
    X i j = 0 := by
  have h := row_eq_zero_of_isMinOn (K := Kᴴ) (X := Xᴴ) (Ω := {kl | (kl.2, kl.1) ∈ Ω}) (i := j)
    (fun i => hj i) (fun kl hkl => by
      rw [conjTranspose_apply, conjTranspose_apply, hX (kl.2, kl.1) hkl])
    (isMinOn_iff.mpr fun Y hY => by
      rw [nuclearNorm_conjTranspose, ← nuclearNorm_conjTranspose Y]
      refine isMinOn_iff.mp hmin _ fun kl hkl => ?_
      have h1 := hY (kl.2, kl.1) hkl
      rw [conjTranspose_apply] at h1 ⊢
      rw [h1, star_star]) i
  rwa [conjTranspose_apply, star_eq_zero] at h

/-- A table of ones with an unseen row and an unseen column: the zero matrix is the minimum
when nothing is seen, and its unseen row and column are zero. -/
example (j : Fin 2) : (0 : Matrix (Fin 2) (Fin 2) ℂ) 0 j = 0 :=
  row_eq_zero_of_isMinOn (K := of fun _ _ => 1) (Ω := ∅) (i := 0) (fun _ => by simp)
    (fun _ h => absurd h (by simp))
    (isMinOn_iff.mpr fun Y _ => by rw [nuclearNorm_zero]; exact nuclearNorm_nonneg Y) j

example (i : Fin 2) : (0 : Matrix (Fin 2) (Fin 2) ℂ) i 0 = 0 :=
  col_eq_zero_of_isMinOn (K := of fun _ _ => 1) (Ω := ∅) (j := 0) (fun _ => by simp)
    (fun _ h => absurd h (by simp))
    (isMinOn_iff.mpr fun Y _ => by rw [nuclearNorm_zero]; exact nuclearNorm_nonneg Y) i

end Transformer.Grokking.FlatSpectrum
