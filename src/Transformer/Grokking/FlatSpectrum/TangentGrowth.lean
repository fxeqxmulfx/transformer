module

public import Transformer.Grokking.FlatSpectrum.RankMinimum

/-!
# The nuclear norm near `K`, along the tangent space

Source: `review/flat_spectrum.md`, *Rank above one* (added), item 6.1, the lemmas of its proof.
Let `K = U diag(s) Vᴴ` with orthonormal columns in `U` and `V` and `s > 0`, and let
`H = U Aᴴ + B Vᴴ`. With `R = diag(√s)` and `R' = diag(1/√s)`, the factors `P = U R + t B R'` and
`Q = V R + t A R'` have `P Qᴴ = K + t H + t² B R'² Aᴴ`, and Lemma 1 (`nuclearNorm_le_of_eq_mul`)
bounds `‖P Qᴴ‖_S1` by `(‖P‖²_S2 + ‖Q‖²_S2)/2`, whose expansion in `t`
(`frobeniusSq_add_smul`, from `Interpolation`) starts with `∑ s + t Re tr((U Vᴴ)ᴴ H)`. So
`‖K + t H‖_S1 ≤ ‖K‖_S1 + t Re tr((U Vᴴ)ᴴ H) + C t²` for every real `t`
(`exists_nuclearNorm_add_smul_tangent_le`): along `T` the nuclear norm has no kink at `K`. Every
`H` is a part in `T` plus `(1 − U Uᴴ) H (1 − V Vᴴ)`, which adds at most `|t|` times its nuclear
norm (`exists_nuclearNorm_add_smul_le`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset
open scoped ComplexOrder

variable {m n r : Type*} [Fintype m] [Fintype n] [Fintype r] [DecidableEq m] [DecidableEq n]
  [DecidableEq r]

omit [DecidableEq m] in
/-- **No kink along the tangent space**: with `K = U diag(s) Vᴴ`, `s > 0`, and
`H = U Aᴴ + B Vᴴ`, some `C` has `‖K + t H‖_S1 ≤ ‖K‖_S1 + t Re tr((U Vᴴ)ᴴ H) + C t²` for every
real `t`. -/
theorem exists_nuclearNorm_add_smul_tangent_le {U : Matrix m r ℂ} {V : Matrix n r ℂ}
    (hU : Uᴴ * U = 1) (hV : Vᴴ * V = 1) {s : r → ℝ} (hs : ∀ k, 0 < s k) {K : Matrix m n ℂ}
    (hK : K = U * diagonal (fun k => (s k : ℂ)) * Vᴴ) (A : Matrix n r ℂ) (B : Matrix m r ℂ) :
    ∃ C : ℝ, ∀ t : ℝ, nuclearNorm (K + (t : ℂ) • (U * Aᴴ + B * Vᴴ)) ≤
      nuclearNorm K + t * ((U * Vᴴ)ᴴ * (U * Aᴴ + B * Vᴴ)).trace.re + C * t ^ 2 := by
  set R : Matrix r r ℂ := diagonal fun k => ((√(s k) : ℝ) : ℂ) with hRdef
  set R' : Matrix r r ℂ := diagonal fun k => (((√(s k))⁻¹ : ℝ) : ℂ) with hR'def
  have hsq (k : r) : √(s k) * √(s k) = s k := Real.mul_self_sqrt (hs k).le
  have hinv (k : r) : √(s k) * (√(s k))⁻¹ = 1 := mul_inv_cancel₀ (Real.sqrt_pos.2 (hs k)).ne'
  have hRR : R * R = diagonal fun k => (s k : ℂ) := by
    rw [hRdef, diagonal_mul_diagonal]
    congr 1
    funext k
    rw [← Complex.ofReal_mul, hsq]
  have hRR' : R * R' = 1 := by
    rw [hRdef, hR'def, diagonal_mul_diagonal, ← diagonal_one]
    congr 1
    funext k
    rw [← Complex.ofReal_mul, hinv, Complex.ofReal_one]
  have hR'R : R' * R = 1 := by
    rw [hRdef, hR'def, diagonal_mul_diagonal, ← diagonal_one]
    congr 1
    funext k
    rw [← Complex.ofReal_mul, mul_comm, hinv, Complex.ofReal_one]
  have hRh : Rᴴ = R := by
    rw [hRdef, diagonal_conjTranspose]
    congr 1
    funext k
    simp
  have hR'h : R'ᴴ = R' := by
    rw [hR'def, diagonal_conjTranspose]
    congr 1
    funext k
    simp
  set E := B * R' * R' * Aᴴ with hEdef
  refine ⟨(frobeniusSq (B * R') + frobeniusSq (A * R')) / 2 + nuclearNorm E, fun t => ?_⟩
  -- the factorization `P Qᴴ = K + t H + t² E`
  have hPQ : (U * R + (t : ℂ) • (B * R')) * (V * R + (t : ℂ) • (A * R'))ᴴ =
      K + (t : ℂ) • (U * Aᴴ + B * Vᴴ) + ((t : ℂ) * t) • E := by
    simp only [conjTranspose_add, conjTranspose_smul, conjTranspose_mul, hRh, hR'h,
      Complex.star_def, Complex.conj_ofReal, Matrix.add_mul, Matrix.mul_add, Matrix.smul_mul,
      Matrix.mul_smul, smul_smul, smul_add]
    have e1 : U * R * (R * Vᴴ) = K := by
      rw [hK, Matrix.mul_assoc U R, ← Matrix.mul_assoc R R, hRR, Matrix.mul_assoc]
    have e2 : U * R * (R' * Aᴴ) = U * Aᴴ := by
      rw [Matrix.mul_assoc U R, ← Matrix.mul_assoc R R', hRR', Matrix.one_mul]
    have e3 : B * R' * (R * Vᴴ) = B * Vᴴ := by
      rw [Matrix.mul_assoc B R', ← Matrix.mul_assoc R' R, hR'R, Matrix.one_mul]
    have e4 : B * R' * (R' * Aᴴ) = E := by
      rw [hEdef, Matrix.mul_assoc (B * R')]
    rw [e1, e2, e3, e4]
    abel
  have hsplit : K + (t : ℂ) • (U * Aᴴ + B * Vᴴ) =
      (U * R + (t : ℂ) • (B * R')) * (V * R + (t : ℂ) • (A * R'))ᴴ + (-((t : ℂ) * t)) • E := by
    rw [hPQ, neg_smul]
    abel
  -- the two Frobenius norms
  obtain ⟨-, hKtr, hKnorm⟩ := svd_hypotheses hU hV fun k => (hs k).le
  have hUR : frobeniusSq (U * R) = ∑ k, s k := by
    rw [frobeniusSq_eq_re_trace, conjTranspose_mul, hRh, Matrix.mul_assoc,
      ← Matrix.mul_assoc Uᴴ, hU, Matrix.one_mul, hRR, trace_diagonal, Complex.re_sum]
    simp
  have hVR : frobeniusSq (V * R) = ∑ k, s k := by
    rw [frobeniusSq_eq_re_trace, conjTranspose_mul, hRh, Matrix.mul_assoc,
      ← Matrix.mul_assoc Vᴴ, hV, Matrix.one_mul, hRR, trace_diagonal, Complex.re_sum]
    simp
  have hUB : ((U * R)ᴴ * (B * R')).trace = (Uᴴ * B).trace := by
    rw [conjTranspose_mul, hRh, Matrix.mul_assoc, Matrix.trace_mul_comm, Matrix.mul_assoc,
      Matrix.mul_assoc, hR'R, Matrix.mul_one]
  have hVA : ((V * R)ᴴ * (A * R')).trace = (Vᴴ * A).trace := by
    rw [conjTranspose_mul, hRh, Matrix.mul_assoc, Matrix.trace_mul_comm, Matrix.mul_assoc,
      Matrix.mul_assoc, hR'R, Matrix.mul_one]
  -- the slope
  have hslope : ((U * Vᴴ)ᴴ * (U * Aᴴ + B * Vᴴ)).trace.re =
      (Uᴴ * B).trace.re + (Vᴴ * A).trace.re := by
    have h1 : ((U * Vᴴ)ᴴ * (U * Aᴴ)).trace = star (Vᴴ * A).trace := by
      rw [← trace_conjTranspose, conjTranspose_mul, conjTranspose_conjTranspose,
        conjTranspose_mul, conjTranspose_conjTranspose, Matrix.mul_assoc, ← Matrix.mul_assoc Uᴴ,
        hU, Matrix.one_mul, Matrix.trace_mul_comm]
    have h2 : ((U * Vᴴ)ᴴ * (B * Vᴴ)).trace = (Uᴴ * B).trace := by
      rw [conjTranspose_mul, conjTranspose_conjTranspose, Matrix.mul_assoc,
        Matrix.trace_mul_comm, Matrix.mul_assoc, Matrix.mul_assoc, hV, Matrix.mul_one]
    rw [Matrix.mul_add, trace_add, Complex.add_re, h1, h2, Complex.star_def, Complex.conj_re,
      add_comm]
  have hE : nuclearNorm ((-((t : ℂ) * t)) • E) = t ^ 2 * nuclearNorm E := by
    rw [nuclearNorm_smul, norm_neg, ← Complex.ofReal_mul, Complex.norm_real,
      Real.norm_of_nonneg (mul_self_nonneg t), sq]
  have hfac := nuclearNorm_le_of_eq_mul _ _
    (rfl : (U * R + (t : ℂ) • (B * R')) * (V * R + (t : ℂ) • (A * R'))ᴴ = _)
  rw [frobeniusSq_add_smul, frobeniusSq_add_smul, hUR, hVR, hUB, hVA] at hfac
  rw [hsplit]
  refine (nuclearNorm_add_le _ _).trans ?_
  rw [hE, hslope, hK, hKnorm]
  nlinarith [hfac]

example := exists_nuclearNorm_add_smul_tangent_le (U := (1 : Matrix (Fin 2) (Fin 2) ℂ))
  (V := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (by simp) (by simp) (s := ![1, 2])
  (fun k => by fin_cases k <;> norm_num) rfl 1 1

/-- **The nuclear norm near `K`**: with `K = U diag(s) Vᴴ`, `s > 0`, every `H` has some `C` with
`‖K + t H‖_S1 ≤ ‖K‖_S1 + t Re tr((U Vᴴ)ᴴ H) + |t| ‖(1 − U Uᴴ) H (1 − V Vᴴ)‖_S1 + C t²` for every
real `t`: `H` is `U (Uᴴ H) + ((1 − U Uᴴ) H V) Vᴴ`, in `T`, plus `(1 − U Uᴴ) H (1 − V Vᴴ)`, which
`U Vᴴ` does not see. -/
theorem exists_nuclearNorm_add_smul_le {U : Matrix m r ℂ} {V : Matrix n r ℂ}
    (hU : Uᴴ * U = 1) (hV : Vᴴ * V = 1) {s : r → ℝ} (hs : ∀ k, 0 < s k) {K : Matrix m n ℂ}
    (hK : K = U * diagonal (fun k => (s k : ℂ)) * Vᴴ) (H : Matrix m n ℂ) :
    ∃ C : ℝ, ∀ t : ℝ, nuclearNorm (K + (t : ℂ) • H) ≤ nuclearNorm K +
      t * ((U * Vᴴ)ᴴ * H).trace.re + |t| * nuclearNorm ((1 - U * Uᴴ) * H * (1 - V * Vᴴ)) +
        C * t ^ 2 := by
  obtain ⟨C, hC⟩ := exists_nuclearNorm_add_smul_tangent_le hU hV hs hK (Hᴴ * U)
    ((1 - U * Uᴴ) * H * V)
  refine ⟨C, fun t => ?_⟩
  set P := (1 - U * Uᴴ) * H * (1 - V * Vᴴ) with hPdef
  have hdec : H = (U * (Hᴴ * U)ᴴ + (1 - U * Uᴴ) * H * V * Vᴴ) + P := by
    rw [conjTranspose_mul, conjTranspose_conjTranspose, hPdef]
    simp only [Matrix.sub_mul, Matrix.mul_sub, Matrix.one_mul, Matrix.mul_one, Matrix.mul_assoc]
    abel
  have htr : ((U * Vᴴ)ᴴ * P).trace = 0 := by
    have h : Uᴴ * (1 - U * Uᴴ) = 0 := by
      rw [Matrix.mul_sub, Matrix.mul_one, ← Matrix.mul_assoc, hU, Matrix.one_mul, sub_self]
    rw [conjTranspose_mul, conjTranspose_conjTranspose, hPdef]
    simp only [Matrix.mul_assoc]
    rw [← Matrix.mul_assoc Uᴴ (1 - U * Uᴴ), h, Matrix.zero_mul, Matrix.mul_zero, trace_zero]
  have hslope : ((U * Vᴴ)ᴴ * (U * (Hᴴ * U)ᴴ + (1 - U * Uᴴ) * H * V * Vᴴ)).trace.re =
      ((U * Vᴴ)ᴴ * H).trace.re := by
    conv_rhs => rw [hdec, Matrix.mul_add, trace_add, Complex.add_re, htr, Complex.zero_re, add_zero]
  have hsplit : K + (t : ℂ) • H =
      (K + (t : ℂ) • (U * (Hᴴ * U)ᴴ + (1 - U * Uᴴ) * H * V * Vᴴ)) + (t : ℂ) • P := by
    conv_lhs => rw [hdec]
    rw [smul_add, add_assoc]
  have h := hC t
  rw [hslope] at h
  rw [hsplit]
  refine (nuclearNorm_add_le _ _).trans ?_
  rw [nuclearNorm_smul, Complex.norm_real, Real.norm_eq_abs]
  linarith

example := exists_nuclearNorm_add_smul_le (U := (1 : Matrix (Fin 2) (Fin 2) ℂ))
  (V := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (by simp) (by simp) (s := ![1, 2])
  (fun k => by fin_cases k <;> norm_num) rfl 1

end Transformer.Grokking.FlatSpectrum
