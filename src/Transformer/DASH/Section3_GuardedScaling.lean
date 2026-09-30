/-
# DASH — automatic scaling with a certified strict margin

arXiv:2602.02016v2, §3.4. Twice the PI estimate is accepted only when
it strictly exceeds a certified upper bound. Otherwise twice that
bound is used, with scale one at zero. This repairs both the unsafe
PI guarantee and the CN endpoint introduced by exact normalization.
-/

import Transformer.DASH.Section3_RowBounds

open scoped Matrix

noncomputable section

namespace Transformer.DASH

variable {n : ℕ}

/-- The corrected scaling algorithm. The candidate `r` may come from
any PI pool, including zero or poorly aligned starting vectors.
Source: arXiv:2602.02016v2, §3.4, corrected factor-two normalization. -/
def guardedScale (A : Matrix (Fin n) (Fin n) ℝ) (r : ℝ) : ℝ :=
  let u := certifiedSpectralBound A
  if u < 2 * r then 2 * r else if 0 < u then 2 * u else 1

/-- The chosen scale strictly exceeds the computed spectral certificate.
No reliability hypothesis about PI is needed.
Source: arXiv:2602.02016v2, §3.4, repaired upper-bound guarantee. -/
theorem certifiedSpectralBound_lt_guardedScale
    (A : Matrix (Fin n) (Fin n) ℝ) (r : ℝ) :
    certifiedSpectralBound A < guardedScale A r := by
  dsimp only [guardedScale]
  split
  · assumption
  · split
    · linarith
    · have := certifiedSpectralBound_nonneg A
      linarith

/-- Scaling is always positive, including for a zero matrix and estimate.
Source: arXiv:2602.02016v2, §3.4, corrected normalization domain. -/
theorem guardedScale_pos (A : Matrix (Fin n) (Fin n) ℝ) (r : ℝ) :
    0 < guardedScale A r :=
  lt_of_le_of_lt (certifiedSpectralBound_nonneg A)
    (certifiedSpectralBound_lt_guardedScale A r)

/-- The automatic scale strictly exceeds every eigenvalue, independently
of the candidate estimate. Source: arXiv:2602.02016v2, §3.4, repaired PI bound. -/
theorem eigenvalue_lt_guardedScale (Q : Matrix (Fin n) (Fin n) ℝ)
    (s : Fin n → ℝ) (r : ℝ) (hQ : Orthogonal Q) (i : Fin n) :
    s i < guardedScale (spectralMatrix Q s) r :=
  lt_of_le_of_lt (le_trans (le_abs_self _) (eigenvalue_le_certifiedSpectralBound Q s hQ i))
    (certifiedSpectralBound_lt_guardedScale _ _)

/-- The guarded eigenvalue theorem has admissible frames,
arXiv:2602.02016v2, §3.4. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  simp [Orthogonal, Muon.OrthonormalColumns]

/-- A PI candidate passing the independent certificate is used unchanged.
Source: arXiv:2602.02016v2, §3.4, checked version of `2λ_max^PI`. -/
theorem guardedScale_accepts (A : Matrix (Fin n) (Fin n) ℝ) (r : ℝ)
    (h : certifiedSpectralBound A < 2 * r) : guardedScale A r = 2 * r := by
  simp only [guardedScale, ite_eq_left h]

/-- The acceptance hypothesis is nonempty, arXiv:2602.02016v2, §3.4. -/
example : certifiedSpectralBound (0 : Matrix (Fin 1) (Fin 1) ℝ) < 2 * (1 : ℝ) := by
  norm_num [certifiedSpectralBound, frobeniusNorm, Muon.squaredFrobenius, absoluteRowBound]

/-- A rejected estimate falls back to a certified bound with a strict margin.
Source: arXiv:2602.02016v2, §3.4, corrected PI fallback. -/
theorem guardedScale_fallback (A : Matrix (Fin n) (Fin n) ℝ) (r : ℝ)
    (h : ¬ certifiedSpectralBound A < 2 * r) (hu : 0 < certifiedSpectralBound A) :
    guardedScale A r = 2 * certifiedSpectralBound A := by
  simp only [guardedScale, ite_eq_right h, ite_eq_left hu]

/-- The fallback hypotheses are satisfiable, arXiv:2602.02016v2, §3.4. -/
example : ¬ certifiedSpectralBound (1 : Matrix (Fin 1) (Fin 1) ℝ) < 2 * (0 : ℝ) ∧
    0 < certifiedSpectralBound (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  have hrow : (fun i : Fin 1 => ∑ j : Fin 1, |(1 : Matrix (Fin 1) (Fin 1) ℝ) i j|) =
      (1 : Fin 1 → ℝ) := by funext i; fin_cases i; simp [Matrix.one_apply]
  have hr : absoluteRowBound (1 : Matrix (Fin 1) (Fin 1) ℝ) = 1 := by
    unfold absoluteRowBound
    rw [hrow]
    exact norm_one
  rw [certifiedSpectralBound, hr]
  norm_num [frobeniusNorm, Muon.squaredFrobenius]

/-- Normalize using the actual guarded choice, not a supplied upper bound.
Source: arXiv:2602.02016v2, §3.4, repaired solver preprocessing. -/
def guardedNormalize (A : Matrix (Fin n) (Fin n) ℝ) (r : ℝ) :
    Matrix (Fin n) (Fin n) ℝ := (guardedScale A r)⁻¹ • A

/-- The actual normalized matrix has the expected scaled eigenvalues.
Source: arXiv:2602.02016v2, §3.4, normalization before CN/NDB. -/
theorem guardedNormalize_spectral (Q : Matrix (Fin n) (Fin n) ℝ)
    (s : Fin n → ℝ) (r : ℝ) :
    guardedNormalize (spectralMatrix Q s) r =
      spectralMatrix Q (fun i => s i / guardedScale (spectralMatrix Q s) r) := by
  simp only [guardedNormalize, spectralMatrix_smul, div_eq_mul_inv, mul_comm]

/-- Every normalized absolute eigenvalue is strictly below one, even when
the candidate estimate misses the dominant eigenspace.
Source: arXiv:2602.02016v2, §3.4, corrected strict spectral domain. -/
theorem guarded_spectrum_abs_lt_one (Q : Matrix (Fin n) (Fin n) ℝ)
    (s : Fin n → ℝ) (r : ℝ) (hQ : Orthogonal Q) (i : Fin n) :
    |s i / guardedScale (spectralMatrix Q s) r| < 1 := by
  rw [abs_div, abs_of_pos (guardedScale_pos _ _), div_lt_one (guardedScale_pos _ _)]
  exact lt_of_le_of_lt (eigenvalue_le_certifiedSpectralBound Q s hQ i)
    (certifiedSpectralBound_lt_guardedScale _ _)

/-- An orthogonal frame for the automatic certificate exists,
arXiv:2602.02016v2, §3.4. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  simp [Orthogonal, Muon.OrthonormalColumns]

/-- Positive spectra automatically land in the open interval `(0,1)`.
This fixes the printed CN unit endpoint as well as the PI scaling claim.
Source: arXiv:2602.02016v2, §3.2–3.4. -/
theorem guarded_spectrum_bounds (Q : Matrix (Fin n) (Fin n) ℝ)
    (s : Fin n → ℝ) (r : ℝ) (hQ : Orthogonal Q) (hs : ∀ i, 0 < s i) (i : Fin n) :
    0 < s i / guardedScale (spectralMatrix Q s) r ∧
      s i / guardedScale (spectralMatrix Q s) r < 1 := by
  have hpos := div_pos (hs i) (guardedScale_pos (spectralMatrix Q s) r)
  exact ⟨hpos, by simpa only [abs_of_pos hpos] using
    guarded_spectrum_abs_lt_one Q s r hQ i⟩

/-- The positive spectral hypotheses are satisfiable,
arXiv:2602.02016v2, §3.2–3.4. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧
    (∀ i : Fin 1, (0 : ℝ) < (fun _ => (1 : ℝ)) i) := by
  norm_num [Orthogonal, Muon.OrthonormalColumns]

/-- The complete PI preprocessing pipeline includes the independent guard.
The pool and iteration budget are unchanged from the source.
Source: arXiv:2602.02016v2, §3.4–3.5, corrected multi-PI normalization. -/
def pooledGuardedScale {ι : Type} [Fintype ι] [Nonempty ι]
    (A : Matrix (Fin n) (Fin n) ℝ) (starts : ι → Fin n → ℝ) (k : ℕ) : ℝ :=
  guardedScale A (pooledRayleigh A starts k)

end Transformer.DASH
