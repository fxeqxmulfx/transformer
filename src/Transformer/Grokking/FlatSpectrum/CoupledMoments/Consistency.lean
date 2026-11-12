module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Loss
public import Transformer.Grokking.FlatSpectrum.AddReadoutSpectrum

/-!
# Shared complex codes carry consistent class information

Source: `review/flat_spectrum.md`, CM3. An orthogonal complete code determines the shared
probabilities uniquely; the Fourier code on any nonzero finite cyclic group is one such
code. A four-class witness shows that independent unit disks admit moment values no common
distribution can produce. This is a strict restriction, not an unseen-label theorem.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments

open Matrix Finset

variable {m n C F : Type*} [Fintype C] [Fintype F] [DecidableEq C]

/-- A full orthogonal complex code determines all probability coordinates (CM3). -/
theorem moment_injective_of_gram (φ : Matrix F C ℂ) {ρ : ℂ} (hρ : ρ ≠ 0)
    (hGram : φᴴ * φ = ρ • 1) {P Q : Table m n C}
    (h : ∀ f, moment (φ f) P = moment (φ f) Q) : P = Q := by
  funext i j c
  have hm : φ *ᵥ (fun d => (P i j d : ℂ)) = φ *ᵥ (fun d => (Q i j d : ℂ)) := by
    funext f
    have he := congrFun (congrFun (h f) i) j
    simpa [moment, mulVec, dotProduct, mul_comm] using he
  have hg := congrArg (fun v => φᴴ *ᵥ v) hm
  simp only [mulVec_mulVec, hGram, smul_mulVec, one_mulVec] at hg
  have hc := congrFun hg c
  exact Complex.ofReal_injective (mul_left_cancel₀ hρ hc)

/-- Fourier coordinates preserve the entire label distribution for every positive
vocabulary size, without using any task labels or arithmetic rule (CM3). -/
theorem fourier_moments_injective {N : ℕ} [NeZero N] {P Q : Table m n (ZMod N)}
    (h : ∀ f, moment (fourierMatrix N f) P = moment (fourierMatrix N f) Q) : P = Q :=
  moment_injective_of_gram (fourierMatrix N) (Nat.cast_ne_zero.mpr (NeZero.ne N))
    conjTranspose_fourierMatrix_mul h

example := moment_injective_of_gram (fourierMatrix 3)
  (by norm_num : (3 : ℂ) ≠ 0) conjTranspose_fourierMatrix_mul
  (P := pointMass (fun (_ : Fin 1) (_ : Fin 1) => (0 : ZMod 3))) (fun _ => rfl)

example := fourier_moments_injective
  (P := pointMass (fun (_ : Fin 1) (_ : Fin 1) => (0 : ZMod 3))) (fun _ => rfl)

/-- For probability tables the zero mode is already fixed by total mass one, so the
nonzero Fourier modes alone determine the table (CM3). -/
theorem nonzero_fourier_moments_injective {N : ℕ} [NeZero N]
    {P Q : Table m n (ZMod N)} (hP : Rows P) (hQ : Rows Q)
    (h : ∀ f, f ≠ 0 → moment (fourierMatrix N f) P = moment (fourierMatrix N f) Q) : P = Q := by
  apply fourier_moments_injective
  intro f
  by_cases hf : f = 0
  · subst f
    ext i j
    simp only [moment, fourierMatrix, Matrix.of_apply, zero_mul, AddChar.map_zero_eq_one,
      mul_one, ← Complex.ofReal_sum, hP.2, hQ.2]
  · exact h f hf

example := nonzero_fourier_moments_injective
  (rows_pointMass (fun (_ : Fin 1) (_ : Fin 1) => (0 : ZMod 3)))
  (rows_pointMass (fun (_ : Fin 1) (_ : Fin 1) => (0 : ZMod 3))) (fun _ _ => rfl)

/-- First and second powers of the four equally spaced unit codes (CM3). -/
def quarterCode (f : Fin 2) : Fin 4 → ℂ :=
  if f = 0 then ![1, Complex.I, -1, -Complex.I] else ![1, -1, 1, -1]

/-- Saturating the first reading forces the second reading to be `+1` (CM3). -/
theorem second_eq_one_of_first {P : Table (Fin 1) (Fin 1) (Fin 4)} (hP : Rows P)
    (hfirst : (moment (quarterCode 0) P 0 0).re = 1) :
    moment (quarterCode 1) P 0 0 = 1 := by
  have hs := hP.2 0 0
  have h0 := hP.1 0 0 0
  have h1 := hP.1 0 0 1
  have h2 := hP.1 0 0 2
  have h3 := hP.1 0 0 3
  simp [moment, quarterCode, Fin.sum_univ_succ, Complex.mul_re] at hfirst
  simp [Fin.sum_univ_succ] at hs
  have e0 : P 0 0 0 = 1 := by linarith
  have e1 : P 0 0 1 = 0 := by linarith
  have e2 : P 0 0 2 = 0 := by linarith
  have e3 : P 0 0 3 = 0 := by linarith
  simp [moment, quarterCode, Fin.sum_univ_succ, e0, e1, e2, e3]

example := second_eq_one_of_first
  (rows_pointMass (fun (_ : Fin 1) (_ : Fin 1) => (0 : Fin 4)))
  (by rw [moment_pointMass]; norm_num [quarterCode])

/-- Independent disks permit first reading `+1` and second reading `-1`; no common
class probabilities do. Coupling is therefore strictly stronger than separate disks (CM3). -/
theorem independent_readings_not_joint :
    ‖(1 : ℂ)‖ ≤ 1 ∧ ‖(-1 : ℂ)‖ ≤ 1 ∧
    ¬∃ P : Table (Fin 1) (Fin 1) (Fin 4), Rows P ∧
      moment (quarterCode 0) P 0 0 = 1 ∧ moment (quarterCode 1) P 0 0 = -1 := by
  refine ⟨by norm_num, by norm_num, ?_⟩
  rintro ⟨P, hP, h1, h2⟩
  have he := second_eq_one_of_first hP (by rw [h1]; simp)
  rw [h2] at he
  norm_num at he

end Transformer.Grokking.FlatSpectrum.CoupledMoments
