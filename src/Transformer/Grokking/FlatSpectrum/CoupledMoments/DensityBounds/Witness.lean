module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DensityBounds.Selection

/-!
# Mixed and genuinely perturbed witnesses

Source: `review/flat_spectrum.md`, CM22. The examples distinguish a positive
feature error from a bookkeeping tolerance, and a positive objective gap from
a real-score tie. Mixtures remain feasible even at zero tolerance.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.DensityBounds

open Finset Matrix DensityExtraction FourierExtraction

/-- The earlier genuine mixture remains feasible at zero error, CM22. -/
theorem mixed_relaxed : Relaxed (basis id tripleFrequency) 0
    (diagonal fun i => (mixedWeights i : ℂ)) := by
  exact exact_relaxed mixed_feasible le_rfl

/-- Zero amplitude error still does not imply purity, CM22. This refutes that
claim on the actual PSD trace-one domain, with separated frequencies. -/
theorem zero_error_not_pure : ∃ M, Relaxed (basis id tripleFrequency) 0 M ∧
    ∀ i, M ≠ projector i := by
  exact ⟨_, mixed_relaxed, mixed_not_pure⟩

/-- Two separated frequencies modulo four for a closed positive-gap example. -/
def twoFrequency : Fin 2 → ZMod 4 := ![0, 1]

/-- The two ordered differences are distinct and nonzero, CM22. -/
theorem two_separated : Separated twoFrequency := by
  unfold Separated twoFrequency
  decide

/-- The example's nontrivial character has real part zero, CM22. -/
theorem character_four_one : ZMod.stdAddChar (1 : ZMod 4) = Complex.I := by
  have h := ZMod.stdAddChar_coe (N := 4) (1 : ℤ)
  norm_num only [Int.cast_one, Nat.cast_ofNat, mul_one] at h
  rw [show (2 * (Real.pi : ℂ) * Complex.I / 4) = (Real.pi : ℂ) / 2 * Complex.I by ring] at h
  simpa only [Complex.exp_pi_div_two_mul_I] using h

/-- At rows one and zero, the real scores have a strict gap of one, CM22. -/
theorem two_gap : ∀ j : Fin 2, j ≠ 0 → (1 : ℝ) ≤
    (ZMod.stdAddChar (((1 : ZMod 4) - 0) * twoFrequency 0)).re -
      (ZMod.stdAddChar (((1 : ZMod 4) - 0) * twoFrequency j)).re := by
  intro j hj
  fin_cases j
  · exact (hj rfl).elim
  · norm_num [twoFrequency, character_four_one]

/-- An actual supplied feature perturbation: multiply the exact basis by
`11/10`. This example does not assume that centering or normalization implies
closeness; it computes the entry error itself, CM22. -/
noncomputable def perturbedBasis : Matrix (ZMod 4) (Fin 2) ℂ :=
  (11 / 10 : ℂ) • basis id twoFrequency

/-- The supplied and structural feature entries differ by exactly `1/10`,
CM22. In particular, the perturbation is genuinely nonzero. -/
theorem perturbed_error (x : ZMod 4) (i : Fin 2) :
    ‖perturbedBasis x i - basis id twoFrequency x i‖ = 1 / 10 := by
  change ‖(11 / 10 : ℂ) * basis id twoFrequency x i - basis id twoFrequency x i‖ = _
  rw [show (11 / 10 : ℂ) * basis id twoFrequency x i - basis id twoFrequency x i =
    (1 / 10 : ℂ) * basis id twoFrequency x i by ring]
  norm_num [norm_mul, basis, ZMod.stdAddChar_apply, Circle.norm_coe]

/-- The explicit perturbation budget admits every exact mode in this example,
CM22. The conservative kernel bound is `4*(1/10)*(2+1/10)=21/25`. -/
theorem perturbed_admitted (i : Fin 2) : Relaxed perturbedBasis (21 / 25) (projector i) := by
  apply reference_admitted perturbedBasis (basis id twoFrequency)
    (δ := 1 / 10) (R := 1) (projector_feasible id twoFrequency i) (by norm_num) (by norm_num)
  · intro x j
    exact (perturbed_error x j).le
  · intro x j
    simp [basis, ZMod.stdAddChar_apply, Circle.norm_coe]
  · norm_num

/-- A genuinely mixed approximate optimizer for the strict-gap example. -/
noncomputable def twoWeights : Fin 2 → ℝ := ![3 / 4, 1 / 4]

/-- The example's weights form a probability distribution, CM22. -/
theorem two_weights : Weights twoWeights := by
  constructor
  · intro i
    fin_cases i <;> norm_num [twoWeights]
  · norm_num [twoWeights, Fin.sum_univ_two]

/-- The mixed point is exactly feasible, CM22. -/
theorem two_feasible : Feasible (basis id twoFrequency)
    (diagonal fun i => (twoWeights i : ℂ)) := by
  exact diagonal_feasible id twoFrequency two_weights

/-- The mixed point is within `1/4` of every feasible real objective, CM22.
This inhabits the optimizer comparison with actual mixed mass and a positive
score gap, rather than assuming a recovered pure mode. -/
theorem two_near (N : Matrix (Fin 2) (Fin 2) ℂ) (hN : Relaxed (basis id twoFrequency) 0 N) :
    (reading (basis id twoFrequency) N (1 : ZMod 4) 0).re ≤
      (reading (basis id twoFrequency) (diagonal fun i => (twoWeights i : ℂ)) 1 0).re + 1 / 4 := by
  obtain ⟨w, hw, rfl⟩ := (feasible_iff_diagonal (Equiv.refl (ZMod 4)) two_separated N).mp
    ((relaxed_zero _ _).mp hN)
  have hs := weighted_bound hw (fun j => (ZMod.stdAddChar (twoFrequency j)).re) 1
    (fun j hj => by simpa only [ZMod.stdAddChar_apply, Circle.norm_coe] using
      Complex.re_le_norm (ZMod.stdAddChar (twoFrequency j)))
  norm_num [diagonal_reading, twoWeights, Fin.sum_univ_two, twoFrequency,
    character_four_one] at hs ⊢
  exact hs

end Transformer.Grokking.FlatSpectrum.CoupledMoments.DensityBounds
