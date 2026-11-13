module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DensityBounds.Basic

/-!
# Observable feature balancing and a feasible density

Source: `review/flat_spectrum.md`, CM24. Normalize actual nonzero feature rows,
not an unknown Fourier reference. The uniform density then has unit readings.
This establishes feasibility, not purity, centering or hidden-mode recovery.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.BalancedFeatures

open Finset Matrix DensityExtraction DensityBounds

variable {ι X : Type*} [Fintype ι]

/-- The measured squared Euclidean energy of a feature row, CM24. -/
noncomputable def rowEnergy (B : Matrix X ι ℂ) (x : X) : ℝ :=
  ∑ i, Complex.normSq (B x i)

/-- Radial row normalization to squared norm equal to the coefficient count, CM24.
Zero rows remain zero; their completion is a separate, explicit construction. -/
noncomputable def normalize (B : Matrix X ι ℂ) : Matrix X ι ℂ :=
  fun x i => (Real.sqrt (Fintype.card ι / rowEnergy B x) : ℂ) * B x i

/-- The isotropic coefficient density, CM24. -/
noncomputable def uniformDensity [DecidableEq ι] : Matrix ι ι ℂ :=
  diagonal (fun _ => ((1 / Fintype.card ι : ℝ) : ℂ))

/-- Squared row energy is nonnegative without any subspace hypothesis, CM24. -/
theorem energy_nonneg (B : Matrix X ι ℂ) (x : X) : 0 ≤ rowEnergy B x :=
  sum_nonneg fun i _ => Complex.normSq_nonneg (B x i)

/-- A genuinely nonzero feature row has positive measured energy, CM24. -/
theorem energy_pos (B : Matrix X ι ℂ) (x : X) (h : ∃ i, B x i ≠ 0) :
    0 < rowEnergy B x := by
  obtain ⟨i, hi⟩ := h
  exact sum_pos' (fun j _ => Complex.normSq_nonneg (B x j))
    ⟨i, mem_univ i, Complex.normSq_pos.mpr hi⟩

/-- The proposed normalization has the claimed actual row energy, CM24. -/
theorem normalized_energy (B : Matrix X ι ℂ) (x : X) (h : 0 < rowEnergy B x) :
    rowEnergy (normalize B) x = Fintype.card ι := by
  unfold rowEnergy normalize
  simp_rw [Complex.normSq_mul, Complex.normSq_ofReal, ← sq]
  rw [← mul_sum, Real.sq_sqrt (div_nonneg (Nat.cast_nonneg _) h.le)]
  exact div_mul_cancel₀ _ h.ne'

variable [DecidableEq ι] [Nonempty ι]

/-- The explicit isotropic matrix is PSD and has trace one, CM24. -/
theorem uniform_density : Density (uniformDensity (ι := ι)) := by
  apply (diagonal_density _).mpr
  refine ⟨fun _ => by positivity, ?_⟩
  have hd : (Fintype.card ι : ℝ) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
  simp [hd]

omit [Nonempty ι] in
/-- Its actual kernel diagonal is exactly measured row energy divided by
coefficient count, CM24. No orthogonality of feature columns is assumed. -/
theorem uniform_reading (B : Matrix X ι ℂ) (x : X) :
    reading B uniformDensity x x = ((rowEnergy B x / Fintype.card ι : ℝ) : ℂ) := by
  rw [uniformDensity, reading_diagonal]
  simp_rw [Complex.star_def, Complex.mul_conj]
  rw [← mul_sum, ← Complex.ofReal_sum]
  simp [rowEnergy, div_eq_mul_inv, mul_comm]

/-- Nonzero rows yield a known feasible point in the exact convex density
domain after observable normalization, CM24. -/
theorem normalized_feasible (B : Matrix X ι ℂ) (h : ∀ x, ∃ i, B x i ≠ 0) :
    Feasible (normalize B) uniformDensity := by
  refine ⟨uniform_density, fun x => ?_⟩
  rw [uniform_reading, normalized_energy B x (energy_pos B x (h x))]
  have hd : (Fintype.card ι : ℝ) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
  simp [hd]

/-- Feasibility and convexity hold together for every nonnegative tolerance,
CM24. The preprocessing is not asserted to be linear or jointly convex. -/
theorem normalized_domain (B : Matrix X ι ℂ) (h : ∀ x, ∃ i, B x i ≠ 0)
    {ε : ℝ} (hε : 0 ≤ ε) :
    Convex ℝ {M | Relaxed (normalize B) ε M} ∧
      Relaxed (normalize B) ε uniformDensity := by
  exact ⟨relaxed_convex _ _, exact_relaxed (normalized_feasible B h) hε⟩

omit [DecidableEq ι] [Nonempty ι] in
/-- A zero row cannot satisfy the original unit-diagonal constraint, CM24.
Explicit completion changes that feature row; it cannot certify the old domain. -/
theorem zero_row_obstruction (B : Matrix X ι ℂ) (x : X) (h : ∀ i, B x i = 0)
    (M : Matrix ι ι ℂ) : ¬ Feasible B M := by
  intro hf
  have hz : reading B M x x = 0 := by simp [reading_eq, h]
  have hc := hf.2 x
  rw [hz] at hc
  exact zero_ne_one hc

/-- Unequal, nonzero measured rows provide a closed normalization witness, CM24. -/
def sampleFeatures : Matrix Bool Bool ℂ :=
  fun x i => if x then (if i then 3 else 4) else (if i then 0 else 2)

example := energy_nonneg sampleFeatures false
example := energy_pos sampleFeatures true ⟨false, by norm_num [sampleFeatures]⟩
example := normalized_energy sampleFeatures false (by norm_num [rowEnergy, sampleFeatures])
example := uniform_density (ι := Bool)
example := uniform_reading sampleFeatures true
example := normalized_feasible sampleFeatures
  (by intro x; cases x <;> exact ⟨false, by norm_num [sampleFeatures]⟩)
example := normalized_domain sampleFeatures
  (by intro x; cases x <;> exact ⟨false, by norm_num [sampleFeatures]⟩)
  (ε := 1 / 10) (by norm_num)
example := zero_row_obstruction (fun x i : Bool => if x then (if i then 3 else 2) else 0)
  false (by simp) (uniformDensity (ι := Bool))

end Transformer.Grokking.FlatSpectrum.CoupledMoments.BalancedFeatures
