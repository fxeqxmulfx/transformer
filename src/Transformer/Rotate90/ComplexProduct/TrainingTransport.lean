module

public import Transformer.Rotate90.ComplexProduct.SourceWitness
public import Transformer.Grokking.FlatSpectrum.FeatureCodes.TrainingTransport

/-!
# Training-state transport through the actual tied source readout

Source: `review/flat_spectrum.md`, *Training-state transport*, TT3–TT5.
Training contrast variation and an explicit query-state reconstruction control
the actual raw source gap. There is no fitted decoder or source-alignment term.
Each competitor may choose its own valid transport and nonzero scale. The
actual final RMS denominator and numerical execution error remain explicit.
Task symmetry supplies a query label separately; optimizer discovery is not proved.
-/

@[expose] public section
noncomputable section

namespace Transformer.Rotate90.ComplexProduct

open Landscape.Neural Grokking.FlatSpectrum.FeatureCodes
open scoped BigOperators

variable {V K H I J : Type*}

/-- Difference of two actual tied embedding rows in real coordinates, TT3. -/
def contrastRow (E : V → K → ℂ) (target v : V) : Channels K → ℝ :=
  fun k => realify (E target) k - realify (E v) k

/-- The real linear functional is exactly the actual raw source contrast,
TT3, including the independent real/complex correspondence. -/
theorem linearScore_contrastRow [Fintype K] (E : V → K → ℂ) (z : K → ℂ)
    (target v : V) :
    linearScore (contrastRow E target v) (realify z) = readout E z target - readout E z v := by
  rw [← realify_readout, ← realify_readout]
  simp only [linearScore, contrastRow, tiedReadout, mul_comm, mul_sub, Finset.sum_sub_distrib]

/-- Choose a valid center-error bound separately for each competitor, TT3.
The choice may depend on the query state. Every candidate must first be a
valid upper bound; no query target error is needed to choose among them. -/
theorem executed_correct_of_center_bounds [Fintype K] [Fintype H]
    {eps : ℝ} (heps : 0 < eps) (w : Weights V K H) (a b target : V)
    (center : Channels K → ℝ) (B : V → J → ℝ) (executed : V → ℝ) {η : ℝ}
    (herr : ∀ v, |executed v - realLogit eps w a b v| ≤ η)
    (hbound : ∀ v, v ≠ target → ∀ j,
      |linearScore (contrastRow w.embedding target v) (realify (state eps w a b)) -
        linearScore (contrastRow w.embedding target v) center| ≤ B v j)
    (hbudget : ∀ v, v ≠ target → ∃ j,
      B v j + 2 * η * denominator eps (state eps w a b) <
        linearScore (contrastRow w.embedding target v) center) :
    ∀ v, v ≠ target → executed v < executed target := by
  intro v hv
  obtain ⟨j, hb⟩ := hbudget v hv
  have h := hbound v hv j
  rw [linearScore_contrastRow] at h
  have hlower := (abs_le.mp h).1
  have hd := denominator_pos heps (state eps w a b)
  have ht := (abs_le.mp (herr target)).1
  have hv' := (abs_le.mp (herr v)).2
  rw [realLogit_eq_readout_div] at ht hv'
  have hraw : 2 * η * denominator eps (state eps w a b) <
      readout w.embedding (state eps w a b) target -
        readout w.embedding (state eps w a b) v := by linarith
  have hn := (lt_div_iff₀ hd).mpr hraw
  rw [sub_div] at hn
  linarith

/-- The Euclidean-center baseline reaches the actual source, TT2–TT3.
The center can be the mean of the training states in a label-preserving orbit. -/
theorem executed_correct_of_euclidean_center [Fintype K] [Fintype H]
    {eps : ℝ} (heps : 0 < eps) (w : Weights V K H) (a b target : V)
    (center : Channels K → ℝ) (executed : V → ℝ) {η : ℝ}
    (herr : ∀ v, |executed v - realLogit eps w a b v| ≤ η)
    (hbudget : ∀ v, v ≠ target →
      Real.sqrt ((∑ k, (realify (state eps w a b) k - center k) ^ 2) *
        ∑ k, contrastRow w.embedding target v k ^ 2) +
          2 * η * denominator eps (state eps w a b) <
            linearScore (contrastRow w.embedding target v) center) :
    ∀ v, v ≠ target → executed v < executed target := by
  apply executed_correct_of_center_bounds heps w a b target center
    (fun v (_ : Unit) => Real.sqrt ((∑ k, (realify (state eps w a b) k - center k) ^ 2) *
      ∑ k, contrastRow w.embedding target v k ^ 2)) executed herr
  · intro v hv j
    exact center_contrast_error_le center _ _
  · intro v hv
    exact ⟨(), hbudget v hv⟩

/-- **A training-state transport budget guarantees the executed decision**,
TT3. All terms controlling the raw margin come from the tied weights, training
states, chosen center and query state. No held-out target error is assumed. -/
theorem executed_correct_of_training_transport [Fintype K] [Fintype H] [Fintype I]
    {eps : ℝ} (heps : 0 < eps) (w : Weights V K H) (a b target : V)
    (T : I → Channels K → ℝ) (center : Channels K → ℝ)
    (coeff : V → I → ℝ) (scale : V → ℝ)
    (hscale : ∀ v, v ≠ target → scale v ≠ 0)
    (executed : V → ℝ) {η : ℝ}
    (herr : ∀ v, |executed v - realLogit eps w a b v| ≤ η)
    (hbudget : ∀ v, v ≠ target →
      Real.sqrt (transportEnergy (coeff v)
        (transportResidual T center (realify (state eps w a b)) (coeff v)) (scale v) *
          trainingContrastEnergy T center (contrastRow w.embedding target v) (scale v)) +
        2 * η * denominator eps (state eps w a b) <
          linearScore (contrastRow w.embedding target v) center) :
    ∀ v, v ≠ target → executed v < executed target := by
  apply executed_correct_of_center_bounds heps w a b target center
    (fun v (_ : Unit) => Real.sqrt (transportEnergy (coeff v)
      (transportResidual T center (realify (state eps w a b)) (coeff v)) (scale v) *
        trainingContrastEnergy T center (contrastRow w.embedding target v) (scale v)))
    executed herr
  · intro v hv j
    exact training_transport_error_le T center _ _ (coeff v) (hscale v hv)
  · intro v hv
    exact ⟨(), hbudget v hv⟩

example := linearScore_contrastRow copyWeights.embedding (state 1 copyWeights 0 0) 0 1

example (q : Fin 2) : ∀ v : Fin 2, v ≠ q →
    realLogit 1 copyWeights q 0 v < realLogit 1 copyWeights q 0 q := by
  apply executed_correct_of_center_bounds (eps := 1) (by norm_num) copyWeights q 0 q
    (realify (state 1 copyWeights q 0)) (fun _ (_ : Unit) => 0)
    (realLogit 1 copyWeights q 0) (η := 0) (by intro v; simp)
  · intro v hv j
    simp
  · intro v hv
    refine ⟨(), ?_⟩
    simp only [mul_zero, zero_mul, add_zero]
    rw [linearScore_contrastRow, copy_gap q v hv]
    positivity

example (q : Fin 2) : ∀ v : Fin 2, v ≠ q →
    realLogit 1 copyWeights q 0 v < realLogit 1 copyWeights q 0 q := by
  apply executed_correct_of_euclidean_center (eps := 1) (by norm_num) copyWeights q 0 q
    (realify (state 1 copyWeights q 0)) (realLogit 1 copyWeights q 0) (η := 0)
    (by intro v; simp)
  intro v hv
  simp only [sub_self, zero_pow (by decide : 2 ≠ 0), Finset.sum_const_zero,
    zero_mul, Real.sqrt_zero, mul_zero, add_zero]
  rw [linearScore_contrastRow, copy_gap q v hv]
  positivity

/-- The exact training-center case instantiates the actual source theorem,
TT5. Nonzero transport residuals are illustrated separately in the generic
training-transport examples; this is not a learned active-product witness. -/
theorem copy_training_transport_classifies (q : Fin 2) : ∀ v : Fin 2, v ≠ q →
    realLogit 1 copyWeights q 0 v < realLogit 1 copyWeights q 0 q := by
  let x := realify (state 1 copyWeights q 0)
  refine executed_correct_of_training_transport (I := Fin 1) (eps := 1) (by norm_num)
    copyWeights q 0 q (fun _ => x) x (fun _ _ => 0) (fun _ => 1)
    (by intro v hv; norm_num) (realLogit 1 copyWeights q 0) (η := 0)
    (by intro v; simp) ?_
  intro v hv
  simp only [x, transportResidual, sub_self, zero_mul, Finset.sum_const_zero,
    transportEnergy, zero_pow (by decide : 2 ≠ 0), zero_div,
    add_zero, mul_zero, Real.sqrt_zero]
  rw [linearScore_contrastRow, copy_gap q v hv]
  positivity

example := copy_training_transport_classifies 0 1 (by decide)

end Transformer.Rotate90.ComplexProduct
