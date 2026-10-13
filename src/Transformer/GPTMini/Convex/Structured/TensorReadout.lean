import Transformer.GPTMini.Convex.Structured.TensorHeads

/-!
# Genuine residual output and tied Euclidean readout

Source: GPTMini.Model.unembed's actual inner product, GPTMini.RMSNorm's
final normalization, and the compact mixed tensor head at 5bdff3c.
This proposal writes inferred means into ten genuine residual axes.
Its final attention contribution subtracts the internally recovered
raw row so the ordinary residual addition leaves exactly that output.
FFN is deferred; this module does not modify the tied readout formula.

The full token embedding still contains all free learned Q/K/value
fields. Their contribution to final logits vanishes because the actual
output is supported on the separate code axes, not because the weights
are frozen. Final RMSNorm introduces its derived positive common scale;
actual greedy decoding therefore matches the verified mixed head for
all parameters. Causal full-array blocks and stack/loss integration
remain separate obligations.
-/

namespace Transformer.GPTMini.Convex.Structured

open Transformer.GPTMini.TokenInterface
open scoped BigOperators Classical
noncomputable section

variable {V C d T : ℕ}

/-- Actual Euclidean output writes all ten inferred scalar means into the physical decoder axes.
Source: TensorEmbedding's code-axis layout and the unchanged residual vector interface. -/
def tensorCodeOutput (hwidth : 64 ≤ d) (coordinates : Fin 10 → ℝ) : EucSpace d :=
  ∑ axis, EuclideanSpace.single (tensorCodeAxis hwidth axis) (coordinates axis)

/-- Distinct output coordinates occupy distinct actual residual axes.
Source: the physical shift by 52, without aliasing learned input or protected axes. -/
theorem tensorCodeAxis_injective (hwidth : 64 ≤ d) : Function.Injective (tensorCodeAxis hwidth) := by
  intro a b he
  have hv := congrArg Fin.val he
  simp only [tensorCodeAxis] at hv
  apply Fin.ext
  omega

example : (64 : ℕ) ≤ 128 := by omega

/-- Reading the real output vector evaluates its finite sum of physical coordinate writes.
Source: genuine Euclidean standard-basis vectors and finite additive coordinate evaluation. -/
theorem tensorCodeOutput_apply (hwidth : 64 ≤ d) (coordinates : Fin 10 → ℝ) (position : Fin d) :
    tensorCodeOutput hwidth coordinates position =
      ∑ axis, if position = tensorCodeAxis hwidth axis then coordinates axis else 0 := by
  change WithLp.ofLp (∑ axis, EuclideanSpace.single (tensorCodeAxis hwidth axis) (coordinates axis)) position = _
  rw [WithLp.ofLp_sum]
  simp only [Finset.sum_apply, PiLp.single_apply]

example : (64 : ℕ) ≤ 64 := by omega

/-- Every decoder axis of the actual residual output is exactly its computed head coordinate.
Source: injectivity of the physical output-axis assignment and real standard-basis addition. -/
theorem tensorCodeOutput_code (hwidth : 64 ≤ d) (coordinates : Fin 10 → ℝ) (position : Fin 10) :
    tensorCodeOutput hwidth coordinates (tensorCodeAxis hwidth position) = coordinates position := by
  rw [tensorCodeOutput_apply]
  rw [Finset.sum_eq_single position]
  · simp only [ite_true]
  · intro other hother hne
    have haxes : tensorCodeAxis hwidth position ≠ tensorCodeAxis hwidth other := by
      intro he
      exact hne ((tensorCodeAxis_injective hwidth he).symm)
    exact ite_eq_right haxes
  · intro hnot
    exact (hnot (Finset.mem_univ position)).elim

example : (64 : ℕ) ≤ 128 := by omega

/-- All learned token fields, protected anchor, position and spare axes are zero in the actual final output.
Source: the real ten-axis support, derived rather than supplied as a correct-readout premise. -/
theorem tensorCodeOutput_offcode (hwidth : 64 ≤ d) (coordinates : Fin 10 → ℝ) (position : Fin d)
    (hoff : position.val < 52 ∨ 62 ≤ position.val) : tensorCodeOutput hwidth coordinates position = 0 := by
  rw [tensorCodeOutput_apply]
  apply Finset.sum_eq_zero
  intro axis haxis
  have hne : position ≠ tensorCodeAxis hwidth axis := by
    intro he
    have hv := congrArg Fin.val he
    have ha := axis.isLt
    simp only [tensorCodeAxis] at hv
    omega
  exact ite_eq_right hne

example : (64 : ℕ) ≤ 64 ∧ ((63 : Fin 64).val < 52 ∨ 62 ≤ (63 : Fin 64).val) := by omega

/-- The actual tied inner product contains exactly the ten inferred means, for every free token table.
Source: GPTMini.Model.unembed's unchanged inner product and TensorEmbedding's true readonly decoder coordinates. -/
theorem tensorCodeOutput_tied (hsize : V ≤ 1024) (hwidth : 64 ≤ d) (θ : SharedParameters V C)
    (coordinates : Fin 10 → ℝ) (token : Fin V) :
    inner (𝕜 := ℝ) (tensorCodeOutput hwidth coordinates) (tensorEmbedding hsize θ token) =
      ∑ axis, coordinates axis * outputCoordinate (outputDigit (vocabularyCode hsize token)) axis := by
  rw [tensorCodeOutput, sum_inner]
  simp only [EuclideanSpace.inner_single_left, conj_trivial, tensorEmbedding_code]

example : (548 : ℕ) ≤ 1024 ∧ (64 : ℕ) ≤ 128 := by omega

/-- The genuine final RMSNorm contributes its actual common multiplier to the unchanged tied readout.
Source: GPTMini.RMSNorm.rmsNormEps and the derived true Euclidean output/code dot product. -/
theorem tensorCodeOutput_rms_tied (hsize : V ≤ 1024) (hwidth : 64 ≤ d) (eps : ℝ)
    (θ : SharedParameters V C) (coordinates : Fin 10 → ℝ) (token : Fin V) :
    inner (𝕜 := ℝ) (rmsNormEps eps (tensorCodeOutput hwidth coordinates)) (tensorEmbedding hsize θ token) =
      (Real.sqrt (d : ℝ) / Real.sqrt (‖tensorCodeOutput hwidth coordinates‖ ^ 2 + (d : ℝ) * eps)) *
        (∑ axis, coordinates axis * outputCoordinate (outputDigit (vocabularyCode hsize token)) axis) := by
  rw [rmsNormEps, real_inner_smul_left, tensorCodeOutput_tied]

example : (68 : ℕ) ≤ 1024 ∧ (64 : ℕ) ≤ 64 := by omega

/-- True final attention subtraction cancels the raw row through ordinary residual addition.
Source: GPTMini.Block.forward's unchanged residual and exact internal recovery from actual positive-epsilon prenorm. -/
theorem tensorFinalResidual (hwidth : 64 ≤ d) (eps : ℝ) (heps : 0 < eps) (x : EucSpace d)
    (hanchor : x (tensorAnchorAxis hwidth) = 1) (coordinates : Fin 10 → ℝ) :
    x + (tensorCodeOutput hwidth coordinates -
      tensorAnchorRecovery (tensorAnchorAxis hwidth) (rmsNormEps eps x)) = tensorCodeOutput hwidth coordinates := by
  rw [tensorAnchorRecovery_rms (by omega) eps heps _ x hanchor]
  abel

example : (64 : ℕ) ≤ 64 ∧ (0 : ℝ) < 1 / 100000 ∧
    (WithLp.toLp 2 (fun _ : Fin 64 => (1 : ℝ)) : EucSpace 64) (tensorAnchorAxis (by omega)) = 1 := by
  exact ⟨by omega, by norm_num, rfl⟩

/-- Actual final-RMS tied logits from the real mixed tensor output equal its inferred score times the true scale.
Source: genuine output support/readout and TensorHeads' ten-coordinate compact mixture identity. -/
theorem tensorMixedOutput_rms_tied (hsize : V ≤ 1024) (hwidth : 64 ≤ d) (eps : ℝ)
    (θ : SharedParameters V C) (ψ : TensorHeadParameters C) (x : Fin T → EucSpace d)
    (hcap : T ≤ C) (query : Fin T) (token : Fin V) :
    inner (𝕜 := ℝ) (rmsNormEps eps (tensorCodeOutput hwidth (tensorMixedCoordinates hwidth ψ x hcap query)))
      (tensorEmbedding hsize θ token) =
      (Real.sqrt (d : ℝ) /
        Real.sqrt (‖tensorCodeOutput hwidth (tensorMixedCoordinates hwidth ψ x hcap query)‖ ^ 2 + (d : ℝ) * eps)) *
          tensorMixedScore hwidth ψ x hcap query (vocabularyCode hsize token) := by
  rw [tensorCodeOutput_rms_tied, ← tensorMixedScore_axes]

example : (36 : ℕ) ≤ 1024 ∧ (64 : ℕ) ≤ 128 ∧ (3 : ℕ) ≤ 128 := by omega

/-- Greedy decoding of genuine final-RMS tied tensor logits is the verified mixed greedy decoder for all weights.
Source: exact real prenorm inference transfer and positivity of the actual final RMS scale, including tied logits. -/
theorem tensorMixedOutput_best (hV : 0 < V) (hsize : V ≤ 1024) (hwidth : 64 ≤ d)
    (eps : ℝ) (heps : 0 < eps) (θ : BindingParameters V C) (tokens : List (Fin V))
    (hcap : tokens.length ≤ C) (query : Fin tokens.length) :
    bestToken hV (fun token => inner (𝕜 := ℝ)
      (rmsNormEps eps (tensorCodeOutput hwidth (tensorMixedCoordinates hwidth (tensorHeadParameters θ)
        (tensorSequence hsize hwidth eps θ tokens hcap) hcap query))) (tensorEmbedding hsize θ.1 token)) =
      mixedGreedy hV hsize θ tokens hcap query := by
  simp_rw [tensorMixedOutput_rms_tied, tensorMixedScore_sequence hsize hwidth eps heps]
  apply mixedGreedy_positive_scale
  exact tensorRMSScale_pos (by omega) eps heps _

example : (0 : ℕ) < 548 ∧ (548 : ℕ) ≤ 1024 ∧ (64 : ℕ) ≤ 64 ∧
    (0 : ℝ) < 1 / 100000 ∧ ([1, 36, 292, 36] : List (Fin 548)).length ≤ 64 := by
  exact ⟨by omega, by omega, by omega, by norm_num, by decide⟩

end
end Transformer.GPTMini.Convex.Structured
