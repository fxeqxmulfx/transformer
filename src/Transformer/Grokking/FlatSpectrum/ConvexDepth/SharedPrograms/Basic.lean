module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.ContextMixture.Loss
public import Transformer.Grokking.FlatSpectrum.ConvexDepth.FourierCode

/-!
# Shared categorical programs with a complete complex copy head

Source: `review/convex/moments.md`, CM52. Each fixed program reads an input and
returns a vocabulary symbol. The shared identity decoder is an explicit copy
prior, not a consequence of flatness. Only mixture coefficients are learned;
the feature bank and the full conjugate Fourier head stay fixed.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.SharedPrograms

open scoped BigOperators
open Landscape.Neural (crossEntropy)

variable {J X C : Type*} [Fintype J] [Fintype C] [DecidableEq C]

/-- A program indexes the full complex vocabulary code, CM52. No observed
label or target function is an argument of this fixed feature map. -/
def bank (U : Matrix C C ℂ) (program : J → X → C) (j : J) : C × Bool → X → ℝ :=
  DecoderConfidence.features U (program j)

/-- The canonical shared copy decoder uses every raw vocabulary class, CM52.
Both complex coordinates are retained, including classes absent from training. -/
noncomputable def copyHead (U : Matrix C C ℂ) (margin : ℝ) : C × Bool → C → ℝ :=
  complexCoefficients (complexInitializer U id margin)

/-- Actual complete complex products are the weighted categorical votes of
the input programs, CM52. This holds for arbitrary real mixture coefficients. -/
theorem mixed_logits (U : Matrix C C ℂ) (hU : U ∈ Matrix.unitaryGroup C ℂ)
    (program : J → X → C) (a : J → ℝ) (margin : ℝ) (x : X) (c : C) :
    headLogits (ContextMixture.features (bank U program) a) (copyHead U margin) x c =
      ∑ j, a j * if c = program j x then margin else 0 := by
  rw [ContextMixture.logits]
  simp only [Finset.sum_apply, Pi.smul_apply, smul_eq_mul]
  apply Finset.sum_congr rfl
  intro j _
  change a j * headLogits (DecoderConfidence.features U (program j))
    (complexCoefficients (complexInitializer U id margin)) x c = _
  rw [DecoderConfidence.logits U hU]
  rfl

/-- The actual initialized tensor obeys the existing sufficient column
budget, CM52. This charges both real and imaginary entries of every class. -/
theorem copyHead_budget (U : Matrix C C ℂ) (hU : U ∈ Matrix.unitaryGroup C ℂ)
    (margin : ℝ) : copyHead U margin ∈ headBudget (2 * |margin| * Fintype.card C) := by
  exact complexInitializer_budget U hU id margin

/-- Flat-source coordinate bounds survive the program mixture, CM52. The
empirical task matrix need not be flat and programs may repeat code states. -/
theorem mixed_features_bound (U : Matrix C C ℂ) (hU : U ∈ Matrix.unitaryGroup C ℂ)
    (program : J → X → C) {a : J → ℝ} (ha : a ∈ ContextMixture.simplex)
    (x : X) (k : C × Bool) :
    |ContextMixture.features (bank U program) a k x| ≤ 1 := by
  apply ContextMixture.features_bound _ ha x
  intro j k
  exact DecoderConfidence.features_bound U hU (program j) x k

/-- Ordinary full-vocabulary CE is convex in the learned mixture, CM52.
The canonical head is fixed; this is not joint head/mixture factor training. -/
theorem ce_convex (U : Matrix C C ℂ) (program : J → X → C) (margin : ℝ)
    (x : X) (y : C) :
    ConvexOn ℝ ContextMixture.simplex (fun a => crossEntropy
      (headLogits (ContextMixture.features (bank U program) a) (copyHead U margin) x) y) := by
  exact ContextMixture.ce_convex _ _ _ _

/-- Agreement of all nonzero-weight programs gives the exact declared logits,
CM52. Identifying their agreed symbol with the unknown truth is a separate step. -/
theorem agreed_logits (U : Matrix C C ℂ) (hU : U ∈ Matrix.unitaryGroup C ℂ)
    (program : J → X → C) (a : J → ℝ) (hs : ∑ j, a j = 1) (margin : ℝ)
    (x : X) (y : C) (hag : ∀ j, a j ≠ 0 → program j x = y) :
    headLogits (ContextMixture.features (bank U program) a) (copyHead U margin) x =
      fun c => if c = y then margin else 0 := by
  funext c
  rw [mixed_logits U hU]
  calc
    _ = ∑ j, a j * (if c = y then margin else 0) := by
      apply Finset.sum_congr rfl
      intro j _
      by_cases hz : a j = 0
      · simp [hz]
      · rw [hag j hz]
    _ = _ := by rw [← Finset.sum_mul, hs, one_mul]

/-- Exact ordinary CE follows for the agreed raw class, CM52. This is an
actual full complex prediction, not a surrogate vote loss. -/
theorem agreed_ce (U : Matrix C C ℂ) (hU : U ∈ Matrix.unitaryGroup C ℂ)
    (program : J → X → C) (a : J → ℝ) (hs : ∑ j, a j = 1) (margin : ℝ)
    (x : X) (y : C) (hag : ∀ j, a j ≠ 0 → program j x = y) :
    crossEntropy
      (headLogits (ContextMixture.features (bank U program) a) (copyHead U margin) x) y =
      Real.log (1 + ((Fintype.card C : ℝ) - 1) * Real.exp (-margin)) := by
  rw [agreed_logits U hU program a hs margin x y hag]
  exact fixed_margin_ce _ _

/-- Every competing class has a strict smaller actual logit when the shared
agreement margin is positive, CM52. -/
theorem agreed_strict (U : Matrix C C ℂ) (hU : U ∈ Matrix.unitaryGroup C ℂ)
    (program : J → X → C) (a : J → ℝ) (hs : ∑ j, a j = 1) {margin : ℝ}
    (hm : 0 < margin) (x : X) (y : C) (hag : ∀ j, a j ≠ 0 → program j x = y)
    {c : C} (hc : c ≠ y) :
    headLogits (ContextMixture.features (bank U program) a) (copyHead U margin) x c <
      headLogits (ContextMixture.features (bank U program) a) (copyHead U margin) x y := by
  rw [agreed_logits U hU program a hs margin x y hag]
  simpa [hc] using hm

/-- A vocabulary renaming commutes with the actual shared readout whenever
it commutes with each input program, CM52. This holds for every learned mixture. -/
theorem logits_equivariant (U : Matrix C C ℂ) (hU : U ∈ Matrix.unitaryGroup C ℂ)
    (program : J → X → C) (a : J → ℝ) (margin : ℝ) (rename : X → X) (σ : C ≃ C)
    (x : X) (hp : ∀ j, program j (rename x) = σ (program j x)) (c : C) :
    headLogits (ContextMixture.features (bank U program) a) (copyHead U margin)
        (rename x) (σ c) =
      headLogits (ContextMixture.features (bank U program) a) (copyHead U margin) x c := by
  rw [mixed_logits U hU, mixed_logits U hU]
  apply Finset.sum_congr rfl
  intro j _
  rw [hp j]
  simp only [Equiv.apply_eq_iff_eq]

example := mixed_logits fourier4 fourier4_unitary
  (fun _ : Unit => id) (fun _ => 1) 24 (2 : Fin 4) 2

example := copyHead_budget fourier4 fourier4_unitary 24

example := mixed_features_bound fourier4 fourier4_unitary
  (fun _ : Unit => id) (a := fun _ => 1) (by constructor <;> simp)
  (2 : Fin 4) (1, true)

example := ce_convex fourier4 (fun _ : Unit => id) 24 (2 : Fin 4) 2

example := agreed_logits fourier4 fourier4_unitary
  (fun _ : Unit => id) (fun _ => 1) (by simp) 24 (2 : Fin 4) 2 (by simp)

example := agreed_ce fourier4 fourier4_unitary
  (fun _ : Unit => id) (fun _ => 1) (by simp) 24 (2 : Fin 4) 2 (by simp)

example := agreed_strict fourier4 fourier4_unitary
  (fun _ : Unit => id) (fun _ => 1) (by simp) (margin := 24) (by norm_num)
  (2 : Fin 4) 2 (by simp) (c := 0) (by decide)

example := logits_equivariant fourier4 fourier4_unitary
  (fun _ : Unit => id) (fun _ => 1) 24 (Equiv.swap (0 : Fin 4) 2)
  (Equiv.swap 0 2) (0 : Fin 4) (by simp) 0

end Transformer.Grokking.FlatSpectrum.ConvexDepth.SharedPrograms
