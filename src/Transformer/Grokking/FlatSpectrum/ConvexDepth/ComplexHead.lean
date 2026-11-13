module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.JointHead
public import Transformer.Grokking.FlatSpectrum.ConvexDepth.Transport

/-!
# Actual complex categorical heads and conjugate initialization

Source: `review/flat_spectrum.md`, CFt5. Native causal pairs at `7ce28651`
use both complex coordinates in prediction. The label assignment here is a
recorded dictionary (including its explicit default), not an unseen target.
The feature code is fixed; CD8 applies to all real coefficient coordinates.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth

open scoped BigOperators
open Matrix
open Landscape.Neural (crossEntropy)

variable {I C : Type*} [Fintype I]

/-- The actual realification of a fixed complex feature row, CFt5. -/
def complexFeatures (U : Matrix I I ℂ) : I × Bool → I → ℝ :=
  fun k s => if k.2 then -(U s k.1).im else (U s k.1).re

/-- Both real and imaginary head coordinates are independent CD8 parameters,
CFt5. This map imposes no factorization of the learned tensor. -/
def complexCoefficients (H : Matrix I C ℂ) : I × Bool → C → ℝ :=
  fun k c => if k.2 then (H k.1 c).im else (H k.1 c).re

/-- The native real/imaginary dot product is the real part of the actual
complex matrix product, CFt5. Neither coordinate is decorative. -/
theorem complex_head_logits (U : Matrix I I ℂ) (H : Matrix I C ℂ) (s : I) (c : C) :
    headLogits (complexFeatures U) (complexCoefficients H) s c = ((U * H) s c).re := by
  simp only [headLogits, Fintype.sum_prod_type, Fintype.sum_bool,
    complexFeatures, complexCoefficients, Bool.false_eq_true, ite_false, ite_true,
    Matrix.mul_apply, Complex.re_sum, Complex.mul_re]
  apply Finset.sum_congr rfl
  intro k hk
  ring

omit [Fintype I] in
/-- Every real coefficient tensor is represented, so complex notation does
not shrink the convex CD8 head domain (CFt5). -/
theorem complexCoefficients_surjective (W : I × Bool → C → ℝ) :
    ∃ H : Matrix I C ℂ, complexCoefficients H = W := by
  refine ⟨fun k c => ⟨W (k, false) c, W (k, true) c⟩, ?_⟩
  funext k c
  rcases k with ⟨i, b⟩
  cases b <;> rfl

/-- The ordinary CE objective is convex in all actual real and imaginary
coordinates with a fixed code, CFt5 and CD8. -/
theorem complex_headLoss_convex [Fintype C] {S : Type*} [Fintype S]
    (U : Matrix I I ℂ) (inputs : S → I) (labels : S → C) (R : ℝ) :
    ConvexOn ℝ (headBudget R) (headLoss (complexFeatures U) inputs labels) :=
  headLoss_convex _ _ _ _

/-- Every entry of a unitary categorical code has modulus at most one,
CFt5. The bound follows from its actual unit-energy columns. -/
theorem unitary_code_entry_norm [DecidableEq I] (U : Matrix I I ℂ)
    (hU : U ∈ Matrix.unitaryGroup I ℂ) (s k : I) : ‖U s k‖ ≤ 1 := by
  have hg : U.conjTranspose * U = 1 := Unitary.star_mul_self_of_mem hU
  have hc : ∑ i, ‖U i k‖^2 = 1 := by
    have h := congrArg (fun A : Matrix I I ℂ => (A k k).re) hg
    rw [conjTranspose_mul_self_apply_self, Complex.ofReal_re] at h
    simpa using h
  have h := Finset.single_le_sum (f := fun i : I => ‖U i k‖^2)
    (fun i _ => sq_nonneg _) (Finset.mem_univ s)
  rw [hc] at h
  nlinarith [norm_nonneg (U s k)]

/-- Both actual real feature coordinates obey the unitary code's bound,
CFt5; the imaginary sign does not affect its absolute value. -/
theorem complexFeatures_bound [DecidableEq I] (U : Matrix I I ℂ)
    (hU : U ∈ Matrix.unitaryGroup I ℂ) (k : I × Bool) (s : I) :
    |complexFeatures U k s| ≤ 1 := by
  rcases k with ⟨i, b⟩
  cases b
  · exact (Complex.abs_re_le_norm _).trans (unitary_code_entry_norm U hU s i)
  · simpa [complexFeatures] using
      (Complex.abs_im_le_norm _).trans (unitary_code_entry_norm U hU s i)

/-- The fixed code bounds logits of every feasible learned head, not just
the initializer, CFt5. This is the capacity use of flatness in CD8. -/
theorem complex_headLogits_bound [DecidableEq I] (U : Matrix I I ℂ)
    (hU : U ∈ Matrix.unitaryGroup I ℂ) {W : I × Bool → C → ℝ} {R : ℝ}
    (hW : W ∈ headBudget R) (s : I) (c : C) :
    |headLogits (complexFeatures U) W s c| ≤ R := by
  simpa only [mul_one] using headLogits_bound (complexFeatures U) hW
    (M := 1) (by norm_num) s (fun k => complexFeatures_bound U hU k s) c

variable [DecidableEq C]

/-- Explicit state margins from the recorded assignment; no answer rule is
used by the code matrix, CFt5. -/
def stateMargins (labels : I → C) (a : ℝ) : Matrix I C ℂ :=
  fun s c => if c = labels s then (a : ℂ) else 0

/-- The actual conjugate-transform initializer, matching the finite native
sum over recorded states in `model/initialize.rs` at `7ce28651` (CFt5). -/
noncomputable def complexInitializer (U : Matrix I I ℂ) (labels : I → C) (a : ℝ) :
    Matrix I C ℂ := U.conjTranspose * stateMargins labels a

/-- Expanded initializer coordinates show exactly which observations and
complex code entries the construction uses, CFt5. -/
theorem complexInitializer_apply (U : Matrix I I ℂ) (labels : I → C) (a : ℝ)
    (k : I) (c : C) :
    complexInitializer U labels a k c =
      ∑ s, star (U s k) * if c = labels s then (a : ℂ) else 0 := by
  rw [complexInitializer, Matrix.mul_apply]
  apply Finset.sum_congr rfl
  intro s hs
  rw [Matrix.conjTranspose_apply, stateMargins]

variable [DecidableEq I]

/-- Unitary inversion reproduces the declared margins in the actual head,
CFt5. Unitarity is later proved for every normalized Fourier query block. -/
theorem complexInitializer_logits (U : Matrix I I ℂ)
    (hU : U ∈ Matrix.unitaryGroup I ℂ) (labels : I → C) (a : ℝ) (s : I) :
    headLogits (complexFeatures U) (complexCoefficients (complexInitializer U labels a)) s =
      fun c => if c = labels s then a else 0 := by
  have hu : U * U.conjTranspose = 1 := Unitary.mul_star_self_of_mem hU
  funext c
  rw [complex_head_logits, complexInitializer, ← Matrix.mul_assoc, hu, Matrix.one_mul]
  by_cases h : c = labels s <;> simp [stateMargins, h]

/-- Exact usual full-vocabulary CE at every declared categorical state,
CFt5. It is not a held-out identification certificate. -/
theorem complexInitializer_ce [Fintype C] (U : Matrix I I ℂ)
    (hU : U ∈ Matrix.unitaryGroup I ℂ) (labels : I → C) (a : ℝ) (s : I) :
    crossEntropy
      (headLogits (complexFeatures U) (complexCoefficients (complexInitializer U labels a)) s)
      (labels s) = Real.log (1 + ((Fintype.card C : ℝ) - 1) * Real.exp (-a)) := by
  rw [complexInitializer_logits U hU]
  exact fixed_margin_ce _ _

/-- Positive margins make the assigned class uniquely preferred, CFt5.
The assigned class may be an explicit default rather than the unknown truth. -/
theorem complexInitializer_strict (U : Matrix I I ℂ)
    (hU : U ∈ Matrix.unitaryGroup I ℂ) (labels : I → C) {a : ℝ} (ha : 0 < a)
    (s : I) {c : C} (hc : c ≠ labels s) :
    headLogits (complexFeatures U) (complexCoefficients (complexInitializer U labels a)) s c <
      headLogits (complexFeatures U) (complexCoefficients (complexInitializer U labels a)) s
        (labels s) := by
  rw [complexInitializer_logits U hU]
  simpa [hc] using ha

example := complexInitializer_logits fourier4 fourier4_unitary
  (fun i : Fin 4 => decide (i.val < 2)) 24 0

example := complexInitializer_ce fourier4 fourier4_unitary
  (fun i : Fin 4 => decide (i.val < 2)) 24 0

example := complexInitializer_strict fourier4 fourier4_unitary
  (fun _ : Fin 4 => true) (a := 24) (by norm_num) 0 (c := false) (by decide)

example := unitary_code_entry_norm fourier4 fourier4_unitary 0 1

example := complexFeatures_bound fourier4 fourier4_unitary (0, true) 1

example := complex_headLogits_bound fourier4 fourier4_unitary
  (C := Bool) (W := 0) (R := 1) (zero_mem_headBudget (by norm_num)) 0 true

end Transformer.Grokking.FlatSpectrum.ConvexDepth
