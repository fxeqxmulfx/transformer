module

public import Transformer.Rotate90.ComplexProduct.ApproximateRecovery
public import Transformer.Rotate90.ComplexProduct.SourceWitness
public import Transformer.Grokking.FlatSpectrum.FeatureCodes.AnswerContrast

/-!
# Answer-mass error reaches the actual normalized source classifier

Source: `review/flat_spectrum.md`, *Combined answer contrasts*, AC4–AC5.
The answer representation has signed unit mass. Each competitor keeps its own
decoder coefficient range, ideal gap and source-alignment error. The actual
final RMS denominator and numerical execution error remain in the budget.
Pooling may remove arbitrarily large latent errors. Neither the mass error nor
source alignment is inferred from training accuracy or a fitted probe alone.
-/

@[expose] public section
noncomputable section

namespace Transformer.Rotate90.ComplexProduct

open scoped BigOperators
open Grokking.FlatSpectrum.FeatureCodes

variable {V K H A : Type*} [Fintype K] [Fintype H] [Fintype A] [DecidableEq A]

/-- **A separate combined budget for each actual class contrast**, AC4.
The answer coordinate type can be smaller than the output vocabulary. An error
bound on answer mass suffices; recovering all latent coordinates is unnecessary. -/
theorem executed_correct_of_answer_mass {eps : ℝ} (heps : 0 < eps)
    (w : Weights V K H) (a b target : V) (answer : A)
    (C : V → A → ℝ) (p : A → ℝ) (hp : ∑ j, p j = 1)
    (lo hi α : V → ℝ) {ε η : ℝ}
    (hmass : ∑ j, |p j - if j = answer then 1 else 0| ≤ ε)
    (hrange : ∀ v, v ≠ target → ∀ j,
      lo v ≤ C target j - C v j ∧ C target j - C v j ≤ hi v)
    (halign : ∀ v, v ≠ target →
      |(readout w.embedding (state eps w a b) target -
        readout w.embedding (state eps w a b) v) -
          (answerScore C p target - answerScore C p v)| ≤ α v)
    (executed : V → ℝ) (herr : ∀ v, |executed v - realLogit eps w a b v| ≤ η)
    (hbudget : ∀ v, v ≠ target → α v + ((hi v - lo v) / 2) * ε +
      2 * η * denominator eps (state eps w a b) < C target answer - C v answer) :
    ∀ v, v ≠ target → executed v < executed target := by
  intro v hv
  have hr := hrange v hv answer
  have hnonneg : 0 ≤ (hi v - lo v) / 2 := by linarith
  have hrecover := (answer_contrast_error_le C p answer target v hp (hrange v hv)).trans
    (mul_le_mul_of_nonneg_left hmass hnonneg)
  have he := contrast_error_le_of_alignment (halign v hv) hrecover
  have hlower := (abs_le.mp he).1
  have hb := hbudget v hv
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

/-- A nonzero answer-mass error in the concrete two-label source witness,
AC5. Both recovery and source alignment consume margin. This remains a
residual-path witness, not an active-product or optimizer-discovery theorem. -/
theorem copy_answer_mass_classifies (q : Fin 2) : ∀ v : Fin 2, v ≠ q →
    realLogit 1 copyWeights q 0 v < realLogit 1 copyWeights q 0 q := by
  let r : ℝ := (Real.sqrt 2)⁻¹
  let C : Fin 2 → Fin 2 → ℝ := fun v j =>
    (if v = 0 then 1 else -1) * (if j = 0 then 1 else -1) * r
  let p : Fin 2 → ℝ := fun j => if j = q then 7 / 8 else 1 / 8
  have hr : 0 < r := by dsimp [r]; positivity
  have hsource (j v : Fin 2) :
      readout copyWeights.embedding (state 1 copyWeights j 0) v = C v j := by
    rw [copy_readout]
    rfl
  refine executed_correct_of_answer_mass (eps := 1) (by norm_num) copyWeights q 0 q q C p
    (by fin_cases q <;> norm_num [p, Fin.sum_univ_two])
    (fun _ => -(2 * r)) (fun _ => 2 * r)
    (fun _ => r / 2) (ε := 1 / 4) (η := 0) ?_ ?_ ?_
    (realLogit 1 copyWeights q 0) (by intro v; simp) ?_
  · fin_cases q <;> norm_num [p, Fin.sum_univ_two]
  · intro v hv j
    fin_cases q <;> fin_cases v <;> fin_cases j <;>
      norm_num [C] at * <;> constructor <;> linarith
  · intro v hv
    rw [hsource, hsource]
    fin_cases q <;> fin_cases v <;> norm_num [C, p, answerScore, Fin.sum_univ_two] at * <;>
      apply abs_le.mpr <;> constructor <;> linarith
  · intro v hv
    fin_cases q <;> fin_cases v <;> norm_num [C] at * <;> linarith

example := copy_answer_mass_classifies 0 1 (by decide)

end Transformer.Rotate90.ComplexProduct
