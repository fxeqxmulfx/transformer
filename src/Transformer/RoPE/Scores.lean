module

public import Transformer.RoPE.Basic
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv

/-!
# Content-dependent rotary scores

Wu, Du and Peng, arXiv:2609.39929v1, Appendix A.2, Proposition 2; Men et al.,
arXiv:2405.14591v1, §4.2. Query and key vectors remain explicit: the scalar
contrast objective alone does not describe an attention head's score.
-/

@[expose] public section

open scoped BigOperators

namespace Transformer.RoPE

/-- One frequency's score with its content-dependent cosine/sine coefficients. -/
noncomputable def score (a b omega distance : ℝ) : ℝ :=
  a * Real.cos (omega * distance) + b * Real.sin (omega * distance)

/-- Rotate the query through relative angle `omega * distance` and dot it
with the key. This uses the query-minus-key relative-position convention. -/
noncomputable def pairScore (query key : ℝ × ℝ) (omega distance : ℝ) : ℝ :=
  (query.1 * Real.cos (omega * distance) - query.2 * Real.sin (omega * distance)) * key.1 +
  (query.1 * Real.sin (omega * distance) + query.2 * Real.cos (omega * distance)) * key.2

/-- Proposition 2, real-coordinate version of the complex score reduction. -/
theorem pair_score_expansion (query key : ℝ × ℝ) (omega distance : ℝ) :
    pairScore query key omega distance = score
      (query.1 * key.1 + query.2 * key.2)
      (query.1 * key.2 - query.2 * key.1) omega distance := by
  unfold pairScore score
  ring

/-- Proposition 2, converse: every pair of real spectral coefficients can
actually be realized by query/key vectors. -/
theorem every_score_realizable (a b omega : ℝ) :
    ∃ query key : ℝ × ℝ, ∀ distance,
      pairScore query key omega distance = score a b omega distance := by
  refine ⟨(1, 0), (a, b), fun distance => ?_⟩
  rw [pair_score_expansion]
  simp

/-- Proposition 2 extends to whole fully rotary heads with arbitrary spectral
amplitudes; the same query/key construction works simultaneously at all offsets. -/
theorem every_head_realizable {n : ℕ} (a b omega : Fin n → ℝ) :
    ∃ query key : Fin n → ℝ × ℝ, ∀ distance,
      ∑ i, pairScore (query i) (key i) (omega i) distance =
        ∑ i, score (a i) (b i) (omega i) distance := by
  refine ⟨fun _ => (1, 0), fun i => (a i, b i), fun distance => ?_⟩
  apply Finset.sum_congr rfl
  intro i _
  rw [pair_score_expansion]
  simp

/-- The rotating scalar score and its quadrature have constant total squared
amplitude. This identity underlies the bandwise norm bounds in Appendix A. -/
theorem score_quadrature (a b omega distance : ℝ) :
    score a b omega distance ^ 2 + score (-b) a omega distance ^ 2 = a ^ 2 + b ^ 2 := by
  unfold score
  nlinarith [Real.sin_sq_add_cos_sq (omega * distance),
    sq_nonneg (a * Real.cos (omega * distance) + b * Real.sin (omega * distance)),
    sq_nonneg (-b * Real.cos (omega * distance) + a * Real.sin (omega * distance))]

/-- Appendix A: a score is bounded by the magnitude of its complex coefficient. -/
theorem score_amplitude_bound (a b omega distance : ℝ) :
    |score a b omega distance| ≤ Real.sqrt (a ^ 2 + b ^ 2) := by
  apply Real.abs_le_sqrt
  have h := score_quadrature a b omega distance
  nlinarith [sq_nonneg (score (-b) a omega distance)]

/-- Appendix A: the finite head score is bounded without assuming uniform
query/key norms or a Gaussian activation distribution. -/
theorem head_amplitude_bound {n : ℕ} (a b omega : Fin n → ℝ) (distance : ℝ) :
    |∑ i, score (a i) (b i) (omega i) distance| ≤
      ∑ i, Real.sqrt (a i ^ 2 + b i ^ 2) := by
  exact (Finset.abs_sum_le_sum_abs _ _).trans
    (Finset.sum_le_sum fun i _ => score_amplitude_bound _ _ _ _)

end Transformer.RoPE
