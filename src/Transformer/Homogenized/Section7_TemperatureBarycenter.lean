module

public import Transformer.Homogenized.Section7_TemperatureSoftmax
public import Transformer.Homogenized.AttnAverage
public import Transformer.Homogenized.SimplexBary

/-!
# Actual temperature derivatives of the empirical softmax barycenter

ArXiv:2604.01978v1, §7, proof of `thm:clustering_small_beta`,
the first and second barycenter derivatives in part (b) of
`eq:s_mu_small_beta_clean`. The normalized finite sum below is
identified with the original empirical barycenter at its original
query scores. Its derivative and second derivative are genuine
derivatives of that same sum.

For unit tokens and scores bounded by `M`, their norms are bounded
by `2 M` and `6 M²`. Both estimates sum the original softmax weights
before bounding, so their constants are independent of token count.
For an original query the score bound can be `‖A x‖`, avoiding the
larger Frobenius norm of the entire matrix in the later Gaussian
second-moment calculation. No symmetry, Gaussian moment or Taylor
remainder is postulated by these definitions or estimates.
-/

@[expose] public section

open Real
open scoped BigOperators

namespace Transformer.Homogenized

/-- The actual normalized finite barycenter at temperature `t`, with
fixed original tokens and scores. Source: arXiv:2604.01978v1, §7,
the definition of `w_k` in `eq:s_mu_small_beta_clean`. -/
noncomputable def temperatureBarycenter {d n : ℕ} (X : Idx n → EucSpace d)
    (a : Idx n → ℝ) (t : ℝ) : EucSpace d :=
  ∑ k : Idx n, Perspective.softmaxWeight (fun l => t * a l) k • X k

/-- At the genuine query scores this is the original empirical
barycenter. Source: arXiv:2604.01978v1, §7, part (b) of
`eq:s_mu_small_beta_clean`, with the original `A x` convention. -/
theorem temperatureBarycenter_eq_softBary {d n : ℕ} (X : Idx n → EucSpace d)
    (A : Matrix (Fin d) (Fin d) ℝ) (i : Idx n) (t : ℝ) :
    temperatureBarycenter X (fun k => inner ℝ (Matrix.toEuclideanLin A (X i)) (X k)) t =
      softBary t A (empMeasure X) (X i) := by
  rw [softBary_empMeasure]
  rfl

/-- The original query norm bounds every original token score.
Source: arXiv:2604.01978v1, §7, `|⟨A x,y⟩| ≤ ‖A‖` in part (b),
with the sharper query norm retained before the Gaussian expectation. -/
theorem abs_query_token_score_le_norm {d n : ℕ} (X : Idx n → EucSpace d)
    (hX : ∀ k, ‖X k‖ = 1) (A : Matrix (Fin d) (Fin d) ℝ) (i k : Idx n) :
    |inner ℝ (Matrix.toEuclideanLin A (X i)) (X k)| ≤ ‖Matrix.toEuclideanLin A (X i)‖ := by
  simpa only [hX, mul_one] using abs_real_inner_le_norm (Matrix.toEuclideanLin A (X i)) (X k)

/-- Unit tokens control an arbitrary original scalar-weighted sum
by its absolute weights. Source: arXiv:2604.01978v1, §7,
the triangle inequality in the barycenter derivative estimate. -/
theorem norm_sum_smul_unit_le_sum_abs {d n : ℕ} (X : Idx n → EucSpace d)
    (hX : ∀ k, ‖X k‖ = 1) (c : Idx n → ℝ) :
    ‖∑ k : Idx n, c k • X k‖ ≤ ∑ k : Idx n, |c k| := by
  apply (norm_sum_le _ _).trans_eq
  simp only [norm_smul, Real.norm_eq_abs, hX, mul_one]

/-- The original finite barycenter has the genuine sum of the
temperature derivatives. Source: arXiv:2604.01978v1, §7,
the displayed first barycenter derivative in `eq:s_mu_small_beta_clean`. -/
theorem hasDerivAt_temperatureBarycenter {d n : ℕ} (hn : 0 < n)
    (X : Idx n → EucSpace d) (a : Idx n → ℝ) (t : ℝ) :
    HasDerivAt (temperatureBarycenter X a)
      (∑ k : Idx n, deriv (fun r : ℝ => Perspective.softmaxWeight (fun l => r * a l) k) t • X k) t := by
  have h := HasDerivAt.fun_sum (u := Finset.univ) fun k hk =>
    (hasDerivAt_softmaxWeight_temperature hn a k t).smul_const (X k)
  simp_rw [← (hasDerivAt_softmaxWeight_temperature hn a _ t).deriv] at h
  exact h

/-- The actual first barycenter derivative is bounded by twice the
original score bound. Source: arXiv:2604.01978v1, §7,
the estimate `‖∂_β m‖ ≤ 2 ‖A‖`, refined to the query's score bound. -/
theorem norm_deriv_temperatureBarycenter_le {d n : ℕ} (hn : 0 < n)
    (X : Idx n → EucSpace d) (hX : ∀ k, ‖X k‖ = 1) (a : Idx n → ℝ)
    (M : ℝ) (ha : ∀ k, |a k| ≤ M) (t : ℝ) :
    ‖deriv (temperatureBarycenter X a) t‖ ≤ 2 * M := by
  rw [(hasDerivAt_temperatureBarycenter hn X a t).deriv]
  apply (norm_sum_smul_unit_le_sum_abs X hX _).trans
  calc
    _ ≤ ∑ k : Idx n, 2 * M * Perspective.softmaxWeight (fun l => t * a l) k :=
      Finset.sum_le_sum fun k hk => abs_deriv_softmaxWeight_temperature_le hn a M ha k t
    _ = _ := by rw [← Finset.mul_sum, Perspective.sum_softmaxWeight hn _, mul_one]

/-- The genuine barycenter derivative is itself differentiable,
with the original coordinate second derivatives.
Source: arXiv:2604.01978v1, §7, the second differentiation in
the proof of `eq:s_mu_small_beta_clean`. -/
theorem hasDerivAt_deriv_temperatureBarycenter {d n : ℕ} (hn : 0 < n)
    (X : Idx n → EucSpace d) (a : Idx n → ℝ) (t : ℝ) :
    HasDerivAt (deriv (temperatureBarycenter X a))
      (∑ k : Idx n, deriv (deriv (fun r : ℝ => Perspective.softmaxWeight (fun l => r * a l) k)) t • X k) t := by
  have he : deriv (temperatureBarycenter X a) = fun r =>
      ∑ k : Idx n, deriv (fun s : ℝ => Perspective.softmaxWeight (fun l => s * a l) k) r • X k :=
    funext fun r => (hasDerivAt_temperatureBarycenter hn X a r).deriv
  rw [he]
  have h := HasDerivAt.fun_sum (u := Finset.univ) fun k hk =>
    (hasDerivAt_deriv_softmaxWeight_temperature hn a k t).smul_const (X k)
  simp_rw [← (hasDerivAt_deriv_softmaxWeight_temperature hn a _ t).deriv] at h
  exact h

/-- **The genuine second barycenter derivative has a uniform
quadratic score bound, without a token-count factor.**
Source: arXiv:2604.01978v1, §7, the bound on `∂_β² m` in
`eq:s_mu_small_beta_clean`, with the explicit constant six. -/
theorem norm_deriv_deriv_temperatureBarycenter_le {d n : ℕ} (hn : 0 < n)
    (X : Idx n → EucSpace d) (hX : ∀ k, ‖X k‖ = 1) (a : Idx n → ℝ)
    (M : ℝ) (ha : ∀ k, |a k| ≤ M) (t : ℝ) :
    ‖deriv (deriv (temperatureBarycenter X a)) t‖ ≤ 6 * M ^ 2 := by
  rw [(hasDerivAt_deriv_temperatureBarycenter hn X a t).deriv]
  apply (norm_sum_smul_unit_le_sum_abs X hX _).trans
  calc
    _ ≤ ∑ k : Idx n, 6 * M ^ 2 * Perspective.softmaxWeight (fun l => t * a l) k :=
      Finset.sum_le_sum fun k hk => abs_deriv_deriv_softmaxWeight_temperature_le hn a M ha k t
    _ = _ := by rw [← Finset.mul_sum, Perspective.sum_softmaxWeight hn _, mul_one]

/-- The actual empirical barycenter remains in the closed unit ball.
Source: arXiv:2604.01978v1, §7, `‖m_{β,A}[μ](x)‖ ≤ 1` in part (b). -/
theorem norm_temperatureBarycenter_le_one {d n : ℕ} (hn : 0 < n)
    (X : Idx n → EucSpace d) (hX : ∀ k, ‖X k‖ = 1) (a : Idx n → ℝ) (t : ℝ) :
    ‖temperatureBarycenter X a t‖ ≤ 1 := by
  apply (norm_sum_smul_unit_le_sum_abs X hX _).trans_eq
  simp only [abs_of_nonneg (Perspective.softmaxWeight_nonneg _ _), Perspective.sum_softmaxWeight hn _]

/-- The original finite barycenter is smooth in its actual temperature.
Source: arXiv:2604.01978v1, §7, the Taylor expansion in part (b). -/
theorem contDiff_temperatureBarycenter {d n : ℕ} (hn : 0 < n)
    (X : Idx n → EucSpace d) (a : Idx n → ℝ) :
    ContDiff ℝ (⊤ : ℕ∞) (temperatureBarycenter X a) := by
  apply ContDiff.sum
  intro k hk
  exact (contDiff_softmaxWeight_temperature hn a k).smul contDiff_const

/-- Nonconstant scores and original orthogonal unit tokens jointly
satisfy the nonempty, unit and uniform score-bound hypotheses. -/
example : ∃ (X : Idx 2 → EucSpace 2) (a : Idx 2 → ℝ),
    (0 : ℕ) < 2 ∧ (∀ k, ‖X k‖ = 1) ∧ (∀ k, |a k| ≤ 1) ∧
      temperatureBarycenter X a 0 = (1 / 2 : ℝ) • (X 0 + X 1) := by
  let X : Idx 2 → EucSpace 2 := fun k => EuclideanSpace.single k 1
  refine ⟨X, ![1, 0], by decide, fun k => by simp [X], ?_, ?_⟩
  · intro k
    fin_cases k <;> norm_num
  · simp only [temperatureBarycenter, zero_mul, Perspective.softmaxWeight,
      Real.exp_zero, Fin.sum_univ_two]
    norm_num

end Transformer.Homogenized
