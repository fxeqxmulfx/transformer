module

public import Transformer.Homogenized.Section7_TemperatureBarycenter
public import Mathlib.Analysis.Calculus.Taylor
public import Mathlib.Analysis.InnerProductSpace.Calculus

/-!
# Genuine temperature Taylor bound for the original barycenter correlation

ArXiv:2604.01978v1, §7, proof of `thm:clustering_small_beta`,
part (b) of `eq:s_mu_small_beta_clean`. The scalar observable is the
inner product of the actual normalized empirical barycenters at two
fixed score rows. At original query scores it is exactly the original
headwise correlation whose expectation defines `s_μ(x,x')`.

The genuine product rule retains both second barycenter derivatives
and the two mixed first derivatives. The proved uniform bounds imply
`|F''(β)| ≤ 10 (M² + N²)`, with no token-count factor. Taylor's theorem
then bounds the original linear remainder by
`10 β² (M² + N²)` for `β ∈ [0,1]`. The linear term is the actual
derivative at zero. Its subsequent Gaussian mean-zero calculation
and the two original Gaussian query moments are separate obligations.
Neither the symmetry cancellation nor an expectation estimate is
inserted as a premise of the temperature derivative calculation.
-/

@[expose] public section

open Real Set
open scoped BigOperators

namespace Transformer.Homogenized

/-- The genuine original headwise correlation at two score rows.
Source: arXiv:2604.01978v1, §7, the inner product expanded in
the proof of `eq:s_mu_small_beta_clean`. -/
noncomputable def temperatureBarycenterCorrelation {d n : ℕ} (X : Idx n → EucSpace d)
    (a b : Idx n → ℝ) (t : ℝ) : ℝ :=
  inner ℝ (temperatureBarycenter X a t) (temperatureBarycenter X b t)

/-- At the actual query scores this is exactly the original
headwise empirical correlation, including self-pairs.
Source: arXiv:2604.01978v1, §7, `eq:s_mu_small_beta_clean`. -/
theorem temperatureBarycenterCorrelation_eq_inner_softBary {d n : ℕ}
    (X : Idx n → EucSpace d) (A : Matrix (Fin d) (Fin d) ℝ) (i j : Idx n) (t : ℝ) :
    temperatureBarycenterCorrelation X
      (fun k => inner ℝ (Matrix.toEuclideanLin A (X i)) (X k))
      (fun k => inner ℝ (Matrix.toEuclideanLin A (X j)) (X k)) t =
      inner ℝ (softBary t A (empMeasure X) (X i)) (softBary t A (empMeasure X) (X j)) := by
  rw [temperatureBarycenterCorrelation, temperatureBarycenter_eq_softBary,
    temperatureBarycenter_eq_softBary]

/-- The genuine correlation derivative differentiates both original
barycenters. Source: arXiv:2604.01978v1, §7, the expansion in part (b). -/
theorem hasDerivAt_temperatureBarycenterCorrelation {d n : ℕ} (hn : 0 < n)
    (X : Idx n → EucSpace d) (a b : Idx n → ℝ) (t : ℝ) :
    HasDerivAt (temperatureBarycenterCorrelation X a b)
      (inner ℝ (temperatureBarycenter X a t) (deriv (temperatureBarycenter X b) t) +
        inner ℝ (deriv (temperatureBarycenter X a) t) (temperatureBarycenter X b t)) t := by
  exact ((hasDerivAt_temperatureBarycenter hn X a t).differentiableAt.hasDerivAt).inner ℝ
    (hasDerivAt_temperatureBarycenter hn X b t).differentiableAt.hasDerivAt

/-- The actual second correlation derivative keeps both mixed
first-derivative terms. Source: arXiv:2604.01978v1, §7, the
second-order expansion underlying `eq:s_mu_small_beta_clean`. -/
theorem hasDerivAt_deriv_temperatureBarycenterCorrelation {d n : ℕ} (hn : 0 < n)
    (X : Idx n → EucSpace d) (a b : Idx n → ℝ) (t : ℝ) :
    HasDerivAt (deriv (temperatureBarycenterCorrelation X a b))
      (inner ℝ (temperatureBarycenter X a t) (deriv (deriv (temperatureBarycenter X b)) t) +
        2 * inner ℝ (deriv (temperatureBarycenter X a) t) (deriv (temperatureBarycenter X b) t) +
        inner ℝ (deriv (deriv (temperatureBarycenter X a)) t) (temperatureBarycenter X b t)) t := by
  have he : deriv (temperatureBarycenterCorrelation X a b) = fun r =>
      inner ℝ (temperatureBarycenter X a r) (deriv (temperatureBarycenter X b) r) +
        inner ℝ (deriv (temperatureBarycenter X a) r) (temperatureBarycenter X b r) :=
    funext fun r => (hasDerivAt_temperatureBarycenterCorrelation hn X a b r).deriv
  rw [he]
  have h1 (c : Idx n → ℝ) := (hasDerivAt_temperatureBarycenter hn X c t).differentiableAt.hasDerivAt
  have h2 (c : Idx n → ℝ) := (hasDerivAt_deriv_temperatureBarycenter hn X c t).differentiableAt.hasDerivAt
  have h := ((h1 a).inner ℝ (h2 b)).fun_add ((h2 a).inner ℝ (h1 b))
  convert h using 1
  ring

/-- The genuine second correlation derivative has a quadratic bound
in the two original score bounds, independently of token count.
Source: arXiv:2604.01978v1, §7, the Taylor estimate in part (b). -/
theorem abs_deriv_deriv_temperatureBarycenterCorrelation_le {d n : ℕ} (hn : 0 < n)
    (X : Idx n → EucSpace d) (hX : ∀ k, ‖X k‖ = 1) (a b : Idx n → ℝ)
    (M N : ℝ) (ha : ∀ k, |a k| ≤ M) (hb : ∀ k, |b k| ≤ N) (t : ℝ) :
    |deriv (deriv (temperatureBarycenterCorrelation X a b)) t| ≤ 10 * (M ^ 2 + N ^ 2) := by
  have hM : 0 ≤ M := (abs_nonneg (a ⟨0, hn⟩)).trans (ha ⟨0, hn⟩)
  have hN : 0 ≤ N := (abs_nonneg (b ⟨0, hn⟩)).trans (hb ⟨0, hn⟩)
  have hA := norm_temperatureBarycenter_le_one hn X hX a t
  have hB := norm_temperatureBarycenter_le_one hn X hX b t
  have hDA := norm_deriv_temperatureBarycenter_le hn X hX a M ha t
  have hDB := norm_deriv_temperatureBarycenter_le hn X hX b N hb t
  have hDDA := norm_deriv_deriv_temperatureBarycenter_le hn X hX a M ha t
  have hDDB := norm_deriv_deriv_temperatureBarycenter_le hn X hX b N hb t
  have hAB : |inner ℝ (temperatureBarycenter X a t) (deriv (deriv (temperatureBarycenter X b)) t)| ≤
      6 * N ^ 2 := (abs_real_inner_le_norm _ _).trans (by nlinarith [mul_le_mul hA hDDB (norm_nonneg _) zero_le_one])
  have hBA : |inner ℝ (deriv (deriv (temperatureBarycenter X a)) t) (temperatureBarycenter X b t)| ≤
      6 * M ^ 2 := (abs_real_inner_le_norm _ _).trans (by nlinarith [mul_le_mul hDDA hB (norm_nonneg _) (by positivity)])
  have hDD : |inner ℝ (deriv (temperatureBarycenter X a) t) (deriv (temperatureBarycenter X b) t)| ≤
      4 * M * N := (abs_real_inner_le_norm _ _).trans (by nlinarith [mul_le_mul hDA hDB (norm_nonneg _) (by positivity)])
  rw [(hasDerivAt_deriv_temperatureBarycenterCorrelation hn X a b t).deriv]
  have hs := abs_add_three
    (inner ℝ (temperatureBarycenter X a t) (deriv (deriv (temperatureBarycenter X b)) t))
    (2 * inner ℝ (deriv (temperatureBarycenter X a) t) (deriv (temperatureBarycenter X b) t))
    (inner ℝ (deriv (deriv (temperatureBarycenter X a)) t) (temperatureBarycenter X b t))
  norm_num [abs_mul] at hs
  nlinarith [sq_nonneg (M - N)]

/-- The original correlation is smooth in its actual temperature.
Source: arXiv:2604.01978v1, §7, the Taylor expansion in part (b). -/
theorem contDiff_temperatureBarycenterCorrelation {d n : ℕ} (hn : 0 < n)
    (X : Idx n → EucSpace d) (a b : Idx n → ℝ) :
    ContDiff ℝ (⊤ : ℕ∞) (temperatureBarycenterCorrelation X a b) :=
  (contDiff_temperatureBarycenter hn X a).inner ℝ (contDiff_temperatureBarycenter hn X b)

/-- **The actual headwise correlation has a uniform quadratic
temperature remainder after its genuine linear term.**
Source: arXiv:2604.01978v1, §7, `eq:s_mu_small_beta_clean`,
the Taylor step before the Gaussian first-order cancellation. -/
theorem temperatureBarycenterCorrelation_taylor_bound {d n : ℕ} (hn : 0 < n)
    (X : Idx n → EucSpace d) (hX : ∀ k, ‖X k‖ = 1) (a b : Idx n → ℝ)
    (M N : ℝ) (ha : ∀ k, |a k| ≤ M) (hb : ∀ k, |b k| ≤ N)
    (β : ℝ) (hβ : β ∈ Icc (0 : ℝ) 1) :
    |temperatureBarycenterCorrelation X a b β - temperatureBarycenterCorrelation X a b 0 -
        β * deriv (temperatureBarycenterCorrelation X a b) 0| ≤ 10 * β ^ 2 * (M ^ 2 + N ^ 2) := by
  let f := temperatureBarycenterCorrelation X a b
  have hf : ContDiff ℝ 2 f := (contDiff_temperatureBarycenterCorrelation hn X a b).of_le (by simp)
  have hu := uniqueDiffOn_Icc (show (0 : ℝ) < 1 by norm_num)
  have hd : derivWithin f (Icc (0 : ℝ) 1) 0 = deriv f 0 :=
    (hf.differentiable (by norm_num) 0).hasDerivAt.hasDerivWithinAt.derivWithin (hu 0 (by norm_num))
  have h := taylor_mean_remainder_bound (n := 1) (show (0 : ℝ) ≤ 1 by norm_num) hf.contDiffOn hβ
    (C := 10 * (M ^ 2 + N ^ 2)) (fun y hy => by
      rw [iteratedDerivWithin_eq_iteratedDeriv hu (hf.contDiffAt) hy, iteratedDeriv_succ, iteratedDeriv_one,
        Real.norm_eq_abs]
      exact abs_deriv_deriv_temperatureBarycenterCorrelation_le hn X hX a b M N ha hb y)
  norm_num [taylorWithinEval_succ, taylor_within_zero_eval, iteratedDerivWithin_one, hd] at h
  have he : |f β - f 0 - β * deriv f 0| = |f β - (f 0 + β * deriv f 0)| := by
    congr 1
    ring
  change |f β - f 0 - β * deriv f 0| ≤ _
  rw [he]
  convert h using 1
  ring

/-- Two nonconstant original score rows, unit orthogonal tokens and
a positive temperature jointly satisfy every Taylor-bound hypothesis. -/
example : ∃ (X : Idx 2 → EucSpace 2) (a b : Idx 2 → ℝ),
    (0 : ℕ) < 2 ∧ (∀ k, ‖X k‖ = 1) ∧ (∀ k, |a k| ≤ 1) ∧
      (∀ k, |b k| ≤ 2) ∧ (1 / 2 : ℝ) ∈ Icc (0 : ℝ) 1 := by
  refine ⟨fun k => EuclideanSpace.single k 1, ![1, 0], ![0, 2], by decide,
    fun k => by simp, ?_, ?_, by norm_num⟩
  · intro k
    fin_cases k <;> norm_num
  · intro k
    fin_cases k <;> norm_num

end Transformer.Homogenized
