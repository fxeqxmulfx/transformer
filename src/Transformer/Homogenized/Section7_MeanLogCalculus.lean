module

public import Transformer.Homogenized.Section7_MeanOverlapCalculus
public import Mathlib.Analysis.SpecialFunctions.Log.Deriv

/-!
# Genuine calculus of the regularized logarithmic empirical mean

ArXiv:2604.01978v1, §7, proof of `thm:clustering_small_beta`,
the original mean decomposition `eq:m_decomp_clean` and the maximal
estimate after `eq:gronwall_e_clean`. The auxiliary observable here
is the logarithm of the actual mean overlap plus a positive constant.
The mean itself remains the original normalized sum of all overlaps.

Its nonnegativity is already proved as the squared norm of the
original centroid. The logarithm therefore has no singularity anywhere
in the ambient configuration space, so it is a genuine smooth test
for the original sphere martingale problem, including configurations
whose centroid is zero.

The first and second derivatives below are actual derivatives.
Their sphere Hessian retains the original sphere correction and the
negative quadratic first-derivative term. This is the genuine Itô
correction required for logarithmic control of the original mean;
neither a noise estimate nor a martingale identity is postulated.
-/

@[expose] public section

open Real
open scoped BigOperators

namespace Transformer.Homogenized

/-- The regularized logarithm of the original empirical mean.
Source: arXiv:2604.01978v1, §7, the mean observable in
`eq:m_decomp_clean`, transformed for its maximal comparison. -/
noncomputable def logMeanOverlap {d n : ℕ} (a : ℝ) (X : Idx n → EucSpace d) : ℝ :=
  Real.log (meanOverlap X + a)

/-- Positive regularization removes the logarithmic singularity on
the entire original configuration space. Source: arXiv:2604.01978v1,
§7, the nonnegative squared-centroid mean in `eq:def_m_smallbeta`. -/
theorem meanOverlap_add_pos {d n : ℕ} {a : ℝ} (ha : 0 < a) (X : Idx n → EucSpace d) :
    0 < meanOverlap X + a := add_pos_of_nonneg_of_pos (meanOverlap_nonneg X) ha

/-- The auxiliary logarithm is a genuine smooth ambient test for the
original martingale problem. Source: arXiv:2604.01978v1, §7,
the mean in `eq:m_decomp_clean` and Appendix, `def:ito.formula`. -/
theorem contDiff_logMeanOverlap {d n : ℕ} {a : ℝ} (ha : 0 < a) :
    ContDiff ℝ (⊤ : ℕ∞) (logMeanOverlap (d := d) (n := n) a) := by
  have hm : ContDiff ℝ (⊤ : ℕ∞) (meanOverlap (d := d) (n := n)) := contDiff_meanOverlap
  exact (hm.add contDiff_const).log fun X => (meanOverlap_add_pos ha X).ne'

/-- The actual logarithmic differential retains the original mean
differential. Source: arXiv:2604.01978v1, §7, the differential of
the original mean in `eq:m_decomp_clean`. -/
theorem hasFDerivAt_logMeanOverlap {d n : ℕ} {a : ℝ} (ha : 0 < a) (X : Idx n → EucSpace d) :
    HasFDerivAt (logMeanOverlap a) ((meanOverlap X + a)⁻¹ • meanOverlapForm X) X :=
  ((hasFDerivAt_meanOverlap X).add_const a).log (meanOverlap_add_pos ha X).ne'

/-- Evaluation of the genuine first logarithmic derivative.
Source: arXiv:2604.01978v1, §7, the original mean noise differential. -/
theorem fderiv_logMeanOverlap {d n : ℕ} {a : ℝ} (ha : 0 < a) (X : Idx n → EucSpace d) :
    fderiv ℝ (logMeanOverlap a) X = (meanOverlap X + a)⁻¹ • meanOverlapForm X :=
  (hasFDerivAt_logMeanOverlap ha X).fderiv

/-- The genuine logarithmic second derivative has the negative
square of the original differential, as well as its actual Hessian.
Source: arXiv:2604.01978v1, §7, `eq:m_decomp_clean` and the
second-order chain rule in Appendix, `lem:Ito_formula`. -/
theorem iteratedFDeriv_two_logMeanOverlap {d n : ℕ} {a : ℝ} (ha : 0 < a)
    (X V : Idx n → EucSpace d) :
    iteratedFDeriv ℝ 2 (logMeanOverlap a) X ![V, V] =
      (meanOverlap X + a)⁻¹ * meanOverlapForm V V -
        ((meanOverlap X + a) ^ 2)⁻¹ * (meanOverlapForm X V) ^ 2 := by
  have hfirst : fderiv ℝ (logMeanOverlap (d := d) (n := n) a) =
      fun Y => (meanOverlap Y + a)⁻¹ • meanOverlapForm Y := funext (fderiv_logMeanOverlap ha)
  have hi := (hasDerivAt_inv (meanOverlap_add_pos ha X).ne').comp_hasFDerivAt X
    ((hasFDerivAt_meanOverlap X).add_const a)
  have hd := hi.fun_smul (meanOverlapForm (d := d) (n := n)).hasFDerivAt
  change HasFDerivAt (fun Y => (meanOverlap Y + a)⁻¹ • meanOverlapForm Y) _ X at hd
  rw [iteratedFDeriv_two_apply, hfirst, hd.fderiv]
  simp only [Function.comp_apply, Matrix.cons_val_zero, Matrix.cons_val_one, add_apply, smul_apply,
    ContinuousLinearMap.smulRight_apply, smul_eq_mul]
  ring

/-- **The actual logarithmic sphere Hessian includes the genuine
negative quadratic correction.** Source: arXiv:2604.01978v1,
§7, the original mean observable and Appendix, `def:ito.formula`.
The regularization changes the auxiliary test, not the original mean
or the original sphere generator. -/
theorem sphHess_logMeanOverlap {d n : ℕ} {a : ℝ} (ha : 0 < a) (X V : Idx n → EucSpace d) :
    sphHess (logMeanOverlap a) X V =
      (meanOverlap X + a)⁻¹ * sphHess meanOverlap X V -
        ((meanOverlap X + a) ^ 2)⁻¹ * (meanOverlapForm X V) ^ 2 := by
  have hs : (∑ k : Idx n, fderiv ℝ (logMeanOverlap a) X (Pi.single k (X k)) * ‖V k‖ ^ 2) =
      (meanOverlap X + a)⁻¹ * ∑ k : Idx n, meanOverlapForm X (Pi.single k (X k)) * ‖V k‖ ^ 2 := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro k hk
    rw [fderiv_logMeanOverlap ha]
    simp only [smul_apply, smul_eq_mul, mul_assoc]
  rw [sphHess, iteratedFDeriv_two_logMeanOverlap ha, hs, sphHess, iteratedFDeriv_two_apply,
    fderiv_fderiv_meanOverlap]
  simp only [Matrix.cons_val_zero, Matrix.cons_val_one, fderiv_meanOverlap]
  ring

/-- The genuine logarithmic sphere Hessian is jointly continuous
in its original base and direction. Source: arXiv:2604.01978v1,
§7, the original mean generator's path integral. -/
theorem continuous_sphHess_logMeanOverlap {d n : ℕ} {a : ℝ} (ha : 0 < a) :
    Continuous (fun P : (Idx n → EucSpace d) × (Idx n → EucSpace d) =>
      sphHess (logMeanOverlap a) P.1 P.2) := by
  have hm : Continuous (meanOverlap (d := d) (n := n)) := contDiff_meanOverlap.continuous
  have hc : Continuous (fun P : (Idx n → EucSpace d) × (Idx n → EucSpace d) =>
      (meanOverlap P.1 + a)⁻¹) :=
    ((hm.comp continuous_fst).add continuous_const).inv₀
      fun P => (meanOverlap_add_pos ha P.1).ne'
  have hF : Continuous (fun P : (Idx n → EucSpace d) × (Idx n → EucSpace d) =>
      meanOverlapForm P.1 P.2) :=
    ((meanOverlapForm (d := d) (n := n)).continuous.comp continuous_fst).clm_apply continuous_snd
  simp only [sphHess_logMeanOverlap ha, ← inv_pow]
  exact (hc.mul continuous_sphHess_meanOverlap).sub
    ((hc.pow 2).mul (hF.pow 2))

/-- Unit original configurations bound the logarithmic observable
between its two genuine endpoint values.
Source: arXiv:2604.01978v1, §7, `m(X) ∈ [0,1]` in the mean argument. -/
theorem logMeanOverlap_mem_Icc {d n : ℕ} {a : ℝ} (ha : 0 < a)
    (X : Idx n → EucSpace d) (hX : ∀ k, ‖X k‖ = 1) :
    logMeanOverlap a X ∈ Set.Icc (Real.log a) (Real.log (1 + a)) := by
  have hm := meanOverlap_mem_Icc X hX
  exact ⟨Real.log_le_log ha (by linarith [hm.1]),
    Real.log_le_log (meanOverlap_add_pos ha X) (by linarith [hm.2])⟩

/-- The actual logarithmic test has an explicit finite bound on
original unit configurations. Source: arXiv:2604.01978v1, §7,
the bounded smooth tests used for the mean's maximal estimate. -/
theorem abs_logMeanOverlap_le {d n : ℕ} {a : ℝ} (ha : 0 < a)
    (X : Idx n → EucSpace d) (hX : ∀ k, ‖X k‖ = 1) :
    |logMeanOverlap a X| ≤ |Real.log a| + |Real.log (1 + a)| := by
  have h := logMeanOverlap_mem_Icc ha X hX
  apply abs_le.mpr
  constructor
  · linarith [h.1, neg_abs_le (Real.log a), abs_nonneg (Real.log (1 + a))]
  · linarith [h.2, le_abs_self (Real.log (1 + a)), abs_nonneg (Real.log a)]

/-- Positive regularization and original unit queries jointly realize
every hypothesis, with nonzero genuine first and second logarithmic derivatives. -/
example : ∃ X : Idx 2 → EucSpace 2, (0 : ℝ) < 1 ∧ (∀ k, ‖X k‖ = 1) ∧
    fderiv ℝ (logMeanOverlap 1) X X = 2 / 3 ∧
      iteratedFDeriv ℝ 2 (logMeanOverlap 1) X ![X, X] = 2 / 9 := by
  let X : Idx 2 → EucSpace 2 := fun k => EuclideanSpace.single k 1
  have hm : meanOverlap X = 1 / 2 := by
    norm_num [meanOverlap, X, Fin.sum_univ_two, EuclideanSpace.inner_single_left, PiLp.single_apply]
  refine ⟨X, one_pos, fun k => by simp [X], ?_, ?_⟩
  · rw [fderiv_logMeanOverlap one_pos, smul_apply, smul_eq_mul, meanOverlapForm_self, hm]
    norm_num
  · rw [iteratedFDeriv_two_logMeanOverlap one_pos, meanOverlapForm_self, hm]
    norm_num

end Transformer.Homogenized
