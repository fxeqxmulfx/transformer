/-
# Joint witnesses for coefficient sequence hypotheses

Concrete embeddings, monomials, small divisor tails, and convergent scalar
evaluation supply the hypotheses of the copied sequence lemmas. Auxiliary
examples for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2.
-/

import Transformer.AnalyticPreparation.PreparationReconstruction

open scoped ENNReal NNReal Topology

namespace Transformer.AnalyticPreparation

/-- The successor embedding omits the zero index, as required by both
off-range embedding lemmas. Appendix D.1 of arXiv:2510.22026v2. -/
example : ∃ e : ℕ ↪ ℕ, 0 ∉ Set.range e ∧
    L1Coeff.embedFun e (monomialSeq 1) 0 = 0 ∧
    L1Coeff.embed e (monomialSeq 1) 0 = 0 := by
  let e : ℕ ↪ ℕ := ⟨Nat.succ, Nat.succ_injective⟩
  have h : 0 ∉ Set.range e := by rintro ⟨n, hn⟩; exact Nat.succ_ne_zero n hn
  exact ⟨e, h, L1Coeff.embedFun_apply_of_not_mem e _ _ h,
    L1Coeff.embed_apply_of_not_mem e _ _ h⟩

/-- A zero operator perturbation and nonzero linear right-hand side
satisfy the analytic inverse-operator hypotheses. Appendix D.1 of
arXiv:2510.22026v2. -/
example : AnalyticAt ℝ (fun x : ℝ =>
    Ring.inverse (1 + (0 : ℝ →L[ℝ] (ℝ →L[ℝ] ℝ)) x)
      ((ContinuousLinearMap.id ℝ ℝ) x)) 0 := by
  exact analyticAt_inverseOneAdd_apply 0 (ContinuousLinearMap.id ℝ ℝ) 0 (by simp)

/-- Both sides of the cutoff and shift index conditions occur for a
degree-one monomial. Appendix D.1 of arXiv:2510.22026v2. -/
example : seqLowShift 1 (monomialSeq 1) 0 = 0 ∧
    seqLowShift 1 (monomialSeq 1) 2 = monomialSeq 1 1 ∧
    seqLowCut 1 (monomialSeq 1) 0 = monomialSeq 1 0 ∧
    seqLowCut 1 (monomialSeq 1) 1 = 0 := by
  exact ⟨seqLowShift_apply_of_lt _ _ (by norm_num),
    seqLowShift_apply_of_le _ _ (by norm_num),
    seqLowCut_apply_of_lt _ _ (by norm_num), seqLowCut_apply_of_le _ _ le_rfl⟩

/-- The small-tail assumption of all sequence division identities and
analytic quotient/remainder maps is jointly satisfiable with a nonzero
dividend. Appendix D.1 of arXiv:2510.22026v2. -/
example : ∃ p f : OriginSeq, ‖p‖ < 1 ∧ f ≠ 0 ∧
    AnalyticAt ℝ (seqDivisionQuotientGlobal 1) (p, f) ∧
    AnalyticAt ℝ (seqDivisionRemainderGlobal 1) (p, f) ∧
    f = seqLowShift 1 (seqDivisionQuotientGlobal 1 (p, f)) +
      convolution (seqDivisionQuotientGlobal 1 (p, f)) p +
        seqDivisionRemainderGlobal 1 (p, f) ∧
    seqHighShift 1 (seqDivisionRemainderGlobal 1 (p, f)) = 0 := by
  have hp : ‖(0 : OriginSeq)‖ < 1 := by simp
  have hf : monomialSeq 1 ≠ 0 := by
    intro hzero
    have heq := congrArg (fun f : OriginSeq => f 1) hzero
    simp at heq
  exact ⟨0, monomialSeq 1, hp, hf,
    analyticAt_seqDivisionQuotientGlobal _ _ hp,
    analyticAt_seqDivisionRemainderGlobal _ _ hp,
    seqDivisionGlobal_factorization _ _ _ hp, seqHighShift_divisionRemainderGlobal _ _ _ hp⟩

/-- The normalization and positive small-weight assumptions are jointly
satisfiable at exact degree one. Appendix D.1 of arXiv:2510.22026v2. -/
example : ∃ (f : OriginSeq) (t : ℝ≥0) (ht : t ≤ 1), 0 < t ∧
    (∀ j < 1, f j = 0) ∧ f 1 ≠ 0 ∧
    ‖normalizedScale t ht f 1 - monomialSeq 1‖ ≤
      ((t : ℝ) / ‖f 1‖) * ‖f‖ ∧
    ‖(normalizedScale t ht f 1 - monomialSeq 1) 2‖ ≤
      ((t : ℝ) / ‖f 1‖) * ‖f 2‖ := by
  have ht : (1 / 2 : ℝ≥0) ≤ 1 := by norm_num
  have ht0 : (0 : ℝ≥0) < 1 / 2 := by norm_num
  have hlow : ∀ j < 1, monomialSeq 1 j = 0 := by
    intro j hj
    exact monomialSeq_apply_ne (ne_of_lt hj)
  have hfd : monomialSeq 1 1 ≠ 0 := by simp
  exact ⟨monomialSeq 1, 1 / 2, ht, ht0, hlow, hfd,
    norm_normalizedScale_sub_monomial_le ht ht0 _ _ hlow hfd,
    normalizedScale_sub_monomial_apply_lt ht ht0 _ _ _ hlow hfd⟩

/-- Evaluation converges at a nonzero point of the unit interval, and a
degree-one sequence has no terms of degree at least two. The analytic
coefficient family is nonconstant. Appendix D.1 of arXiv:2510.22026v2. -/
example : Summable (fun k : ℕ => monomialSeq 1 k * (1 / 2 : ℝ) ^ k) ∧
    evalL1PowerSeries (monomialSeq 1) (1 / 2) = (1 / 2 : ℝ) ^ 1 ∧
    AnalyticAt ℝ (fun x : ℝ × ℝ =>
      evalL1PowerSeries (x.1 • monomialSeq 1) x.2) 0 := by
  have hw : ‖(1 / 2 : ℝ)‖ < 1 := by norm_num
  have hq : AnalyticAt ℝ (fun x : ℝ => x • monomialSeq 1) 0 :=
    (ContinuousLinearMap.toSpanSingleton ℝ (monomialSeq 1)).analyticAt 0
  refine ⟨summable_l1_mul_pow _ hw.le, evalL1PowerSeries_monomialSeq _ hw, ?_⟩
  simpa using Transformer.AnalyticPreparation.AnalyticAt.evalL1PowerSeries hq 1

end Transformer.AnalyticPreparation
