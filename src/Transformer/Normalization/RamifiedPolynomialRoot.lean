/-
# Real Newton--Puiseux root lifting in arbitrary degree

A real analytic monic polynomial family with central polynomial `y^d`
and real roots accumulating from positive parameters has an analytic root
after a positive integral power substitution. The proof uses centering,
Newton scaling, compactness, and real preparation to decrease the degree.
-/

import Transformer.Normalization.PolynomialDegreeReduction
import Transformer.Normalization.PolynomialRootCluster
import Transformer.Normalization.PreparedRootAccumulation

open Filter Set

namespace Transformer.Normalization

/-- An analytic monic polynomial of arbitrary degree with central
polynomial `y^d` has a real analytic ramified root whenever real roots
accumulate from the positive parameter side. No simplicity, discriminant
condition, or bound on the degree is assumed. The polynomial identity
holds on a full real neighborhood of the new parameter origin.
Auxiliary Newton--Puiseux input for singular curve selection in
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem analytic_ramified_polynomial_root (d : ℕ) (a : Fin d → ℝ → ℝ)
    (ha : ∀ i, AnalyticAt ℝ (a i) 0) (ha0 : ∀ i, a i 0 = 0)
    (hroots : ∃ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0),
      ∃ y : ℝ, polynomialFamily d a (t, y) = 0) :
    ∃ (q : ℕ) (g : ℝ → ℝ), 0 < q ∧ AnalyticAt ℝ g 0 ∧ g 0 = 0 ∧
      ∀ᶠ s in nhds (0 : ℝ), polynomialFamily d a (s ^ q, g s) = 0 := by
  classical
  induction d using Nat.strong_induction_on with
  | h d ih =>
    cases d with
    | zero =>
      obtain ⟨t, y, hy⟩ := hroots.exists
      simp [polynomialFamily] at hy
    | succ n =>
      obtain ⟨c, b, hc, hc0, hb, hb0, hcenter, htranslate⟩ :=
        analytic_polynomial_centering a ha ha0
      have hbroots : ∃ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0),
          ∃ y : ℝ, polynomialFamily (n + 1) b (t, y) = 0 := by
        apply hroots.mono
        rintro t ⟨y, hy⟩
        refine ⟨y - c t, ?_⟩
        rw [← htranslate t (y - c t), sub_add_cancel]
        exact hy
      by_cases hzero : ∀ i, ∀ᶠ t in nhds (0 : ℝ), b i t = 0
      · refine ⟨1, c, by omega, hc, hc0, ?_⟩
        filter_upwards [eventually_all.mpr hzero] with s hs
        have hB : polynomialFamily (n + 1) b (s, 0) = 0 := by
          simp [polynomialFamily, hs]
        simpa only [pow_one, zero_add] using (htranslate s 0).trans hB
      have hnonzero : ∃ i, ¬ ∀ᶠ t in nhds (0 : ℝ), b i t = 0 := not_forall.mp hzero
      obtain ⟨p, q, beta, hp, hq, hbeta, hbeta0, hpreserve, hcoeff⟩ :=
        analytic_newton_coefficient_scaling b hb hb0 hnonzero
      have hbetaroots := newton_normalized_roots_frequent b beta hq hcoeff hbroots
      obtain ⟨r, hroot, hacc⟩ := analytic_polynomial_root_cluster beta hbeta hbetaroots
      have hbetacenter : beta (Fin.last n) 0 = 0 := by
        have hbzero : ∀ᶠ t in nhds (0 : ℝ), b (Fin.last n) t = 0 := by simp [hcenter]
        simp only [hpreserve (Fin.last n) hbzero]
      obtain ⟨m, A, U, _, hmd, hA, hA0, hU, hU0, hprepare⟩ :=
        centered_analytic_polynomial_degree_reduction beta hbeta hbetacenter hbeta0 r hroot
      have hAroots := frequent_roots_of_polynomial_preparation beta A r U
        hU.continuousAt hU0 hprepare hacc
      obtain ⟨k, g, hk, hg, hg0, hgroot⟩ := ih m hmd A hA hA0 hAroots
      have hQ : 0 < k * q := Nat.mul_pos hk hq
      have hP : 0 < k * p := Nat.mul_pos hk hp
      have hpower : AnalyticAt ℝ (fun s : ℝ => s ^ (k * q)) 0 := analyticAt_id.fun_pow _
      have hcat : AnalyticAt ℝ c ((0 : ℝ) ^ (k * q)) := by simpa [hQ.ne'] using hc
      have hccomp : AnalyticAt ℝ (fun s : ℝ => c (s ^ (k * q))) 0 :=
        hcat.comp (f := fun s : ℝ => s ^ (k * q)) (x := 0) hpower
      let gamma : ℝ → ℝ := fun s => s ^ (k * p) * (r + g s) + c (s ^ (k * q))
      have hgamma : AnalyticAt ℝ gamma 0 :=
        ((analyticAt_id.fun_pow _).fun_mul (analyticAt_const.fun_add hg)).fun_add hccomp
      have hgamma0 : gamma 0 = 0 := by simp [gamma, hP.ne', hQ.ne', hc0]
      refine ⟨k * q, gamma, hQ, hgamma, hgamma0, ?_⟩
      have hpowerK : Tendsto (fun s : ℝ => s ^ k) (nhds 0) (nhds 0) := by
        simpa [hk.ne'] using
          ((analyticAt_id (𝕜 := ℝ) (z := (0 : ℝ))).fun_pow k).continuousAt.tendsto
      have hpath : Tendsto (fun s : ℝ => (s ^ k, g s)) (nhds 0)
          (nhds (0 : ℝ × ℝ)) := by
        simpa only [hg0, Prod.mk_zero_zero] using hpowerK.prodMk_nhds hg.continuousAt.tendsto
      filter_upwards [hgroot, hpath.eventually hprepare,
        hpowerK.eventually (newton_polynomial_factorization b beta hcoeff)] with s hsroot hsP hsN
      have hnormalized : polynomialFamily (n + 1) beta (s ^ k, r + g s) = 0 := by
        rw [hsP, hsroot, mul_zero]
      have hscaled : polynomialFamily (n + 1) b
          (s ^ (k * q), s ^ (k * p) * (r + g s)) = 0 := by
        change ∀ y : ℝ, polynomialFamily (n + 1) b ((s ^ k) ^ q, (s ^ k) ^ p * y) =
          (s ^ k) ^ (p * (n + 1)) * polynomialFamily (n + 1) beta (s ^ k, y) at hsN
        have h := hsN (r + g s)
        rw [hnormalized, mul_zero] at h
        simpa only [← pow_mul] using h
      change polynomialFamily (n + 1) a
        (s ^ (k * q), s ^ (k * p) * (r + g s) + c (s ^ (k * q))) = 0
      rw [htranslate]
      exact hscaled

/-- The degree-five singular polynomial `y⁵ - t²` has a real root for
every positive parameter. All hypotheses of the unrestricted-degree
lifting theorem hold jointly; odd degree and a multiple central root
exercise the Newton induction. Auxiliary for Appendix D.1, `lem: loj`,
of arXiv:2510.22026v2. -/
example : let a : Fin 5 → ℝ → ℝ := fun i t => if i = 0 then -(t ^ 2) else 0
    ∃ (q : ℕ) (g : ℝ → ℝ), 0 < q ∧ AnalyticAt ℝ g 0 ∧ g 0 = 0 ∧
      ∀ᶠ s in nhds (0 : ℝ), polynomialFamily 5 a (s ^ q, g s) = 0 := by
  apply analytic_ramified_polynomial_root
  · intro i
    split_ifs
    · exact (analyticAt_id.fun_pow 2).fun_neg
    · exact analyticAt_const
  · intro i
    simp
  · have : NeBot (nhdsWithin (0 : ℝ) (Ioi 0)) := nhdsWithin_Ioi_neBot le_rfl
    have hpositive : ∀ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0), 0 < t := self_mem_nhdsWithin
    apply hpositive.frequently.mono
    intro t ht
    refine ⟨(t ^ 2) ^ ((5 : ℝ)⁻¹), ?_⟩
    simpa [polynomialFamily, Fin.sum_univ_succ, sub_eq_add_neg] using
      sub_eq_zero.mpr (Real.rpow_inv_natCast_pow (sq_nonneg t) (by omega : (5 : ℕ) ≠ 0))

end Transformer.Normalization
