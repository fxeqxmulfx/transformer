/-
# Stable minima among finitely many analytic branches

Analytic sign constraints stabilize on the positive side. A finite family
of analytic objective values therefore has one admissible branch which
minimizes the objective at every sufficiently small positive parameter.
-/

import Transformer.Normalization.AnalyticSignStability
import Mathlib.Data.Finset.Max
import Mathlib.Order.Filter.Finite

open Filter Set

namespace Transformer.Normalization

/-- A finite family of accumulating analytic sign constraints holds on a
whole sufficiently small positive interval. Auxiliary for the quantified
minimum step of Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem analytic_finite_sign_constraints_of_frequent {l : ℕ}
    (C : Fin l → ℝ → ℝ) (hC : ∀ i, AnalyticAt ℝ (C i) 0)
    (requirement : Fin l → AnalyticSignRequirement)
    (h : ∃ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0), ∀ i,
      (requirement i).Holds (C i t)) :
    ∀ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0), ∀ i,
      (requirement i).Holds (C i t) := by
  apply eventually_all.mpr
  intro i
  exact analytic_sign_requirement_of_frequent (C i) (hC i) (requirement i)
    (h.mono (fun t ht => ht i))

/-- One fixed branch in a finite analytic family minimizes the objective
among all admissible branches on an entire small positive interval.
Admissibility may impose any finite analytic equality, strict inequality,
or nonnegative inequality. Only accumulating admissible parameters are
assumed, rather than an existing analytic minimizer. Auxiliary for removing
finite fiber quantifiers in Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
theorem analytic_finite_minimum_branch {k l : ℕ}
    (V : Fin k → ℝ → ℝ) (hV : ∀ j, AnalyticAt ℝ (V j) 0)
    (C : Fin k → Fin l → ℝ → ℝ) (hC : ∀ j i, AnalyticAt ℝ (C j i) 0)
    (requirement : Fin l → AnalyticSignRequirement)
    (h : ∃ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0), ∃ j : Fin k,
      ∀ i, (requirement i).Holds (C j i t)) :
    ∃ j : Fin k, ∀ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0),
      (∀ i, (requirement i).Holds (C j i t)) ∧
        ∀ b : Fin k, (∀ i, (requirement i).Holds (C b i t)) → V j t ≤ V b t := by
  classical
  let eligible : Fin k → ℝ → Prop := fun j t => ∀ i, (requirement i).Holds (C j i t)
  have hmin : ∃ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0), ∃ j : Fin k,
      eligible j t ∧ ∀ b, eligible b t → V j t ≤ V b t := by
    apply h.mono
    rintro t ⟨j, hj⟩
    let candidates : Finset (Fin k) := Finset.univ.filter (fun b => eligible b t)
    have hcandidates : candidates.Nonempty :=
      ⟨j, Finset.mem_filter.mpr ⟨Finset.mem_univ j, hj⟩⟩
    obtain ⟨b, hb, hleast⟩ := candidates.exists_min_image (fun a => V a t) hcandidates
    refine ⟨b, (Finset.mem_filter.mp hb).2, fun a ha => ?_⟩
    exact hleast a (Finset.mem_filter.mpr ⟨Finset.mem_univ a, ha⟩)
  obtain ⟨j, hj⟩ := frequently_exists.mp hmin
  have hselected : ∀ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0), eligible j t :=
    analytic_finite_sign_constraints_of_frequent (C j) (hC j) requirement
      (hj.mono (fun t ht => ht.1))
  have hcompare (b : Fin k) : ∀ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0),
      eligible b t → V j t ≤ V b t := by
    by_cases hb : ∃ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0), eligible b t
    · have hbnear : ∀ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0), eligible b t :=
        analytic_finite_sign_constraints_of_frequent (C b) (hC b) requirement hb
      have hdiff : ∃ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0),
          AnalyticSignRequirement.nonnegative.Holds (V b t - V j t) :=
        (hj.and_eventually hbnear).mono (fun t ht => sub_nonneg.mpr (ht.1.2 b ht.2))
      have hbound := analytic_sign_requirement_of_frequent
        (fun t => V b t - V j t) ((hV b).fun_sub (hV j)) .nonnegative hdiff
      exact hbound.mono (fun t ht _ => sub_nonneg.mp ht)
    · exact (not_frequently.mp hb).mono (fun t ht hbt => (ht hbt).elim)
  refine ⟨j, ?_⟩
  filter_upwards [hselected, eventually_all.mpr hcompare] with t ht hc
  exact ⟨ht, hc⟩

/-- The lower objective branch `-t` is excluded by positivity, while the
branch with value `t` is admissible. These nonconstant analytic functions
satisfy all hypotheses of both finite selection theorems simultaneously.
Auxiliary example for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
example : let V : Fin 2 → ℝ → ℝ := fun j t => if j = 0 then -t else t
    let C : Fin 2 → Fin 1 → ℝ → ℝ := fun j _ t => V j t
    ∃ j : Fin 2, ∀ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0),
      (∀ i : Fin 1, 0 < C j i t) ∧ ∀ b : Fin 2,
        (∀ i : Fin 1, 0 < C b i t) → V j t ≤ V b t := by
  intro V C
  have hV (j : Fin 2) : AnalyticAt ℝ (V j) 0 := by
    fin_cases j
    · exact analyticAt_id.fun_neg
    · exact analyticAt_id
  apply analytic_finite_minimum_branch V hV C (fun j _ => hV j) (fun _ => .positive)
  have : NeBot (nhdsWithin (0 : ℝ) (Ioi 0)) := nhdsWithin_Ioi_neBot le_rfl
  have hp : ∀ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0), 0 < t := self_mem_nhdsWithin
  exact hp.frequently.mono (fun t ht => ⟨1, fun _ => ht⟩)

end Transformer.Normalization
