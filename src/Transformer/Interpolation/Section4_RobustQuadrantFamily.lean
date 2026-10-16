/-
# Common robust geometry for original quadrant law matching

ArXiv:2411.04551v3, §1, `thm: targets.atoms`, and §3.2,
protected finite transfers. Choose one positive generic intermediate
family relative to both the old mean poles and the target atoms.
The generic exclusions protect previously moved means in the first
stage and both unmoved and already targeted means in the second.

All finitely many independent triples admit one positive perturbation
radius. Intersect it with positive-quadrant cap radii for both the
intermediate and final atom families. Thus concentrated laws and
their actual perturbed means meet the needed plane conditions without
assuming preservation of their mean directions during contraction.
-/

import Transformer.Interpolation.Section4_RobustPlanes
import Transformer.Interpolation.Section4_GenericQuadrantFamily
import Transformer.Interpolation.Section3_SeparatedCaps

open Real MeasureTheory Set

namespace Transformer.Interpolation

variable {d N : ℕ}

/-- Generic positive intermediate atoms admit one common robust
radius for both stages of original law matching. Source auxiliary:
arXiv:2411.04551v3, §1 and §3.2, finite protected interpolation. -/
theorem exists_robust_quadrant_matching_family (hd : 3 ≤ d) (hN : 1 ≤ N)
    (X Y : Idx N → SSphere d) (hXinj : Function.Injective X) (hYinj : Function.Injective Y)
    (hX : ∀ k, X k ∈ positiveQuadrant d) (hY : ∀ k, Y k ∈ positiveQuadrant d) :
    ∃ (Z : Idx N → SSphere d) (r : ℝ), Function.Injective Z ∧ 0 < r ∧ r ≤ 1 ∧
      (∀ k, Metric.closedBall (Z k) r ⊆ positiveQuadrant d) ∧
      (∀ k, Metric.closedBall (Y k) r ⊆ positiveQuadrant d) ∧
      (∀ j k, k ≠ j → ∀ α β : ℝ,
        (X k : EucSpace d) ≠ α • (X j : EucSpace d) + β • (Z j : EucSpace d)) ∧
      (∀ j k, k ≠ j → ∀ m : EucSpace d, dist m (Z k : EucSpace d) ≤ r →
        ∀ α β : ℝ, m ≠ α • (X j : EucSpace d) + β • (Z j : EucSpace d)) ∧
      (∀ j k, k ≠ j → ∀ p m : EucSpace d,
        dist p (Z j : EucSpace d) ≤ r → dist m (Z k : EucSpace d) ≤ r →
        ∀ α β : ℝ, m ≠ α • p + β • (Y j : EucSpace d)) ∧
      ∀ j k, k ≠ j → ∀ p m : EucSpace d,
        dist p (Z j : EucSpace d) ≤ r → dist m (Y k : EucSpace d) ≤ r →
        ∀ α β : ℝ, m ≠ α • p + β • (Y j : EucSpace d) := by
  classical
  let i : Fin d := ⟨0, by omega⟩
  let : Nonempty (Idx N) := ⟨⟨0, by omega⟩⟩
  let A : (Idx N ⊕ Idx N) → EucSpace d :=
    Sum.elim (fun k => (X k : EucSpace d)) (fun k => (Y k : EucSpace d))
  obtain ⟨Z, hZinj, hZ, hplanes, hpairs⟩ := exists_injective_generic_quadrant_family hd A N
  have hnc (j : Idx N) (a : Idx N ⊕ Idx N) (α : ℝ) :
      (Z j : EucSpace d) ≠ α • A a := generic_quadrant_family_noncollinear A Z hplanes j a α
  let J := {jk : Idx N × Idx N // jk.2 ≠ jk.1}
  let P (a : Fin 3 × J) : SSphere d := if a.1 = 0 then X a.2.1.1 else Y a.2.1.1
  let Q (a : Fin 3 × J) : EucSpace d := Z a.2.1.1
  let M (a : Fin 3 × J) : EucSpace d := if a.1 = 2 then (Y a.2.1.2 : EucSpace d) else Z a.2.1.2
  have hP (a : Fin 3 × J) : (P a : EucSpace d) ≠ 0 := by
    intro he
    have hn := mem_sphere_zero_iff_norm.mp (P a).2
    rw [he, norm_zero] at hn
    norm_num at hn
  have hQ (a : Fin 3 × J) (α : ℝ) : Q a ≠ α • (P a : EucSpace d) := by
    rcases a with ⟨c, ⟨⟨j, k⟩, hkj⟩⟩
    fin_cases c
    · exact hnc j (Sum.inl j) α
    · exact hnc j (Sum.inr j) α
    · exact hnc j (Sum.inr j) α
  have hM (a : Fin 3 × J) (α β : ℝ) : M a ≠ α • (P a : EucSpace d) + β • Q a := by
    rcases a with ⟨c, ⟨⟨j, k⟩, hkj⟩⟩
    fin_cases c
    · exact hpairs k j hkj (Sum.inl j) α β
    · exact hpairs k j hkj (Sum.inr j) α β
    · exact generic_old_outside_transfer_plane A Z hplanes j (Sum.inr j) (Sum.inr k)
        (quadrant_unit_noncollinear i (Y j) (Y k) (hY j) (hY k) (fun he => hkj (hYinj he))) α β
  obtain ⟨R, hR, hrob⟩ := exists_finite_robust_plane_radius (fun a => (P a : EucSpace d)) Q M hP hQ hM
  obtain ⟨rz, hrz, hrz1, hcapZ, _⟩ := exists_separated_quadrant_caps Z hZ hZinj
  obtain ⟨ry, hry, _, hcapY, _⟩ := exists_separated_quadrant_caps Y hY hYinj
  let r := min R (min rz ry)
  have hr : 0 < r := by dsimp only [r]; positivity
  have hrR : r ≤ R := min_le_left _ _
  have hrrz : r ≤ rz := (min_le_right _ _).trans (min_le_left _ _)
  have hrry : r ≤ ry := (min_le_right _ _).trans (min_le_right _ _)
  refine ⟨Z, r, hZinj, hr, hrrz.trans hrz1, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro k x hx
    exact hcapZ k ((Metric.mem_closedBall.mp hx).trans hrrz)
  · intro k x hx
    exact hcapY k ((Metric.mem_closedBall.mp hx).trans hrry)
  · intro j k hkj α β
    exact generic_old_outside_transfer_plane A Z hplanes j (Sum.inl j) (Sum.inl k)
      (quadrant_unit_noncollinear i (X j) (X k) (hX j) (hX k) (fun he => hkj (hXinj he))) α β
  · intro j k hkj m hm α β
    exact hrob (0, ⟨(j, k), hkj⟩) _ _ m (by simpa [P] using hR.le) (by simpa [Q] using hR.le)
      (hm.trans hrR) α β
  · intro j k hkj p m hp hm α β he
    apply hrob (1, ⟨(j, k), hkj⟩) (Y j : EucSpace d) p m
      (by simpa [P] using hR.le) (hp.trans hrR) (hm.trans hrR) β α
    exact he.trans (add_comm _ _)
  · intro j k hkj p m hp hm α β he
    apply hrob (2, ⟨(j, k), hkj⟩) (Y j : EucSpace d) p m
      (by simpa [P] using hR.le) (hp.trans hrR) (hm.trans hrR) β α
    exact he.trans (add_comm _ _)

/-- The robust geometry applies to actual concentrated laws, including
the selected law's actual normalized barycenter. Source auxiliary:
arXiv:2411.04551v3, §1 and §3.2; the mean direction may vary within
its concentration cap. -/
theorem plane_exclusion_for_concentrated_laws (z w p q : SSphere d)
    (ν σ : Perspective.ProbSphere d) (r : ℝ) (hr : 0 ≤ r) (hr1 : r ≤ 1)
    (hp : (p : EucSpace d) = barycenterDirection ν)
    (hν : (ν : Measure (SSphere d)).support ⊆ Metric.closedBall z r)
    (hσ : (σ : Measure (SSphere d)).support ⊆ Metric.closedBall w r)
    (hrob : ∀ p' m' : EucSpace d, dist p' (z : EucSpace d) ≤ r →
      dist m' (w : EucSpace d) ≤ r → ∀ α β : ℝ, m' ≠ α • p' + β • (q : EucSpace d)) :
    ∀ α β : ℝ, barycenter d σ ≠ α • (p : EucSpace d) + β • (q : EucSpace d) := by
  have hpcap := barycenter_pole_mem_closedBall z p ν r hr hr1 hν hp
  have hpdist : dist (p : EucSpace d) (z : EucSpace d) ≤ r := by
    simpa only [Metric.mem_closedBall, Subtype.dist_eq] using hpcap
  have hmdist : dist (barycenter d σ) (w : EucSpace d) ≤ r := by
    simpa only [dist_eq_norm] using norm_barycenter_sub_le_support_radius σ w r hσ
  exact hrob (p : EucSpace d) (barycenter d σ) hpdist hmdist

/-- A genuine nontrivial swap of two positive rational atom families
satisfies all dimension, nonempty-family and injectivity hypotheses. -/
example : ∃ X Y : Idx 2 → SSphere 3, 3 ≤ 3 ∧ 1 ≤ 2 ∧
    Function.Injective X ∧ Function.Injective Y ∧
      (∀ k, X k ∈ positiveQuadrant 3) ∧ (∀ k, Y k ∈ positiveQuadrant 3) := by
  let X (k : Idx 2) := quadrantThreePoint k.castSucc
  let Y (k : Idx 2) := X (Fin.rev k)
  have hXinj : Function.Injective X := by
    intro k l he
    exact Fin.castSucc_injective 2 (quadrantThreePoint_injective he)
  refine ⟨X, Y, le_rfl, by decide, hXinj, hXinj.comp Fin.rev_injective, ?_, ?_⟩
  · intro k
    exact quadrantThreePoint_mem k.castSucc
  · intro k
    exact quadrantThreePoint_mem (Fin.rev k).castSucc

/-- Actual Dirac laws at constructed generic points inhabit all
concentration, mean-direction and robust-plane hypotheses together. -/
example : ∃ (z w p q : SSphere 3) (ν σ : Perspective.ProbSphere 3) (r : ℝ),
    0 ≤ r ∧ r ≤ 1 ∧ (p : EucSpace 3) = barycenterDirection ν ∧
      (ν : Measure (SSphere 3)).support ⊆ Metric.closedBall z r ∧
      (σ : Measure (SSphere 3)).support ⊆ Metric.closedBall w r ∧
      ∀ p' m' : EucSpace 3, dist p' (z : EucSpace 3) ≤ r → dist m' (w : EucSpace 3) ≤ r →
        ∀ α β : ℝ, m' ≠ α • p' + β • (q : EucSpace 3) := by
  let X (k : Idx 2) := quadrantThreePoint k.castSucc
  have hinj : Function.Injective X := by
    intro k l he
    exact Fin.castSucc_injective 2 (quadrantThreePoint_injective he)
  have hQ (k : Idx 2) : X k ∈ positiveQuadrant 3 := quadrantThreePoint_mem k.castSucc
  obtain ⟨Z, r, _, hr, hr1, _, _, _, _, hrob, _⟩ :=
    exists_robust_quadrant_matching_family (d := 3) le_rfl (by decide) X X hinj hinj hQ hQ
  refine ⟨Z 0, Z 1, Z 0, X 0, Perspective.diracProb 3 (Z 0),
    Perspective.diracProb 3 (Z 1), r, hr.le, hr1, ?_, ?_, ?_, hrob 0 1 (by decide)⟩
  · simp [barycenterDirection, barycenter, Perspective.diracProb, mem_sphere_zero_iff_norm.mp (Z 0).2]
  · intro x hx
    rw [eq_of_mem_support_dirac hx]
    exact Metric.mem_closedBall.mpr (by simpa only [dist_self] using hr.le)
  · intro x hx
    rw [eq_of_mem_support_dirac hx]
    exact Metric.mem_closedBall.mpr (by simpa only [dist_self] using hr.le)

end Transformer.Interpolation
