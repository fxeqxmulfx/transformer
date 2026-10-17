/-
# Disjoint collapses and collapse chains

ArXiv:2411.04551v3, §2.2, Step 3 of the proof of `prop: compression`:
the mass of each packing ball travels along a chain of balls into its
target ball, one application of `lem: tubular.mass.movement` per link,
and keeps the fraction `(1 - ε̄)^L` of its mass
(`eq: mass.lower.bound`).

Three bookkeeping results on the ball collapses of
`exists_collapse_phase`. Pairwise disjoint balls collapsed one after
another into open sets inside them each keep the fraction `1 - ε` of
their mass. A chain of points, each next small ball inside the current
collapse ball, carries a small ball of mass along it and keeps
`(1 - ε)^n` of it after `n` links. Finitely many chains run one after
another carry every small ball along its own chain, provided the tube
of each chain misses the small balls that wait for later chains and
those already brought by earlier ones.
-/

import Transformer.Interpolation.Section2_CollapsePhases

open Real MeasureTheory Set

namespace Transformer.Interpolation

variable {d : ℕ}

/-- **Disjoint collapses.** Pairwise disjoint proper balls, each with an
open set inside it around a point, are collapsed one after the other:
the phases are local in the union of the balls, and every open set ends
with the fraction `1 - ε` of the initial mass of its ball. Source:
arXiv:2411.04551v3, §2.2, Step 3, the packing balls. -/
theorem exists_disjoint_collapses (i j : Fin d) (hij : i ≠ j) {ι : Type*} (S : Finset ι)
    (z ω : ι → SSphere d) (R : ι → ℝ) (A : ι → Set (SSphere d))
    (hR : ∀ p ∈ S, 0 < R p) (hR2 : ∀ p ∈ S, R p ≤ 2) (hA : ∀ p ∈ S, IsOpen (A p))
    (hAB : ∀ p ∈ S, A p ⊆ Metric.ball (z p) (R p)) (hω : ∀ p ∈ S, ω p ∈ A p)
    (hdisj : (S : Set ι).PairwiseDisjoint fun p => Metric.ball (z p) (R p))
    (ε : ℝ) (hε : 0 < ε) (hε1 : ε ≤ 1) (ν : Perspective.ProbSphere d) :
    ∃ L : List (AutonomousPhase d), AllLocalIn L (⋃ p ∈ S, Metric.ball (z p) (R p)) ∧
      ∀ p ∈ S, ENNReal.ofReal (1 - ε) * (ν : Measure (SSphere d)) (Metric.ball (z p) (R p)) ≤
        (phaseSequenceLaw L ν : Measure (SSphere d)) (A p) := by
  classical
  induction S using Finset.induction_on generalizing ν with
  | empty => exact ⟨[], fun P hP => by simp at hP, fun p hp => by simp at hp⟩
  | @insert q S hq ih =>
    have hqS := Finset.mem_insert_self q S
    have hsub (p : ι) (hp : p ∈ S) : p ∈ insert q S := Finset.mem_insert_of_mem hp
    obtain ⟨P, Φ, hP, hmass⟩ := exists_collapse_phase i j hij (z q) (ω q) (R q) (hR q hqS)
      (hR2 q hqS) (A q) (hA q hqS) (hAB q hqS) (hω q hqS) ε hε hε1 ν
    obtain ⟨L, hL, hLm⟩ := ih (fun p hp => hR p (hsub p hp)) (fun p hp => hR2 p (hsub p hp))
      (fun p hp => hA p (hsub p hp)) (fun p hp => hAB p (hsub p hp))
      (fun p hp => hω p (hsub p hp))
      (hdisj.subset (Finset.coe_subset.mpr (Finset.subset_insert q S))) (P.2 ν 1)
    have hne (p : ι) (hp : p ∈ S) : Disjoint (Metric.ball (z p) (R p)) (Metric.ball (z q) (R q)) :=
      hdisj (Finset.mem_coe.mpr (hsub p hp)) (Finset.mem_coe.mpr hqS) fun h => hq (h ▸ hp)
    refine ⟨P :: L, ?_, fun p hp => ?_⟩
    · intro Q hQ
      rcases List.mem_cons.mp hQ with rfl | hQ
      · exact ⟨Φ, hP.mono (subset_biUnion_of_mem (u := fun p => Metric.ball (z p) (R p)) hqS)⟩
      · obtain ⟨Ψ, hΨ⟩ := hL Q hQ
        exact ⟨Ψ, hΨ.mono (biUnion_subset_biUnion_left fun p hp => hsub p hp)⟩
    · rw [phaseSequenceLaw_cons]
      rcases Finset.mem_insert.mp hp with rfl | hp
      · refine hmass.trans (hL.le_of_disjoint _ _ (Set.disjoint_left.mpr fun x hxA hxU => ?_))
        obtain ⟨p', hp', hx⟩ := mem_iUnion₂.mp hxU
        exact Set.disjoint_left.mp (hne p' hp') hx (hAB p hqS hxA)
      · exact (mul_le_mul_right (hP.le_of_disjoint ν _ (hne p hp)) _).trans (hLm p hp)

/-- **A collapse chain.** Along points `a 0, …, a n` whose next small
ball `B(a (l+1), r)` lies in the current collapse ball `B(a l, R)`, the
collapses carry the small ball at `a 0` to the one at `a n`, keeping the
fraction `(1 - ε)^n` of its mass; the phases are local in the tube of
collapse balls. Source: arXiv:2411.04551v3, §2.2, Step 3, the chains of
balls `eq: homotopy`, and `eq: mass.lower.bound`. -/
theorem exists_collapse_chain (i j : Fin d) (hij : i ≠ j) (a : ℕ → SSphere d) (n : ℕ)
    (R r : ℝ) (hr : 0 < r) (hrR : r ≤ R) (hR2 : R ≤ 2)
    (hstep : ∀ l < n, Metric.ball (a (l + 1)) r ⊆ Metric.ball (a l) R)
    (ε : ℝ) (hε : 0 < ε) (hε1 : ε ≤ 1) (ν : Perspective.ProbSphere d) :
    ∃ L : List (AutonomousPhase d), AllLocalIn L (⋃ l < n, Metric.ball (a l) R) ∧
      ENNReal.ofReal (1 - ε) ^ n * (ν : Measure (SSphere d)) (Metric.ball (a 0) r) ≤
        (phaseSequenceLaw L ν : Measure (SSphere d)) (Metric.ball (a n) r) := by
  induction n generalizing a ν with
  | zero => exact ⟨[], fun P hP => by simp at hP, by rw [pow_zero, one_mul]; exact le_rfl⟩
  | succ n ih =>
    obtain ⟨P, Φ, hP, hmass⟩ := exists_collapse_phase i j hij (a 0) (a 1) R (hr.trans_le hrR)
      hR2 (Metric.ball (a 1) r) Metric.isOpen_ball (hstep 0 n.succ_pos)
      (Metric.mem_ball_self hr) ε hε hε1 ν
    obtain ⟨L, hL, hLm⟩ := ih (fun l => a (l + 1)) (fun l hl => hstep (l + 1) (by omega))
      (P.2 ν 1)
    refine ⟨P :: L, ?_, ?_⟩
    · intro Q hQ
      rcases List.mem_cons.mp hQ with rfl | hQ
      · exact ⟨Φ, hP.mono (subset_biUnion_of_mem (u := fun l => Metric.ball (a l) R)
          n.succ_pos)⟩
      · obtain ⟨Ψ, hΨ⟩ := hL Q hQ
        refine ⟨Ψ, hΨ.mono (iUnion₂_subset fun l hl => ?_)⟩
        exact subset_biUnion_of_mem (u := fun l => Metric.ball (a l) R) (by omega : l + 1 < n + 1)
    · rw [phaseSequenceLaw_cons, pow_succ, mul_assoc]
      refine (mul_le_mul_right ?_ _).trans hLm
      exact (mul_le_mul_right (measure_mono (Metric.ball_subset_ball hrR)) _).trans hmass

/-- **Collapse routes.** Chains run one after another in the listed
order carry every small ball along its own chain, keeping the fraction
`(1 - ε)^n` of its mass for a chain of `n` links, when the tube of each
chain misses the starting small ball of every later chain and the tube
of every later chain misses its own final small ball. The phases are
local in the union of the tubes. Source: arXiv:2411.04551v3, §2.2,
Step 3, the chains of all packing balls. -/
theorem exists_collapse_routes (i j : Fin d) (hij : i ≠ j) {ι : Type*}
    (a : ι → ℕ → SSphere d) (n : ι → ℕ) (R r : ℝ) (hr : 0 < r) (hrR : r ≤ R) (hR2 : R ≤ 2)
    (order : List ι)
    (hstep : ∀ p ∈ order, ∀ l < n p, Metric.ball (a p (l + 1)) r ⊆ Metric.ball (a p l) R)
    (hsep : order.Pairwise fun p q =>
      Disjoint (⋃ l < n p, Metric.ball (a p l) R) (Metric.ball (a q 0) r) ∧
        Disjoint (⋃ l < n q, Metric.ball (a q l) R) (Metric.ball (a p (n p)) r))
    (ε : ℝ) (hε : 0 < ε) (hε1 : ε ≤ 1) (ν : Perspective.ProbSphere d) :
    ∃ L : List (AutonomousPhase d),
      AllLocalIn L (⋃ p ∈ order, ⋃ l < n p, Metric.ball (a p l) R) ∧
      ∀ p ∈ order, ENNReal.ofReal (1 - ε) ^ n p *
          (ν : Measure (SSphere d)) (Metric.ball (a p 0) r) ≤
        (phaseSequenceLaw L ν : Measure (SSphere d)) (Metric.ball (a p (n p)) r) := by
  induction order generalizing ν with
  | nil => exact ⟨[], fun P hP => by simp at hP, fun p hp => by simp at hp⟩
  | cons p rest ih =>
    obtain ⟨hhead, hrest⟩ := List.pairwise_cons.mp hsep
    obtain ⟨Lp, hLp, hmp⟩ := exists_collapse_chain i j hij (a p) (n p) R r hr hrR hR2
      (hstep p List.mem_cons_self) ε hε hε1 ν
    obtain ⟨L', hL', hm'⟩ := ih (fun q hq => hstep q (List.mem_cons_of_mem p hq)) hrest
      (phaseSequenceLaw Lp ν)
    refine ⟨Lp ++ L', ?_, fun q hq => ?_⟩
    · refine AllLocalIn.append (hLp.mono ?_) (hL'.mono ?_)
      · exact subset_biUnion_of_mem (u := fun q => ⋃ l < n q, Metric.ball (a q l) R)
          List.mem_cons_self
      · exact biUnion_subset_biUnion_left fun q hq => List.mem_cons_of_mem p hq
    · rw [phaseSequenceLaw_append]
      rcases List.mem_cons.mp hq with rfl | hq
      · refine hmp.trans (hL'.le_of_disjoint _ _ (Set.disjoint_left.mpr fun x hx hxU => ?_))
        obtain ⟨q', hq', hxq⟩ := mem_iUnion₂.mp hxU
        exact Set.disjoint_left.mp (hhead q' hq').2 hxq hx
      · refine (mul_le_mul_right ?_ _).trans (hm' q hq)
        exact hLp.le_of_disjoint ν _ (hhead q hq).1.symm

/-- The hypotheses of `exists_collapse_chain` are satisfiable: the
constant chain at a point, with radii one half and one. -/
example (z : SSphere 2) : (0 : ℝ) < 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧ (1 : ℝ) ≤ 2 ∧
    ∀ l < 3, Metric.ball ((fun _ : ℕ => z) (l + 1)) (1 / 2) ⊆
      Metric.ball ((fun _ : ℕ => z) l) 1 :=
  ⟨by norm_num, by norm_num, by norm_num,
    fun _ _ => Metric.ball_subset_ball (by norm_num)⟩

/-- The hypotheses of `exists_disjoint_collapses` are satisfiable: one
ball of radius one around a point, collapsed into itself. -/
example (z : SSphere 2) :
    ((({()} : Finset Unit)) : Set Unit).PairwiseDisjoint (fun _ => Metric.ball z 1) ∧
      IsOpen (Metric.ball z 1) ∧ z ∈ Metric.ball z 1 :=
  ⟨by rw [Finset.coe_singleton]; exact Set.pairwiseDisjoint_singleton _ _, Metric.isOpen_ball,
    Metric.mem_ball_self one_pos⟩

end Transformer.Interpolation
