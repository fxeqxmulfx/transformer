/-
# Collapsing balls and routing them to their targets

ArXiv:2411.04551v3, §2.2, proof of `prop: compression`, Step 3, in
dimension `d ≥ 3`: finitely many disjoint balls inside an open region
closed under geodesic hulls are each collapsed into a small ball around a
start point, and the start points are routed to end points near their
targets along great circle arcs, each route avoiding the other start and
end points. The phases are local in the region, and the mass of each ball
arrives up to a prescribed fraction.
-/

import Transformer.Interpolation.Section2_GenericRoutePoints
import Transformer.Interpolation.Section2_RouteChains
import Transformer.Interpolation.Section2_CollapseChains

open scoped ENNReal
open Real MeasureTheory Set Filter Topology

namespace Transformer.Interpolation

variable {d : ℕ}

/-- **Routed collapses.** In dimension `d ≥ 3`, finitely many disjoint
balls of radius at most `2` inside an open region closed under geodesic
hulls, and a point of the region for each ball: phases local in the
region carry all but a fraction `η` of the mass of each ball into a
small ball inside the ball of radius `s` around its point, the small
balls being pairwise disjoint. Source: arXiv:2411.04551v3, §2.2, proof
of `prop: compression`, Step 3, the collapse of the packing balls and
their chains of balls. -/
theorem exists_routed_collapses (hd : 3 ≤ d) (ν : Perspective.ProbSphere d)
    (U : Set (SSphere d)) (hU : IsOpen U) (hconv : ∀ A ⊆ U, convG d A ⊆ U) {P : ℕ}
    (z : Fin P → SSphere d) (ρ : Fin P → ℝ) (hρ : ∀ p, 0 < ρ p ∧ ρ p ≤ 2)
    (hBU : ∀ p, Metric.ball (z p) (ρ p) ⊆ U)
    (hdisj : Pairwise (Function.onFun Disjoint fun p => Metric.ball (z p) (ρ p)))
    (t : Fin P → SSphere d) (ht : ∀ p, t p ∈ U) (s : ℝ) (hs : 0 < s) (η : ℝ) (hη : 0 < η)
    (hη1 : η ≤ 1) :
    ∃ (L : List (AutonomousPhase d)) (e : Fin P → SSphere d) (r : ℝ), AllLocalIn L U ∧
      0 < r ∧ (∀ p, Metric.ball (e p) r ⊆ Metric.ball (t p) s) ∧
      Pairwise (Function.onFun Disjoint fun p => Metric.ball (e p) r) ∧
      ∀ p, ENNReal.ofReal (1 - η) * (ν : Measure (SSphere d)) (Metric.ball (z p) (ρ p)) ≤
        (phaseSequenceLaw L ν : Measure (SSphere d)) (Metric.ball (e p) r) := by
  classical
  have hij : (⟨0, by omega⟩ : Fin d) ≠ ⟨1, by omega⟩ := Fin.ne_of_val_ne zero_ne_one
  obtain ⟨ω, e, hmem, hline, hstart, hend⟩ := exists_generic_pairs hd
    (fun p => Metric.ball (z p) (ρ p)) (fun p => Metric.ball (t p) (s / 2) ∩ U)
    (fun _ => Metric.isOpen_ball) (fun _ => Metric.isOpen_ball.inter hU)
    (fun p => ⟨z p, Metric.mem_ball_self (hρ p).1⟩)
    (fun p => ⟨t p, Metric.mem_ball_self (half_pos hs), ht p⟩)
  have hev := eventually_route_chains U hU hconv ω e (fun p => hBU p (hmem p).1)
    (fun p => (hmem p).2.2) hline hstart hend
  have hlt {c : ℝ} (hc : 0 < c) : ∀ᶠ R in 𝓝[>] (0 : ℝ), R < c :=
    eventually_nhdsWithin_of_eventually_nhds (eventually_lt_nhds hc)
  have hz : ∀ᶠ R in 𝓝[>] (0 : ℝ), ∀ p, R < 2 * (ρ p - dist (ω p) (z p)) :=
    eventually_all.mpr fun p => hlt (by linarith [Metric.mem_ball.mp (hmem p).1])
  have hee : ∀ᶠ R in 𝓝[>] (0 : ℝ), ∀ p q, p ≠ q → R < dist (e p) (e q) := by
    refine eventually_all.mpr fun p => eventually_all.mpr fun q => ?_
    by_cases hpq : p = q
    · exact Eventually.of_forall fun _ h => absurd hpq h
    · refine (hlt (dist_pos.mpr fun h => hend p q hpq 0 1 ?_)).mono fun _ h _ => h
      rw [zero_smul, one_smul, zero_add, h]
  obtain ⟨R, hR, ⟨a, n, hends, hstep, hsub, hsep⟩, hR2, hRs, hRz, hRe⟩ :=
    (eventually_mem_nhdsWithin.and (hev.and ((hlt two_pos).and ((hlt hs).and
      (hz.and hee))))).exists
  have hR0 : 0 < R := hR
  set N := ∑ p, n p
  have hε : 0 < η / (N + 1) := by positivity
  have hε1 : η / (N + 1) ≤ 1 :=
    div_le_one_of_le₀ (by linarith [(N.cast_nonneg : (0 : ℝ) ≤ N)]) (by positivity)
  obtain ⟨L₁, hL₁, hm₁⟩ := exists_disjoint_collapses _ _ hij Finset.univ z ω ρ
    (fun p => Metric.ball (ω p) (R / 2)) (fun p _ => (hρ p).1) (fun p _ => (hρ p).2)
    (fun _ _ => Metric.isOpen_ball) (fun p _ => Metric.ball_subset_ball' (by linarith [hRz p]))
    (fun p _ => Metric.mem_ball_self (half_pos hR0)) (fun p _ q _ hpq => hdisj hpq)
    (η / (N + 1)) hε hε1 ν
  have hsepL : (List.finRange P).Pairwise fun p q =>
      Disjoint (⋃ l < n p, Metric.ball (a p l) R) (Metric.ball (a q 0) (R / 2)) ∧
        Disjoint (⋃ l < n q, Metric.ball (a q l) R) (Metric.ball (a p (n p)) (R / 2)) :=
    (List.nodup_finRange P).pairwise_of_forall_ne fun p _ q _ hpq => by
      rw [(hends q).1, (hends p).2]
      exact ⟨disjoint_iUnion₂_left.mpr fun l hl => (hsep p q hpq l hl).1,
        disjoint_iUnion₂_left.mpr fun l hl => (hsep q p (Ne.symm hpq) l hl).2⟩
  obtain ⟨L₂, hL₂, hm₂⟩ := exists_collapse_routes _ _ hij a n R (R / 2) (half_pos hR0)
    (half_le_self hR0.le) hR2.le (List.finRange P) (fun p _ l hl => hstep p l hl) hsepL
    (η / (N + 1)) hε hε1 (phaseSequenceLaw L₁ ν)
  refine ⟨L₁ ++ L₂, e, R / 2, (hL₁.mono (iUnion₂_subset fun p _ => hBU p)).append
    (hL₂.mono (iUnion₂_subset fun p _ => iUnion₂_subset fun l hl => hsub p l hl)),
    half_pos hR0, fun p => Metric.ball_subset_ball' ?_, fun p q hpq => ?_, fun p => ?_⟩
  · linarith [Metric.mem_ball.mp (hmem p).2.1]
  · exact Metric.ball_disjoint_ball (by linarith [hRe p q hpq])
  · have hnp : (n p : ℝ) ≤ N := by
      exact_mod_cast Finset.single_le_sum (fun q _ => Nat.zero_le (n q)) (Finset.mem_univ p)
    have hk : ((n p + 1 : ℕ) : ℝ) * (η / (N + 1)) ≤ η := by
      rw [mul_div_assoc', div_le_iff₀ (by positivity)]
      push_cast
      nlinarith
    have hc : ENNReal.ofReal (1 - η) ≤ ENNReal.ofReal (1 - η / (N + 1)) ^ (n p + 1) := by
      rw [← ENNReal.ofReal_pow (by linarith)]
      refine ENNReal.ofReal_le_ofReal ?_
      have hb := one_add_mul_le_pow (a := -(η / (N + 1))) (by linarith) (n p + 1)
      rw [mul_neg, ← sub_eq_add_neg, ← sub_eq_add_neg] at hb
      linarith
    have h₁ := hm₁ p (Finset.mem_univ p)
    have h₂ := hm₂ p (List.mem_finRange p)
    rw [(hends p).1, (hends p).2] at h₂
    rw [phaseSequenceLaw_append]
    calc ENNReal.ofReal (1 - η) * (ν : Measure (SSphere d)) (Metric.ball (z p) (ρ p))
        ≤ ENNReal.ofReal (1 - η / (N + 1)) ^ (n p + 1) *
            (ν : Measure (SSphere d)) (Metric.ball (z p) (ρ p)) := mul_le_mul_left hc _
      _ = ENNReal.ofReal (1 - η / (N + 1)) ^ n p * (ENNReal.ofReal (1 - η / (N + 1)) *
            (ν : Measure (SSphere d)) (Metric.ball (z p) (ρ p))) := by rw [pow_succ, mul_assoc]
      _ ≤ _ := (mul_le_mul_right h₁ _).trans h₂

/-- The hypotheses of `exists_routed_collapses` are satisfiable: one ball
of radius one in the whole sphere of `ℝ³`, radius and slack one. -/
example : 3 ≤ 3 ∧ IsOpen (univ : Set (SSphere 3)) ∧
    (∀ A ⊆ (univ : Set (SSphere 3)), convG 3 A ⊆ univ) ∧
    (∀ _p : Fin 1, (0 : ℝ) < 1 ∧ (1 : ℝ) ≤ 2) ∧
    Pairwise (Function.onFun Disjoint fun _p : Fin 1 =>
      Metric.ball (⟨EuclideanSpace.single 0 1, by simp⟩ : SSphere 3) 1) ∧
    (0 : ℝ) < 1 ∧ (1 : ℝ) ≤ 1 :=
  ⟨le_rfl, isOpen_univ, fun _ _ => subset_univ _, fun _ => ⟨one_pos, one_le_two⟩,
    fun p q h => absurd (Subsingleton.elim p q) h, one_pos, le_rfl⟩

end Transformer.Interpolation
