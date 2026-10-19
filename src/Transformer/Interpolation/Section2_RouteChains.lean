module

public import Transformer.Interpolation.Section2_GreatArcs
public import Mathlib.Topology.MetricSpace.Thickening

/-!
# Chains of balls along generic great circle arcs

ArXiv:2411.04551v3, §2.2, proof of `prop: compression`, Step 3: chains
of balls `𝓑_{ℓ,n}` inside the region carry the mass of each packing ball
to the anchor ball. Routes along great circle arcs between generic start
and end points, inside an open set closed under the geodesic hull, give
for every small radius `R` chains of points with steps below `R / 2`
whose `R`-balls stay inside the open set and miss the `R / 2`-balls
around the start and end points of the other routes: each arc is a
compact subset of the open set off those finitely many points, so a
thickening of it is too.
-/

@[expose] public section

open Set Filter Topology

namespace Transformer.Interpolation

variable {d : ℕ}

/-- **Route chains.** Start and end points in an open set closed under
the geodesic hull, every end point off the line of its start point and
the start and end points of every route off the planes of the other
routes: for every small `R` there are chains from each start point to its
end point, with steps below `R / 2`, whose `R`-balls stay in the open set
and miss the `R / 2`-balls around the start and end points of the other
routes. Source auxiliary: arXiv:2411.04551v3, §2.2, proof of
`prop: compression`, Step 3, `eq: homotopy`. -/
theorem eventually_route_chains {P : ℕ} (U : Set (SSphere d)) (hU : IsOpen U)
    (hconv : ∀ A ⊆ U, convG d A ⊆ U) (ω e : Fin P → SSphere d)
    (hωU : ∀ p, ω p ∈ U) (heU : ∀ p, e p ∈ U)
    (hline : ∀ p, ∀ α : ℝ, (e p : EucSpace d) ≠ α • (ω p : EucSpace d))
    (hstart : ∀ p q, p ≠ q → ∀ α β : ℝ,
      (ω q : EucSpace d) ≠ α • (ω p : EucSpace d) + β • (e p : EucSpace d))
    (hend : ∀ p q, p ≠ q → ∀ α β : ℝ,
      (e p : EucSpace d) ≠ α • (ω q : EucSpace d) + β • (e q : EucSpace d)) :
    ∀ᶠ R in 𝓝[>] (0 : ℝ), ∃ (a : Fin P → ℕ → SSphere d) (n : Fin P → ℕ),
      (∀ p, a p 0 = ω p ∧ a p (n p) = e p) ∧
      (∀ p, ∀ l < n p, Metric.ball (a p (l + 1)) (R / 2) ⊆ Metric.ball (a p l) R) ∧
      (∀ p, ∀ l < n p, Metric.ball (a p l) R ⊆ U) ∧
      ∀ p q, p ≠ q → ∀ l < n p, Disjoint (Metric.ball (a p l) R) (Metric.ball (ω q) (R / 2)) ∧
        Disjoint (Metric.ball (a p l) R) (Metric.ball (e q) (R / 2)) := by
  let K (p : Fin P) : Set (SSphere d) := arcSphere (ω p) (e p) (hline p) '' Icc 0 1
  let W (p : Fin P) : Set (SSphere d) := U ∩ ⋂ q, ⋂ (_ : q ≠ p), ({ω q}ᶜ ∩ {e q}ᶜ)
  have hW (p : Fin P) : IsOpen (W p) := hU.inter (isOpen_iInter_of_finite fun q =>
    isOpen_iInter_of_finite fun _ => isOpen_compl_singleton.inter isOpen_compl_singleton)
  have hKW (p : Fin P) : K p ⊆ W p := by
    rintro _ ⟨t, ht, rfl⟩
    refine ⟨hconv {ω p, e p} (insert_subset (hωU p) (singleton_subset_iff.mpr (heU p)))
      (arcSphere_mem_convG _ _ _ t ht), mem_iInter₂.mpr fun q hqp => ⟨?_, ?_⟩⟩
    · exact arcSphere_ne_of_avoid _ _ _ _ (hstart p q (Ne.symm hqp)) t
    · exact arcSphere_ne_of_avoid _ _ _ _ (hend q p hqp) t
  have hδ (p : Fin P) := ((isCompact_Icc.image (continuous_arcSphere (ω p) (e p)
    (hline p))).exists_thickening_subset_open (hW p) (hKW p))
  choose δ hδpos hδW using hδ
  have hev : ∀ᶠ R in 𝓝[>] (0 : ℝ), ∀ p, R < 2 * δ p / 3 := eventually_all.mpr fun p =>
    eventually_nhdsWithin_of_eventually_nhds (eventually_lt_nhds (by linarith [hδpos p]))
  filter_upwards [self_mem_nhdsWithin, hev] with R hR hRδ
  have hR : 0 < R := hR
  have hchain (p : Fin P) := exists_arc_chain (ω p) (e p) (hline p) (R / 2) (half_pos hR)
  choose n hn hstep using hchain
  let a (p : Fin P) (l : ℕ) : SSphere d := arcSphere (ω p) (e p) (hline p) ((l : ℝ) / n p)
  have hK (p : Fin P) {l : ℕ} (hl : l ≤ n p) : a p l ∈ K p :=
    ⟨(l : ℝ) / n p, ⟨by positivity, div_le_one_of_le₀ (by exact_mod_cast hl) (by positivity)⟩,
      rfl⟩
  have hnear (p : Fin P) {l : ℕ} (hl : l ≤ n p) {y : SSphere d} (hy : dist y (a p l) < δ p) :
      y ∈ W p := hδW p (Metric.mem_thickening_iff.mpr ⟨a p l, hK p hl, hy⟩)
  have hfar (p q : Fin P) (hpq : p ≠ q) {l : ℕ} (hl : l ≤ n p) (y : SSphere d)
      (hy : y = ω q ∨ y = e q) (z : SSphere d) (hz1 : z ∈ Metric.ball (a p l) R)
      (hz2 : z ∈ Metric.ball y (R / 2)) : False := by
    have hdist : dist y (a p l) < δ p := by
      rw [Metric.mem_ball] at hz1 hz2
      linarith [dist_triangle y z (a p l), dist_comm y z, hRδ p]
    have hmem := (mem_iInter₂.mp (hnear p hl hdist).2) q (Ne.symm hpq)
    rcases hy with rfl | rfl
    · exact hmem.1 rfl
    · exact hmem.2 rfl
  refine ⟨a, n, fun p => ⟨?_, ?_⟩, fun p l hl => ?_, fun p l hl y hy => ?_,
    fun p q hpq l hl => ⟨disjoint_left.mpr fun z hz1 hz2 => hfar p q hpq hl.le _ (Or.inl rfl) z
      hz1 hz2, disjoint_left.mpr fun z hz1 hz2 => hfar p q hpq hl.le _ (Or.inr rfl) z hz1 hz2⟩⟩
  · change arcSphere _ _ _ (((0 : ℕ) : ℝ) / n p) = ω p
    rw [Nat.cast_zero, zero_div, arcSphere_zero]
  · change arcSphere _ _ _ ((n p : ℝ) / n p) = e p
    rw [div_self (Nat.cast_ne_zero.mpr (hn p).ne'), arcSphere_one]
  · exact Metric.ball_subset_ball' (by linarith [hstep p l hl])
  · exact (hnear p hl.le (by linarith [Metric.mem_ball.mp hy, hRδ p, hδpos p])).1

/-- The hypotheses of `eventually_route_chains` are satisfiable: one route
between two orthogonal points of the whole circle. -/
example : ∃ ω e : Fin 1 → SSphere 2, IsOpen (univ : Set (SSphere 2)) ∧
    (∀ A ⊆ (univ : Set (SSphere 2)), convG 2 A ⊆ univ) ∧
    (∀ p, ∀ α : ℝ, (e p : EucSpace 2) ≠ α • (ω p : EucSpace 2)) ∧
    ∀ p q : Fin 1, p ≠ q → ∀ α β : ℝ,
      (ω q : EucSpace 2) ≠ α • (ω p : EucSpace 2) + β • (e p : EucSpace 2) := by
  refine ⟨fun _ => ⟨EuclideanSpace.single 0 1, by simp⟩,
    fun _ => ⟨EuclideanSpace.single 1 1, by simp⟩, isOpen_univ, fun _ _ => subset_univ _,
    fun p α h => ?_, fun p q hpq => absurd (Subsingleton.elim p q) hpq⟩
  have h1 := congrArg (fun v : EucSpace 2 => v 1) h
  simp at h1

end Transformer.Interpolation
