/-
# The emergence of clusters in self-attention dynamics — the limits `a` and
  `b`, and bounded trajectories

§9 of arXiv:2305.05465v6, continued: the two limits `e:defab` that `l:fj`
produces, and the corollary `c:bounded` that reads them back on the tokens.

**What the source says and what is carried here.**

* Both statements are consequences of `l:fj`, and both are proved here from
  its conclusion taken as an explicit hypothesis — the monotonicity on
  `[0,+∞)` for `e:defab`, the uniform bound on each coordinate for
  `c:bounded`.  `l:fj` itself is in `Transformer.Clusters.Section9_Eigen`.

* `c:bounded` assumes that `V` has only real non-negative eigenvalues.  That
  hypothesis is used twice: to put each `φ*_k` under `l:fj` (`λ_k ≥ 0`), and
  to expand a vector in the eigenbasis, `z = Σ_k φ*_k(z) φ_k`.  The first is
  the coordinate bound below, the second is `hexp`; together they are the
  source's one-line proof.

Source: arXiv:2305.05465v6, `e:defab`, `c:bounded`.
-/

import Transformer.Clusters.Section9_Eigen

open scoped BigOperators
open Real Filter Topology

namespace Transformer
namespace Clusters

variable {d n m : ℕ}

/-! ### Bounded monotone convergence on a half-line -/

/-- A function antitone on `[0,+∞)` and bounded below there converges, and its
limit is a lower bound.  This is what turns the monotonicity of `l:fj` into
the limit `b` of `e:defab`. -/
theorem exists_tendsto_of_antitoneOn {g : ℝ → ℝ} (hg : AntitoneOn g (Set.Ici 0))
    {c : ℝ} (hc : ∀ t : ℝ, 0 ≤ t → c ≤ g t) :
    ∃ b : ℝ, Tendsto g atTop (nhds b) ∧ ∀ t : ℝ, 0 ≤ t → b ≤ g t := by
  have hmem : ∀ t : ℝ, max t 0 ∈ Set.Ici (0 : ℝ) := fun t => Set.mem_Ici.mpr (le_max_right _ _)
  have hanti : Antitone fun t : ℝ => g (max t 0) := fun s t hst =>
    hg (hmem s) (hmem t) (max_le_max hst le_rfl)
  have hbdd : BddBelow (Set.range fun t : ℝ => g (max t 0)) := by
    refine ⟨c, ?_⟩
    rintro _ ⟨t, rfl⟩
    exact hc _ (hmem t)
  have hlim := tendsto_atTop_ciInf hanti hbdd
  have heq : (fun t : ℝ => g (max t 0)) =ᶠ[atTop] g := by
    filter_upwards [eventually_ge_atTop (0 : ℝ)] with t ht
    rw [max_eq_left ht]
  refine ⟨⨅ t : ℝ, g (max t 0), Tendsto.congr' heq hlim, fun t ht => ?_⟩
  have := ciInf_le hbdd t
  rwa [max_eq_left ht] at this

/-- The dual statement: a function monotone on `[0,+∞)` and bounded above
there converges, and its limit is an upper bound. -/
theorem exists_tendsto_of_monotoneOn {g : ℝ → ℝ} (hg : MonotoneOn g (Set.Ici 0))
    {c : ℝ} (hc : ∀ t : ℝ, 0 ≤ t → g t ≤ c) :
    ∃ a : ℝ, Tendsto g atTop (nhds a) ∧ ∀ t : ℝ, 0 ≤ t → g t ≤ a := by
  obtain ⟨b, hb, hbt⟩ := exists_tendsto_of_antitoneOn (g := fun t => -g t)
    (fun s hs t ht hst => neg_le_neg (hg hs ht hst)) (c := -c)
    (fun t ht => neg_le_neg (hc t ht))
  refine ⟨-b, by simpa using hb.neg, fun t ht => ?_⟩
  have := hbt t ht
  linarith

/-! ### `e:defab` -/

/-- **Equation (e:defab).**  The two limits

  `a = lim_t min_j φ*_1(z_j(t))`,  `b = lim_t max_j φ*_1(z_j(t))`

exist, and `min_j φ*_1(z_j(t)) ≤ a`, `b ≤ max_j φ*_1(z_j(t))` for every
`t ≥ 0` — at `t = 0` that is the source's parenthetical remark.

The monotonicity conclusions of `l:fj` are taken as explicit hypotheses.

Source: arXiv:2305.05465v6, `e:defab`. -/
theorem exists_tendsto_maxCoord_minCoord (f : EucSpace d →L[ℝ] ℝ)
    (Z : ℝ → Idx (m + 1) → EucSpace d)
    (hmax : AntitoneOn (fun t => maxCoord f (Z t)) (Set.Ici 0))
    (hmin : MonotoneOn (fun t => minCoord f (Z t)) (Set.Ici 0)) :
    (∃ b : ℝ, Tendsto (fun t => maxCoord f (Z t)) atTop (nhds b) ∧
        ∀ t : ℝ, 0 ≤ t → b ≤ maxCoord f (Z t)) ∧
      ∃ a : ℝ, Tendsto (fun t => minCoord f (Z t)) atTop (nhds a) ∧
        ∀ t : ℝ, 0 ≤ t → minCoord f (Z t) ≤ a := by
  have h0 : (0 : ℝ) ∈ Set.Ici (0 : ℝ) := Set.mem_Ici.mpr le_rfl
  constructor
  · refine exists_tendsto_of_antitoneOn hmax (c := minCoord f (Z 0)) fun t ht => ?_
    exact (hmin h0 (Set.mem_Ici.mpr ht) ht).trans (minCoord_le_maxCoord f (Z t))
  · refine exists_tendsto_of_monotoneOn hmin (c := maxCoord f (Z 0)) fun t ht => ?_
    exact (minCoord_le_maxCoord f (Z t)).trans (hmax h0 (Set.mem_Ici.mpr ht) ht)

/-- The hypotheses of `exists_tendsto_maxCoord_minCoord` are satisfiable. -/
example (f : EucSpace d →L[ℝ] ℝ) (z : EucSpace d) :
    AntitoneOn (fun _ : ℝ => maxCoord f (fun _ : Idx (m + 1) => z)) (Set.Ici 0) ∧
      MonotoneOn (fun _ : ℝ => minCoord f (fun _ : Idx (m + 1) => z)) (Set.Ici 0) :=
  ⟨antitoneOn_const, monotoneOn_const⟩

/-! ### `c:bounded` -/

/-- **Corollary (c:bounded).**  If `V` has only real non-negative eigenvalues
then `z_i(·) ∈ L^∞([0,+∞))`.

The hypothesis enters through its two consequences, both explicit here: the
eigenbasis expansion `z = Σ_k φ*_k(z) φ_k`, and the uniform bound on each
coordinate `φ*_k(z_i(·))`, which is `l:fj` applied to each `φ*_k` — legitimate
because every `λ_k ≥ 0`.

Source: arXiv:2305.05465v6, `c:bounded`. -/
theorem bounded_of_coord_bounded {ι : Type*} [Fintype ι] (f : ι → (EucSpace d →L[ℝ] ℝ))
    (φ : ι → EucSpace d) (hexp : ∀ z : EucSpace d, z = ∑ k : ι, f k z • φ k)
    (Z : ℝ → Idx n → EucSpace d) (i : Idx n)
    (hb : ∀ k : ι, ∃ R : ℝ, ∀ t : ℝ, 0 ≤ t → |f k (Z t i)| ≤ R) :
    ∃ R : ℝ, ∀ t : ℝ, 0 ≤ t → ‖Z t i‖ ≤ R := by
  choose R hR using hb
  refine ⟨∑ k : ι, |R k| * ‖φ k‖, fun t ht => ?_⟩
  calc ‖Z t i‖ = ‖∑ k : ι, f k (Z t i) • φ k‖ := by rw [← hexp]
    _ ≤ ∑ k : ι, ‖f k (Z t i) • φ k‖ := norm_sum_le _ _
    _ = ∑ k : ι, |f k (Z t i)| * ‖φ k‖ := by
        simp [norm_smul, Real.norm_eq_abs]
    _ ≤ ∑ k : ι, |R k| * ‖φ k‖ :=
        Finset.sum_le_sum fun k _ =>
          mul_le_mul_of_nonneg_right ((hR k t ht).trans (le_abs_self _)) (norm_nonneg _)

/-- The standard basis of `ℝ^d` expands every vector, so the hypothesis `hexp`
is satisfiable. -/
theorem eq_sum_proj_single (z : EucSpace d) :
    z = ∑ k : Fin d, EuclideanSpace.proj k z •
      (EuclideanSpace.single k (1 : ℝ) : EucSpace d) := by
  ext i
  simp [Pi.single_apply]

/-- The hypotheses of `bounded_of_coord_bounded` are satisfiable: the standard
basis, and a configuration that does not move. -/
example (z : EucSpace d) (i : Idx (m + 1)) :
    (∀ w : EucSpace d, w = ∑ k : Fin d, EuclideanSpace.proj k w •
        (EuclideanSpace.single k (1 : ℝ) : EucSpace d)) ∧
      ∀ k : Fin d, ∃ R : ℝ, ∀ t : ℝ, 0 ≤ t →
        |EuclideanSpace.proj k ((fun (_ : ℝ) (_ : Idx (m + 1)) => z) t i)| ≤ R :=
  ⟨eq_sum_proj_single, fun k => ⟨|EuclideanSpace.proj k z|, fun _ _ => le_rfl⟩⟩

end Clusters
end Transformer
