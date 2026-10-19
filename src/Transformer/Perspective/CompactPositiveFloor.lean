module

public import Transformer.Perspective.CompactMaxCurve

/-!
# Preservation of a positive floor for a compact family

Auxiliary for arXiv:2411.04551v3, §2, `prop: targets.atoms`. The
hemisphere height derivative has the required sign at a minimizing
particle only while the minimum is nonnegative.

If a height could reach half its positive initial floor, compactness
would give a first such time. Before that time every height is above
the half-floor, and continuity gives the same bound at the first time.
The compact minimum principle then preserves the full initial floor
on that interval, a contradiction. Consequently all heights remain
positive, and the minimum principle preserves the full initial floor
on every future interval. No derivative of the minimum is assumed.
-/

@[expose] public section

open Set Filter
open scoped Topology

namespace Transformer.Perspective

variable {ι : Type*} [TopologicalSpace ι] [CompactSpace ι] [Nonempty ι]

/-- A positive initial floor persists if every nonnegative minimizing
curve has nonnegative derivative. Auxiliary for arXiv:2411.04551v3,
§2, `prop: targets.atoms`, the hemisphere-invariance step. -/
theorem compact_positive_floor_of_deriv_nonneg
    (f f' : ι → ℝ → ℝ) (hc : Continuous (fun q : ι × ℝ => f q.1 q.2))
    (hd : ∀ i t, 0 ≤ t → HasDerivAt (f i) (f' i t) t)
    (hnn : ∀ t, 0 ≤ t → ∀ i, (∀ j, f i t ≤ f j t) → 0 ≤ f i t → 0 ≤ f' i t)
    (r₀ : ℝ) (hr₀ : 0 < r₀) (h0 : ∀ i, r₀ ≤ f i 0)
    (t : ℝ) (ht : 0 ≤ t) (i : ι) : r₀ ≤ f i t := by
  classical
  -- Compactness supplies an attained minimum at each time; its
  -- continuity and differentiability are not assumptions.
  have hmin_exists (r : ℝ) : ∃ k : ι, ∀ j, f k r ≤ f j r := by
    obtain ⟨k, _, hk⟩ := (isCompact_univ : IsCompact (univ : Set ι)).exists_isMinOn
      (univ_nonempty) (hc.comp (continuous_id.prodMk continuous_const)).continuousOn
    exact ⟨k, fun j => hk (mem_univ j)⟩
  choose k hk using hmin_exists
  let R (r : ℝ) := f (k r) r
  have hmin (r : ℝ) (j : ι) : R r ≤ f j r := hk r j
  have hatt (r : ℝ) : ∃ j, R r = f j r := ⟨k r, rfl⟩
  have hR0 : r₀ ≤ R 0 := h0 (k 0)
  have compare (T : ℝ) (hT : 0 ≤ T) (hpos : ∀ r ∈ Icc 0 T, ∀ j, 0 ≤ f j r) (j : ι) :
      r₀ ≤ f j T := by
    have hmono := compact_min_curve_le_of_deriv_nonneg f f' R 0 T hT hc.continuousOn
      (fun j r hr => hd j r hr.1) (fun r _ => hmin r) (fun r _ => hatt r)
      (fun r hr j hj => hnn r hr.1 j (fun l => by rw [hj]; exact hmin r l) (hpos r hr j))
    exact hR0.trans (hmono.trans (hmin T j))
  -- A first half-floor contact would provide an entire interval
  -- on which the unconditional minimum comparison applies.
  let S : Set (ι × ℝ) := (univ ×ˢ Icc 0 t) ∩ {q | f q.1 q.2 ≤ r₀ / 2}
  have hcomp : IsCompact S := (isCompact_univ.prod isCompact_Icc).inter_right
    (isClosed_le hc continuous_const)
  have hnone : ¬ S.Nonempty := by
    intro hne
    obtain ⟨⟨j, τ⟩, hτ, hfirst⟩ := hcomp.exists_isMinOn hne continuous_snd.continuousOn
    have hτ0 : 0 < τ := by
      have hh := h0 j
      have hnonneg : 0 ≤ τ := hτ.1.2.1
      have hbad : f j τ ≤ r₀ / 2 := hτ.2
      by_contra hn
      have he : τ = 0 := le_antisymm (le_of_not_gt hn) hnonneg
      rw [he] at hbad
      linarith
    have hbefore (z : ℝ) (hz : 0 ≤ z) (hzτ : z < τ) (l : ι) : r₀ / 2 < f l z := by
      by_contra hn
      have hzmem : (l, z) ∈ S := ⟨⟨mem_univ _, hz, hzτ.le.trans hτ.1.2.2⟩, le_of_not_gt hn⟩
      have he : τ ≤ z := hfirst hzmem
      linarith
    have hbound (l : ι) : r₀ / 2 ≤ f l τ := by
      -- The half-floor bound at the contact time follows from the
      -- left limit of each actual differentiable curve.
      have hlim : Tendsto (f l) (𝓝[<] τ) (𝓝 (f l τ)) :=
        (hd l τ hτ0.le).continuousAt.continuousWithinAt
      apply ge_of_tendsto hlim
      filter_upwards [(nhdsWithin_le_nhds (s := Iio τ)) (Ioi_mem_nhds hτ0), self_mem_nhdsWithin] with z hz hzτ
      exact (hbefore z hz.le hzτ l).le
    have hpos (r : ℝ) (hr : r ∈ Icc 0 τ) (l : ι) : 0 ≤ f l r := by
      rcases hr.2.eq_or_lt with he | he
      · rw [he]
        linarith [hbound l]
      · linarith [hbefore r hr.1 he l]
    have hgood := compare τ hτ0.le hpos j
    have hbad : f j τ ≤ r₀ / 2 := hτ.2
    linarith
  have hpos (r : ℝ) (hr : r ∈ Icc 0 t) (l : ι) : 0 ≤ f l r := by
    have hh : r₀ / 2 < f l r := by
      by_contra hn
      exact hnone ⟨(l, r), ⟨mem_univ _, hr⟩, le_of_not_gt hn⟩
    linarith
  exact compare t ht hpos i

/-- A specified attained compact minimum stays above its positive
initial value even when the derivative sign is known only at
nonnegative minima. Auxiliary for arXiv:2411.04551v3, §2,
`prop: targets.atoms`, extending the finite `Perspective.MinCurve`. -/
theorem compact_min_curve_le_of_deriv_nonneg_when_nonneg
    (f f' : ι → ℝ → ℝ) (R : ℝ → ℝ)
    (hc : Continuous (fun q : ι × ℝ => f q.1 q.2))
    (hd : ∀ i t, 0 ≤ t → HasDerivAt (f i) (f' i t) t)
    (hmin : ∀ r i, R r ≤ f i r) (hatt : ∀ r, ∃ i, R r = f i r)
    (hnn : ∀ r, 0 ≤ r → ∀ i, 0 ≤ R r → f i r = R r → 0 ≤ f' i r)
    (hR0 : 0 < R 0) (t : ℝ) (ht : 0 ≤ t) : R 0 ≤ R t := by
  have hsign (r : ℝ) (hr : 0 ≤ r) (i : ι) (hi : ∀ j, f i r ≤ f j r)
      (hn : 0 ≤ f i r) : 0 ≤ f' i r := by
    obtain ⟨j, hj⟩ := hatt r
    have he : f i r = R r := le_antisymm (by rw [hj]; exact hi j) (hmin r i)
    exact hnn r hr i (by rwa [← he]) he
  have hfloor := compact_positive_floor_of_deriv_nonneg f f' hc hd hsign (R 0) hR0 (hmin 0)
  obtain ⟨i, hi⟩ := hatt t
  rw [hi]
  exact hfloor t ht i

/-- A family indexed by the whole unit interval, with positive initial
heights and actual increasing time curves, jointly realizes all floor,
conditional derivative, attainment and nonnegative-time hypotheses. -/
example : ∃ (f f' : Icc (0 : ℝ) 1 → ℝ → ℝ) (R : ℝ → ℝ),
    Continuous (fun q : Icc (0 : ℝ) 1 × ℝ => f q.1 q.2) ∧
    (∀ i r, HasDerivAt (f i) (f' i r) r) ∧
    (∀ r, 0 ≤ r → ∀ i, (∀ j, f i r ≤ f j r) → 0 ≤ f i r → 0 ≤ f' i r) ∧
    (∀ i, (1 : ℝ) ≤ f i 0) ∧ (∀ r i, R r ≤ f i r) ∧
    (∀ r, ∃ i, R r = f i r) ∧
    (∀ r, 0 ≤ r → ∀ i, 0 ≤ R r → f i r = R r → 0 ≤ f' i r) ∧
    (0 : ℝ) < 1 ∧ 0 < R 0 ∧ (0 : ℝ) ≤ 1 := by
  let f (i : Icc (0 : ℝ) 1) (r : ℝ) := (i : ℝ) + 1 + r
  refine ⟨f, (fun _ _ => 1), (fun r => 1 + r), ?_, ?_, ?_, ?_, ?_, ?_, ?_,
    zero_lt_one, by norm_num, zero_le_one⟩
  · exact ((continuous_subtype_val.comp continuous_fst).add continuous_const).add continuous_snd
  · intro i r
    convert (hasDerivAt_id r).const_add ((i : ℝ) + 1) using 1
    funext x
    rfl
  · intros
    norm_num
  · intro i
    dsimp only [f]
    linarith [i.2.1]
  · intro r i
    dsimp only [f]
    linarith [i.2.1]
  · intro r
    exact ⟨⟨0, le_rfl, zero_le_one⟩, by simp [f]⟩
  · intros
    norm_num

end Transformer.Perspective
