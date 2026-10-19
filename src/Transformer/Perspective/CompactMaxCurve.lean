module

public import Transformer.Perspective.MaxCurve
public import Mathlib.Topology.Order.Compact

/-!
# A maximum principle for compact families of differentiable curves

Auxiliary for arXiv:2411.04551v3, §2, `prop: targets.atoms`, extending
the finite-family argument in `Transformer.Perspective.MaxCurve` to
arbitrary compact supports.

Subtract a small increasing affine barrier from every curve. The joint
continuous function attains a maximum on the compact index space times
a finite time interval. If the maximum exceeds the initial barrier,
its time is later than the initial endpoint. Its index also maximizes
the original family at that time, so its derivative is strictly negative.
The left slope at a maximum is nonnegative, a contradiction. Letting the
barrier slope tend to zero proves the comparison.

The maximum function itself needs no differentiability hypothesis, and
the time derivatives need no continuity in the index. Duality gives the
corresponding minimum principle.
-/

@[expose] public section

open Set Filter
open scoped Topology

namespace Transformer.Perspective

variable {ι : Type*} [TopologicalSpace ι] [CompactSpace ι]

/-- A compact family's maximum cannot increase if every attaining
curve has nonpositive derivative. Auxiliary for arXiv:2411.04551v3,
§2, `prop: targets.atoms`; all hypotheses concern the stated interval. -/
theorem compact_max_curve_le_of_deriv_nonpos
    (f f' : ι → ℝ → ℝ) (R : ℝ → ℝ) (s t : ℝ) (hst : s ≤ t)
    (hc : ContinuousOn (fun q : ι × ℝ => f q.1 q.2) (univ ×ˢ Icc s t))
    (hd : ∀ i r, r ∈ Icc s t → HasDerivAt (f i) (f' i r) r)
    (hmax : ∀ r, r ∈ Icc s t → ∀ i, f i r ≤ R r)
    (hatt : ∀ r, r ∈ Icc s t → ∃ i, R r = f i r)
    (hnp : ∀ r, r ∈ Icc s t → ∀ i, f i r = R r → f' i r ≤ 0) : R t ≤ R s := by
  have hbarrier (ε : ℝ) (hε : 0 < ε) : R t ≤ R s + ε * (t - s) := by
    by_contra hbad
    have hbad' : R s + ε * (t - s) < R t := lt_of_not_ge hbad
    let H (q : ι × ℝ) := f q.1 q.2 - ε * (q.2 - s)
    obtain ⟨i₀, _⟩ := hatt s ⟨le_rfl, hst⟩
    have hcomp : IsCompact ((univ : Set ι) ×ˢ Icc s t) := isCompact_univ.prod isCompact_Icc
    have hH : ContinuousOn H (univ ×ˢ Icc s t) :=
      hc.sub
        (continuous_const.mul (continuous_snd.sub continuous_const)).continuousOn
    obtain ⟨⟨i, r⟩, hr, hmaxH⟩ := hcomp.exists_isMaxOn ⟨(i₀, s), mem_univ _, le_rfl, hst⟩ hH
    obtain ⟨j, hj⟩ := hatt t ⟨hst, le_rfl⟩
    have hhigh : R s < H (i, r) := by
      have hh : H (j, t) ≤ H (i, r) :=
        hmaxH (show (j, t) ∈ (univ : Set ι) ×ˢ Icc s t from ⟨mem_univ _, hst, le_rfl⟩)
      change f j t - ε * (t - s) ≤ f i r - ε * (r - s) at hh
      change R s < f i r - ε * (r - s)
      rw [← hj] at hh
      linarith
    have hsr : s < r := by
      rcases hr.2.1.eq_or_lt with he | he
      · change s = r at he
        change R s < f i r - ε * (r - s) at hhigh
        rw [← he] at hhigh
        have hh0 := hmax s ⟨le_rfl, hst⟩ i
        linarith
      · exact he
    have hactive : f i r = R r := by
      obtain ⟨k, hk⟩ := hatt r hr.2
      have hh : H (k, r) ≤ H (i, r) :=
        hmaxH (show (k, r) ∈ (univ : Set ι) ×ˢ Icc s t from ⟨mem_univ _, hr.2⟩)
      change f k r - ε * (r - s) ≤ f i r - ε * (r - s) at hh
      rw [← hk] at hh
      exact le_antisymm (hmax r hr.2 i) (by linarith)
    have hdH : HasDerivAt (fun z => H (i, z)) (f' i r - ε) r := by
      convert (hd i r hr.2).fun_sub (((hasDerivAt_id r).sub_const s).const_mul ε) using 1 <;>
        simp [H]
    have hslope : Tendsto (slope (fun z => H (i, z)) r) (𝓝[<] r) (𝓝 (f' i r - ε)) :=
      (hasDerivAt_iff_tendsto_slope.mp hdH).mono_left
        (nhdsWithin_mono r fun z hz => ne_of_lt hz)
    have hneg : f' i r - ε < 0 := by linarith [hnp r hr.2 i hactive]
    have hfalse : ∀ᶠ z in 𝓝[<] r, False := by
      filter_upwards [hslope.eventually_lt_const hneg,
        (nhdsWithin_le_nhds (s := Iio r)) (Ioi_mem_nhds hsr), self_mem_nhdsWithin] with z hz hzs hzr
      have hh := hmaxH (show (i, z) ∈ (univ : Set ι) ×ˢ Icc s t from
        ⟨mem_univ _, hzs.le, hzr.le.trans hr.2.2⟩)
      have hn : 0 ≤ slope (fun z => H (i, z)) r z := by
        rw [slope_def_field]
        exact div_nonneg_of_nonpos (sub_nonpos.mpr hh) (sub_nonpos.mpr hzr.le)
      exact (not_lt_of_ge hn) hz
    obtain ⟨z, hz⟩ := hfalse.exists
    exact hz
  apply le_of_forall_pos_le_add
  intro δ hδ
  have hpos : 0 < t - s + 1 := by linarith
  have hh := hbarrier (δ / (t - s + 1)) (div_pos hδ hpos)
  have he : δ / (t - s + 1) * (t - s) ≤ δ := by
    rw [div_mul_eq_mul_div, div_le_iff₀ hpos]
    nlinarith
  linarith

/-- The compact maximum principle on a future time half-line.
Auxiliary for arXiv:2411.04551v3, §2, `prop: targets.atoms`. -/
theorem compact_max_curve_le_of_deriv_nonpos_Ici
    (f f' : ι → ℝ → ℝ) (R : ℝ → ℝ) (a : ℝ)
    (hc : ContinuousOn (fun q : ι × ℝ => f q.1 q.2) (univ ×ˢ Ici a))
    (hd : ∀ i r, a ≤ r → HasDerivAt (f i) (f' i r) r)
    (hmax : ∀ r, a ≤ r → ∀ i, f i r ≤ R r)
    (hatt : ∀ r, a ≤ r → ∃ i, R r = f i r)
    (hnp : ∀ r, a ≤ r → ∀ i, f i r = R r → f' i r ≤ 0)
    (s t : ℝ) (has : a ≤ s) (hst : s ≤ t) : R t ≤ R s :=
  compact_max_curve_le_of_deriv_nonpos f f' R s t hst
    (hc.mono (fun _ hq => ⟨hq.1, has.trans hq.2.1⟩))
    (fun i r hr => hd i r (has.trans hr.1)) (fun r hr => hmax r (has.trans hr.1))
    (fun r hr => hatt r (has.trans hr.1)) (fun r hr => hnp r (has.trans hr.1))

/-- A compact family's minimum cannot decrease if every attaining
curve has nonnegative derivative. Auxiliary for arXiv:2411.04551v3,
§2, `prop: targets.atoms`. -/
theorem compact_min_curve_le_of_deriv_nonneg
    (f f' : ι → ℝ → ℝ) (R : ℝ → ℝ) (s t : ℝ) (hst : s ≤ t)
    (hc : ContinuousOn (fun q : ι × ℝ => f q.1 q.2) (univ ×ˢ Icc s t))
    (hd : ∀ i r, r ∈ Icc s t → HasDerivAt (f i) (f' i r) r)
    (hmin : ∀ r, r ∈ Icc s t → ∀ i, R r ≤ f i r)
    (hatt : ∀ r, r ∈ Icc s t → ∃ i, R r = f i r)
    (hnn : ∀ r, r ∈ Icc s t → ∀ i, f i r = R r → 0 ≤ f' i r) : R s ≤ R t := by
  have hh := compact_max_curve_le_of_deriv_nonpos
    (fun i r => -f i r) (fun i r => -f' i r) (fun r => -R r) s t hst hc.neg
    (fun i r hr => (hd i r hr).neg) (fun r hr i => neg_le_neg (hmin r hr i))
    (fun r hr => by obtain ⟨i, hi⟩ := hatt r hr; exact ⟨i, congrArg Neg.neg hi⟩)
    (fun r hr i hi => by have hn := hnn r hr i (neg_injective hi); linarith)
  linarith

/-- An infinite compact family, indexed by the whole unit interval,
with actual time derivatives and attained maxima, witnesses every
maximum-principle hypothesis jointly. Negating it witnesses the
minimum-principle hypotheses. -/
example : ∃ (f f' : Icc (0 : ℝ) 1 → ℝ → ℝ) (R : ℝ → ℝ),
    Continuous (fun q : Icc (0 : ℝ) 1 × ℝ => f q.1 q.2) ∧
    (∀ i r, HasDerivAt (f i) (f' i r) r) ∧
    (∀ r i, f i r ≤ R r) ∧ (∀ r, ∃ i, R r = f i r) ∧
    (∀ r i, f i r = R r → f' i r ≤ 0) ∧ (0 : ℝ) ≤ 1 := by
  let f (i : Icc (0 : ℝ) 1) (r : ℝ) := (i : ℝ) - r
  refine ⟨f, (fun _ _ => -1), (fun r => 1 - r), ?_, ?_, ?_, ?_, ?_, zero_le_one⟩
  · exact (continuous_subtype_val.comp continuous_fst).sub continuous_snd
  · intro i r
    convert (hasDerivAt_id r).const_sub (i : ℝ) using 1
    funext x
    rfl
  · intro r i
    dsimp only [f]
    linarith [i.2.2]
  · intro r
    exact ⟨⟨1, zero_le_one, le_rfl⟩, rfl⟩
  · intro r i hi
    norm_num

/-- An increasing infinite compact family jointly witnesses the
minimum-principle hypotheses on every finite time interval. -/
example : ∃ (f f' : Icc (0 : ℝ) 1 → ℝ → ℝ) (R : ℝ → ℝ),
    Continuous (fun q : Icc (0 : ℝ) 1 × ℝ => f q.1 q.2) ∧
    (∀ i r, HasDerivAt (f i) (f' i r) r) ∧
    (∀ r i, R r ≤ f i r) ∧ (∀ r, ∃ i, R r = f i r) ∧
    (∀ r i, f i r = R r → 0 ≤ f' i r) ∧ (0 : ℝ) ≤ 1 := by
  let f (i : Icc (0 : ℝ) 1) (r : ℝ) := (i : ℝ) + r
  refine ⟨f, (fun _ _ => 1), id, ?_, ?_, ?_, ?_, ?_, zero_le_one⟩
  · exact (continuous_subtype_val.comp continuous_fst).add continuous_snd
  · intro i r
    convert (hasDerivAt_id r).const_add (i : ℝ) using 1
    funext x
    rfl
  · intro r i
    dsimp only [f, id_eq]
    linarith [i.2.1]
  · intro r
    exact ⟨⟨0, le_rfl, zero_le_one⟩, by simp [f]⟩
  · intro r i hi
    norm_num

end Transformer.Perspective
