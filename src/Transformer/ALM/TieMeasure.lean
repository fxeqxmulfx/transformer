/-
# The merge path is null, in every dimension

`Transformer.ALM.TieHyperplane` computes the tie locus of two keys and calls
it a hyperplane — "a proper, nonempty affine subspace", and in
`Transformer.ALM.TieBreak` "a set of measure zero in the query".  Neither was
stated: `score_midpoint_eq` gives a point of the locus, `exists_score_ne` a
point outside it, and the two were never put together, so nothing in the
development said the tie locus is small.  In dimension one smallness was
visible — `sScore_eq_iff` pins the tie to the single query `2q = k₁ + k₂` —
and the generalization to dimension `m` lost exactly that, since an equation
between scores is not by itself a statement about how many queries satisfy it.

`tieLocus_eq_bisector` identifies the locus with an affine subspace,
`tieLocus_ne_top` makes it proper from `exists_score_ne`, and
`volume_tieLocus` concludes it is Lebesgue-null.  `volume_tie_locus_family` is
the statement the machine needs: for *any* stored family, in any dimension,
the queries at which two distinct keys tie form a null set, so the merge loops
of `HullHalf::query` are entered for almost no query at all, and the
single-maximizer hypothesis of `Transformer.ALM.SoftmaxValue` holds almost
everywhere.

Source: `transformer_vm/attention/hull2d_cht.h`, lines 252-310.
-/

import Transformer.ALM.TieHyperplane
import Mathlib.MeasureTheory.Measure.Lebesgue.EqHaar
import Mathlib.MeasureTheory.Measure.Haar.InnerProductSpace

open scoped BigOperators

namespace Transformer
namespace ALM

variable {m n : ℕ}

/-- The queries at which two keys score equally: the merge path of
`HullHalf::query` is reachable exactly here. -/
def tieLocus (k₁ k₂ : EucSpace m) : Set (EucSpace m) := {q | score q k₁ = score q k₂}

/-- **The tie locus is the perpendicular bisector, as an affine subspace.**
Not merely an equation between scores: the solution set is the translate of
the hyperplane orthogonal to `k₁ - k₂` through the midpoint of the keys. -/
theorem tieLocus_eq_bisector (k₁ k₂ : EucSpace m) :
    tieLocus k₁ k₂
      = (AffineSubspace.mk' ((2 : ℝ)⁻¹ • (k₁ + k₂)) ((ℝ ∙ (k₁ - k₂))ᗮ) : Set (EucSpace m)) := by
  ext q
  show score q k₁ = score q k₂ ↔ _
  rw [score_eq_iff_inner_eq_zero]
  simp only [SetLike.mem_coe, AffineSubspace.mem_mk', vsub_eq_sub,
    Submodule.mem_orthogonal_singleton_iff_inner_right]
  rw [show q - (2 : ℝ)⁻¹ • (k₁ + k₂) = (2 : ℝ)⁻¹ • ((2 : ℝ) • q - (k₁ + k₂)) by
    rw [smul_sub, smul_smul]; norm_num]
  rw [real_inner_smul_right]
  constructor
  · intro h; rw [h]; ring
  · intro h; linarith [h]

/-- **And it is proper.**  Some query separates two distinct keys — the
query that *is* one of them — so the subspace is not the whole space. -/
theorem tieLocus_ne_top {k₁ k₂ : EucSpace m} (hne : k₁ ≠ k₂) :
    AffineSubspace.mk' ((2 : ℝ)⁻¹ • (k₁ + k₂)) ((ℝ ∙ (k₁ - k₂))ᗮ) ≠ ⊤ := by
  obtain ⟨q, hq⟩ := exists_score_ne k₁ k₂ hne
  intro htop
  refine hq ?_
  have : q ∈ tieLocus k₁ k₂ := by
    rw [tieLocus_eq_bisector, htop]
    trivial
  exact this

/-- **So two keys tie at almost no query.**  In dimension one this is
`sScore_eq_iff`'s single point; in dimension `m` it is a hyperplane, and the
statement that survives the generalization is that its Lebesgue measure is
zero. -/
theorem volume_tieLocus {k₁ k₂ : EucSpace m} (hne : k₁ ≠ k₂) :
    MeasureTheory.volume (tieLocus k₁ k₂) = 0 := by
  rw [tieLocus_eq_bisector]
  exact MeasureTheory.Measure.addHaar_affineSubspace _ _ (tieLocus_ne_top hne)

/-- The hypothesis is satisfiable, and not only in dimension one: the two
plane keys `(0,0)` and `(2,0)` are distinct, and the line of queries `(1, t)`
on which they tie — exhibited in `Transformer.ALM.TieHyperplane` — is null. -/
example : MeasureTheory.volume
    (tieLocus (WithLp.toLp 2 ![(0 : ℝ), 0]) (WithLp.toLp 2 ![(2 : ℝ), 0])) = 0 := by
  refine volume_tieLocus ?_
  intro h
  have := congrFun (congrArg (WithLp.ofLp) h) 0
  norm_num at this

/-- **And the machine's merge path is null for the whole stored family.**  A
query reaches the merge loops only if two of the stored keys tie at it; there
are finitely many pairs, and each contributes a null set.  So in every
dimension, for every family, almost every query has a unique best key: the
tie-break modes decide nothing on a set of positive measure, and the
single-maximizer hypothesis of `Transformer.ALM.SoftmaxValue` is satisfied
almost everywhere. -/
theorem volume_tie_locus_family (K : Fin n → EucSpace m) :
    MeasureTheory.volume
        {q : EucSpace m | ∃ i j, K i ≠ K j ∧ score q (K i) = score q (K j)} = 0 := by
  have hsub : {q : EucSpace m | ∃ i j, K i ≠ K j ∧ score q (K i) = score q (K j)}
      ⊆ ⋃ p : Fin n × Fin n, {q : EucSpace m | K p.1 ≠ K p.2 ∧ score q (K p.1) = score q (K p.2)} := by
    rintro q ⟨i, j, hij, hq⟩
    exact Set.mem_iUnion.mpr ⟨(i, j), hij, hq⟩
  refine MeasureTheory.measure_mono_null hsub (MeasureTheory.measure_iUnion_null fun p => ?_)
  rcases eq_or_ne (K p.1) (K p.2) with h | h
  · convert MeasureTheory.measure_empty (μ := MeasureTheory.volume (α := EucSpace m))
    exact Set.eq_empty_of_forall_notMem fun q hq => hq.1 h
  · exact MeasureTheory.measure_mono_null (fun q hq => hq.2) (volume_tieLocus h)

/-- **So almost every query separates the stored keys.**  The usable form of
the bound above: outside a null set, no two distinct stored keys score alike,
which is the hypothesis `hno` of
`Transformer.ALM.SoftmaxValue.head_output_at_index_untied` and the reason
`resolve` is handed a single line. -/
theorem ae_no_tie (K : Fin n → EucSpace m) :
    ∀ᵐ q : EucSpace m, ∀ i j, K i ≠ K j → score q (K i) ≠ score q (K j) := by
  rw [MeasureTheory.ae_iff]
  refine MeasureTheory.measure_mono_null (fun q hq => ?_) (volume_tie_locus_family K)
  simp only [Set.mem_ofPred_eq, not_forall, not_not] at hq
  obtain ⟨i, j, hij, hq⟩ := hq
  exact ⟨i, j, hij, hq⟩

/-- **And almost every query has a unique maximizer.**  For a family of
distinct keys the argmax is single-valued off a null set, in every dimension:
what `argmax_unique_of_query_mem` gives on the lookup path, the measure gives
everywhere else. -/
theorem ae_argmax_unique (K : Fin n → EucSpace m) (hinj : Function.Injective K) :
    ∀ᵐ q : EucSpace m, ∀ i j, score q (K i) = score q (K j) → i = j := by
  filter_upwards [ae_no_tie K] with q hq i j hij
  by_contra hne
  exact hq i j (fun h => hne (hinj h)) hij

/-- The hypothesis is satisfiable, and the conclusion has content: the two
distinct plane keys `(0,0)` and `(2,0)` do tie somewhere, and the set where
they do is null. -/
example : Function.Injective
    (fun j : Fin 2 => (WithLp.toLp 2 ![(2 * (j : ℕ) : ℝ), 0] : EucSpace 2)) := by
  intro a b h
  have h0 := congrFun (congrArg WithLp.ofLp h) 0
  simp only [Matrix.cons_val_zero] at h0
  refine Fin.ext ?_
  have : ((a : ℕ) : ℝ) = ((b : ℕ) : ℝ) := by linarith
  exact_mod_cast this

/-- And the statement is not vacuous: a family with two distinct keys really
does have a nonempty tie locus, reached at their midpoint. -/
example : (WithLp.toLp 2 ![(1 : ℝ), 0] : EucSpace 2) ∈
    tieLocus (WithLp.toLp 2 ![(0 : ℝ), 0]) (WithLp.toLp 2 ![(2 : ℝ), 0]) := by
  show score _ _ = score _ _
  unfold score
  simp [PiLp.inner_apply, RCLike.inner_apply, norm_sq_eq_sum, Fin.sum_univ_two]
  norm_num

end ALM
end Transformer
