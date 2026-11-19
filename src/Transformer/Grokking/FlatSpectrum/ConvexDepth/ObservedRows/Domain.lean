module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.ObservedRows.Basic

/-!
# A bijective coordinate reduction of the original bounded domain

Source: `review/convex/moments.md`, CM65. Use only CM53's actual training
dictionary and exact inconsistent-candidate filter. The full CM64 domain,
including every admissible unknown completion, is preserved in both directions.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.ObservedRows

open scoped BigOperators
open LatentPrograms (Parameters lookup retained)

variable {J X S C : Type*} [Fintype J] [Fintype C] [DecidableEq S] [DecidableEq C]

/-- The partial raw dictionary learned solely from original observations,
CM65. Repeated records are permitted; conflicting candidates have zero mass. -/
def dictionary (program : J → X → S) (samples : List (X × C)) : J → S → Option C :=
  fun j => lookup (program j) samples

/-- Every original feasible tensor restricts to the explicit reduced
domain, CM65. The only discarded candidates are the proved zero blocks. -/
theorem restrict_mem (program : J → X → S) (order : List J) (samples : List (X × C))
    {δ R : ℝ} {q : Parameters J S C}
    (hq : q ∈ BoundedLatent.domain program order samples δ R) :
    restrict (dictionary program samples) q ∈
      domain (dictionary program samples) (retained program order samples) δ R := by
  refine ⟨hq.1.1, ?_, ?_, ?_, hq.2.1, hq.2.2⟩
  · intro j s c
    exact hq.1.2.1 j s.1 c
  · intro j s
    exact hq.1.2.2.1 j s.1
  · intro j hj
    change q.1 j = 0
    by_contra hn
    exact hj (BoundedLatent.nonzero_retained program order samples hq.1 j hn)

/-- Reconstructing a restricted feasible tensor gives every original
coefficient back, CM65, not just its predictions on the observed inputs. -/
theorem expand_restrict (program : J → X → S) (order : List J)
    (samples : List (X × C)) {δ R : ℝ} {q : Parameters J S C}
    (hq : q ∈ BoundedLatent.domain program order samples δ R) :
    expand (dictionary program samples) (restrict (dictionary program samples) q) = q := by
  refine Prod.ext ?_ ?_
  · rfl
  · funext j s c
    cases h : dictionary program samples j s with
    | none =>
      rw [expand_unknown _ _ j ⟨s, h⟩]
      rfl
    | some y =>
      rw [expand_known _ _ j s y h]
      obtain ⟨t, ht, hs, hy⟩ := LatentPrograms.lookup_some_mem (program j) samples s y h
      have ha := hq.1.2.2.2.2 j t ht c
      rw [hs, hy] at ha
      exact ha.symm

/-- Every reduced feasible point expands to the original CM64 domain,
CM65. Known entries use observed labels; every unknown row remains arbitrary
within its full shared-mass simplex. -/
theorem expand_mem (program : J → X → S) (order : List J) (samples : List (X × C))
    {δ R : ℝ} {r : Reduced (dictionary program samples)}
    (hr : r ∈ domain (dictionary program samples) (retained program order samples) δ R) :
    expand (dictionary program samples) r ∈
      BoundedLatent.domain program order samples δ R := by
  refine ⟨⟨hr.1, ?_, ?_, ?_, ?_⟩, hr.2.2.2.2.1, hr.2.2.2.2.2⟩
  · intro j s c
    cases h : dictionary program samples j s with
    | none =>
      rw [expand_unknown _ _ j ⟨s, h⟩]
      exact hr.2.1 j ⟨s, h⟩ c
    | some y =>
      rw [expand_known _ _ j s y h]
      split_ifs
      · exact hr.1 j
      · exact le_rfl
  · intro j s
    cases h : dictionary program samples j s with
    | none =>
      have he (c : C) := expand_unknown (dictionary program samples) r j ⟨s, h⟩ c
      simp_rw [he]
      exact hr.2.2.1 j ⟨s, h⟩
    | some y =>
      simp_rw [expand_known _ r j s y h]
      simp [expand]
  · intro j hj
    apply hr.2.2.2.1 j
    intro kept
    exact hj ((LatentPrograms.mem_retained program order samples j).mp kept).1
  · intro j t ht c
    by_cases hj : j ∈ retained program order samples
    · exact expand_known _ r j (program j t.1) t.2
        (LatentPrograms.retained_observed program order samples j hj t ht) c
    · have hz := hr.2.2.2.1 j hj
      cases h : dictionary program samples j (program j t.1) with
      | none =>
        have hn := (LatentPrograms.lookup_none_iff (program j) samples (program j t.1)).mp h
        exact False.elim (hn t ht rfl)
      | some y =>
        rw [expand_known _ r j (program j t.1) y h]
        change (if c = y then r.1 j else 0) = if c = t.2 then r.1 j else 0
        simp [hz]

/-- The explicit reduced constraints are exactly the preimage of the
original domain, CM65; this is an equivalence, not just a feasible initializer. -/
theorem expand_mem_iff (program : J → X → S) (order : List J)
    (samples : List (X × C)) (δ R : ℝ) (r : Reduced (dictionary program samples)) :
    expand (dictionary program samples) r ∈ BoundedLatent.domain program order samples δ R ↔
      r ∈ domain (dictionary program samples) (retained program order samples) δ R := by
  constructor
  · intro h
    have hr := restrict_mem program order samples h
    rwa [restrict_expand] at hr
  · exact expand_mem program order samples

/-- Reconstruction covers precisely the entire original feasible set,
CM65. No unknown decoder row or compatible program has been fixed or removed. -/
theorem expanded_domain (program : J → X → S) (order : List J)
    (samples : List (X × C)) (δ R : ℝ) :
    expand (dictionary program samples) ''
        domain (dictionary program samples) (retained program order samples) δ R =
      BoundedLatent.domain program order samples δ R := by
  ext q
  constructor
  · rintro ⟨r, hr, rfl⟩
    exact expand_mem program order samples hr
  · intro hq
    exact ⟨restrict (dictionary program samples) q, restrict_mem program order samples hq,
      expand_restrict program order samples hq⟩

end Transformer.Grokking.FlatSpectrum.ConvexDepth.ObservedRows
