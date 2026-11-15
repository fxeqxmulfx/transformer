module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.BoundedLatent.Complex

/-!
# Extending a fixed input-program family without losing old feasible tensors

Source: `review/flat_spectrum.md`, CM67. The additional programs are fixed
independently of labels. Zero new blocks embed the old full joint domain;
no surviving candidate is removed and no unknown decoder row is completed.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.FamilyExtension

open scoped BigOperators
open LatentPrograms (Parameters)

variable {J K X S C : Type*}

/-- Disjoint enumeration preserves every old and new input program, CM67.
Even equal functions retain their separately declared candidate coordinates. -/
def programs (old : J → X → S) (added : K → X → S) : J ⊕ K → X → S :=
  Sum.elim old added

/-- Append the complete new enumeration after the old one, CM67. -/
def order (old : List J) (added : List K) : List (J ⊕ K) :=
  old.map Sum.inl ++ added.map Sum.inr

/-- Embed all old masses and decoder entries, with zero mass and every
raw coordinate zero in each added block, CM67. -/
def embed (q : Parameters J S C) : Parameters (J ⊕ K) S C :=
  (Sum.elim q.1 (fun _ => 0), Sum.elim q.2 (fun _ _ _ => 0))

/-- No old mass or unknown raw decoder coordinate is identified by the
embedding, CM67; the full old parameter tensor can be recovered. -/
theorem embed_injective : Function.Injective (embed (J := J) (K := K) (S := S) (C := C)) := by
  intro a b h
  apply Prod.ext
  · funext j
    exact congrArg (fun q => q.1 (Sum.inl j)) h
  · funext j s c
    exact congrArg (fun q => q.2 (Sum.inl j) s c) h

variable [Fintype J] [Fintype K]

/-- Adding zero blocks preserves the actual total confidence, CM67. -/
theorem mass_embed (q : Parameters J S C) :
    BoundedLatent.mass (embed (K := K) q) = BoundedLatent.mass q := by
  simp [BoundedLatent.mass, Fintype.sum_sum_type, embed]

variable [Fintype C] [DecidableEq C]

/-- Every old feasible full tensor remains feasible with all added input
programs present, CM67. This checks all rows and anchors, including rows never
seen in the observations; no assumption on the added program functions is used. -/
theorem embed_mem (old : J → X → S) (added : K → X → S)
    (oldOrder : List J) (addedOrder : List K) (samples : List (X × C))
    {δ R : ℝ} {q : Parameters J S C}
    (hq : q ∈ BoundedLatent.domain old oldOrder samples δ R) :
    embed q ∈ BoundedLatent.domain (programs old added)
      (order oldOrder addedOrder) samples δ R := by
  refine ⟨⟨?_, ?_, ?_, ?_, ?_⟩, ?_, ?_⟩
  · intro j
    cases j with
    | inl j => exact hq.1.1 j
    | inr k => exact le_rfl
  · intro j s c
    cases j with
    | inl j => exact hq.1.2.1 j s c
    | inr k => exact le_rfl
  · intro j s
    cases j with
    | inl j => exact hq.1.2.2.1 j s
    | inr k => simp [embed]
  · intro j hj
    cases j with
    | inl j =>
      apply hq.1.2.2.2.1 j
      intro hm
      exact hj (List.mem_append_left _ (List.mem_map_of_mem hm))
    | inr k => rfl
  · intro j t ht c
    cases j with
    | inl j => exact hq.1.2.2.2.2 j t ht c
    | inr k => simp [embed]
  · simpa only [mass_embed] using hq.2.1
  · simpa only [mass_embed] using hq.2.2

variable [DecidableEq S]

omit [Fintype J] [Fintype K] [Fintype C] in
/-- Old candidate consistency is unchanged by appending any family, CM67.
Every old compatible candidate remains retained, in its original copy. -/
theorem retained_inl_iff (old : J → X → S) (added : K → X → S)
    (oldOrder : List J) (addedOrder : List K) (samples : List (X × C)) (j : J) :
    Sum.inl j ∈ LatentPrograms.retained (programs old added)
      (order oldOrder addedOrder) samples ↔
        j ∈ LatentPrograms.retained old oldOrder samples := by
  simp [LatentPrograms.mem_retained, order, programs]

omit [Fintype J] [Fintype K] [Fintype C] in
/-- The additional branch also retains exactly its consistent candidates,
CM67; this is a full union, not a fitted prefix or top-k selection. -/
theorem retained_inr_iff (old : J → X → S) (added : K → X → S)
    (oldOrder : List J) (addedOrder : List K) (samples : List (X × C)) (k : K) :
    Sum.inr k ∈ LatentPrograms.retained (programs old added)
      (order oldOrder addedOrder) samples ↔
        k ∈ LatentPrograms.retained added addedOrder samples := by
  simp [LatentPrograms.mem_retained, order, programs]

end Transformer.Grokking.FlatSpectrum.ConvexDepth.FamilyExtension
