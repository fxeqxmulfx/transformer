module

public import Transformer.Grokking.Convex.ConvexDepth.BoundedLatent

/-!
# Exact coordinates after eliminating observed decoder rows

Source: `review/convex/moments.md`, CM65. A fixed partial dictionary identifies
which rows are already affine copies of a program mass. Only the other rows
remain coordinates. Reconstruction preserves all of those unknown entries.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexDepth.ObservedRows

open scoped BigOperators
open LatentPrograms (Parameters)

variable {J S C : Type*}

/-- States without an observed dictionary entry, CM65. -/
abbrev Unknown (labels : J → S → Option C) (j : J) := {s : S // labels j s = none}

/-- Shared masses and every unknown raw-class row, CM65. Observed rows have
no independent coordinates; their values are determined by the anchor equations. -/
abbrev Reduced (labels : J → S → Option C) :=
  (J → ℝ) × ((j : J) → Unknown labels j → C → ℝ)

variable [DecidableEq C]

/-- Reconstruct the actual full joint tensor, CM65. The dictionary is fixed
before optimization; its class indicators multiply only a learned mass. -/
def expand (labels : J → S → Option C) (r : Reduced labels) : Parameters J S C :=
  (r.1, fun j s c => match h : labels j s with
    | none => r.2 j ⟨s, h⟩ c
    | some y => if c = y then r.1 j else 0)

/-- Retain every unobserved row and every shared mass, CM65. -/
def restrict (labels : J → S → Option C) (q : Parameters J S C) : Reduced labels :=
  (q.1, fun j s c => q.2 j s.1 c)

/-- Reconstruction leaves every unknown coordinate unchanged, CM65. -/
theorem expand_unknown (labels : J → S → Option C) (r : Reduced labels)
    (j : J) (s : Unknown labels j) (c : C) :
    (expand labels r).2 j s.1 c = r.2 j s c := by
  dsimp only [expand]
  split
  · rfl
  · have hn := s.2
    simp_all

/-- An observed row is exactly its mass times its recorded class indicator,
CM65, including zero mass and every other raw class. -/
theorem expand_known (labels : J → S → Option C) (r : Reduced labels)
    (j : J) (s : S) (y : C) (h : labels j s = some y) (c : C) :
    (expand labels r).2 j s c = if c = y then r.1 j else 0 := by
  dsimp only [expand]
  split <;> simp_all

/-- Restriction after reconstruction is the identity on all reduced
coordinates, CM65; no feasibility or guessed completion is needed. -/
theorem restrict_expand (labels : J → S → Option C) (r : Reduced labels) :
    restrict labels (expand labels r) = r := by
  refine Prod.ext ?_ ?_
  · rfl
  · funext j s c
    exact expand_unknown labels r j s c

/-- The full reconstruction is linear, hence affine, in the actual learned
coordinates, CM65. The partial dictionary does not vary with the parameters. -/
theorem expand_combination (labels : J → S → Option C) (a b : Reduced labels)
    (t u : ℝ) :
    expand labels (t • a + u • b) = t • expand labels a + u • expand labels b := by
  refine Prod.ext ?_ ?_
  · rfl
  · funext j s c
    change (expand labels (t • a + u • b)).2 j s c =
      t * (expand labels a).2 j s c + u * (expand labels b).2 j s c
    cases h : labels j s with
    | none =>
      rw [expand_unknown labels _ j ⟨s, h⟩, expand_unknown labels a j ⟨s, h⟩,
        expand_unknown labels b j ⟨s, h⟩]
      rfl
    | some y =>
      rw [expand_known labels _ j s y h, expand_known labels a j s y h,
        expand_known labels b j s y h]
      split_ifs
      · rfl
      · ring

variable [Fintype J] [Fintype C]

/-- Explicit reduced constraints, CM65. Each unknown row still has the
program's shared mass and all its raw-class coordinates; only anchors disappear. -/
def domain (labels : J → S → Option C) (order : List J) (δ R : ℝ) :
    Set (Reduced labels) := {r | (∀ j, 0 ≤ r.1 j) ∧
      (∀ j s c, 0 ≤ r.2 j s c) ∧ (∀ j s, ∑ c, r.2 j s c = r.1 j) ∧
      (∀ j, j ∉ order → r.1 j = 0) ∧ δ ≤ ∑ j, r.1 j ∧ (∑ j, r.1 j) ≤ R}

omit [DecidableEq C] in
/-- The reduced variables have an explicit convex feasible domain, CM65.
This is not a product of independently learned routing and decoder factors. -/
theorem domain_convex (labels : J → S → Option C) (order : List J) (δ R : ℝ) :
    Convex ℝ (domain labels order δ R) := by
  intro a ha b hb t u ht hu htu
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro j
    exact add_nonneg (mul_nonneg ht (ha.1 j)) (mul_nonneg hu (hb.1 j))
  · intro j s c
    exact add_nonneg (mul_nonneg ht (ha.2.1 j s c)) (mul_nonneg hu (hb.2.1 j s c))
  · intro j s
    change (∑ c, (t * a.2 j s c + u * b.2 j s c)) = t * a.1 j + u * b.1 j
    rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum,
      ha.2.2.1 j s, hb.2.2.1 j s]
  · intro j hj
    change t * a.1 j + u * b.1 j = 0
    rw [ha.2.2.2.1 j hj, hb.2.2.2.1 j hj]
    ring
  · change δ ≤ ∑ j, (t * a.1 j + u * b.1 j)
    rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum]
    have h := add_le_add (mul_le_mul_of_nonneg_left ha.2.2.2.2.1 ht)
      (mul_le_mul_of_nonneg_left hb.2.2.2.2.1 hu)
    simpa [← add_mul, htu] using h
  · change (∑ j, (t * a.1 j + u * b.1 j)) ≤ R
    rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum]
    have h := add_le_add (mul_le_mul_of_nonneg_left ha.2.2.2.2.2 ht)
      (mul_le_mul_of_nonneg_left hb.2.2.2.2.2 hu)
    simpa [← add_mul, htu] using h

end Transformer.Grokking.Convex.ConvexDepth.ObservedRows
