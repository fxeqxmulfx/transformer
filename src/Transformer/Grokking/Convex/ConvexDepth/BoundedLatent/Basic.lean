module

public import Transformer.Grokking.Convex.ConvexDepth.LatentPrograms.Joint

/-!
# Bounded joint latent decoder masses

Source: `review/convex/moments.md`, CM64. Extend CM53's explicit joint
tensor constraints by a bounded total mass, allowing CE to learn confidence.
The program bank is fixed; eliminating inconsistent programs preserves exactly
the same convex feasible set, including every unknown decoder row.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexDepth.BoundedLatent

open scoped BigOperators
open LatentPrograms (Parameters)

variable {J X S C : Type*} [Fintype J] [Fintype C] [DecidableEq C]

/-- Total learned program mass, CM64. -/
def mass (q : Parameters J S C) : ℝ := ∑ j, q.1 j

/-- Explicit homogeneous joint constraints, CM64. No independently learned
mass and decoder factors occur in these coefficients. -/
def cone (program : J → X → S) (order : List J) (samples : List (X × C)) :
    Set (Parameters J S C) := {q | (∀ j, 0 ≤ q.1 j) ∧
      (∀ j s c, 0 ≤ q.2 j s c) ∧ (∀ j s, ∑ c, q.2 j s c = q.1 j) ∧
      (∀ j, j ∉ order → q.1 j = 0) ∧
      ∀ j, ∀ t ∈ samples, ∀ c,
        q.2 j (program j t.1) c = if c = t.2 then q.1 j else 0}

/-- Bounds constrain the learned total, not a product of learned factors,
CM64. Every decoder row, including unseen states, remains in the domain. -/
def domain (program : J → X → S) (order : List J) (samples : List (X × C))
    (δ R : ℝ) : Set (Parameters J S C) :=
  {q | q ∈ cone program order samples ∧ δ ≤ mass q ∧ mass q ≤ R}

omit [Fintype J] in
/-- Shared row masses and observed anchors form a convex cone, CM64.
This proof extends CM53 Joint.domain_convex with its unit-mass constraint removed. -/
theorem cone_convex (program : J → X → S) (order : List J)
    (samples : List (X × C)) : Convex ℝ (cone program order samples) := by
  intro a ha b hb t u ht hu htu
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
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
  · intro j z hz c
    change t * a.2 j (program j z.1) c + u * b.2 j (program j z.1) c =
      if c = z.2 then t * a.1 j + u * b.1 j else 0
    rw [ha.2.2.2.2 j z hz c, hb.2.2.2.2 j z hz c]
    split_ifs <;> ring

omit [Fintype C] [DecidableEq C] in
/-- The actual total is linear in all joint variables, CM64. -/
theorem mass_combination (a b : Parameters J S C) (t u : ℝ) :
    mass (t • a + u • b) = t * mass a + u * mass b := by
  change (∑ j, (t * a.1 j + u * b.1 j)) = t * (∑ j, a.1 j) + u * (∑ j, b.1 j)
  rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum]

/-- Lower and upper total-mass bounds retain the full convex domain, CM64. -/
theorem domain_convex (program : J → X → S) (order : List J)
    (samples : List (X × C)) (δ R : ℝ) : Convex ℝ (domain program order samples δ R) := by
  intro a ha b hb t u ht hu htu
  refine ⟨cone_convex program order samples ha.1 hb.1 ht hu htu, ?_, ?_⟩
  · change δ ≤ mass (t • a + u • b)
    rw [mass_combination]
    have h := add_le_add (mul_le_mul_of_nonneg_left ha.2.1 ht)
      (mul_le_mul_of_nonneg_left hb.2.1 hu)
    simpa [← add_mul, htu] using h
  · change mass (t • a + u • b) ≤ R
    rw [mass_combination]
    have h := add_le_add (mul_le_mul_of_nonneg_left ha.2.2 ht)
      (mul_le_mul_of_nonneg_left hb.2.2 hu)
    simpa [← add_mul, htu] using h

omit [Fintype J] in
/-- Every joint decoder coordinate is bounded by its complete row mass,
CM64; this also covers states absent from all observations. -/
theorem coefficient_le_mass (program : J → X → S) (order : List J)
    (samples : List (X × C)) {q : Parameters J S C}
    (hq : q ∈ cone program order samples) (j : J) (s : S) (c : C) : q.2 j s c ≤ q.1 j := by
  have h := Finset.single_le_sum (f := fun d : C => q.2 j s d)
    (fun d (_ : d ∈ Finset.univ) => hq.2.1 j s d) (Finset.mem_univ c)
  rwa [hq.2.2.1 j s] at h

omit [Fintype J] in
/-- Zero mass annihilates every coordinate of that program, CM64. -/
theorem coefficient_zero (program : J → X → S) (order : List J)
    (samples : List (X × C)) {q : Parameters J S C}
    (hq : q ∈ cone program order samples) (j : J) (hj : q.1 j = 0) (s : S) (c : C) :
    q.2 j s c = 0 := by
  apply le_antisymm
  · simpa [hj] using coefficient_le_mass program order samples hq j s c
  · exact hq.2.1 j s c

variable [DecidableEq S]

omit [Fintype J] in
/-- Conflicting observed labels force the entire fixed program block to
zero, CM64. No proposed target program enters this necessary condition. -/
theorem nonzero_retained (program : J → X → S) (order : List J)
    (samples : List (X × C)) {q : Parameters J S C}
    (hq : q ∈ cone program order samples) (j : J) (hj : q.1 j ≠ 0) :
    j ∈ LatentPrograms.retained program order samples := by
  apply (LatentPrograms.mem_retained program order samples j).mpr
  refine ⟨?_, ?_⟩
  · by_contra hn
    exact hj (hq.2.2.2.1 j hn)
  · intro t ht u hu he
    by_contra hn
    have a := hq.2.2.2.2 j t ht t.2
    have b := hq.2.2.2.2 j u hu t.2
    rw [ite_eq_left rfl, he] at a
    rw [ite_eq_right hn] at b
    exact hj (a.symm.trans b)

/-- Removing exactly the inconsistent fixed candidates changes no feasible
tensor, CM64. This does not justify truncating a surviving candidate family. -/
theorem retained_domain_eq (program : J → X → S) (order : List J)
    (samples : List (X × C)) (δ R : ℝ) :
    domain program (LatentPrograms.retained program order samples) samples δ R =
      domain program order samples δ R := by
  ext q
  constructor
  · intro h
    refine ⟨⟨h.1.1, h.1.2.1, h.1.2.2.1, ?_, h.1.2.2.2.2⟩, h.2⟩
    intro j hj
    apply h.1.2.2.2.1 j
    intro hk
    exact hj ((LatentPrograms.mem_retained program order samples j).mp hk).1
  · intro h
    refine ⟨⟨h.1.1, h.1.2.1, h.1.2.2.1, ?_, h.1.2.2.2.2⟩, h.2⟩
    intro j hj
    by_contra hn
    exact hj (nonzero_retained program order samples h.1 j hn)

end Transformer.Grokking.Convex.ConvexDepth.BoundedLatent
