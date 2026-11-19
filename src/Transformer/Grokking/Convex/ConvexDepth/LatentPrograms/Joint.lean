module

public import Transformer.Grokking.Convex.ConvexDepth.LatentPrograms.Learner

/-!
# Joint convex program and decoder coefficients

Source: `review/convex/moments.md`, CM53. Each joint nonnegative row has the
program's shared mass. Observations impose affine raw-label constraints.
Independent learned routing and decoder factors are never multiplied.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexDepth.LatentPrograms

open scoped BigOperators

/-- Shared program masses and joint raw decoder masses, CM53. -/
abbrev Parameters (J S C : Type*) := (J → ℝ) × (J → S → C → ℝ)

variable {J X S C : Type*} [Fintype J] [Fintype C] [DecidableEq C]

/-- The jointly affine observed domain, CM53. Every latent state's full
raw-class row has the same program mass, including states absent from samples. -/
def domain (program : J → X → S) (order : List J) (samples : List (X × C)) :
    Set (Parameters J S C) := {q | q.1 ∈ ContextMixture.simplex ∧
      (∀ j s c, 0 ≤ q.2 j s c) ∧ (∀ j s, ∑ c, q.2 j s c = q.1 j) ∧
      (∀ j, j ∉ order → q.1 j = 0) ∧
      ∀ j, ∀ t ∈ samples, ∀ c,
        q.2 j (program j t.1) c = if c = t.2 then q.1 j else 0}

/-- Shared row masses, support and observed labels are affine constraints,
CM53. Their intersection with nonnegative coefficients is convex. -/
theorem domain_convex (program : J → X → S) (order : List J)
    (samples : List (X × C)) : Convex ℝ (domain program order samples) := by
  intro a ha b hb t u ht hu htu
  refine ⟨ContextMixture.simplex_convex ha.1 hb.1 ht hu htu, ?_, ?_, ?_, ?_⟩
  · intro j s c
    change 0 ≤ t * a.2 j s c + u * b.2 j s c
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

/-- Nonnegative row entries are bounded by their shared program mass,
CM53. This bound later charges every complete complex coefficient. -/
theorem coefficient_le_weight (program : J → X → S) (order : List J)
    (samples : List (X × C)) {q : Parameters J S C} (hq : q ∈ domain program order samples)
    (j : J) (s : S) (c : C) : q.2 j s c ≤ q.1 j := by
  have h := Finset.single_le_sum (f := fun d : C => q.2 j s d)
    (fun d (_ : d ∈ Finset.univ) => hq.2.1 j s d)
    (Finset.mem_univ c)
  rw [hq.2.2.1 j s] at h
  exact h

/-- Zero program mass forces every raw decoder coordinate to zero, CM53;
there is no uncharged latent row hidden outside active support. -/
theorem coefficient_zero (program : J → X → S) (order : List J)
    (samples : List (X × C)) {q : Parameters J S C} (hq : q ∈ domain program order samples)
    (j : J) (hj : q.1 j = 0) (s : S) (c : C) : q.2 j s c = 0 := by
  apply le_antisymm
  · simpa [hj] using coefficient_le_weight program order samples hq j s c
  · exact hq.2.1 j s c

/-- A declared decoder tuple supplies joint coefficients without adding
independently learned factors to the domain, CM53. -/
def ofDecoders (weights : J → ℝ) (decoder : J → S → C) : Parameters J S C :=
  (weights, fun j s c => if c = decoder j s then weights j else 0)

/-- Every consistent supported decoder mixture is a point of the joint
convex domain, CM53. This is a constructor, not a restriction to factored training. -/
theorem ofDecoders_mem (program : J → X → S) (order : List J)
    (samples : List (X × C)) (weights : J → ℝ) (decoder : J → S → C)
    (hw : weights ∈ ContextMixture.simplex) (hs : ∀ j, j ∉ order → weights j = 0)
    (hf : ∀ j, weights j ≠ 0 → ∀ t ∈ samples, decoder j (program j t.1) = t.2) :
    ofDecoders weights decoder ∈ domain program order samples := by
  refine ⟨hw, ?_, ?_, hs, ?_⟩
  · intro j s c
    change 0 ≤ if c = decoder j s then weights j else 0
    split_ifs
    · exact hw.1 j
    · exact le_rfl
  · intro j s
    simp [ofDecoders]
  · intro j t ht c
    change (if c = decoder j (program j t.1) then weights j else 0) =
      if c = t.2 then weights j else 0
    by_cases hz : weights j = 0
    · simp [hz]
    · rw [hf j hz t ht]

variable [DecidableEq S]

/-- An active joint block must be an observationally consistent program,
CM53. Two different labels at one state force its entire mass to zero. -/
theorem nonzero_mem_retained (program : J → X → S) (order : List J)
    (samples : List (X × C)) {q : Parameters J S C} (hq : q ∈ domain program order samples)
    (j : J) (hj : q.1 j ≠ 0) : j ∈ retained program order samples := by
  apply (mem_retained program order samples j).mpr
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

/-- A nonconstant feasible joint example used to instantiate every
domain hypothesis below; the active decoder is learned by exampleSamples. -/
def examplePoint : Parameters Bool Bool Bool :=
  ofDecoders (fun j => if j then 1 else 0) (fun j s => s == j)

/-- The examples' actual nonconstant head coefficients are feasible, CM53. -/
theorem examplePoint_mem : examplePoint ∈ domain exampleProgram [false, true] exampleSamples := by
  apply ofDecoders_mem
  · constructor
    · intro j; cases j <;> norm_num
    · norm_num [Fintype.sum_bool]
  · intro j hj
    cases j <;> simp at hj
  · intro j hj t ht
    cases j
    · simp at hj
    · simp only [exampleSamples, List.mem_cons, List.not_mem_nil, or_false] at ht
      rcases ht with rfl | rfl <;> rfl

example := domain_convex exampleProgram [false, true] exampleSamples

example := ofDecoders_mem exampleProgram [false, true] exampleSamples
  (fun j => if j then 1 else 0) (fun j s => s == j)
  examplePoint_mem.1 examplePoint_mem.2.2.2.1
  (by
    intro j hj t ht
    cases j
    · norm_num at hj
    · simp only [exampleSamples, List.mem_cons, List.not_mem_nil, or_false] at ht
      rcases ht with rfl | rfl <;> rfl)

example := coefficient_le_weight exampleProgram [false, true] exampleSamples
  examplePoint_mem true false false

example := coefficient_zero exampleProgram [false, true] exampleSamples
  examplePoint_mem false (by rfl) true true

example := nonzero_mem_retained exampleProgram [false, true] exampleSamples
  examplePoint_mem true (by norm_num [examplePoint, ofDecoders])

end Transformer.Grokking.Convex.ConvexDepth.LatentPrograms
