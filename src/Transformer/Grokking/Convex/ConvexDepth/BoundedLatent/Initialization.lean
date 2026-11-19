module

public import Transformer.Grokking.Convex.ConvexDepth.BoundedLatent.Basic
public import Transformer.Grokking.Convex.ConvexDepth.LatentPrograms.Initialization

/-!
# Feasible bounded confidence from observed latent dictionaries

Source: `review/convex/moments.md`, CM64. Scale a proved CM53 feasible point to
construct one joint tensor. This constructor does not parameterize the full
convex domain by a product of independently learned factors.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexDepth.BoundedLatent

open scoped BigOperators
open LatentPrograms (Parameters)

variable {J X S C : Type*} [Fintype J] [Fintype C] [DecidableEq C]

omit [Fintype J] in
/-- The explicit homogeneous constraints are stable under nonnegative
scaling, CM64; all decoder states scale with their shared program mass. -/
theorem cone_smul (program : J → X → S) (order : List J)
    (samples : List (X × C)) {q : Parameters J S C}
    (hq : q ∈ cone program order samples) {ρ : ℝ} (hρ : 0 ≤ ρ) :
    ρ • q ∈ cone program order samples := by
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · intro j
    exact mul_nonneg hρ (hq.1 j)
  · intro j s c
    exact mul_nonneg hρ (hq.2.1 j s c)
  · intro j s
    change (∑ c, ρ * q.2 j s c) = ρ * q.1 j
    rw [← Finset.mul_sum, hq.2.2.1 j s]
  · intro j hj
    change ρ * q.1 j = 0
    rw [hq.2.2.2.1 j hj, mul_zero]
  · intro j t ht c
    change ρ * q.2 j (program j t.1) c = if c = t.2 then ρ * q.1 j else 0
    rw [hq.2.2.2.2 j t ht c]
    split_ifs <;> ring

omit [Fintype C] [DecidableEq C] in
/-- The shared total scales with the actual joint tensor, CM64. -/
theorem mass_smul (q : Parameters J S C) (ρ : ℝ) : mass (ρ • q) = ρ * mass q := by
  change (∑ j, ρ * q.1 j) = ρ * ∑ j, q.1 j
  rw [Finset.mul_sum]

/-- Every CM53 feasible tensor gives a bounded joint point at any admitted
nonnegative confidence, CM64. Both unobserved and observed rows are retained. -/
theorem scaled_mem (program : J → X → S) (order : List J)
    (samples : List (X × C)) {q : Parameters J S C}
    (hq : q ∈ LatentPrograms.domain program order samples) {ρ δ R : ℝ}
    (hρ : 0 ≤ ρ) (hδ : δ ≤ ρ) (hR : ρ ≤ R) :
    ρ • q ∈ domain program order samples δ R := by
  have hc : q ∈ cone program order samples :=
    ⟨hq.1.1, hq.2.1, hq.2.2.1, hq.2.2.2.1, hq.2.2.2.2⟩
  have hm : mass q = 1 := hq.1.2
  refine ⟨cone_smul program order samples hc hρ, ?_, ?_⟩
  · simpa [mass_smul, hm] using hδ
  · simpa [mass_smul, hm] using hR

variable [DecidableEq J] [DecidableEq S]

/-- A retained input program and its observed dictionary initialize a
feasible bounded-confidence tensor, CM64; no target decoder is an input. -/
theorem retained_vertex_mem (program : J → X → S) (order : List J)
    (samples : List (X × C)) (default : C) (j : J)
    (hj : j ∈ LatentPrograms.retained program order samples) {ρ δ R : ℝ}
    (hρ : 0 ≤ ρ) (hδ : δ ≤ ρ) (hR : ρ ≤ R) :
    ρ • LatentPrograms.vertex program samples default j ∈
      domain program order samples δ R := by
  exact scaled_mem program order samples
    (LatentPrograms.vertex_mem program order samples default j hj) hρ hδ hR

omit [DecidableEq J] in
/-- A positive lower mass makes a feasible bounded model impossible when
the exact observed filter has no program, CM64. This is a domain refusal. -/
theorem nonempty_retained (program : J → X → S) (order : List J)
    (samples : List (X × C)) {δ R : ℝ} (hδ : 0 < δ)
    {q : Parameters J S C} (hq : q ∈ domain program order samples δ R) :
    LatentPrograms.retained program order samples ≠ [] := by
  intro empty
  have hz : ∀ j, q.1 j = 0 := by
    intro j
    by_contra hn
    have hj := nonzero_retained program order samples hq.1 j hn
    rw [empty] at hj
    exact List.not_mem_nil hj
  have hm : mass q = 0 := by simp [mass, hz]
  have hb := hq.2.1
  rw [hm] at hb
  linarith

/-- With positive admitted lower mass, observed compatibility exactly
characterizes feasibility, CM64. Unknown targets are absent from this test. -/
theorem nonempty_iff (program : J → X → S) (order : List J)
    (samples : List (X × C)) (default : C) {δ R : ℝ} (hδ : 0 < δ) (hR : δ ≤ R) :
    (domain program order samples δ R).Nonempty ↔
      LatentPrograms.retained program order samples ≠ [] := by
  constructor
  · rintro ⟨q, hq⟩
    exact nonempty_retained program order samples hδ hq
  · intro hn
    cases he : LatentPrograms.retained program order samples with
    | nil => exact False.elim (hn he)
    | cons j rest =>
      have hj : j ∈ LatentPrograms.retained program order samples := by rw [he]; simp
      exact ⟨δ • LatentPrograms.vertex program samples default j,
        retained_vertex_mem program order samples default j hj (le_of_lt hδ) (le_refl δ) hR⟩

/-- Nonconstant feasible joint tensor for the general domain examples,
CM64. Its observed dictionary is learned by CM53's two-coordinate filter. -/
def examplePoint : Parameters Bool Bool Bool := (4 : ℝ) • LatentPrograms.examplePoint

/-- The examples' nonconstant point has confidence four inside bounds one
and eight, CM64; these hypotheses are feasible, rather than merely assumed. -/
theorem examplePoint_mem : examplePoint ∈ domain LatentPrograms.exampleProgram
    [false, true] LatentPrograms.exampleSamples 1 8 := by
  exact scaled_mem _ _ _ LatentPrograms.examplePoint_mem
    (by norm_num : (0 : ℝ) ≤ 4) (by norm_num : (1 : ℝ) ≤ 4) (by norm_num : (4 : ℝ) ≤ 8)

example := cone_convex LatentPrograms.exampleProgram [false, true] LatentPrograms.exampleSamples

example := domain_convex LatentPrograms.exampleProgram [false, true]
  LatentPrograms.exampleSamples 1 8

example := coefficient_le_mass LatentPrograms.exampleProgram [false, true]
  LatentPrograms.exampleSamples examplePoint_mem.1 true false false

example := coefficient_zero LatentPrograms.exampleProgram [false, true]
  LatentPrograms.exampleSamples examplePoint_mem.1 false (by norm_num
    [examplePoint, LatentPrograms.examplePoint, LatentPrograms.ofDecoders]) true true

example := nonzero_retained LatentPrograms.exampleProgram [false, true]
  LatentPrograms.exampleSamples examplePoint_mem.1 true (by norm_num
    [examplePoint, LatentPrograms.examplePoint, LatentPrograms.ofDecoders])

example := retained_domain_eq LatentPrograms.exampleProgram [false, true]
  LatentPrograms.exampleSamples 1 8

example := cone_smul LatentPrograms.exampleProgram [false, true]
  LatentPrograms.exampleSamples examplePoint_mem.1 (by norm_num : (0 : ℝ) ≤ 2)

example := scaled_mem LatentPrograms.exampleProgram [false, true]
  LatentPrograms.exampleSamples LatentPrograms.examplePoint_mem (by norm_num : (0 : ℝ) ≤ 3)
  (by norm_num : (1 : ℝ) ≤ 3) (by norm_num : (3 : ℝ) ≤ 8)

example := retained_vertex_mem LatentPrograms.exampleProgram [false, true]
  LatentPrograms.exampleSamples false true (by decide) (by norm_num : (0 : ℝ) ≤ 3)
  (by norm_num : (1 : ℝ) ≤ 3) (by norm_num : (3 : ℝ) ≤ 8)

example := nonempty_retained LatentPrograms.exampleProgram [false, true]
  LatentPrograms.exampleSamples (by norm_num : (0 : ℝ) < 1) examplePoint_mem

example := nonempty_iff LatentPrograms.exampleProgram [false, true]
  LatentPrograms.exampleSamples false (by norm_num : (0 : ℝ) < 1) (by norm_num : (1 : ℝ) ≤ 8)

end Transformer.Grokking.Convex.ConvexDepth.BoundedLatent
