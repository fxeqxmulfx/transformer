module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.LatentPrograms.Complex

/-!
# Training-only feasible initialization of joint latent decoders

Source: `review/flat_spectrum.md`, CM53. The initializer chooses the first
retained candidate itself and completes unknown entries with a declared
default. Feasibility is proved; no successful optimizer is a hypothesis.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.LatentPrograms

open scoped BigOperators

variable {J X S C : Type*} [Fintype J] [Fintype C]
  [DecidableEq J] [DecidableEq S] [DecidableEq C]

/-- A joint vertex for a candidate and its training-learned decoder,
CM53. The supplied default is retained only outside observed latent support. -/
def vertex (program : J → X → S) (samples : List (X × C)) (default : C) (chosen : J) :
    Parameters J S C := ofDecoders (fun j => if j = chosen then 1 else 0)
      (fun j => completion (program j) samples default)

/-- Every retained candidate supplies a feasible actual joint vertex,
CM53, including all original raw labels and every unobserved latent row. -/
theorem vertex_mem (program : J → X → S) (order : List J) (samples : List (X × C))
    (default : C) (j : J) (hj : j ∈ retained program order samples) :
    vertex program samples default j ∈ domain program order samples := by
  apply ofDecoders_mem
  · constructor
    · intro k
      change 0 ≤ if k = j then (1 : ℝ) else 0
      split_ifs <;> norm_num
    · simp
  · intro k hk
    have hn : k ≠ j := by
      intro he
      subst k
      exact hk ((mem_retained program order samples j).mp hj).1
    simp [hn]
  · intro k hk t ht
    have he : k = j := by
      by_contra hn
      simp [hn] at hk
    subst k
    exact completion_of_lookup (program j) samples default (program j t.1) t.2
      (retained_observed program order samples j hj t ht)

/-- Select the first retained candidate, then construct its joint tensor,
CM53. No unknown target program or raw decoder is an input. -/
def initializeTensor (program : J → X → S) (order : List J) (samples : List (X × C))
    (default : C) : Option (Parameters J S C) :=
  (retained program order samples).head?.map (vertex program samples default)

omit [Fintype J] [Fintype C] in
/-- The executable initializer refuses exactly when its observed filter
has no candidate, CM53. No optimizer failure is hidden in this condition. -/
theorem initialize_none_iff (program : J → X → S) (order : List J)
    (samples : List (X × C)) (default : C) :
    initializeTensor program order samples default = none ↔
      retained program order samples = [] := by
  cases he : retained program order samples <;> simp [initializeTensor, he]

/-- Successful executable initialization always yields a feasible tensor,
CM53. This statement does not assert the default is the unknown raw target. -/
theorem initialize_mem (program : J → X → S) (order : List J) (samples : List (X × C))
    (default : C) (q : Parameters J S C)
    (hq : initializeTensor program order samples default = some q) :
    q ∈ domain program order samples := by
  cases he : retained program order samples with
  | nil => simp [initializeTensor, he] at hq
  | cons j rest =>
    have hv : vertex program samples default j = q := by simpa [initializeTensor, he] using hq
    rw [← hv]
    apply vertex_mem program order samples default j
    rw [he]
    simp

/-- Observable nonempty candidate support guarantees that the explicit
initializer returns a feasible point, CM53. -/
theorem initialize_exists (program : J → X → S) (order : List J)
    (samples : List (X × C)) (default : C) (hne : retained program order samples ≠ []) :
    ∃ q, initializeTensor program order samples default = some q ∧
      q ∈ domain program order samples := by
  cases he : retained program order samples with
  | nil => exact False.elim (hne he)
  | cons j rest =>
    refine ⟨vertex program samples default j, ?_, ?_⟩
    · simp [initializeTensor, he]
    · apply vertex_mem program order samples default j
      rw [he]
      simp

/-- The exact observed joint domain is feasible if and only if the
executable filter retains a candidate, CM53. An empty bank is an actual
representation obstruction for these constraints, not merely a failed search. -/
theorem domain_nonempty_iff (program : J → X → S) (order : List J)
    (samples : List (X × C)) (default : C) :
    (domain program order samples).Nonempty ↔ retained program order samples ≠ [] := by
  constructor
  · rintro ⟨q, hq⟩ he
    have hz : ∀ j, q.1 j = 0 := by
      intro j
      by_contra hn
      have hj := nonzero_mem_retained program order samples hq j hn
      rw [he] at hj
      exact List.not_mem_nil hj
    have hs := hq.1.2
    simp only [hz, Finset.sum_const_zero] at hs
    norm_num at hs
  · intro hne
    obtain ⟨q, _, hq⟩ := initialize_exists program order samples default hne
    exact ⟨q, hq⟩

/-- Unknown-program realizability proves initialization succeeds using
only observations, CM53. The target and its witnesses remain proof premises. -/
theorem realizable_initialize (program : J → X → S) (order : List J)
    (samples : List (X × C)) (default : C) (target : X → C)
    (hreal : ∃ j ∈ order, ∃ decoder : S → C, ∀ x, decoder (program j x) = target x)
    (hobs : ∀ t ∈ samples, t.2 = target t.1) :
    ∃ q, initializeTensor program order samples default = some q ∧
      q ∈ domain program order samples := by
  apply initialize_exists program order samples default
  exact retained_nonempty program order samples target hreal hobs

omit [DecidableEq J] in
/-- Any supported simplex mixture of all training-learned dictionaries
is feasible, CM53. Native full/mini/forward modes may share this initialization. -/
theorem learned_mixture_mem (program : J → X → S) (order : List J)
    (samples : List (X × C)) (default : C) (weights : J → ℝ)
    (hw : weights ∈ ContextMixture.simplex)
    (hs : ∀ j, j ∉ retained program order samples → weights j = 0) :
    ofDecoders weights (fun j => completion (program j) samples default) ∈
      domain program order samples := by
  apply ofDecoders_mem program order samples weights _ hw
  · intro j hj
    apply hs j
    intro hk
    exact hj ((mem_retained program order samples j).mp hk).1
  · intro j hj t ht
    have hk : j ∈ retained program order samples := by
      by_contra hn
      exact hj (hs j hn)
    exact completion_of_lookup (program j) samples default (program j t.1) t.2
      (retained_observed program order samples j hk t ht)

example := vertex_mem exampleProgram [false, true] exampleSamples false true (by decide)

example := initialize_none_iff exampleProgram [false, true]
  [((false, false), false), ((false, false), true)] false

example := initialize_mem exampleProgram [false, true] exampleSamples false
  (vertex exampleProgram exampleSamples false true) (by rfl)

example := initialize_exists exampleProgram [false, true] exampleSamples false (by decide)

example := domain_nonempty_iff exampleProgram [false, true] exampleSamples false

example := realizable_initialize exampleProgram [false, true] exampleSamples false Prod.snd
  ⟨true, by simp, id, by simp [exampleProgram]⟩ (by simp [exampleSamples])

example := learned_mixture_mem exampleProgram [false, true] exampleSamples false
  (fun j => if j then 1 else 0)
  (by
    constructor
    · intro j; cases j <;> norm_num
    · norm_num [Fintype.sum_bool])
  (by intro j hj; cases j <;> norm_num [retained, exampleSamples, lookup, exampleProgram] at *)

end Transformer.Grokking.FlatSpectrum.ConvexDepth.LatentPrograms
