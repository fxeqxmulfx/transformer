module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.LatentPrograms.Witness
public import Transformer.Grokking.FlatSpectrum.ConvexCausal.Generation

/-!
# Actual generation from accepted latent-program queries

Source: `review/flat_spectrum.md`, CM54. CM53 identifies individual accepted
queries, while CCa7's global rule recovery is stronger than this learner's
contract. Only acceptance on the finite target trajectory is required here.
The actual complete complex head reads its own emitted prefix and includes EOS.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.LatentPrograms

open ConvexCausal (best best_max generate)

variable {C : Type*} [DecidableEq C]

/-- Prefixes queried by a causal rule before EOS or fuel exhaustion, CM54.
The prefix whose next token is EOS is included; no later prefix is queried. -/
def queryPrefixes (next : List C → C) (eos : C) : ℕ → List C → List (List C)
  | 0, _ => []
  | n + 1, history => history ::
      if next history = eos then []
      else queryPrefixes next eos n (history ++ [next history])

/-- Agreement only along the target's visited prefixes suffices for equality
of actual free generation, CM54. Off-trajectory outputs can differ. -/
theorem generate_on_queries (f g : List C → C) (eos : C) (n : ℕ) (history : List C)
    (h : ∀ xs ∈ queryPrefixes g eos n history, f xs = g xs) :
    generate f eos n history = generate g eos n history := by
  induction n generalizing history with
  | zero => rfl
  | succ n ih =>
    have hx : f history = g history := h history (by simp [queryPrefixes])
    simp only [generate, hx]
    by_cases he : g history = eos
    · simp only [he, ite_true]
    · simp only [he, ite_false]
      apply congrArg (List.cons (g history))
      apply ih
      intro xs hxs
      apply h xs
      simp [queryPrefixes, he, hxs]

variable {J X S : Type*} [Fintype J] [Fintype S] [Fintype C] [Nonempty C]
  [DecidableEq S]

/-- A maximum readout of the complete complex logits returns the accepted
class for every feasible joint tensor at positive margin, CM54. -/
theorem accepted_best (U : Matrix S S ℂ) (hU : U ∈ Matrix.unitaryGroup S ℂ)
    (program : J → X → S) (order : List J) (samples : List (X × C)) {margin : ℝ}
    (hm : 0 < margin) {q : Parameters J S C} (hq : q ∈ domain program order samples)
    (x : X) (y : C) (hp : predict program order samples x = some y) :
    best (headLogits (features U program) (head U margin q) x) = y := by
  by_contra hne
  have hs := accepted_strict U hU program order samples hm hq x y hp
    (best (headLogits (features U program) (head U margin q) x)) hne
  exact (not_lt_of_ge (best_max _ y)) hs

/-- Structural realizability and acceptance at every queried target prefix
transfer CM53's complete complex readout through generation and EOS, CM54.
The premise does not assert coverage of all possible prefixes. -/
theorem accepted_generation (U : Matrix S S ℂ) (hU : U ∈ Matrix.unitaryGroup S ℂ)
    (program : J → List C → S) (order : List J) (samples : List (List C × C))
    {margin : ℝ} (hm : 0 < margin) {q : Parameters J S C}
    (hq : q ∈ domain program order samples) (target : List C → C)
    (hreal : ∃ j ∈ order, ∃ decoder : S → C, ∀ x, decoder (program j x) = target x)
    (hobs : ∀ t ∈ samples, t.2 = target t.1) (eos : C) (n : ℕ) (history : List C)
    (hcover : ∀ xs ∈ queryPrefixes target eos n history,
      ∃ y, predict program order samples xs = some y) :
    generate (fun xs => best (headLogits (features U program) (head U margin q) xs))
      eos n history = generate target eos n history := by
  apply generate_on_queries
  intro xs hxs
  obtain ⟨y, hy⟩ := hcover xs hxs
  exact (accepted_best U hU program order samples hm hq xs y hy).trans
    (predict_sound program order samples target hreal hobs xs y hy)

/-- Training computes a feasible bounded complete head before generating,
CM54. Target realizability, truthful observations and trajectory coverage
remain explicit; no optimizer or unseen-label oracle supplies the tensor. -/
theorem initialized_generation [DecidableEq J]
    (U : Matrix S S ℂ) (hU : U ∈ Matrix.unitaryGroup S ℂ)
    (program : J → List C → S) (order : List J) (samples : List (List C × C))
    (default : C) {margin : ℝ} (hm : 0 < margin) (target : List C → C)
    (hreal : ∃ j ∈ order, ∃ decoder : S → C, ∀ x, decoder (program j x) = target x)
    (hobs : ∀ t ∈ samples, t.2 = target t.1) (eos : C) (n : ℕ) (history : List C)
    (hcover : ∀ xs ∈ queryPrefixes target eos n history,
      ∃ y, predict program order samples xs = some y) :
    ∃ q, initializeTensor program order samples default = some q ∧
      q ∈ domain program order samples ∧
      head U margin q ∈ headBudget (2 * |margin| * Fintype.card S) ∧
      generate (fun xs => best (headLogits (features U program) (head U margin q) xs))
        eos n history = generate target eos n history := by
  obtain ⟨q, hi, hq⟩ := realizable_initialize program order samples default target hreal hobs
  exact ⟨q, hi, hq, head_budget U hU program order samples margin hq,
    accepted_generation U hU program order samples hm hq target hreal hobs eos n history hcover⟩

/-! A nonconstant witness generates from prefixes absent from training. -/

/-- Two training prefixes cover both latent states of the orientation bank. -/
def generationSamples : List (List Bool × Bool) := [([], false), ([false], true)]

/-- This witness initializes from the observed labels, not a supplied decoder. -/
noncomputable def generationPoint : Parameters Bool Bool Bool :=
  vertex ConvexCausal.alternating generationSamples false false

/-- The observed dictionary yields a feasible witness, CM54. -/
theorem generationPoint_mem : generationPoint ∈
    domain ConvexCausal.alternating [false, true] generationSamples := by
  exact vertex_mem ConvexCausal.alternating [false, true] generationSamples false false (by decide)

example := generate_on_queries (fun xs : List Bool => !xs.isEmpty)
  (ConvexCausal.alternating false) true 4 [] (by decide)

example := accepted_best (1 : Matrix Bool Bool ℂ) (Matrix.unitaryGroup Bool ℂ).one_mem
  exampleProgram [false, true] exampleSamples (by norm_num : (0 : ℝ) < 24)
  examplePoint_mem (true, true) true (by decide)

example := accepted_generation (1 : Matrix Bool Bool ℂ) (Matrix.unitaryGroup Bool ℂ).one_mem
  ConvexCausal.alternating [false, true] generationSamples (by norm_num : (0 : ℝ) < 24)
  generationPoint_mem (ConvexCausal.alternating false)
  ⟨false, by simp, id, by intro xs; rfl⟩
  (by simp [generationSamples, ConvexCausal.alternating]) true 4 [true, true]
  (by decide)

example := initialized_generation (1 : Matrix Bool Bool ℂ) (Matrix.unitaryGroup Bool ℂ).one_mem
  ConvexCausal.alternating [false, true] generationSamples false
  (by norm_num : (0 : ℝ) < 24) (ConvexCausal.alternating false)
  ⟨false, by simp, id, by intro xs; rfl⟩
  (by simp [generationSamples, ConvexCausal.alternating]) true 4 [true, true]
  (by decide)

/-- The actual complex model emits a token and EOS from unseen prefixes,
CM54. The second prediction reads its own first token. -/
example : generate (fun xs => best (headLogits (features (1 : Matrix Bool Bool ℂ)
    ConvexCausal.alternating) (head 1 24 generationPoint) xs)) true 4 [true, true] =
    [false, true] := by
  rw [accepted_generation (1 : Matrix Bool Bool ℂ) (Matrix.unitaryGroup Bool ℂ).one_mem
    ConvexCausal.alternating [false, true] generationSamples (by norm_num : (0 : ℝ) < 24)
    generationPoint_mem (ConvexCausal.alternating false)
    ⟨false, by simp, id, by intro xs; rfl⟩
    (by simp [generationSamples, ConvexCausal.alternating]) true 4 [true, true] (by decide)]
  decide

end Transformer.Grokking.FlatSpectrum.ConvexDepth.LatentPrograms
