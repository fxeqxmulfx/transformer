module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.SharedPrograms.Learner

/-!
# Training-compatible convex mixtures and certified unseen complex answers

Source: `review/convex/moments.md`, CM52. Observations impose linear zero-weight
constraints on inconsistent programs. Consensus and structural realizability
then identify the actual complex prediction. The fixed source code supplies
capacity bounds only; it supplies neither the copy prior nor realizability.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.SharedPrograms

open scoped BigOperators
open Landscape.Neural (crossEntropy)

variable {J X C : Type*} [Fintype J] [Fintype C] [DecidableEq C]

/-- The observed-consistent simplex face, CM52. Its constraints inspect only
observed inputs and labels; the raw label at a future query is never consulted. -/
def compatibleWeights (program : J → X → C) (order : List J) (samples : List (X × C)) :
    Set (J → ℝ) := {a | a ∈ ContextMixture.simplex ∧
      ∀ j, j ∉ retained program order samples → a j = 0}

omit [Fintype C] in
/-- Any retained program supplies a feasible forward-only vertex, CM52.
The vertex is selected from the observed filter, not the unknown target index. -/
theorem vertex_compatible [DecidableEq J] (program : J → X → C) (order : List J)
    (samples : List (X × C)) (j : J) (hj : j ∈ retained program order samples) :
    (fun k => if k = j then (1 : ℝ) else 0) ∈ compatibleWeights program order samples := by
  refine ⟨⟨?_, ?_⟩, ?_⟩
  · intro k
    dsimp only
    split_ifs <;> norm_num
  · simp
  · intro k hk
    by_cases hkj : k = j
    · subst k
      exact False.elim (hk hj)
    · simp [hkj]

omit [Fintype C] in
/-- Eliminating inconsistent vertices preserves a convex domain, CM52. -/
theorem compatibleWeights_convex (program : J → X → C) (order : List J)
    (samples : List (X × C)) : Convex ℝ (compatibleWeights program order samples) := by
  intro a ha b hb t u ht hu htu
  refine ⟨ContextMixture.simplex_convex ha.1 hb.1 ht hu htu, ?_⟩
  intro j hj
  simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul, ha.2 j hj, hb.2 j hj,
    mul_zero, add_zero]

/-- Ordinary CE remains convex on the observed-consistent face, CM52. -/
theorem compatible_ce_convex (U : Matrix C C ℂ) (program : J → X → C)
    (order : List J) (samples : List (X × C)) (margin : ℝ) (x : X) (y : C) :
    ConvexOn ℝ (compatibleWeights program order samples) (fun a => crossEntropy
      (headLogits (ContextMixture.features (bank U program) a) (copyHead U margin) x) y) := by
  refine ⟨compatibleWeights_convex program order samples, ?_⟩
  intro a ha b hb t u ht hu htu
  exact (ce_convex U program margin x y).2 ha.1 hb.1 ht hu htu

/-- Every feasible mixture has the same exact observed logits, CM52. This
also explains why further observed CE cannot distinguish surviving programs. -/
theorem observed_logits (U : Matrix C C ℂ) (hU : U ∈ Matrix.unitaryGroup C ℂ)
    (program : J → X → C) (order : List J) (samples : List (X × C))
    {a : J → ℝ} (ha : a ∈ compatibleWeights program order samples)
    (margin : ℝ) (x : X) (y : C) (hxy : (x, y) ∈ samples) :
    headLogits (ContextMixture.features (bank U program) a) (copyHead U margin) x =
      fun c => if c = y then margin else 0 := by
  apply agreed_logits U hU program a ha.1.2 margin x y
  intro j hj
  by_cases hkept : j ∈ retained program order samples
  · exact ((mem_retained program order samples j).mp hkept).2 (x, y) hxy
  · exact False.elim (hj (ha.2 j hkept))

/-- The executable consensus result determines the complete complex logit
vector for every compatible mixture, CM52; no particular vertex is assumed. -/
theorem accepted_logits (U : Matrix C C ℂ) (hU : U ∈ Matrix.unitaryGroup C ℂ)
    (program : J → X → C) (order : List J) (samples : List (X × C))
    {a : J → ℝ} (ha : a ∈ compatibleWeights program order samples)
    (margin : ℝ) (x : X) (y : C) (hp : predict program order samples x = some y) :
    headLogits (ContextMixture.features (bank U program) a) (copyHead U margin) x =
      fun c => if c = y then margin else 0 := by
  apply agreed_logits U hU program a ha.1.2 margin x y
  intro j hj
  by_cases hkept : j ∈ retained program order samples
  · exact ((consensus_some_iff program _ x y).mp hp).2 j hkept
  · exact False.elim (hj (ha.2 j hkept))

/-- Structural realizability turns the checked consensus into the unknown
raw target's exact full logits, CM52. No observed-class coverage is required. -/
theorem target_logits (U : Matrix C C ℂ) (hU : U ∈ Matrix.unitaryGroup C ℂ)
    (program : J → X → C) (order : List J) (samples : List (X × C))
    {a : J → ℝ} (ha : a ∈ compatibleWeights program order samples)
    (target : X → C) (hreal : ∃ j ∈ order, ∀ x, program j x = target x)
    (hobs : ∀ s ∈ samples, s.2 = target s.1) (margin : ℝ) (x : X) (y : C)
    (hp : predict program order samples x = some y) :
    headLogits (ContextMixture.features (bank U program) a) (copyHead U margin) x =
      fun c => if c = target x then margin else 0 := by
  have hy := predict_sound program order samples target hreal hobs x y hp
  rw [accepted_logits U hU program order samples ha margin x y hp, hy]

/-- The identified raw answer has the ordinary finite-margin CE, CM52. -/
theorem target_ce (U : Matrix C C ℂ) (hU : U ∈ Matrix.unitaryGroup C ℂ)
    (program : J → X → C) (order : List J) (samples : List (X × C))
    {a : J → ℝ} (ha : a ∈ compatibleWeights program order samples)
    (target : X → C) (hreal : ∃ j ∈ order, ∀ x, program j x = target x)
    (hobs : ∀ s ∈ samples, s.2 = target s.1) (margin : ℝ) (x : X) (y : C)
    (hp : predict program order samples x = some y) :
    crossEntropy
      (headLogits (ContextMixture.features (bank U program) a) (copyHead U margin) x)
      (target x) = Real.log (1 + ((Fintype.card C : ℝ) - 1) * Real.exp (-margin)) := by
  rw [target_logits U hU program order samples ha target hreal hobs margin x y hp]
  exact fixed_margin_ce _ _

/-- The forward vertex selected by the separating witness observations. -/
def witnessVertex (j : Bool) : ℝ := if j = true then 1 else 0

/-- The witness mixture is feasible by the executable observed filter. -/
theorem witnessVertex_mem :
    witnessVertex ∈ compatibleWeights witnessProgram [false, true] witnessSamples := by
  exact vertex_compatible witnessProgram [false, true] witnessSamples true (by decide)

/-- The actual complete complex model places margin24 on class3, absent
from training labels, after the learner identifies last-coordinate copying. -/
theorem witness_complex_logits :
    headLogits (ContextMixture.features (bank fourier4 witnessProgram) witnessVertex)
      (copyHead fourier4 24) (0, 3) = fun c => if c = 3 then 24 else 0 := by
  apply target_logits fourier4 fourier4_unitary witnessProgram [false, true]
    witnessSamples witnessVertex_mem Prod.snd _ _ 24 (0, 3) 3 (by decide)
  · exact ⟨true, by simp, by simp [witnessProgram]⟩
  · simp [witnessSamples]

/-- The missing raw class has ordinary full-vocabulary CE at the proved
positive margin, CM52; it was not supplied by a training label. -/
theorem witness_complex_ce :
    crossEntropy
      (headLogits (ContextMixture.features (bank fourier4 witnessProgram) witnessVertex)
        (copyHead fourier4 24) (0, 3)) 3 = Real.log (1 + 3 * Real.exp (-24)) ∧
      ∀ s ∈ witnessSamples, s.2 ≠ 3 := by
  constructor
  · rw [witness_complex_logits]
    have h := fixed_margin_ce (3 : Fin 4) 24
    norm_num at h
    exact h
  · decide

example := compatibleWeights_convex witnessProgram [false, true] witnessSamples

example := compatible_ce_convex fourier4 witnessProgram [false, true] witnessSamples
  24 (0, 3) 3

example := observed_logits fourier4 fourier4_unitary witnessProgram [false, true]
  witnessSamples witnessVertex_mem 24 (0, 1) 1 (by decide)

example := accepted_logits fourier4 fourier4_unitary witnessProgram [false, true]
  witnessSamples witnessVertex_mem 24 (0, 3) 3 (by decide)

example := target_ce fourier4 fourier4_unitary witnessProgram [false, true]
  witnessSamples witnessVertex_mem Prod.snd
  ⟨true, by simp, by simp [witnessProgram]⟩ (by simp [witnessSamples]) 24 (0, 3) 3 (by decide)

end Transformer.Grokking.FlatSpectrum.ConvexDepth.SharedPrograms
