import Transformer.GPTMini.Convex.Structured.MarkovObjective
import Mathlib.Data.Fintype.Card

/-!
# Shared compact raw parameter coordinates

Source: the 52-field learned pointer layout at 87ffa1b and the jointly
convex causal state/value objective at 95456c1. This proposal uses the
same 52 free token fields for both heads. State transitions read the
first 36 fields; pointer Q/K/value groups use 16/16/20 disjoint fields.
Task/mode parameter assignments remain separate, as in Basis training.

Additional freely learned coordinates contain one positional potential
per context slot, chronology, six initial logits, 120 state emission
logits and two mixture-head logits. Positional potentials are proposed
to learn table exclusion without a fixed record mask or a latent role
bank. This module realizes the state head's actual shared raw lookups
and proves its unrestricted complete objective convex in this space.
It does not prove pointer role capacity, mixture convexity, raw task
capability or prenorm/residual/tied readout integration. The count 64
includes ten fixed decoder axes, a protected constant and one position
axis; a count alone is not a realized embedding/attention block.
-/

namespace Transformer.GPTMini.Convex.Structured

open scoped BigOperators Classical
noncomputable section

variable {V C : ℕ}

/-- Free shared non-token parameters: positions, chronology, initial states, state values and head selection.
Source: the proposed compact two-head parameter layout; context slots are not examples or prefix prototypes. -/
abbrev SharedGlobal (C : ℕ) :=
  Fin C ⊕ (Unit ⊕ (Fin 6 ⊕ ((Fin 6 × (Fin 5 × Fin 4)) ⊕ Fin 2)))

/-- All actual trainable coordinates, with one shared 52-field row per vocabulary token.
Source: the proposed unrestricted raw parameter space, independent of dataset or number of prefixes. -/
abbrev SharedField (V C : ℕ) := (Fin V × Fin 52) ⊕ SharedGlobal C

/-- Ordinary free real parameters on the exact shared finite coordinate set.
Source: the proposed embedding/head weights, without a constraint projection or a replacement optimizer. -/
abbrev SharedParameters (V C : ℕ) := SharedField V C → ℝ

/-- Query matching channels occupy the first sixteen actual token fields.
Source: the four-by-four learned Q log-potential layout. -/
def querySlot (g c : Fin 4) : Fin 52 :=
  ⟨4 * g.val + c.val, by have hg := g.isLt; have hc := c.isLt; omega⟩

/-- Independently learned key matching channels occupy the next sixteen token fields.
Source: the four-by-four K layout, disjoint from the actual Q coordinates. -/
def keySlot (g c : Fin 4) : Fin 52 :=
  ⟨16 + 4 * g.val + c.val, by have hg := g.isLt; have hc := c.isLt; omega⟩

/-- Jointly learned value channels occupy the final twenty token fields.
Source: the five-by-four value layout; fixed output codes are separate decoder coordinates. -/
def valueSlot (h : Fin 5) (d : Fin 4) : Fin 52 :=
  ⟨32 + 4 * h.val + d.val, by have hh := h.isLt; have hd := d.isLt; omega⟩

/-- A six-by-six token-conditioned transition table fits in the first thirty-six free token fields.
Source: the proposed learned state head, with all transition logits trained rather than task transitions hardcoded. -/
def transitionSlot (previous next : Fin 6) : Fin 52 :=
  ⟨6 * previous.val + next.val, by have hp := previous.isLt; have hn := next.isLt; omega⟩

/-- All matching/value groups occupy their actual disjoint ranges in the same free token table.
Source: the proposed 16/16/20 raw coordinate allocation, rather than merely a total field count. -/
theorem sharedPointerSlot_ranges (g c : Fin 4) (h : Fin 5) (d : Fin 4) :
    (querySlot g c).val < 16 ∧ 16 ≤ (keySlot g c).val ∧ (keySlot g c).val < 32 ∧
      32 ≤ (valueSlot h d).val ∧ (valueSlot h d).val < 52 := by
  simp only [querySlot, keySlot, valueSlot]
  have hg := g.isLt
  have hc := c.isLt
  have hh := h.isLt
  have hd := d.isLt
  omega

/-- Every previous/next state pair has its own independently trainable raw token field.
Source: the actual six-by-six row-major table embedding; no state transitions are accidentally tied. -/
theorem transitionSlot_injective : Function.Injective (fun pair : Fin 6 × Fin 6 => transitionSlot pair.1 pair.2) := by
  rintro ⟨previous, next⟩ ⟨other, after⟩ heq
  have hv := congrArg Fin.val heq
  change 6 * previous.val + next.val = 6 * other.val + after.val at hv
  have hp := previous.isLt
  have hn := next.isLt
  have ho := other.isLt
  have ha := after.isLt
  have hpEq : previous.val = other.val := by omega
  have hnEq : next.val = after.val := by omega
  exact Prod.ext (Fin.ext hpEq) (Fin.ext hnEq)

/-- An actual token-field read is a coordinate linear map of the full jointly trainable parameter space.
Source: the shared raw vocabulary table, with no separate per-occurrence weights. -/
def sharedTokenRead (token : Fin V) (slot : Fin 52) : SharedParameters V C →ₗ[ℝ] ℝ :=
  LinearMap.proj (.inl (token, slot))

/-- Initial-state logits are free shared head coordinates.
Source: the proposed actual six-way initial row, without a supplied correct start distribution. -/
def sharedInitialRead (state : Fin 6) : SharedParameters V C →ₗ[ℝ] ℝ :=
  LinearMap.proj (.inr (.inr (.inr (.inl state))))

/-- Conditional output logits are free shared state/group/channel head coordinates.
Source: the proposed actual six-by-five-by-four emission table, jointly trained with token transitions. -/
def sharedEmissionRead (state : Fin 6) (h : Fin 5) (d : Fin 4) : SharedParameters V C →ₗ[ℝ] ℝ :=
  LinearMap.proj (.inr (.inr (.inr (.inr (.inl (state, (h, d)))))))

/-- The real state transition lookup reads the current raw token's shared free transition coordinate.
Source: the 36 actual token fields indexed by previous and next encoder states. -/
def sharedTransitionRead (token : Fin V) (previous next : Fin 6) : SharedParameters V C →ₗ[ℝ] ℝ :=
  sharedTokenRead token (transitionSlot previous next)

/-- The exact global head/position parameter count is linear in the context cap.
Source: the actual finite raw coordinate set, with no path, prototype or interaction-bank parameter. -/
theorem sharedGlobal_card (C : ℕ) : Fintype.card (SharedGlobal C) = C + 129 := by
  simp only [SharedGlobal, Fintype.card_sum, Fintype.card_prod, Fintype.card_fin, Fintype.card_unit]

/-- Total learned storage is 52 vocabulary fields plus context slots and 129 global head coordinates.
Source: the actual shared parameter type; it does not grow with training examples or complete latent histories. -/
theorem sharedField_card (V C : ℕ) : Fintype.card (SharedField V C) = 52 * V + C + 129 := by
  simp only [SharedField, SharedGlobal, Fintype.card_sum, Fintype.card_prod, Fintype.card_fin, Fintype.card_unit]
  omega

/-- The proposed raw fields, fixed decoder, constant and learned position fit the original small width exactly.
Source: the actual 52-field layout and its twelve separate residual/readout axes; operator realization remains required. -/
theorem sharedEmbedding_width : 52 + 10 + 1 + 1 = (64 : ℕ) := by omega

/-- Complete state/value training on actual raw vocabulary tokens is globally convex in every shared coordinate.
Source: genuine linear table/head reads into MarkovObjective's exact complete likelihood, without frozen state or value weights. -/
theorem sharedMarkovNLL_convex (tokens : List (Fin V))
    (observed : MarkovConfiguration (Fin 6) (Fin 5) (Fin 4) tokens.length) :
    ConvexOn ℝ Set.univ (markovNLL (sharedInitialRead (V := V) (C := C))
      sharedTransitionRead sharedEmissionRead tokens observed) :=
  markovNLL_convex _ _ _ _ _

/-- The actual shared-coordinate loss is exactly the same full model used by compact causal value inference.
Source: true raw initial/transition/emission lookups and the proved joint probability identity. -/
theorem sharedMarkovNLL_eq (tokens : List (Fin V))
    (observed : MarkovConfiguration (Fin 6) (Fin 5) (Fin 4) tokens.length) (θ : SharedParameters V C) :
    markovNLL sharedInitialRead sharedTransitionRead sharedEmissionRead tokens observed θ =
      -Real.log (markovJoint (fun s => θ (.inr (.inr (.inr (.inl s)))))
        (fun token previous next => θ (.inl (token, transitionSlot previous next))) tokens
        (fun s h d => θ (.inr (.inr (.inr (.inr (.inl (s, (h, d)))))))) observed) := by
  rw [markovNLL_eq]
  simp only [sharedInitialRead, sharedTransitionRead, sharedTokenRead, sharedEmissionRead, LinearMap.proj_apply]

/-- The largest Basis vocabulary uses a compact shared table at its actual recall context cap.
Source: recall vocabulary 548 and context cap 64, with the proposed actual head/global field count. -/
theorem sharedRecall_parameters : Fintype.card (SharedField 548 64) = 28689 := by
  rw [sharedField_card]

/-- The last state transition occupies field 35, while the last pointer value occupies field 51.
Source: actual raw slot arithmetic, showing both complete table layouts fit the same token embedding. -/
example : (transitionSlot 5 5).val = 35 ∧ (valueSlot 4 3).val = 51 := by
  decide

end
end Transformer.GPTMini.Convex.Structured
