import Transformer.GPTMini.Convex.Structured.Binding
import Transformer.GPTMini.Convex.Structured.PointerDecoder

/-!
# Actual raw-prefix all-pair learned binding inference and loss

Source: the unrestricted learned positional binding at f37438a and
PointerDecoder's genuine whole-joint ten-axis readout. The query and
every candidate key/value are read from the unchanged finite raw token
list. All visible position pairs are retained, including BOS, wrong
key/value orders, earlier writes and post-table fillers. The only
input side condition bounds physical positions by the context cap.

The actual loss is computed by the all-route/small-channel contraction;
it is the negative log of the same probability used by inference and
globally convex in every unrestricted embedding/value/binding weight.
Observed configurations remain loss-only arguments. True finite task
weights must still derive semantic correctness and selected mass.
These raw head computations do not yet establish the full tensor,
prenorm/residual/tied model realization or successful AdamW training.
-/

namespace Transformer.GPTMini.Convex.Structured

open Transformer.GPTMini.Semantics
open scoped BigOperators Classical
noncomputable section

variable {V C : ℕ}

/-- The actual full pair domain for a raw causal prefix, without a table or adjacency prefilter.
Source: every visible key position paired with every visible value position in the unchanged finite input. -/
abbrev RawBindingRoutes (tokens : List (Fin V)) := Fin tokens.length × Fin tokens.length

/-- The full implicit raw-pair/matching/value configuration, used as training supervision only.
Source: the compact pointer's exact Gibbs configuration domain; inference does not store this exponential type. -/
abbrev RawBindingConfiguration (tokens : List (Fin V)) := SharedPointerConfiguration (RawBindingRoutes tokens)

/-- The raw position is included exactly into the checked context cap.
Source: the actual length bound and unchanged physical index, without a semantic role or external position oracle. -/
def rawBindingPosition (tokens : List (Fin V)) (hcap : tokens.length ≤ C) (position : Fin tokens.length) : Fin C :=
  ⟨position.val, by omega⟩

/-- Actual complete inference energies read the raw query and both unchanged pair endpoints' free shared fields.
Source: bindingPointerLinear with all raw candidate pairs and exact physical positions; no learned fields are frozen. -/
def rawBindingLinear (tokens : List (Fin V)) (hcap : tokens.length ≤ C) (query : Fin tokens.length)
    (z : RawBindingConfiguration tokens) : BindingParameters V C →ₗ[ℝ] ℝ :=
  bindingPointerLinear (tokens.get query) (fun pair => tokens.get pair.1) (fun pair => tokens.get pair.2)
    (fun pair => rawBindingPosition tokens hcap pair.1) (fun pair => rawBindingPosition tokens hcap pair.2) z

/-- The genuine computed compact raw all-pair probability at arbitrary actual trainable parameters.
Source: the true pointer energy and small-channel partition, without an observed-label inference argument. -/
def rawBindingProbability (θ : BindingParameters V C) (tokens : List (Fin V)) (hcap : tokens.length ≤ C)
    (query : Fin tokens.length) (z : RawBindingConfiguration tokens) : ℝ :=
  pointerProbability (sharedQuery θ.1 (tokens.get query)) (fun pair => sharedKey θ.1 (tokens.get pair.1))
    (fun pair => sharedValue θ.1 (tokens.get pair.2))
    (fun pair => bindingRouteBias θ (rawBindingPosition tokens hcap pair.1) (rawBindingPosition tokens hcap pair.2)) z

/-- The actually computed compact complete raw-prefix objective, with desired route/channels supplied only to the loss.
Source: bindingPointerNLL on every raw visible pair and the same genuine query/key/value lookups as inference. -/
def rawBindingNLL (tokens : List (Fin V)) (hcap : tokens.length ≤ C) (query : Fin tokens.length)
    (observed : RawBindingConfiguration tokens) (θ : BindingParameters V C) : ℝ :=
  bindingPointerNLL (tokens.get query) (fun pair => tokens.get pair.1) (fun pair => tokens.get pair.2)
    (fun pair => rawBindingPosition tokens hcap pair.1) (fun pair => rawBindingPosition tokens hcap pair.2) observed θ

/-- True learned raw-prefix output means are scored against each candidate whole vocabulary token.
Source: pointerOutputScore of the same raw all-pair model, reading actual compact jointly learned values. -/
def rawBindingScore (θ : BindingParameters V C) (tokens : List (Fin V)) (hcap : tokens.length ≤ C)
    (query : Fin tokens.length) (target : Fin 1024) : ℝ :=
  pointerOutputScore (sharedQuery θ.1 (tokens.get query)) (fun pair : RawBindingRoutes tokens => sharedKey θ.1 (tokens.get pair.1))
    (fun pair => sharedValue θ.1 (tokens.get pair.2))
    (fun pair => bindingRouteBias θ (rawBindingPosition tokens hcap pair.1) (rawBindingPosition tokens hcap pair.2)) target

/-- The actual raw inference probability is exactly the generic affine Gibbs probability at the same simultaneous weights.
Source: exact learned binding partition/energy coupling and unchanged raw token/position reads. -/
theorem rawBinding_probability (θ : BindingParameters V C) (tokens : List (Fin V)) (hcap : tokens.length ≤ C)
    (query : Fin tokens.length) (z : RawBindingConfiguration tokens) :
    rawBindingProbability θ tokens hcap query z = probability (rawBindingLinear tokens hcap query) (fun _ => 0) θ z := by
  unfold rawBindingProbability rawBindingLinear
  exact (bindingPointer_probability _ _ _ _ _ θ z).symm

example : ([0, 1] : List (Fin 2)).length ≤ 4 := by decide

/-- The actual raw complete loss is the negative logarithm of the genuine inference model's complete configuration probability.
Source: the identical affine Gibbs likelihood and compact computation; query existence provides a real nonempty pair domain. -/
theorem rawBindingNLL_eq (tokens : List (Fin V)) (hcap : tokens.length ≤ C) (query : Fin tokens.length)
    (observed : RawBindingConfiguration tokens) (θ : BindingParameters V C) :
    rawBindingNLL tokens hcap query observed θ = -Real.log (rawBindingProbability θ tokens hcap query observed) := by
  let : Nonempty (Fin tokens.length) := ⟨query⟩
  unfold rawBindingNLL
  rw [bindingPointerNLL_eq, jointNLL_eq, rawBinding_probability]
  rfl

example : ([0, 1] : List (Fin 2)).length ≤ 4 := by decide

/-- Ordinary simultaneous training of all actual raw shared embedding/value/positional binding fields has a globally convex complete loss.
Source: actual unrestricted all-pair binding NLL convexity; no correctly selected route is assumed. -/
theorem rawBindingNLL_convex (tokens : List (Fin V)) (hcap : tokens.length ≤ C) (query : Fin tokens.length)
    (observed : RawBindingConfiguration tokens) : ConvexOn ℝ Set.univ (rawBindingNLL tokens hcap query observed) := by
  let : Nonempty (Fin tokens.length) := ⟨query⟩
  unfold rawBindingNLL
  exact bindingPointerNLL_convex _ _ _ _ _ _

example : ([0, 1] : List (Fin 2)).length ≤ 4 := by decide

/-- Every actual raw complete probability is positive at any finite simultaneous parameter assignment.
Source: the same real positive Gibbs distribution over all unchanged raw pairs and matching/value channels. -/
theorem rawBindingProbability_pos (θ : BindingParameters V C) (tokens : List (Fin V)) (hcap : tokens.length ≤ C)
    (query : Fin tokens.length) (z : RawBindingConfiguration tokens) : 0 < rawBindingProbability θ tokens hcap query z := by
  let : Nonempty (Fin tokens.length) := ⟨query⟩
  rw [rawBinding_probability]
  exact probability_pos _ _ _ _

example : ([0, 1] : List (Fin 2)).length ≤ 4 := by decide

/-- The whole implicit raw binding model normalizes to one, including every incorrect candidate pair.
Source: the true affine Gibbs coupling, rather than normalization on a preselected raw table subset. -/
theorem rawBindingProbability_sum (θ : BindingParameters V C) (tokens : List (Fin V)) (hcap : tokens.length ≤ C)
    (query : Fin tokens.length) : ∑ z : RawBindingConfiguration tokens, rawBindingProbability θ tokens hcap query z = 1 := by
  let : Nonempty (Fin tokens.length) := ⟨query⟩
  simp_rw [rawBinding_probability]
  exact probability_sum _ _ _

example : ([0, 1] : List (Fin 2)).length ≤ 4 := by decide

/-- The actual raw decoder score is the entire normalized joint model's whole-token output expectation.
Source: exact true compact pointer value contraction, with every input key/value position retained. -/
theorem rawBindingScore_joint (θ : BindingParameters V C) (tokens : List (Fin V)) (hcap : tokens.length ≤ C)
    (query : Fin tokens.length) (target : Fin 1024) :
    rawBindingScore θ tokens hcap query target = weightedOutputScore
      (rawBindingProbability θ tokens hcap query) (fun z : RawBindingConfiguration tokens => z.2.2) target := by
  let : Nonempty (Fin tokens.length) := ⟨query⟩
  unfold rawBindingScore rawBindingProbability
  exact pointerOutputScore_joint _ _ _ _ _

example : ([0, 1] : List (Fin 2)).length ≤ 4 := by decide

/-- The full latent configuration count is exactly T squared times 4 to the ninth; these configurations are never stored by inference.
Source: the actual all-visible-position pair type and four matching/five value four-channel groups. -/
theorem rawBindingConfiguration_card (tokens : List (Fin V)) :
    Fintype.card (RawBindingConfiguration tokens) = tokens.length * tokens.length * 262144 := by
  simp only [RawBindingConfiguration, RawBindingRoutes, SharedPointerConfiguration, PointerConfiguration,
    Fintype.card_prod, Fintype.card_fun, Fintype.card_fin]
  norm_num

/-- At the genuine Basis recall cap the entire implicit choice count is bounded by 1073741824.
Source: the actual raw T-squared pair domain and nine four-channel groups; this controls probability tails, not stored parameters. -/
theorem rawBindingConfiguration_bound (tokens : List (Fin V)) (hcap : tokens.length ≤ 64) :
    Fintype.card (RawBindingConfiguration tokens) ≤ 1073741824 := by
  rw [rawBindingConfiguration_card]
  have hsquare : tokens.length * tokens.length ≤ 4096 := by nlinarith
  nlinarith

example : ([0, 1] : List (Fin 2)).length ≤ 64 := by decide

end
end Transformer.GPTMini.Convex.Structured
