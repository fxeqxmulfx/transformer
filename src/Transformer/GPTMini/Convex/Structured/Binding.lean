import Transformer.GPTMini.Convex.Structured.SharedPointer
import Mathlib.LinearAlgebra.Prod

/-!
# Jointly learned key/value binding over all visible position pairs

Source: the actual affine-energy shared pointer at 7598fbc and the
raw adjacent MQAR semantics at cbafbe9. This proposed extension learns
one unrestricted potential for every possible relative displacement.
The forward considers all supplied key/value position pairs; adjacency
is neither a fixed input interaction nor a hard routing mask. Complete
configuration labels remain training-only data supervision.

Every actual raw Q/K/value, absolute-position, chronology and relative
binding coordinate enters the complete energy linearly. The computed
small-channel normalizer equals its true joint Gibbs partition, so the
actual complete NLL is globally convex in their entire free domain.
Storage remains linear in vocabulary and context cap. An all-pair route
sum costs quadratically many positions per query, but never enumerates
latent channel combinations. Full raw recall, successful AdamW training
and tensor/residual/tied block realization are subsequent obligations.
-/

namespace Transformer.GPTMini.Convex.Structured

open scoped BigOperators Classical
noncomputable section

variable {V C : ℕ} {J : Type*} [Fintype J]

/-- Unrestricted actual shared weights and one free scalar for each physical relative displacement.
Source: the proposed learned binding extension; weights are shared across every raw prefix and record. -/
abbrev BindingParameters (V C : ℕ) := SharedParameters V C × (Fin (C + (C - 1)) → ℝ)

/-- Every signed visible key/value displacement has its own finite physical-position index.
Source: the shift by C-1 covers differences from -(C-1) to C-1, without filtering any pair. -/
def bindingRelativeIndex (key value : Fin C) : Fin (C + (C - 1)) :=
  ⟨value.val + C - 1 - key.val, by have hk := key.isLt; have hv := value.isLt; omega⟩

/-- The relative coordinate for a successor is derived exactly from the actual two raw positions.
Source: signed displacement indexing; no predecessor or paired-token encoder is supplied to inference. -/
theorem bindingRelative_adjacent (key value : Fin C) :
    (bindingRelativeIndex key value).val = C ↔ key.val + 1 = value.val := by
  simp only [bindingRelativeIndex]
  have hk := key.isLt
  have hv := value.isLt
  omega

/-- Actual route bias combines free absolute position, chronology and free learned key/value displacement.
Source: the proposed affine physical-position potentials, with no data-label argument in the forward. -/
def bindingRouteBias (θ : BindingParameters V C) (key value : Fin C) : ℝ :=
  sharedRouteBias θ.1 value + θ.2 (bindingRelativeIndex key value)

/-- The true complete route/channel energy as a linear map of every simultaneous raw trainable coordinate.
Source: sharedPointerLinear plus the independently learned relative-position projection. -/
def bindingPointerLinear (query : Fin V) (keys values : J → Fin V) (keyPositions valuePositions : J → Fin C)
    (z : SharedPointerConfiguration J) : BindingParameters V C →ₗ[ℝ] ℝ :=
  (sharedPointerLinear query keys values valuePositions z).comp
    (LinearMap.fst ℝ (SharedParameters V C) (Fin (C + (C - 1)) → ℝ)) +
    (LinearMap.proj (bindingRelativeIndex (keyPositions z.1) (valuePositions z.1)) :
      (Fin (C + (C - 1)) → ℝ) →ₗ[ℝ] ℝ).comp
        (LinearMap.snd ℝ (SharedParameters V C) (Fin (C + (C - 1)) → ℝ))

omit [Fintype J] in
/-- Actual simultaneous linear reads equal the real pointer energy including the freely learned binding potential.
Source: genuine coordinate projections and shared pointer evaluation, rather than an assumed correct key/value score. -/
theorem bindingPointerLinear_apply (query : Fin V) (keys values : J → Fin V) (keyPositions valuePositions : J → Fin C)
    (z : SharedPointerConfiguration J) (θ : BindingParameters V C) :
    bindingPointerLinear query keys values keyPositions valuePositions z θ =
      pointerEnergy (sharedQuery θ.1 query) (fun j => sharedKey θ.1 (keys j))
        (fun j => sharedValue θ.1 (values j)) (fun j => bindingRouteBias θ (keyPositions j) (valuePositions j)) z := by
  simp only [bindingPointerLinear, LinearMap.add_apply, LinearMap.comp_apply,
    LinearMap.fst_apply, LinearMap.snd_apply, LinearMap.proj_apply, sharedPointerLinear_apply,
    pointerEnergy, bindingRouteBias]
  ring

/-- The actually computed all-route/small-channel normalizer equals the whole affine joint partition.
Source: exact Factorial contraction, applied to the same learned raw binding energy. -/
theorem bindingPointerPartition_eq (query : Fin V) (keys values : J → Fin V) (keyPositions valuePositions : J → Fin C)
    (θ : BindingParameters V C) :
    pointerPartition (sharedQuery θ.1 query) (fun j => sharedKey θ.1 (keys j))
      (fun j => sharedValue θ.1 (values j)) (fun j => bindingRouteBias θ (keyPositions j) (valuePositions j)) =
      partition (bindingPointerLinear query keys values keyPositions valuePositions) (fun _ => 0) θ := by
  unfold partition
  simp only [energy, add_zero, bindingPointerLinear_apply]
  convert pointerPartition_eq (sharedQuery θ.1 query) (fun j => sharedKey θ.1 (keys j))
    (fun j => sharedValue θ.1 (values j)) (fun j => bindingRouteBias θ (keyPositions j) (valuePositions j))

/-- The actual contracted complete likelihood with observed route/channels kept outside inference.
Source: the true compact partition and raw learned energy, with no frozen matching, values or binding weights. -/
def bindingPointerNLL (query : Fin V) (keys values : J → Fin V) (keyPositions valuePositions : J → Fin C)
    (observed : SharedPointerConfiguration J) (θ : BindingParameters V C) : ℝ :=
  Real.log (pointerPartition (sharedQuery θ.1 query) (fun j => sharedKey θ.1 (keys j))
    (fun j => sharedValue θ.1 (values j)) (fun j => bindingRouteBias θ (keyPositions j) (valuePositions j))) -
  pointerEnergy (sharedQuery θ.1 query) (fun j => sharedKey θ.1 (keys j))
    (fun j => sharedValue θ.1 (values j)) (fun j => bindingRouteBias θ (keyPositions j) (valuePositions j)) observed

/-- The computed learned-binding loss is exactly the actual full affine Gibbs NLL at every raw parameter assignment.
Source: the real compact normalizer identity and simultaneous complete-energy linearity. -/
theorem bindingPointerNLL_eq (query : Fin V) (keys values : J → Fin V) (keyPositions valuePositions : J → Fin C)
    (observed : SharedPointerConfiguration J) (θ : BindingParameters V C) :
    bindingPointerNLL query keys values keyPositions valuePositions observed θ =
      jointNLL (bindingPointerLinear query keys values keyPositions valuePositions) (fun _ => 0) observed θ := by
  unfold bindingPointerNLL jointNLL
  rw [bindingPointerPartition_eq]
  simp only [energy, add_zero, bindingPointerLinear_apply]

/-- Joint Q/K/value and key/value-position learning is globally convex in all unrestricted actual raw weights.
Source: the identical computed complete loss and affine joint Gibbs convexity; output-only CE is not this objective. -/
theorem bindingPointerNLL_convex [Nonempty J] (query : Fin V) (keys values : J → Fin V)
    (keyPositions valuePositions : J → Fin C) (observed : SharedPointerConfiguration J) :
    ConvexOn ℝ Set.univ (bindingPointerNLL query keys values keyPositions valuePositions observed) := by
  have h : bindingPointerNLL query keys values keyPositions valuePositions observed =
      jointNLL (bindingPointerLinear query keys values keyPositions valuePositions) (fun _ => 0) observed := by
    funext θ
    exact bindingPointerNLL_eq _ _ _ _ _ _ θ
  rw [h]
  exact jointNLL_convex _ _ _

/-- Actual compact inference uses the identical full joint distribution optimized by the learned-binding objective.
Source: the true exponential energy and exact contracted partition, independent of correctness of any route. -/
theorem bindingPointer_probability (query : Fin V) (keys values : J → Fin V) (keyPositions valuePositions : J → Fin C)
    (θ : BindingParameters V C) (z : SharedPointerConfiguration J) :
    probability (bindingPointerLinear query keys values keyPositions valuePositions) (fun _ => 0) θ z =
      pointerProbability (sharedQuery θ.1 query) (fun j => sharedKey θ.1 (keys j))
        (fun j => sharedValue θ.1 (values j)) (fun j => bindingRouteBias θ (keyPositions j) (valuePositions j)) z := by
  unfold probability pointerProbability
  rw [← bindingPointerPartition_eq]
  simp only [energy, add_zero, bindingPointerLinear_apply]

/-- Learned binding adds exactly 2C-1 scalars to the genuine common shared coordinate count for a nonempty context.
Source: every physical signed displacement, with no pair-indexed or per-example learned table. -/
theorem bindingParameters_count (V C : ℕ) (hC : 0 < C) :
    Fintype.card (SharedField V C) + Fintype.card (Fin (C + (C - 1))) = 52 * V + 3 * C + 128 := by
  rw [sharedField_card, Fintype.card_fin]
  omega

example : (0 : ℕ) < 64 := by omega

/-- The largest Basis vocabulary and actual recall cap need only 28816 jointly trained shared scalars.
Source: the exact 52-token-field/state-head layout plus all 127 signed binding offsets at context 64. -/
theorem bindingRecall_parameters :
    Fintype.card (SharedField 548 64) + Fintype.card (Fin (64 + (64 - 1))) = 28816 := by
  rw [bindingParameters_count 548 64 (by omega)]

/-- Every visible raw position pair is a real candidate under the same jointly convex learned-binding objective.
Source: a two-position control with all four pairs and unrestricted shared content/value/relative weights. -/
example : ConvexOn ℝ Set.univ (bindingPointerNLL (V := 2) (C := 2) 0
    (fun pair : Fin 2 × Fin 2 => pair.1) (fun pair => pair.2) (fun pair => pair.1) (fun pair => pair.2)
    ((0, 1), ((fun _ => 0), (fun _ => 0)))) := bindingPointerNLL_convex _ _ _ _ _ _

end
end Transformer.GPTMini.Convex.Structured
