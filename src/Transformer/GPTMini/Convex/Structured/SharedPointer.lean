import Transformer.GPTMini.Convex.Structured.SharedSlots
import Transformer.GPTMini.Convex.Structured.PointerValues

/-!
# Joint pointer training with learned positional potentials

Source: the exact compact pointer contraction at 5371b5f and the actual
common raw parameter coordinates at bdbea00. Query, key and values read
their shared free token fields. Each candidate route also reads its free
physical-position potential and learned chronology. There is no fixed
table-record mask, table-role input or per-prefix parameter table.

The complete energy is realized as an actual linear map of every raw
token/position/chronology weight. The computed compact NLL equals the
genuine joint Gibbs NLL and is globally convex in the unrestricted shared
parameter domain. Its inference probability is exactly the same model.
This is an algebraic construction, not yet a proof that the positional
weights learn table exclusion or that the raw model solves Basis recall.
Mixture training and actual prenorm/residual/tied integration remain open.
-/

namespace Transformer.GPTMini.Convex.Structured

open scoped BigOperators Classical
noncomputable section

variable {V C : ℕ} {J : Type*} [Fintype J]

/-- The implicit route/matching/value configuration; its exponential channel product is never stored by inference.
Source: the exact four-group matching/five-group value pointer proposal. -/
abbrev SharedPointerConfiguration (J : Type*) :=
  PointerConfiguration (G := Fin 4) (C := Fin 4) (H := Fin 5) (D := Fin 4) (J := J)

/-- Actual free query fields from the shared raw token embedding.
Source: the first sixteen coordinates, without a fixed input code or learned-weight product. -/
def sharedQuery (θ : SharedParameters V C) (token : Fin V) (g c : Fin 4) : ℝ :=
  θ (.inl (token, querySlot g c))

/-- Actual independent key log potentials from the same vocabulary table at the physical predecessor token.
Source: the next sixteen coordinates, retaining ordinary learned content matching. -/
def sharedKey (θ : SharedParameters V C) (token : Fin V) (g c : Fin 4) : ℝ :=
  θ (.inl (token, keySlot g c))

/-- Actual jointly trained value channel potentials at the physical candidate value token.
Source: the final twenty raw token fields, with no frozen prototype values. -/
def sharedValue (θ : SharedParameters V C) (token : Fin V) (h : Fin 5) (d : Fin 4) : ℝ :=
  θ (.inl (token, valueSlot h d))

/-- Free raw positional coordinates are linear reads of the same jointly trained parameter vector.
Source: one learned potential per context position, without a predefined table gate. -/
def sharedPositionRead (position : Fin C) : SharedParameters V C →ₗ[ℝ] ℝ :=
  LinearMap.proj (.inr (.inl position))

/-- Chronology is one additional free scalar shared by every route in every prefix.
Source: the proposed learned physical-position preference. -/
def sharedChronologyRead : SharedParameters V C →ₗ[ℝ] ℝ :=
  LinearMap.proj (.inr (.inr (.inl ())))

/-- Actual route bias combines a freely learned positional potential with freely learned chronology.
Source: the proposed table exclusion/latest-write representation, evaluated without semantic role labels at inference. -/
def sharedRouteBias (θ : SharedParameters V C) (position : Fin C) : ℝ :=
  θ (.inr (.inl position)) + θ (.inr (.inr (.inl ()))) * (position.val : ℝ)

/-- The real complete energy is a linear map of raw shared Q/K/value/position/chronology coordinates simultaneously.
Source: the actual free log-potential formula, with all physical lookup indices supplied only by raw input positions. -/
def sharedPointerLinear (query : Fin V) (keys values : J → Fin V) (positions : J → Fin C)
    (z : SharedPointerConfiguration J) : SharedParameters V C →ₗ[ℝ] ℝ :=
  sharedPositionRead (positions z.1) + ((positions z.1).val : ℝ) • sharedChronologyRead +
    (∑ g, (sharedTokenRead query (querySlot g (z.2.1 g)) +
      sharedTokenRead (keys z.1) (keySlot g (z.2.1 g)))) +
    ∑ h, sharedTokenRead (values z.1) (valueSlot h (z.2.2 h))

omit [Fintype J] in
/-- The actual linear parameter map evaluates to the genuine jointly learned compact pointer energy.
Source: true coordinate reads and physical-position bias, not an assumed correct route score. -/
theorem sharedPointerLinear_apply (query : Fin V) (keys values : J → Fin V) (positions : J → Fin C)
    (z : SharedPointerConfiguration J) (θ : SharedParameters V C) :
    sharedPointerLinear query keys values positions z θ =
      pointerEnergy (sharedQuery θ query) (fun j => sharedKey θ (keys j))
        (fun j => sharedValue θ (values j)) (fun j => sharedRouteBias θ (positions j)) z := by
  simp only [sharedPointerLinear, LinearMap.add_apply, LinearMap.smul_apply, LinearMap.sum_apply,
    sharedPositionRead, sharedChronologyRead, sharedTokenRead, LinearMap.proj_apply, smul_eq_mul,
    pointerEnergy, channelEnergy, sharedQuery, sharedKey, sharedValue, sharedRouteBias]
  ring

/-- The compact normalizer is the exact full affine-energy partition at actual shared raw parameters.
Source: finite route/channel contraction with the same genuine linear parameter map. -/
theorem sharedPointerPartition_eq (query : Fin V) (keys values : J → Fin V) (positions : J → Fin C)
    (θ : SharedParameters V C) :
    pointerPartition (sharedQuery θ query) (fun j => sharedKey θ (keys j))
      (fun j => sharedValue θ (values j)) (fun j => sharedRouteBias θ (positions j)) =
      partition (sharedPointerLinear query keys values positions) (fun _ => 0) θ := by
  unfold partition
  simp only [energy, add_zero, sharedPointerLinear_apply]
  convert pointerPartition_eq (sharedQuery θ query) (fun j => sharedKey θ (keys j))
    (fun j => sharedValue θ (values j)) (fun j => sharedRouteBias θ (positions j))

/-- The computed compact training objective with observed routes/channels external to inference.
Source: the true small-sum pointer normalizer and the same learned raw complete energy. -/
def sharedPointerNLL (query : Fin V) (keys values : J → Fin V) (positions : J → Fin C)
    (observed : SharedPointerConfiguration J) (θ : SharedParameters V C) : ℝ :=
  Real.log (pointerPartition (sharedQuery θ query) (fun j => sharedKey θ (keys j))
    (fun j => sharedValue θ (values j)) (fun j => sharedRouteBias θ (positions j))) -
  pointerEnergy (sharedQuery θ query) (fun j => sharedKey θ (keys j))
    (fun j => sharedValue θ (values j)) (fun j => sharedRouteBias θ (positions j)) observed

/-- The actual contracted shared-table loss is precisely the full affine Gibbs objective.
Source: exact small-sum partition contraction and the genuine linear raw parameter energy. -/
theorem sharedPointerNLL_eq (query : Fin V) (keys values : J → Fin V) (positions : J → Fin C)
    (observed : SharedPointerConfiguration J) (θ : SharedParameters V C) :
    sharedPointerNLL query keys values positions observed θ =
      jointNLL (sharedPointerLinear query keys values positions) (fun _ => 0) observed θ := by
  unfold sharedPointerNLL jointNLL
  rw [sharedPointerPartition_eq]
  simp only [energy, add_zero, sharedPointerLinear_apply]

/-- Joint pointer training stays globally convex with all raw matching, value, position and chronology weights free.
Source: the actual computed loss's exact affine-energy identity, on the whole shared parameter domain. -/
theorem sharedPointerNLL_convex [Nonempty J] (query : Fin V) (keys values : J → Fin V) (positions : J → Fin C)
    (observed : SharedPointerConfiguration J) :
    ConvexOn ℝ Set.univ (sharedPointerNLL query keys values positions observed) := by
  have h : sharedPointerNLL query keys values positions observed =
      jointNLL (sharedPointerLinear query keys values positions) (fun _ => 0) observed := by
    funext θ
    exact sharedPointerNLL_eq _ _ _ _ _ θ
  rw [h]
  exact jointNLL_convex _ _ _

/-- Actual complete shared-pointer losses are nonnegative at every free parameter assignment.
Source: the genuine joint likelihood, independently of route correctness or learned table exclusion. -/
theorem sharedPointerNLL_nonneg (query : Fin V) (keys values : J → Fin V) (positions : J → Fin C)
    (observed : SharedPointerConfiguration J) (θ : SharedParameters V C) :
    0 ≤ sharedPointerNLL query keys values positions observed θ := by
  rw [sharedPointerNLL_eq]
  exact jointNLL_nonneg _ _ _ _

/-- Compact inference is the same joint distribution trained by the actual shared-position/Q/K/value objective.
Source: the exact real energy and partition identity, rather than a separately normalized surrogate predictor. -/
theorem sharedPointer_probability (query : Fin V) (keys values : J → Fin V) (positions : J → Fin C)
    (θ : SharedParameters V C) (z : SharedPointerConfiguration J) :
    probability (sharedPointerLinear query keys values positions) (fun _ => 0) θ z =
      pointerProbability (sharedQuery θ query) (fun j => sharedKey θ (keys j))
        (fun j => sharedValue θ (values j)) (fun j => sharedRouteBias θ (positions j)) z := by
  unfold probability pointerProbability
  rw [← sharedPointerPartition_eq]
  simp only [energy, add_zero, sharedPointerLinear_apply]

/-- A real two-record input permits convex training of the same free shared raw fields and physical positions.
Source: actual raw vocabulary/position lookups, with observed matching/value channels kept outside inference. -/
example : ConvexOn ℝ Set.univ (sharedPointerNLL (V := 2) (C := 4) 0 ![0, 1] ![1, 0] ![1, 3]
    (0, ((fun _ => 0), (fun _ => 0)))) := sharedPointerNLL_convex _ _ _ _ _

end
end Transformer.GPTMini.Convex.Structured
