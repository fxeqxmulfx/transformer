module

public import Transformer.GainMatroid.Section2_Classes
public import Transformer.GainMatroid.Section2_DeletionCycles

/-!
# Balanced cycles along an edge map, and joints of gain graphs

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.3, for Lemma 5.8 (§5.2). An injective edge map
that keeps ends and gains matches balanced cycles (`OrientedGraph.mem_balancedCycles_image_iff`):
the cycles correspond (`OrientedGraph.IsCycle.map_edge`), and one switching makes both sides
identity. Over an abelian group a loop is a joint exactly when its gain is not the identity, since
switching leaves the gain on a loop unchanged (`OrientedGraph.isJoint_gainBiased_iff`).
`G.withLoop v` adds one loop `none` at `v` to `G`.
-/

@[expose] public section

namespace Transformer.GainMatroid

namespace OrientedGraph

variable {V E E' Γ : Type*} {G : OrientedGraph V E} {H : OrientedGraph V E'} {ι : E' → E}

/-- An injective edge map keeping the ends and the gains matches the balanced cycles. -/
@[to_additive mem_addBalancedCycles_image_iff
  /-- An injective edge map keeping the ends and the additive gains matches the balanced
  cycles. -/]
theorem mem_balancedCycles_image_iff [DecidableEq V] [Group Γ] (hinj : Function.Injective ι)
    (hι : ∀ f, G.tail (ι f) = H.tail f ∧ G.head (ι f) = H.head f) {γ : E' → Γ} {γ' : E → Γ}
    (hγ : ∀ f, γ' (ι f) = γ f) {C : Set E'} :
    C ∈ H.balancedCycles γ ↔ ι '' C ∈ G.balancedCycles γ' := by
  rw [mem_balancedCycles_iff, mem_balancedCycles_iff]
  constructor
  · rintro ⟨hC, η, hη⟩
    refine ⟨hC.map_edge hinj hι, η, ?_⟩
    rintro _ ⟨f, hf, rfl⟩
    have h := hη f hf
    rwa [switch_apply, ← (hι f).1, ← (hι f).2, ← hγ, ← switch_apply] at h
  · rintro ⟨hC, η, hη⟩
    refine ⟨hC.of_map_edge hinj hι, η, fun f hf => ?_⟩
    have h := hη (ι f) ⟨f, hf, rfl⟩
    rwa [switch_apply, (hι f).1, (hι f).2, hγ, ← switch_apply] at h

/-- A loop is a joint of a gain graph over an abelian group exactly when its gain is not `1`. -/
@[to_additive isJoint_addGainBiased_iff
  /-- A loop is a joint of an additive gain graph exactly when its gain is not `0`. -/]
theorem isJoint_gainBiased_iff [DecidableEq V] [CommGroup Γ] {γ : E → Γ} {e : E}
    (he : G.tail e = G.head e) : (G.gainBiased γ).IsJoint e ↔ γ e ≠ 1 := by
  refine ⟨fun h hγ => h.2 (mem_balancedCycles_iff.2 ⟨isCycle_singleton_of_loop he, 1, ?_⟩),
    fun hγ => ⟨he, fun hb => ?_⟩⟩
  · rintro f rfl
    rw [switch_one, hγ]
  · obtain ⟨-, η, hη⟩ := mem_balancedCycles_iff.1 hb
    have h := hη e rfl
    rw [switch_apply, he, mul_comm, ← mul_assoc, mul_inv_cancel, one_mul] at h
    exact hγ h

/-- `G` with one more edge `none`, a loop at `v`. -/
def withLoop (G : OrientedGraph V E) (v : V) : OrientedGraph V (Option E) :=
  ⟨fun e => e.elim v G.tail, fun e => e.elim v G.head⟩

end OrientedGraph

open OrientedGraph

/-- The pumpkin with a loop at `0`, of gain `-1`. -/
def pumpkinLoopGain : Option (Fin 3) → ℚˣ := fun e => e.elim (-1) fun _ => 1

/-- The pumpkin with a loop at `0`, of additive gain `1`. -/
def pumpkinLoopAddGain : Option (Fin 3) → ℚ := fun e => e.elim 1 fun _ => 0

example := mem_balancedCycles_image_iff (G := pumpkin.withLoop 0) (H := pumpkin)
  (γ := fun _ => (1 : ℚˣ)) (γ' := pumpkinLoopGain) (Option.some_injective _)
  (fun _ => ⟨rfl, rfl⟩) (fun _ => rfl) (C := {0, 1})

example := mem_addBalancedCycles_image_iff (G := pumpkin.withLoop 0) (H := pumpkin)
  (γ := fun _ => (0 : ℚ)) (γ' := pumpkinLoopAddGain) (Option.some_injective _)
  (fun _ => ⟨rfl, rfl⟩) (fun _ => rfl) (C := {0, 1})

example := isJoint_gainBiased_iff (G := pumpkin.withLoop 0) (γ := pumpkinLoopGain) (e := none) rfl

example :=
  isJoint_addGainBiased_iff (G := pumpkin.withLoop 0) (γ := pumpkinLoopAddGain) (e := none) rfl

/-- The loop of `pumpkin.withLoop 0` is a joint under `pumpkinLoopGain`. -/
example : ((pumpkin.withLoop 0).gainBiased pumpkinLoopGain).IsJoint none :=
  (isJoint_gainBiased_iff rfl).2 fun h => by
    have h' : ((-1 : ℚˣ) : ℚ) = ((1 : ℚˣ) : ℚ) := congrArg Units.val h
    norm_num at h'

end Transformer.GainMatroid
