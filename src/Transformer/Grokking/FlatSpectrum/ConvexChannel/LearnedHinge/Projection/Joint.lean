module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Projection.Raw

/-!
# Certified projection of every coordinate in the actual complex model

Source: `review/flat_spectrum.md`, CM79. Independent component certificates
sum to the strong Euclidean projection inequality for the whole model. The
metric includes every slot, output object, signed readout and complex component.
This does not preserve a factor-norm regularizer or supply an efficient
enumeration of the complete pattern family.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Projection

open scoped BigOperators
open PatternReduction

variable {O J : Type*} [Fintype O] [DecidableEq O] [Fintype J] {T : ℕ}

/-- Sum the exact squared update distance over the whole complex parameter
array, CM79. Both components and both readout signs are explicitly present. -/
def jointDistance (p q : Parameters J O (ObjectWeights O T)) : ℝ :=
  ∑ j, ∑ c, ∑ s : Bool, ∑ k : Bool,
    squareDistance (project k (p j c s)) (project k (q j c s))

/-- Component certificates certify the whole model's nearest feasible
point, CM79. The competitor may be any point in the full original domain. -/
theorem joint_projection {p q z : Parameters J O (ObjectWeights O T)}
    (mask : J → Pattern (Prefix O T))
    (dual : J → O → Bool → Bool → Prefix O T → ℝ)
    (h : ∀ j c s k, Certificate (fun x => signedRow (mask j (x, k)) x)
      (project k (p j c s)) (project k (q j c s)) (dual j c s k))
    (hz : z ∈ complexDomain rawInput Set.univ mask) :
    jointDistance p q + jointDistance q z ≤ jointDistance p z := by
  simp only [jointDistance, ← Finset.sum_add_distrib]
  apply Finset.sum_le_sum
  intro j hj
  apply Finset.sum_le_sum
  intro c hc
  apply Finset.sum_le_sum
  intro s hs
  apply Finset.sum_le_sum
  intro k hk
  apply certificate_distance (h j c s k)
  intro tokens
  rw [signedRow_dot, ← raw_components]
  exact hz j c s (tokens, k) (Set.mem_univ _)

/-- A complete certified update satisfies both feasibility and the strong
whole-model distance bound, CM79. These properties hold independently of
whether the proposal was formed from a minibatch gradient or forward queries. -/
theorem certified_update {p q : Parameters J O (ObjectWeights O T)}
    (mask : J → Pattern (Prefix O T))
    (dual : J → O → Bool → Bool → Prefix O T → ℝ)
    (h : ∀ j c s k, Certificate (fun x => signedRow (mask j (x, k)) x)
      (project k (p j c s)) (project k (q j c s)) (dual j c s k)) :
    q ∈ complexDomain rawInput Set.univ mask ∧
      ∀ z ∈ complexDomain rawInput Set.univ mask,
        jointDistance p q + jointDistance q z ≤ jointDistance p z :=
  ⟨certificates_mem mask dual h, fun _ hz => joint_projection mask dual h hz⟩

example := joint_projection (O := Bool) (T := 1) (J := Unit) (p := 0) (q := 0) (z := 0)
  (fun _ _ => true) (fun _ _ _ _ _ => 0)
  (by
    intro j c s k
    apply identity_certificate
    intro tokens
    simp [dot, project])
  (zero_mem _ _ _)
example := certified_update (O := Bool) (T := 1) (J := Unit) (p := 0) (q := 0)
  (fun _ _ => true) (fun _ _ _ _ _ => 0) (by
    intro j c s k
    apply identity_certificate
    intro tokens
    simp [dot, project])

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Projection
