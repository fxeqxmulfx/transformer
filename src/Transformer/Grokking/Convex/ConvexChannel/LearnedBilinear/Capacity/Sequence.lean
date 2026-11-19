module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedBilinear.Capacity.Basic
public import Transformer.Grokking.Convex.ConvexChannel.LearnedBilinear.Updates.Scaling

/-!
# Capacity certificates for the actual next-original-object model

Source: `review/convex/moments.md`, CM91. Inputs remain sequences of immutable
objects and outputs remain original objects. The finite coefficient check
certifies the actual learned complex forward; it does not replace inference
with a fixed feature predictor. The extra squared budget must be justified
for the target family, rather than inferred from observed loss alone.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedBilinear.Capacity

open scoped BigOperators
open LearnedHinge (Prefix)
open Identification

variable {O : Type*} [Fintype O] [DecidableEq O] {T N : ℕ}

/-- All original observed forms, including zero self-comparisons, CM91. -/
def observedForms (tokens : Fin N → Prefix O (T + 1)) (labels : Fin N → O) :
    Fin N × O → RealCoordinate O T → ℝ :=
  fun j => coefficient (tokens j.1) (labels j.1) j.2

/-- A finite squared-capacity certificate, CM91. A nonzero full residual is
allowed. This predicate uses training labels, query objects and the declared
budget; it contains no true query label or unknown target parameter. -/
def BoundedCertificate (tokens : Fin N → Prefix O (T + 1)) (labels : Fin N → O)
    (query : Prefix O (T + 1)) (a b : O) (B beta : ℝ) (w : Fin N → O → ℝ) : Prop :=
  (∀ i c, 0 ≤ w i c) ∧ (∀ i, w i (labels i) = 0) ∧
  0 ≤ (∑ i, ∑ c, w i c) - beta ∧
  B * energy (residual (observedForms tokens labels) (coefficient query a b)
    (fun j => w j.1 j.2)) ≤ ((∑ i, ∑ c, w i c) - beta) ^ 2

/-- Capacity-bounded ordinary CE remains convex in the complete joint real
coordinates, CM91. Both learned encoders are still recovered and executed. -/
theorem bounded_loss_convex (tokens : Prefix O (T + 1)) (target : O) (B : ℝ) :
    ConvexOn ℝ {r : RealCoordinate O T → ℝ | energy r ≤ B}
      (fun r => sequenceLoss tokens target (decode r)) := by
  have h := (sequenceLoss_convex tokens target).comp_linearMap Updates.decodeLinear
  have hu : ConvexOn ℝ Set.univ
      (fun r : RealCoordinate O T → ℝ => sequenceLoss tokens target (decode r)) := by
    simpa only [Set.preimage_univ, Function.comp_def, Updates.decodeLinear,
      LinearMap.coe_mk, AddHom.coe_mk] using h
  exact hu.subset (Set.subset_univ _) (energyBall_convex B)

/-- A checked residual budget forces the actual query margin for every
fitting model within capacity, CM91. No claim of certificate completeness is
made; arbitrary movement along CM90's silent coordinate need not stay feasible. -/
theorem bounded_certificate_sound (tokens : Fin N → Prefix O (T + 1))
    (labels : Fin N → O) (query : Prefix O (T + 1)) (a b : O) (B beta : ℝ)
    (w : Fin N → O → ℝ) (hw : BoundedCertificate tokens labels query a b B beta w)
    (p : SequenceParameters O T) (hp : FitsObserved tokens labels p)
    (hcap : energy (encode p) ≤ B) : beta ≤ margin p query a b := by
  let lower : Fin N × O → ℝ := fun j => if j.2 = labels j.1 then 0 else 1
  have hmass : (∑ j : Fin N × O, w j.1 j.2 * lower j) = ∑ i, ∑ c, w i c := by
    rw [Fintype.sum_prod_type]
    apply Finset.sum_congr rfl
    intro i hi
    apply Finset.sum_congr rfl
    intro c hc
    by_cases he : c = labels i
    · subst c
      simp [lower, hw.2.1]
    · simp [lower, he]
  have hfit : ∀ j : Fin N × O,
      lower j ≤ reading (observedForms tokens labels j) (encode p) := by
    intro j
    change lower j ≤ ∑ z, coefficient (tokens j.1) (labels j.1) j.2 z * encode p z
    rw [← margin_expansion]
    by_cases he : j.2 = labels j.1
    · simp [lower, he, margin]
    · simpa [lower, he] using hp j.1 j.2 he
  rw [margin_expansion]
  apply bounded_transfer (observedForms tokens labels) (coefficient query a b) (encode p)
    (fun j => w j.1 j.2) lower B beta (fun j => hw.1 j.1 j.2) hfit hcap
  · simpa only [hmass] using hw.2.2.1
  · simpa only [hmass] using hw.2.2.2

/-- Zero residual recovers CM87 sound certificates, CM91. The bounded
extension does not invalidate the earlier unrestricted identification result. -/
theorem of_exact_certificate (tokens : Fin N → Prefix O (T + 1)) (labels : Fin N → O)
    (query : Prefix O (T + 1)) (a b : O) (B beta : ℝ) (w : Fin N → O → ℝ)
    (hw : Certificate tokens labels query a b beta w) :
    BoundedCertificate tokens labels query a b B beta w := by
  refine ⟨hw.1, hw.2.1, by linarith [hw.2.2.1], ?_⟩
  have hz : residual (observedForms tokens labels) (coefficient query a b)
      (fun j => w j.1 j.2) = 0 := by
    funext z
    simp only [residual, observedForms, Fintype.sum_prod_type, hw.2.2.2 z, sub_self,
      Pi.zero_apply]
  rw [hz]
  simpa [energy] using sq_nonneg ((∑ i, ∑ c, w i c) - beta)

end Transformer.Grokking.Convex.ConvexChannel.LearnedBilinear.Capacity
