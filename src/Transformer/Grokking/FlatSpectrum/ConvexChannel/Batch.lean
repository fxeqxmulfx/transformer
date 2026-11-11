module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Representation

/-!
# Feasible updates for independently visited query blocks

Source: `review/flat_spectrum.md`, *Convex channel minibatch updates*, CM1.
A minibatch may visit only some queries. Each visited query can use its own
convex step size; unvisited queries use zero. The joint tensor remains in
the same affine/nonnegative feasible domain. This proves feasibility and
the actual logit update, not convergence of a stochastic optimizer.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel

open scoped BigOperators

variable {Q K V C : Type*} [Fintype K] [Fintype C]

/-- A convex update with one step size per query block (CM1). -/
def mixRows (η : Q → ℝ) (A B : Channel Q K V C) : Channel Q K V C :=
  fun q k v c => (1 - η q) * A q k v c + η q * B q k v c

/-- Query-dependent steps preserve the entire joint-channel domain, including
the value-independent routing marginals (CM1). -/
theorem feasible_mixRows {A B : Channel Q K V C} (hA : Feasible A) (hB : Feasible B)
    (η : Q → ℝ) (hη : ∀ q, 0 ≤ η q ∧ η q ≤ 1) : Feasible (mixRows η A B) := by
  refine ⟨?_, ?_, ?_⟩
  · intro q k v c
    exact add_nonneg (mul_nonneg (sub_nonneg.mpr (hη q).2) (hA.1 q k v c))
      (mul_nonneg (hη q).1 (hB.1 q k v c))
  · intro q k v w
    simp only [mixRows, Finset.sum_add_distrib, ← Finset.mul_sum,
      hA.2.1 q k v w, hB.2.1 q k v w]
  · intro q v
    simp only [mixRows, Finset.sum_add_distrib, ← Finset.mul_sum,
      hA.2.2 q v, hB.2.2 q v]
    ring

omit [Fintype C] in
/-- The actual logits undergo the same query-specific convex update (CM1). -/
theorem logits_mixRows (η : Q → ℝ) (A B : Channel Q K V C) (q : Q) (D : K → V) :
    logits (mixRows η A B) q D = (1 - η q) • logits A q D + η q • logits B q D := by
  funext c
  simp only [logits, mixRows, Finset.sum_add_distrib, ← Finset.mul_sum,
    Pi.add_apply, Pi.smul_apply, smul_eq_mul]

omit [Fintype K] [Fintype C] in
/-- A query absent from the minibatch retains every trained joint coefficient
when its step size is zero; it is not reset to an oracle's default row (CM1). -/
theorem mixRows_zero (η : Q → ℝ) (A B : Channel Q K V C) (q : Q) (hη : η q = 0) :
    mixRows η A B q = A q := by
  funext k v c
  simp [mixRows, hη]

example : Feasible (mixRows (fun q : Bool => if q then (2 : ℝ) / 3 else 1 / 3)
    (generator (fun _ : Bool => (id : Bool → Bool))) (generator (fun _ => Bool.not))) := by
  apply feasible_mixRows (feasible_generator _) (feasible_generator _)
  intro q
  cases q <;> norm_num

example := mixRows_zero (fun q : Bool => if q then (1 : ℝ) / 2 else 0)
  (generator (fun _ : Bool => (id : Bool → Bool))) (generator (fun _ => Bool.not)) false
  (by simp)

end Transformer.Grokking.FlatSpectrum.ConvexChannel
