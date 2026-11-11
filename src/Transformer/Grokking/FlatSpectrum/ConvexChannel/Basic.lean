module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Defs

/-!
# Convex channels generate probability-mass logits

Source: `review/flat_spectrum.md`, *Convex channels and capacity*, CC1-CC2.
The domain is jointly convex in all routing/output interaction coordinates.
The output is linear in those coordinates. The fixed unit logit scale is
retained from ConvexRecall; it bounds confidence, not argmax expressivity.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel

open scoped BigOperators

variable {Q K V C S : Type*} [Fintype K] [Fintype C] [Fintype S]

/-- Joint routing and value generation have a convex feasible domain (CC2). -/
theorem convex_feasible : Convex ℝ {W : Channel Q K V C | Feasible W} := by
  intro A hA B hB a b ha hb hab
  refine ⟨?_, ?_, ?_⟩
  · intro q k v c
    exact add_nonneg (mul_nonneg ha (hA.1 q k v c)) (mul_nonneg hb (hB.1 q k v c))
  · intro q k v w
    change (∑ c, (a * A q k v c + b * B q k v c)) =
      ∑ c, (a * A q k w c + b * B q k w c)
    simp only [Finset.sum_add_distrib, ← Finset.mul_sum, hA.2.1 q k v w, hB.2.1 q k v w]
  · intro q v
    change (∑ k, ∑ c, (a * A q k v c + b * B q k v c)) = 1
    simp only [Finset.sum_add_distrib, ← Finset.mul_sum, hA.2.2 q v, hB.2.2 q v]
    simpa using hab

omit [Fintype C] in
/-- Every learned coordinate enters the logits linearly (CC2). -/
theorem logits_linear (A B : Channel Q K V C) (a b : ℝ) (q : Q) (D : K → V) :
    logits (a • A + b • B) q D = a • logits A q D + b • logits B q D := by
  funext c
  simp [logits, Finset.sum_add_distrib, Finset.mul_sum]

/-- Ordinary output cross-entropy remains jointly convex in every learned
entry, including the entries that generate new output values (CC2). -/
theorem answerLoss_convex (query : S → Q) (data : S → K → V) (label : S → C) :
    ConvexOn ℝ Set.univ (answerLoss query data label) := by
  refine ⟨convex_univ, ?_⟩
  intro A _ B _ a b ha hb hab
  change (∑ s, Landscape.Neural.crossEntropy (logits (a • A + b • B) (query s) (data s))
    (label s)) ≤ _
  simp only [answerLoss, smul_eq_mul, Finset.mul_sum, ← Finset.sum_add_distrib]
  apply Finset.sum_le_sum
  intro s _
  rw [logits_linear]
  exact (ConvexRecall.convex_crossEntropy (label s)).2 (Set.mem_univ _) (Set.mem_univ _) ha hb hab

/-- Each generated logit is nonnegative (CC1). -/
lemma logits_nonneg {W : Channel Q K V C} (hW : Feasible W) (q : Q) (D : K → V) (c : C) :
    0 ≤ logits W q D c :=
  Finset.sum_nonneg fun k _ => hW.1 q k (D k) c

/-- For every input dictionary, the output mass is exactly one. Routing
mass independence is what permits different values at different keys (CC1). -/
theorem sum_logits [Nonempty V] {W : Channel Q K V C} (hW : Feasible W) (q : Q) (D : K → V) :
    ∑ c, logits W q D c = 1 := by
  obtain ⟨v⟩ := ‹Nonempty V›
  simp only [logits]
  rw [Finset.sum_comm]
  calc
    _ = ∑ k, ∑ c, W q k v c := Finset.sum_congr rfl fun k _ => hW.2.1 q k (D k) v
    _ = 1 := hW.2.2 q v

/-- The unit output-mass constraint bounds every class logit by one (CC1). -/
lemma logits_le_one [Nonempty V] {W : Channel Q K V C} (hW : Feasible W)
    (q : Q) (D : K → V) (c : C) : logits W q D c ≤ 1 := by
  have h := Finset.single_le_sum (s := Finset.univ) (f := logits W q D)
    (fun j _ => logits_nonneg hW q D j) (Finset.mem_univ c)
  simpa [sum_logits hW q D] using h

end Transformer.Grokking.FlatSpectrum.ConvexChannel
