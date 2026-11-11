module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Oracle

/-!
# Sharing a joint channel across input-aligned queries

Source: `review/flat_spectrum.md`, *Repairing minibatch and full-batch failures*,
MF3. A fixed, input-only permutation aligns the categorical slots of each query.
One joint tensor is trained on the aligned dictionaries. Expanding it into the
old query-indexed tensor is linear and preserves feasibility. Ordinary CE,
including a fixed logit multiplier, remains convex in all trained coordinates.

This removes independent decoder rows for equivalent aligned queries. The
equivalence is an explicit structural choice, not a consequence of flatness.
No labels or task-specific arithmetic enter the alignment or the forward map.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel

open scoped BigOperators

variable {Q K V C S : Type*} [Fintype K] [Fintype C] [Fintype S]

/-- Relabel the input slots using a fixed query-dependent permutation, MF3. -/
def alignedData (align : Q → K ≃ K) (q : Q) (D : K → V) : K → V :=
  fun k => D ((align q).symm k)

/-- One directly trained joint tensor, expanded to the query-indexed domain. -/
def tiedChannel (align : Q → K ≃ K) (W : Channel Unit K V C) : Channel Q K V C :=
  fun q k v c => W () (align q k) v c

omit [Fintype S] in
/-- **Sharing preserves the entire feasible domain**, MF3: the only change is
a fixed permutation of the routing slots for each query. -/
theorem feasible_tiedChannel (align : Q → K ≃ K) {W : Channel Unit K V C}
    (hW : Feasible W) : Feasible (tiedChannel align W) := by
  refine ⟨fun q k v c => hW.1 () (align q k) v c,
    fun q k v w => hW.2.1 () (align q k) v w, ?_⟩
  intro q v
  change (∑ k, ∑ c, W () (align q k) v c) = 1
  rw [(align q).sum_comp (fun k => ∑ c, W () k v c)]
  exact hW.2.2 () v

omit [Fintype K] [Fintype C] [Fintype S] in
/-- **The shared model is linear in its joint parameters**, MF3. No product of
independently learned routing and decoding factors is introduced. -/
theorem tiedChannel_linear (align : Q → K ≃ K) (A B : Channel Unit K V C) (a b : ℝ) :
    tiedChannel align (a • A + b • B) = a • tiedChannel align A + b • tiedChannel align B := by
  funext q k v c
  change a * A () (align q k) v c + b * B () (align q k) v c = _
  rfl

omit [Fintype C] [Fintype S] in
/-- The expanded model reads precisely the aligned input, MF3. -/
theorem logits_tiedChannel (align : Q → K ≃ K) (W : Channel Unit K V C) (q : Q) (D : K → V) :
    logits (tiedChannel align W) q D = logits W () (alignedData align q D) := by
  funext c
  change (∑ k, W () (align q k) (D k) c) =
    ∑ k, W () k (D ((align q).symm k)) c
  simpa using (align q).sum_comp (fun k => W () k (D ((align q).symm k)) c)

/-- Training shared parameters is ordinary answer CE on aligned inputs, MF3. -/
theorem answerLoss_tiedChannel (align : Q → K ≃ K) (query : S → Q) (data : S → K → V)
    (label : S → C) (W : Channel Unit K V C) :
    answerLoss query data label (tiedChannel align W) =
      answerLoss (fun _ : S => ()) (fun s => alignedData align (query s) (data s)) label W := by
  simp only [answerLoss, logits_tiedChannel]

/-- **The shared full-batch objective remains convex**, MF3. Its domain is the
existing convex feasible set for a single joint query block. -/
theorem shared_answerLoss_convex (align : Q → K ≃ K) (query : S → Q)
    (data : S → K → V) (label : S → C) :
    ConvexOn ℝ Set.univ (fun W : Channel Unit K V C =>
      answerLoss query data label (tiedChannel align W)) := by
  simp only [answerLoss_tiedChannel]
  exact answerLoss_convex _ _ _

omit [Fintype C] [Fintype S] in
/-- **Equal aligned dictionaries give equal predictions across queries**, MF3.
The hypothesis is additional input structure, not spectral flatness. -/
theorem logits_tied_eq_of_aligned_eq (align : Q → K ≃ K) (W : Channel Unit K V C)
    {q q' : Q} {D D' : K → V}
    (h : alignedData align q D = alignedData align q' D') :
    logits (tiedChannel align W) q D = logits (tiedChannel align W) q' D' := by
  rw [logits_tiedChannel, logits_tiedChannel, h]

/-- Ordinary CE with a fixed logit multiplier, MF3. The multiplier is not a
jointly trained parameter. -/
noncomputable def scaledAnswerLoss (scale : ℝ) (query : S → Q) (data : S → K → V)
    (label : S → C) (W : Channel Q K V C) : ℝ :=
  ∑ s, Landscape.Neural.crossEntropy (scale • logits W (query s) (data s)) (label s)

/-- **A fixed logit scale preserves convexity**, MF3. Positive scales also keep
the ordering of a fixed model's logits; no scale-selection guarantee is asserted. -/
theorem scaledAnswerLoss_convex (scale : ℝ) (query : S → Q) (data : S → K → V)
    (label : S → C) : ConvexOn ℝ Set.univ (scaledAnswerLoss scale query data label) := by
  refine ⟨convex_univ, ?_⟩
  intro A _ B _ a b ha hb hab
  change (∑ s, Landscape.Neural.crossEntropy
    (scale • logits (a • A + b • B) (query s) (data s)) (label s)) ≤ _
  simp only [scaledAnswerLoss, smul_eq_mul, Finset.mul_sum, ← Finset.sum_add_distrib]
  apply Finset.sum_le_sum
  intro s _
  rw [logits_linear]
  have heq : scale • (a • logits A (query s) (data s) + b • logits B (query s) (data s)) =
      a • (scale • logits A (query s) (data s)) + b • (scale • logits B (query s) (data s)) := by
    funext c
    simp only [Pi.smul_apply, Pi.add_apply, smul_eq_mul]
    ring
  rw [heq]
  exact (ConvexRecall.convex_crossEntropy (label s)).2
    (Set.mem_univ _) (Set.mem_univ _) ha hb hab

example := feasible_tiedChannel (fun _ : Bool => Equiv.refl Bool)
  (feasible_vertex (fun _ : Unit => false) (fun _ _ => (id : Bool → Bool)))

example := logits_tied_eq_of_aligned_eq (fun _ : Bool => Equiv.refl Bool)
  (vertex (fun _ : Unit => false) (fun _ _ => (id : Bool → Bool)))
  (q := false) (q' := true) (D := id) (D' := id) rfl

example := shared_answerLoss_convex (fun _ : Bool => Equiv.refl Bool) (id : Bool → Bool)
  (fun _ => (id : Bool → Bool)) id

example := scaledAnswerLoss_convex (10 : ℝ) (id : Bool → Bool)
  (fun _ => (id : Bool → Bool)) id

end Transformer.Grokking.FlatSpectrum.ConvexChannel
