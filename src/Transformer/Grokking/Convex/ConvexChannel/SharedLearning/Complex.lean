module

public import Transformer.Grokking.Convex.ConvexChannel.SharedLearning.Ambiguity
public import Transformer.Grokking.Convex.ConvexChannel.Capacity
public import Transformer.Grokking.Convex.ConvexDepth.BoxHead

/-!
# The shared learner's complete complex forward and capacity bounds

Source: `review/convex/moments.md`, CM73. A fixed unitary code acts on the value
alphabet separately in every categorical slot. Its complete conjugate head is
linear in the jointly trained routing/output tensor. The actual multiplication
equals the shared channel's logits, retaining every real and imaginary term.
No observed-label equations or target program enter this construction.

The source bounds every feature by one and the entire per-class head budget
by twice the value-alphabet size. Storage remains 2|K||V||C| real coordinates;
the joint learner has |K||V||C| coordinates. CC6's bound is applied to the
actual represented output operator, not used to supply unseen answers.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.SharedLearning

open scoped BigOperators
open ConvexDepth FlatSpectrum
open Landscape.Neural (crossEntropy)

variable {Q K V C S X : Type*} [Fintype K] [Fintype V] [DecidableEq V]

/-- Fixed categorical aligned inputs in complete complex coordinates, CM73. -/
def encodedFeatures (U : Matrix V V ℂ) (align : Q → K ≃ K) :
    K × (V × Bool) → (Q × (K → V)) → ℝ :=
  fun f x => complexFeatures U f.2 (alignedData align x.1 x.2 f.1)

/-- Every joint coefficient enters a complete conjugate head linearly, CM73.
The domain is still Feasible A, independent of all observed labels. -/
noncomputable def encodedHead (U : Matrix V V ℂ) (A : Channel Unit K V C) :
    K × (V × Bool) → C → ℝ :=
  fun f c => complexCoefficients (tableHead U (A () f.1)) f.2 c

/-- Expanding all actual complex products recovers the shared model's whole
class-logit vector, CM73. Neither a scalar-only decoder nor an ideal argmax
is substituted for the forward computation. -/
theorem encoded_logits (U : Matrix V V ℂ) (hU : U ∈ Matrix.unitaryGroup V ℂ)
    (align : Q → K ≃ K) (A : Channel Unit K V C) (q : Q) (D : K → V) :
    headLogits (encodedFeatures U align) (encodedHead U A) (q, D) =
      logits (tiedChannel align A) q D := by
  rw [logits_tiedChannel]
  funext c
  unfold headLogits
  rw [Fintype.sum_prod_type]
  change (∑ k, headLogits (complexFeatures U)
    (complexCoefficients (tableHead U (A () k))) (alignedData align q D k) c) = _
  apply Finset.sum_congr rfl
  intro k _
  exact congrFun (tableHead_logits U hU (A () k) (alignedData align q D k)) c

omit [Fintype K] in
/-- Every actual source coordinate obeys the fixed unitary bound, CM73. -/
theorem encoded_feature_bound (U : Matrix V V ℂ) (hU : U ∈ Matrix.unitaryGroup V ℂ)
    (align : Q → K ≃ K) (f : K × (V × Bool)) (x : Q × (K → V)) :
    |encodedFeatures U align f x| ≤ 1 :=
  complexFeatures_bound U hU f.2 (alignedData align x.1 x.2 f.1)

variable [Fintype C] [Nonempty V]

/-- The full shared head satisfies a label-independent coefficient budget,
CM73. Each block is charged; common routing mass sums to one across all slots. -/
theorem encoded_head_budget (U : Matrix V V ℂ) (hU : U ∈ Matrix.unitaryGroup V ℂ)
    {A : Channel Unit K V C} (hA : Feasible A) :
    encodedHead U A ∈ headBudget (2 * Fintype.card V) := by
  obtain ⟨v₀⟩ := ‹Nonempty V›
  let mass : K → ℝ := fun k => ∑ c, A () k v₀ c
  have hm (k : K) : 0 ≤ mass k := Finset.sum_nonneg fun c _ => hA.1 () k v₀ c
  have hb (k : K) (v : V) (c : C) : |A () k v c| ≤ mass k := by
    rw [abs_of_nonneg (hA.1 () k v c)]
    have h := Finset.single_le_sum (s := Finset.univ) (f := A () k v)
      (fun d _ => hA.1 () k v d) (Finset.mem_univ c)
    exact h.trans_eq (hA.2.1 () k v v₀)
  intro c
  rw [Fintype.sum_prod_type]
  calc
    _ ≤ ∑ k, 2 * mass k * (Fintype.card V : ℝ) := by
      apply Finset.sum_le_sum
      intro k _
      exact tableHead_budget U hU (A () k) (hm k) (hb k) c
    _ = (2 * Fintype.card V) * ∑ k, mass k := by
      rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro k _
      ring
    _ = _ := by rw [hA.2.2 () v₀, mul_one]

omit [Nonempty V] in
/-- Ordinary CE of the actual full complex multiplication is convex in every
jointly trained entry, CM73. Records retain their original finite indices. -/
theorem encoded_loss_convex [Fintype S] (U : Matrix V V ℂ)
    (hU : U ∈ Matrix.unitaryGroup V ℂ) (align : Q → K ≃ K)
    (query : S → Q) (data : S → K → V) (label : S → C) :
    ConvexOn ℝ Set.univ (fun A : Channel Unit K V C => ∑ s,
      crossEntropy (headLogits (encodedFeatures U align) (encodedHead U A)
        (query s, data s)) (label s)) := by
  simp only [encoded_logits U hU]
  exact shared_answerLoss_convex align query data label

omit [Nonempty V] in
/-- Empirical optimality of the actual complex CE identifies all shared raw
answers under the observed separation and coverage conditions, CM73. -/
theorem trained_encoded_logits [Fintype S] [DecidableEq K] [DecidableEq C] [Nontrivial C]
    (U : Matrix V V ℂ) (hU : U ∈ Matrix.unitaryGroup V ℂ)
    (align : Q → K ≃ K) (query : S → Q) (data : S → K → V) (label : S → C)
    (slot : K) (decode : V → C)
    (hlabel : ∀ s, label s = decode (alignedData align (query s) (data s) slot))
    (hcover : Covers (fun s => alignedData align (query s) (data s)) slot)
    (hsep : SeparatesSlots (fun s => alignedData align (query s) (data s)) label slot)
    {A : Channel Unit K V C} (hA : Feasible A)
    (hmin : IsMinOn (fun B => ∑ s, crossEntropy
      (headLogits (encodedFeatures U align) (encodedHead U B) (query s, data s)) (label s))
      {B | Feasible B} A) (q : Q) (D : K → V) (c : C) :
    headLogits (encodedFeatures U align) (encodedHead U A) (q, D) c =
      if c = decode (alignedData align q D slot) then 1 else 0 := by
  let : Nonempty V := ⟨D slot⟩
  simp only [encoded_logits U hU] at hmin ⊢
  exact trained_shared_logits align query data label slot decode hlabel hcover hsep hA hmin q D c

omit [DecidableEq V] [Nonempty V] in
/-- All real and imaginary head coordinates remain in the storage count,
CM73. A norm bound independent of slot count is not a storage reduction. -/
theorem encoded_coordinate_count : Fintype.card ((K × (V × Bool)) × C) =
    2 * Fintype.card K * Fintype.card V * Fintype.card C := by
  simp only [Fintype.card_prod, Fintype.card_bool]
  ring

omit [Nonempty V] in
/-- CC6 bounds the actual complex learner's scalar output operator, CM73.
The target flatness hypothesis supplies a necessary capacity bound only. -/
theorem encoded_flat_error_bound [Fintype Q] [Fintype X] [DecidableEq Q] [DecidableEq X]
    (U : Matrix V V ℂ) (hU : U ∈ Matrix.unitaryGroup V ℂ) (align : Q → K ≃ K)
    {T : Matrix Q X ℂ} {σ₂ : ℝ} (hσ : 0 < σ₂) (hT : IsFlat T σ₂)
    (φ : C → ℂ) (A : Channel Unit K V C) (data : X → K → V) :
    σ₂ * ((min (Fintype.card Q) (Fintype.card X) : ℕ) - Fintype.card (K × V)) ≤
      frobeniusSq (T - (show Matrix Q X ℂ from fun q x => ∑ c,
        (headLogits (encodedFeatures U align) (encodedHead U A) (q, data x) c : ℂ) * φ c)) := by
  have he : (show Matrix Q X ℂ from fun q x => ∑ c,
      (headLogits (encodedFeatures U align) (encodedHead U A) (q, data x) c : ℂ) * φ c) =
      ConvexChannel.outputKernel φ (tiedChannel align A) data := by
    ext q x
    simp only [ConvexChannel.outputKernel, encoded_logits U hU]
  rw [he]
  exact flat_error_bound hσ hT φ (tiedChannel align A) data

/-- A genuine imaginary source and head coordinate are both retained, CM73.
The Fourier witness uses a generated class with one selected value. -/
theorem encoded_imaginary_witness :
    complexFeatures fourier4 (1, true) (1 : Fin 4) = -1 / 2 ∧
    encodedHead fourier4 (selected false (fun v : Fin 4 => decide (v = 1)))
      (false, (1, true)) true = -1 / 2 := by
  norm_num [encodedHead, selected, Forward.channelOf, complexFeatures,
    complexCoefficients, tableHead, complexTable, Matrix.mul_apply,
    Matrix.conjTranspose_apply, fourier4, Fin.sum_univ_succ]

example := encoded_logits fourier4 fourier4_unitary (fun _ : Unit => Equiv.refl Bool)
  (selected false (fun v : Fin 4 => decide (v = 1))) () (fun k => if k then 0 else 1)

example := encoded_feature_bound fourier4 fourier4_unitary
  (fun _ : Unit => Equiv.refl Bool) (false, (1, true)) ((), fun _ => 1)

example := encoded_head_budget fourier4 fourier4_unitary
  (feasible_selected false (fun v : Fin 4 => decide (v = 1)))

example := encoded_loss_convex fourier4 fourier4_unitary
  (fun _ : Unit => Equiv.refl Bool) (fun _ : Bool => ())
  (fun b k => if b = k then (1 : Fin 4) else 0) (id : Bool → Bool)

example := trained_encoded_logits (1 : Matrix Bool Bool ℂ) (Matrix.unitaryGroup Bool ℂ).one_mem
  witnessAlign witnessQuery witnessData witnessLabel 0 Bool.not
  (by simpa only [witness_aligned] using witness_realizable)
  (by simpa only [witness_aligned] using witness_coverage)
  (by simpa only [witness_aligned] using witness_separation)
  (feasible_selected 0 Bool.not)
  (by simpa only [encoded_logits _ (Matrix.unitaryGroup Bool ℂ).one_mem,
    ConvexChannel.answerLoss] using witness_optimum)
  true unseenData false

example := encoded_coordinate_count (K := Fin 3) (V := Bool) (C := Bool)

-- Rank eight versus four categorical features gives a positive bound, even
-- though the actual constant initialization has no access to the target.
example := encoded_flat_error_bound (1 : Matrix Bool Bool ℂ)
  (Matrix.unitaryGroup Bool ℂ).one_mem (fun _ : Fin 8 => Equiv.refl Bool)
  (by norm_num : (0 : ℝ) < 1)
  (unitary_isFlat (1 : Matrix (Fin 8) (Fin 8) ℂ) (Matrix.unitaryGroup (Fin 8) ℂ).one_mem)
  boolSign (initial : Channel Unit Bool Bool Bool)
  (![id, Bool.not, id, Bool.not, id, Bool.not, id, Bool.not] : Fin 8 → Bool → Bool)

end Transformer.Grokking.Convex.ConvexChannel.SharedLearning
