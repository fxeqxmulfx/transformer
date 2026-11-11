module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Representation

/-!+# Answer CE identifies the atoms contradicted by training

Source: `review/flat_spectrum.md`, *Convex channels and capacity*, CC5.
A realizable generated-recall rule attains the minimum sample CE. Equality
of total loss forces target mass one on each training sample, which rules
out every active parameter atom that would generate a different answer.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel

open scoped BigOperators
open Landscape.Neural (crossEntropy)

variable {K V C S : Type*} [Fintype K] [Fintype C] [Fintype S]
  [Nonempty V] [DecidableEq K] [DecidableEq C]

/-- The generated rule's ideal logits minimize CE on each input (CC5). -/
theorem sample_loss_generator_le (g : K → V → C) {W : Channel K K V C}
    (hW : Feasible W) (q : K) (D : K → V) :
    crossEntropy (logits (generator g) q D) (g q (D q)) ≤
      crossEntropy (logits W q D) (g q (D q)) := by
  have hz : logits (generator g) q D = ConvexRecall.targetLogits (g q (D q)) := by
    funext c
    exact logits_generator g q D c
  rw [hz]
  exact ConvexRecall.target_crossEntropy_le _ (logits_nonneg hW q D)
    (logits_le_one hW q D _)

/-- A realizable generated rule is an attained global CE optimizer.
Labels remain ordinary answer labels, not parameter supervision (CC5). -/
theorem answerLoss_generator_le (g : K → V → C) (query : S → K)
    (data : S → K → V) (label : S → C)
    (hlabel : ∀ s, label s = g (query s) (data s (query s)))
    {W : Channel K K V C} (hW : Feasible W) :
    answerLoss query data label (generator g) ≤ answerLoss query data label W := by
  apply Finset.sum_le_sum
  intro s _
  rw [hlabel s]
  exact sample_loss_generator_le g hW (query s) (data s)

/-- An optimum exists for any realizable generated-recall dataset (CC5). -/
theorem isMinOn_generator (g : K → V → C) (query : S → K)
    (data : S → K → V) (label : S → C)
    (hlabel : ∀ s, label s = g (query s) (data s (query s))) :
    IsMinOn (answerLoss query data label) {W | Feasible W} (generator g) := by
  intro W hW
  exact answerLoss_generator_le g query data label hlabel hW

/-- Equal total CE forces perfect target mass at every observed sample,
not just a correct argmax (CC5). -/
theorem sample_target_mass_one [Nontrivial C] (g : K → V → C)
    (query : S → K) (data : S → K → V) (label : S → C)
    (hlabel : ∀ s, label s = g (query s) (data s (query s)))
    {W : Channel K K V C} (hW : Feasible W)
    (hmin : answerLoss query data label W ≤ answerLoss query data label (generator g))
    (s : S) : logits W (query s) (data s) (label s) = 1 := by
  have hle : ∀ j ∈ (Finset.univ : Finset S),
      crossEntropy (logits (generator g) (query j) (data j)) (label j) ≤
        crossEntropy (logits W (query j) (data j)) (label j) := by
    intro j _
    rw [hlabel j]
    exact sample_loss_generator_le g hW (query j) (data j)
  have he := le_antisymm (answerLoss_generator_le g query data label hlabel hW) hmin
  have hs := (Finset.sum_eq_sum_iff_of_le hle).mp he s (Finset.mem_univ s)
  have hz : logits (generator g) (query s) (data s) =
      ConvexRecall.targetLogits (label s) := by
    funext c
    simp only [logits_generator, hlabel s, ConvexRecall.targetLogits]
  rw [hz] at hs
  exact ConvexRecall.target_mass_eq_one _ (logits_nonneg hW (query s) (data s))
    (logits_le_one hW (query s) (data s) _) hs.symm.le

omit [DecidableEq K] in
/-- A sample with unit target mass gives zero weight to every active
parameter atom predicting a different class (CC5). -/
theorem atom_eq_zero {W : Channel K K V C} (hW : Feasible W)
    (q : K) (D : K → V) (t : C) (ht : logits W q D t = 1)
    (k : K) (c : C) (hc : c ≠ t) : W q k (D k) c = 0 := by
  have hs := Finset.sum_erase_add Finset.univ (logits W q D) (Finset.mem_univ t)
  rw [sum_logits hW q D, ht] at hs
  have hlog := Finset.single_le_sum (s := Finset.univ.erase t)
    (f := logits W q D) (fun j _ => logits_nonneg hW q D j)
    (Finset.mem_erase.mpr ⟨hc, Finset.mem_univ c⟩)
  have hatom := Finset.single_le_sum (s := Finset.univ)
    (f := fun j => W q j (D j) c) (fun j _ => hW.1 q j (D j) c) (Finset.mem_univ k)
  change W q k (D k) c ≤ logits W q D c at hatom
  linarith [hW.1 q k (D k) c]

example := sample_loss_generator_le (fun _ : Bool => Bool.not)
  (feasible_generator (fun _ : Bool => Bool.not)) true (id : Bool → Bool)

example := answerLoss_generator_le (fun _ : Bool => Bool.not) (id : Bool → Bool)
  (fun _ => id) Bool.not (fun _ => rfl) (feasible_generator (fun _ : Bool => Bool.not))

example := isMinOn_generator (fun _ : Bool => Bool.not) (id : Bool → Bool)
  (fun _ => id) Bool.not (fun _ => rfl)

example := sample_target_mass_one (fun _ : Bool => Bool.not) (id : Bool → Bool)
  (fun _ => id) Bool.not (fun _ => rfl) (feasible_generator (fun _ : Bool => Bool.not))
  le_rfl true

example := atom_eq_zero (feasible_generator (fun _ : Bool => Bool.not))
  true (id : Bool → Bool) false (by simp [logits_generator]) false true (by decide)

end Transformer.Grokking.FlatSpectrum.ConvexChannel
