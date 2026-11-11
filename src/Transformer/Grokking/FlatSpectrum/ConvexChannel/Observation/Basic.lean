module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Representation

/-!
# Observed logits at an ordinary answer-CE optimum

Source: `review/flat_spectrum.md`, *Observable features determine unseen
predictions*, O1. Realizability here means that a feasible reference channel
gives unit target mass on the observed examples. It imposes no test-output
condition. Every global optimizer has the same observed logit vectors.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.Observation

open scoped BigOperators
open Landscape.Neural (crossEntropy)

variable {Q K V C S : Type*} [Fintype K] [Fintype C] [Fintype S]
  [Nonempty V] [DecidableEq C]

/-- A channel exactly realizes the observed answer labels at unit logit
scale. This condition concerns training samples only (O1). -/
def Fits (query : S → Q) (data : S → K → V) (label : S → C)
    (W : Channel Q K V C) : Prop :=
  ∀ s, logits W (query s) (data s) = ConvexRecall.targetLogits (label s)

/-- An exact training witness attains the global lower bound for ordinary
answer CE over the feasible channel domain (O1). -/
theorem fits_loss_le (query : S → Q) (data : S → K → V) (label : S → C)
    {A W : Channel Q K V C} (hfit : Fits query data label A) (hW : Feasible W) :
    answerLoss query data label A ≤ answerLoss query data label W := by
  apply Finset.sum_le_sum
  intro s _
  rw [hfit s]
  exact ConvexRecall.target_crossEntropy_le _ (logits_nonneg hW (query s) (data s))
    (logits_le_one hW (query s) (data s) _)

/-- A training witness is an attained optimizer of the actual answer
loss, without auxiliary feature or routing labels (O1). -/
theorem isMinOn_of_fits (query : S → Q) (data : S → K → V) (label : S → C)
    {A : Channel Q K V C} (hfit : Fits query data label A) :
    IsMinOn (answerLoss query data label) {W | Feasible W} A := by
  intro W hW
  exact fits_loss_le query data label hfit hW

/-- Any channel attaining the witness's ordinary CE has unit target
mass on each observed sample (O1). -/
theorem observed_mass_one [Nontrivial C] (query : S → Q) (data : S → K → V)
    (label : S → C) {A W : Channel Q K V C}
    (hfit : Fits query data label A) (hW : Feasible W)
    (hmin : answerLoss query data label W ≤ answerLoss query data label A) (s : S) :
    logits W (query s) (data s) (label s) = 1 := by
  have hle : ∀ j ∈ (Finset.univ : Finset S),
      crossEntropy (logits A (query j) (data j)) (label j) ≤
        crossEntropy (logits W (query j) (data j)) (label j) := by
    intro j _
    rw [hfit j]
    exact ConvexRecall.target_crossEntropy_le _ (logits_nonneg hW (query j) (data j))
      (logits_le_one hW (query j) (data j) _)
  have he := le_antisymm (fits_loss_le query data label hfit hW) hmin
  have hs := (Finset.sum_eq_sum_iff_of_le hle).mp he s (Finset.mem_univ s)
  rw [hfit s] at hs
  exact ConvexRecall.target_mass_eq_one _ (logits_nonneg hW (query s) (data s))
    (logits_le_one hW (query s) (data s) _) hs.symm.le

/-- Nonnegative unit-mass logits with target mass one are exactly the
target vector. The common softmax-shift ambiguity is removed by unit mass (O1). -/
theorem simplex_eq_target {z : C → ℝ} (t : C) (hz : ∀ c, 0 ≤ z c)
    (hsum : ∑ c, z c = 1) (ht : z t = 1) : z = ConvexRecall.targetLogits t := by
  funext c
  by_cases hc : c = t
  · subst c
    simpa [ConvexRecall.targetLogits] using ht
  · have h := Finset.sum_erase_add Finset.univ z (Finset.mem_univ t)
    rw [hsum, ht] at h
    have hle := Finset.single_le_sum (s := Finset.univ.erase t) (f := z)
      (fun j _ => hz j) (Finset.mem_erase.mpr ⟨hc, Finset.mem_univ c⟩)
    have hc0 : z c = 0 := by linarith [hz c]
    simp [ConvexRecall.targetLogits, hc, hc0]

/-- Every feasible optimum agrees with the reference channel on every
observed logit, not only on the training argmax (O1). -/
theorem observed_logits_eq [Nontrivial C] (query : S → Q) (data : S → K → V)
    (label : S → C) {A W : Channel Q K V C}
    (hfit : Fits query data label A) (hW : Feasible W)
    (hmin : answerLoss query data label W ≤ answerLoss query data label A) (s : S) :
    logits W (query s) (data s) = logits A (query s) (data s) := by
  rw [hfit s]
  exact simplex_eq_target _ (logits_nonneg hW (query s) (data s))
    (sum_logits hW (query s) (data s)) (observed_mass_one query data label hfit hW hmin s)

end Transformer.Grokking.FlatSpectrum.ConvexChannel.Observation
