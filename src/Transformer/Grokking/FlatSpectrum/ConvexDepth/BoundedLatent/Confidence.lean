module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.BoundedLatent.Prediction

/-!
# What confidence learning changes in ordinary cross-entropy

Source: `review/flat_spectrum.md`, CM64. Exact observed anchors leave one learned
confidence value in the observed CE, rather than CM53's constant fixed margin.
Increasing that value cannot identify additional latent states or unknown labels.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.BoundedLatent

open scoped BigOperators
open Landscape.Neural (crossEntropy)
open LatentPrograms (Parameters features head)

variable {J X S C : Type*} [Fintype J] [Fintype S] [Fintype C]
  [DecidableEq S] [DecidableEq C]

/-- The ordinary observed CE is the finite-margin CE at the actual learned
total mass, CM64. This identity uses every raw class of the complex head. -/
theorem observed_ce (U : Matrix S S ℂ) (hU : U ∈ Matrix.unitaryGroup S ℂ)
    (program : J → X → S) (order : List J) (samples : List (X × C)) {δ R : ℝ}
    {q : Parameters J S C} (hq : q ∈ domain program order samples δ R)
    (t : X × C) (ht : t ∈ samples) :
    crossEntropy (headLogits (features U program) (head U 1 q) t.1) t.2 =
      Real.log (1 + ((Fintype.card C : ℝ) - 1) * Real.exp (-mass q)) := by
  rw [observed_logits U hU program order samples hq t ht]
  exact fixed_margin_ce _ _

/-- Increasing feasible total mass cannot increase observed ordinary CE,
CM64. It changes confidence, with no assertion about an unseen target. -/
theorem observed_ce_antitone (U : Matrix S S ℂ) (hU : U ∈ Matrix.unitaryGroup S ℂ)
    (program : J → X → S) (order : List J) (samples : List (X × C)) {δ R : ℝ}
    {q r : Parameters J S C} (hq : q ∈ domain program order samples δ R)
    (hr : r ∈ domain program order samples δ R) (hm : mass q ≤ mass r)
    (t : X × C) (ht : t ∈ samples) :
    crossEntropy (headLogits (features U program) (head U 1 r) t.1) t.2 ≤
      crossEntropy (headLogits (features U program) (head U 1 q) t.1) t.2 := by
  rw [observed_ce U hU program order samples hr t ht,
    observed_ce U hU program order samples hq t ht]
  let : Nonempty C := ⟨t.2⟩
  have hn : (1 : ℝ) ≤ Fintype.card C := by exact_mod_cast Fintype.card_pos (α := C)
  have hfactor : 0 ≤ (Fintype.card C : ℝ) - 1 := sub_nonneg.mpr hn
  have hpos := mul_nonneg hfactor (le_of_lt (Real.exp_pos (-mass r)))
  apply Real.log_le_log (by linarith)
  exact add_le_add (le_refl 1)
    (mul_le_mul_of_nonneg_left (Real.exp_le_exp.mpr (neg_le_neg hm)) hfactor)

/-- Every nonempty observed batch has the same CE function of total mass,
CM64. This includes repeated record values and a short final minibatch. The
identity explains why confidence optimization cannot select unseen decoders. -/
theorem observed_batch_ce {I : Type*} (U : Matrix S S ℂ) (hU : U ∈ Matrix.unitaryGroup S ℂ)
    (program : J → X → S) (order : List J) (samples : List (X × C)) {δ R : ℝ}
    {q : Parameters J S C} (hq : q ∈ domain program order samples δ R)
    (inputs : I → X) (labels : I → C) (batch : Finset I) (hne : batch.Nonempty)
    (hmem : ∀ i ∈ batch, (inputs i, labels i) ∈ samples) :
    batchLoss U program inputs labels batch q =
      Real.log (1 + ((Fintype.card C : ℝ) - 1) * Real.exp (-mass q)) := by
  have hn : (batch.card : ℝ) ≠ 0 := by
    exact_mod_cast (Finset.card_ne_zero.mpr hne)
  unfold batchLoss
  have hs := Finset.sum_congr rfl (fun i hi =>
    observed_ce U hU program order samples hq (inputs i, labels i) (hmem i hi))
  rw [hs]
  simp [hn]

end Transformer.Grokking.FlatSpectrum.ConvexDepth.BoundedLatent
