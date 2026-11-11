module

public import Transformer.Grokking.FlatSpectrum.ConvexRecall.Loss

/-!
# Answer-only training determines the matching matrix

Source: `review/flat_spectrum.md`, *Convex answer-loss recall*, CR3.
Every stochastic matching matrix has at least the identity's ordinary
answer CE, sample by sample. If the training dictionaries separate each
queried key from every other key, equality of total losses forces the
identity. Consequently every global optimizer recalls all dictionaries,
including those absent from the training set. This is a theorem about the
actual answer loss on supplied inputs, not auxiliary matching supervision.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexRecall

open scoped BigOperators
open Landscape.Neural (crossEntropy)

variable {κ ν S : Type*} [Fintype κ] [Fintype ν] [Fintype S]
  [DecidableEq κ] [DecidableEq ν]

omit [Fintype ν] in
/-- Identity matching gives the ideal logits for the observed answer (CR3). -/
lemma logits_one_eq_target (q : κ) (D : κ → ν) :
    logits (1 : Matrix κ κ ℝ) q D = targetLogits (D q) := by
  funext c
  simp [logits_one, targetLogits, eq_comm]

/-- Every individual answer loss is minimized by correct matching (CR3). -/
theorem sample_loss_one_le {A : Matrix κ κ ℝ} (hA : IsStochastic A)
    (q : κ) (D : κ → ν) :
    crossEntropy (logits (1 : Matrix κ κ ℝ) q D) (D q) ≤
      crossEntropy (logits A q D) (D q) := by
  rw [logits_one_eq_target]
  exact target_crossEntropy_le _ (logits_nonneg hA q D) (logits_le_one hA q D (D q))

/-- The identity attains a global minimum of the ordinary answer CE for
every dataset. No existence of a trained optimum is assumed (CR3). -/
theorem answerLoss_one_le (query : S → κ) (data : S → κ → ν)
    {A : Matrix κ κ ℝ} (hA : IsStochastic A) :
    answerLoss query data 1 ≤ answerLoss query data A :=
  Finset.sum_le_sum fun s _ => sample_loss_one_le hA (query s) (data s)

/-- A global minimum on the declared convex parameter domain exists (CR3). -/
theorem isMinOn_answerLoss_one (query : S → κ) (data : S → κ → ν) :
    IsMinOn (answerLoss query data) {A : Matrix κ κ ℝ | IsStochastic A} 1 := by
  intro A hA
  exact answerLoss_one_le query data hA

/-- Attaining the global minimum forces unit mass on the correct value
at every observed sample (`review/flat_spectrum.md`, CR3). -/
theorem sample_target_mass_one [Nontrivial ν] (query : S → κ) (data : S → κ → ν)
    {A : Matrix κ κ ℝ} (hA : IsStochastic A)
    (hmin : answerLoss query data A ≤ answerLoss query data 1) (s : S) :
    logits A (query s) (data s) (data s (query s)) = 1 := by
  have hle : ∀ j ∈ (Finset.univ : Finset S),
      crossEntropy (logits (1 : Matrix κ κ ℝ) (query j) (data j)) (data j (query j)) ≤
        crossEntropy (logits A (query j) (data j)) (data j (query j)) :=
    fun j _ => sample_loss_one_le hA (query j) (data j)
  have he : answerLoss query data 1 = answerLoss query data A :=
    le_antisymm (answerLoss_one_le query data hA) hmin
  have hs := (Finset.sum_eq_sum_iff_of_le hle).mp he s (Finset.mem_univ s)
  rw [logits_one_eq_target] at hs
  exact target_mass_eq_one _ (logits_nonneg hA (query s) (data s))
    (logits_le_one hA (query s) (data s) _) hs.symm.le

/-- **Ordinary answer CE identifies the matching matrix.** If training
dictionaries distinguish each query from every wrong key, every feasible
matrix attaining the minimum is the identity (`review/flat_spectrum.md`, CR3).
Separation concerns only the input dictionaries; no learned correctness
condition or matching labels occur in its premise. -/
theorem eq_one_of_answerLoss_le [Nontrivial ν] (query : S → κ) (data : S → κ → ν)
    (hsep : Separates query data) {A : Matrix κ κ ℝ} (hA : IsStochastic A)
    (hmin : answerLoss query data A ≤ answerLoss query data 1) : A = 1 := by
  have hoff (q k : κ) (hk : k ≠ q) : A q k = 0 := by
    obtain ⟨s, hq, hdiff⟩ := hsep q k hk
    have hmass := sample_target_mass_one query data hA hmin s
    rw [hq] at hmass
    exact weight_eq_zero_of_target_mass hA q (data s) hmass k hdiff
  have hdiag (q : κ) : A q q = 1 := by
    have hs : ∑ k, A q k = A q q :=
      Finset.sum_eq_single q (fun k _ hk => hoff q k hk) (by simp)
    exact hs.symm.trans (hA.2 q)
  funext q k
  rw [Matrix.one_apply]
  by_cases h : q = k
  · subst k
    simpa using hdiag q
  · simpa [h] using hoff q k (Ne.symm h)

/-- **Every global answer-loss optimizer recalls every dictionary.** The
test dictionary is arbitrary and need not occur in training (CR3). -/
theorem trained_logits [Nontrivial ν] (query : S → κ) (data : S → κ → ν)
    (hsep : Separates query data) {A : Matrix κ κ ℝ} (hA : IsStochastic A)
    (hmin : IsMinOn (answerLoss query data) {B | IsStochastic B} A)
    (q : κ) (D : κ → ν) (c : ν) :
    logits A q D c = if D q = c then 1 else 0 := by
  rw [eq_one_of_answerLoss_le query data hsep hA (hmin stochastic_one)]
  exact logits_one q D c

/-- The true answer is the unique largest logit on every test dictionary,
so any argmax decoder returns it (`review/flat_spectrum.md`, CR3). -/
theorem trained_strict_argmax [Nontrivial ν] (query : S → κ) (data : S → κ → ν)
    (hsep : Separates query data) {A : Matrix κ κ ℝ} (hA : IsStochastic A)
    (hmin : IsMinOn (answerLoss query data) {B | IsStochastic B} A)
    (q : κ) (D : κ → ν) (c : ν) (hc : c ≠ D q) :
    logits A q D c < logits A q D (D q) := by
  rw [trained_logits query data hsep hA hmin, trained_logits query data hsep hA hmin]
  simp [Ne.symm hc]

/-- One two-key dictionary, queried at both keys, separates the keys.
Its labels are its actual values, one false and one true. -/
theorem separates_bool_id : Separates (id : Bool → Bool) (fun _ : Bool => (id : Bool → Bool)) := by
  intro q k hk
  exact ⟨q, rfl, hk⟩

example := sample_loss_one_le (stochastic_one (κ := Bool)) true (id : Bool → Bool)

example := answerLoss_one_le (id : Bool → Bool) (fun _ => id) (stochastic_one (κ := Bool))

example := sample_target_mass_one (id : Bool → Bool) (fun _ => id)
  (stochastic_one (κ := Bool)) le_rfl true

example := eq_one_of_answerLoss_le (id : Bool → Bool) (fun _ => id)
  separates_bool_id (stochastic_one (κ := Bool)) le_rfl

/-- All training hypotheses hold, while the tested dictionary reverses
both values and was never present in the two-sample training dataset. -/
example : logits (1 : Matrix Bool Bool ℝ) true Bool.not false = 1 :=
  trained_logits (id : Bool → Bool) (fun _ => id) separates_bool_id stochastic_one
    (isMinOn_answerLoss_one _ _) true Bool.not false

example : logits (1 : Matrix Bool Bool ℝ) true Bool.not true <
    logits (1 : Matrix Bool Bool ℝ) true Bool.not false :=
  trained_strict_argmax (id : Bool → Bool) (fun _ => id) separates_bool_id stochastic_one
    (isMinOn_answerLoss_one _ _) true Bool.not true (by decide)

end Transformer.Grokking.FlatSpectrum.ConvexRecall
