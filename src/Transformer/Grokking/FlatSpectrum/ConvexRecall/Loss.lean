module

public import Transformer.Grokking.FlatSpectrum.ConvexRecall.Basic
public import Transformer.Grokking.FlatSpectrum.Training.Logits
public import Transformer.Grokking.NaiveLoss.Section4_LogitScaling

/-!
# Ordinary cross-entropy identifies maximal target mass

Source: `review/flat_spectrum.md`, *Convex answer-loss recall*, CR2-CR3;
XE1 for the usual log-sum-exp cross-entropy. The objective is convex in
all matching coordinates. Among nonnegative logits bounded by unit target
mass, the target's one-hot vector minimizes the loss. With at least two
classes, equality forces target mass one. No matching labels are used.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexRecall

open scoped BigOperators
open Landscape.Neural (crossEntropy sum_exp_pos)
open Geometry (probability)

variable {C : Type*} [Fintype C]

/-- The first-order inequality of XE1 implies convexity in the complete
logit vector (`review/flat_spectrum.md`, CR2). -/
theorem convex_crossEntropy (t : C) : ConvexOn ℝ Set.univ (fun z : C → ℝ => crossEntropy z t) := by
  refine ⟨convex_univ, ?_⟩
  intro x _ y _ a b ha hb hab
  let z : C → ℝ := a • x + b • y
  have hinc (c : C) : a * (x c - z c) + b * (y c - z c) = 0 := by
    calc
      _ = (a * x c + b * y c) - (a + b) * z c := by ring
      _ = 0 := by rw [hab]; simp [z]
  have hsum :
      a * (∑ c, probability z c * (x c - z c)) +
      b * (∑ c, probability z c * (y c - z c)) =
        ∑ c, probability z c * (a * (x c - z c) + b * (y c - z c)) := by
    rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
    apply Finset.sum_congr rfl
    intro c _
    ring
  have hzero :
      a * (∑ c, probability z c * (x c - z c) - (x t - z t)) +
      b * (∑ c, probability z c * (y c - z c) - (y t - z t)) = 0 := by
    calc
      _ = (a * (∑ c, probability z c * (x c - z c)) +
          b * (∑ c, probability z c * (y c - z c))) -
          (a * (x t - z t) + b * (y t - z t)) := by ring
      _ = 0 := by rw [hsum]; simp [hinc]
  have hx := mul_le_mul_of_nonneg_left (Training.crossEntropy_first_order z x t) ha
  have hy := mul_le_mul_of_nonneg_left (Training.crossEntropy_first_order z y t) hb
  have hce : a * crossEntropy z t + b * crossEntropy z t = crossEntropy z t := by
    rw [← add_mul, hab, one_mul]
  change crossEntropy z t ≤ a * crossEntropy x t + b * crossEntropy y t
  nlinarith only [hx, hy, hzero, hce]

variable [DecidableEq C]

/-- The ideal logits: unit mass on the correct value (CR3). -/
def targetLogits (t : C) (c : C) : ℝ := if c = t then 1 else 0

omit [DecidableEq C] in
/-- Ordinary CE written in relative logits, for comparing margins (CR3). -/
lemma crossEntropy_relative (z : C → ℝ) (t : C) :
    crossEntropy z t = Real.log (∑ c, Real.exp (z c - z t)) :=
  (NaiveLoss.crossEntropy_eq_standard z t).symm

omit [Fintype C] in
/-- Every relative competitor logit is at least the ideal value when
logits are nonnegative and the target logit is at most one (CR3). -/
lemma target_relative_le {z : C → ℝ} (t : C) (hz : ∀ c, 0 ≤ z c) (ht : z t ≤ 1) (c : C) :
    targetLogits t c - targetLogits t t ≤ z c - z t := by
  by_cases h : c = t
  · subst c
    simp
  · simp only [targetLogits, h, ↓reduceIte]
    linarith [hz c]

/-- The ideal target logits minimize ordinary CE over the probability
simplex; the stated weaker bounds suffice (CR3). -/
theorem target_crossEntropy_le {z : C → ℝ} (t : C) (hz : ∀ c, 0 ≤ z c) (ht : z t ≤ 1) :
    crossEntropy (targetLogits t) t ≤ crossEntropy z t := by
  rw [crossEntropy_relative, crossEntropy_relative]
  apply Real.log_le_log (sum_exp_pos _ t)
  exact Finset.sum_le_sum fun c _ => Real.exp_le_exp.mpr (target_relative_le t hz ht c)

/-- With another class present, a target mass below one has strictly
larger ordinary CE than the ideal logits (CR3). -/
theorem target_crossEntropy_lt [Nontrivial C] {z : C → ℝ} (t : C)
    (hz : ∀ c, 0 ≤ z c) (ht : z t < 1) :
    crossEntropy (targetLogits t) t < crossEntropy z t := by
  obtain ⟨c, hc⟩ := exists_ne t
  rw [crossEntropy_relative, crossEntropy_relative]
  apply Real.log_lt_log (sum_exp_pos _ t)
  refine Finset.sum_lt_sum
    (fun d _ => Real.exp_le_exp.mpr (target_relative_le t hz ht.le d)) ?_
  refine ⟨c, Finset.mem_univ c, Real.exp_lt_exp.mpr ?_⟩
  simp only [targetLogits, hc, ↓reduceIte]
  linarith [hz c]

/-- Attaining the ideal CE forces target mass one, rather than merely
assuming correct training predictions (CR3). -/
theorem target_mass_eq_one [Nontrivial C] {z : C → ℝ} (t : C)
    (hz : ∀ c, 0 ≤ z c) (ht : z t ≤ 1)
    (hce : crossEntropy z t ≤ crossEntropy (targetLogits t) t) : z t = 1 := by
  by_contra h
  exact (not_lt_of_ge hce) (target_crossEntropy_lt t hz (lt_of_le_of_ne ht h))

variable {κ S : Type*} [Fintype κ] [Fintype S]

/-- Ordinary answer CE is jointly convex on the whole unrestricted
matching-matrix space, hence on its stochastic domain (CR2). -/
theorem answerLoss_convex (query : S → κ) (data : S → κ → C) :
    ConvexOn ℝ Set.univ (answerLoss query data) := by
  refine ⟨convex_univ, ?_⟩
  intro A _ B _ a b ha hb hab
  change (∑ s, crossEntropy (logits (a • A + b • B) (query s) (data s)) (data s (query s))) ≤ _
  simp only [answerLoss, smul_eq_mul, Finset.mul_sum, ← Finset.sum_add_distrib]
  apply Finset.sum_le_sum
  intro s _
  rw [logits_add, logits_smul, logits_smul]
  exact (convex_crossEntropy (data s (query s))).2 (Set.mem_univ _) (Set.mem_univ _) ha hb hab

/-- The identity's correct one-hot logits inhabit every loss hypothesis. -/
example : (∀ c : Bool, 0 ≤ targetLogits true c) ∧ targetLogits true true = 1 := by
  constructor
  · intro c
    cases c <;> norm_num [targetLogits]
  · simp [targetLogits]

example := target_crossEntropy_le (z := targetLogits true) true
  (fun c => by cases c <;> norm_num [targetLogits]) (by simp [targetLogits])

/-- A uniform matching row has a strictly worse loss on a distinguishing
two-value dictionary. -/
example : crossEntropy (targetLogits true) true < crossEntropy (fun _ : Bool => 1 / 2) true :=
  target_crossEntropy_lt true (fun _ => by norm_num) (by norm_num)

example := target_mass_eq_one (z := targetLogits true) true
  (fun c => by cases c <;> norm_num [targetLogits]) (by simp [targetLogits]) le_rfl

example := answerLoss_convex (S := Bool) (κ := Bool) (fun q => q) (fun _ => id)

end Transformer.Grokking.FlatSpectrum.ConvexRecall
