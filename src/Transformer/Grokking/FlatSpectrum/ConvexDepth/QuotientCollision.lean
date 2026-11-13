module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.JointHead

/-!
# Ordinary cross-entropy cannot resolve a shared-feature collision

Source: `review/flat_spectrum.md`, CM31. The quotient head at `49105730`
uses the same logit vector for every input with the same learned code. If
two observed raw labels in that fiber differ, their mean CE is at least
log2, regardless of the head budget or optimizer. A decoded quotient answer
can still be correct; CM30 does not imply raw-label interpolation.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth

open scoped BigOperators
open Landscape.Neural (crossEntropy)

variable {C X Z : Type*} [Fintype C]

/-- Two distinct raw labels sharing one logit vector incur at least twice
log2 in ordinary full-vocabulary CE, CM31. No optimizer is assumed. -/
theorem shared_logits_ce_lower (v : C → ℝ) {c d : C} (hcd : c ≠ d) :
    2 * Real.log 2 ≤ crossEntropy v c + crossEntropy v d := by
  classical
  let S := ∑ k, Real.exp (v k)
  have hs : Real.exp (v c) + Real.exp (v d) ≤ S := by
    have h := Finset.sum_le_sum_of_subset_of_nonneg
      (f := fun k => Real.exp (v k))
      (Finset.subset_univ ({c, d} : Finset C))
      (fun k _ _ => (Real.exp_pos (v k)).le)
    simpa [Finset.sum_pair hcd, S] using h
  have hS : 0 < S := Landscape.Neural.sum_exp_pos v c
  have hp : 4 * Real.exp (v c + v d) ≤ S ^ 2 := by
    rw [Real.exp_add]
    nlinarith [sq_nonneg (Real.exp (v c) - Real.exp (v d)),
      Real.exp_pos (v c), Real.exp_pos (v d)]
  have hlog := Real.log_le_log (by positivity : 0 < 4 * Real.exp (v c + v d)) hp
  rw [Real.log_mul (by norm_num) (ne_of_gt (Real.exp_pos _)), Real.log_exp,
    Real.log_pow] at hlog
  have hfour : Real.log (4 : ℝ) = 2 * Real.log 2 := by
    rw [show (4 : ℝ) = 2 ^ 2 by norm_num, Real.log_pow]
    norm_num
  rw [hfour] at hlog
  norm_num at hlog
  dsimp [crossEntropy]
  change 2 * Real.log 2 ≤ (Real.log S - v c) + (Real.log S - v d)
  linarith

/-- Any predictor factoring through a colliding latent feature has the
same quantitative training obstruction, CM31. -/
theorem latent_collision_mean_ce (latent : X → Z) (head : Z → C → ℝ)
    (labels : X → C) {x y : X} (hxy : latent x = latent y)
    (hlabels : labels x ≠ labels y) :
    Real.log 2 ≤
      (crossEntropy (head (latent x)) (labels x) +
        crossEntropy (head (latent y)) (labels y)) / 2 := by
  rw [← hxy]
  have h := shared_logits_ce_lower (head (latent x)) hlabels
  linarith

/-- A convex combined head does not repair identical fixed features by
training longer: the same loss floor holds for every coefficient tensor. -/
theorem feature_collision_mean_ce {K : Type*} [Fintype K]
    (features : K → X → ℝ) (W : K → C → ℝ) (labels : X → C)
    {x y : X} (hxy : ∀ k, features k x = features k y)
    (hlabels : labels x ≠ labels y) :
    Real.log 2 ≤
      (crossEntropy (headLogits features W x) (labels x) +
        crossEntropy (headLogits features W y) (labels y)) / 2 := by
  have heq : headLogits features W x = headLogits features W y := by
    funext c
    apply Finset.sum_congr rfl
    intro k hk
    rw [hxy k]
  rw [← heq]
  have h := shared_logits_ce_lower (headLogits features W x) hlabels
  linarith

/-- A collision contributes a nonzero floor to the actual empirical sum,
even when every other example can fit. CM31 retains their nonnegative CE. -/
theorem headLoss_collision_lower {K S : Type*} [Fintype K] [Fintype S]
    (features : K → X → ℝ) (W : K → C → ℝ) (inputs : S → X) (labels : S → C)
    {s t : S} (hfeatures : ∀ k, features k (inputs s) = features k (inputs t))
    (hlabels : labels s ≠ labels t) :
    2 * Real.log 2 ≤ headLoss features inputs labels W := by
  classical
  have hst : s ≠ t := fun h => hlabels (congrArg labels h)
  have hpair := feature_collision_mean_ce
    (fun k i => features k (inputs i)) W labels hfeatures hlabels
  have hs := Finset.sum_le_sum_of_subset_of_nonneg
    (f := fun i => crossEntropy (headLogits features W (inputs i)) (labels i))
    (Finset.subset_univ ({s, t} : Finset S))
    (fun i _ _ => Landscape.Neural.crossEntropy_nonneg _ _)
  simp only [Finset.sum_pair hst] at hs
  change 2 * Real.log 2 ≤ ∑ i, crossEntropy (headLogits features W (inputs i)) (labels i)
  change Real.log 2 ≤
    (crossEntropy (headLogits features W (inputs s)) (labels s) +
      crossEntropy (headLogits features W (inputs t)) (labels t)) / 2 at hpair
  linarith

/-- The collision floor for the mean ordinary loss scales with the number
of observed samples; it does not vanish by increasing the head budget. -/
theorem headLoss_collision_mean {K S : Type*} [Fintype K] [Fintype S]
    (features : K → X → ℝ) (W : K → C → ℝ) (inputs : S → X) (labels : S → C)
    {s t : S} (hfeatures : ∀ k, features k (inputs s) = features k (inputs t))
    (hlabels : labels s ≠ labels t) :
    (2 * Real.log 2) / Fintype.card S ≤
      headLoss features inputs labels W / Fintype.card S := by
  exact div_le_div_of_nonneg_right
    (headLoss_collision_lower features W inputs labels hfeatures hlabels) (Nat.cast_nonneg _)

/-- The two-class collision bound is sharp: a shared constant vector gives
exactly log2 per example, CM31. Increasing its common offset cannot help. -/
theorem shared_logits_lower_sharp (a : ℝ) :
    crossEntropy (fun _ : Bool => a) false +
      crossEntropy (fun _ : Bool => a) true = 2 * Real.log 2 := by
  simp only [crossEntropy, Fintype.sum_bool]
  rw [show Real.exp a + Real.exp a = 2 * Real.exp a by ring,
    Real.log_mul (by norm_num) (ne_of_gt (Real.exp_pos a)), Real.log_exp]
  ring

omit [Fintype C] in
/-- A deterministic decoder of a shared code cannot get both distinct raw
labels right. This is independent of CE and of any flatness certificate. -/
theorem latent_collision_no_exact (latent : X → Z) (prediction : Z → C) (labels : X → C)
    {x y : X} (hxy : latent x = latent y) (hlabels : labels x ≠ labels y) :
    ¬ (prediction (latent x) = labels x ∧ prediction (latent y) = labels y) := by
  rintro ⟨hx, hy⟩
  apply hlabels
  calc
    labels x = prediction (latent x) := hx.symm
    _ = prediction (latent y) := congrArg prediction hxy
    _ = labels y := hy

example := shared_logits_ce_lower (fun b : Bool => if b then 1 else -1)
  (c := false) (d := true) (by decide)

example := latent_collision_mean_ce (fun _ : Bool => ())
  (fun (_ : Unit) (c : Bool) => if c then 1 else -1) id
  (x := false) (y := true) rfl (by decide)

example := feature_collision_mean_ce (fun (_ : Unit) (_ : Bool) => (1 : ℝ))
  (fun (_ : Unit) (c : Bool) => if c then 1 else -1) id
  (x := false) (y := true) (fun _ => rfl) (by decide)

example := headLoss_collision_lower (fun (_ : Unit) (_ : Bool) => (1 : ℝ))
  (fun (_ : Unit) (c : Bool) => if c then 1 else -1) id id
  (s := false) (t := true) (fun _ => rfl) (by decide)

example := headLoss_collision_mean (fun (_ : Unit) (_ : Bool) => (1 : ℝ))
  (fun (_ : Unit) (c : Bool) => if c then 1 else -1) id id
  (s := false) (t := true) (fun _ => rfl) (by decide)

example := latent_collision_no_exact (fun _ : Bool => ()) (fun _ : Unit => false) id
  (x := false) (y := true) rfl (by decide)

end Transformer.Grokking.FlatSpectrum.ConvexDepth
