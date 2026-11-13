module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.BoxUpdate
public import Mathlib.Analysis.SpecialFunctions.Log.Deriv

/-!
# Ordinary cross-entropy directions and margin order

Source: `review/flat_spectrum.md`, CM33. Derive the full-vocabulary CE
direction from its exponential partition, then compare actual losses when
every target-versus-rival gap increases. Monotone clipping preserves target
and rival score order under any nonnegative CE step from a feasible row.
No fitted label or optimizer convergence conclusion is assumed.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth

open Landscape.Neural (crossEntropy sum_exp_pos)
open Geometry (probability probability_pos probability_le_one sum_probability)
open GradientEvidence (weightedExp weightedExp_pos crossEntropy_relative_change)

variable {C : Type*} [Fintype C] [DecidableEq C]

/-- The ordinary CE coordinate direction, with every original class
present in the denominator, CM33. Its derivative is proved below. -/
noncomputable def rowCEGradient (z : C → ℝ) (target : C) (c : C) : ℝ :=
  probability z c - if c = target then 1 else 0

/-- The usual softmax-minus-label vector is the actual derivative of CE
in every logit direction, CM33; it is not an assumed gradient oracle. -/
theorem rowCE_hasDerivAt (z v : C → ℝ) (target : C) :
    HasDerivAt (fun a : ℝ => crossEntropy (fun c => z c + a * v c) target)
      (∑ c, rowCEGradient z target c * v c) 0 := by
  have hz (c : C) : HasDerivAt (fun a : ℝ => z c + a * v c) (v c) 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).mul_const (v c)).const_add (z c)
  have he (c : C) : HasDerivAt (fun a : ℝ => Real.exp (z c + a * v c))
      (Real.exp (z c) * v c) 0 := by
    simpa using (hz c).exp
  have hs := HasDerivAt.fun_sum (u := Finset.univ) (fun c _ => he c)
  have hp : (∑ c, Real.exp (z c + (0 : ℝ) * v c)) ≠ 0 := by
    simpa using (sum_exp_pos z target).ne'
  have h := (hs.log hp).fun_sub (hz target)
  change HasDerivAt (fun a : ℝ => Real.log (∑ c, Real.exp (z c + a * v c)) -
    (z target + a * v target)) _ 0
  convert h using 1
  simp only [rowCEGradient, sub_mul, Finset.sum_sub_distrib, ite_mul, one_mul,
    zero_mul, Finset.sum_ite_eq', Finset.mem_univ, ite_true, add_zero]
  rw [Finset.sum_div]
  congr 1
  apply Finset.sum_congr rfl
  intro c hc
  simp only [probability]
  ring

/-- The target derivative is nonpositive, so subtracting it moves the
target score upwards before projection, CM33. -/
theorem rowCEGradient_target_nonpos (z : C → ℝ) (target : C) :
    rowCEGradient z target target ≤ 0 := by
  simp only [rowCEGradient, ite_true]
  exact sub_nonpos.mpr (probability_le_one z target)

/-- Every rival derivative is nonnegative, without assuming the target
currently wins, CM33. -/
theorem rowCEGradient_other_nonneg (z : C → ℝ) (target : C) {c : C}
    (hc : c ≠ target) : 0 ≤ rowCEGradient z target c := by
  simp only [rowCEGradient, ite_eq_right hc, sub_zero]
  exact (probability_pos z c).le

omit [DecidableEq C] in
/-- Nondecreasing target margins imply nonincreasing actual CE, CM33.
This compares the full competing-class partition, not a surrogate loss. -/
theorem rowCE_le_of_margins (z next : C → ℝ) (target : C)
    (hgap : ∀ c, z target - z c ≤ next target - next c) :
    crossEntropy next target ≤ crossEntropy z target := by
  have hsum : weightedExp z (fun c => (next c - next target) - (z c - z target)) ≤ 1 := by
    calc
      _ ≤ ∑ c, probability z c * 1 := by
        apply Finset.sum_le_sum
        intro c hc
        apply mul_le_mul_of_nonneg_left _ (probability_pos z c).le
        have hd : (next c - next target) - (z c - z target) ≤ 0 := by
          linarith [hgap c]
        simpa using Real.exp_le_exp.mpr hd
      _ = 1 := by simpa using sum_probability z target
  have hlog := Real.log_le_log (weightedExp_pos z _ target) hsum
  rw [Real.log_one] at hlog
  rw [← crossEntropy_relative_change z next target] at hlog
  linarith

example := rowCE_hasDerivAt (fun c : Bool => if c then (2 : ℝ) else -1)
  (fun c => if c then 3 else -2) true

example := rowCEGradient_target_nonpos (fun c : Bool => if c then (-1 : ℝ) else 2) true

example := rowCEGradient_other_nonneg (fun c : Bool => if c then (-1 : ℝ) else 2)
  true (c := false) (by decide)

example := rowCE_le_of_margins (fun c : Bool => if c then (2 : ℝ) else -1)
  (fun c => if c then 3 else -2) true (by intro c; cases c <;> norm_num)

/-- The coordinatewise projection of the derived ordinary-CE direction,
CM33. The effective rate includes the original minibatch normalization. -/
noncomputable def rowCEStep (A rate : ℝ) (z : C → ℝ) (target : C) : C → ℝ :=
  fun c => boxClip A (z c - rate * rowCEGradient z target c)

/-- Scalar clipping preserves order, CM33. Together with CM32's
fixed-point property this prevents projection from reversing a score move. -/
theorem boxClip_mono (A : ℝ) : Monotone (boxClip A) := by
  intro x y hxy
  exact max_le_max_left _ (min_le_min_left _ hxy)

/-- Every CE proposal is projected into the box, CM33. -/
theorem rowCEStep_bound {A : ℝ} (hA : 0 ≤ A) (rate : ℝ)
    (z : C → ℝ) (target c : C) : |rowCEStep A rate z target c| ≤ A := by
  exact boxClip_bound hA _

/-- A feasible target score cannot decrease under any nonnegative
ordinary-CE step, including a step that reaches the box boundary, CM33. -/
theorem rowCEStep_target_mono {A rate : ℝ} (hr : 0 ≤ rate) (z : C → ℝ)
    (target : C) (hz : |z target| ≤ A) : z target ≤ rowCEStep A rate z target target := by
  have hmove : z target ≤ z target - rate * rowCEGradient z target target := by
    have h := mul_nonpos_of_nonneg_of_nonpos hr (rowCEGradient_target_nonpos z target)
    linarith
  have h := boxClip_mono A hmove
  rw [boxClip_eq hz] at h
  exact h

/-- A feasible rival score cannot increase under the same update,
CM33. The hypothesis supplies its distinct class, not its desired outcome. -/
theorem rowCEStep_other_antimono {A rate : ℝ} (hr : 0 ≤ rate) (z : C → ℝ)
    (target : C) {c : C} (hc : c ≠ target) (hz : |z c| ≤ A) :
    rowCEStep A rate z target c ≤ z c := by
  have hmove : z c - rate * rowCEGradient z target c ≤ z c := by
    have h := mul_nonneg hr (rowCEGradient_other_nonneg z target hc)
    linarith
  have h := boxClip_mono A hmove
  rw [boxClip_eq hz] at h
  exact h

/-- Every target-versus-class gap is nondecreasing, CM33. The target's
own zero gap is included, which is needed for the complete CE partition. -/
theorem rowCEStep_margin {A rate : ℝ} (hr : 0 ≤ rate) (z : C → ℝ) (target : C)
    (hz : ∀ c, |z c| ≤ A) (c : C) :
    z target - z c ≤ rowCEStep A rate z target target - rowCEStep A rate z target c := by
  by_cases hc : c = target
  · subst c
    simp only [sub_self, le_refl]
  · have ht := rowCEStep_target_mono hr z target (hz target)
    have ho := rowCEStep_other_antimono hr z target hc (hz c)
    linarith

/-- The complete row CE never rises after this projected CE update,
CM33. This does not apply to arbitrary proposals or mixed-label collisions. -/
theorem rowCEStep_loss_le {A rate : ℝ} (hr : 0 ≤ rate) (z : C → ℝ) (target : C)
    (hz : ∀ c, |z c| ≤ A) : crossEntropy (rowCEStep A rate z target) target ≤
      crossEntropy z target := by
  exact rowCE_le_of_margins z _ target (rowCEStep_margin hr z target hz)

/-- A previously strict raw-label winner stays strict, CM33. -/
theorem rowCEStep_preserves_winner {A rate : ℝ} (hr : 0 ≤ rate) (z : C → ℝ)
    (target : C) (hz : ∀ c, |z c| ≤ A) {c : C} (hwin : z c < z target) :
    rowCEStep A rate z target c < rowCEStep A rate z target target := by
  have h := rowCEStep_margin hr z target hz c
  linarith

example := boxClip_mono 32 (show (24 : ℝ) ≤ 1000 by norm_num)

example := rowCEStep_bound (by norm_num : (0 : ℝ) ≤ 32) 1000
  (fun c : Bool => if c then 24 else 0) true false

example := rowCEStep_target_mono (by norm_num : (0 : ℝ) ≤ 1)
  (fun c : Bool => if c then 24 else 0) true (by norm_num : |(24 : ℝ)| ≤ 32)

example := rowCEStep_other_antimono (by norm_num : (0 : ℝ) ≤ 1)
  (fun c : Bool => if c then 24 else 0) true (c := false) (by decide)
  (by norm_num : |(0 : ℝ)| ≤ 32)

example := rowCEStep_margin (by norm_num : (0 : ℝ) ≤ 1)
  (fun c : Bool => if c then 24 else 0) true
  (by intro c; cases c <;> norm_num : ∀ c : Bool, |(if c then (24 : ℝ) else 0)| ≤ 32) false

example := rowCEStep_loss_le (by norm_num : (0 : ℝ) ≤ 1)
  (fun c : Bool => if c then 24 else 0) true
  (by intro c; cases c <;> norm_num : ∀ c : Bool, |(if c then (24 : ℝ) else 0)| ≤ 32)

example := rowCEStep_preserves_winner (by norm_num : (0 : ℝ) ≤ 1000)
  (fun c : Bool => if c then 24 else 0) true
  (by intro c; cases c <;> norm_num : ∀ c : Bool, |(if c then (24 : ℝ) else 0)| ≤ 32)
  (c := false) (by norm_num)

end Transformer.Grokking.FlatSpectrum.ConvexDepth
