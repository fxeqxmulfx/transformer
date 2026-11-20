module

public import Transformer.RoPE.LeReRoPE.Window
public import Mathlib.Analysis.Calculus.Deriv.Comp
public import Mathlib.Analysis.Calculus.Deriv.Prod
public import Mathlib.Analysis.Calculus.FDeriv.Linear

/-!
# LeReRoPE: the gradients of the heads' windows and slopes

In no paper. Each head `a` learns its window `w_a` and slope `s_a`
(`Transformer.RoPE.LeReRoPE.windowPosition`). A pair `i` of the head `h i`, `d_i` apart, sits at
`p_i = min(d_i, w_{h i}) + s_{h i}·max(d_i − w_{h i}, 0)`. For every loss `F` of the pairs'
positions, differentiable there with the derivative `F'`:

* `hasDerivAt_slope_loss`: `∂F/∂s_a = F'(e)`, `e_i = max(d_i − w_a, 0)` on the head's pairs and
  `0` on the others. Each far pair moves by its own excess over the window;
* `hasDerivAt_window_loss`: `∂F/∂w_a = (1 − s_a) F'(u)`, `u` the indicator of the head's pairs
  beyond the window, if no pair sits at the window itself. All far pairs move by one amount,
  whatever their distances: the window's gradient sees which pairs are far, not how far;
* `hasDerivAt_loss_of_window_open`: a window beyond every distance of its head gets no gradient,
  and neither does its slope. Once a window has opened past a batch's distances, no gradient of
  that batch closes it again;
* `hasDerivAt_window_loss_of_slope_one`: at `s_a = 1` the head is RoPE, and its window gets no
  gradient at all.
-/

@[expose] public section

namespace Transformer.RoPE.LeReRoPE

variable {ι H : Type*}

/-- The positions of the pairs `i`, `d i` apart in the heads `h i`, under the heads' windows `w`
and slopes `s`. -/
noncomputable def headPositions (h : ι → H) (d : ι → ℝ) (w s : H → ℝ) : ι → ℝ :=
  fun i => windowPosition (w (h i)) (s (h i)) (d i)

variable [Fintype ι] [DecidableEq H]

/-- **The slope's gradient**: `∂F/∂s_a = F'(e)`, with `e_i = max(d_i − w_a, 0)` on the head's
pairs and `0` on the others. -/
theorem hasDerivAt_slope_loss (h : ι → H) (d : ι → ℝ) (w s : H → ℝ) (a : H)
    {F : (ι → ℝ) → ℝ} {F' : (ι → ℝ) →L[ℝ] ℝ} (hF : HasFDerivAt F F' (headPositions h d w s)) :
    HasDerivAt (fun σ => F (headPositions h d w (Function.update s a σ)))
      (F' fun i => if h i = a then max (d i - w a) 0 else 0) (s a) := by
  have hx : HasDerivAt (fun σ => headPositions h d w (Function.update s a σ))
      (fun i => if h i = a then max (d i - w a) 0 else 0) (s a) := by
    refine hasDerivAt_pi.2 fun i => ?_
    by_cases hi : h i = a
    · rw [ite_eq_left hi]
      simp only [headPositions, hi, Function.update_self]
      exact hasDerivAt_windowPosition_slope (w a) (d i) (s a)
    · rw [ite_eq_right hi]
      simp only [headPositions, Function.update_of_ne hi]
      exact hasDerivAt_const _ _
  have hF' : HasFDerivAt F F' (headPositions h d w (Function.update s a (s a))) := by
    rwa [Function.update_eq_self]
  exact hF'.comp_hasDerivAt (s a) hx

/-- The loss `p₀ − p₁` of branch amplitudes' two inputs, `32` and `64` back in one head with the
window `16`, satisfies the hypothesis of `hasDerivAt_slope_loss`. -/
example (s : ℝ) :=
  hasDerivAt_slope_loss (fun _ : Fin 2 => ()) ![32, 64] (fun _ => 16) (fun _ => s) ()
    (ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : Fin 2 => ℝ) 0 -
      ContinuousLinearMap.proj 1).hasFDerivAt

/-- **The window's gradient**: if no pair of the head `a` sits at its window,
`∂F/∂w_a = (1 − s_a) F'(u)`, with `u_i = 1` on the head's pairs beyond the window and `0` on the
others. -/
theorem hasDerivAt_window_loss (h : ι → H) (d : ι → ℝ) (w s : H → ℝ) (a : H)
    (hd : ∀ i, h i = a → d i ≠ w a) {F : (ι → ℝ) → ℝ} {F' : (ι → ℝ) →L[ℝ] ℝ}
    (hF : HasFDerivAt F F' (headPositions h d w s)) :
    HasDerivAt (fun ω => F (headPositions h d (Function.update w a ω) s))
      ((1 - s a) * F' fun i => if h i = a ∧ w a < d i then 1 else 0) (w a) := by
  have hx : HasDerivAt (fun ω => headPositions h d (Function.update w a ω) s)
      ((1 - s a) • fun i => if h i = a ∧ w a < d i then (1 : ℝ) else 0) (w a) := by
    refine hasDerivAt_pi.2 fun i => ?_
    simp only [Pi.smul_apply, smul_eq_mul]
    by_cases hi : h i = a
    · rcases lt_or_gt_of_ne (hd i hi) with hlt | hgt
      · rw [ite_eq_right fun hc => lt_asymm hlt hc.2, mul_zero]
        simp only [headPositions, hi, Function.update_self]
        exact hasDerivAt_windowPosition_window_of_gt (s a) hlt
      · rw [ite_eq_left ⟨hi, hgt⟩, mul_one]
        simp only [headPositions, hi, Function.update_self]
        exact hasDerivAt_windowPosition_window_of_lt (s a) hgt
    · rw [ite_eq_right fun hc => hi hc.1, mul_zero]
      simp only [headPositions, Function.update_of_ne hi]
      exact hasDerivAt_const _ _
  have hF' : HasFDerivAt F F' (headPositions h d (Function.update w a (w a)) s) := by
    rwa [Function.update_eq_self]
  have h' := hF'.comp_hasDerivAt (w a) hx
  rwa [map_smul, smul_eq_mul] at h'

/-- The same loss satisfies the hypotheses of `hasDerivAt_window_loss`: neither input sits at the
window `16`. -/
example (s : ℝ) :=
  hasDerivAt_window_loss (fun _ : Fin 2 => ()) ![32, 64] (fun _ => 16) (fun _ => s) ()
    (fun i _ => by fin_cases i <;> norm_num)
    (ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : Fin 2 => ℝ) 0 -
      ContinuousLinearMap.proj 1).hasFDerivAt

/-- **An open window stays open**: if every pair of the head `a` is closer than its window, the
loss's derivatives in the window and in the slope are both `0`. -/
theorem hasDerivAt_loss_of_window_open (h : ι → H) (d : ι → ℝ) (w s : H → ℝ) (a : H)
    (hopen : ∀ i, h i = a → d i < w a) {F : (ι → ℝ) → ℝ} {F' : (ι → ℝ) →L[ℝ] ℝ}
    (hF : HasFDerivAt F F' (headPositions h d w s)) :
    HasDerivAt (fun ω => F (headPositions h d (Function.update w a ω) s)) 0 (w a) ∧
      HasDerivAt (fun σ => F (headPositions h d w (Function.update s a σ))) 0 (s a) := by
  constructor
  · convert hasDerivAt_window_loss h d w s a (fun i hi => (hopen i hi).ne) hF using 1
    have hu : (fun i => if h i = a ∧ w a < d i then (1 : ℝ) else 0) = 0 := by
      funext i
      rw [ite_eq_right fun hc => lt_asymm (hopen i hc.1) hc.2]
      rfl
    rw [hu, map_zero, mul_zero]
  · convert hasDerivAt_slope_loss h d w s a hF using 1
    have he : (fun i => if h i = a then max (d i - w a) 0 else 0) = 0 := by
      funext i
      split_ifs with hi
      · exact max_eq_right (sub_nonpos.2 (hopen i hi).le)
      · rfl
    rw [he, map_zero]

/-- With the window `100`, both inputs of branch amplitudes are inside it. -/
example (s : ℝ) :=
  hasDerivAt_loss_of_window_open (fun _ : Fin 2 => ()) ![32, 64] (fun _ => 100) (fun _ => s) ()
    (fun i _ => by fin_cases i <;> norm_num)
    (ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : Fin 2 => ℝ) 0 -
      ContinuousLinearMap.proj 1).hasFDerivAt

omit [Fintype ι] in
/-- **At the slope `1` the window is invisible**: the head is RoPE, the loss does not depend on
its window, and the window's derivative is `0`. -/
theorem hasDerivAt_window_loss_of_slope_one (h : ι → H) (d : ι → ℝ) (w s : H → ℝ) (a : H)
    (hs : s a = 1) (F : (ι → ℝ) → ℝ) :
    HasDerivAt (fun ω => F (headPositions h d (Function.update w a ω) s)) 0 (w a) := by
  have hc : (fun ω => F (headPositions h d (Function.update w a ω) s)) =
      fun _ => F (headPositions h d w s) := by
    funext ω
    congr 1
    funext i
    by_cases hi : h i = a
    · simp only [headPositions, hi, hs, Function.update_self, windowPosition_one]
    · simp only [headPositions, Function.update_of_ne hi]
  rw [hc]
  exact hasDerivAt_const _ _

example (F : (Fin 2 → ℝ) → ℝ) :=
  hasDerivAt_window_loss_of_slope_one (fun _ : Fin 2 => ()) ![32, 64] (fun _ => 16) (fun _ => 1)
    () rfl F

end Transformer.RoPE.LeReRoPE
