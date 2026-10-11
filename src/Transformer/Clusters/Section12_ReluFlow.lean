import Transformer.Clusters.Section12_ReluBounds

/-!
# A global clock-driven ODE for the ReLU test configuration

The four-token test of §12, `eq: mlp.transfo` in
arXiv:2305.05465v6 needs a genuine global trajectory. We construct
one by first solving an auxiliary autonomous equation on
`ℝ × (ℝ × ℝ)`. Its clock derivative is the fixed scale `w`, and
the scale derivative is zero. The moving coordinate is evaluated
at `min(y, -6w)` before applying ReLU. This explicit truncation
is solely an existence construction, not a changed attention
equation or an assumption that the original flow exists.

The truncated field is locally Lipschitz and has linear growth
bounded by `12 ‖state‖`, so the proved global existence theorem
applies. At the initial state `(0, -8, 1)`, the scale is one and
the clock is real time. The integrable bound from the preceding
module will keep the moving coordinate below minus six, so the
truncation is inactive on the resulting trajectory. That equality
will be proved before using the trajectory as a solution of the
source's actual feed-forward equation.
-/

open Real
namespace Transformer.Clusters

/-- An explicit auxiliary truncation for the global ODE construction
used to test §12, `eq: mlp.transfo`. It is later proved inactive
on the constructed trajectory. -/
noncomputable def reluClockCut (s : ReluClockState) : ReluClockState :=
  (s.1, (min s.2.1 (-6 * s.2.2), s.2.2))

/-- The auxiliary autonomous field for the four-token test of
§12, `eq: mlp.transfo`; the clock has derivative `w`, the scale
is fixed, and the moving drift is explicitly truncated. -/
noncomputable def reluClockField (s : ReluClockState) : ReluClockState :=
  (s.2.2, (max (reluClockDrift (reluClockCut s)) 0, 0))

/-- The existence truncation is locally Lipschitz.
Source: arXiv:2305.05465v6, §12, the auxiliary global existence
construction for the four-token test of `eq: mlp.transfo`. -/
theorem locallyLipschitz_reluClockCut : LocallyLipschitz reluClockCut := by
  have ht : LocallyLipschitz (fun s : ReluClockState => s.1) :=
    (by fun_prop : ContDiff ℝ 1 (fun s : ReluClockState => s.1)).locallyLipschitz
  have hy : LocallyLipschitz (fun s : ReluClockState => s.2.1) :=
    (by fun_prop : ContDiff ℝ 1 (fun s : ReluClockState => s.2.1)).locallyLipschitz
  have hw : LocallyLipschitz (fun s : ReluClockState => s.2.2) :=
    (by fun_prop : ContDiff ℝ 1 (fun s : ReluClockState => s.2.2)).locallyLipschitz
  have hb : LocallyLipschitz (fun s : ReluClockState => -6 * s.2.2) :=
    (by fun_prop : ContDiff ℝ 1 (fun s : ReluClockState => -6 * s.2.2)).locallyLipschitz
  exact ht.prodMk ((hy.min hb).prodMk hw)

/-- The ReLU clock field is locally Lipschitz despite the activation's kink.
Source: arXiv:2305.05465v6, §12, `eq: mlp.transfo`, the
auxiliary field used to construct its four-token trajectory. -/
theorem locallyLipschitz_reluClockField : LocallyLipschitz reluClockField := by
  have hw : LocallyLipschitz (fun s : ReluClockState => s.2.2) :=
    (by fun_prop : ContDiff ℝ 1 (fun s : ReluClockState => s.2.2)).locallyLipschitz
  have hd : LocallyLipschitz (fun s => reluClockDrift (reluClockCut s)) :=
    (contDiff_reluClockDrift.of_le (by simp)).locallyLipschitz.comp
      locallyLipschitz_reluClockCut
  exact hw.prodMk ((hd.max_const 0).prodMk (LocallyLipschitz.const 0))

/-- The truncated moving coordinate has a linear norm bound.
Source: arXiv:2305.05465v6, §12, the explicit existence
truncation for the four-token test of `eq: mlp.transfo`. -/
theorem norm_reluClockCut_y_le (s : ReluClockState) :
    ‖(reluClockCut s).2.1‖ ≤ 6 * ‖s‖ := by
  have hy : ‖s.2.1‖ ≤ ‖s‖ := (norm_fst_le s.2).trans (norm_snd_le s)
  have hw : ‖s.2.2‖ ≤ ‖s‖ := (norm_snd_le s.2).trans (norm_snd_le s)
  calc
    _ ≤ max ‖s.2.1‖ ‖-6 * s.2.2‖ := abs_min_le_max_abs_abs
    _ ≤ 6 * ‖s‖ := by
      apply max_le
      · nlinarith [norm_nonneg s]
      · rw [norm_mul]
        simpa using mul_le_mul_of_nonneg_left hw (by norm_num : (0 : ℝ) ≤ 6)

/-- The unchanged scale is bounded by the state norm.
Source: arXiv:2305.05465v6, §12, the auxiliary state
for its four-token test configuration. -/
theorem norm_reluClockCut_w_le (s : ReluClockState) :
    ‖(reluClockCut s).2.2‖ ≤ ‖s‖ :=
  (norm_snd_le s.2).trans (norm_snd_le s)

/-- ReLU cannot increase the norm of a scalar drift.
Source: arXiv:2305.05465v6, §12, the activation in `eq: mlp.transfo`. -/
theorem norm_relu_max_le (r : ℝ) : ‖max r 0‖ ≤ ‖r‖ := by
  simpa only [Real.norm_eq_abs, abs_zero, max_eq_left (abs_nonneg r)] using
    (abs_max_le_max_abs_abs (a := r) (b := (0 : ℝ)))

/-- The explicitly truncated attention drift has linear growth.
Source: arXiv:2305.05465v6, §12, `eq: mlp.transfo`;
the convex-average estimate bounds the auxiliary construction. -/
theorem norm_reluClockField_y_le (s : ReluClockState) :
    ‖(reluClockField s).2.1‖ ≤ 12 * ‖s‖ := by
  calc
    _ ≤ ‖reluClockDrift (reluClockCut s)‖ := norm_relu_max_le _
    _ ≤ 2 * max ‖(reluClockCut s).2.1‖ ‖(reluClockCut s).2.2‖ :=
      norm_reluFourDrift_le _ _ _ _
    _ ≤ 2 * (6 * ‖s‖) := by
      apply mul_le_mul_of_nonneg_left _ (by norm_num)
      exact max_le (norm_reluClockCut_y_le s)
        ((norm_reluClockCut_w_le s).trans (by nlinarith [norm_nonneg s]))
    _ = _ := by ring

/-- The full autonomous field has linear growth on its proper state space.
Source: arXiv:2305.05465v6, `p:wellposedparticles` and §12,
the global existence construction for the ReLU test trajectory. -/
theorem norm_reluClockField_le (s : ReluClockState) :
    ‖reluClockField s‖ ≤ 12 * ‖s‖ := by
  apply norm_prod_le_iff.mpr
  constructor
  · have hw : ‖s.2.2‖ ≤ ‖s‖ := (norm_snd_le s.2).trans (norm_snd_le s)
    change ‖s.2.2‖ ≤ _
    nlinarith [norm_nonneg s]
  · apply norm_prod_le_iff.mpr
    exact ⟨norm_reluClockField_y_le s, by simp [reluClockField]⟩

/-- The auxiliary origin is an equilibrium, as follows from linear growth.
Source: arXiv:2305.05465v6, §12, the autonomous existence
construction for the four-token ReLU test configuration. -/
theorem reluClockField_zero : reluClockField 0 = 0 := by
  apply norm_le_zero_iff.mp
  simpa using norm_reluClockField_le 0

/-- The auxiliary field has a concrete stationary solution.
Source: arXiv:2305.05465v6, §12, the existence construction;
this equilibrium does not supply the distinct-token counterexample. -/
example : ∀ t : ℝ,
    HasDerivAt (fun _ : ℝ => (0 : ReluClockState)) (reluClockField 0) t := by
  intro t
  rw [reluClockField_zero]
  exact hasDerivAt_const t 0

/-- The initial state for the distinct-token construction is nonzero.
Source: arXiv:2305.05465v6, §12, the four-token test configuration. -/
theorem reluClock_initial_ne_zero : (0, ((-8 : ℝ), (1 : ℝ))) ≠ (0 : ReluClockState) := by
  intro h
  have hw := congrArg (fun s : ReluClockState => s.2.2) h
  norm_num at hw

/-- A global auxiliary trajectory exists at the concrete initial state.
Source: arXiv:2305.05465v6, §12, `eq: mlp.transfo`;
proved existence precedes verification that its truncation is inactive. -/
theorem exists_reluClock_solution :
    ∃ γ : ℝ → ReluClockState, γ 0 = (0, (-8, 1)) ∧
      ∀ t : ℝ, HasDerivAt γ (reluClockField (γ t)) t := by
  exact GlobalFlow.exists_global locallyLipschitz_reluClockField
    (C := 12) (by norm_num) norm_reluClockField_le (0, (-8, 1))

end Transformer.Clusters
