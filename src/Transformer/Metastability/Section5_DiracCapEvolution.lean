/-
# Evolution of a cap under a stationary Dirac law

ArXiv:2410.06833v1, §5, Step 2 of the proof of `thm: metastability MF`
and `claim: de sortie de cap`. The characteristic flow for `μ(t) = δ_w`
is independent of `β`, since normalization cancels its exponential kernel.

For a boundary point `b` with `⟨b,w⟩ = 1-ε`, the entire transported cap
has minimum `η(t) = C(t,1-ε)`. Its variance around that boundary trajectory
is exactly `V(t) = 1-η(t)`. The longitudinal coordinate increases for
nonnegative time, so the whole geometric cap stays in its original cap,
and hence in the double cap used to define `escapeWindow`.

These are auxiliary identities for a counterexample to the claimed
upper bound on `T_*`. They concern the actual global characteristic flow,
including points outside the support of the stationary measure.
-/

import Transformer.Metastability.Section5_DiracCaps

open Real MeasureTheory

namespace Transformer.Metastability

open Perspective

/-- Exact longitudinal deficit for the Dirac flow of `eq: flow.map`.

Source: arXiv:2410.06833v1, §5, auxiliary to `claim: de sortie de cap`. -/
theorem one_sub_diracAlong (t c : ℝ) (hc : c ∈ Set.Icc (-1 : ℝ) 1) :
    1 - diracAlong t c = 2 * (1 - c) / diracDen t c := by
  have hd := ne_of_gt (diracDen_pos t c hc)
  unfold diracAlong
  field_simp [hd]
  unfold diracDen
  ring

/-- The Dirac characteristic moves toward its stationary center in forward time.

Source: arXiv:2410.06833v1, §5, auxiliary to `claim: de sortie de cap`. -/
theorem diracAlong_forward_bounds (t c : ℝ)
    (hc : c ∈ Set.Icc (-1 : ℝ) 1) (ht : 0 ≤ t) :
    c ≤ diracAlong t c ∧ diracAlong t c ≤ 1 := by
  have hd := diracDen_pos t c hc
  have he : 1 ≤ Real.exp (2 * t) := by
    simpa using (Real.exp_le_exp.mpr (show (0 : ℝ) ≤ 2 * t by linarith))
  have hcprod : 0 ≤ (1 + c) * (1 - c) := mul_nonneg (by linarith [hc.1])
    (by linarith [hc.2])
  constructor
  · rw [diracAlong, le_div_iff₀ hd]
    dsimp [diracDen]
    nlinarith [mul_nonneg hcprod (sub_nonneg.mpr he)]
  · rw [diracAlong, div_le_iff₀ hd]
    dsimp [diracDen]
    linarith [hc.2]

/-- An exponential upper bound on the exact deficit, sufficient to prove
nonemptiness of the stopping set without assuming the cap-exit claim.

Source: arXiv:2410.06833v1, §5, `claim: de sortie de cap`. -/
theorem one_sub_diracAlong_upper_exp (t c : ℝ) (hc : c ∈ Set.Icc (0 : ℝ) 1) :
    1 - diracAlong t c ≤ 2 * (1 - c) * Real.exp (-2 * t) := by
  have hci : c ∈ Set.Icc (-1 : ℝ) 1 := ⟨by linarith [hc.1], hc.2⟩
  have he := Real.exp_pos (2 * t)
  have hD : Real.exp (2 * t) ≤ diracDen t c := by
    dsimp [diracDen]
    nlinarith [mul_nonneg hc.1 he.le, hc.2]
  rw [one_sub_diracAlong t c hci]
  calc
    2 * (1 - c) / diracDen t c ≤ 2 * (1 - c) / Real.exp (2 * t) :=
      div_le_div_of_nonneg_left (by linarith [hc.2]) he hD
    _ = 2 * (1 - c) * Real.exp (-2 * t) := by
      rw [show -2 * t = -(2 * t) by ring, Real.exp_neg, div_eq_mul_inv]

/-- The minimum of the whole transported cap follows its boundary trajectory.

Source: arXiv:2410.06833v1, §5, Step 1, definition of `η_q`. -/
theorem capMin_diracFlow (d : ℕ) (w b : SSphere d) (ε t : ℝ)
    (hb : inner (𝕜 := ℝ) (b : EucSpace d) (w : EucSpace d) = 1 - ε) :
    capMin d (diracFlow d w) w ε t = diracAlong t (1 - ε) := by
  rw [capMin_eq_of_isCapArgmin d _ w ε _ (isCapArgmin_diracFlow d w b ε hb) t,
    inner_diracFlow, hb]

/-- The source's variance about a cap minimizer equals the longitudinal deficit
for a Dirac law at its center, at every real time.

Source: arXiv:2410.06833v1, §5, Step 2, definition of `V_q`. -/
theorem capVariance_diracFlow (d : ℕ) (w b : SSphere d) (ε t : ℝ) (hε : 0 ≤ ε)
    (hb : inner (𝕜 := ℝ) (b : EucSpace d) (w : EucSpace d) = 1 - ε) :
    capVariance d (diracProb d w) (diracFlow d w) w ε
      (fun s => diracFlow d w s b) t = 1 - diracAlong t (1 - ε) := by
  classical
  have hw : w ∈ sphericalCap d w (2 * ε) := by
    change 1 - 2 * ε ≤ inner (𝕜 := ℝ) (w : EucSpace d) (w : EucSpace d)
    rw [real_inner_self_eq_norm_mul_norm, mem_sphere_zero_iff_norm.mp w.2]
    linarith
  have hμ : (diracProb d w : Measure (SSphere d)) = Measure.dirac w := rfl
  rw [capVariance, hμ, restrict_dirac, ite_eq_left hw, integral_dirac,
    diracFlow_self, norm_sub_sq_real,
    mem_sphere_zero_iff_norm.mp w.2,
    mem_sphere_zero_iff_norm.mp (diracFlow d w t b).2,
    real_inner_comm (diracFlow d w t b : EucSpace d) (w : EucSpace d),
    inner_diracFlow, hb]
  ring

/-- The entire geometric cap remains in the escape window at every nonnegative time.
The stationary measure's smaller support is not substituted for the cap.

Source: arXiv:2410.06833v1, §5, Step 1, definition of `T_esc`. -/
theorem mem_escapeWindow_diracFlow (d k : ℕ) (w : SSphere d) (ε t : ℝ)
    (hε : 0 ≤ ε) (ht : 0 ≤ t) :
    t ∈ escapeWindow d k (fun _ => w) (diracFlow d w) ε := by
  refine ⟨ht, ?_⟩
  intro s hs q z hz
  rcases hz with ⟨z', hz', rfl⟩
  change 1 - 2 * ε ≤ inner (𝕜 := ℝ) (diracFlow d w s z' : EucSpace d) (w : EucSpace d)
  rw [inner_diracFlow]
  have hc := real_inner_mem_Icc_of_norm_eq_one (mem_sphere_zero_iff_norm.mp z'.2)
    (mem_sphere_zero_iff_norm.mp w.2)
  have hforward := (diracAlong_forward_bounds s _ hc hs.1).1
  change 1 - ε ≤ inner (𝕜 := ℝ) (z' : EucSpace d) (w : EucSpace d) at hz'
  linarith

/-- Nonpolar coordinates and positive time witness both coefficient bounds. -/
example : (0 : ℝ) ∈ Set.Icc (-1 : ℝ) 1 ∧ (0 : ℝ) ≤ 1 ∧
    (0 : ℝ) ∈ Set.Icc (0 : ℝ) 1 ∧
    0 ≤ diracAlong 1 0 ∧ diracAlong 1 0 ≤ 1 ∧
    1 - diracAlong 1 0 ≤ 2 * (1 - 0) * Real.exp (-2 * 1) :=
  ⟨by norm_num, by norm_num, by norm_num,
    (diracAlong_forward_bounds 1 0 (by norm_num) (by norm_num)).1,
    (diracAlong_forward_bounds 1 0 (by norm_num) (by norm_num)).2,
    one_sub_diracAlong_upper_exp 1 0 (by norm_num)⟩

/-- The deficit identity has an admissible coordinate away from both poles. -/
example : (0 : ℝ) ∈ Set.Icc (-1 : ℝ) 1 ∧
    1 - diracAlong 1 0 = 2 * (1 - 0) / diracDen 1 0 :=
  ⟨by norm_num, one_sub_diracAlong 1 0 (by norm_num)⟩

/-- The circle boundary meets the cap-minimum and variance hypotheses at positive time. -/
example : (0 : ℝ) ≤ 1 / 10000 ∧
    inner (𝕜 := ℝ) (varianceCounterBoundary : EucSpace 2)
      (varianceCounterCentre : EucSpace 2) = 1 - (1 / 10000 : ℝ) ∧
    capVariance 2 (diracProb 2 varianceCounterCentre) (diracFlow 2 varianceCounterCentre)
      varianceCounterCentre (1 / 10000)
      (fun s => diracFlow 2 varianceCounterCentre s varianceCounterBoundary) 1 =
      1 - diracAlong 1 (1 - 1 / 10000) :=
  ⟨by norm_num, varianceCounterBoundary_coordinate,
    capVariance_diracFlow 2 _ _ _ 1 (by norm_num) varianceCounterBoundary_coordinate⟩

/-- The escape-window hypotheses are attained at positive time. -/
example : (0 : ℝ) ≤ 1 / 10000 ∧ (0 : ℝ) ≤ 1 ∧
    (1 : ℝ) ∈ escapeWindow 2 1 (fun _ => varianceCounterCentre)
      (diracFlow 2 varianceCounterCentre) (1 / 10000) :=
  ⟨by norm_num, by norm_num, mem_escapeWindow_diracFlow 2 1 _ _ 1 (by norm_num) (by norm_num)⟩

end Transformer.Metastability
