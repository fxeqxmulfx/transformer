/-
# Transported caps for a Dirac characteristic flow

ArXiv:2410.06833v1, §5, proof of `thm: metastability MF`, Steps 1 and 2.
A boundary point with `⟨b,w⟩ = 1-ε` minimizes the longitudinal coordinate
of the initial cap. The explicit flow preserves the ordering of these
coordinates, so its trajectory is a minimizer of the whole transported
cap at every time. This is the source's geometric cap, not just the support
of the initial measure, which may be smaller.

At time zero, `μ₀ = δ_w` gives `η = 1-ε` and `V = ε` around this
minimizer. The two explicit circle points below realize the boundary
condition at `ε = 1/10000`. The center stays fixed and its push-forward
is exactly the stationary solution, as required by the characteristic
representation in `eq: flow.map`.
-/

import Transformer.Metastability.Section5_DiracDynamics

open scoped BigOperators
open Real MeasureTheory

namespace Transformer.Metastability

open Perspective

/-- A cap boundary trajectory is a genuine minimizer in Step 1 of the mean-field proof.

Source: arXiv:2410.06833v1, §5. -/
theorem isCapArgmin_diracFlow (d : ℕ) (w b : SSphere d) (ε : ℝ)
    (hb : inner (𝕜 := ℝ) (b : EucSpace d) (w : EucSpace d) = 1 - ε) :
    IsCapArgmin d (diracFlow d w) w ε (fun t => diracFlow d w t b) := by
  intro t
  refine ⟨⟨b, ?_, rfl⟩, ?_⟩
  · change 1 - ε ≤ inner (𝕜 := ℝ) (b : EucSpace d) (w : EucSpace d)
    exact hb.ge
  · rintro y ⟨z, hz, rfl⟩
    rw [inner_diracFlow, inner_diracFlow]
    apply diracAlong_mono
    · exact real_inner_mem_Icc_of_norm_eq_one (mem_sphere_zero_iff_norm.mp b.2)
        (mem_sphere_zero_iff_norm.mp w.2)
    · exact real_inner_mem_Icc_of_norm_eq_one (mem_sphere_zero_iff_norm.mp z.2)
        (mem_sphere_zero_iff_norm.mp w.2)
    · rw [hb]
      exact hz

/-- The initial minimum in Step 1 of the mean-field proof is the cap boundary coordinate.

Source: arXiv:2410.06833v1, §5. -/
theorem capMin_diracFlow_zero (d : ℕ) (w b : SSphere d) (ε : ℝ)
    (hb : inner (𝕜 := ℝ) (b : EucSpace d) (w : EucSpace d) = 1 - ε) :
    capMin d (diracFlow d w) w ε 0 = 1 - ε := by
  rw [capMin_eq_of_isCapArgmin d _ w ε _ (isCapArgmin_diracFlow d w b ε hb) 0,
    diracFlow_zero, hb]

/-- Step 2 defines variance around a minimizer of the entire cap.
For a Dirac mass at the center it is exactly `ε`, even though the mass itself does not move.

Source: arXiv:2410.06833v1, §5. -/
theorem capVariance_diracFlow_zero (d : ℕ) (w b : SSphere d) (ε : ℝ) (hε : 0 ≤ ε)
    (hb : inner (𝕜 := ℝ) (b : EucSpace d) (w : EucSpace d) = 1 - ε) :
    capVariance d (diracProb d w) (diracFlow d w) w ε
      (fun t => diracFlow d w t b) 0 = ε := by
  classical
  have hw : w ∈ sphericalCap d w (2 * ε) := by
    change 1 - 2 * ε ≤ inner (𝕜 := ℝ) (w : EucSpace d) (w : EucSpace d)
    rw [real_inner_self_eq_norm_mul_norm, mem_sphere_zero_iff_norm.mp w.2]
    linarith
  have hμ : (diracProb d w : Measure (SSphere d)) = Measure.dirac w := rfl
  rw [capVariance, hμ, restrict_dirac, ite_eq_left hw, integral_dirac,
    diracFlow_zero, diracFlow_zero, norm_sub_sq_real,
    mem_sphere_zero_iff_norm.mp w.2, mem_sphere_zero_iff_norm.mp b.2,
    real_inner_comm (b : EucSpace d) (w : EucSpace d), hb]
  ring

/-- The initial cap lies inside its double, so time zero belongs to the escape window in Step 1.

Source: arXiv:2410.06833v1, §5. -/
theorem zero_mem_escapeWindow_diracFlow (d : ℕ) (k : ℕ) (w : SSphere d)
    (ε : ℝ) (hε : 0 ≤ ε) :
    (0 : ℝ) ∈ escapeWindow d k (fun _ => w) (diracFlow d w) ε := by
  refine ⟨le_rfl, ?_⟩
  intro s hs q z hz
  have hs0 : s = 0 := le_antisymm hs.2 hs.1
  subst s
  rcases hz with ⟨z', hz', he⟩
  rw [diracFlow_zero] at he
  subst z
  change 1 - 2 * ε ≤ inner (𝕜 := ℝ) (z' : EucSpace d) (w : EucSpace d)
  change 1 - ε ≤ inner (𝕜 := ℝ) (z' : EucSpace d) (w : EucSpace d) at hz'
  linarith

/-- The characteristic representation of `eq: flow.map` agrees with the stationary Dirac solution.

Source: arXiv:2410.06833v1, §5. -/
theorem map_diracFlow_dirac (d : ℕ) (w : SSphere d) (t : ℝ) :
    (diracProb d w : Measure (SSphere d)).map (diracFlow d w t) = diracProb d w := by
  change (Measure.dirac w).map (diracFlow d w t) = Measure.dirac w
  rw [Measure.map_dirac, diracFlow_self]

/-- The circle center `(1,0)` used to refute `eq: v.small`.

Source: arXiv:2410.06833v1, §5. -/
noncomputable def varianceCounterCentre : SSphere 2 :=
  ⟨EuclideanSpace.single (0 : Fin 2) (1 : ℝ), by
    simp [PiLp.norm_single]⟩

/-- A unit circle point with coordinate `1-1/10000` along the center.
Its orthogonal coordinate is the positive square root required by unit norm.

Source: arXiv:2410.06833v1, §5. -/
noncomputable def varianceCounterBoundary : SSphere 2 :=
  ⟨WithLp.toLp 2 (![9999 / 10000, Real.sqrt (1 - (9999 / 10000 : ℝ) ^ 2)] : Fin 2 → ℝ), by
    rw [mem_sphere_zero_iff_norm]
    have hsq : ‖WithLp.toLp 2
        (![9999 / 10000, Real.sqrt (1 - (9999 / 10000 : ℝ) ^ 2)] : Fin 2 → ℝ)‖ ^ 2 = 1 := by
      rw [EuclideanSpace.real_norm_sq_eq, Fin.sum_univ_two]
      dsimp
      rw [Real.sq_sqrt (by norm_num)]
      norm_num
    nlinarith [norm_nonneg (WithLp.toLp 2
      (![9999 / 10000, Real.sqrt (1 - (9999 / 10000 : ℝ) ^ 2)] : Fin 2 → ℝ))]⟩

/-- The two explicit circle points meet the exact boundary condition used in `eq: v.small`.

Source: arXiv:2410.06833v1, §5. -/
theorem varianceCounterBoundary_coordinate :
    inner (𝕜 := ℝ) (varianceCounterBoundary : EucSpace 2)
      (varianceCounterCentre : EucSpace 2) = 1 - (1 / 10000 : ℝ) := by
  norm_num [varianceCounterBoundary, varianceCounterCentre,
    EuclideanSpace.inner_eq_star_dotProduct, dotProduct, Fin.sum_univ_two,
    EuclideanSpace.single]

/-- The minimizer and cap-minimum hypotheses are attained on the circle. -/
example : inner (𝕜 := ℝ) (varianceCounterBoundary : EucSpace 2)
    (varianceCounterCentre : EucSpace 2) = 1 - (1 / 10000 : ℝ) :=
  varianceCounterBoundary_coordinate

/-- The variance lemma's positivity and boundary hypotheses hold together. -/
example : (0 : ℝ) ≤ 1 / 10000 ∧
    inner (𝕜 := ℝ) (varianceCounterBoundary : EucSpace 2)
      (varianceCounterCentre : EucSpace 2) = 1 - (1 / 10000 : ℝ) :=
  ⟨by norm_num, varianceCounterBoundary_coordinate⟩

/-- The escape-window hypothesis is satisfied by the counterexample's cap. -/
example : (0 : ℝ) ≤ 1 / 10000 ∧
    (0 : ℝ) ∈ escapeWindow 2 1 (fun _ => varianceCounterCentre)
      (diracFlow 2 varianceCounterCentre) (1 / 10000) :=
  ⟨by norm_num, zero_mem_escapeWindow_diracFlow 2 1 _ _ (by norm_num)⟩

end Transformer.Metastability
