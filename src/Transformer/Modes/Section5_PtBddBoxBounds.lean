import Transformer.Modes.Section5_PtBddTailMass
import Mathlib.Analysis.Normed.Group.Bounded

/-
# The number of modes of a Gaussian KDE — consequences of a continuous density

The continuity conclusion of `lem: pt.bdd`, arXiv:2412.09080v3, §4.1 and
§5.5, forces the probability of a centered rectangle of radius `r` to be
`O(r²)`: the density is bounded on a fixed compact rectangle and its area
is `4r²`. This implication requires neither Fourier inversion nor any
regularity of the sampling map.

The last two lemmas give an explicit large radius at which a positive
quadratic exponent exceeds any proposed bound. They implement the comparison
between that area estimate and the Gaussian tail estimate in
`Section5_PtBddTailMass.lean`.

Source: arXiv:2412.09080v3, §4.1, `lem: pt.bdd`; §5.5, its proposed proof.
-/

open Real MeasureTheory
open scoped ENNReal

namespace Transformer.Modes

/-- Increasing the rectangle radius increases the test set for the joint
density in §5.5, `lem: pt.bdd`. -/
theorem centeredBox_mono {r R : ℝ} (h : r ≤ R) : centeredBox r ⊆ centeredBox R := by
  intro z hz
  rw [mem_centeredBox] at hz ⊢
  exact ⟨hz.1.trans h, hz.2.trans h⟩

/-- The radius comparison has positive witnesses. -/
example : (1 : ℝ) ≤ 2 := by norm_num

/-- Every nonnegative test radius gives a nonempty rectangle containing
the origin; the origin is never excluded from the probability test set.
Source: arXiv:2412.09080v3, §5.5, the joint law near `(0, 0)`. -/
theorem centeredBox_nonempty {r : ℝ} (hr : 0 ≤ r) : (centeredBox r).Nonempty := by
  refine ⟨(0, 0), ?_⟩
  simpa only [mem_centeredBox, abs_zero] using And.intro hr hr

/-- A positive radius witnesses nonemptiness without a degenerate rectangle. -/
example : (0 : ℝ) ≤ 1 := zero_le_one

/-- The rectangle used to test the density has area `4r²`.
Source: arXiv:2412.09080v3, §5.5, the planar law in `lem: pt.bdd`. -/
theorem volume_centeredBox {r : ℝ} (hr : 0 ≤ r) :
    volume (centeredBox r) = ENNReal.ofReal (4 * r ^ 2) := by
  rw [centeredBox, Measure.volume_eq_prod, Measure.prod_prod, Real.volume_Icc,
    ← ENNReal.ofReal_mul (by linarith : 0 ≤ r - -r)]
  congr 1
  ring

/-- A positive radius witnesses the area formula's hypothesis. -/
example : (0 : ℝ) ≤ 1 := zero_le_one

/-- A density bounded above by `B` on a rectangle gives mass at most its
area times `B`. Nonnegativity of the density is handled by `ofReal`.
Source: arXiv:2412.09080v3, §5.5, boundedness asserted in `lem: pt.bdd`. -/
theorem density_box_le_of_bound {P : Measure (ℝ × ℝ)} {q : ℝ × ℝ → ℝ}
    {B r : ℝ} (hB : 0 ≤ B) (hr : 0 ≤ r) (hq : ∀ z ∈ centeredBox r, q z ≤ B)
    (hP : P = volume.withDensity fun z => ENNReal.ofReal (q z)) :
    P.real (centeredBox r) ≤ 4 * B * r ^ 2 := by
  have hbound : P (centeredBox r) ≤ ENNReal.ofReal (4 * B * r ^ 2) := by
    rw [hP, withDensity_apply _ (measurableSet_centeredBox r)]
    calc
      ∫⁻ z in centeredBox r, ENNReal.ofReal (q z)
          ≤ ∫⁻ _z in centeredBox r, ENNReal.ofReal B := by
            refine setLIntegral_mono measurable_const fun z hz => ENNReal.ofReal_le_ofReal ?_
            exact hq z hz
      _ = ENNReal.ofReal B * ENNReal.ofReal (4 * r ^ 2) := by
            rw [setLIntegral_const, volume_centeredBox hr]
      _ = ENNReal.ofReal (4 * B * r ^ 2) := by
            rw [← ENNReal.ofReal_mul hB]
            congr 1
            ring
  have hreal := ENNReal.toReal_mono ENNReal.ofReal_ne_top hbound
  simpa only [measureReal_def, ENNReal.toReal_ofReal (by positivity :
    0 ≤ 4 * B * r ^ 2)] using hreal

/-- Unit density and radius satisfy all the bounded-density assumptions. -/
example : (0 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 ∧
    (∀ z ∈ centeredBox 1, (fun _ : ℝ × ℝ => (1 : ℝ)) z ≤ 1) ∧
    (volume.withDensity (fun _ : ℝ × ℝ => ENNReal.ofReal (1 : ℝ))) =
      volume.withDensity (fun _ : ℝ × ℝ => ENNReal.ofReal (1 : ℝ)) :=
  ⟨zero_le_one, zero_le_one, fun _ _ => le_rfl, rfl⟩

/-- A continuous density forces quadratic decay of small rectangle masses.
Only continuity on one compact rectangle is needed; `lem: pt.bdd` asserts
continuity on the entire plane.
Source: arXiv:2412.09080v3, §4.1 and §5.5, `lem: pt.bdd`. -/
theorem continuous_density_box_le {P : Measure (ℝ × ℝ)} {q : ℝ × ℝ → ℝ}
    (hq : ContinuousOn q (centeredBox 1))
    (hP : P = volume.withDensity fun z => ENNReal.ofReal (q z)) :
    ∃ C : ℝ, 0 ≤ C ∧ ∀ r : ℝ, 0 ≤ r → r ≤ 1 →
      P.real (centeredBox r) ≤ C * r ^ 2 := by
  have hcompact : IsCompact (centeredBox 1) := isCompact_Icc.prod isCompact_Icc
  obtain ⟨B, hB⟩ := hcompact.exists_bound_of_continuousOn hq
  refine ⟨4 * max B 0, by positivity, ?_⟩
  intro r hr hr1
  refine density_box_le_of_bound (le_max_right B 0) hr ?_ hP
  intro z hz
  calc
    q z ≤ |q z| := le_abs_self _
    _ ≤ B := by simpa only [Real.norm_eq_abs] using hB z (centeredBox_mono hr1 hz)
    _ ≤ max B 0 := le_max_left _ _

/-- Constant unit density satisfies both assumptions of the rectangle estimate.
The estimate is local, so a probability-measure hypothesis is unnecessary. -/
example : ContinuousOn (fun _ : ℝ × ℝ => (1 : ℝ)) (centeredBox 1) ∧
    (volume.withDensity (fun _ : ℝ × ℝ => ENNReal.ofReal (1 : ℝ))) =
      volume.withDensity (fun _ : ℝ × ℝ => ENNReal.ofReal (1 : ℝ)) :=
  ⟨continuous_const.continuousOn, rfl⟩

/-- A positive quadratic term dominates a linear term and a constant at
an explicitly selectable remote radius.
Source: arXiv:2412.09080v3, §5.5, comparison of Gaussian tail mass and area. -/
theorem exists_far_quadratic_ge {κ M : ℝ} (hκ : 0 < κ) (hM : 0 ≤ M) (b L : ℝ) :
    ∃ z : ℝ, 1 ≤ z ∧ L ≤ z ∧ M + b * z ≤ κ * z ^ 2 := by
  let z := max (max 1 L) ((b + M) / κ)
  have hz1 : 1 ≤ z := (le_max_left _ _).trans (le_max_left _ _)
  have hzL : L ≤ z := (le_max_right _ _).trans (le_max_left _ _)
  have hzq : (b + M) / κ ≤ z := le_max_right _ _
  have hmul : b + M ≤ κ * z := by
    rwa [div_le_iff₀ hκ, mul_comm] at hzq
  refine ⟨z, hz1, hzL, ?_⟩
  have hscaled := mul_le_mul_of_nonneg_left hmul (by linarith : 0 ≤ z)
  have hconst := mul_nonneg (by linarith : 0 ≤ z - 1) hM
  nlinarith

/-- Unit positive quadratic coefficient and nonnegative constant satisfy
the radius lemma. The linear coefficient and lower cutoff are arbitrary. -/
example : (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 := ⟨one_pos, zero_le_one⟩

/-- The resulting exponential exceeds any proposed constant, beyond any
chosen radius. This supplies the strict inequality in the density refutation.
Source: arXiv:2412.09080v3, §5.5, counterexample to `lem: pt.bdd`. -/
theorem exists_far_exp_gt {κ : ℝ} (hκ : 0 < κ) (b d M L : ℝ) :
    ∃ z : ℝ, 1 ≤ z ∧ L ≤ z ∧ M < Real.exp (κ * z ^ 2 - b * z - d) := by
  obtain ⟨z, hz1, hzL, hz⟩ := exists_far_quadratic_ge hκ
    (le_max_right (M + d) 0) b L
  refine ⟨z, hz1, hzL, ?_⟩
  have hMd : M + d ≤ max (M + d) 0 := le_max_left _ _
  have hexp := Real.add_one_le_exp (κ * z ^ 2 - b * z - d)
  linarith

/-- A positive quadratic coefficient witnesses the exponential comparison's
only assumption. -/
example : (0 : ℝ) < 1 := one_pos

end Transformer.Modes
