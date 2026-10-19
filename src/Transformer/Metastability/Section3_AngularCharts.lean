module

public import Transformer.Metastability.AngularEnergy
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Angle

/-!
# Angular chart representatives preserve the actual energy and gradient

ArXiv:2410.06833v1, §3.2, `eq: cones_2`, `eq: tau.small`, and
`lem: PL.borjan`. A token angle may be any real representative of its
torus point. Before estimating the width of a cap, we choose its
representative within `π` of that cap's center. This changes neither the
interaction energy nor any angular gradient component.
-/

@[expose] public section

open scoped BigOperators
open Real

namespace Transformer.Metastability

/-- The real representative of `θ` in the chart centered at `ω`.
Source: arXiv:2410.06833v1, §3.2, the cap chart in `eq: tau.small`. -/
noncomputable def capChart (ω θ : ℝ) : ℝ := ω + ((θ - ω : ℝ) : Real.Angle).toReal

/-- Choosing the chart preserves the torus point. Source:
arXiv:2410.06833v1, §3.2, `eq: cones_2`. -/
theorem capChart_angle (ω θ : ℝ) : (capChart ω θ : Real.Angle) = (θ : Real.Angle) := by
  rw [capChart, Real.Angle.coe_add, Real.Angle.coe_toReal, Real.Angle.coe_sub]
  abel

/-- The chosen representative is within `π` of its chart center.
Source: arXiv:2410.06833v1, §3.2, the chart of `eq: tau.small`. -/
theorem capChart_dist_le_pi (ω θ : ℝ) : |capChart ω θ - ω| ≤ π := by
  simpa only [capChart, add_sub_cancel_left] using
    Real.Angle.abs_toReal_le_pi ((θ - ω : ℝ) : Real.Angle)

/-- Equal torus angles have equal real cosines. This transfers cap
membership and exponential interactions between chart representatives.
Source: arXiv:2410.06833v1, §3.2, `eq: cones_2`. -/
theorem cos_eq_of_angle_eq {x y : ℝ} (h : (x : Real.Angle) = (y : Real.Angle)) :
    Real.cos x = Real.cos y := by
  rw [← Real.Angle.cos_coe x, ← Real.Angle.cos_coe y, h]

/-- Equal torus angles also have equal real sines, preserving the
angular gradient. Source: arXiv:2410.06833v1, §3.2, `lem: PL.borjan`. -/
theorem sin_eq_of_angle_eq {x y : ℝ} (h : (x : Real.Angle) = (y : Real.Angle)) :
    Real.sin x = Real.sin y := by
  rw [← Real.Angle.sin_coe x, ← Real.Angle.sin_coe y, h]

/-- The cosine of an angle difference is invariant under replacement
of either representative. Source: arXiv:2410.06833v1, §3.2, `𝖤_β`. -/
theorem cos_sub_eq_of_angles_eq {x x' y y' : ℝ}
    (hx : (x : Real.Angle) = (x' : Real.Angle))
    (hy : (y : Real.Angle) = (y' : Real.Angle)) : Real.cos (x - y) = Real.cos (x' - y') := by
  apply cos_eq_of_angle_eq
  rw [Real.Angle.coe_sub, Real.Angle.coe_sub, hx, hy]

/-- The corresponding sine difference is invariant as well.
Source: arXiv:2410.06833v1, §3.2, the angular gradient of `𝖤_β`. -/
theorem sin_sub_eq_of_angles_eq {x x' y y' : ℝ}
    (hx : (x : Real.Angle) = (x' : Real.Angle))
    (hy : (y : Real.Angle) = (y' : Real.Angle)) : Real.sin (x - y) = Real.sin (x' - y') := by
  apply sin_eq_of_angle_eq
  rw [Real.Angle.coe_sub, Real.Angle.coe_sub, hx, hy]

/-- The actual angular energy depends only on the torus configuration.
Source: arXiv:2410.06833v1, §3.2, `𝖤_β` in `lem: PL.borjan`. -/
theorem angularEβ_congr_angles (n : ℕ) (β : ℝ) (Θ Ψ : Idx n → ℝ)
    (h : ∀ i : Idx n, (Θ i : Real.Angle) = (Ψ i : Real.Angle)) :
    angularEβ n β Θ = angularEβ n β Ψ := by
  unfold angularEβ
  congr 1
  apply Finset.sum_congr rfl
  intro i _
  apply Finset.sum_congr rfl
  intro j _
  rw [cos_sub_eq_of_angles_eq (h i) (h j)]

/-- Every component of the true energy gradient is preserved by chart
replacement. Source: arXiv:2410.06833v1, §3.2, `lem: PL.borjan`. -/
theorem angularGrad_congr_angles (n : ℕ) (β : ℝ) (Θ Ψ : Idx n → ℝ)
    (h : ∀ i : Idx n, (Θ i : Real.Angle) = (Ψ i : Real.Angle)) (l : Idx n) :
    angularGrad n β Θ l = angularGrad n β Ψ l := by
  unfold angularGrad
  congr 2
  apply Finset.sum_congr rfl
  intro j _
  rw [cos_sub_eq_of_angles_eq (h l) (h j), sin_sub_eq_of_angles_eq (h l) (h j)]

/-- The angle-equality inputs above are realized by different real
representatives: the two nonempty constant tuples `0` and `2π`. -/
example : (0 : Real.Angle) = ((2 * π : ℝ) : Real.Angle) ∧
    (∀ i : Idx 2, (((fun _ : Idx 2 => (0 : ℝ)) i) : Real.Angle) =
      (((fun _ : Idx 2 => (2 * π : ℝ)) i) : Real.Angle)) ∧
    (0 : ℝ) ≠ 2 * π := by
  refine ⟨Real.Angle.coe_two_pi.symm, fun _ => Real.Angle.coe_two_pi.symm, ?_⟩
  linarith [Real.pi_pos]

/-- Replace each token by the representative centered at its assigned cap.
The assignment is explicit data, chosen from cap membership in the PL proof.
Source: arXiv:2410.06833v1, §3.2, `eq: cones_2`, `eq: tau.small`. -/
noncomputable def chartTuple {n k : ℕ} (ω : Idx k → ℝ) (p : Idx n → Idx k)
    (Θ : Idx n → ℝ) : Idx n → ℝ := fun i => capChart (ω (p i)) (Θ i)

/-- The tuple replacement preserves every torus coordinate.
Source: arXiv:2410.06833v1, §3.2, `lem: PL.borjan`. -/
theorem chartTuple_angle {n k : ℕ} (ω : Idx k → ℝ) (p : Idx n → Idx k)
    (Θ : Idx n → ℝ) (i : Idx n) :
    (chartTuple ω p Θ i : Real.Angle) = (Θ i : Real.Angle) :=
  capChart_angle (ω (p i)) (Θ i)

/-- Each replaced token lies in the chart of its assigned cap.
Source: arXiv:2410.06833v1, §3.2, `eq: tau.small`. -/
theorem chartTuple_own_chart {n k : ℕ} (ω : Idx k → ℝ) (p : Idx n → Idx k)
    (Θ : Idx n → ℝ) (i : Idx n) : |chartTuple ω p Θ i - ω (p i)| ≤ π :=
  capChart_dist_le_pi (ω (p i)) (Θ i)

/-- Changing the representative preserves its cosine relative to any
cap center, including a center different from its own chart's center.
Source: arXiv:2410.06833v1, §3.2, `eq: cones_2`. -/
theorem capChart_cos_sub (ω θ φ : ℝ) : Real.cos (capChart ω θ - φ) = Real.cos (θ - φ) :=
  cos_sub_eq_of_angles_eq (capChart_angle ω θ) rfl

/-- Cap membership is preserved for every cap, including caps different
from the one used to choose the representative.
Source: arXiv:2410.06833v1, §3.2, `eq: cones_2`. -/
theorem chartTuple_cap_cos {n k : ℕ} (ω : Idx k → ℝ) (p : Idx n → Idx k)
    (Θ : Idx n → ℝ) (i : Idx n) (q : Idx k) :
    Real.cos (chartTuple ω p Θ i - ω q) = Real.cos (Θ i - ω q) :=
  capChart_cos_sub (ω (p i)) (Θ i) (ω q)

/-- **Chart replacement preserves the actual angular energy.**
The chosen cap centers may differ between tokens; only each token's torus
point matters. Source: arXiv:2410.06833v1, §3.2, `lem: PL.borjan`. -/
theorem angularEβ_chartTuple {n k : ℕ} (β : ℝ) (ω : Idx k → ℝ)
    (p : Idx n → Idx k) (Θ : Idx n → ℝ) :
    angularEβ n β (chartTuple ω p Θ) = angularEβ n β Θ := by
  apply angularEβ_congr_angles
  intro i
  exact chartTuple_angle ω p Θ i

/-- **Chart replacement preserves every actual angular gradient component.**
Source: arXiv:2410.06833v1, §3.2, the Euclidean gradient norm in
`lem: PL.borjan`. -/
theorem angularGrad_chartTuple {n k : ℕ} (β : ℝ) (ω : Idx k → ℝ)
    (p : Idx n → Idx k) (Θ : Idx n → ℝ) (i : Idx n) :
    angularGrad n β (chartTuple ω p Θ) i = angularGrad n β Θ i := by
  apply angularGrad_congr_angles
  intro j
  exact chartTuple_angle ω p Θ j

end Transformer.Metastability
