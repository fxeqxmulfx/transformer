module

public import Transformer.Perspective.Section2_ExtendedField

/-!
# A smooth extension of unnormalized attention with linear growth

Sources: arXiv:2312.10794v5, §2, `USA`, and arXiv:2512.01868v4, §4,
`thm:clustering_finite`. The source's exponential field is bounded on the
product of unit spheres but grows rapidly in the ambient product space.
We damp every exponential by the same positive smooth factor
`exp(|beta| * (n - sum_j norm(y_j)^2))` and use the existing polynomial
projection scale. Both factors equal one on the actual state space.
The next module derives sphere invariance from this ambient ODE.
This extension is a proof construction, not a change to the source dynamics.
-/

@[expose] public section

open scoped BigOperators
open Real

namespace Transformer.Perspective

variable {d n : ℕ}

/-- Damped exponential average, agreeing with the source's unnormalized
average on unit tuples. Sources: arXiv:2312.10794v5, §2, `USA`;
arXiv:2512.01868v4, §4, the global-existence assertion. -/
noncomputable def usaExtensionAverage (β : ℝ) (Y : Idx n → EucSpace d)
    (i : Idx n) : EucSpace d :=
  (n : ℝ)⁻¹ • ∑ j : Idx n,
    Real.exp (β * inner (𝕜 := ℝ) (Y i) (Y j) +
      |β| * ((n : ℝ) - ∑ k : Idx n, inner (𝕜 := ℝ) (Y k) (Y k))) • Y j

/-- Smooth ambient field whose restriction will be proved to be `USA`.
The two damping factors only control growth outside the unit state space.
Sources: arXiv:2312.10794v5, §2; arXiv:2512.01868v4, §4. -/
noncomputable def usaGlobalField (β : ℝ) (Y : Idx n → EucSpace d) : Idx n → EucSpace d :=
  saExtensionScale Y • fun i => proj d (Y i) (usaExtensionAverage β Y i)

/-- The damped average is the exact source average on unit tuples.
Sources: arXiv:2312.10794v5, §2, `USA`; arXiv:2512.01868v4, `eq:USA`. -/
theorem usaExtensionAverage_eq (β : ℝ) (Y : Idx n → EucSpace d)
    (hY : ∀ i, ‖Y i‖ = 1) (i : Idx n) :
    usaExtensionAverage β Y i =
      (n : ℝ)⁻¹ • ∑ j, Real.exp (β * inner (𝕜 := ℝ) (Y i) (Y j)) • Y j := by
  have hs : ∑ k : Idx n, inner (𝕜 := ℝ) (Y k) (Y k) = (n : ℝ) := by
    simp [hY]
  unfold usaExtensionAverage
  rw [hs, sub_self, mul_zero]
  simp only [add_zero]

example : ∀ i : Idx 1, ‖(fun _ => (basePoint 0 : EucSpace 1)) i‖ = 1 := by
  intro i
  simp [basePoint, PiLp.norm_single]

/-- The complete extended field restricts to the source's exact `USA`
velocity. Sources: arXiv:2312.10794v5, §2; arXiv:2512.01868v4, §4. -/
theorem usaGlobalField_apply (β : ℝ) (Y : Idx n → EucSpace d)
    (hY : ∀ i, ‖Y i‖ = 1) (i : Idx n) :
    usaGlobalField β Y i = proj d (Y i)
      ((n : ℝ)⁻¹ • ∑ j, Real.exp (β * inner (𝕜 := ℝ) (Y i) (Y j)) • Y j) := by
  change saExtensionScale Y • proj d (Y i) (usaExtensionAverage β Y i) = _
  rw [saExtensionScale_eq_one Y hY, one_smul, usaExtensionAverage_eq β Y hY i]

example : ∀ i : Idx 1, ‖(fun _ => (basePoint 0 : EucSpace 1)) i‖ = 1 := by
  intro i
  simp [basePoint, PiLp.norm_single]

/-- Every damped exponent has a uniform upper bound on the ambient space.
Cauchy--Schwarz absorbs its interaction into the squared-norm damping.
Source: the global-existence proof for `thm:clustering_finite`, §4 of
arXiv:2512.01868v4, applied to the actual exponential kernel. -/
theorem usaExtensionExponent_le (β : ℝ) (Y : Idx n → EucSpace d) (i j : Idx n) :
    β * inner (𝕜 := ℝ) (Y i) (Y j) +
      |β| * ((n : ℝ) - ∑ k : Idx n, inner (𝕜 := ℝ) (Y k) (Y k)) ≤ |β| * n := by
  let S := ∑ k : Idx n, inner (𝕜 := ℝ) (Y k) (Y k)
  have hi : ‖Y i‖ ^ 2 ≤ S := by
    rw [← real_inner_self_eq_norm_sq]
    exact Finset.single_le_sum (fun k _ => real_inner_self_nonneg) (Finset.mem_univ i)
  have hj : ‖Y j‖ ^ 2 ≤ S := by
    rw [← real_inner_self_eq_norm_sq]
    exact Finset.single_le_sum (fun k _ => real_inner_self_nonneg) (Finset.mem_univ j)
  have hp : ‖Y i‖ * ‖Y j‖ ≤ S := by nlinarith [sq_nonneg (‖Y i‖ - ‖Y j‖)]
  have hb : β * inner (𝕜 := ℝ) (Y i) (Y j) ≤ |β| * S := by
    calc _ ≤ |β * inner (𝕜 := ℝ) (Y i) (Y j)| := le_abs_self _
      _ = |β| * |inner (𝕜 := ℝ) (Y i) (Y j)| := abs_mul _ _
      _ ≤ |β| * (‖Y i‖ * ‖Y j‖) :=
        mul_le_mul_of_nonneg_left (abs_real_inner_le_norm _ _) (abs_nonneg _)
      _ ≤ |β| * S := mul_le_mul_of_nonneg_left hp (abs_nonneg _)
  change _ + |β| * ((n : ℝ) - S) ≤ _
  linarith

/-- All coordinate averages are smooth, including away from the sphere.
Source: arXiv:2312.10794v5, §2, the exponential kernel of `USA`. -/
theorem contDiff_usaExtensionAverage (β : ℝ) (i : Idx n) :
    ContDiff ℝ (⊤ : ℕ∞) (fun Y : Idx n → EucSpace d => usaExtensionAverage β Y i) := by
  have hc (j : Idx n) : ContDiff ℝ (⊤ : ℕ∞) (fun Y : Idx n → EucSpace d => Y j) := by
    fun_prop
  unfold usaExtensionAverage
  apply ContDiff.const_smul
  apply ContDiff.sum
  intro j _
  have he : ContDiff ℝ (⊤ : ℕ∞) (fun Y : Idx n → EucSpace d =>
      Real.exp (β * inner (𝕜 := ℝ) (Y i) (Y j) +
        |β| * ((n : ℝ) - ∑ k : Idx n, inner (𝕜 := ℝ) (Y k) (Y k)))) := by
    apply ContDiff.exp
    apply ContDiff.add
    · exact contDiff_const.mul ((hc i).inner ℝ (hc j))
    · apply ContDiff.mul contDiff_const
      apply ContDiff.sub contDiff_const
      exact ContDiff.sum fun k _ => (hc k).inner ℝ (hc k)
  exact he.smul (hc j)

/-- The ambient extension is smooth on the whole product space.
Source: the global-existence assertion of arXiv:2512.01868v4, §4. -/
theorem contDiff_usaGlobalField (β : ℝ) :
    ContDiff ℝ (⊤ : ℕ∞) (usaGlobalField (d := d) (n := n) β) := by
  apply contDiff_saExtensionScale.smul
  apply contDiff_pi.mpr
  intro i
  have hc : ContDiff ℝ (⊤ : ℕ∞) (fun Y : Idx n → EucSpace d => Y i) := by fun_prop
  exact (contDiff_usaExtensionAverage β i).sub
    ((hc.inner ℝ (contDiff_usaExtensionAverage β i)).smul hc)

/-- The average grows at most linearly, despite the original exponential
weights. Source: the auxiliary extension of the `USA` kernel in §2 of
arXiv:2312.10794v5, used for arXiv:2512.01868v4, §4. -/
theorem norm_usaExtensionAverage_le (β : ℝ) (Y : Idx n → EucSpace d) (i : Idx n) :
    ‖usaExtensionAverage β Y i‖ ≤ Real.exp (|β| * n) * ‖Y‖ := by
  have hn : (0 : ℝ) < n := by exact_mod_cast Fin.pos_iff_nonempty.mpr ⟨i⟩
  rw [usaExtensionAverage, norm_smul, Real.norm_of_nonneg (inv_nonneg.mpr hn.le)]
  apply (mul_le_mul_of_nonneg_left (norm_sum_le _ _) (inv_nonneg.mpr hn.le)).trans
  calc (n : ℝ)⁻¹ * ∑ j, ‖Real.exp _ • Y j‖
      ≤ (n : ℝ)⁻¹ * ∑ _j : Idx n, Real.exp (|β| * n) * ‖Y‖ := by
        apply mul_le_mul_of_nonneg_left _ (inv_nonneg.mpr hn.le)
        apply Finset.sum_le_sum
        intro j _
        rw [norm_smul, Real.norm_of_nonneg (Real.exp_pos _).le]
        exact mul_le_mul (Real.exp_le_exp.mpr (usaExtensionExponent_le β Y i j))
          (norm_le_pi_norm Y j) (norm_nonneg _) (Real.exp_pos _).le
    _ = _ := by simp [hn.ne', ← mul_assoc]

/-- The full extension has linear growth; the rational projection scale
absorbs the quadratic factor from the ambient tangent projection.
Sources: arXiv:2312.10794v5, §2; arXiv:2512.01868v4, §4. -/
theorem norm_usaGlobalField_le (β : ℝ) (Y : Idx n → EucSpace d) :
    ‖usaGlobalField β Y‖ ≤ (((n : ℝ) + 1) * Real.exp (|β| * n)) * ‖Y‖ := by
  apply (pi_norm_le_iff_of_nonneg (by positivity)).mpr
  intro i
  let C := Real.exp (|β| * n)
  let v := usaExtensionAverage β Y i
  have hv : ‖v‖ ≤ C * ‖Y‖ := norm_usaExtensionAverage_le β Y i
  have hi : |inner (𝕜 := ℝ) (Y i) v| ≤ ‖Y i‖ * (C * ‖Y‖) :=
    (abs_real_inner_le_norm _ _).trans (mul_le_mul_of_nonneg_left hv (norm_nonneg _))
  have hp : ‖proj d (Y i) v‖ ≤ (1 + ‖Y i‖ ^ 2) * (C * ‖Y‖) := by
    calc ‖proj d (Y i) v‖ ≤ ‖v‖ + ‖inner (𝕜 := ℝ) (Y i) v • Y i‖ := norm_sub_le _ _
      _ = ‖v‖ + |inner (𝕜 := ℝ) (Y i) v| * ‖Y i‖ := by
        rw [norm_smul, Real.norm_eq_abs]
      _ ≤ C * ‖Y‖ + (‖Y i‖ * (C * ‖Y‖)) * ‖Y i‖ :=
        add_le_add hv (mul_le_mul_of_nonneg_right hi (norm_nonneg _))
      _ = _ := by ring
  have hs : ‖Y i‖ ^ 2 ≤ ∑ j : Idx n, inner (𝕜 := ℝ) (Y j) (Y j) := by
    rw [← real_inner_self_eq_norm_sq]
    exact Finset.single_le_sum (fun j _ => real_inner_self_nonneg) (Finset.mem_univ i)
  have hscale : saExtensionScale Y * (1 + ‖Y i‖ ^ 2) ≤ (n : ℝ) + 1 := by
    rw [saExtensionScale, div_mul_eq_mul_div, div_le_iff₀ (saExtensionDenom_pos Y)]
    exact mul_le_mul_of_nonneg_left (add_le_add le_rfl hs) (by positivity)
  change ‖saExtensionScale Y • proj d (Y i) v‖ ≤ _
  rw [norm_smul, Real.norm_of_nonneg (saExtensionScale_pos Y).le]
  calc _ ≤ saExtensionScale Y * ((1 + ‖Y i‖ ^ 2) * (C * ‖Y‖)) :=
        mul_le_mul_of_nonneg_left hp (saExtensionScale_pos Y).le
    _ = (saExtensionScale Y * (1 + ‖Y i‖ ^ 2)) * (C * ‖Y‖) := by ring
    _ ≤ ((n : ℝ) + 1) * (C * ‖Y‖) := mul_le_mul_of_nonneg_right hscale (by positivity)
    _ = _ := by ring

end Transformer.Perspective
