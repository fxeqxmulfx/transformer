/-
# Normalization — a radial invariant of Pre-LN and Peri-LN

The polar identities in arXiv:2510.22026v2, §2, equation (NA) and Table 2,
imply that `r(t)^2 (‖θ(t)‖^2 - 1)` is constant. This algebraic invariant is
valid even at a zero radius, where the formal equations use `0⁻¹ = 0`.
Consequently a unit initial direction stays unit whenever its radius is
nonzero. This is the first step in proving radius positivity and the sphere
constraint directly from `SchemeDynamics`, without assuming that constraint
along the path.

Pre-LN uses the forcing `A`; Peri-LN uses `A/‖A‖`. The invariant depends only
on the two polar equations, so no continuity of the forcing is assumed.
-/

import Transformer.Normalization.Convergence
import Mathlib.Analysis.Calculus.MeanValue

open scoped BigOperators
open Real Set

namespace Transformer.Normalization

/-- A forward path with zero derivative is constant, by the mean value
theorem. This is the calculus step used for the polar identities of
arXiv:2510.22026v2, §2, equation (NA). -/
theorem eq_initial_of_hasDerivWithinAt_zero {E : Type*}
    [NormedAddCommGroup E] [NormedSpace ℝ E] (f : ℝ → E)
    (hf : ∀ t : ℝ, 0 ≤ t → HasDerivWithinAt f 0 (Ici 0) t)
    (t : ℝ) (ht : 0 ≤ t) : f t = f 0 := by
  have hle := Convex.norm_image_sub_le_of_norm_hasDerivWithin_le
    (f := f) (f' := fun _ => 0) (C := 0) (x := (0 : ℝ)) (y := t)
    (fun u hu => hf u hu)
    (fun _ _ => by simp) (convex_Ici (0 : ℝ))
    (by simp) (by simpa using ht)
  exact sub_eq_zero.mp (norm_le_zero_iff.mp (by simpa using hle))

/-- A constant path realizes the derivative and time hypotheses. -/
example : (∀ t : ℝ, 0 ≤ t →
    HasDerivWithinAt (fun _ : ℝ => (1 : ℝ)) 0 (Ici 0) t) ∧ (0 : ℝ) ≤ 1 :=
  ⟨fun t _ => hasDerivWithinAt_const t _ _, zero_le_one⟩

/-- The polar equations make the derivative of the radial norm defect zero.
No unit-norm or nonzero-radius hypothesis enters this identity.
Source: arXiv:2510.22026v2, §2, the polar identities following equation (NA). -/
theorem hasDerivWithinAt_radial_norm_defect {d : ℕ}
    (x : ℝ → EucSpace d) (r : ℝ → ℝ) (F : EucSpace d) (s : Set ℝ) (t : ℝ)
    (hx : HasDerivWithinAt x ((r t)⁻¹ • proj d (x t) F) s t)
    (hr : HasDerivWithinAt r (inner (𝕜 := ℝ) (x t) F) s t) :
    HasDerivWithinAt (fun u => r u ^ 2 * (‖x u‖ ^ 2 - 1)) 0 s t := by
  convert (hr.fun_pow 2).mul (hx.norm_sq.sub_const 1) using 1
  simp only [Nat.cast_ofNat, Nat.reduceSub, pow_one, real_inner_smul_right,
    proj, inner_sub_right, real_inner_self_eq_norm_sq]
  by_cases hzero : r t = 0
  · simp [hzero]
  · field_simp
    ring

/-- Constant unit directions, radius one and zero forcing satisfy both
polar derivative hypotheses. -/
example (d : ℕ) :
    HasDerivWithinAt (fun _ : ℝ => (basePoint d : EucSpace (d + 1)))
      ((1 : ℝ)⁻¹ • proj (d + 1) (basePoint d : EucSpace (d + 1)) 0) (Ici 0) 0 ∧
    HasDerivWithinAt (fun _ : ℝ => (1 : ℝ))
      (inner (𝕜 := ℝ) (basePoint d : EucSpace (d + 1)) 0) (Ici 0) 0 := by
  constructor <;> simpa [proj] using hasDerivWithinAt_const (0 : ℝ) (Ici 0) _

/-- The radial norm defect is constant on the forward half-line, including
times with zero radius. Source: arXiv:2510.22026v2, §2, the polar identities
following equation (NA) and the Pre-LN/Peri-LN rows of Table 2. -/
theorem radial_norm_defect_eq_initial {d : ℕ} (x : ℝ → EucSpace d)
    (r : ℝ → ℝ) (F : ℝ → EucSpace d)
    (hx : ∀ t : ℝ, 0 ≤ t →
      HasDerivWithinAt x ((r t)⁻¹ • proj d (x t) (F t)) (Ici 0) t)
    (hr : ∀ t : ℝ, 0 ≤ t →
      HasDerivWithinAt r (inner (𝕜 := ℝ) (x t) (F t)) (Ici 0) t)
    (t : ℝ) (ht : 0 ≤ t) :
    r t ^ 2 * (‖x t‖ ^ 2 - 1) = r 0 ^ 2 * (‖x 0‖ ^ 2 - 1) := by
  exact eq_initial_of_hasDerivWithinAt_zero _
    (fun u hu => hasDerivWithinAt_radial_norm_defect x r (F u) (Ici 0) u
      (hx u hu) (hr u hu)) t ht

/-- The global polar hypotheses are met by the same stationary unit path. -/
example (d : ℕ) :
    (∀ t : ℝ, 0 ≤ t →
      HasDerivWithinAt (fun _ : ℝ => (basePoint d : EucSpace (d + 1)))
        ((1 : ℝ)⁻¹ • proj (d + 1) (basePoint d : EucSpace (d + 1)) 0) (Ici 0) t) ∧
    (∀ t : ℝ, 0 ≤ t → HasDerivWithinAt (fun _ : ℝ => (1 : ℝ))
      (inner (𝕜 := ℝ) (basePoint d : EucSpace (d + 1)) 0) (Ici 0) t) ∧ (0 : ℝ) ≤ 1 := by
  refine ⟨fun t _ => ?_, fun t _ => ?_, zero_le_one⟩ <;>
    simpa [proj] using hasDerivWithinAt_const _ (Ici 0) _

/-- Unit initial data make the radial norm defect vanish. This is an
invariant derived from the equations, rather than a sphere constraint added
to their definition. Source: arXiv:2510.22026v2, §2, equation (NA), Table 2. -/
theorem radial_norm_defect_zero {d : ℕ} (x : ℝ → EucSpace d)
    (r : ℝ → ℝ) (F : ℝ → EucSpace d)
    (hx : ∀ t : ℝ, 0 ≤ t →
      HasDerivWithinAt x ((r t)⁻¹ • proj d (x t) (F t)) (Ici 0) t)
    (hr : ∀ t : ℝ, 0 ≤ t →
      HasDerivWithinAt r (inner (𝕜 := ℝ) (x t) (F t)) (Ici 0) t)
    (h0 : ‖x 0‖ = 1) (t : ℝ) (ht : 0 ≤ t) :
    r t ^ 2 * (‖x t‖ ^ 2 - 1) = 0 := by
  rw [radial_norm_defect_eq_initial x r F hx hr t ht, h0]
  norm_num

/-- A stationary unit path meets every hypothesis of `radial_norm_defect_zero`. -/
example (d : ℕ) :
    (∀ t : ℝ, 0 ≤ t →
      HasDerivWithinAt (fun _ : ℝ => (basePoint d : EucSpace (d + 1)))
        ((1 : ℝ)⁻¹ • proj (d + 1) (basePoint d : EucSpace (d + 1)) 0) (Ici 0) t) ∧
    (∀ t : ℝ, 0 ≤ t → HasDerivWithinAt (fun _ : ℝ => (1 : ℝ))
      (inner (𝕜 := ℝ) (basePoint d : EucSpace (d + 1)) 0) (Ici 0) t) ∧
    ‖(basePoint d : EucSpace (d + 1))‖ = 1 ∧ (0 : ℝ) ≤ 1 := by
  refine ⟨fun t _ => ?_, fun t _ => ?_, by simp [basePoint, PiLp.norm_single],
    zero_le_one⟩ <;> simpa [proj] using hasDerivWithinAt_const _ (Ici 0) _

/-- The invariant for the actual Pre-LN and Peri-LN attention equations.
The forcing is `A` for Pre-LN and `‖A‖⁻¹ A` for Peri-LN, exactly as in
Table 2. The identity also holds at a vanishing attention vector.
Source: arXiv:2510.22026v2, §2, equation (NA), Table 2. -/
theorem SchemeDynamics.radial_norm_defect_zero {d n : ℕ}
    {β τ : ℝ} {Q K V : ℝ → ParamMatrix d} {α : ℝ → ℝ} {scheme : Scheme}
    {θ : ℝ → Idx n → EucSpace d} {r : ℝ → Idx n → ℝ}
    (hD : SchemeDynamics d n β Q K V α τ scheme θ r)
    (hscheme : scheme = .pre ∨ scheme = .peri) (h0 : ∀ j, ‖θ 0 j‖ = 1)
    (t : ℝ) (ht : 0 ≤ t) (j : Idx n) :
    r t j ^ 2 * (‖θ t j‖ ^ 2 - 1) = 0 := by
  rcases hscheme with rfl | rfl
  · exact Transformer.Normalization.radial_norm_defect_zero (fun u => θ u j) (fun u => r u j)
      (fun u => attentionVec d n β (Q u) (K u) (V u) (θ u) j)
      (fun u hu => (hD u hu j).1) (fun u hu => (hD u hu j).2) (h0 j) t ht
  · apply Transformer.Normalization.radial_norm_defect_zero (fun u => θ u j) (fun u => r u j)
      (fun u => ‖attentionVec d n β (Q u) (K u) (V u) (θ u) j‖⁻¹ •
        attentionVec d n β (Q u) (K u) (V u) (θ u) j) ?_ ?_ (h0 j) t ht
    · intro u hu
      simpa only [speedFactor, mul_inv, proj_smul, smul_smul] using (hD u hu j).1
    · intro u hu
      simpa only [radialDerivative, real_inner_smul_right, div_eq_mul_inv, mul_comm]
        using (hD u hu j).2

/-- A genuine attention solution realizes the invariant's hypotheses: zero
value parameters give zero forcing, constant unit directions and radii one.
Both schemes are included. -/
example (d : ℕ) (β : ℝ) :
    (∀ scheme : Scheme, SchemeDynamics (d + 1) 1 β 0 0 0 0 0 scheme
      (fun _ _ => (basePoint d : EucSpace (d + 1))) (fun _ _ => 1)) ∧
    (∀ j : Idx 1, ‖(fun _ _ => (basePoint d : EucSpace (d + 1))) (0 : ℝ) j‖ = 1) ∧
    (0 : ℝ) ≤ 1 ∧ (.pre = Scheme.pre ∨ Scheme.pre = .peri) := by
  refine ⟨?_, fun _ => by simp [basePoint, PiLp.norm_single], zero_le_one, Or.inl rfl⟩
  intro scheme t _ j
  have hrad : radialDerivative (d + 1) 1 β 0 0 0
      (fun _ _ => (basePoint d : EucSpace (d + 1))) 0 scheme t j = 0 := by
    cases scheme <;> simp [radialDerivative, attentionVec]
  constructor
  · simpa [attentionVec, proj] using
      hasDerivWithinAt_const t (Ici 0) (basePoint d : EucSpace (d + 1))
  · rw [hrad]
    exact hasDerivWithinAt_const _ _ _

end Transformer.Normalization
