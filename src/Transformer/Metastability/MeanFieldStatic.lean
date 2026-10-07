/-
# Metastability — a static mean-field solution

A Dirac mass `δ_p` solves the mean-field equation at rest in every dimension.
Its field at another unit vector `x` is `Proj_x p`; the normalized kernel
cancels exactly. The full characteristic flow above dimension one is
constructed in `Section5_DiracDynamics`.

On `𝕊^0 = {±p}` the mass does not move, and neither does anything
else: on `𝕊^0` every tangent vector is zero.  The constant curve at `δ_p`
therefore solves `eq: mean.field.pde`, the identity is its characteristic
flow, and with one cap around `p` and a large `β` it is a `(β, ε)`-separated
measure.  This is the configuration that witnesses the hypotheses of
`Metastability.metastability_mf`, the cap-exit claim and
the estimate refuted by `Metastability.variance_small_counterexample`
(arXiv:2410.06833v1, §5).
-/

import Transformer.Metastability.MeanField
import Transformer.Interpolation.Basic

open Real MeasureTheory

namespace Transformer
namespace Metastability

open Perspective

/-- The normalized field in `eq: mean.field.pde` at `δ_w` equals `Proj_z w`.
The positive exponential cancels exactly against its partition function.

Source: arXiv:2410.06833v1, §5. -/
theorem MFVel_diracProb (d : ℕ) (β : ℝ) (w : SSphere d) (z : EucSpace d) :
    MFVel d β (diracProb d w) z = proj d z w := by
  have hμ : (diracProb d w : Measure (SSphere d)) = Measure.dirac w := rfl
  simp [MFVel, partitionMu, hμ, Real.exp_ne_zero]

/-- The mass has zero velocity in `eq: mean.field.pde`, in every dimension.

Source: arXiv:2410.06833v1, §5. -/
theorem MFVel_diracProb_self (d : ℕ) (β : ℝ) (w : SSphere d) :
    MFVel d β (diracProb d w) (w : EucSpace d) = 0 := by
  rw [MFVel_diracProb]
  simpa only [one_smul] using proj_smul_self (mem_sphere_zero_iff_norm.mp w.2) 1

/-- A constant Dirac law solves `eq: mean.field.pde` in every dimension.
Every test-function integral is constant and the velocity at the mass is zero.

Source: arXiv:2410.06833v1, §5. -/
theorem meanFieldPDE_dirac (d : ℕ) (β : ℝ) (w : SSphere d) :
    meanFieldPDE d β (fun _ => diracProb d w) := by
  intro φ _ t
  have hμ : (diracProb d w : Measure (SSphere d)) = Measure.dirac w := rfl
  rw [hμ, integral_dirac, MFVel_diracProb_self, inner_zero_right]
  exact hasDerivAt_const t _

/-- Both zero velocity and stationarity have genuine instances on the circle. -/
example : MFVel 2 1000 (diracProb 2 (basePoint 1)) (basePoint 1 : EucSpace 2) = 0 ∧
    meanFieldPDE 2 1000 (fun _ => diracProb 2 (basePoint 1)) :=
  ⟨MFVel_diracProb_self 2 1000 (basePoint 1), meanFieldPDE_dirac 2 1000 (basePoint 1)⟩

/-- On `𝕊^0` a unit vector spans the line: `v = ⟨x, v⟩ x`. -/
theorem eq_inner_smul_of_dim_one (x v : EucSpace 1) (hx : ‖x‖ = 1) :
    v = inner (𝕜 := ℝ) x v • x := by
  have h2 := EuclideanSpace.real_norm_sq_eq x
  rw [hx, Fin.sum_univ_one] at h2
  ext i
  fin_cases i
  simp [EuclideanSpace.inner_eq_star_dotProduct, dotProduct]
  have : x.ofLp 0 * x.ofLp 0 = 1 := by nlinarith [h2]
  rw [mul_assoc, this, mul_one]

/-- On `𝕊^0` the mean-field velocity of a Dirac mass vanishes everywhere. -/
theorem MFVel_diracProb_dim_one (β : ℝ) (p : SSphere 1) (x : EucSpace 1) (hx : ‖x‖ = 1) :
    MFVel 1 β (diracProb 1 p) x = 0 := by
  have hμ : ((diracProb 1 p : ProbSphere 1) : Measure (SSphere 1)) = Measure.dirac p := rfl
  rw [MFVel, hμ, integral_dirac]
  have : proj 1 x (p : EucSpace 1) = 0 := by
    rw [proj]
    exact sub_eq_zero.mpr (eq_inner_smul_of_dim_one x _ hx)
  rw [this, smul_zero]

/-- The constant curve at `δ_p` solves `eq: mean.field.pde` on `𝕊^0`. -/
theorem meanFieldPDE_const_diracProb (β : ℝ) (p : SSphere 1) :
    meanFieldPDE 1 β (fun _ => diracProb 1 p) := by
  intro φ _ t
  have hμ : ((diracProb 1 p : ProbSphere 1) : Measure (SSphere 1)) = Measure.dirac p := rfl
  rw [hμ, integral_dirac,
    MFVel_diracProb_dim_one β p _ (mem_sphere_zero_iff_norm.mp p.2), inner_zero_right]
  exact hasDerivAt_const t _

/-- The identity is the characteristic flow of the constant curve at `δ_p`. -/
theorem isMFFlow_id_diracProb (β : ℝ) (p : SSphere 1) :
    IsMFFlow 1 β (fun _ => diracProb 1 p) (fun _ x => x) :=
  ⟨fun _ => measurable_id, fun _ => rfl, fun x t => by
    rw [MFVel_diracProb_dim_one β p _ (mem_sphere_zero_iff_norm.mp x.2)]
    exact hasDerivAt_const t _⟩

/-- On `𝕊^0` a cap of width `ε < 2` is its centre alone. -/
theorem eq_of_mem_sphericalCap_dim_one (p y : SSphere 1) (ε : ℝ) (hε : ε < 2)
    (hy : y ∈ sphericalCap 1 p ε) : y = p := by
  have hp : ‖(p : EucSpace 1)‖ = 1 := mem_sphere_zero_iff_norm.mp p.2
  have hy1 : ‖(y : EucSpace 1)‖ = 1 := mem_sphere_zero_iff_norm.mp y.2
  have hdecomp := eq_inner_smul_of_dim_one (p : EucSpace 1) (y : EucSpace 1) hp
  have hc : |inner (𝕜 := ℝ) (p : EucSpace 1) (y : EucSpace 1)| = 1 := by
    have := congrArg norm hdecomp
    rwa [norm_smul, hp, mul_one, Real.norm_eq_abs, hy1, eq_comm] at this
  have hge : 1 - ε ≤ inner (𝕜 := ℝ) (p : EucSpace 1) (y : EucSpace 1) := by
    have := hy; simp only [sphericalCap, Set.mem_ofPred_eq] at this
    rwa [real_inner_comm] at this
  have h1 : inner (𝕜 := ℝ) (p : EucSpace 1) (y : EucSpace 1) = 1 := by
    rcases abs_eq (zero_le_one) |>.mp hc with h | h
    · exact h
    · linarith
  apply Subtype.ext
  rw [hdecomp, h1, one_smul]

/-- `p` is the minimiser of the cap around itself, at every time, under the
identity flow. -/
theorem isCapArgmin_id_dim_one (p : SSphere 1) (ε : ℝ) (hε0 : 0 ≤ ε) (hε : ε < 2) :
    IsCapArgmin 1 (fun _ x => x) p ε (fun _ => p) := by
  have hpp : inner (𝕜 := ℝ) (p : EucSpace 1) (p : EucSpace 1) = 1 := by
    rw [real_inner_self_eq_norm_mul_norm, mem_sphere_zero_iff_norm.mp p.2]; ring
  refine fun _ => ⟨⟨p, ?_, rfl⟩, ?_⟩
  · simp only [sphericalCap, Set.mem_ofPred_eq]; linarith
  · rintro y ⟨y', hy', rfl⟩
    rw [eq_of_mem_sphericalCap_dim_one p y' ε hε hy']

/-- One cap around `p` at `β = 1000`, `ε = 1/32` is separated:
`8ε < γ(β) = 1 - α - 8ε - β⁻¹ log(2k²/ε)`, with `α = 0` since there is no
second cap to be close to. -/
theorem separated_one_cap (p : SSphere 1) :
    8 * (1 / 32 : ℝ) < γβ 1 1000 (αDist 1 1 (fun _ => p) (1 / 32)) (1 / 32) := by
  have hα : αDist 1 1 (fun _ => p) (1 / 32) = 0 := by
    have : { c : ℝ | ∃ i j : Idx 1, i ≠ j ∧
        ∃ x ∈ sphericalCap 1 ((fun _ => p) i) (2 * (1 / 32)),
          ∃ y ∈ sphericalCap 1 ((fun _ => p) j) (2 * (1 / 32)),
            c = inner (𝕜 := ℝ) ((x : EucSpace 1)) ((y : EucSpace 1)) } = ∅ := by
      ext c
      simp only [Set.mem_ofPred_eq, Set.mem_empty_iff_false, iff_false]
      rintro ⟨i, j, hij, -⟩
      exact hij (Subsingleton.elim i j)
    rw [αDist, this, Real.sSup_empty]
  rw [γβ, hα]
  have hlog := Real.log_le_sub_one_of_pos (by norm_num : (0 : ℝ) < 2 * ((1 : ℕ) : ℝ) ^ 2 / (1 / 32))
  norm_num at hlog ⊢
  linarith

/-- The hypotheses of the cap-exit claim and the refuted equation `eq: v.small` are satisfiable: on
`𝕊^0`, one cap around `p`, `ν = μ_0 = δ_p` at rest, the identity flow with
`x ≡ p` its minimising selection, `β = 1000`, `ε = 1/32`, and `c = λ` the
midpoint of `(8ε, γ)` (`Metastability.MeanFieldStatic`). -/
example : let p := basePoint 0
    let γ := γβ 1 1000 (αDist 1 1 (fun _ => p) (1 / 32)) (1 / 32)
    (1 : ℝ) < 1000 ∧ (0 : ℝ) < 1 / 32 ∧ (1 / 32 : ℝ) < 1 / 16 ∧ 0 < 1 ∧
      (∀ q : Idx 1, ((fun _ => diracProb 1 p) q : Measure (SSphere 1)).support
        ⊆ sphericalCap 1 ((fun _ => p) q) (1 / 32)) ∧
      ((diracProb 1 p : ProbSphere 1) : Measure (SSphere 1))
        = (((1 : ℕ) : ℝ)⁻¹).toNNReal •
            ∑ q : Idx 1, ((fun _ => diracProb 1 p) q : Measure (SSphere 1)) ∧
      8 * (1 / 32 : ℝ) < γ ∧
      meanFieldPDE 1 1000 (fun _ => diracProb 1 p) ∧
      IsMFFlow 1 1000 (fun _ => diracProb 1 p) (fun _ x => x) ∧
      IsCapArgmin 1 (fun _ x => x) p (1 / 32) (fun _ => p) ∧
      0 < (8 * (1 / 32) + γ) / 2 ∧
      8 * (1 / 32 : ℝ) < (8 * (1 / 32) + γ) / 2 ∧ (8 * (1 / 32) + γ) / 2 < γ := by
  intro p γ
  have hsep : 8 * (1 / 32 : ℝ) < γ := separated_one_cap p
  refine ⟨by norm_num, by norm_num, by norm_num, one_pos, fun _ y hy => ?_, by simp, hsep,
    meanFieldPDE_const_diracProb 1000 p, isMFFlow_id_diracProb 1000 p,
    isCapArgmin_id_dim_one p _ (by norm_num) (by norm_num), by linarith, by linarith,
    by linarith⟩
  rw [Interpolation.eq_of_mem_support_dirac hy]
  simp only [sphericalCap, Set.mem_ofPred_eq, real_inner_self_eq_norm_mul_norm,
    mem_sphere_zero_iff_norm.mp p.2]
  norm_num

end Metastability
end Transformer
