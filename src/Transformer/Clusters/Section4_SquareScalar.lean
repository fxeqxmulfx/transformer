module

public import Transformer.Landscape.GlobalFlow.Existence
public import Mathlib.Analysis.SpecialFunctions.ExpDeriv

/-!
# A scalar equation for four symmetric planar tokens

To test the codimension conjecture `c:codim` of arXiv:2305.05465v6,
§4, put four tokens at the corners `(±u, ±u)` and take `Q = K = V = I₂`.
The attention weights factor over the two coordinate signs. Each
coordinate then solves `u' = u tanh(u²)`, written here using only
real exponentials. This is the original, unrescaled token amplitude.

The scalar field is smooth and has linear growth, so the proved
global ODE existence theorem applies at every initial amplitude.
Two elementary drift bounds control its normalized amplitude:
`u' ≥ u - 1` everywhere, and `u' ≤ u` when `u ≥ 0`.

The first bound follows from `exp(u²) ≥ 1 + u²` and
`2u ≤ 1 + u²`; it does not need an asymptotic approximation.
These facts will give a strictly positive limit after multiplying
by `exp(-t)`, and hence four distinct limiting corners.
The construction tests the source's universal conjecture; it is
not a changed definition of its attention dynamics.
-/

@[expose] public section

open Real Filter Topology
namespace Transformer.Clusters

/-- The actual scalar drift of a coordinate of the four-corner configuration.
Source: arXiv:2305.05465v6, §4 `c:codim` and `eq:trans_dyn`,
evaluated at four equally spaced sign combinations in dimension two. -/
noncomputable def squareScalarDrift (u : ℝ) : ℝ :=
  u * (Real.exp (u ^ 2) - Real.exp (-u ^ 2)) /
    (Real.exp (u ^ 2) + Real.exp (-u ^ 2))

/-- The four-corner partition denominator is strictly positive.
Source: arXiv:2305.05465v6, `eq:trans_dyn`, its literal exponential weights. -/
theorem squareScalar_denominator_pos (u : ℝ) :
    0 < Real.exp (u ^ 2) + Real.exp (-u ^ 2) := by positivity

/-- The four-corner scalar field is smooth at every order.
Source: arXiv:2305.05465v6, §4 `c:codim`, the scalar specialization
of the smooth field in `p:wellposedparticles`. -/
theorem contDiff_squareScalarDrift : ContDiff ℝ ⊤ squareScalarDrift := by
  unfold squareScalarDrift
  have hnum : ContDiff ℝ ⊤ (fun u : ℝ => u * (Real.exp (u ^ 2) - Real.exp (-u ^ 2))) := by fun_prop
  have hden : ContDiff ℝ ⊤ (fun u : ℝ => Real.exp (u ^ 2) + Real.exp (-u ^ 2)) := by fun_prop
  exact hnum.fun_div hden (fun u => ne_of_gt (squareScalar_denominator_pos u))

/-- The actual four-corner scalar field has the linear growth bound needed for global existence.
Source: arXiv:2305.05465v6, `p:wellposedparticles`,
the attention average specialized to the planar square. -/
theorem norm_squareScalarDrift_le (u : ℝ) : ‖squareScalarDrift u‖ ≤ ‖u‖ := by
  have hx := Real.exp_pos (u ^ 2)
  have hy := Real.exp_pos (-u ^ 2)
  have hden := squareScalar_denominator_pos u
  have hnum : |Real.exp (u ^ 2) - Real.exp (-u ^ 2)| ≤
      Real.exp (u ^ 2) + Real.exp (-u ^ 2) := by
    exact abs_le.mpr ⟨by linarith, by linarith⟩
  have hratio : |Real.exp (u ^ 2) - Real.exp (-u ^ 2)| /
      (Real.exp (u ^ 2) + Real.exp (-u ^ 2)) ≤ 1 := (div_le_one hden).mpr hnum
  calc
    _ = |u| * (|Real.exp (u ^ 2) - Real.exp (-u ^ 2)| /
        (Real.exp (u ^ 2) + Real.exp (-u ^ 2))) := by
      rw [squareScalarDrift, Real.norm_eq_abs, abs_div, abs_mul, abs_of_pos hden]
      ring
    _ ≤ |u| * 1 := mul_le_mul_of_nonneg_left hratio (abs_nonneg u)
    _ = _ := by rw [mul_one, Real.norm_eq_abs]

/-- The square amplitude grows at least as fast as the affine field `u - 1`.
Source: arXiv:2305.05465v6, §4 `c:codim`, the lower amplitude
comparison for the four-corner test of its universal conclusion. -/
theorem sub_one_le_squareScalarDrift (u : ℝ) : u - 1 ≤ squareScalarDrift u := by
  have hx : u ^ 2 + 1 ≤ Real.exp (u ^ 2) := Real.add_one_le_exp _
  have hy0 := (Real.exp_pos (-u ^ 2)).le
  have hy1 : Real.exp (-u ^ 2) ≤ 1 := Real.exp_le_one_iff.mpr (by nlinarith [sq_nonneg u])
  have hp : 2 * u ≤ u ^ 2 + 1 := by nlinarith [sq_nonneg (u - 1)]
  have hxy : 2 * u * Real.exp (-u ^ 2) ≤ Real.exp (u ^ 2) + Real.exp (-u ^ 2) := by
    calc
      _ ≤ (u ^ 2 + 1) * Real.exp (-u ^ 2) := mul_le_mul_of_nonneg_right hp hy0
      _ ≤ (u ^ 2 + 1) * 1 := mul_le_mul_of_nonneg_left hy1 (by positivity)
      _ ≤ _ := by linarith
  unfold squareScalarDrift
  apply (le_div_iff₀ (squareScalar_denominator_pos u)).mpr
  nlinarith

/-- A positive square amplitude grows no faster than the linear field `u`.
Source: arXiv:2305.05465v6, `eq:trans_dyn`, the
convex attention average of the four coordinate signs. -/
theorem squareScalarDrift_le_self {u : ℝ} (hu : 0 ≤ u) : squareScalarDrift u ≤ u := by
  unfold squareScalarDrift
  apply (div_le_iff₀ (squareScalar_denominator_pos u)).mpr
  nlinarith [mul_nonneg hu (Real.exp_pos (-u ^ 2)).le]

example : (0 : ℝ) ≤ 2 := by norm_num

/-- Every positive square amplitude has strictly positive outward drift.
Source: arXiv:2305.05465v6, `eq:trans_dyn`, the
imbalance between the matching and opposite coordinate signs. -/
theorem squareScalarDrift_pos {u : ℝ} (hu : 0 < u) : 0 < squareScalarDrift u := by
  unfold squareScalarDrift
  apply div_pos _ (squareScalar_denominator_pos u)
  apply mul_pos hu
  exact sub_pos.mpr (Real.exp_lt_exp.mpr (by nlinarith [mul_pos hu hu]))

example : (0 : ℝ) < 2 := by norm_num

/-- The scalar field vanishes at the collapsed square.
Source: arXiv:2305.05465v6, `eq:trans_dyn`, its zero configuration. -/
theorem squareScalarDrift_zero : squareScalarDrift 0 = 0 := by
  simp only [squareScalarDrift, zero_mul, zero_div]

/-- Every initial amplitude has a genuine scalar solution defined at all real times.
Source: arXiv:2305.05465v6, `p:wellposedparticles`, applied
to the actual linear-growth field of the four-corner configuration. -/
theorem exists_squareScalar_solution (u₀ : ℝ) :
    ∃ u : ℝ → ℝ, u 0 = u₀ ∧ ∀ t : ℝ, HasDerivAt u (squareScalarDrift (u t)) t := by
  apply Landscape.GlobalFlow.exists_global (contDiff_squareScalarDrift.of_le (by simp)).locallyLipschitz
    zero_le_one (C := 1)
  intro u
  simpa only [one_mul] using norm_squareScalarDrift_le u

/-- An amplitude started at two dominates `1 + exp t` at nonnegative times.
Source: arXiv:2305.05465v6, §4 `c:codim`, the actual
four-corner amplitude comparison obtained from the affine drift bound. -/
theorem one_add_exp_le_squareScalar_solution (u : ℝ → ℝ)
    (hu : ∀ t : ℝ, HasDerivAt u (squareScalarDrift (u t)) t) (h0 : u 0 = 2)
    {t : ℝ} (ht : 0 ≤ t) : 1 + Real.exp t ≤ u t := by
  let g := fun s => Real.exp (-s) * (u s - 1)
  have hD (s : ℝ) : HasDerivAt g
      (Real.exp (-s) * (squareScalarDrift (u s) - u s + 1)) s := by
    have he : HasDerivAt (fun r : ℝ => Real.exp (-r)) (-Real.exp (-s)) s := by
      simpa only [Pi.neg_apply, mul_neg, mul_one] using ((hasDerivAt_id' (x := s)).neg.exp)
    convert he.fun_mul ((hu s).sub_const 1) using 1
    ring
  have hmono : Monotone g := monotone_of_hasDerivAt_nonneg hD (fun s =>
    mul_nonneg (Real.exp_pos _).le (by linarith [sub_one_le_squareScalarDrift (u s)]))
  have hge : 1 ≤ Real.exp (-t) * (u t - 1) := by
    have hcomp := hmono ht
    simp [g, h0] at hcomp
    linarith
  have hexp : Real.exp t * Real.exp (-t) = 1 := by
    rw [← Real.exp_add, add_neg_cancel, Real.exp_zero]
  have hbound : Real.exp t ≤ u t - 1 := by
    calc
      _ = Real.exp t * 1 := (mul_one _).symm
      _ ≤ Real.exp t * (Real.exp (-t) * (u t - 1)) :=
        mul_le_mul_of_nonneg_left hge (Real.exp_pos _).le
      _ = _ := by rw [← mul_assoc, hexp, one_mul]
  linarith

/-- The comparison hypotheses hold for an actual global solution started at two. -/
example : ∃ u : ℝ → ℝ, u 0 = 2 ∧
    (∀ t : ℝ, HasDerivAt u (squareScalarDrift (u t)) t) ∧ (0 : ℝ) ≤ 1 := by
  obtain ⟨u, h0, hu⟩ := exists_squareScalar_solution 2
  exact ⟨u, h0, hu, zero_le_one⟩

end Transformer.Clusters
