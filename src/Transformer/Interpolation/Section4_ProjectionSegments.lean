/-
# A uniformly active gate along an exact projection transfer

ArXiv:2411.04551v3, §4, the local point transfer in
`prop: interpolation.neural.ode`, using the tangent field of §3.1.
The hyperbolic projection denominator is a nonnegative linear
combination of two convex exponential functions. It is therefore
convex. If it is one at both endpoints of a segment, it is at most
one throughout that segment.

A gate direction perpendicular to the forcing axis has constant
numerator along the actual projection orbit. Its original height
therefore stays above its initial positive height on this segment.
Any smaller nonnegative threshold gives a uniformly positive actual
ReLU gate, which will permit an exact change to the gated clock.
-/

import Transformer.Interpolation.Section4_ProjectionTransfers
import Mathlib.Analysis.Convex.SpecificFunctions.Basic
import Mathlib.Analysis.Convex.Jensen

open Real MeasureTheory Set

namespace Transformer.Interpolation

variable {d : ℕ}

/-- The exact original projection denominator is convex in time.
Source auxiliary: arXiv:2411.04551v3, §3.1, the hyperbolic flow
used for local point matching in §4. -/
theorem convexOn_projectionDenom (a p : SSphere d) :
    ConvexOn ℝ univ (fun t => projectionDenom (a : EucSpace d) t p) := by
  let h := inner ℝ (a : EucSpace d) (p : EucSpace d)
  have hb : |h| ≤ 1 := by
    simpa only [mem_sphere_zero_iff_norm.mp a.2, mem_sphere_zero_iff_norm.mp p.2, mul_one] using
      abs_real_inner_le_norm (a : EucSpace d) (p : EucSpace d)
  have he : ConvexOn ℝ univ (fun t : ℝ => exp (-t)) := by
    simpa only [preimage_univ, Function.comp_def, LinearMap.neg_apply, LinearMap.id_apply] using
      convexOn_exp.comp_linearMap (-(LinearMap.id : ℝ →ₗ[ℝ] ℝ))
  have hp : 0 ≤ (1 + h) / 2 := by linarith [(abs_le.mp hb).1]
  have hm : 0 ≤ (1 - h) / 2 := by linarith [(abs_le.mp hb).2]
  have hf := (ConvexOn.smul hp convexOn_exp).add (ConvexOn.smul hm he)
  convert hf using 1
  funext t
  change projectionDenom (a : EucSpace d) t p =
    ((1 + h) / 2) * exp t + ((1 - h) / 2) * exp (-t)
  rw [projectionDenom, cosh_eq, sinh_eq]
  dsimp only [h]
  ring

/-- A projection transfer with denominator one at its terminal time
has denominator at most one throughout its actual time segment.
Source auxiliary: arXiv:2411.04551v3, §4, local point transfer. -/
theorem projectionDenom_le_one_on_transfer (a p : SSphere d) (t : ℝ)
    (hend : projectionDenom (a : EucSpace d) t p = 1) (s : ℝ) (hs : s ∈ Icc 0 t) :
    projectionDenom (a : EucSpace d) s p ≤ 1 := by
  have hh := (convexOn_projectionDenom a p).le_max_of_mem_Icc
    (mem_univ 0) (mem_univ t) hs
  simpa only [projectionDenom_zero, hend, max_self] using hh

/-- An orthogonal gate direction has its original height divided by
exactly the original projection denominator. Source auxiliary:
arXiv:2411.04551v3, §4, the plane of the point transfer. -/
theorem inner_projectionFlow_orthogonal (a p : SSphere d) (m : EucSpace d)
    (ham : inner ℝ (a : EucSpace d) m = 0) (s : ℝ) :
    inner ℝ (sphereProjectionFlow a s p : EucSpace d) m =
      inner ℝ (p : EucSpace d) m / projectionDenom (a : EucSpace d) s p := by
  change inner ℝ ((projectionDenom (a : EucSpace d) s p)⁻¹ •
    projectionNumerator (a : EucSpace d) s p) m = _
  rw [real_inner_smul_left, projectionNumerator, inner_add_left,
    real_inner_smul_left, ham, mul_zero, add_zero, div_eq_mul_inv, mul_comm]

/-- Every nonnegative orthogonal gate height stays above its initial
value on the exact transfer segment. Source auxiliary:
arXiv:2411.04551v3, §4, the active region of the local point transfer. -/
theorem projection_transfer_height_ge (a p : SSphere d) (m : EucSpace d)
    (ham : inner ℝ (a : EucSpace d) m = 0)
    (hp : 0 ≤ inner ℝ (p : EucSpace d) m) (t : ℝ)
    (hend : projectionDenom (a : EucSpace d) t p = 1) (s : ℝ) (hs : s ∈ Icc 0 t) :
    inner ℝ (p : EucSpace d) m ≤ inner ℝ (sphereProjectionFlow a s p : EucSpace d) m := by
  rw [inner_projectionFlow_orthogonal a p m ham s]
  apply (le_div_iff₀ (projectionDenom_pos _ _ (mem_sphere_zero_iff_norm.mp a.2)
    (mem_sphere_zero_iff_norm.mp p.2) s)).mpr
  simpa only [mul_one] using mul_le_mul_of_nonneg_left
    (projectionDenom_le_one_on_transfer a p t hend s hs) hp

/-- A threshold strictly below the initial height provides one fixed
positive lower bound for the original ReLU gate throughout the whole
transfer segment. Source auxiliary: arXiv:2411.04551v3, §4,
local control by the original one-neuron perceptron. -/
theorem projection_transfer_gate_floor (a p : SSphere d) (m : EucSpace d)
    (ham : inner ℝ (a : EucSpace d) m = 0) (c : ℝ) (hc : 0 ≤ c)
    (hcp : c < inner ℝ (p : EucSpace d) m) (t : ℝ)
    (hend : projectionDenom (a : EucSpace d) t p = 1) :
    0 < inner ℝ (p : EucSpace d) m - c ∧ ∀ s ∈ Icc 0 t,
      inner ℝ (p : EucSpace d) m - c ≤
        max (-c - inner ℝ (sphereProjectionFlow a s p : EucSpace d) (-m)) 0 := by
  refine ⟨sub_pos.mpr hcp, ?_⟩
  intro s hs
  have hh := projection_transfer_height_ge a p m ham (hc.trans hcp.le) t hend s hs
  have hm := le_max_left (-c - inner ℝ (sphereProjectionFlow a s p : EucSpace d) (-m)) 0
  rw [inner_neg_right] at hm ⊢
  linarith

/-- The same actual gate is bounded by its original input and bias
norms, at every time on the whole unit orbit. Source auxiliary:
arXiv:2411.04551v3, §1, the perceptron in `eq: vf`, and §4,
the finite positive clock of a local point transfer. -/
theorem projection_gate_upper (a p : SSphere d) (m : EucSpace d) (c s : ℝ) :
    max (-c - inner ℝ (sphereProjectionFlow a s p : EucSpace d) (-m)) 0 ≤ |c| + ‖m‖ := by
  have hi := abs_real_inner_le_norm (sphereProjectionFlow a s p : EucSpace d) m
  rw [mem_sphere_zero_iff_norm.mp (sphereProjectionFlow a s p).2, one_mul] at hi
  rw [inner_neg_right]
  apply max_le _ (by positivity)
  linarith [neg_abs_le c, le_abs_self (inner ℝ (sphereProjectionFlow a s p : EucSpace d) m)]

/-- The actual ReLU gate along the original projection orbit is
continuous in intrinsic time. Source auxiliary:
arXiv:2411.04551v3, §4, the local transfer's change of clock. -/
theorem continuous_projection_gate (a p : SSphere d) (m : EucSpace d) (c : ℝ) :
    Continuous (fun s => max (-c - inner ℝ (sphereProjectionFlow a s p : EucSpace d) (-m)) 0) := by
  have hc : Continuous (fun s => (sphereProjectionFlow a s p : EucSpace d)) :=
    continuous_subtype_val.comp ((continuous_sphereProjectionFlow a).comp
      (continuous_id.prodMk continuous_const))
  exact (continuous_const.sub (hc.inner continuous_const)).max continuous_const

/-- A genuine positive-time transfer between orthogonal unit atoms
has a nonzero midpoint gate, a strictly positive margin, and an
interior segment time satisfying every hypothesis jointly. -/
example : ∃ (a p : SSphere 2) (m : EucSpace 2) (t c : ℝ),
    0 < t ∧ projectionDenom (a : EucSpace 2) t p = 1 ∧
    inner ℝ (a : EucSpace 2) m = 0 ∧ 0 ≤ inner ℝ (p : EucSpace 2) m ∧
    0 ≤ c ∧ c < inner ℝ (p : EucSpace 2) m ∧ t / 2 ∈ Icc (0 : ℝ) t := by
  let p : SSphere 2 := ⟨PiLp.single 2 (0 : Fin 2) (1 : ℝ), by simp⟩
  let q : SSphere 2 := ⟨PiLp.single 2 (1 : Fin 2) (1 : ℝ), by simp⟩
  have hpq : p ≠ q := by
    intro he
    have hh := congrArg (fun z : SSphere 2 => (z : EucSpace 2) (0 : Fin 2)) he
    norm_num [p, q] at hh
  have hanti : q ≠ antipode 2 p := by
    intro he
    have hh := congrArg (fun z : SSphere 2 => (z : EucSpace 2) (1 : Fin 2)) he
    norm_num [p, q, antipode] at hh
  obtain ⟨a, t, ht, hD, _, ham⟩ := exists_exact_projection_transfer p q hpq hanti
  let m : EucSpace 2 := (p : EucSpace 2) + (q : EucSpace 2)
  have hmp : inner ℝ (p : EucSpace 2) m = 1 := by
    simp [m, p, q, inner_add_right, EuclideanSpace.inner_single_left]
  refine ⟨a, p, m, t, 1 / 2, ht, hD, ham, ?_, by norm_num, ?_, ?_⟩
  · rw [hmp]
    exact zero_le_one
  · rw [hmp]
    norm_num
  · constructor <;> linarith

end Transformer.Interpolation
