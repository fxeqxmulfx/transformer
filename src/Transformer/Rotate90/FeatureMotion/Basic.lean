module

public import Transformer.Grokking.GradientEvidence.ForecastSecant
public import Mathlib.Analysis.InnerProductSpace.PiL2

/-!
# Checkpoint bounds for moving linear layers and RMSNorm

Source: `review/flat_spectrum.md`, *Full encoder motion*, EM1–EM2.
Only old inputs, old operator norms and displacement caps occur in the
computed bounds. RMSNorm retains epsilon and the actual coordinate count.
-/

@[expose] public section
noncomputable section
namespace Transformer.Rotate90.FeatureMotion

open Grokking.GradientEvidence Landscape.Neural
open scoped BigOperators

variable {E F : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
  [NormedAddCommGroup F] [NormedSpace ℝ F]

/-- EM1: finite linear-layer motion, including weight/input interaction. -/
theorem linear_motion (A A' : E →L[ℝ] F) (x x' : E) (R r : ℝ)
    (hA : ‖A' - A‖ ≤ R) (hx : ‖x' - x‖ ≤ r) :
    ‖A' x' - A x‖ ≤ (‖A‖ + R) * r + R * ‖x‖ := by
  have hR := (norm_nonneg (A' - A)).trans hA
  have hr := (norm_nonneg (x' - x)).trans hx
  have he : A' x' - A x = A (x' - x) + (A' - A) x + (A' - A) (x' - x) := by
    simp only [map_sub, sub_apply]
    abel
  rw [he]
  have h1 := (A.le_opNorm (x' - x)).trans
    (mul_le_mul_of_nonneg_left hx (norm_nonneg A))
  have h2 := ((A' - A).le_opNorm x).trans
    (mul_le_mul_of_nonneg_right hA (norm_nonneg x))
  have h3 := ((A' - A).le_opNorm (x' - x)).trans
    (mul_le_mul hA hx (norm_nonneg _) hR)
  have ht := (norm_add_le (A (x' - x) + (A' - A) x) ((A' - A) (x' - x))).trans
    (add_le_add (norm_add_le _ _) le_rfl)
  nlinarith

example := linear_motion (0 : ℝ →L[ℝ] ℝ)
  (ContinuousLinearMap.id ℝ ℝ) 2 3 1 1 (by norm_num) (by norm_num)

/-- EM1: a certified checkpoint operator cap, such as the Frobenius or
Gram bound FT2, can replace the exact operator norm in the constructor. -/
theorem linear_motion_of_bound (A A' : E →L[ℝ] F) (x x' : E) (L R r : ℝ)
    (hL : ‖A‖ ≤ L) (hA : ‖A' - A‖ ≤ R) (hx : ‖x' - x‖ ≤ r) :
    ‖A' x' - A x‖ ≤ (L + R) * r + R * ‖x‖ := by
  exact (linear_motion A A' x x' R r hA hx).trans
    (add_le_add (mul_le_mul_of_nonneg_right (add_le_add hL le_rfl)
      ((norm_nonneg _).trans hx)) le_rfl)

example := linear_motion_of_bound (0 : ℝ →L[ℝ] ℝ)
  (ContinuousLinearMap.id ℝ ℝ) 2 3 0 1 1 (by norm_num) (by norm_num) (by norm_num)

variable {I : Type*} [Fintype I]

/-- EM2: Euclidean form of source RMSNorm, with its actual width. -/
def rms (eps : ℝ) (x : EuclideanSpace ℝ I) : EuclideanSpace ℝ I :=
  Real.sqrt (Fintype.card I) • softNormalize (Fintype.card I * eps) x

/-- EM2: anchored propagation coefficient, computable from the old input. -/
def rmsGain (eps : ℝ) (x : EuclideanSpace ℝ I) : ℝ :=
  2 * Real.sqrt (Fintype.card I) / softNorm (Fintype.card I * eps) x

variable [Nonempty I]

/-- EM2: the Euclidean expression is the source coordinate RMSNorm;
no normalization layer or epsilon has been removed. -/
theorem rms_eq_source (eps : ℝ) (x : EuclideanSpace ℝ I) :
    rms eps x = WithLp.toLp 2 (rmsNorm eps x) := by
  have hd : (0 : ℝ) < Fintype.card I := by exact_mod_cast Fintype.card_pos
  have hs : 0 < Real.sqrt (Fintype.card I) := Real.sqrt_pos.mpr hd
  have he : ‖x‖ ^ 2 + Fintype.card I * eps =
      (Fintype.card I : ℝ) * (eps + meanSquare x) := by
    rw [meanSquare, ← EuclideanSpace.real_norm_sq_eq]
    field_simp
    ring
  ext i
  simp only [rms, softNormalize, softNorm, he, Real.sqrt_mul hd.le, smul_smul]
  change (Real.sqrt (Fintype.card I) *
    (Real.sqrt (Fintype.card I) * Real.sqrt (eps + meanSquare x))⁻¹) * x i = _
  simp [rmsNorm, hs.ne', div_eq_mul_inv, mul_comm]

/-- EM2: source normalization bounds every feature norm by sqrt(width). -/
theorem rms_norm_le (eps : ℝ) (heps : 0 < eps) (x : EuclideanSpace ℝ I) :
    ‖rms eps x‖ ≤ Real.sqrt (Fintype.card I) := by
  have hd : (0 : ℝ) < Fintype.card I := by exact_mod_cast Fintype.card_pos
  have hs := softNorm_pos (Fintype.card I * eps) (mul_pos hd heps) x
  have hn := norm_le_softNorm (Fintype.card I * eps) (mul_pos hd heps).le x
  rw [rms, softNormalize, norm_smul, norm_smul, Real.norm_eq_abs, Real.norm_eq_abs,
    abs_of_nonneg (Real.sqrt_nonneg _), abs_of_pos (inv_pos.mpr hs)]
  have h : (softNorm (Fintype.card I * eps) x)⁻¹ * ‖x‖ ≤ 1 := by
    rw [← div_eq_inv_mul, div_le_one hs]
    exact hn
  nlinarith [Real.sqrt_nonneg (Fintype.card I)]

/-- EM2: full-ball RMS motion uses the old denominator only. -/
theorem rms_motion (eps : ℝ) (heps : 0 < eps) (x x' : EuclideanSpace ℝ I)
    (r : ℝ) (hx : ‖x' - x‖ ≤ r) : ‖rms eps x' - rms eps x‖ ≤ rmsGain eps x * r := by
  have hd : (0 : ℝ) < Fintype.card I := by exact_mod_cast Fintype.card_pos
  exact (scaled_softNormalize_error _ _ (Real.sqrt_nonneg _) (mul_pos hd heps) x x').trans
    (mul_le_mul_of_nonneg_left hx (by
      have := softNorm_pos _ (mul_pos hd heps) x
      positivity))

/-- EM2: a second, globally valid cap prevents normalization radii from
exceeding the diameter of its range. -/
theorem rms_diameter (eps : ℝ) (heps : 0 < eps) (x x' : EuclideanSpace ℝ I) :
    ‖rms eps x' - rms eps x‖ ≤ 2 * Real.sqrt (Fintype.card I) := by
  have h := (norm_sub_le (rms eps x') (rms eps x)).trans
    (add_le_add (rms_norm_le eps heps x') (rms_norm_le eps heps x))
  linarith

/-- EM2: intersect the range diameter, global anchored bound and local
secant bound. Every term is fixed before the later input is observed. -/
def rmsRadius (eps : ℝ) (x : EuclideanSpace ℝ I) (r : ℝ) : ℝ :=
  let s := softNorm (Fintype.card I * eps) x
  min (2 * Real.sqrt (Fintype.card I)) (min (rmsGain eps x * r)
    (if r < s then Real.sqrt (Fintype.card I) * r / Real.sqrt (s * (s - r))
     else rmsGain eps x * r))

/-- EM2: the intersected radius encloses the actual normalized motion. -/
theorem rms_motion_radius (eps : ℝ) (heps : 0 < eps) (x x' : EuclideanSpace ℝ I)
    (r : ℝ) (hx : ‖x' - x‖ ≤ r) : ‖rms eps x' - rms eps x‖ ≤ rmsRadius eps x r := by
  have hd : (0 : ℝ) < Fintype.card I := by exact_mod_cast Fintype.card_pos
  apply le_min (rms_diameter eps heps x x')
  apply le_min (rms_motion eps heps x x' r hx)
  split
  · exact scaled_softNormalize_local_error _ _ (Real.sqrt_nonneg _) (mul_pos hd heps)
      x x' r hx (by assumption)
  · exact rms_motion eps heps x x' r hx

/-- EM2: nonnegative input caps produce nonnegative output caps. -/
theorem rmsRadius_nonneg (eps : ℝ) (heps : 0 < eps) (x : EuclideanSpace ℝ I)
    (r : ℝ) (hr : 0 ≤ r) : 0 ≤ rmsRadius eps x r := by
  simpa only [sub_self, norm_zero] using rms_motion_radius eps heps x x r (by simpa using hr)

example := rms_eq_source 1 (WithLp.toLp 2 (fun _ : Fin 2 => (2 : ℝ)))
example := rms_norm_le 1 (by norm_num) (WithLp.toLp 2 (fun _ : Fin 2 => (2 : ℝ)))
example := rms_motion 1 (by norm_num) (WithLp.toLp 2 (fun _ : Fin 1 => (2 : ℝ)))
  (WithLp.toLp 2 (fun _ => 3)) 1 (by norm_num [EuclideanSpace.norm_eq])
example := rms_diameter 1 (by norm_num) (WithLp.toLp 2 (fun _ : Fin 1 => (2 : ℝ)))
  (WithLp.toLp 2 (fun _ => -3))
example := rms_motion_radius 1 (by norm_num) (WithLp.toLp 2 (fun _ : Fin 1 => (2 : ℝ)))
  (WithLp.toLp 2 (fun _ => 3)) 1 (by norm_num [EuclideanSpace.norm_eq])
example := rmsRadius_nonneg 1 (by norm_num) (WithLp.toLp 2 (fun _ : Fin 1 => (2 : ℝ)))
  1 (by norm_num)

end Transformer.Rotate90.FeatureMotion
