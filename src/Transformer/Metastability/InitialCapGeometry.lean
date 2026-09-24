/-
# Quadratic radial-projection bound for Gaussian-mixture initialization

The distance between two unit vectors controls their inner-product deficit
quadratically. Combined with the existing radial-projection estimate, this
puts a projected sample in a cap of height `2δ²` when its scaled distance to
the centre is at most `δ`. The linear cap estimate in `InitialGeometry` loses
this square and is too weak for the mixture tail bound.

Source: arXiv:2410.06833v1, §4, proof of `prop: mixture.of.gaussians`.
-/

import Transformer.Metastability.InitialGeometry

open Real

namespace Transformer
namespace Metastability

/-- A sample within distance `δ` of a unit centre has radial projection in
the cap of height `2δ²`. The squared height follows from
`‖y-w‖² = 2 - 2⟨y,w⟩` for unit vectors.
Source: arXiv:2410.06833v1, §4, proof of `prop: mixture.of.gaussians`. -/
theorem mem_sphericalCap_of_radialProjection_sq_close (d : ℕ) (x : EucSpace d)
    (w y : SSphere d) (δ : ℝ) (hx : x ≠ 0)
    (hy : (y : EucSpace d) = ‖x‖⁻¹ • x)
    (hclose : ‖x - (w : EucSpace d)‖ ≤ δ) :
    y ∈ sphericalCap d w (2 * δ ^ 2) := by
  have hdist : ‖(y : EucSpace d) - (w : EucSpace d)‖ ≤ 2 * δ := by
    rw [hy]
    exact (norm_radialProjection_sub_le d x w hx).trans (by linarith)
  have hδ : 0 ≤ δ := (norm_nonneg _).trans hclose
  have hdist_sq : ‖(y : EucSpace d) - (w : EucSpace d)‖ ^ 2 ≤ 4 * δ ^ 2 := by
    nlinarith [norm_nonneg ((y : EucSpace d) - (w : EucSpace d))]
  have hyn : ‖(y : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp y.2
  have hwn : ‖(w : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp w.2
  have hsq := norm_sub_sq_real (y : EucSpace d) (w : EucSpace d)
  rw [hyn, hwn] at hsq
  change 1 - 2 * δ ^ 2 ≤ inner (𝕜 := ℝ) ((y : EucSpace d)) ((w : EucSpace d))
  nlinarith

/-- The squared-distance version of the cap criterion, with the cap height
itself on the right-hand side.
Source: arXiv:2410.06833v1, §4, proof of `prop: mixture.of.gaussians`. -/
theorem mem_sphericalCap_of_radialProjection_norm_sq_le (d : ℕ) (x : EucSpace d)
    (w y : SSphere d) (ε : ℝ) (hx : x ≠ 0)
    (hy : (y : EucSpace d) = ‖x‖⁻¹ • x)
    (hclose : 2 * ‖x - (w : EucSpace d)‖ ^ 2 ≤ ε) :
    y ∈ sphericalCap d w ε := by
  have hε : 0 ≤ ε / 2 := by nlinarith [sq_nonneg ‖x - (w : EucSpace d)‖]
  have hroot : ‖x - (w : EucSpace d)‖ ≤ Real.sqrt (ε / 2) :=
    Real.le_sqrt_of_sq_le (by linarith)
  have hcap := mem_sphericalCap_of_radialProjection_sq_close d x w y
    (Real.sqrt (ε / 2)) hx hy hroot
  rwa [Real.sq_sqrt hε, show 2 * (ε / 2) = ε by ring] at hcap

/-- If each nonzero sample is quadratically close to some unit centre, its
radial projection is a separated configuration.
Source: arXiv:2410.06833v1, §4, `prop: mixture.of.gaussians`. -/
theorem projectedSeparated_of_sq_near_centers (d n : ℕ) (β ε : ℝ) (r : ℕ)
    (w : Idx r → SSphere d) (hcent : isCentered d n β ε r w)
    (X : Idx n → EucSpace d) (hX : ∀ i, X i ≠ 0)
    (hnear : ∀ i, ∃ q : Idx r,
      2 * ‖X i - (w q : EucSpace d)‖ ^ 2 ≤ ε) :
    X ∈ projectedSeparated d n β ε := by
  let Y : SphereTuple d n := fun i =>
    ⟨‖X i‖⁻¹ • X i, mem_sphere_zero_iff_norm.mpr (by
      simpa using (norm_smul_inv_norm (𝕜 := ℝ) (hX i)))⟩
  refine ⟨Y, fun _ => rfl, ?_⟩
  refine ⟨r, hcent.1, w, ?_, hcent.2⟩
  intro i
  obtain ⟨q, hq⟩ := hnear i
  exact ⟨q, mem_sphericalCap_of_radialProjection_norm_sq_le d (X i) (w q) (Y i)
    ε (hX i) rfl hq⟩

/-- The quadratic criterion after scaling by the positive mixture-centre
radius `√r`.
Source: arXiv:2410.06833v1, §4, `prop: mixture.of.gaussians`. -/
theorem projectedSeparated_of_sq_near_scaled_centers (d n : ℕ) (β ε : ℝ) (r : ℕ)
    (hr : 0 < r) (w : Idx r → SSphere d) (hcent : isCentered d n β ε r w)
    (X : Idx n → EucSpace d) (hX : ∀ i, X i ≠ 0)
    (hnear : ∀ i, ∃ q : Idx r,
      2 * ‖(Real.sqrt (r : ℝ))⁻¹ • X i - (w q : EucSpace d)‖ ^ 2 ≤ ε) :
    X ∈ projectedSeparated d n β ε := by
  let a : ℝ := Real.sqrt (r : ℝ)
  have ha : 0 < a := Real.sqrt_pos.2 (by exact_mod_cast hr)
  let Z : Idx n → EucSpace d := fun i => a⁻¹ • X i
  have hrecover : (fun i => a • Z i) = X := by
    funext i
    dsimp [Z]
    rw [smul_smul, mul_inv_cancel₀ ha.ne', one_smul]
  have hZ : ∀ i, Z i ≠ 0 := by
    intro i hi
    apply hX i
    have hi' := congrFun hrecover i
    rw [hi, smul_zero] at hi'
    exact hi'.symm
  have hsep : Z ∈ projectedSeparated d n β ε :=
    projectedSeparated_of_sq_near_centers d n β ε r w hcent Z hZ hnear
  have hscaled := (projectedSeparated_smul_pos_iff d n β ε a ha Z).2 hsep
  rw [hrecover] at hscaled
  exact hscaled

/-- At a unit centre, zero displacement satisfies the quadratic bound. -/
example (w : SSphere 1) :
    ‖(w : EucSpace 1) - (w : EucSpace 1)‖ ≤ (0 : ℝ) := by simp

end Metastability
end Transformer
