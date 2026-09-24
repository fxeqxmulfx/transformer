/-
# Geometry of radial projection for Gaussian-mixture initialization

The deterministic step of `prop: mixture.of.gaussians`: a sample close to a
unit centre remains in a spherical cap after radial projection. The
probabilistic concentration estimate is stated separately in `Initial`.

Source: arXiv:2410.06833v1, §4, `prop: mixture.of.gaussians`.
-/

import Transformer.Metastability.Basic

open Real

namespace Transformer
namespace Metastability

variable (d n : ℕ)

/-- **Definition (d: separated_mixtures).**

`(w_1,…,w_r)` is `(β, ε)`-*centered* (for a sample of size `n`) if there are
at most `n` centres and the caps `𝒮_q(ε)` around them satisfy `eq: gamma` of
`hyp: init`:

  `γ(β) = 1 - α(ε) - 8 ε - β⁻¹ log(2 n² / ε) > 0`.

This is `isSeparated` with the membership clause dropped: it constrains the
centres alone, not the sample. Source: arXiv:2410.06833v1, §4. -/
def isCentered
    (β ε : ℝ) (r : ℕ) (w : Idx r → SSphere d) : Prop :=
  r ≤ n ∧ 0 < γβ n β (αDist d r w ε) ε

/-- The event that the radial projections `X_i / ‖X_i‖` of a sample form a
`(β, ε)`-separated configuration on the sphere. A sample with some `X_i = 0`
has no projection and is outside the event, since no point of the sphere
equals `‖0‖⁻¹ • 0 = 0`.
Source: arXiv:2410.06833v1, §4, `prop: mixture.of.gaussians`. -/
def projectedSeparated (β ε : ℝ) : Set (Idx n → EucSpace d) :=
  { X | ∃ Y : SphereTuple d n,
      (∀ i : Idx n, (Y i : EucSpace d) = ‖X i‖⁻¹ • X i) ∧ isSeparated d n β ε Y }

/-- Positive rescaling does not change radial projection.
Source: arXiv:2410.06833v1, §4, `prop: mixture.of.gaussians`. -/
theorem radialProjection_smul_pos (a : ℝ) (ha : 0 < a) (x : EucSpace d) :
    ‖a • x‖⁻¹ • (a • x) = ‖x‖⁻¹ • x := by
  rw [norm_smul, Real.norm_eq_abs, abs_of_pos ha, smul_smul]
  congr 1
  have ha0 : a ≠ 0 := ha.ne'
  calc
    (a * ‖x‖)⁻¹ * a = (a⁻¹ * a) * ‖x‖⁻¹ := by ring
    _ = ‖x‖⁻¹ := by simp [ha0]

/-- The separated-projection event is invariant under positive rescaling of
every sample. Source: arXiv:2410.06833v1, §4,
`prop: mixture.of.gaussians`. -/
theorem projectedSeparated_smul_pos_iff (β ε : ℝ) (a : ℝ) (ha : 0 < a)
    (X : Idx n → EucSpace d) :
    (fun i => a • X i) ∈ projectedSeparated d n β ε ↔
      X ∈ projectedSeparated d n β ε := by
  constructor
  · rintro ⟨Y, hY, hsep⟩
    refine ⟨Y, ?_, hsep⟩
    intro i
    rw [← radialProjection_smul_pos d a ha]
    exact hY i
  · rintro ⟨Y, hY, hsep⟩
    refine ⟨Y, ?_, hsep⟩
    intro i
    rw [radialProjection_smul_pos d a ha]
    exact hY i

/-- Radial projection of a nonzero vector can at most double its distance to
a unit vector. Source: arXiv:2410.06833v1, §4, proof of
`prop: mixture.of.gaussians`. -/
theorem norm_radialProjection_sub_le (d : ℕ) (x : EucSpace d) (w : SSphere d)
    (hx : x ≠ 0) :
    ‖‖x‖⁻¹ • x - (w : EucSpace d)‖ ≤ 2 * ‖x - (w : EucSpace d)‖ := by
  have hxn : ‖x‖ ≠ 0 := norm_ne_zero_iff.mpr hx
  have hwn : ‖(w : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp w.2
  have hrep : x = ‖x‖ • (‖x‖⁻¹ • x) := by
    rw [smul_smul, mul_inv_cancel₀ hxn, one_smul]
  have hunit : ‖‖x‖⁻¹ • x‖ = 1 := by
    simpa using (norm_smul_inv_norm (𝕜 := ℝ) hx)
  have hdiff : ‖‖x‖⁻¹ • x - x‖ = |1 - ‖x‖| := by
    have hvec : ‖x‖⁻¹ • x - x = (1 - ‖x‖) • (‖x‖⁻¹ • x) := by
      calc
        ‖x‖⁻¹ • x - x = ‖x‖⁻¹ • x - ‖x‖ • (‖x‖⁻¹ • x) :=
          congrArg (fun z => ‖x‖⁻¹ • x - z) hrep
        _ = (1 - ‖x‖) • (‖x‖⁻¹ • x) := by module
    rw [hvec, norm_smul, hunit]
    simp
  have htriangle := norm_sub_le_norm_sub_add_norm_sub (‖x‖⁻¹ • x) x (w : EucSpace d)
  have hreverse := abs_norm_sub_norm_le x (w : EucSpace d)
  rw [hwn] at hreverse
  rw [hdiff] at htriangle
  rw [abs_sub_comm] at htriangle
  linarith

/-- A radial projection within `δ` of a unit centre lies in its cap of
height `2δ`. Source: arXiv:2410.06833v1, §4, proof of
`prop: mixture.of.gaussians`. -/
theorem mem_sphericalCap_of_radialProjection_close (d : ℕ) (x : EucSpace d)
    (w y : SSphere d) (δ : ℝ) (hx : x ≠ 0)
    (hy : (y : EucSpace d) = ‖x‖⁻¹ • x)
    (hclose : ‖x - (w : EucSpace d)‖ ≤ δ) :
    y ∈ sphericalCap d w (2 * δ) := by
  have hwn : ‖(w : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp w.2
  have hinner : inner (𝕜 := ℝ) ((w : EucSpace d) - (y : EucSpace d))
      (w : EucSpace d) ≤ 2 * δ := by
    calc
      inner (𝕜 := ℝ) ((w : EucSpace d) - (y : EucSpace d)) (w : EucSpace d)
          ≤ ‖(w : EucSpace d) - (y : EucSpace d)‖ * ‖(w : EucSpace d)‖ :=
            real_inner_le_norm _ _
      _ = ‖(y : EucSpace d) - (w : EucSpace d)‖ := by rw [hwn, norm_sub_rev, mul_one]
      _ ≤ 2 * ‖x - (w : EucSpace d)‖ := by
        rw [hy]
        exact norm_radialProjection_sub_le d x w hx
      _ ≤ 2 * δ := by gcongr
  change 1 - 2 * δ ≤ inner (𝕜 := ℝ) ((y : EucSpace d)) (w : EucSpace d)
  rw [inner_sub_left, real_inner_self_eq_norm_sq, hwn] at hinner
  nlinarith

/-- The radial-projection hypotheses are realized by a unit vector at its
centre and by the positive scale `1`. Source: arXiv:2410.06833v1, §4,
`prop: mixture.of.gaussians`. -/
example (w : SSphere 1) :
    (0 : ℝ) < 1 ∧ (w : EucSpace 1) ≠ 0 ∧
      (w : EucSpace 1) = ‖(w : EucSpace 1)‖⁻¹ • (w : EucSpace 1) ∧
      ‖(w : EucSpace 1) - (w : EucSpace 1)‖ ≤ (0 : ℝ) := by
  have hw : ‖(w : EucSpace 1)‖ = 1 := mem_sphere_zero_iff_norm.mp w.2
  refine ⟨one_pos, ?_, ?_, by simp⟩
  · intro h
    rw [h] at hw
    norm_num at hw
  · rw [hw]
    norm_num

/-- A sufficient deterministic condition for `prop: mixture.of.gaussians`:
if every nonzero sample is within `ε/2` of a centre, radial projection
produces a separated configuration. This stronger near-centre event is not
asserted to have the probability in the proposition.
Source: arXiv:2410.06833v1, §4,
`prop: mixture.of.gaussians`. -/
theorem projectedSeparated_of_near_centers (β ε : ℝ) (r : ℕ)
    (w : Idx r → SSphere d) (hcent : isCentered d n β ε r w)
    (X : Idx n → EucSpace d) (hX : ∀ i, X i ≠ 0)
    (hnear : ∀ i, ∃ q : Idx r, ‖X i - (w q : EucSpace d)‖ ≤ ε / 2) :
    X ∈ projectedSeparated d n β ε := by
  let Y : SphereTuple d n := fun i =>
    ⟨‖X i‖⁻¹ • X i, mem_sphere_zero_iff_norm.mpr (by
      simpa using (norm_smul_inv_norm (𝕜 := ℝ) (hX i)))⟩
  refine ⟨Y, fun _ => rfl, ?_⟩
  refine ⟨r, hcent.1, w, ?_, hcent.2⟩
  intro i
  obtain ⟨q, hq⟩ := hnear i
  refine ⟨q, ?_⟩
  simpa only [show 2 * (ε / 2) = ε by ring] using
    (mem_sphericalCap_of_radialProjection_close d (X i) (w q) (Y i)
      (ε / 2) (hX i) rfl hq)

/-- The centre-rescaled form used by the Gaussian mixture: after dividing a
sample by `√r`, proximity to one of the unit centres suffices for separated
radial projections. Source: arXiv:2410.06833v1, §4,
`prop: mixture.of.gaussians`. -/
theorem projectedSeparated_of_near_scaled_centers (β ε : ℝ) (r : ℕ)
    (hr : 0 < r) (w : Idx r → SSphere d) (hcent : isCentered d n β ε r w)
    (X : Idx n → EucSpace d) (hX : ∀ i, X i ≠ 0)
    (hnear : ∀ i, ∃ q : Idx r,
      ‖(Real.sqrt (r : ℝ))⁻¹ • X i - (w q : EucSpace d)‖ ≤ ε / 2) :
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
    projectedSeparated_of_near_centers d n β ε r w hcent Z hZ hnear
  have hscaled := (projectedSeparated_smul_pos_iff d n β ε a ha Z).2 hsep
  rw [hrecover] at hscaled
  exact hscaled

end Metastability
end Transformer
