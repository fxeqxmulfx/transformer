/-
# Actual coordinates and projection of the previous orthonormal frame

ArXiv:2410.06833v1, §4, `prop: concentration unif`, and
arXiv:2312.10794v5, Appendix D, Step 2, almost orthogonality.
The original previous frame is represented by its actual finite list
of ambient vectors. Synthesis is their actual linear combination;
analysis is the vector of their actual inner products with the sample.
Their composition is the actual orthogonal projection when the frame
is orthonormal. No bound or orthogonality is built into a definition.

Orthonormality proves that synthesis preserves the coordinate norm,
identifies scalar products with ambient directions, and makes the
actual residual orthogonal to every previous vector. The exact norm
identity supplies the original projection threshold. Joint continuity
also keeps these actual constructions measurable in the previous
frame and the new sample, as required for product initialization.
-/

import Transformer.Metastability.Section4_SphericalNet

open Real
open scoped BigOperators

namespace Transformer.Metastability

/-- The actual ambient linear combination of a finite frame.
Source: arXiv:2410.06833v1, §4, the original orthogonal projection
in `prop: concentration unif`; arXiv:2312.10794v5, Appendix D, Step 2. -/
noncomputable def frameSynthesis {d m : ℕ} (v : Fin m → EucSpace d)
    (b : EucSpace m) : EucSpace d := ∑ i : Fin m, b i • v i

/-- The actual vector of inner products against the previous frame.
Source: arXiv:2410.06833v1, §4, the original projection estimate. -/
noncomputable def frameCoordinates {d m : ℕ} (v : Fin m → EucSpace d)
    (x : EucSpace d) : EucSpace m :=
  WithLp.toLp 2 (fun i : Fin m => inner (𝕜 := ℝ) (v i) x)

/-- The actual finite-frame projection, before any normalization.
Source: arXiv:2410.06833v1, §4, the original inductive approximation;
arXiv:2312.10794v5, Appendix D, Step 2. It is defined for every frame. -/
noncomputable def frameProjection {d m : ℕ} (v : Fin m → EucSpace d)
    (x : EucSpace d) : EucSpace d := frameSynthesis v (frameCoordinates v x)

/-- Actual orthonormal synthesis preserves the squared coordinate norm.
Source: arXiv:2410.06833v1, §4, the original projection geometry. -/
theorem norm_frameSynthesis_sq {d m : ℕ} (v : Fin m → EucSpace d)
    (hv : Orthonormal ℝ v) (b : EucSpace m) : ‖frameSynthesis v b‖ ^ 2 = ‖b‖ ^ 2 := by
  rw [← real_inner_self_eq_norm_sq, frameSynthesis]
  rw [hv.inner_sum (fun i => b i) (fun i => b i) Finset.univ]
  rw [EuclideanSpace.real_norm_sq_eq]
  apply Finset.sum_congr rfl
  intro i hi
  simp only [RCLike.conj_to_real, pow_two]

/-- Actual orthonormal synthesis preserves the coordinate norm.
Source: arXiv:2410.06833v1, §4, the original finite-direction
projection estimate, with its original unit ambient directions. -/
theorem norm_frameSynthesis {d m : ℕ} (v : Fin m → EucSpace d)
    (hv : Orthonormal ℝ v) (b : EucSpace m) : ‖frameSynthesis v b‖ = ‖b‖ :=
  (sq_eq_sq₀ (norm_nonneg _) (norm_nonneg _)).mp (norm_frameSynthesis_sq v hv b)

/-- Actual coordinate pairing is actual ambient directional pairing.
Source: arXiv:2410.06833v1, §4, the original scalar projection estimate.
This identity needs no orthonormality or dimension restriction. -/
theorem inner_frameSynthesis {d m : ℕ} (v : Fin m → EucSpace d)
    (b : EucSpace m) (x : EucSpace d) :
    inner (𝕜 := ℝ) (frameSynthesis v b) x =
      inner (𝕜 := ℝ) b (frameCoordinates v x) := by
  rw [frameSynthesis, sum_inner, PiLp.inner_apply]
  apply Finset.sum_congr rfl
  intro i hi
  simp only [real_inner_smul_left, frameCoordinates, WithLp.ofLp_toLp, Real.inner_apply]

/-- The actual projection norm is the actual coordinate-vector norm.
Source: arXiv:2410.06833v1, §4, the threshold in the original
orthogonal-projection concentration estimate. -/
theorem norm_frameProjection {d m : ℕ} (v : Fin m → EucSpace d)
    (hv : Orthonormal ℝ v) (x : EucSpace d) :
    ‖frameProjection v x‖ = ‖frameCoordinates v x‖ :=
  norm_frameSynthesis v hv (frameCoordinates v x)

/-- The actual projection preserves every previous frame coordinate.
Source: arXiv:2410.06833v1, §4, the original inductive projection. -/
theorem inner_frameProjection {d m : ℕ} (v : Fin m → EucSpace d)
    (hv : Orthonormal ℝ v) (x : EucSpace d) (i : Fin m) :
    inner (𝕜 := ℝ) (v i) (frameProjection v x) = inner (𝕜 := ℝ) (v i) x := by
  exact hv.inner_right_sum (fun j => inner (𝕜 := ℝ) (v j) x) (Finset.mem_univ i)

/-- The actual residual is orthogonal to every previous vector.
Source: arXiv:2410.06833v1, §4, the original construction of the next
unit vector; arXiv:2312.10794v5, Appendix D, Step 2. -/
theorem inner_frameResidual {d m : ℕ} (v : Fin m → EucSpace d)
    (hv : Orthonormal ℝ v) (x : EucSpace d) (i : Fin m) :
    inner (𝕜 := ℝ) (v i) (x - frameProjection v x) = 0 := by
  rw [inner_sub_right, inner_frameProjection v hv x i, sub_self]

/-- Pairing the actual projection with its original point gives its
exact squared norm. Source: arXiv:2410.06833v1, §4, the original
Pythagorean identity in the inductive approximation. -/
theorem inner_frameProjection_self {d m : ℕ} (v : Fin m → EucSpace d)
    (hv : Orthonormal ℝ v) (x : EucSpace d) :
    inner (𝕜 := ℝ) (frameProjection v x) x = ‖frameProjection v x‖ ^ 2 := by
  rw [frameProjection, inner_frameSynthesis, real_inner_self_eq_norm_sq,
    norm_frameSynthesis v hv]

/-- The actual residual satisfies the original exact Pythagorean
identity. Source: arXiv:2410.06833v1, §4, the normalization geometry
for the original near-orthogonal configuration. -/
theorem norm_frameResidual_sq {d m : ℕ} (v : Fin m → EucSpace d)
    (hv : Orthonormal ℝ v) (x : EucSpace d) :
    ‖x - frameProjection v x‖ ^ 2 = ‖x‖ ^ 2 - ‖frameProjection v x‖ ^ 2 := by
  rw [norm_sub_sq_real, real_inner_comm (frameProjection v x) x,
    inner_frameProjection_self v hv x]
  ring

/-- The actual orthogonal projection is contractive.
Source: arXiv:2410.06833v1, §4, the original finite-frame geometry
used in the inductive orthogonal approximation. -/
theorem norm_frameProjection_le {d m : ℕ} (v : Fin m → EucSpace d)
    (hv : Orthonormal ℝ v) (x : EucSpace d) : ‖frameProjection v x‖ ≤ ‖x‖ := by
  have h := norm_frameResidual_sq v hv x
  have hnonneg := sq_nonneg ‖x - frameProjection v x‖
  nlinarith [norm_nonneg x, norm_nonneg (frameProjection v x)]

/-- Actual synthesis depends continuously on both its frame and its
coefficients. Source: arXiv:2410.06833v1, §4, the measurable original
construction using only the previous sample points. -/
theorem continuous_frameSynthesis {d m : ℕ} :
    Continuous (fun p : (Fin m → EucSpace d) × EucSpace m => frameSynthesis p.1 p.2) := by
  unfold frameSynthesis
  fun_prop

/-- The actual coordinate vector depends continuously on both its
frame and its original point. Source: arXiv:2410.06833v1, §4,
conditioning the original projection estimate on the previous frame. -/
theorem continuous_frameCoordinates {d m : ℕ} :
    Continuous (fun p : (Fin m → EucSpace d) × EucSpace d => frameCoordinates p.1 p.2) := by
  unfold frameCoordinates
  apply (PiLp.continuous_toLp 2 (fun _ : Fin m => ℝ)).comp
  fun_prop

/-- The actual projection depends continuously on both the frame and
the original point. Source: arXiv:2410.06833v1, §4, the original
measurable inductive approximation in `prop: concentration unif`. -/
theorem continuous_frameProjection {d m : ℕ} :
    Continuous (fun p : (Fin m → EucSpace d) × EucSpace d => frameProjection p.1 p.2) := by
  unfold frameProjection
  exact continuous_frameSynthesis.comp (continuous_fst.prodMk continuous_frameCoordinates)

/-- The two actual coordinate axes give a nonempty orthonormal frame;
this witnesses every orthonormality hypothesis in the projection lemmas. -/
example : Orthonormal ℝ (fun i : Fin 2 => (EuclideanSpace.single i 1 : EucSpace 2)) := by
  rw [orthonormal_iff_ite]
  intro i j
  simp [EuclideanSpace.inner_single_left]

end Transformer.Metastability
