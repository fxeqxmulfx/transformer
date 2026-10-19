module

public import Transformer.Metastability.Section4_UniformTail

/-!
# A fixed finite net for the original projection estimate

ArXiv:2410.06833v1, §4, `prop: concentration unif`, and
arXiv:2312.10794v5, Appendix D, Step 2, almost orthogonality.
Projection onto a previously chosen orthonormal frame is represented
by its actual coordinate vector. A finite half-net on the coordinate
sphere detects its norm through actual scalar products.

The net depends only on the frame length, not on the ambient dimension,
the uniform law or the sample. Its finite cardinality can therefore be
absorbed into the original dimension threshold depending on `n`.
Compactness constructs the net; no covering estimate is assumed.
The detection factor two is proved by normalizing the actual nonzero
coordinate vector and applying Cauchy–Schwarz to a nearby net point.
-/

@[expose] public section

open Real Set

namespace Transformer.Metastability

/-- The actual coordinate sphere has a finite open half-net, including
the empty sphere in dimension zero. Source: arXiv:2410.06833v1, §4,
the finite-direction reduction of the original projection estimate. -/
theorem exists_spherical_half_net (n : ℕ) :
    ∃ s : Finset (SSphere n), ∀ x : SSphere n, ∃ b ∈ s, dist x b < (1 / 2 : ℝ) := by
  classical
  have hcover : (univ : Set (SSphere n)) ⊆ ⋃ b : SSphere n, Metric.ball b (1 / 2 : ℝ) := by
    intro x hx
    exact mem_iUnion.mpr ⟨x, Metric.mem_ball_self (by norm_num)⟩
  obtain ⟨s, hs⟩ := isCompact_univ.elim_finite_subcover
    (fun b : SSphere n => Metric.ball b (1 / 2 : ℝ))
    (fun _ => Metric.isOpen_ball) hcover
  refine ⟨s, fun x => ?_⟩
  have hx := hs (mem_univ x)
  obtain ⟨b, hb⟩ := mem_iUnion.mp hx
  obtain ⟨hb, hball⟩ := mem_iUnion.mp hb
  exact ⟨b, hb, hball⟩

/-- A fixed finite half-net on the actual coordinate sphere.
Source: arXiv:2410.06833v1, §4, the finite-direction projection estimate.
Its only choice is the finite cover just constructed by compactness. -/
noncomputable def sphericalHalfNet (n : ℕ) : Finset (SSphere n) :=
  Classical.choose (exists_spherical_half_net n)

/-- Every actual unit coordinate vector has a nearby point of the fixed
net. Source: arXiv:2410.06833v1, §4, the original projection bound. -/
theorem sphericalHalfNet_covers (n : ℕ) (x : SSphere n) :
    ∃ b ∈ sphericalHalfNet n, dist x b < (1 / 2 : ℝ) :=
  Classical.choose_spec (exists_spherical_half_net n) x

/-- Positive-dimensional coordinate spheres give nonempty actual nets.
Source: arXiv:2410.06833v1, §4, the finite-direction projection estimate. -/
theorem sphericalHalfNet_nonempty {n : ℕ} (hn : 0 < n) : (sphericalHalfNet n).Nonempty := by
  let x : SSphere n := ⟨EuclideanSpace.single ⟨0, hn⟩ 1,
    mem_sphere_zero_iff_norm.mpr (by simp)⟩
  obtain ⟨b, hb, hclose⟩ := sphericalHalfNet_covers n x
  exact ⟨b, hb⟩

/-- A half-close vector has scalar product at least one half with the
actual unit vector. Source: arXiv:2410.06833v1, §4, the scalar detection
step in the original projection concentration estimate. -/
theorem inner_ge_half_of_close {n : ℕ} {u v : EucSpace n}
    (hu : ‖u‖ = 1) (hclose : ‖u - v‖ ≤ (1 / 2 : ℝ)) :
    (1 / 2 : ℝ) ≤ inner (𝕜 := ℝ) v u := by
  have h := abs_real_inner_le_norm (u - v) u
  rw [inner_sub_left, real_inner_self_eq_norm_sq, hu, one_pow, mul_one] at h
  have hbound : |1 - inner (𝕜 := ℝ) v u| ≤ (1 / 2 : ℝ) := h.trans hclose
  have hleft := (abs_le.mp hbound).2
  linarith

/-- Actual normalization has unit norm at every nonzero coordinate
vector. Source: arXiv:2410.06833v1, §4, the normalization used in the
original projection estimate; the zero case is explicitly excluded. -/
theorem norm_normalized_coordinate {n : ℕ} (z : EucSpace n) (hz : z ≠ 0) :
    ‖‖z‖⁻¹ • z‖ = 1 := by
  have hpos := norm_pos_iff.mpr hz
  rw [norm_smul, Real.norm_eq_abs, abs_of_pos (inv_pos.mpr hpos)]
  exact inv_mul_cancel₀ hpos.ne'

/-- Normalization reconstructs the actual coordinate vector.
Source: arXiv:2410.06833v1, §4, the actual scalar detection argument. -/
theorem norm_smul_normalized_coordinate {n : ℕ} (z : EucSpace n) (hz : z ≠ 0) :
    ‖z‖ • (‖z‖⁻¹ • z) = z := by
  rw [smul_smul, mul_inv_cancel₀ (norm_pos_iff.mpr hz).ne', one_smul]

/-- Some actual net direction detects at least half the norm of a
nonzero coordinate vector. Source: arXiv:2410.06833v1, §4, the
finite-direction reduction of the original projection estimate. -/
theorem sphericalHalfNet_detects_norm {n : ℕ} (z : EucSpace n) (hz : z ≠ 0) :
    ∃ b ∈ sphericalHalfNet n, ‖z‖ / 2 ≤ |inner (𝕜 := ℝ) (b : EucSpace n) z| := by
  let u : SSphere n := ⟨‖z‖⁻¹ • z,
    mem_sphere_zero_iff_norm.mpr (norm_normalized_coordinate z hz)⟩
  obtain ⟨b, hb, hclose⟩ := sphericalHalfNet_covers n u
  have hdist : ‖(u : EucSpace n) - (b : EucSpace n)‖ ≤ (1 / 2 : ℝ) := by
    simpa only [Subtype.dist_eq, dist_eq_norm] using hclose.le
  have hhalf := inner_ge_half_of_close (norm_eq_of_mem_sphere u) hdist
  have hrepr : z = ‖z‖ • (u : EucSpace n) :=
    (norm_smul_normalized_coordinate z hz).symm
  have hpair : inner (𝕜 := ℝ) (b : EucSpace n) z =
      ‖z‖ * inner (𝕜 := ℝ) (b : EucSpace n) (u : EucSpace n) := by
    conv_lhs => rw [hrepr]
    rw [inner_smul_right]
  have hscaled := mul_le_mul_of_nonneg_left hhalf (norm_nonneg z)
  refine ⟨b, hb, ?_⟩
  rw [hpair, abs_of_nonneg (by positivity)]
  linarith

/-- An actual positive norm threshold implies one of the finitely many
actual scalar threshold events. Source: arXiv:2410.06833v1, §4,
the event inclusion used in the original projection union bound. -/
theorem sphericalHalfNet_detects_threshold {n : ℕ} (z : EucSpace n)
    (r : ℝ) (hr : 0 < r) (hz : r ≤ ‖z‖) :
    ∃ b ∈ sphericalHalfNet n, r / 2 ≤ |inner (𝕜 := ℝ) (b : EucSpace n) z| := by
  have hnonzero : z ≠ 0 := norm_pos_iff.mp (hr.trans_le hz)
  obtain ⟨b, hb, hdetect⟩ := sphericalHalfNet_detects_norm z hnonzero
  exact ⟨b, hb, (by linarith : r / 2 ≤ ‖z‖ / 2).trans hdetect⟩

/-- Uniformly small actual net coordinates force the original norm
below its threshold. Source: arXiv:2410.06833v1, §4, the deterministic
finite-direction criterion for the original projection estimate. -/
theorem norm_lt_of_sphericalHalfNet_bounds {n : ℕ} (z : EucSpace n)
    (r : ℝ) (hr : 0 < r)
    (hnet : ∀ b ∈ sphericalHalfNet n, |inner (𝕜 := ℝ) (b : EucSpace n) z| < r / 2) :
    ‖z‖ < r := by
  by_contra h
  obtain ⟨b, hb, hdetect⟩ := sphericalHalfNet_detects_threshold z r hr (le_of_not_gt h)
  exact (not_lt_of_ge hdetect) (hnet b hb)

/-- Weak bounds on all actual net coordinates give the corresponding
weak norm bound. Source: arXiv:2410.06833v1, §4, the deterministic
finite-direction projection criterion, including a zero threshold. -/
theorem norm_le_of_sphericalHalfNet_bounds {n : ℕ} (z : EucSpace n)
    (r : ℝ) (hr : 0 ≤ r)
    (hnet : ∀ b ∈ sphericalHalfNet n, |inner (𝕜 := ℝ) (b : EucSpace n) z| ≤ r / 2) :
    ‖z‖ ≤ r := by
  by_contra h
  have hpos : 0 < ‖z‖ := hr.trans_lt (lt_of_not_ge h)
  obtain ⟨b, hb, hdetect⟩ := sphericalHalfNet_detects_norm z (norm_pos_iff.mp hpos)
  have hbound := hnet b hb
  linarith

/-- Positive dimension, a nonzero coordinate vector, unit normalization
and the close-vector hypotheses occur together on the actual circle. -/
example : (0 : ℕ) < 2 ∧
    (EuclideanSpace.single (0 : Fin 2) 1 : EucSpace 2) ≠ 0 ∧
    ‖(EuclideanSpace.single (0 : Fin 2) 1 : EucSpace 2)‖ = 1 ∧
    ‖(EuclideanSpace.single (0 : Fin 2) 1 : EucSpace 2) -
      EuclideanSpace.single (0 : Fin 2) 1‖ ≤ (1 / 2 : ℝ) ∧ (0 : ℝ) < 1 := by
  refine ⟨by decide, ?_, by simp, by simp, by norm_num⟩
  apply norm_pos_iff.mp
  simp

/-- The actual nonzero coordinate direction also satisfies the strict
and weak finite-net bounds at a positive threshold. -/
example : (0 : ℝ) < 4 ∧ ∀ b ∈ sphericalHalfNet 2,
    |inner (𝕜 := ℝ) (b : EucSpace 2) (EuclideanSpace.single (0 : Fin 2) 1)| < 4 / 2 := by
  refine ⟨by norm_num, fun b hb => ?_⟩
  have h := abs_real_inner_le_norm (b : EucSpace 2) (EuclideanSpace.single (0 : Fin 2) 1)
  rw [norm_eq_of_mem_sphere b] at h
  have he : ‖(EuclideanSpace.single (0 : Fin 2) 1 : EucSpace 2)‖ = 1 := by simp
  rw [he, mul_one] at h
  linarith

end Transformer.Metastability
