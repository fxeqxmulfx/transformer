/-
# The simple leading eigencoordinate and its invariant complement

ArXiv:2305.05465v6, §4, `d:good`, and §9, the proof of
`l:3hyperplanes11`, `eq:phistarvar` and `e:majorizeweights`.
The leading coordinate is a normalized left eigenfunctional, not an
orthogonal coordinate when the invariant splitting is oblique.

The strict spectral gap forces this functional to vanish on the
complement: its value on powers would grow as `λ^k`, whereas the
complementary power bound grows only as `r^k` with `r < λ`.
Consequently `x ↦ f(x) φ` is exactly the leading projection.

A good triple supplies the paranormal value splitting used by the
proved complementary growth estimates. Only the attention form along
the leading line is positive; no global positive-definiteness of
`QᵀK`, orthogonality, or diagonalizability is added here.
-/

import Transformer.Clusters.Section9_Eigen
import Transformer.Clusters.Section10_ComplementaryProjection

open Real Filter Topology

namespace Transformer.Clusters

variable {d : ℕ}

/-- Powers of the value map act on a left eigenfunctional by scalar
powers. Source: arXiv:2305.05465v6, §9, the eigenbasis in `e:majorizeweights`. -/
theorem IsEigenFunctional.pow_apply (V : ParamMatrix d) (f : EucSpace d →L[ℝ] ℝ)
    (lam : ℝ) (hf : IsEigenFunctional V f lam) (k : ℕ) (x : EucSpace d) :
    f ((V ^ k) x) = lam ^ k * f x := by
  induction k generalizing x with
  | zero => simp only [pow_zero, one_apply_eq_self, one_mul]
  | succ k ih =>
    rw [pow_succ V k]
    change f ((V ^ k) (V x)) = lam ^ (k + 1) * f x
    rw [ih, hf, pow_succ lam k]
    ring

example (f : EucSpace d →L[ℝ] ℝ) : IsEigenFunctional (1 : ParamMatrix d) f 1 :=
  isEigenFunctional_one f

/-- A leading left eigenfunctional vanishes on every subspace with
strictly smaller spectral radius. Source: arXiv:2305.05465v6, §9,
the dual leading coordinate in `eq:phistarvar` and `e:majorizeweights`. -/
theorem IsEigenFunctional.eq_zero_on_spectral_complement (V : ParamMatrix d)
    (f : EucSpace d →L[ℝ] ℝ) (lam : ℝ) (hlam : 0 < lam)
    (hf : IsEigenFunctional V f lam) (G : Submodule ℝ (EucSpace d))
    (hG : IsSpectralRadiusLtOn G V lam) (w : EucSpace d) (hw : w ∈ G) : f w = 0 := by
  obtain ⟨C, r, _, hr, hrlam, hbound⟩ := hG
  have hestimate (k : ℕ) : lam ^ k * |f w| ≤ ‖f‖ * (C * r ^ k * ‖w‖) := by
    calc
      _ = ‖f ((V ^ k) w)‖ := by
        rw [hf.pow_apply V f lam k w, Real.norm_eq_abs, abs_mul,
          abs_of_pos (pow_pos hlam k)]
      _ ≤ ‖f‖ * ‖(V ^ k) w‖ := f.le_opNorm _
      _ ≤ _ := mul_le_mul_of_nonneg_left (hbound k w hw) (norm_nonneg f)
  have hnormalized (k : ℕ) : |f w| ≤ (‖f‖ * C * ‖w‖) * (r / lam) ^ k := by
    have hratio : (‖f‖ * C * ‖w‖) * (r / lam) ^ k =
        (‖f‖ * (C * r ^ k * ‖w‖)) / lam ^ k := by
      rw [div_pow]
      ring
    rw [hratio]
    apply (le_div_iff₀ (pow_pos hlam k)).mpr
    simpa only [mul_comm] using hestimate k
  have hlim : Tendsto (fun k : ℕ => (‖f‖ * C * ‖w‖) * (r / lam) ^ k)
      atTop (nhds 0) := by
    simpa only [mul_zero] using
      (tendsto_pow_atTop_nhds_zero_of_lt_one (div_nonneg hr hlam.le)
        ((div_lt_one₀ hlam).mpr hrlam)).const_mul (‖f‖ * C * ‖w‖)
  have hzero := le_of_tendsto_of_tendsto (tendsto_const_nhds (x := |f w|)) hlim
    (Eventually.of_forall hnormalized)
  exact abs_eq_zero.mp (le_antisymm hzero (abs_nonneg _))

example (f : EucSpace 1 →L[ℝ] ℝ) : (0 : ℝ) < 1 ∧
    IsEigenFunctional (1 : ParamMatrix 1) f 1 ∧
    IsSpectralRadiusLtOn (⊥ : Submodule ℝ (EucSpace 1)) 1 1 ∧
    (0 : EucSpace 1) ∈ (⊥ : Submodule ℝ (EucSpace 1)) :=
  ⟨one_pos, isEigenFunctional_one f, (isGoodTripleMulti_one 1).2.2.2.2.2.2,
    Submodule.zero_mem _⟩

/-- A simple-leading good triple supplies the paranormal splitting,
independently of global attention positivity. Source: arXiv:2305.05465v6,
§4, `d:good`, and §9, `r:notdiagonalizable`. -/
theorem IsGoodTripleWith.isParanormalOn (Q K V : ParamMatrix d) (lam : ℝ)
    (φ : EucSpace d) (G : Submodule ℝ (EucSpace d))
    (hgood : IsGoodTripleWith Q K V lam φ G) :
    IsParanormalOn V (Submodule.span ℝ {φ}) G lam := by
  have hleading : ∀ w ∈ Submodule.span ℝ {φ}, V w = lam • w := by
    intro w hw
    obtain ⟨a, rfl⟩ := Submodule.mem_span_singleton.mp hw
    rw [map_smul, hgood.2.2.1, smul_smul, smul_smul]
    congr 1
    ring
  exact ⟨fun w hw => (hleading w hw) ▸ Submodule.smul_mem _ lam hw,
    hgood.2.2.2.1, hgood.2.2.2.2.1, hgood.1, hleading, hgood.2.2.2.2.2.1⟩

example : IsGoodTripleWith (1 : ParamMatrix 1) 1 1 1
    (EuclideanSpace.single 0 (1 : ℝ)) ⊥ := isGoodTripleWith_one

/-- The normalized eigenfunctional constructs the exact leading
projection along the invariant complement. Source: arXiv:2305.05465v6,
§9, the decomposition into `φ₁*` and the complementary eigencoordinates. -/
theorem IsGoodTripleWith.isProjOnto_smulRight (Q K V : ParamMatrix d) (lam : ℝ)
    (φ : EucSpace d) (G : Submodule ℝ (EucSpace d))
    (hgood : IsGoodTripleWith Q K V lam φ G) (f : EucSpace d →L[ℝ] ℝ)
    (hf : IsEigenFunctional V f lam) (hfφ : f φ = 1) :
    IsProjOnto (f.smulRight φ) (Submodule.span ℝ {φ}) G := by
  refine ⟨fun x => Submodule.mem_span_singleton.mpr ⟨f x, rfl⟩, ?_, ?_⟩
  · intro x hx
    obtain ⟨a, rfl⟩ := Submodule.mem_span_singleton.mp hx
    change f (a • φ) • φ = a • φ
    rw [map_smul, hfφ, smul_eq_mul, mul_one]
  · intro x hx
    change f x • φ = 0
    rw [hf.eq_zero_on_spectral_complement V f lam hgood.1 G hgood.2.2.2.2.2.1 x hx,
      zero_smul]

example : IsGoodTripleWith (1 : ParamMatrix 1) 1 1 1
    (EuclideanSpace.single 0 (1 : ℝ)) ⊥ ∧
    IsEigenFunctional (1 : ParamMatrix 1) (EuclideanSpace.proj (𝕜 := ℝ) 0) 1 ∧
    (EuclideanSpace.proj (𝕜 := ℝ) 0) (EuclideanSpace.single 0 (1 : ℝ)) = 1 :=
  ⟨isGoodTripleWith_one, isEigenFunctional_one _, by simp⟩

/-- Every good triple has an actual normalized leading left
eigenfunctional. Source: arXiv:2305.05465v6, §9, the coordinate `φ₁*`.
The construction uses the direct-sum projection and normalizes the
inner product on the leading line; it does not posit an eigenbasis. -/
theorem IsGoodTripleWith.exists_leading_eigenFunctional (Q K V : ParamMatrix d) (lam : ℝ)
    (φ : EucSpace d) (G : Submodule ℝ (EucSpace d))
    (hgood : IsGoodTripleWith Q K V lam φ G) :
    ∃ f : EucSpace d →L[ℝ] ℝ, IsEigenFunctional V f lam ∧ f φ = 1 ∧
      ∀ w ∈ G, f w = 0 := by
  obtain ⟨proj, hproj⟩ := exists_isProjOnto (Submodule.span ℝ {φ}) G hgood.2.2.2.2.1
  let f : EucSpace d →L[ℝ] ℝ := (‖φ‖ ^ 2)⁻¹ • (innerSL ℝ φ).comp proj
  have hφmem : φ ∈ Submodule.span ℝ {φ} :=
    Submodule.mem_span_singleton.mpr ⟨1, one_smul ℝ φ⟩
  have hnorm : ‖φ‖ ^ 2 ≠ 0 := (sq_pos_of_pos (norm_pos_iff.mpr hgood.2.1)).ne'
  refine ⟨f, ?_, ?_, ?_⟩
  · intro x
    change (‖φ‖ ^ 2)⁻¹ * inner (𝕜 := ℝ) φ (proj (V x)) = lam * f x
    rw [hproj.apply_value proj V _ G lam (hgood.isParanormalOn Q K V lam φ G),
      real_inner_smul_right]
    change _ = lam * ((‖φ‖ ^ 2)⁻¹ * inner (𝕜 := ℝ) φ (proj x))
    ring
  · change (‖φ‖ ^ 2)⁻¹ * inner (𝕜 := ℝ) φ (proj φ) = 1
    rw [hproj.2.1 φ hφmem, real_inner_self_eq_norm_sq, inv_mul_cancel₀ hnorm]
  · intro w hw
    change (‖φ‖ ^ 2)⁻¹ * inner (𝕜 := ℝ) φ (proj w) = 0
    rw [hproj.2.2 w hw, inner_zero_right, mul_zero]

example : IsGoodTripleWith (1 : ParamMatrix 1) 1 1 1
    (EuclideanSpace.single 0 (1 : ℝ)) ⊥ := isGoodTripleWith_one

end Transformer.Clusters
