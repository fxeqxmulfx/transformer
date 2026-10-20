module

public import Transformer.Causal.Section4_SubspaceLeader
public import Transformer.Causal.Section4_WeightFloor

/-!
# Leading covectors and the Rayleigh ceiling for causal attention

ArXiv:2411.04990v2, §4, `thm2`. The manuscript's invariant orthogonal
splitting makes every vector in the leading space a left eigenvector too,
although the original value matrix need not be symmetric. The full
Rayleigh quotient is bounded by the leading eigenvalue. These facts give
the exact scalar overlap equation for every original causal row, retaining
the current-token weight and the original masked normalization.
-/

@[expose] public section

open Real
open scoped BigOperators

namespace Transformer.Causal

variable {d n : ℕ}

/-- Leading vectors are also leading covectors under the manuscript's
invariant orthogonal splitting; symmetry of `V` is unnecessary.
Source: arXiv:2411.04990v2, §4, `thm2`. -/
theorem inner_value_of_mem_leading (V : ParamMatrix d) (L : Submodule ℝ (EucSpace d))
    (lam : ℝ) (heig : ∀ v ∈ L, V v = lam • v)
    (hinv : ∀ z ∈ Lᗮ, V z ∈ Lᗮ) (u : EucSpace d) (hu : u ∈ L) (x : EucSpace d) :
    inner ℝ u (V x) = lam * inner ℝ u x := by
  have hp := L.starProjection_eq_self_iff.mpr hu
  calc
    inner ℝ u (V x) = inner ℝ u (L.starProjection (V x)) := by
      rw [← L.inner_starProjection_left_eq_right, hp]
    _ = lam * inner ℝ u x := by
      rw [starProjection_value_eq V L lam heig hinv, real_inner_smul_right,
        ← L.inner_starProjection_left_eq_right, hp]

/-- The original leading/transverse splitting gives the exact Rayleigh
defect, without assuming a symmetric value matrix.
Source: arXiv:2411.04990v2, §4, the strict quadratic condition in `thm2`. -/
theorem rayleigh_defect_eq_transverse (V : ParamMatrix d)
    (L : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (heig : ∀ v ∈ L, V v = lam • v)
    (hinv : ∀ z ∈ Lᗮ, V z ∈ Lᗮ) (x : EucSpace d) :
    lam * ‖x‖ ^ 2 - inner ℝ (V x) x =
      lam * ‖x - L.starProjection x‖ ^ 2 -
        inner ℝ (V (x - L.starProjection x)) (x - L.starProjection x) := by
  let p := L.starProjection x
  let z := x - p
  have hp : p ∈ L := L.starProjection_apply_mem x
  have hz : z ∈ Lᗮ := L.sub_starProjection_mem_orthogonal x
  have hzp : inner ℝ p z = 0 := hz p hp
  have hpz : inner ℝ z p = 0 := by rw [real_inner_comm]; exact hzp
  have hVp : V p = lam • p := heig _ hp
  have hVzp : inner ℝ p (V z) = 0 := hinv _ hz p hp
  have hVz : inner ℝ (V z) p = 0 := by rw [real_inner_comm]; exact hVzp
  have hx : x = p + z := by dsimp [z]; abel
  change lam * ‖x‖ ^ 2 - inner ℝ (V x) x = lam * ‖z‖ ^ 2 - inner ℝ (V z) z
  rw [hx, ← real_inner_self_eq_norm_sq, inner_add_left, inner_add_right,
    inner_add_right, hzp, hpz, map_add, hVp, inner_add_left,
    inner_add_right, inner_add_right, real_inner_smul_left,
    real_inner_smul_left, hzp, hVz, real_inner_self_eq_norm_sq,
    real_inner_self_eq_norm_sq]
  ring

/-- The manuscript's strict transverse condition bounds the original
Rayleigh quotient on the full ambient space.
Source: arXiv:2411.04990v2, §4, `thm2`. -/
theorem rayleigh_le_leading (V : ParamMatrix d) (L : Submodule ℝ (EucSpace d))
    (lam : ℝ) (heig : ∀ v ∈ L, V v = lam • v)
    (hinv : ∀ z ∈ Lᗮ, V z ∈ Lᗮ)
    (hdecay : ∀ z ∈ Lᗮ, z ≠ 0 → inner ℝ (V z) z < lam * ‖z‖ ^ 2)
    (x : EucSpace d) : inner ℝ (V x) x ≤ lam * ‖x‖ ^ 2 := by
  rw [← sub_nonneg, rayleigh_defect_eq_transverse V L lam heig hinv]
  by_cases hz : x - L.starProjection x = 0
  · simp [hz]
  · exact (sub_pos.mpr (hdecay _ (L.sub_starProjection_mem_orthogonal x) hz)).le

/-- The scalar equation of an actual original causal token against an
actual leading vector, with its entire original row of weights.
Source: arXiv:2411.04990v2, §4, `thm2`, and the overlap computation in the proof of `thm1`. -/
theorem hasDerivAt_csaLeadingOverlap (β : ℝ) (Q K V : ParamMatrix d)
    (L : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (heig : ∀ v ∈ L, V v = lam • v) (hinv : ∀ z ∈ Lᗮ, V z ∈ Lᗮ)
    (u : EucSpace d) (hu : u ∈ L)
    (X : ℝ → SphereTuple d n) (hX : CSA d n β Q K V X) (i : Idx n) (t : ℝ) :
    HasDerivAt (fun s => inner ℝ u (X s i : EucSpace d))
      (lam * ∑ j, csaTokenWeight β Q K (fun k => (X t k : EucSpace d)) i j *
          inner ℝ u (X t j : EucSpace d) -
        inner ℝ (X t i : EucSpace d)
          (csaAverage β Q K V (fun k => (X t k : EucSpace d)) i) *
            inner ℝ u (X t i : EucSpace d)) t := by
  have h := (hasDerivAt_const t u).inner ℝ ((CSA_iff_csaAmbientField β Q K V X).mp hX t i)
  simp only [inner_zero_left, add_zero, csaAmbientField, proj, inner_sub_right,
    real_inner_smul_right, csaAverage, inner_sum, inner_value_of_mem_leading V L lam heig hinv u hu] at h
  convert! h using 1
  rw [Finset.mul_sum]
  apply congrArg₂ (· - ·)
  · apply Finset.sum_congr rfl
    intro j _
    ring
  · simp only [csaAverage, inner_sum, real_inner_smul_right]

/-- The original value-projected predecessor force differs from its
leading consensus value by the original value norm times the predecessor
error. Source: arXiv:2411.04990v2, §4, the overlap argument of `thm1`
applied to the leading eigenvector in `thm2`. -/
theorem leadingOverlapInteraction_approx (V : ParamMatrix d) (lam : ℝ)
    (u x p : SSphere d) (heig : V (u : EucSpace d) = lam • (u : EucSpace d)) :
    |inner ℝ (u : EucSpace d) (proj d (x : EucSpace d) (V (p : EucSpace d))) -
      lam * (1 - inner ℝ (u : EucSpace d) (x : EucSpace d) ^ 2)| ≤
        ‖V‖ * ‖(p : EucSpace d) - (u : EucSpace d)‖ := by
  have he : inner ℝ (u : EucSpace d) (proj d (x : EucSpace d) (V (p : EucSpace d))) -
      lam * (1 - inner ℝ (u : EucSpace d) (x : EucSpace d) ^ 2) =
      inner ℝ (u : EucSpace d)
        (proj d (x : EucSpace d) (V ((p : EucSpace d) - (u : EucSpace d)))) := by
    simp only [proj, map_sub, heig, inner_sub_right,
      real_inner_smul_right, real_inner_self_eq_norm_sq, mem_sphere_zero_iff_norm.mp u.2, one_pow]
    rw [real_inner_comm (u : EucSpace d) (x : EucSpace d)]
    ring
  rw [he]
  calc
    _ ≤ ‖(u : EucSpace d)‖ *
        ‖proj d (x : EucSpace d) (V ((p : EucSpace d) - (u : EucSpace d)))‖ := abs_real_inner_le_norm _ _
    _ ≤ ‖V ((p : EucSpace d) - (u : EucSpace d))‖ := by
      rw [mem_sphere_zero_iff_norm.mp u.2, one_mul]
      exact norm_proj_le (mem_sphere_zero_iff_norm.mp x.2) _
    _ ≤ _ := V.le_opNorm _

/-- The genuine non-scalar value projection realizes the spectral,
leading-vector and actual-flow premises at once. No quantitative gap
or convergence hypothesis is supplied.
Source: arXiv:2411.04990v2, §4, `thm2`. -/
example : ∃ (V : ParamMatrix 2) (L : Submodule ℝ (EucSpace 2))
    (u : EucSpace 2) (X : ℝ → SphereTuple 2 2),
    (∀ v ∈ L, V v = (1 : ℝ) • v) ∧ (∀ z ∈ Lᗮ, V z ∈ Lᗮ) ∧
    (∀ z ∈ Lᗮ, z ≠ 0 → inner ℝ (V z) z < (1 : ℝ) * ‖z‖ ^ 2) ∧
    u ∈ L ∧ ‖u‖ = 1 ∧ CSA 2 2 1 0 0 V X := by
  let u : EucSpace 2 := basePoint 1
  let L : Submodule ℝ (EucSpace 2) := ℝ ∙ u
  have hu : u ∈ L := Submodule.mem_span_singleton_self u
  refine ⟨L.starProjection, L, u,
    csaFlow 1 0 0 L.starProjection (fun _ => basePoint 1),
    fun v hv => by simpa using L.starProjection_eq_self_iff.mpr hv, ?_, ?_,
    hu, mem_sphere_zero_iff_norm.mp (basePoint 1).2, (csaFlow_spec _ _ _ _ _).2⟩
  · intro z hz
    rw [L.starProjection_apply_eq_zero_iff.mpr hz]
    exact Lᗮ.zero_mem
  · intro z hz hz0
    rw [L.starProjection_apply_eq_zero_iff.mpr hz, inner_zero_left, one_mul]
    exact sq_pos_of_pos (norm_pos_iff.mpr hz0)

end Transformer.Causal
