module

public import Transformer.Causal.Section4_SubspaceGeometry
public import Mathlib.LinearAlgebra.Dimension.RankNullity

/-!
# Transverse leading covectors and the exceptional initialization

ArXiv:2411.04990v2, §4, `thm2`, for two tokens. These auxiliary
geometric facts make its exceptional set explicit. A polynomial
residual detects whether the follower's leading projection lies
on the leader's leading line. A nonzero residual itself supplies
a leading covector orthogonal to the original leader and positive
on the original follower. A leading space of dimension at least two
also supplies a nonzero normal for every exceptional fiber.
-/

@[expose] public section

open Real

namespace Transformer.Causal

variable {d : ℕ}

/-- A division-free transverse residual of the two actual leading
projections. It is an auxiliary construction for arXiv:2411.04990v2,
§4, `thm2`; no dynamics is replaced by this residual. -/
noncomputable def leadingTransverse (L : Submodule ℝ (EucSpace d))
    (x y : EucSpace d) : EucSpace d :=
  inner ℝ (L.starProjection x) (L.starProjection x) • L.starProjection y -
    inner ℝ (L.starProjection x) (L.starProjection y) • L.starProjection x

/-- Leading covectors see the original vector and its leading
projection identically. Source: arXiv:2411.04990v2, §4, `thm2`,
the orthogonal leading-space decomposition. -/
theorem inner_starProjection_of_mem_left (L : Submodule ℝ (EucSpace d))
    (v : EucSpace d) (hv : v ∈ L) (x : EucSpace d) :
    inner ℝ v (L.starProjection x) = inner ℝ v x := by
  rw [← L.inner_starProjection_left_eq_right, L.starProjection_eq_self_iff.mpr hv]

/-- The actual transverse residual belongs to the leading space.
Source: arXiv:2411.04990v2, §4, the leading space of `thm2`. -/
theorem leadingTransverse_mem (L : Submodule ℝ (EucSpace d)) (x y : EucSpace d) :
    leadingTransverse L x y ∈ L :=
  L.sub_mem (L.smul_mem _ (L.starProjection_apply_mem y))
    (L.smul_mem _ (L.starProjection_apply_mem x))

/-- The actual transverse residual is orthogonal to the original
leader, including its transverse component. Source: arXiv:2411.04990v2,
§4, the initial leading direction in `thm2`. -/
theorem leadingTransverse_inner_leader (L : Submodule ℝ (EucSpace d)) (x y : EucSpace d) :
    inner ℝ (leadingTransverse L x y) x = 0 := by
  rw [← inner_starProjection_of_mem_left L _ (leadingTransverse_mem L x y) x]
  simp only [leadingTransverse, inner_sub_left, real_inner_smul_left]
  rw [real_inner_comm (L.starProjection x) (L.starProjection y)]
  ring

/-- The actual residual's squared norm equals the leader projection's
squared norm times its original follower component.
Source: arXiv:2411.04990v2, §4, the leading projection of `thm2`. -/
theorem leadingTransverse_inner_self (L : Submodule ℝ (EucSpace d)) (x y : EucSpace d) :
    inner ℝ (leadingTransverse L x y) (leadingTransverse L x y) =
      inner ℝ (L.starProjection x) (L.starProjection x) *
        inner ℝ (leadingTransverse L x y) y := by
  conv_lhs => rhs; unfold leadingTransverse
  rw [inner_sub_right, real_inner_smul_right, real_inner_smul_right,
    inner_starProjection_of_mem_left L _ (leadingTransverse_mem L x y),
    inner_starProjection_of_mem_left L _ (leadingTransverse_mem L x y),
    leadingTransverse_inner_leader, mul_zero, sub_zero]

/-- A nonzero actual residual supplies the full covector initialization
used in the original two-token dynamics. Source: arXiv:2411.04990v2,
§4, `thm2`, outside its exceptional leading-line set. -/
theorem leadingTransverse_witness (L : Submodule ℝ (EucSpace d)) (x y : EucSpace d)
    (h : leadingTransverse L x y ≠ 0) :
    ∃ v ∈ L, inner ℝ v x = 0 ∧ 0 < inner ℝ v y := by
  refine ⟨leadingTransverse L x y, leadingTransverse_mem L x y,
    leadingTransverse_inner_leader L x y, ?_⟩
  have hp := real_inner_self_pos.mpr h
  rw [leadingTransverse_inner_self] at hp
  have ha : 0 ≤ inner ℝ (L.starProjection x) (L.starProjection x) := real_inner_self_nonneg
  nlinarith

/-- A vanishing actual residual confines the follower to the great
sphere of every leading covector orthogonal to the original leader.
Source: arXiv:2411.04990v2, §4, the exceptional initialization of `thm2`. -/
theorem inner_eq_zero_of_leadingTransverse_eq_zero (L : Submodule ℝ (EucSpace d))
    (x y v : EucSpace d) (hp : L.starProjection x ≠ 0) (hv : v ∈ L)
    (hx : inner ℝ v x = 0) (h : leadingTransverse L x y = 0) : inner ℝ v y = 0 := by
  have he := congrArg (fun z => inner ℝ v z) h
  simp only [leadingTransverse, inner_sub_right, real_inner_smul_right, inner_zero_right,
    inner_starProjection_of_mem_left L v hv, hx, mul_zero, sub_zero] at he
  exact (mul_eq_zero.mp he).resolve_left (real_inner_self_pos.mpr hp).ne'

/-- A leading space of dimension at least two has a nonzero leading
covector orthogonal to each original leader with nonzero leading
projection. Source: arXiv:2411.04990v2, §4, the multiplicity assumption
of `thm2`; this makes every exceptional two-token fiber a null set. -/
theorem exists_leading_transverse_normal (L : Submodule ℝ (EucSpace d))
    (hdim : 2 ≤ Module.finrank ℝ L) (x : EucSpace d) (hp : L.starProjection x ≠ 0) :
    ∃ v ∈ L, v ≠ 0 ∧ inner ℝ v x = 0 := by
  let p : L := ⟨L.starProjection x, L.starProjection_apply_mem x⟩
  have hp' : p ≠ 0 := fun h => hp (congrArg Subtype.val h)
  obtain ⟨y, hy⟩ := exists_linearIndependent_pair_of_one_lt_finrank
    (show 1 < Module.finrank ℝ L by omega) hp'
  let a := inner ℝ (p : EucSpace d) (p : EucSpace d)
  let b := inner ℝ (p : EucSpace d) (y : EucSpace d)
  let v : EucSpace d := a • (y : EucSpace d) - b • (p : EucSpace d)
  have ha : 0 < a := real_inner_self_pos.mpr hp
  have hv : v ∈ L := L.sub_mem (L.smul_mem _ y.2) (L.smul_mem _ p.2)
  refine ⟨v, hv, ?_, ?_⟩
  · intro h
    have he : (-b) • p + a • y = 0 := by
      apply Subtype.ext
      change (-b) • (p : EucSpace d) + a • (y : EucSpace d) = 0
      change a • (y : EucSpace d) - b • (p : EucSpace d) = 0 at h
      simpa only [neg_smul, sub_eq_add_neg, add_comm] using h
    exact ha.ne' ((LinearIndependent.pair_iff.mp hy) (-b) a he).2
  · rw [← inner_starProjection_of_mem_left L v hv x]
    change inner ℝ v (p : EucSpace d) = 0
    simp only [v, inner_sub_left, real_inner_smul_left]
    rw [real_inner_comm (p : EucSpace d) (y : EucSpace d)]
    change a * b - b * a = 0
    ring

/-- The actual two-projection residual is continuous jointly in both
original initial vectors. Source: arXiv:2411.04990v2, §4, the Borel
exceptional-set construction for the two-token case of `thm2`. -/
theorem continuous_leadingTransverse (L : Submodule ℝ (EucSpace d)) :
    Continuous (fun z : EucSpace d × EucSpace d => leadingTransverse L z.1 z.2) := by
  have hx : Continuous (fun z : EucSpace d × EucSpace d => L.starProjection z.1) :=
    L.starProjection.continuous.comp continuous_fst
  have hy : Continuous (fun z : EucSpace d × EucSpace d => L.starProjection z.2) :=
    L.starProjection.continuous.comp continuous_snd
  exact ((hx.inner (𝕜 := ℝ) hx).smul hy).sub ((hx.inner (𝕜 := ℝ) hy).smul hx)

/-- The two distinct unit coordinate vectors in the full leading
plane give a nonzero residual and a genuine transverse covector.
Source: arXiv:2411.04990v2, §4, the nondegenerate initialization in `thm2`. -/
example : ∃ (x y : EucSpace 2),
    2 ≤ Module.finrank ℝ (⊤ : Submodule ℝ (EucSpace 2)) ∧
    (⊤ : Submodule ℝ (EucSpace 2)).starProjection x ≠ 0 ∧
    leadingTransverse ⊤ x y ≠ 0 ∧
    y ∈ (⊤ : Submodule ℝ (EucSpace 2)) ∧ inner ℝ y x = 0 := by
  refine ⟨EuclideanSpace.single (0 : Fin 2) 1, EuclideanSpace.single (1 : Fin 2) 1, ?_⟩
  simp [leadingTransverse, Submodule.starProjection_top, ContinuousLinearMap.id_apply,
    EuclideanSpace.inner_single_left]

/-- A collinear original pair realizes the exceptional-set premises,
with nonzero leader projection and a nonzero orthogonal leading normal.
Source: arXiv:2411.04990v2, §4, the exceptional fibers of `thm2`. -/
example : ∃ x y v : EucSpace 2,
    (⊤ : Submodule ℝ (EucSpace 2)).starProjection x ≠ 0 ∧
    v ∈ (⊤ : Submodule ℝ (EucSpace 2)) ∧ inner ℝ v x = 0 ∧
    leadingTransverse ⊤ x y = 0 := by
  refine ⟨EuclideanSpace.single (0 : Fin 2) 1, EuclideanSpace.single (0 : Fin 2) 1,
    EuclideanSpace.single (1 : Fin 2) 1, ?_⟩
  simp [leadingTransverse, Submodule.starProjection_top, ContinuousLinearMap.id_apply,
    EuclideanSpace.inner_single_left]

end Transformer.Causal
