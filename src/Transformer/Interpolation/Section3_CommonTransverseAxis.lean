module

public import Transformer.Interpolation.Section3_MeanAxisField

/-!
# A common transverse axis for a finite family of original means

ArXiv:2411.04551v3, §3.2, Step 1 of `lem: induction.barycenter`.
The source cycles through an orthonormal basis transverse to the target
mean. For a finite family, one can instead choose a single transverse
axis whose inner product with every other mean is nonzero.

Each non-collinear vector has a nonzero orthogonal projection onto the
target's orthogonal complement. A finite induction avoids all forbidden
linear hyperplanes: on an affine line there is at most one forbidden
scalar for each nonzero restriction. Normalizing the resulting vector
gives one actual spherical attention axis for all other input laws.

This alternative isolation argument retains the original non-collinearity
hypotheses and reduces the number of attention phases needed by the proof.
-/

@[expose] public section

open Real MeasureTheory Set
open scoped Topology

namespace Transformer.Interpolation

variable {d : ℕ} {ι : Type*}

/-- Finitely many nonzero transverse linear tests can be avoided
simultaneously. Auxiliary for arXiv:2411.04551v3, §3.2, Step 1. -/
theorem exists_common_transverse_vector (p : EucSpace d) (m : ι → EucSpace d) (S : Finset ι)
    (hm : ∀ i ∈ S, ∃ u : EucSpace d, inner ℝ u p = 0 ∧ inner ℝ u (m i) ≠ 0) :
    ∃ v : EucSpace d, inner ℝ v p = 0 ∧ ∀ i ∈ S, inner ℝ v (m i) ≠ 0 := by
  classical
  revert hm
  induction S using Finset.induction_on with
  | empty =>
    intro hm
    exact ⟨0, by simp, by simp⟩
  | @insert i S hi ih =>
    intro hm
    obtain ⟨v, hvp, hv⟩ := ih (fun j hj => hm j (Finset.mem_insert_of_mem hj))
    obtain ⟨u, hup, hu⟩ := hm i (Finset.mem_insert_self i S)
    obtain ⟨a, ha⟩ := ((insert i S).image (fun j => -inner ℝ v (m j) / inner ℝ u (m j))).exists_notMem
    refine ⟨v + a • u, ?_, ?_⟩
    · rw [inner_add_left, real_inner_smul_left, hvp, hup, mul_zero, add_zero]
    · intro j hj
      rw [inner_add_left, real_inner_smul_left]
      by_cases huj : inner ℝ u (m j) = 0
      · have hji : j ≠ i := by intro he; subst j; exact hu huj
        have hjs : j ∈ S := (Finset.mem_insert.mp hj).resolve_left hji
        simpa only [huj, mul_zero, add_zero] using hv j hjs
      · intro he
        apply ha
        exact Finset.mem_image.mpr ⟨j, hj, (div_eq_iff huj).mpr (by linarith)⟩

/-- The actual orthogonal projection of a non-collinear vector has
nonzero transverse pairing. Source: arXiv:2411.04551v3, §3.2, Step 1. -/
theorem exists_transverse_vector_of_noncolinear (p : SSphere d) (m : EucSpace d)
    (hm : ∀ γ : ℝ, m ≠ γ • (p : EucSpace d)) :
    ∃ u : EucSpace d, inner ℝ u (p : EucSpace d) = 0 ∧ inner ℝ u m ≠ 0 := by
  let a := inner ℝ m (p : EucSpace d)
  let u := m - a • (p : EucSpace d)
  have hp : inner ℝ (p : EucSpace d) (p : EucSpace d) = 1 := by
    rw [real_inner_self_eq_norm_sq, mem_sphere_zero_iff_norm.mp p.2]
    norm_num
  have hup : inner ℝ u (p : EucSpace d) = 0 := by
    dsimp only [u, a]
    rw [inner_sub_left, real_inner_smul_left, hp, mul_one, sub_self]
  have hu : u ≠ 0 := by
    intro he
    exact hm a (sub_eq_zero.mp he)
  refine ⟨u, hup, ?_⟩
  have hmdecomp : m = u + a • (p : EucSpace d) := by dsimp only [u]; module
  have hpos := real_inner_self_pos.mpr hu
  rw [hmdecomp, inner_add_right, real_inner_smul_right, hup, mul_zero, add_zero]
  exact ne_of_gt hpos

/-- A nonempty finite family admits one common unit transverse axis.
Auxiliary for arXiv:2411.04551v3, §3.2, Step 1. -/
theorem exists_common_unit_transverse_axis [Fintype ι] [Nonempty ι]
    (p : SSphere d) (m : ι → EucSpace d)
    (hm : ∀ i, ∀ γ : ℝ, m i ≠ γ • (p : EucSpace d)) :
    ∃ w : SSphere d, inner ℝ (w : EucSpace d) (p : EucSpace d) = 0 ∧
      ∀ i, inner ℝ (w : EucSpace d) (m i) ≠ 0 := by
  obtain ⟨v, hvp, hv⟩ := exists_common_transverse_vector (p : EucSpace d) m Finset.univ
    (fun i _ => exists_transverse_vector_of_noncolinear p (m i) (hm i))
  have hne : v ≠ 0 := by
    intro he
    have hh := hv (Classical.arbitrary ι) (Finset.mem_univ _)
    rw [he, inner_zero_left] at hh
    exact hh rfl
  let w : SSphere d := ⟨‖v‖⁻¹ • v, mem_sphere_zero_iff_norm.mpr (norm_smul_inv_norm hne)⟩
  refine ⟨w, ?_, fun i => ?_⟩
  · change inner ℝ (‖v‖⁻¹ • v) (p : EucSpace d) = 0
    rw [real_inner_smul_left, hvp, mul_zero]
  · change inner ℝ (‖v‖⁻¹ • v) (m i) ≠ 0
    rw [real_inner_smul_left]
    exact mul_ne_zero (inv_ne_zero (norm_ne_zero_iff.mpr hne)) (hv i (Finset.mem_univ i))

/-- Normalization of the original target vector preserves every
non-collinearity condition. Auxiliary for arXiv:2411.04551v3, §3.2, Step 1. -/
theorem exists_common_unit_transverse_axis_of_nonzero [Fintype ι] [Nonempty ι]
    (p : EucSpace d) (hp : p ≠ 0) (m : ι → EucSpace d)
    (hm : ∀ i, ∀ γ : ℝ, m i ≠ γ • p) :
    ∃ w : SSphere d, inner ℝ (w : EucSpace d) p = 0 ∧
      ∀ i, inner ℝ (w : EucSpace d) (m i) ≠ 0 := by
  let pn : SSphere d := ⟨‖p‖⁻¹ • p, mem_sphere_zero_iff_norm.mpr (norm_smul_inv_norm hp)⟩
  have hpn (i : ι) (γ : ℝ) : m i ≠ γ • (pn : EucSpace d) := by
    simpa only [pn, smul_smul] using hm i (γ * ‖p‖⁻¹)
  obtain ⟨w, hwp, hw⟩ := exists_common_unit_transverse_axis pn m hpn
  refine ⟨w, ?_, hw⟩
  change inner ℝ (w : EucSpace d) (‖p‖⁻¹ • p) = 0 at hwp
  rw [real_inner_smul_right] at hwp
  exact (mul_eq_zero.mp hwp).resolve_left (inv_ne_zero (norm_ne_zero_iff.mpr hp))

/-- One original rank-one attention axis has nonzero axial mean on
every other input. Auxiliary for arXiv:2411.04551v3, §3.2, Step 1. -/
theorem exists_common_meanAxis_direction [Fintype ι] [Nonempty ι]
    (p : EucSpace d) (hp : p ≠ 0) (μ : ι → Perspective.ProbSphere d)
    (hm : ∀ i, ∀ γ : ℝ, barycenter d (μ i) ≠ γ • p) :
    ∃ w : SSphere d, inner ℝ (w : EucSpace d) p = 0 ∧
      ∀ i, meanAxisMoment (w : EucSpace d) (μ i) ≠ 0 :=
  exists_common_unit_transverse_axis_of_nonzero p hp (fun i => barycenter d (μ i)) hm

/-- Two distinct non-collinear vectors and a unit target jointly realize
all finite-avoidance and transverse-projection hypotheses. The index
family has two elements and both transverse pairings are nonzero. -/
example : ∃ (p : SSphere 2) (m : Fin 2 → EucSpace 2),
    (p : EucSpace 2) ≠ 0 ∧ (∀ i, ∀ γ : ℝ, m i ≠ γ • (p : EucSpace 2)) ∧
    (∀ i ∈ (Finset.univ : Finset (Fin 2)),
      ∃ u : EucSpace 2, inner ℝ u (p : EucSpace 2) = 0 ∧ inner ℝ u (m i) ≠ 0) := by
  let p : SSphere 2 := basePoint 1
  let z : EucSpace 2 := EuclideanSpace.single 1 1
  let m (i : Fin 2) : EucSpace 2 := if i = 0 then z else (p : EucSpace 2) + z
  have hm (i : Fin 2) (γ : ℝ) : m i ≠ γ • (p : EucSpace 2) := by
    intro he
    have hh := congrArg (fun x : EucSpace 2 => x 1) he
    fin_cases i <;> simp [m, p, z, basePoint] at hh
  refine ⟨p, m, ?_, hm, fun i _ => exists_transverse_vector_of_noncolinear p (m i) (hm i)⟩
  intro he
  have hn := mem_sphere_zero_iff_norm.mp p.2
  rw [he, norm_zero] at hn
  exact zero_ne_one hn

/-- Actual nonzero probability laws realize every mean-family hypothesis.
Their common mean is transverse to the nonzero original target vector. -/
example : ∃ (p : EucSpace 2) (μ : Fin 2 → Perspective.ProbSphere 2),
    p ≠ 0 ∧ ∀ i, ∀ γ : ℝ, barycenter 2 (μ i) ≠ γ • p := by
  let p : SSphere 2 := basePoint 1
  let z : SSphere 2 := ⟨EuclideanSpace.single 1 1, by simp⟩
  let μ : Fin 2 → Perspective.ProbSphere 2 := fun _ => Perspective.diracProb 2 z
  refine ⟨p, μ, ?_, ?_⟩
  · intro he
    have hn := mem_sphere_zero_iff_norm.mp p.2
    rw [he, norm_zero] at hn
    exact zero_ne_one hn
  · intro i γ he
    have hb : barycenter 2 (μ i) = (z : EucSpace 2) := by simp [μ, barycenter, Perspective.diracProb]
    rw [hb] at he
    have hh := congrArg (fun x : EucSpace 2 => x 1) he
    simp [p, z, basePoint] at hh

end Transformer.Interpolation
