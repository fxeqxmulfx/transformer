/-
# Common original unit axes transverse to a prescribed plane

ArXiv:2411.04551v3, §1, `thm: targets.atoms`, and §3.2,
the rank-one attention isolation mechanism. A protected atomic transfer
uses an axis orthogonal to both selected endpoints, with nonzero
pairing against every unselected mean. The corresponding geometric
hypothesis is that no unselected mean lies in their linear plane.

Orthogonal subtraction gives a nonzero test for each such mean.
A finite affine-line argument avoids all forbidden scalar pairings
while retaining both orthogonality constraints. Normalization then
gives a single actual unit attention axis for the whole finite family.
No original mean or scalar multiple is redefined by these statements.
The selected directions may be linearly dependent; the same construction
then treats their one-dimensional span.
-/

import Transformer.Interpolation.Section3_CommonTransverseAxis

open Real MeasureTheory Set

namespace Transformer.Interpolation

variable {d : ℕ} {ι : Type*}

/-- A vector outside the selected pair's actual linear plane has
nonzero pairing with a vector orthogonal to both selected directions.
Source auxiliary: arXiv:2411.04551v3, §3.2, finite isolation for
`thm: targets.atoms` in §1. -/
theorem exists_plane_transverse_vector (p : SSphere d) (q m : EucSpace d)
    (hm : ∀ α β : ℝ, m ≠ α • (p : EucSpace d) + β • q) :
    ∃ v : EucSpace d, inner ℝ v (p : EucSpace d) = 0 ∧ inner ℝ v q = 0 ∧ inner ℝ v m ≠ 0 := by
  have hpp : inner ℝ (p : EucSpace d) (p : EucSpace d) = 1 := by
    rw [real_inner_self_eq_norm_sq, mem_sphere_zero_iff_norm.mp p.2]
    norm_num
  let a := inner ℝ q (p : EucSpace d)
  let u := q - a • (p : EucSpace d)
  have hup : inner ℝ u (p : EucSpace d) = 0 := by
    simp only [u, inner_sub_left, real_inner_smul_left, hpp, mul_one, a, sub_self]
  by_cases hu : u = 0
  · have hq : q = a • (p : EucSpace d) := sub_eq_zero.mp hu
    obtain ⟨v, hvp, hvm⟩ := exists_transverse_vector_of_noncolinear p m
      (fun γ => by simpa only [zero_smul, add_zero] using hm γ 0)
    refine ⟨v, hvp, ?_, hvm⟩
    rw [hq, real_inner_smul_right, hvp, mul_zero]
  have hpu : inner ℝ (p : EucSpace d) u = 0 := (real_inner_comm _ _).trans hup
  have huu : inner ℝ u u ≠ 0 := ne_of_gt (real_inner_self_pos.mpr hu)
  let α := inner ℝ m (p : EucSpace d)
  let β := inner ℝ m u / inner ℝ u u
  let v := m - α • (p : EucSpace d) - β • u
  have hvp : inner ℝ v (p : EucSpace d) = 0 := by
    simp only [v, inner_sub_left, real_inner_smul_left, hpp, hup,
      mul_one, mul_zero, α, sub_self]
  have hvu : inner ℝ v u = 0 := by
    simp only [v, inner_sub_left, real_inner_smul_left, hpu, mul_zero, sub_zero]
    dsimp only [β]
    rw [div_mul_cancel₀ _ huu, sub_self]
  have hq : q = u + a • (p : EucSpace d) := by dsimp only [u]; module
  have hvq : inner ℝ v q = 0 := by
    rw [hq, inner_add_right, real_inner_smul_right, hvu, hvp, mul_zero, add_zero]
  have hv : v ≠ 0 := by
    intro hv0
    change m - α • (p : EucSpace d) - β • u = 0 at hv0
    have hm' : m = β • u + α • (p : EucSpace d) :=
      sub_eq_iff_eq_add.mp (sub_eq_zero.mp hv0)
    apply hm (α - β * a) β
    rw [hm']
    dsimp only [u]
    module
  refine ⟨v, hvp, hvq, ?_⟩
  have hdecomp : m = v + α • (p : EucSpace d) + β • u := by dsimp only [v]; module
  rw [hdecomp]
  simp only [inner_add_right, real_inner_smul_right, hvp, hvu, mul_zero, add_zero]
  exact ne_of_gt (real_inner_self_pos.mpr hv)

/-- Finitely many original tests can be avoided simultaneously while
retaining both selected orthogonality constraints. Source auxiliary:
arXiv:2411.04551v3, §3.2, the common attention isolation axis. -/
theorem exists_common_plane_transverse_vector (p q : EucSpace d) (m : ι → EucSpace d) (S : Finset ι)
    (hm : ∀ i ∈ S, ∃ u : EucSpace d,
      inner ℝ u p = 0 ∧ inner ℝ u q = 0 ∧ inner ℝ u (m i) ≠ 0) :
    ∃ v : EucSpace d, inner ℝ v p = 0 ∧ inner ℝ v q = 0 ∧ ∀ i ∈ S, inner ℝ v (m i) ≠ 0 := by
  classical
  revert hm
  induction S using Finset.induction_on with
  | empty =>
    intro hm
    exact ⟨0, by simp, by simp, by simp⟩
  | @insert i S hi ih =>
    intro hm
    obtain ⟨v, hvp, hvq, hv⟩ := ih (fun j hj => hm j (Finset.mem_insert_of_mem hj))
    obtain ⟨u, hup, huq, hu⟩ := hm i (Finset.mem_insert_self i S)
    obtain ⟨a, ha⟩ := ((insert i S).image (fun j => -inner ℝ v (m j) / inner ℝ u (m j))).exists_notMem
    refine ⟨v + a • u, ?_, ?_, ?_⟩
    · rw [inner_add_left, real_inner_smul_left, hvp, hup, mul_zero, add_zero]
    · rw [inner_add_left, real_inner_smul_left, hvq, huq, mul_zero, add_zero]
    · intro j hj
      rw [inner_add_left, real_inner_smul_left]
      by_cases huj : inner ℝ u (m j) = 0
      · have hji : j ≠ i := by intro he; subst j; exact hu huj
        have hjs : j ∈ S := (Finset.mem_insert.mp hj).resolve_left hji
        simpa only [huj, mul_zero, add_zero] using hv j hjs
      · intro he
        apply ha
        exact Finset.mem_image.mpr ⟨j, hj, (div_eq_iff huj).mpr (by linarith)⟩

/-- A nonempty finite family of original means outside the selected
plane admits one actual unit attention axis orthogonal to both selected
vectors and nonorthogonal to every mean. Source auxiliary:
arXiv:2411.04551v3, §1 and §3.2, protected finite point matching. -/
theorem exists_common_unit_plane_normal [Fintype ι] [Nonempty ι]
    (p : SSphere d) (q : EucSpace d) (m : ι → EucSpace d)
    (hm : ∀ i, ∀ α β : ℝ, m i ≠ α • (p : EucSpace d) + β • q) :
    ∃ w : SSphere d, inner ℝ (w : EucSpace d) (p : EucSpace d) = 0 ∧
      inner ℝ (w : EucSpace d) q = 0 ∧ ∀ i, inner ℝ (w : EucSpace d) (m i) ≠ 0 := by
  obtain ⟨v, hvp, hvq, hv⟩ := exists_common_plane_transverse_vector (p : EucSpace d) q m Finset.univ
    (fun i _ => exists_plane_transverse_vector p q (m i) (hm i))
  have hne : v ≠ 0 := by
    intro he
    have hh := hv (Classical.arbitrary ι) (Finset.mem_univ _)
    rw [he, inner_zero_left] at hh
    exact hh rfl
  let w : SSphere d := ⟨‖v‖⁻¹ • v, mem_sphere_zero_iff_norm.mpr (norm_smul_inv_norm hne)⟩
  refine ⟨w, ?_, ?_, fun i => ?_⟩
  · change inner ℝ (‖v‖⁻¹ • v) (p : EucSpace d) = 0
    rw [real_inner_smul_left, hvp, mul_zero]
  · change inner ℝ (‖v‖⁻¹ • v) q = 0
    rw [real_inner_smul_left, hvq, mul_zero]
  · change inner ℝ (‖v‖⁻¹ • v) (m i) ≠ 0
    rw [real_inner_smul_left]
    exact mul_ne_zero (inv_ne_zero (norm_ne_zero_iff.mpr hne)) (hv i (Finset.mem_univ i))

/-- The common plane-normal axis has nonzero actual mean component
on every other original law. Source auxiliary: arXiv:2411.04551v3,
§1 and §3.2, the original nonlinear attention isolation. -/
theorem exists_common_unit_mean_plane_normal [Fintype ι] [Nonempty ι]
    (p : SSphere d) (q : EucSpace d) (ν : ι → Perspective.ProbSphere d)
    (hν : ∀ i, ∀ α β : ℝ, barycenter d (ν i) ≠ α • (p : EucSpace d) + β • q) :
    ∃ w : SSphere d, inner ℝ (w : EucSpace d) (p : EucSpace d) = 0 ∧
      inner ℝ (w : EucSpace d) q = 0 ∧ ∀ i, meanAxisMoment (w : EucSpace d) (ν i) ≠ 0 :=
  exists_common_unit_plane_normal p q (fun i => barycenter d (ν i)) hν

/-- A genuine coordinate plane and two original vectors outside it
jointly realize every single-vector, finite-test and normalization
hypothesis, with a nonempty two-element family. -/
example : ∃ (p : SSphere 3) (q : EucSpace 3) (m : Fin 2 → EucSpace 3),
    (∀ i, ∀ α β : ℝ, m i ≠ α • (p : EucSpace 3) + β • q) ∧
    (∀ i ∈ (Finset.univ : Finset (Fin 2)), ∃ u : EucSpace 3,
      inner ℝ u (p : EucSpace 3) = 0 ∧ inner ℝ u q = 0 ∧ inner ℝ u (m i) ≠ 0) := by
  let p : SSphere 3 := ⟨PiLp.single 2 (0 : Fin 3) (1 : ℝ), by simp⟩
  let q : EucSpace 3 := PiLp.single 2 (1 : Fin 3) 1
  let z : EucSpace 3 := PiLp.single 2 (2 : Fin 3) 1
  let m (i : Fin 2) := if i = 0 then z else (p : EucSpace 3) + z
  have hm (i : Fin 2) (α β : ℝ) : m i ≠ α • (p : EucSpace 3) + β • q := by
    intro he
    have hh := congrArg (fun x : EucSpace 3 => x (2 : Fin 3)) he
    fin_cases i <;> norm_num [m, p, q, z] at hh
  exact ⟨p, q, m, hm, fun i _ => exists_plane_transverse_vector p q (m i) (hm i)⟩

/-- Actual nonzero original Dirac probabilities also realize the
mean-plane hypotheses for the same genuine coordinate plane. -/
example : ∃ (p : SSphere 3) (q : EucSpace 3) (ν : Fin 1 → Perspective.ProbSphere 3),
    ∀ i, ∀ α β : ℝ, barycenter 3 (ν i) ≠ α • (p : EucSpace 3) + β • q := by
  let p : SSphere 3 := ⟨PiLp.single 2 (0 : Fin 3) (1 : ℝ), by simp⟩
  let q : EucSpace 3 := PiLp.single 2 (1 : Fin 3) 1
  let z : SSphere 3 := ⟨PiLp.single 2 (2 : Fin 3) (1 : ℝ), by simp⟩
  let ν : Fin 1 → Perspective.ProbSphere 3 := fun _ => Perspective.diracProb 3 z
  refine ⟨p, q, ν, ?_⟩
  intro i α β he
  have hh := congrArg (fun x : EucSpace 3 => x (2 : Fin 3)) he
  norm_num [ν, barycenter, Perspective.diracProb, p, q, z] at hh

end Transformer.Interpolation
