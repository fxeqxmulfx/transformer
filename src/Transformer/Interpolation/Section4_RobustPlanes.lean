/-
# Stable original plane exclusions and concentrated barycenters

ArXiv:2411.04551v3, §1, `thm: targets.atoms`, and §3.2,
protected finite transfers. Three independent original vectors retain
their independence under small simultaneous perturbations. Thus a
mean excluded from a selected-target plane stays excluded when the
selected pole and the protected mean vary slightly after concentration.

The geometric statement uses genuine linear independence in the
original ambient Euclidean space. A finite tuple of independent
vectors is an interior point of the set of independent tuples.
A smaller common coordinate radius gives closed perturbation bounds.

The original barycenter of a law supported near a unit center is
within the same Euclidean radius of that center. This links the open
plane exclusion to actual original concentrated probabilities.
-/

import Transformer.Interpolation.Section4_ConcentratedReplacement
import Mathlib.Analysis.Normed.Module.FiniteDimension
import Mathlib.LinearAlgebra.LinearIndependent.Lemmas

open Real MeasureTheory Set

namespace Transformer.Interpolation

variable {d : ℕ}

/-- Actual non-collinear selected vectors and an excluded original
mean form an independent triple. Source auxiliary:
arXiv:2411.04551v3, §1 and §3.2, protected plane choices. -/
theorem independent_triple_of_plane_exclusion (p q m : EucSpace d) (hp : p ≠ 0)
    (hq : ∀ α : ℝ, q ≠ α • p) (hm : ∀ α β : ℝ, m ≠ α • p + β • q) :
    LinearIndependent ℝ (Fin.cons m (![q, p] : Fin 2 → EucSpace d)) := by
  have hpair : LinearIndependent ℝ (![q, p] : Fin 2 → EucSpace d) :=
    linearIndependent_fin2.mpr ⟨hp, fun α he => hq α he.symm⟩
  apply linearIndependent_finCons.mpr
  refine ⟨hpair, ?_⟩
  intro hmem
  have hrange : Set.range (![q, p] : Fin 2 → EucSpace d) = {q, p} := by
    ext x
    constructor
    · rintro ⟨k, rfl⟩
      fin_cases k <;> simp
    · intro hx
      simp only [Set.mem_insert_iff, Set.mem_singleton_iff] at hx
      rcases hx with rfl | rfl
      · exact ⟨0, rfl⟩
      · exact ⟨1, rfl⟩
  rw [hrange] at hmem
  obtain ⟨α, β, he⟩ := Submodule.mem_span_pair.mp hmem
  apply hm β α
  exact he.symm.trans (add_comm _ _)

/-- Independence of the actual original triple excludes its first
vector from the plane of the other two. Source auxiliary:
arXiv:2411.04551v3, §1 and §3.2, reverse plane interpretation. -/
theorem plane_exclusion_of_independent_triple (p q m : EucSpace d)
    (h : LinearIndependent ℝ (Fin.cons m (![q, p] : Fin 2 → EucSpace d))) :
    ∀ α β : ℝ, m ≠ α • p + β • q := by
  have hm := (linearIndependent_finCons.mp h).2
  let P := Submodule.span ℝ (Set.range (![q, p] : Fin 2 → EucSpace d))
  have hp : p ∈ P := Submodule.subset_span ⟨1, rfl⟩
  have hq : q ∈ P := Submodule.subset_span ⟨0, rfl⟩
  intro α β he
  apply hm
  rw [he]
  exact P.add_mem (P.smul_mem α hp) (P.smul_mem β hq)

/-- An original plane exclusion persists under one positive common
closed perturbation radius for all three vectors. Source auxiliary:
arXiv:2411.04551v3, §1 and §3.2, robust choices after concentration. -/
theorem exists_robust_plane_exclusion_radius (p q m : EucSpace d) (hp : p ≠ 0)
    (hq : ∀ α : ℝ, q ≠ α • p) (hm : ∀ α β : ℝ, m ≠ α • p + β • q) :
    ∃ r : ℝ, 0 < r ∧ ∀ p' q' m' : EucSpace d,
      dist p' p ≤ r → dist q' q ≤ r → dist m' m ≤ r →
      ∀ α β : ℝ, m' ≠ α • p' + β • q' := by
  let v : Fin 3 → EucSpace d := Fin.cons m ![q, p]
  have hv : LinearIndependent ℝ v := independent_triple_of_plane_exclusion p q m hp hq hm
  have hopen : IsOpen {f : Fin 3 → EucSpace d | LinearIndependent ℝ f} :=
    isOpen_setOfPred_linearIndependent
  obtain ⟨R, hR, hball⟩ := Metric.isOpen_iff.mp hopen v hv
  refine ⟨R / 2, half_pos hR, ?_⟩
  intro p' q' m' hp' hq' hm'
  apply plane_exclusion_of_independent_triple
  apply hball
  apply Metric.mem_ball.mpr
  apply (dist_pi_lt_iff hR).mpr
  intro k
  fin_cases k
  · change dist m' m < R
    exact hm'.trans_lt (half_lt_self hR)
  · change dist q' q < R
    exact hq'.trans_lt (half_lt_self hR)
  · change dist p' p < R
    exact hp'.trans_lt (half_lt_self hR)

/-- Every finite family of original protected planes has a common
positive perturbation radius. Empty families are permitted. Source
auxiliary: arXiv:2411.04551v3, §1 and §3.2, uniform finite geometry. -/
theorem exists_finite_robust_plane_radius {ι : Type*} [Fintype ι]
    (p q m : ι → EucSpace d) (hp : ∀ i, p i ≠ 0)
    (hq : ∀ i, ∀ α : ℝ, q i ≠ α • p i)
    (hm : ∀ i, ∀ α β : ℝ, m i ≠ α • p i + β • q i) :
    ∃ r : ℝ, 0 < r ∧ ∀ i, ∀ p' q' m' : EucSpace d,
      dist p' (p i) ≤ r → dist q' (q i) ≤ r → dist m' (m i) ≤ r →
      ∀ α β : ℝ, m' ≠ α • p' + β • q' := by
  classical
  choose ρ hρ hrob using fun i => exists_robust_plane_exclusion_radius (p i) (q i) (m i)
    (hp i) (hq i) (hm i)
  have hmin (S : Finset ι) : ∃ r : ℝ, 0 < r ∧ ∀ i ∈ S, r ≤ ρ i := by
    induction S using Finset.induction_on with
    | empty =>
      refine ⟨1, one_pos, ?_⟩
      intro i hi
      cases hi
    | @insert i S _ ih =>
      obtain ⟨r, hr, hle⟩ := ih
      have hρi := hρ i
      refine ⟨min r (ρ i), by positivity, ?_⟩
      intro k hk
      rcases Finset.mem_insert.mp hk with rfl | hk
      · exact min_le_right _ _
      · exact (min_le_left _ _).trans (hle k hk)
  obtain ⟨r, hr, hle⟩ := hmin Finset.univ
  refine ⟨r, hr, ?_⟩
  intro i p' q' m' hp' hq' hm'
  exact hrob i p' q' m' (hp'.trans (hle i (Finset.mem_univ i)))
    (hq'.trans (hle i (Finset.mem_univ i))) (hm'.trans (hle i (Finset.mem_univ i)))

/-- The original barycenter of a concentrated probability remains
within its actual support radius of the unit center. Source auxiliary:
arXiv:2411.04551v3, §1 and §3.2, means after selective contraction. -/
theorem norm_barycenter_sub_le_support_radius (ν : Perspective.ProbSphere d)
    (p : SSphere d) (r : ℝ)
    (hν : (ν : Measure (SSphere d)).support ⊆ Metric.closedBall p r) :
    ‖barycenter d ν - (p : EucSpace d)‖ ≤ r := by
  have hi : Integrable (fun x : SSphere d => (x : EucSpace d)) (ν : Measure (SSphere d)) :=
    continuous_subtype_val.integrable_of_hasCompactSupport (HasCompactSupport.of_compactSpace _)
  have he : (∫ x : SSphere d, ((x : EucSpace d) - (p : EucSpace d))
      ∂(ν : Measure (SSphere d))) = barycenter d ν - (p : EucSpace d) := by
    rw [integral_sub hi (integrable_const _), integral_const, probReal_univ, one_smul]
    rfl
  rw [← he]
  have hb : ∀ᵐ x : SSphere d ∂(ν : Measure (SSphere d)),
      ‖(x : EucSpace d) - (p : EucSpace d)‖ ≤ r := by
    filter_upwards [Measure.support_mem_ae (μ := (ν : Measure (SSphere d)))] with x hx
    simpa only [Metric.mem_closedBall, Subtype.dist_eq, dist_eq_norm] using hν hx
  simpa only [probReal_univ, mul_one] using norm_integral_le_of_norm_le_const hb

/-- Three genuine coordinate vectors are independent and exclude
each other from the indicated planes. A genuine coordinate Dirac
also realizes the concentrated-probability hypothesis. -/
example : ∃ (p : SSphere 3) (q m : EucSpace 3) (ν : Perspective.ProbSphere 3),
    (p : EucSpace 3) ≠ 0 ∧ (∀ α : ℝ, q ≠ α • (p : EucSpace 3)) ∧
      (∀ α β : ℝ, m ≠ α • (p : EucSpace 3) + β • q) ∧
      (ν : Measure (SSphere 3)).support ⊆ Metric.closedBall p 1 := by
  let p : SSphere 3 := ⟨PiLp.single 2 (0 : Fin 3) (1 : ℝ), by simp⟩
  let q : EucSpace 3 := PiLp.single 2 (1 : Fin 3) 1
  let m : EucSpace 3 := PiLp.single 2 (2 : Fin 3) 1
  refine ⟨p, q, m, Perspective.diracProb 3 p, ?_, ?_, ?_, ?_⟩
  · intro he
    have hh := congrArg (fun x : EucSpace 3 => x (0 : Fin 3)) he
    norm_num [p] at hh
  · intro α he
    have hh := congrArg (fun x : EucSpace 3 => x (1 : Fin 3)) he
    norm_num [p, q] at hh
  · intro α β he
    have hh := congrArg (fun x : EucSpace 3 => x (2 : Fin 3)) he
    norm_num [p, q, m] at hh
  · intro x hx
    rw [eq_of_mem_support_dirac hx]
    simp

/-- A nonempty finite family of genuine coordinate triples satisfies
all simultaneous plane-avoidance hypotheses. -/
example : ∃ p q m : Fin 1 → EucSpace 3, (∀ i, p i ≠ 0) ∧
    (∀ i, ∀ α : ℝ, q i ≠ α • p i) ∧
      ∀ i, ∀ α β : ℝ, m i ≠ α • p i + β • q i := by
  let e (j : Fin 3) : EucSpace 3 := PiLp.single 2 j 1
  refine ⟨(fun _ => e 0), (fun _ => e 1), (fun _ => e 2), ?_, ?_, ?_⟩
  · intro i he
    have hh := congrArg (fun x : EucSpace 3 => x (0 : Fin 3)) he
    norm_num [e] at hh
  · intro i α he
    have hh := congrArg (fun x : EucSpace 3 => x (1 : Fin 3)) he
    norm_num [e] at hh
  · intro i α β he
    have hh := congrArg (fun x : EucSpace 3 => x (2 : Fin 3)) he
    norm_num [e] at hh

end Transformer.Interpolation
