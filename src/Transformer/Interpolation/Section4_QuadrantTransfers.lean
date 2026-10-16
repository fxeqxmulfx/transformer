/-
# Exact protected transfers inside the original positive quadrant

ArXiv:2411.04551v3, §1, `thm: targets.atoms`, and §3.2,
the rank-one attention isolation construction.
When the selected endpoints lie in the positive quadrant and every
unselected original mean lies outside their plane, the proved common
plane-normal axis supplies every attention separation hypothesis.
Orthogonality to the positive selected input excludes both poles from
all unselected positive-quadrant supports.

Three actual full-network phases then move the selected unit atom
exactly and restore each unselected original law. Their prescribed-time
realization has four constant pieces, zero scores, constructed global
weak solutions, and the same endpoints for every original weak law.
The hypotheses have genuine positive, nonpolar, three-dimensional data.
-/

import Transformer.Interpolation.Section4_ProtectedTransfers
import Transformer.Interpolation.Section4_TransversePlanes
import Transformer.Interpolation.Section3_BarycenterContraction

open Real MeasureTheory Set

namespace Transformer.Interpolation

variable {d : ℕ} {ι : Type*}

/-- Two original positive-quadrant unit points cannot be antipodes.
Source auxiliary: arXiv:2411.04551v3, §3.2, the quadrant reduction. -/
theorem quadrant_ne_antipode (i : Fin d) (p q : SSphere d)
    (hp : p ∈ positiveQuadrant d) (hq : q ∈ positiveQuadrant d) : q ≠ antipode d p := by
  intro he
  have hh := congrArg (fun z : SSphere d => (z : EucSpace d) i) he
  change (q : EucSpace d) i = -(p : EucSpace d) i at hh
  have hp' : 0 < (p : EucSpace d) i := hp i
  have hq' : 0 < (q : EucSpace d) i := hq i
  linarith

/-- Actual positive-quadrant data admit a protected exact atomic
transfer once every other original mean lies outside the selected
plane. Source auxiliary: arXiv:2411.04551v3, §1 and §3.2. -/
theorem exists_quadrant_transfer_phases [Fintype ι] [Nonempty ι]
    (i : Fin d) (p q : SSphere d) (hp : p ∈ positiveQuadrant d)
    (hq : q ∈ positiveQuadrant d) (hpq : p ≠ q) (η : ι → Perspective.ProbSphere d)
    (hQ : ∀ k, (η k : Measure (SSphere d)).support ⊆ positiveQuadrant d)
    (hplane : ∀ k, ∀ α β : ℝ, barycenter d (η k) ≠ α • (p : EucSpace d) + β • (q : EucSpace d)) :
    ∃ L : List (AutonomousPhase d), L.length = 3 ∧
      (∀ P ∈ L, IsAutonomousPhase P ∧ P.1.B = 0) ∧
      phaseSequenceLaw L (Perspective.diracProb d p) = Perspective.diracProb d q ∧
      ∀ k, phaseSequenceLaw L (η k) = η k := by
  obtain ⟨w, hwp, hwq, hw⟩ := exists_common_unit_mean_plane_normal p (q : EucSpace d) η hplane
  exact exists_protected_transfer_phases i p q w hpq (quadrant_ne_antipode i p q hp hq)
    hwp hwq η hw (fun k => orthogonal_poles_not_support i (p : EucSpace d) hp w hwp (η k) (hQ k))

/-- Every positive prescribed physical horizon admits the exact
original quadrant transfer with four pieces. Source auxiliary:
arXiv:2411.04551v3, §1, `thm: targets.atoms`, and §3.2, clock rescaling. -/
theorem exists_quadrant_transfer_controls [Fintype ι] [Nonempty ι]
    (i : Fin d) (p q : SSphere d) (hp : p ∈ positiveQuadrant d)
    (hq : q ∈ positiveQuadrant d) (hpq : p ≠ q) (η : ι → Perspective.ProbSphere d)
    (hQ : ∀ k, (η k : Measure (SSphere d)).support ⊆ positiveQuadrant d)
    (hplane : ∀ k, ∀ α β : ℝ, barycenter d (η k) ≠ α • (p : EucSpace d) + β • (q : EucSpace d))
    (H : ℝ) (hH : 0 < H) :
    ∃ θ : TimeParams d, PiecewiseConstant d θ H 4 ∧ (∀ s, (θ s).B = 0) ∧
      (∀ ρ : Perspective.ProbSphere d, ∃ μ : ℝ → Perspective.ProbSphere d,
        μ 0 = ρ ∧ cauchyPB d θ μ) ∧
      (∀ μ : ℝ → Perspective.ProbSphere d, μ 0 = Perspective.diracProb d p → cauchyPB d θ μ →
        μ H = Perspective.diracProb d q) ∧
      ∀ k, ∀ μ : ℝ → Perspective.ProbSphere d, μ 0 = η k → cauchyPB d θ μ → μ H = η k := by
  obtain ⟨L, hlen, hL, hselected, hrest⟩ := exists_quadrant_transfer_phases i p q hp hq hpq η hQ hplane
  obtain ⟨θ, hpc, hB, hex, hend⟩ := exists_phase_sequence_realization i L hL H hH
  refine ⟨θ, by simpa only [hlen] using hpc, hB, hex, ?_, ?_⟩
  · intro μ h0 hμ
    rw [hend μ hμ, h0, hselected]
  · intro k μ h0 hμ
    rw [hend μ hμ, h0, hrest k]

/-- Three genuine rational unit points lie strictly in the original
positive quadrant. Source auxiliary: arXiv:2411.04551v3, §3.2,
the satisfiable plane-separation hypotheses for a protected transfer. -/
noncomputable def quadrantThreePoint (j : Fin 3) : SSphere 3 :=
  ⟨(EuclideanSpace.equiv (Fin 3) ℝ).symm (fun i => if i = j then 1 / 3 else 2 / 3), by
    rw [mem_sphere_zero_iff_norm]
    apply (sq_eq_sq₀ (norm_nonneg _) zero_le_one).mp
    rw [EuclideanSpace.real_norm_sq_eq]
    fin_cases j <;> norm_num [Fin.sum_univ_succ]⟩

/-- The actual rational points satisfy every positive coordinate
condition. Source auxiliary: arXiv:2411.04551v3, §3.2. -/
theorem quadrantThreePoint_mem (j : Fin 3) : quadrantThreePoint j ∈ positiveQuadrant 3 := by
  intro i
  change 0 < if i = j then (1 / 3 : ℝ) else 2 / 3
  split_ifs <;> norm_num

/-- The three actual positive unit points are distinct. Source
auxiliary: arXiv:2411.04551v3, §1, the injective atomic data used
to witness the point-transfer hypotheses. -/
theorem quadrantThreePoint_injective : Function.Injective quadrantThreePoint := by
  intro i j he
  by_contra hij
  have hh := congrArg (fun z : SSphere 3 => (z : EucSpace 3) i) he
  norm_num [quadrantThreePoint, hij] at hh

/-- The actual three-point unit family contains no antipodal pair.
Source auxiliary: arXiv:2411.04551v3, §3.2, positive-quadrant transfers. -/
theorem quadrantThreePoint_ne_antipode (i j : Fin 3) :
    quadrantThreePoint j ≠ antipode 3 (quadrantThreePoint i) :=
  quadrant_ne_antipode 0 _ _ (quadrantThreePoint_mem i) (quadrantThreePoint_mem j)

/-- Each rational quadrant point lies outside the actual plane of
the other two. Source auxiliary: arXiv:2411.04551v3, §3.2,
the satisfiable plane condition in the protected transfer. -/
theorem quadrantThreePoint_not_plane (i j k : Fin 3)
    (hij : i ≠ j) (hki : k ≠ i) (hkj : k ≠ j) (α β : ℝ) :
    (quadrantThreePoint k : EucSpace 3) ≠
      α • (quadrantThreePoint i : EucSpace 3) + β • (quadrantThreePoint j : EucSpace 3) := by
  intro he
  have hi := congrArg (fun x : EucSpace 3 => x i) he
  have hj := congrArg (fun x : EucSpace 3 => x j) he
  have hk := congrArg (fun x : EucSpace 3 => x k) he
  norm_num [quadrantThreePoint, hij, Ne.symm hij, hki, Ne.symm hki, hkj, Ne.symm hkj] at hi hj hk
  linarith

/-- Distinct positive selected endpoints, a genuinely positive third
unit Dirac outside their plane, and a positive horizon jointly realize
all protected-quadrant transfer hypotheses. -/
example : ∃ (p q : SSphere 3) (η : Fin 1 → Perspective.ProbSphere 3),
    (0 : ℝ) < 1 ∧ (0 : Fin 3) ≠ 1 ∧ (2 : Fin 3) ≠ 0 ∧ (2 : Fin 3) ≠ 1 ∧
    p ∈ positiveQuadrant 3 ∧ q ∈ positiveQuadrant 3 ∧ p ≠ q ∧
    (∀ k, (η k : Measure (SSphere 3)).support ⊆ positiveQuadrant 3) ∧
    ∀ k, ∀ α β : ℝ, barycenter 3 (η k) ≠ α • (p : EucSpace 3) + β • (q : EucSpace 3) := by
  let p := quadrantThreePoint 0
  let q := quadrantThreePoint 1
  let v := quadrantThreePoint 2
  let η : Fin 1 → Perspective.ProbSphere 3 := fun _ => Perspective.diracProb 3 v
  refine ⟨p, q, η, one_pos, by decide, by decide, by decide,
    quadrantThreePoint_mem 0, quadrantThreePoint_mem 1, ?_, ?_, ?_⟩
  · intro he
    have hh := congrArg (fun z : SSphere 3 => (z : EucSpace 3) (0 : Fin 3)) he
    norm_num [p, q, quadrantThreePoint] at hh
  · intro k z hz
    rw [eq_of_mem_support_dirac hz]
    exact quadrantThreePoint_mem 2
  · intro k α β he
    have h0 := congrArg (fun x : EucSpace 3 => x (0 : Fin 3)) he
    have h1 := congrArg (fun x : EucSpace 3 => x (1 : Fin 3)) he
    have h2 := congrArg (fun x : EucSpace 3 => x (2 : Fin 3)) he
    norm_num [η, barycenter, Perspective.diracProb, p, q, v, quadrantThreePoint] at h0 h1 h2
    linarith

end Transformer.Interpolation
