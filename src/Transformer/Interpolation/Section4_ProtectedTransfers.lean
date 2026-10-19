module

public import Transformer.Interpolation.Section4_TransferPhases
public import Transformer.Interpolation.Section3_MeanAxisAttraction

/-!
# Three original phases for a protected atomic transfer

ArXiv:2411.04551v3, §1, `thm: targets.atoms`, using the
rank-one attention isolation mechanism of §3.2, Steps 1–3.
Gather and restore here use the original attention component. The
middle phase is the actual original one-neuron transfer. These
auxiliary controls therefore act in the full original network.

A unit axis transverse to both selected endpoints fixes those atoms
under attention. All unselected laws with nonzero axial mean and
supports missing both poles can be gathered near those poles.
The middle gate is inactive throughout those pole neighborhoods.
Inverse attention then restores each such entire probability law
exactly, while the selected atom reaches its prescribed target.
-/

@[expose] public section

open Real MeasureTheory Set

namespace Transformer.Interpolation

variable {d : ℕ}

/-- A probability supported in sufficiently small neighborhoods of
orthogonal poles is fixed by the complete original gated neuron.
Source auxiliary: arXiv:2411.04551v3, §3.2, protected middle phases. -/
theorem ridgeLaw_fixed_of_two_pole_support (a m : EucSpace d) (w : SSphere d)
    (c r T : ℝ) (hwm : inner ℝ (w : EucSpace d) m = 0) (hr : r * ‖m‖ ≤ c)
    (ν : Perspective.ProbSphere d)
    (hν : (ν : Measure (SSphere d)).support ⊆
      Metric.ball w r ∪ Metric.ball (antipode d w) r) : ridgeLaw a (-m) (-c) ν T = ν := by
  apply ridgeLaw_inactive_support
  have hb (z pole : SSphere d) (hp : inner ℝ (pole : EucSpace d) m = 0)
      (hz : dist z pole < r) : -c - inner ℝ (z : EucSpace d) (-m) ≤ 0 := by
    have hn : ‖(z : EucSpace d) - (pole : EucSpace d)‖ ≤ r := by
      simpa only [Subtype.dist_eq, dist_eq_norm] using hz.le
    have hi := real_inner_le_norm ((z : EucSpace d) - (pole : EucSpace d)) m
    rw [inner_sub_left, hp, sub_zero] at hi
    have hm := mul_le_mul_of_nonneg_right hn (norm_nonneg m)
    rw [inner_neg_right]
    linarith
  intro z hz
  rcases hν hz with hz | hz
  · exact hb z w hwm hz
  · apply hb z (antipode d w) ?_ hz
    change inner ℝ (-(w : EucSpace d)) m = 0
    rw [inner_neg_left, hwm, neg_zero]

/-- Actual complete controls move the selected unit atom exactly and
restore every law in a finite unselected family. Three phases
suffice once a common transverse axis is supplied. Source auxiliary:
arXiv:2411.04551v3, §1, `thm: targets.atoms`, and §3.2, exact isolation. -/
theorem exists_protected_transfer_phases {ι : Type*} [Fintype ι]
    (i : Fin d) (p q w : SSphere d) (hpq : p ≠ q) (hanti : q ≠ antipode d p)
    (hwp : inner ℝ (w : EucSpace d) (p : EucSpace d) = 0)
    (hwq : inner ℝ (w : EucSpace d) (q : EucSpace d) = 0)
    (η : ι → Perspective.ProbSphere d)
    (hη : ∀ k, meanAxisMoment (w : EucSpace d) (η k) ≠ 0)
    (hmiss : ∀ k, w ∉ (η k : Measure (SSphere d)).support ∧
      antipode d w ∉ (η k : Measure (SSphere d)).support) :
    ∃ L : List (AutonomousPhase d), L.length = 3 ∧
      (∀ P ∈ L, IsAutonomousPhase P ∧ P.1.B = 0) ∧
      phaseSequenceLaw L (Perspective.diracProb d p) = Perspective.diracProb d q ∧
      ∀ k, phaseSequenceLaw L (η k) = η k := by
  obtain ⟨a, c, T, hc, _, hend, _⟩ := exists_exact_gated_transfer p q hpq hanti
  let m : EucSpace d := (p : EucSpace d) + (q : EucSpace d)
  have hwm : inner ℝ (w : EucSpace d) m = 0 := by simp only [m, inner_add_right, hwp, hwq, add_zero]
  let r := c / (2 * (‖m‖ + 1))
  have hr : 0 < r := by dsimp only [r]; positivity
  have hrad : r * ‖m‖ ≤ c := by
    have he : r * (2 * (‖m‖ + 1)) = c := div_mul_cancel₀ c (by positivity)
    nlinarith [mul_nonneg hr.le (norm_nonneg m)]
  have he : ∀ᶠ t : ℝ in Filter.atTop, ∀ k,
      (meanAxisLaw w 1 (η k) t : Measure (SSphere d)).support ⊆
        Metric.ball w r ∪ Metric.ball (antipode d w) r :=
    Filter.eventually_all.mpr (fun k => eventually_meanAxisLaw_two_poles w 1 (η k)
      one_pos (hη k) (hmiss k).1 (hmiss k).2 r hr)
  obtain ⟨t₁, hpre, _⟩ := (he.and (Filter.eventually_gt_atTop (0 : ℝ))).exists
  refine ⟨transferPhaseList i (a : EucSpace d) m w c t₁ T,
    transferPhaseList_length i _ _ _ _ _ _, transferPhaseList_valid i _ _ _ _ _ _,
    transferPhaseList_dirac i _ _ w p q c t₁ T hwp hwq hend, ?_⟩
  intro k
  apply transferPhaseList_restore_fixed
  exact ridgeLaw_fixed_of_two_pole_support (a : EucSpace d) m w c r T hwm hrad _ (hpre k)

/-- The protected transfer has actual original piecewise constant
controls at any positive prescribed horizon. All original weak laws
have the stated exact endpoints. Source auxiliary:
arXiv:2411.04551v3, §1, `thm: targets.atoms`, and §3.2, time rescaling. -/
theorem exists_protected_transfer_controls {ι : Type*} [Fintype ι]
    (i : Fin d) (p q w : SSphere d) (hpq : p ≠ q) (hanti : q ≠ antipode d p)
    (hwp : inner ℝ (w : EucSpace d) (p : EucSpace d) = 0)
    (hwq : inner ℝ (w : EucSpace d) (q : EucSpace d) = 0)
    (η : ι → Perspective.ProbSphere d)
    (hη : ∀ k, meanAxisMoment (w : EucSpace d) (η k) ≠ 0)
    (hmiss : ∀ k, w ∉ (η k : Measure (SSphere d)).support ∧
      antipode d w ∉ (η k : Measure (SSphere d)).support) (H : ℝ) (hH : 0 < H) :
    ∃ θ : TimeParams d, PiecewiseConstant d θ H 4 ∧ (∀ s, (θ s).B = 0) ∧
      (∀ ρ : Perspective.ProbSphere d, ∃ μ : ℝ → Perspective.ProbSphere d,
        μ 0 = ρ ∧ cauchyPB d θ μ) ∧
      (∀ μ : ℝ → Perspective.ProbSphere d, μ 0 = Perspective.diracProb d p → cauchyPB d θ μ →
        μ H = Perspective.diracProb d q) ∧
      ∀ k, ∀ μ : ℝ → Perspective.ProbSphere d, μ 0 = η k → cauchyPB d θ μ → μ H = η k := by
  obtain ⟨L, hlen, hL, hselected, hrest⟩ :=
    exists_protected_transfer_phases i p q w hpq hanti hwp hwq η hη hmiss
  obtain ⟨θ, hpc, hB, hex, hend⟩ := exists_phase_sequence_realization i L hL H hH
  refine ⟨θ, by simpa only [hlen] using hpc, hB, hex, ?_, ?_⟩
  · intro μ h0 hμ
    rw [hend μ hμ, h0, hselected]
  · intro k μ h0 hμ
    rw [hend μ hμ, h0, hrest k]

/-- A genuine selected orthogonal unit pair, transverse unit axis and
one other nonpolar unit Dirac meet every protected-transfer premise.
The other atom has a strictly nonzero axial mean. -/
example : ∃ (p q w : SSphere 3) (η : Fin 1 → Perspective.ProbSphere 3),
    (0 : ℝ) < 1 ∧ p ≠ q ∧ q ≠ antipode 3 p ∧
    inner ℝ (w : EucSpace 3) (p : EucSpace 3) = 0 ∧
    inner ℝ (w : EucSpace 3) (q : EucSpace 3) = 0 ∧
    (∀ k, meanAxisMoment (w : EucSpace 3) (η k) ≠ 0) ∧
    (∀ k, w ∉ (η k : Measure (SSphere 3)).support ∧
      antipode 3 w ∉ (η k : Measure (SSphere 3)).support) := by
  let p : SSphere 3 := ⟨PiLp.single 2 (0 : Fin 3) (1 : ℝ), by simp⟩
  let q : SSphere 3 := ⟨PiLp.single 2 (1 : Fin 3) (1 : ℝ), by simp⟩
  let w : SSphere 3 := ⟨PiLp.single 2 (2 : Fin 3) (1 : ℝ), by simp⟩
  let v : SSphere 3 := ⟨(EuclideanSpace.equiv (Fin 3) ℝ).symm ![3 / 5, 0, 4 / 5], by
    rw [mem_sphere_zero_iff_norm]
    apply (sq_eq_sq₀ (norm_nonneg _) zero_le_one).mp
    rw [EuclideanSpace.real_norm_sq_eq]
    norm_num [Fin.sum_univ_succ]⟩
  let η : Fin 1 → Perspective.ProbSphere 3 := fun _ => Perspective.diracProb 3 v
  refine ⟨p, q, w, η, one_pos, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro he
    have hh := congrArg (fun z : SSphere 3 => (z : EucSpace 3) (0 : Fin 3)) he
    norm_num [p, q] at hh
  · intro he
    have hh := congrArg (fun z : SSphere 3 => (z : EucSpace 3) (1 : Fin 3)) he
    norm_num [p, q, antipode] at hh
  · simp [w, p, EuclideanSpace.inner_single_left]
  · simp [w, q, EuclideanSpace.inner_single_left]
  · intro k
    norm_num [η, meanAxisMoment, barycenter, Perspective.diracProb, w, v, EuclideanSpace.inner_single_left]
  · intro k
    constructor
    · intro hw
      have hh := congrArg (fun z : SSphere 3 => (z : EucSpace 3) (0 : Fin 3)) (eq_of_mem_support_dirac hw)
      norm_num [η, w, v] at hh
    · intro hw
      have hh := congrArg (fun z : SSphere 3 => (z : EucSpace 3) (0 : Fin 3)) (eq_of_mem_support_dirac hw)
      norm_num [η, w, v, antipode] at hh

/-- A nonzero orthogonal gate and an actual pole Dirac jointly satisfy
the quantitative support and radius premises of middle-law fixing. -/
example : ∃ (w : SSphere 3) (m : EucSpace 3) (ν : Perspective.ProbSphere 3),
    inner ℝ (w : EucSpace 3) m = 0 ∧ (1 : ℝ) * ‖m‖ ≤ 1 ∧
    (ν : Measure (SSphere 3)).support ⊆ Metric.ball w 1 ∪ Metric.ball (antipode 3 w) 1 := by
  let w : SSphere 3 := ⟨PiLp.single 2 (2 : Fin 3) (1 : ℝ), by simp⟩
  refine ⟨w, PiLp.single 2 (0 : Fin 3) 1, Perspective.diracProb 3 w,
    by simp [w, EuclideanSpace.inner_single_left], by simp, ?_⟩
  intro z hz
  exact Or.inl (by rw [eq_of_mem_support_dirac hz]; simp)

end Transformer.Interpolation
