module

public import Transformer.FrankWolfe.Section5_FirstPhaseStarts
public import Transformer.FrankWolfe.Section5_ProcessLaw
public import Transformer.FrankWolfe.Section5_TransitionAS

/-!
# Counterexample to the printed first-phase clustering time

ArXiv:2508.09628v1, §5, `lem: first.phase`, claims a universal
radius constant at its displayed minimum-distance time. This claim
is false for the original process, even on an actual acute segment.

Take vertices minus one and one, step size `γ = 1/2`, and interior
particles `1-m` and `1/2`. Every original geometric, cell and erosion
hypothesis holds at sufficiently large β. The actual minimum distance
is m, the original constraints give `τ = m/2`, and the printed time
is exactly one. Every possible update of the farther particle stays
at distance at least `1/4` from its assigned positive vertex.

For any proposed universal `C > 1`, put `m = 1/(8C)`. Then the
claimed target radius is `Cτ = 1/16`, so the actual success event has
probability zero, while its claimed lower bound is strictly positive.
The process is the genuine infinite law already constructed from
the original softmax rows. The final theorem negates the full original
universal statement with all its hypotheses and quantifiers retained.
No change of a universal radius constant repairs this printed time.
-/

@[expose] public section

open Real MeasureTheory Set Filter Topology
open scoped BigOperators ENNReal
namespace Transformer.FrankWolfe

/-- The original simultaneous success event at the actual printed time one.
Source: arXiv:2508.09628v1, §5, `lem: first.phase`, counterexample. -/
def firstPhaseSuccess (β C m : ℝ) : Set (ℕ → Idx 4 → EucSpace 1) :=
  {x | (∀ (i : Idx 4) (hi : (i : ℕ) < 2),
      x 1 i ∈ Metric.ball (firstPhaseVertices ⟨i, hi⟩) (β ^ (-(1 : ℝ) / 4))) ∧
    (∀ i : Idx 4, 2 ≤ (i : ℕ) →
      x 1 i ∈ Metric.ball (firstPhaseVertices (firstPhaseAssignment i)) (C * (m / 2)))}

/-- Every actual update of the farther original particle stays at least `1/4` away.
Source: arXiv:2508.09628v1, §5, `lem: first.phase`, counterexample. -/
theorem firstPhase_far_step_dist {m : ℝ} (hm : m ∈ Ioo (0 : ℝ) (1 / 4)) (j : Idx 4) :
    (1 / 4 : ℝ) ≤ dist
      ((1 - (1 / 2 : ℝ)) • firstPhaseStart m 3 + (1 / 2 : ℝ) • firstPhaseStart m j)
      (firstPhaseVertices 1) := by
  let z := (1 - (1 / 2 : ℝ)) • firstPhaseStart m 3 + (1 / 2 : ℝ) • firstPhaseStart m j
  have hcoord := (firstPhaseSegment_coord (firstPhaseStart_mem_segment hm j)).2
  have hz : z 0 ≤ 3 / 4 := by
    change (1 - (1 / 2 : ℝ)) * (1 / 2) + (1 / 2) * firstPhaseStart m j 0 ≤ 3 / 4
    linarith
  calc
    (1 / 4 : ℝ) ≤ |z 0 - 1| := by rw [abs_of_nonpos (by linarith)]; linarith
    _ = dist z (firstPhaseVertices 1) := by
      change _ = dist z (firstPhasePoint 1)
      rw [dist_eq_norm, ← firstPhasePoint_coord z, firstPhasePoint_sub, norm_firstPhasePoint]
      simp only [firstPhasePoint_apply]

/-- The actual original success event has probability zero at the proposed small radius.
Source: arXiv:2508.09628v1, §5, `lem: first.phase`, counterexample. -/
theorem firstPhaseSuccess_measure_zero {m C β : ℝ}
    (hm : m ∈ Ioo (0 : ℝ) (1 / 4)) (hCm : C * (m / 2) ≤ 1 / 4)
    (P : Measure (ℕ → Idx 4 → EucSpace 1))
    (hP : IsSAProcess β (1 / 2) (firstPhaseStart m) P) :
    P (firstPhaseSuccess β C m) = 0 := by
  apply measure_eq_zero_iff_ae_notMem.mpr
  filter_upwards [hP.2.1, hP.ae_exists_step] with x hx₀ hstep
  intro hsuccess
  have hball := hsuccess.2 (3 : Idx 4) (by decide)
  have hdist := Metric.mem_ball.mp hball
  change dist (x 1 3) (firstPhaseVertices 1) < C * (m / 2) at hdist
  obtain ⟨j, hj⟩ := hstep 0 3
  have heq : x 1 3 = (1 - (1 / 2 : ℝ)) • firstPhaseStart m 3 +
      (1 / 2 : ℝ) • firstPhaseStart m j := by simpa only [zero_add, hx₀] using hj
  rw [heq] at hdist
  linarith [firstPhase_far_step_dist hm j]

/-- Actual arbitrarily large temperatures satisfy the original cone and erosion bounds.
Source: arXiv:2508.09628v1, §5, `lem: first.phase`, counterexample. -/
theorem exists_firstPhase_large_beta (B m : ℝ) (hm : 0 < m) :
    ∃ β : ℝ, B ≤ β ∧ 1 < β ∧ β ^ (-(1 : ℝ) / 8) ≤ 1 / 2 ∧
      β ^ (-(1 : ℝ) / 4) ≤ m / 2 := by
  have hd : Tendsto (fun β : ℝ => β ^ (-(1 : ℝ) / 8)) atTop (nhds 0) := by
    simpa only [neg_div] using tendsto_rpow_neg_atTop (by norm_num : (0 : ℝ) < 1 / 8)
  have hr : Tendsto (fun β : ℝ => β ^ (-(1 : ℝ) / 4)) atTop (nhds 0) := by
    simpa only [neg_div] using tendsto_rpow_neg_atTop (by norm_num : (0 : ℝ) < 1 / 4)
  have hevent : ∀ᶠ β : ℝ in atTop, B ≤ β ∧ 1 < β ∧ β ^ (-(1 : ℝ) / 8) ≤ 1 / 2 ∧
      β ^ (-(1 : ℝ) / 4) ≤ m / 2 := by
    filter_upwards [eventually_ge_atTop B, eventually_gt_atTop (1 : ℝ),
      hd.eventually_lt_const (by norm_num : (0 : ℝ) < 1 / 2),
      hr.eventually_lt_const (by linarith : (0 : ℝ) < m / 2)] with β hB hβ hδ hrβ
    exact ⟨hB, hβ, hδ.le, hrβ.le⟩
  exact hevent.exists

/-- The membership, radius, positive universal constant and genuine-process
hypotheses are jointly realized by actual four-particle data. -/
example : (1 : ℝ) < 2 ∧ (1 / 8 : ℝ) ∈ Ioo (0 : ℝ) (1 / 4) ∧ (2 : ℝ) * ((1 / 8) / 2) ≤ 1 / 4 ∧
    IsSAProcess 1 (1 / 2) (firstPhaseStart (1 / 8))
      (saProcessLaw 1 (1 / 2) (firstPhaseStart (1 / 8))) :=
  ⟨by norm_num, by norm_num, by norm_num, isSAProcess_saProcessLaw _ _ _⟩

/-- Every proposed universal radius constant admits an actual process contradicting the bound.
Source: arXiv:2508.09628v1, §5, `lem: first.phase`, counterexample. -/
theorem exists_firstPhase_counterexample (C B : ℝ) (hC : 1 < C) :
    ∃ (m β : ℝ) (P : Measure (ℕ → Idx 4 → EucSpace 1)),
      m ∈ Ioo (0 : ℝ) (1 / 4) ∧ B ≤ β ∧ 1 < β ∧
      β ^ (-(1 : ℝ) / 8) ≤ 1 / 2 ∧ β ^ (-(1 : ℝ) / 4) ≤ m / 2 ∧
      IsSAProcess β (1 / 2) (firstPhaseStart m) P ∧
      P (firstPhaseSuccess β C m) = 0 ∧ 0 < 1 - β ^ (-(1 : ℝ) / 8) := by
  have hC0 : 0 < C := by linarith
  let m := 1 / (8 * C)
  have hm : m ∈ Ioo (0 : ℝ) (1 / 4) := by
    constructor
    · dsimp [m]
      positivity
    · dsimp [m]
      apply (div_lt_iff₀ (by positivity : 0 < 8 * C)).mpr
      linarith
  have hCm : C * (m / 2) ≤ 1 / 4 := by
    have heq : C * (m / 2) = 1 / 16 := by dsimp [m]; field_simp [hC0.ne']; ring
    rw [heq]
    norm_num
  obtain ⟨β, hB, hβ, hδ, hr⟩ := exists_firstPhase_large_beta B m hm.1
  let P := saProcessLaw β (1 / 2) (firstPhaseStart m)
  have hP : IsSAProcess β (1 / 2) (firstPhaseStart m) P := isSAProcess_saProcessLaw _ _ _
  refine ⟨m, β, P, hm, hB, hβ, hδ, hr, hP, firstPhaseSuccess_measure_zero hm hCm P hP, ?_⟩
  linarith

/-- Counterexample to the complete printed `lem: first.phase`: no universal
`C > 1` gives its stated probability at its minimum-distance time.
The actual segment, four original initial particles, genuine softmax process,
eroded cones, minimum m, greatest constraint value τ and integer T₁
all satisfy the original hypotheses. The original claimed success event
has probability zero instead of the strictly positive lower bound.
Source: arXiv:2508.09628v1, §5, `lem: first.phase`. -/
theorem not_first_phase :
    ¬ (∃ C : ℝ, 1 < C ∧
      ∀ (d κ n : ℕ) (K : Set (EucSpace d)) (v : Idx κ → EucSpace d) (c₀ γ : ℝ),
        ConePolytope K v c₀ → γ ∈ Set.Ioo (0 : ℝ) 1 → κ ≤ n →
        ∃ βstar > (0 : ℝ), ∀ β ≥ βstar,
          ∀ (X₀ : Idx n → EucSpace d) (σ : Idx n → Idx κ) (m τ : ℝ) (T₁ : ℕ)
            (P : Measure (ℕ → Idx n → EucSpace d)),
            (∀ i : Idx n, X₀ i ∈ K) →
            (∀ (i : Idx n) (h : (i : ℕ) < κ), X₀ i = v ⟨i, h⟩) →
            (∀ i : Idx n, X₀ i ∈
              cell (ContinuousLinearMap.id ℝ (EucSpace d)) K v (σ i)) →
            (∀ i : Idx n, κ ≤ (i : ℕ) →
              X₀ i ∈ erosion (dominanceCone K v (σ i) (β ^ (-(1 : ℝ) / 8)))
                (β ^ (-(1 : ℝ) / 4))) →
            IsLeast {r : ℝ | ∃ j : Idx n, κ ≤ (j : ℕ) ∧ r = ‖X₀ j - v (σ j)‖} m →
            IsGreatest {r : ℝ | (∀ i j : Idx κ, j ≠ i → r * (2 * ‖v i - v j‖) ≤ c₀) ∧
              r ≤ Real.sqrt (2 * c₀) / 2 ∧ r ≤ (1 - γ) * m} τ →
            T₁ = ⌊Real.log (τ / m) / Real.log (1 - γ)⌋₊ →
            IsSAProcess β γ X₀ P →
            1 - β ^ (-(1 : ℝ) / 8) ≤
              (P {x : ℕ → Idx n → EucSpace d |
                (∀ (i : Idx n) (h : (i : ℕ) < κ),
                    x T₁ i ∈ Metric.ball (v ⟨i, h⟩) (β ^ (-(1 : ℝ) / 4))) ∧
                (∀ i : Idx n, κ ≤ (i : ℕ) →
                    x T₁ i ∈ Metric.ball (v (σ i)) (C * τ))}).toReal) := by
  rintro ⟨C, hC, hmain⟩
  obtain ⟨B, -, hβ⟩ := hmain 1 2 4 firstPhaseSegment firstPhaseVertices 1 (1 / 2)
    conePolytope_firstPhaseSegment (by norm_num) (by norm_num)
  obtain ⟨m, β, P, hm, hB, -, hδ, hr, hP, hzero, hgap⟩ :=
    exists_firstPhase_counterexample C B hC
  have hb := hβ β hB (firstPhaseStart m) firstPhaseAssignment m (m / 2) 1 P
    (firstPhaseStart_mem_segment hm) (firstPhaseStart_vertices m) (firstPhaseStart_mem_cell hm)
    (firstPhaseStart_mem_eroded_cone hm hr hδ) (firstPhaseStart_minimum hm)
    (firstPhase_tau_greatest hm) (firstPhase_time_eq_one hm.1).symm hP
  change 1 - β ^ (-(1 : ℝ) / 8) ≤ (P (firstPhaseSuccess β C m)).toReal at hb
  rw [hzero, ENNReal.toReal_zero] at hb
  linarith

end Transformer.FrankWolfe
