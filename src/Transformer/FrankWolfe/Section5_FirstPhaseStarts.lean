import Transformer.FrankWolfe.Section5_FirstPhaseGeometry
import Mathlib.Analysis.SpecialFunctions.Pow.Asymptotics

/-!
# Actual four-particle data for the printed first-phase time

ArXiv:2508.09628v1, §5, `lem: first.phase`, the displays defining
`m`, `τ` and `T₁`. In the actual unit segment take the two vertices,
a particle at `1-m`, and a particle at `1/2`, where `0 < m < 1/4`.
Both nonvertex particles are assigned to the positive vertex.
Their actual minimum distance is m. The source's complete constraints
therefore give `τ = m/2` at step size `γ = 1/2`, and its printed
logarithmic formula gives `T₁ = 1`.

All original initial-value and cell conditions are verified below.
The actual closed ball about either nonvertex point is contained in
the required dominance cone whenever its radius is at most `m/2`
and the cone margin is at most `1/2`. Thus the source's actual
temperature-dependent erosion holds at every sufficiently large β.
The original minimum, greatest constraint value and integer time
are calculated from the data; none is introduced as an assumption.
-/

open Real MeasureTheory Set
open scoped BigOperators
namespace Transformer.FrankWolfe

/-- The actual two vertices and the two distinct interior particles.
Source: arXiv:2508.09628v1, §5, `lem: first.phase`, original initial data. -/
noncomputable def firstPhaseStart (m : ℝ) : Idx 4 → EucSpace 1 :=
  ![firstPhasePoint (-1), firstPhasePoint 1, firstPhasePoint (1 - m), firstPhasePoint (1 / 2)]

/-- Each original particle is assigned to its actual segment cell.
Source: arXiv:2508.09628v1, §5, `lem: first.phase`, original initial data. -/
def firstPhaseAssignment : Idx 4 → Idx 2 := ![0, 1, 1, 1]

/-- Every original particle of the construction lies in the actual polytope.
Source: arXiv:2508.09628v1, §5, `lem: first.phase`, original initial data. -/
theorem firstPhaseStart_mem_segment {m : ℝ} (hm : m ∈ Ioo (0 : ℝ) (1 / 4)) :
    ∀ i : Idx 4, firstPhaseStart m i ∈ firstPhaseSegment := by
  intro i
  fin_cases i <;> norm_num [firstPhaseStart, abs_of_nonneg (show 0 ≤ 1 - m by linarith [hm.2])]
  linarith [hm.1]

/-- The first two original particles are exactly the two actual vertices.
Source: arXiv:2508.09628v1, §5, `lem: first.phase`, original initial data. -/
theorem firstPhaseStart_vertices (m : ℝ) (i : Idx 4) (hi : (i : ℕ) < 2) :
    firstPhaseStart m i = firstPhaseVertices ⟨i, hi⟩ := by
  fin_cases i
  · rfl
  · rfl
  · change (2 : ℕ) < 2 at hi
    omega
  · change (3 : ℕ) < 2 at hi
    omega

/-- Every original particle lies in its assigned actual cell.
Source: arXiv:2508.09628v1, §5, `lem: first.phase`, original initial data. -/
theorem firstPhaseStart_mem_cell {m : ℝ} (hm : m ∈ Ioo (0 : ℝ) (1 / 4)) :
    ∀ i : Idx 4, firstPhaseStart m i ∈
      cell (ContinuousLinearMap.id ℝ (EucSpace 1)) firstPhaseSegment firstPhaseVertices
        (firstPhaseAssignment i) := by
  intro i
  apply (mem_cell_iff _ firstPhaseSegment_hull _ _).mpr
  refine ⟨firstPhaseStart_mem_segment hm i, fun j => ?_⟩
  fin_cases i <;> fin_cases j <;> simp [firstPhaseStart, firstPhaseAssignment, firstPhaseVertices]
  all_goals linarith [hm.2]

/-- The source's actual minimum nonvertex distance is exactly m.
Source: arXiv:2508.09628v1, §5, `lem: first.phase`, original initial data. -/
theorem firstPhaseStart_minimum {m : ℝ} (hm : m ∈ Ioo (0 : ℝ) (1 / 4)) :
    IsLeast {r : ℝ | ∃ j : Idx 4, 2 ≤ (j : ℕ) ∧
      r = ‖firstPhaseStart m j - firstPhaseVertices (firstPhaseAssignment j)‖} m := by
  constructor
  · refine ⟨2, by decide, ?_⟩
    simp [firstPhaseStart, firstPhaseAssignment, firstPhaseVertices, firstPhasePoint_sub,
      abs_of_nonpos (show -m ≤ 0 by linarith [hm.1])]
  · rintro r ⟨j, hj, rfl⟩
    fin_cases j
    · norm_num at hj
    · norm_num at hj
    · simp [firstPhaseStart, firstPhaseAssignment, firstPhaseVertices, firstPhasePoint_sub,
        abs_of_nonpos (show -m ≤ 0 by linarith [hm.1])]
    · norm_num [firstPhaseStart, firstPhaseAssignment, firstPhaseVertices, firstPhasePoint_sub]
      linarith [hm.2]

/-- The actual closed ball about an interior point stays in the required cone.
Source: arXiv:2508.09628v1, §5, `lem: first.phase`, original initial data. -/
theorem firstPhasePoint_mem_eroded_cone {m r δ a : ℝ}
    (hm : m ∈ Ioo (0 : ℝ) (1 / 4)) (hr : r ≤ m / 2) (hδ : δ ≤ 1 / 2)
    (ha : a ∈ Icc (1 / 2 : ℝ) (1 - m)) :
    firstPhasePoint a ∈ erosion (dominanceCone firstPhaseSegment firstPhaseVertices 1 δ) r := by
  intro x hx
  have hd : |x 0 - a| ≤ r := by
    have hdist := Metric.mem_closedBall.mp hx
    rw [dist_eq_norm, ← firstPhasePoint_coord x, firstPhasePoint_sub, norm_firstPhasePoint] at hdist
    exact hdist
  have hcoord := abs_le.mp hd
  refine ⟨?_, fun j hj => ?_⟩
  · rw [← firstPhasePoint_coord x, firstPhasePoint_mem_segment, abs_le]
    constructor <;> linarith [hm.1, hm.2, ha.1, ha.2, hcoord.1, hcoord.2]
  · fin_cases j
    · change δ ≤ inner (𝕜 := ℝ) (firstPhasePoint 1 - firstPhasePoint (-1)) x
      rw [firstPhasePoint_sub, ← firstPhasePoint_coord x, inner_firstPhasePoint]
      linarith [hm.2, ha.1, hcoord.1]
    · exact False.elim (hj rfl)

/-- Both actual nonvertex particles satisfy the original erosion condition.
Source: arXiv:2508.09628v1, §5, `lem: first.phase`, original initial data. -/
theorem firstPhaseStart_mem_eroded_cone {m r δ : ℝ}
    (hm : m ∈ Ioo (0 : ℝ) (1 / 4)) (hr : r ≤ m / 2) (hδ : δ ≤ 1 / 2) :
    ∀ i : Idx 4, 2 ≤ (i : ℕ) → firstPhaseStart m i ∈
      erosion (dominanceCone firstPhaseSegment firstPhaseVertices (firstPhaseAssignment i) δ) r := by
  intro i hi
  fin_cases i
  · norm_num at hi
  · norm_num at hi
  · exact firstPhasePoint_mem_eroded_cone hm hr hδ ⟨by linarith [hm.2], le_rfl⟩
  · exact firstPhasePoint_mem_eroded_cone hm hr hδ ⟨le_rfl, by linarith [hm.2]⟩

/-- The source's full original radius constraints give exactly `τ = m/2`.
Source: arXiv:2508.09628v1, §5, `lem: first.phase`, original initial data. -/
theorem firstPhase_tau_greatest {m : ℝ} (hm : m ∈ Ioo (0 : ℝ) (1 / 4)) :
    IsGreatest {r : ℝ | (∀ i j : Idx 2, j ≠ i →
      r * (2 * ‖firstPhaseVertices i - firstPhaseVertices j‖) ≤ 1) ∧
      r ≤ Real.sqrt (2 * 1) / 2 ∧ r ≤ (1 - (1 / 2 : ℝ)) * m} (m / 2) := by
  constructor
  · refine ⟨?_, ?_, by ring_nf; exact le_rfl⟩
    · intro i j hij
      fin_cases i <;> fin_cases j <;> simp_all [firstPhaseVertices, firstPhasePoint_sub]
      all_goals linarith [hm.2]
    · have hs := Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 2)
      have hn := Real.sqrt_nonneg 2
      norm_num only [mul_one]
      nlinarith [hm.2]
  · intro r hr
    linarith [hr.2.2]

/-- The source's actual printed logarithmic time equals one step.
Source: arXiv:2508.09628v1, §5, `lem: first.phase`, original initial data. -/
theorem firstPhase_time_eq_one {m : ℝ} (hm : 0 < m) :
    ⌊Real.log ((m / 2) / m) / Real.log (1 - (1 / 2 : ℝ))⌋₊ = 1 := by
  have hratio : (m / 2) / m = 1 / 2 := by field_simp [hm.ne']
  have hlog : Real.log (1 / 2 : ℝ) ≠ 0 := (Real.log_neg (by norm_num) (by norm_num)).ne
  rw [hratio]
  norm_num [hlog]

/-- A genuine positive m, actual interior point, radius and cone margin
jointly realize every hypothesis, including an actual vertex index. -/
example : ((0 : Idx 4) : ℕ) < 2 ∧ (1 / 8 : ℝ) ∈ Ioo (0 : ℝ) (1 / 4) ∧ (1 / 32 : ℝ) ≤ (1 / 8) / 2 ∧
    (1 / 4 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ∈ Icc (1 / 2 : ℝ) (1 - 1 / 8) := by norm_num

end Transformer.FrankWolfe
