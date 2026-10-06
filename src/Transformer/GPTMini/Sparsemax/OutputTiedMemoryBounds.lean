import Transformer.GPTMini.Sparsemax.OutputTiedMemoryExamples

/-!
# Ordinary output error controls all tied intrinsic parameters

New quantitative consequence after arXiv:1602.02068v2, §2.5. The adjacent
affine readout has squared Lipschitz bound `4*gain^2` in the finite sum
squared coordinates, independently of dictionary size. Each endpoint
coordinate appears in at most two adjacent edges; no factor P is needed.

Consequently output-table distance controls complete edge/output parameter
distance with factor `1+4*gain^2`. This is a diagnostic error bound, not an
additional optimization criterion. Physical Q/K and original common values
are determined by these intrinsic parameters through the proved chart.
The bound itself is on chart coordinates, rather than a claim of uniformly
well-conditioned original inverse values or arbitrary embedding gauges.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open scoped BigOperators

/-- Complete squared intrinsic-coordinate error, used only to state guarantees.
Source: the tied fixed-width parameter chart following sparsemax Eq. (1). -/
def periodicParameterError {N D : ℕ}
    (reference x : (Fin N → ℝ) × Matrix (Fin (N + 1)) (Fin D) ℝ) : ℝ :=
  (∑ e, (x.1 e - reference.1 e) ^ 2) + matrixOutputError reference.2 x.2

/-- Complete intrinsic squared error is nonnegative without domain assumptions.
Source: the diagnostic coordinate error following arXiv:1602.02068v2, §2.5. -/
theorem periodicParameterError_nonneg {N D : ℕ}
    (reference x : (Fin N → ℝ) × Matrix (Fin (N + 1)) (Fin D) ℝ) :
    0 ≤ periodicParameterError reference x :=
  add_nonneg (Finset.sum_nonneg (fun _ _ => sq_nonneg _)) (matrixOutputError_nonneg _ _)

/-- Zero complete coordinate error means equality of edges and every output coordinate.
Source: the diagnostic full intrinsic error following sparsemax §2.5. -/
theorem periodicParameterError_eq_zero {N D : ℕ}
    (reference x : (Fin N → ℝ) × Matrix (Fin (N + 1)) (Fin D) ℝ) :
    periodicParameterError reference x = 0 ↔ x = reference := by
  constructor
  · intro h
    have he : 0 ≤ ∑ e, (x.1 e - reference.1 e) ^ 2 :=
      Finset.sum_nonneg (fun _ _ => sq_nonneg _)
    have ho := matrixOutputError_nonneg reference.2 x.2
    unfold periodicParameterError at h
    have hz : (∑ e, (x.1 e - reference.1 e) ^ 2) = 0 := by linarith
    apply Prod.ext
    · have ha := (Finset.sum_eq_zero_iff_of_nonneg
        (fun e _ => sq_nonneg (x.1 e - reference.1 e))).mp hz
      funext e
      exact sub_eq_zero.mp (sq_eq_zero_iff.mp (ha e (Finset.mem_univ e)))
    · exact (matrixOutputError_eq_zero reference.2 x.2).mp (by linarith)
  · intro h
    rw [h]
    unfold periodicParameterError
    rw [(matrixOutputError_eq_zero _ _).mpr rfl]
    simp only [sub_self, zero_pow (by decide : (2 : ℕ) ≠ 0), Finset.sum_const_zero, add_zero]

/-- Every changed complete intrinsic point has positive squared coordinate error.
Source: the diagnostic intrinsic metric after arXiv:1602.02068v2, §2.5. -/
theorem periodicParameterError_pos {N D : ℕ}
    (reference x : (Fin N → ℝ) × Matrix (Fin (N + 1)) (Fin D) ℝ)
    (h : x ≠ reference) : 0 < periodicParameterError reference x := by
  have hn := periodicParameterError_nonneg reference x
  by_contra hp
  have hz : periodicParameterError reference x = 0 := by linarith
  exact h ((periodicParameterError_eq_zero reference x).mp hz)

/-- Opposite genuine learned edge/answer points inhabit the positive-distance premise. -/
example : 0 < periodicParameterError ((taskEnergyParameters 0).1, taskEnergyTarget 0)
    ((taskEnergyParameters 1).1, taskEnergyTarget 1) := by
  apply periodicParameterError_pos
  intro h
  have he := congrArg (fun p : (Fin 2 → ℝ) × Matrix (Fin 3) (Fin 1) ℝ => p.2 0 0) h
  norm_num [taskEnergyTarget, Matrix.of_apply] at he

/-- Adjacent affine edge sharing has a squared Lipschitz constant independent of prototype count.
Source: the new restricted ordinary-output chart after sparsemax §2.5. -/
theorem outputTiedMemoryEdges_sq_bound {N D : ℕ} (offset : Fin N → ℝ) (gain : ℝ)
    (channel : Fin D) (Z W : Matrix (Fin (N + 1)) (Fin D) ℝ) :
    (∑ e, (outputTiedMemoryEdges offset gain channel Z e -
      outputTiedMemoryEdges offset gain channel W e) ^ 2) ≤
        4 * gain ^ 2 * matrixOutputError W Z := by
  let f : Fin (N + 1) → ℝ := fun i => (Z i channel - W i channel) ^ 2
  have hl : (∑ e : Fin N, f e.castSucc) ≤ ∑ i, f i := by
    rw [Fin.sum_univ_castSucc f]
    linarith [sq_nonneg (Z (Fin.last N) channel - W (Fin.last N) channel)]
  have hr : (∑ e : Fin N, f e.succ) ≤ ∑ i, f i := by
    rw [Fin.sum_univ_succ f]
    linarith [sq_nonneg (Z 0 channel - W 0 channel)]
  have hc : (∑ i, f i) ≤ matrixOutputError W Z := by
    apply Finset.sum_le_sum
    intro i hi
    exact Finset.single_le_sum (fun d _ => sq_nonneg (Z i d - W i d)) (Finset.mem_univ channel)
  have he (e : Fin N) :
      (outputTiedMemoryEdges offset gain channel Z e -
        outputTiedMemoryEdges offset gain channel W e) ^ 2 ≤
          2 * gain ^ 2 * (f e.castSucc + f e.succ) := by
    have h : ((Z e.castSucc channel - W e.castSucc channel) +
        (Z e.succ channel - W e.succ channel)) ^ 2 ≤
        2 * (f e.castSucc + f e.succ) := by
      dsimp only [f]
      nlinarith [sq_nonneg ((Z e.castSucc channel - W e.castSucc channel) -
        (Z e.succ channel - W e.succ channel))]
    have hg := mul_le_mul_of_nonneg_left h (sq_nonneg gain)
    unfold outputTiedMemoryEdges
    dsimp only [f] at hg ⊢
    nlinarith only [hg]
  have hs : (∑ e : Fin N, f e.castSucc) + (∑ e : Fin N, f e.succ) ≤ 2 * ∑ i, f i := by
    linarith
  calc
    _ ≤ ∑ e : Fin N, 2 * gain ^ 2 * (f e.castSucc + f e.succ) :=
      Finset.sum_le_sum (fun e _ => he e)
    _ = 2 * gain ^ 2 * ((∑ e : Fin N, f e.castSucc) + ∑ e : Fin N, f e.succ) := by
      rw [← Finset.mul_sum, Finset.sum_add_distrib]
    _ ≤ 2 * gain ^ 2 * (2 * ∑ i, f i) :=
      mul_le_mul_of_nonneg_left hs (by positivity)
    _ ≤ 4 * gain ^ 2 * matrixOutputError W Z := by
      have h := mul_le_mul_of_nonneg_left hc (by positivity : 0 ≤ 4 * gain ^ 2)
      nlinarith only [h]

/-- On the tied domain, ordinary output distance controls every intrinsic learned coordinate.
Source: the new quantitative sharing guarantee following arXiv:1602.02068v2, §2.5. -/
theorem outputTiedEnergy_parameter_bound {N D : ℕ} (floor budget energy : ℝ)
    (offset : Fin N → ℝ) (gain : ℝ) (channel : Fin D)
    (x y : (Fin N → ℝ) × Matrix (Fin (N + 1)) (Fin D) ℝ)
    (hx : x ∈ outputTiedEnergyDomain N D floor budget energy offset gain channel)
    (hy : y ∈ outputTiedEnergyDomain N D floor budget energy offset gain channel) :
    periodicParameterError y x ≤ (1 + 4 * gain ^ 2) * matrixOutputError y.2 x.2 := by
  unfold periodicParameterError
  rw [hx.2, hy.2]
  have h := outputTiedMemoryEdges_sq_bound offset gain channel x.2 y.2
  nlinarith only [h]

/-- Distinct nonidentity answer fits inhabit the complete-coordinate bound at nonzero gain. -/
example : periodicParameterError ((taskEnergyParameters 0).1, taskEnergyTarget 0)
    ((taskEnergyParameters 1).1, taskEnergyTarget 1) ≤
      (1 + 4 * (1 / 24 : ℝ) ^ 2) * matrixOutputError (taskEnergyTarget 0) (taskEnergyTarget 1) :=
  outputTiedEnergy_parameter_bound (3 / 4) (1 / 8) 6 _ _ _ _ _
    (taskOutputTiedPair_mem 1) (taskOutputTiedPair_mem 0)

/-- Ordinary nonconstant answer changes obey the same size-independent edge bound. -/
example : (∑ e, (outputTiedMemoryEdges (fun _ : Fin 2 => (1 / 24 : ℝ)) (1 / 24)
    (0 : Fin 1) (taskEnergyTarget 0) e -
    outputTiedMemoryEdges (fun _ : Fin 2 => (1 / 24 : ℝ)) (1 / 24) 0 (taskEnergyTarget 1) e) ^ 2) ≤
    4 * (1 / 24 : ℝ) ^ 2 * matrixOutputError (taskEnergyTarget 1) (taskEnergyTarget 0) :=
  outputTiedMemoryEdges_sq_bound _ _ _ _ _

end Transformer.GPTMini.Sparsemax
