import Transformer.GPTMini.Sparsemax.OutputTiedMemoryMinimum

/-!
# Ordinary suboptimality controls complete trained parameter error

New guarantee following arXiv:1602.02068v2, §2.5. Comparison with all
feasible interpolation times gives the sharp output-distance growth bound,
without the factor two from a midpoint-only argument. The output minimum
may have positive error, so exact fitting is not an assumption.

Affine parameter sharing transfers ordinary objective suboptimality to
full intrinsic squared distance with factor `1+4*gain^2`, independent of
dictionary size. The error is a diagnostic quantity, not an added criterion.
The earlier proved nonempty-domain optimizer supplies all minimum premises.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

/-- A constrained ordinary-error minimum gives sharp output-coordinate quadratic growth.
Source: all feasible affine comparisons in the actual chart after sparsemax §2.5. -/
theorem outputTiedSquaredError_output_growth {N D : ℕ} (floor budget energy : ℝ)
    (offset : Fin N → ℝ) (gain : ℝ) (channel : Fin D)
    (target : Matrix (Fin (N + 1)) (Fin D) ℝ)
    (x y : (Fin N → ℝ) × Matrix (Fin (N + 1)) (Fin D) ℝ) (hf : 1 / 2 < floor)
    (hx : x ∈ outputTiedEnergyDomain N D floor budget energy offset gain channel)
    (hy : y ∈ outputTiedEnergyDomain N D floor budget energy offset gain channel)
    (hm : IsMinOn (periodicEnergySquaredError (fun j : Fin (N + 1) => j) target)
      (outputTiedEnergyDomain N D floor budget energy offset gain channel) y) :
    matrixOutputError y.2 x.2 ≤
      periodicEnergySquaredError (fun j : Fin (N + 1) => j) target x -
      periodicEnergySquaredError (fun j : Fin (N + 1) => j) target y := by
  let distance := matrixOutputError y.2 x.2
  let gap := periodicEnergySquaredError (fun j : Fin (N + 1) => j) target x -
    periodicEnergySquaredError (fun j : Fin (N + 1) => j) target y
  have hmin := hm hx
  change periodicEnergySquaredError _ _ y ≤ periodicEnergySquaredError _ _ x at hmin
  have hg : 0 ≤ gap := sub_nonneg.mpr hmin
  change distance ≤ gap
  by_contra h
  have hd : gap < distance := lt_of_not_ge h
  have hdpos : 0 < distance := lt_of_le_of_lt hg hd
  let t := (distance - gap) / (2 * distance)
  have ht : 0 < t := div_pos (by linarith) (by linarith)
  have hu : t ≤ 1 := (div_le_iff₀ (by linarith : 0 < 2 * distance)).mpr (by linarith)
  have he : t * (2 * distance) = distance - gap :=
    div_mul_cancel₀ _ (by linarith)
  have hmpt := outputTiedEnergyDomain_convex N D floor budget energy offset gain channel
    hx hy ht.le (by linarith : 0 ≤ 1 - t) (by ring)
  have hc := hm hmpt
  change periodicEnergySquaredError (fun j : Fin (N + 1) => j) target y ≤
    periodicEnergySquaredError (fun j : Fin (N + 1) => j) target (t • x + (1 - t) • y) at hc
  rw [outputTiedSquaredError_affine_gap floor budget energy offset gain channel target
    x y hf hx hy t (1 - t) ht.le (by linarith) (by ring)] at hc
  have hp : 0 < t * t * distance := mul_pos (mul_pos ht ht) hdpos
  have hmultip := congrArg (fun r : ℝ => t * r) he
  dsimp only [gap, distance] at hmultip hp
  nlinarith only [hc, hmultip, hp]

/-- Different learned supports and nonconstant answers inhabit every sharp-growth premise. -/
example : matrixOutputError (taskEnergyTarget 0) (taskEnergyTarget 1) ≤
    periodicEnergySquaredError (fun j : Fin 3 => j) (taskEnergyTarget 0)
      ((taskEnergyParameters 1).1, taskEnergyTarget 1) -
    periodicEnergySquaredError (fun j : Fin 3 => j) (taskEnergyTarget 0)
      ((taskEnergyParameters 0).1, taskEnergyTarget 0) :=
  outputTiedSquaredError_output_growth (3 / 4) (1 / 8) 6
    (fun _ : Fin 2 => (1 / 24 : ℝ)) (1 / 24) (0 : Fin 1) (taskEnergyTarget 0)
    ((taskEnergyParameters 1).1, taskEnergyTarget 1)
    ((taskEnergyParameters 0).1, taskEnergyTarget 0) (by norm_num)
    (taskOutputTiedPair_mem 1) (taskOutputTiedPair_mem 0) (taskOutputTied_isMinOn 0)

/-- Ordinary objective suboptimality controls every intrinsic learned edge/output coordinate.
Source: sharp output growth and the adjacent affine tie after arXiv:1602.02068v2, §2.5. -/
theorem outputTiedSquaredError_parameter_growth {N D : ℕ} (floor budget energy : ℝ)
    (offset : Fin N → ℝ) (gain : ℝ) (channel : Fin D)
    (target : Matrix (Fin (N + 1)) (Fin D) ℝ)
    (x y : (Fin N → ℝ) × Matrix (Fin (N + 1)) (Fin D) ℝ) (hf : 1 / 2 < floor)
    (hx : x ∈ outputTiedEnergyDomain N D floor budget energy offset gain channel)
    (hy : y ∈ outputTiedEnergyDomain N D floor budget energy offset gain channel)
    (hm : IsMinOn (periodicEnergySquaredError (fun j : Fin (N + 1) => j) target)
      (outputTiedEnergyDomain N D floor budget energy offset gain channel) y) :
    periodicParameterError y x ≤ (1 + 4 * gain ^ 2) *
      (periodicEnergySquaredError (fun j : Fin (N + 1) => j) target x -
       periodicEnergySquaredError (fun j : Fin (N + 1) => j) target y) := by
  have hg := mul_le_mul_of_nonneg_left
    (outputTiedSquaredError_output_growth floor budget energy offset gain channel target x y hf hx hy hm)
    (by positivity : 0 ≤ 1 + 4 * gain ^ 2)
  exact (outputTiedEnergy_parameter_bound floor budget energy offset gain channel x y hx hy).trans hg

/-- Distinct nonidentity learned points inhabit every complete parameter-growth premise. -/
example : periodicParameterError ((taskEnergyParameters 0).1, taskEnergyTarget 0)
    ((taskEnergyParameters 1).1, taskEnergyTarget 1) ≤ (1 + 4 * (1 / 24 : ℝ) ^ 2) *
    (periodicEnergySquaredError (fun j : Fin 3 => j) (taskEnergyTarget 0)
      ((taskEnergyParameters 1).1, taskEnergyTarget 1) -
     periodicEnergySquaredError (fun j : Fin 3 => j) (taskEnergyTarget 0)
      ((taskEnergyParameters 0).1, taskEnergyTarget 0)) :=
  outputTiedSquaredError_parameter_growth (3 / 4) (1 / 8) 6
    (fun _ : Fin 2 => (1 / 24 : ℝ)) (1 / 24) (0 : Fin 1) (taskEnergyTarget 0)
    ((taskEnergyParameters 1).1, taskEnergyTarget 1)
    ((taskEnergyParameters 0).1, taskEnergyTarget 0) (by norm_num)
    (taskOutputTiedPair_mem 1) (taskOutputTiedPair_mem 0) (taskOutputTied_isMinOn 0)

/-- Epsilon-suboptimal ordinary training gives a size-independent full parameter error guarantee.
Source: the sharp actual ordinary-loss growth bound following sparsemax §2.5. -/
theorem outputTiedSquaredError_suboptimal_error {N D : ℕ} (floor budget energy : ℝ)
    (offset : Fin N → ℝ) (gain : ℝ) (channel : Fin D)
    (target : Matrix (Fin (N + 1)) (Fin D) ℝ)
    (x y : (Fin N → ℝ) × Matrix (Fin (N + 1)) (Fin D) ℝ) (epsilon : ℝ) (hf : 1 / 2 < floor)
    (hx : x ∈ outputTiedEnergyDomain N D floor budget energy offset gain channel)
    (hy : y ∈ outputTiedEnergyDomain N D floor budget energy offset gain channel)
    (hm : IsMinOn (periodicEnergySquaredError (fun j : Fin (N + 1) => j) target)
      (outputTiedEnergyDomain N D floor budget energy offset gain channel) y)
    (he : periodicEnergySquaredError (fun j : Fin (N + 1) => j) target x -
      periodicEnergySquaredError (fun j : Fin (N + 1) => j) target y ≤ epsilon) :
    periodicParameterError y x ≤ (1 + 4 * gain ^ 2) * epsilon := by
  exact (outputTiedSquaredError_parameter_growth floor budget energy offset gain channel target
    x y hf hx hy hm).trans (mul_le_mul_of_nonneg_left he (by positivity))

/-- A concrete positive ordinary-loss gap inhabits all approximate-training premises. -/
example : periodicParameterError ((taskEnergyParameters 0).1, taskEnergyTarget 0)
    ((taskEnergyParameters 1).1, taskEnergyTarget 1) ≤ (1 + 4 * (1 / 24 : ℝ) ^ 2) * 18 := by
  apply outputTiedSquaredError_suboptimal_error (3 / 4) (1 / 8) 6 _ _ _
    (taskEnergyTarget 0) _ _ 18 (by norm_num)
    (taskOutputTiedPair_mem 1) (taskOutputTiedPair_mem 0) (taskOutputTied_isMinOn 0)
  rw [taskOutputTied_loss_zero 0, sub_zero,
    periodicEnergySquaredError_eq (3 / 4) (1 / 8) 6 _ _ _ (by norm_num) (taskPeriodicEnergyPair_mem 1)]
  norm_num [matrixOutputError, taskEnergyTarget, Fin.sum_univ_succ, Matrix.of_apply]

/-- The proved nonempty-domain optimizer, rather than a supplied minimizer, gives the same error bound.
Source: attained optimization and sharp intrinsic growth following sparsemax §2.5. -/
theorem outputTiedMemoryOptimalPoint_error {N D : ℕ} (floor budget energy : ℝ)
    (offset : Fin N → ℝ) (gain : ℝ) (channel : Fin D)
    (target : Matrix (Fin (N + 1)) (Fin D) ℝ) (hf : 1 / 2 < floor)
    (hn : (outputTiedEnergyDomain N D floor budget energy offset gain channel).Nonempty)
    (x : (Fin N → ℝ) × Matrix (Fin (N + 1)) (Fin D) ℝ)
    (hx : x ∈ outputTiedEnergyDomain N D floor budget energy offset gain channel) :
    periodicParameterError (outputTiedMemoryOptimalPoint floor budget energy offset gain channel target hf hn) x ≤
      (1 + 4 * gain ^ 2) *
      (periodicEnergySquaredError (fun j : Fin (N + 1) => j) target x -
       periodicEnergySquaredError (fun j : Fin (N + 1) => j) target
        (outputTiedMemoryOptimalPoint floor budget energy offset gain channel target hf hn)) :=
  outputTiedSquaredError_parameter_growth floor budget energy offset gain channel target x _ hf hx
    (outputTiedMemoryOptimalPoint_mem floor budget energy offset gain channel target hf hn)
    (outputTiedMemoryOptimalPoint_min floor budget energy offset gain channel target hf hn)

/-- A changed learned answer point inhabits every proved-optimizer error premise. -/
example : periodicParameterError (outputTiedMemoryOptimalPoint (3 / 4) (1 / 8) 6
    (fun _ : Fin 2 => (1 / 24 : ℝ)) (1 / 24) 0 (taskEnergyTarget 0)
    (by norm_num) ⟨_, taskOutputTiedPair_mem 0⟩)
    ((taskEnergyParameters 1).1, taskEnergyTarget 1) ≤ (1 + 4 * (1 / 24 : ℝ) ^ 2) *
    (periodicEnergySquaredError (fun j : Fin 3 => j) (taskEnergyTarget 0)
      ((taskEnergyParameters 1).1, taskEnergyTarget 1) -
     periodicEnergySquaredError (fun j : Fin 3 => j) (taskEnergyTarget 0)
      (outputTiedMemoryOptimalPoint (3 / 4) (1 / 8) 6 (fun _ => (1 / 24 : ℝ)) (1 / 24) 0
        (taskEnergyTarget 0) (by norm_num) ⟨_, taskOutputTiedPair_mem 0⟩)) :=
  outputTiedMemoryOptimalPoint_error _ _ _ _ _ _ _ _ _ _ (taskOutputTiedPair_mem 1)

end Transformer.GPTMini.Sparsemax
