import Transformer.GPTMini.Sparsemax.OutputTiedMemoryGrowth

/-!
# Quantitative feasible descent toward the proved trained point

New ordinary-loss guarantee following arXiv:1602.02068v2, §2.5. The
feasible midpoint toward a minimum lowers the genuine sparsemax/common-value
criterion by at least full parameter distance divided by `4*(1+4*gain^2)`.
Every different feasible point has strict descent at every positive time
up to one, including across changes of sparse support and at positive-error
minima of unattainable answer tables.

The proved unique nonempty-domain optimizer supplies the endpoint; its
existence is not an extra assumption. These are mathematical finite-step
guarantees, rather than a numerical method for computing that endpoint.
The possible local mask, fixed affine readout and full observations remain
the stated restrictions of this constant-width jointly learned block.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

/-- A feasible midpoint gives a full-coordinate finite ordinary-loss gain independent of dictionary size.
Source: quantitative intrinsic curvature and constrained minimum comparison after sparsemax §2.5. -/
theorem outputTiedSquaredError_midpoint_gain {N D : ℕ} (floor budget energy : ℝ)
    (offset : Fin N → ℝ) (gain : ℝ) (channel : Fin D)
    (target : Matrix (Fin (N + 1)) (Fin D) ℝ)
    (x y : (Fin N → ℝ) × Matrix (Fin (N + 1)) (Fin D) ℝ) (hf : 1 / 2 < floor)
    (hx : x ∈ outputTiedEnergyDomain N D floor budget energy offset gain channel)
    (hy : y ∈ outputTiedEnergyDomain N D floor budget energy offset gain channel)
    (hm : IsMinOn (periodicEnergySquaredError (fun j : Fin (N + 1) => j) target)
      (outputTiedEnergyDomain N D floor budget energy offset gain channel) y) :
    periodicEnergySquaredError (fun j : Fin (N + 1) => j) target
      ((1 / 2 : ℝ) • x + (1 / 2 : ℝ) • y) +
      periodicParameterError y x / (4 * (1 + 4 * gain ^ 2)) ≤
        periodicEnergySquaredError (fun j : Fin (N + 1) => j) target x := by
  have hg := outputTiedSquaredError_curvature floor budget energy offset gain channel target
    x y hf hx hy (1 / 2) (1 / 2) (by norm_num) (by norm_num) (by norm_num)
  have hmin := hm hx
  change periodicEnergySquaredError _ _ y ≤ periodicEnergySquaredError _ _ x at hmin
  have hc := mul_le_mul_of_nonneg_left hmin (by positivity : 0 ≤ 1 + 4 * gain ^ 2)
  have hh : periodicParameterError y x ≤ 4 * (1 + 4 * gain ^ 2) *
      (periodicEnergySquaredError (fun j : Fin (N + 1) => j) target x -
       periodicEnergySquaredError (fun j : Fin (N + 1) => j) target
        ((1 / 2 : ℝ) • x + (1 / 2 : ℝ) • y)) := by
    nlinarith only [hg, hc]
  have hd : 0 < 4 * (1 + 4 * gain ^ 2) := by positivity
  have h : periodicParameterError y x / (4 * (1 + 4 * gain ^ 2)) ≤
      periodicEnergySquaredError (fun j : Fin (N + 1) => j) target x -
      periodicEnergySquaredError (fun j : Fin (N + 1) => j) target
        ((1 / 2 : ℝ) • x + (1 / 2 : ℝ) • y) :=
    (div_le_iff₀ hd).mpr (by nlinarith only [hh])
  linarith only [h]

/-- Different genuine learned supports inhabit every quantitative midpoint premise. -/
example : periodicEnergySquaredError (fun j : Fin 3 => j) (taskEnergyTarget 0)
    ((1 / 2 : ℝ) • ((taskEnergyParameters 1).1, taskEnergyTarget 1) +
      (1 / 2 : ℝ) • ((taskEnergyParameters 0).1, taskEnergyTarget 0)) +
    periodicParameterError ((taskEnergyParameters 0).1, taskEnergyTarget 0)
      ((taskEnergyParameters 1).1, taskEnergyTarget 1) / (4 * (1 + 4 * (1 / 24 : ℝ) ^ 2)) ≤
    periodicEnergySquaredError (fun j : Fin 3 => j) (taskEnergyTarget 0)
      ((taskEnergyParameters 1).1, taskEnergyTarget 1) :=
  outputTiedSquaredError_midpoint_gain (3 / 4) (1 / 8) 6 _ _ _ _ _ _ (by norm_num)
    (taskOutputTiedPair_mem 1) (taskOutputTiedPair_mem 0) (taskOutputTied_isMinOn 0)

/-- Every different feasible trained point has strict arbitrarily short descent toward a minimum.
Source: sharp ordinary-loss growth and actual convex feasible paths after sparsemax §2.5.
No zero-error or unchanged-support assumption is required. -/
theorem outputTiedSquaredError_strict_descent {N D : ℕ} (floor budget energy : ℝ)
    (offset : Fin N → ℝ) (gain : ℝ) (channel : Fin D)
    (target : Matrix (Fin (N + 1)) (Fin D) ℝ)
    (x y : (Fin N → ℝ) × Matrix (Fin (N + 1)) (Fin D) ℝ) (hf : 1 / 2 < floor)
    (hx : x ∈ outputTiedEnergyDomain N D floor budget energy offset gain channel)
    (hy : y ∈ outputTiedEnergyDomain N D floor budget energy offset gain channel)
    (hm : IsMinOn (periodicEnergySquaredError (fun j : Fin (N + 1) => j) target)
      (outputTiedEnergyDomain N D floor budget energy offset gain channel) y)
    (hxy : x ≠ y) (t : ℝ) (ht : 0 < t) (hu : t ≤ 1) :
    periodicEnergySquaredError (fun j : Fin (N + 1) => j) target ((1 - t) • x + t • y) <
      periodicEnergySquaredError (fun j : Fin (N + 1) => j) target x := by
  have hZ : x.2 ≠ y.2 := fun h => hxy
    (outputTiedEnergy_output_injective floor budget energy offset gain channel x y hx hy h)
  have hd := matrixOutputError_pos y.2 x.2 hZ
  have hg := outputTiedSquaredError_output_growth floor budget energy offset gain channel
    target x y hf hx hy hm
  have hb : periodicEnergySquaredError (fun j : Fin (N + 1) => j) target y <
      periodicEnergySquaredError (fun j : Fin (N + 1) => j) target x := by linarith
  exact periodicEnergyObjective_strict_descent floor budget energy _ (matrixOutputError target)
    x y hf (matrixOutputError_convex target) hx.1 hy.1 hb t ht hu

/-- Nonconstant ordinary answers and opposite learned edges inhabit all strict-descent premises. -/
example : periodicEnergySquaredError (fun j : Fin 3 => j) (taskEnergyTarget 0)
    ((1 - (1 / 2 : ℝ)) • ((taskEnergyParameters 1).1, taskEnergyTarget 1) +
      (1 / 2 : ℝ) • ((taskEnergyParameters 0).1, taskEnergyTarget 0)) <
    periodicEnergySquaredError (fun j : Fin 3 => j) (taskEnergyTarget 0)
      ((taskEnergyParameters 1).1, taskEnergyTarget 1) := by
  apply outputTiedSquaredError_strict_descent (3 / 4) (1 / 8) 6 _ _ _ _ _ _ (by norm_num)
    (taskOutputTiedPair_mem 1) (taskOutputTiedPair_mem 0) (taskOutputTied_isMinOn 0)
  · intro h
    have he := congrArg (fun p : (Fin 2 → ℝ) × Matrix (Fin 3) (Fin 1) ℝ => p.2 0 0) h
    norm_num [taskEnergyTarget, Matrix.of_apply] at he
  · norm_num
  · norm_num

/-- The proved attained optimizer gives the same finite gain without a supplied minimum.
Source: nonempty-domain optimizer existence and true ordinary-loss curvature after sparsemax §2.5. -/
theorem outputTiedMemoryOptimalPoint_midpoint_gain {N D : ℕ} (floor budget energy : ℝ)
    (offset : Fin N → ℝ) (gain : ℝ) (channel : Fin D)
    (target : Matrix (Fin (N + 1)) (Fin D) ℝ) (hf : 1 / 2 < floor)
    (hn : (outputTiedEnergyDomain N D floor budget energy offset gain channel).Nonempty)
    (x : (Fin N → ℝ) × Matrix (Fin (N + 1)) (Fin D) ℝ)
    (hx : x ∈ outputTiedEnergyDomain N D floor budget energy offset gain channel) :
    let q := outputTiedMemoryOptimalPoint floor budget energy offset gain channel target hf hn
    periodicEnergySquaredError (fun j : Fin (N + 1) => j) target
      ((1 / 2 : ℝ) • x + (1 / 2 : ℝ) • q) +
      periodicParameterError q x / (4 * (1 + 4 * gain ^ 2)) ≤
        periodicEnergySquaredError (fun j : Fin (N + 1) => j) target x :=
  outputTiedSquaredError_midpoint_gain floor budget energy offset gain channel target x _ hf hx
    (outputTiedMemoryOptimalPoint_mem floor budget energy offset gain channel target hf hn)
    (outputTiedMemoryOptimalPoint_min floor budget energy offset gain channel target hf hn)

/-- A genuine learned comparison point inhabits every proved-optimizer finite-gain premise. -/
example :
    let q := outputTiedMemoryOptimalPoint (3 / 4) (1 / 8) 6
      (fun _ : Fin 2 => (1 / 24 : ℝ)) (1 / 24) (0 : Fin 1) (taskEnergyTarget 0)
      (by norm_num) ⟨_, taskOutputTiedPair_mem 0⟩
    periodicEnergySquaredError (fun j : Fin 3 => j) (taskEnergyTarget 0)
      ((1 / 2 : ℝ) • ((taskEnergyParameters 1).1, taskEnergyTarget 1) + (1 / 2 : ℝ) • q) +
      periodicParameterError q ((taskEnergyParameters 1).1, taskEnergyTarget 1) /
        (4 * (1 + 4 * (1 / 24 : ℝ) ^ 2)) ≤
      periodicEnergySquaredError (fun j : Fin 3 => j) (taskEnergyTarget 0)
        ((taskEnergyParameters 1).1, taskEnergyTarget 1) :=
  outputTiedMemoryOptimalPoint_midpoint_gain _ _ _ _ _ _ _ _ _ _ (taskOutputTiedPair_mem 1)

/-- Every nonglobal point descends toward the proved optimizer, even when the best error is positive.
Source: unique attained training and sharp feasible descent after sparsemax §2.5. -/
theorem outputTiedMemoryOptimalPoint_strict_descent {N D : ℕ} (floor budget energy : ℝ)
    (offset : Fin N → ℝ) (gain : ℝ) (channel : Fin D)
    (target : Matrix (Fin (N + 1)) (Fin D) ℝ) (hf : 1 / 2 < floor)
    (hn : (outputTiedEnergyDomain N D floor budget energy offset gain channel).Nonempty)
    (x : (Fin N → ℝ) × Matrix (Fin (N + 1)) (Fin D) ℝ)
    (hx : x ∈ outputTiedEnergyDomain N D floor budget energy offset gain channel)
    (hneq : x ≠ outputTiedMemoryOptimalPoint floor budget energy offset gain channel target hf hn)
    (t : ℝ) (ht : 0 < t) (hu : t ≤ 1) :
    periodicEnergySquaredError (fun j : Fin (N + 1) => j) target
      ((1 - t) • x + t • outputTiedMemoryOptimalPoint floor budget energy offset gain channel target hf hn) <
      periodicEnergySquaredError (fun j : Fin (N + 1) => j) target x :=
  outputTiedSquaredError_strict_descent floor budget energy offset gain channel target x _ hf hx
    (outputTiedMemoryOptimalPoint_mem floor budget energy offset gain channel target hf hn)
    (outputTiedMemoryOptimalPoint_min floor budget energy offset gain channel target hf hn) hneq t ht hu

/-- Opposite learned answers satisfy every strict-descent premise relative to the proved optimizer. -/
example :
    let q := outputTiedMemoryOptimalPoint (3 / 4) (1 / 8) 6
      (fun _ : Fin 2 => (1 / 24 : ℝ)) (1 / 24) (0 : Fin 1) (taskEnergyTarget 0)
      (by norm_num) ⟨_, taskOutputTiedPair_mem 0⟩
    periodicEnergySquaredError (fun j : Fin 3 => j) (taskEnergyTarget 0)
      ((1 - (1 / 2 : ℝ)) • ((taskEnergyParameters 1).1, taskEnergyTarget 1) + (1 / 2 : ℝ) • q) <
      periodicEnergySquaredError (fun j : Fin 3 => j) (taskEnergyTarget 0)
        ((taskEnergyParameters 1).1, taskEnergyTarget 1) := by
  apply outputTiedMemoryOptimalPoint_strict_descent (3 / 4) (1 / 8) 6 _ _ _ _
    (by norm_num) ⟨_, taskOutputTiedPair_mem 0⟩ _ (taskOutputTiedPair_mem 1)
  · rw [taskOutputTied_optimalPoint_eq 0]
    intro h
    have he := congrArg (fun p : (Fin 2 → ℝ) × Matrix (Fin 3) (Fin 1) ℝ => p.2 0 0) h
    norm_num [taskEnergyTarget, Matrix.of_apply] at he
  · norm_num
  · norm_num

end Transformer.GPTMini.Sparsemax
