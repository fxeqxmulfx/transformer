import Transformer.GPTMini.Sparsemax.OutputTiedMemoryBounds

/-!
# Strict ordinary-loss curvature in every tied intrinsic direction

New derived training guarantee after arXiv:1602.02068v2, §2.5. With every
registered prototype observed, the unchanged ordinary vector-answer error
is strictly convex on the tied joint domain. A quantitative affine gap
controls complete edge/output distance with factor `1+4*gain^2`, independent
of dictionary size. No additional geometry criterion is optimized.

These statements hold for arbitrary targets, including unattainable ones.
Every constrained local minimum is global and every attained minimum is
unique in intrinsic coordinates. The fixed structural mask and affine
parameter sharing are explicit architectural restrictions, not a claim of
strict convexity for unrestricted physical Q/K/value parameter training.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

/-- Exact ordinary-loss affine gap on the full genuine tied sparsemax/common-value path.
Source: actual chart affinity and finite output curvature derived after sparsemax §2.5. -/
theorem outputTiedSquaredError_affine_gap {N D : ℕ} (floor budget energy : ℝ)
    (offset : Fin N → ℝ) (gain : ℝ) (channel : Fin D)
    (target : Matrix (Fin (N + 1)) (Fin D) ℝ)
    (x y : (Fin N → ℝ) × Matrix (Fin (N + 1)) (Fin D) ℝ) (hf : 1 / 2 < floor)
    (hx : x ∈ outputTiedEnergyDomain N D floor budget energy offset gain channel)
    (hy : y ∈ outputTiedEnergyDomain N D floor budget energy offset gain channel)
    (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : a + b = 1) :
    periodicEnergySquaredError (fun j : Fin (N + 1) => j) target (a • x + b • y) =
      a * periodicEnergySquaredError (fun j : Fin (N + 1) => j) target x +
      b * periodicEnergySquaredError (fun j : Fin (N + 1) => j) target y -
        a * b * matrixOutputError y.2 x.2 := by
  have hm := outputTiedEnergyDomain_convex N D floor budget energy offset gain channel hx hy ha hb hab
  rw [periodicEnergySquaredError_eq floor budget energy _ _ _ hf hm.1,
    periodicEnergySquaredError_eq floor budget energy _ _ _ hf hx.1,
    periodicEnergySquaredError_eq floor budget energy _ _ _ hf hy.1]
  exact matrixOutputError_affine_gap target x.2 y.2 a b hab

/-- Opposite learned Q/K supports and nonconstant values inhabit all exact affine-gap premises. -/
example : periodicEnergySquaredError (fun j : Fin 3 => j) (taskEnergyTarget 0)
    ((1 / 2 : ℝ) • ((taskEnergyParameters 0).1, taskEnergyTarget 0) +
      (1 / 2 : ℝ) • ((taskEnergyParameters 1).1, taskEnergyTarget 1)) =
    (1 / 2 : ℝ) * periodicEnergySquaredError (fun j : Fin 3 => j) (taskEnergyTarget 0)
      ((taskEnergyParameters 0).1, taskEnergyTarget 0) +
    (1 / 2 : ℝ) * periodicEnergySquaredError (fun j : Fin 3 => j) (taskEnergyTarget 0)
      ((taskEnergyParameters 1).1, taskEnergyTarget 1) -
    (1 / 2 : ℝ) * (1 / 2 : ℝ) * matrixOutputError (taskEnergyTarget 1) (taskEnergyTarget 0) :=
  outputTiedSquaredError_affine_gap (3 / 4) (1 / 8) 6 _ _ _ _ _ _ (by norm_num)
    (taskOutputTiedPair_mem 0) (taskOutputTiedPair_mem 1)
    _ _ (by norm_num) (by norm_num) (by norm_num)

/-- Ordinary answer error is strictly convex in every intrinsic tied parameter direction.
Source: the proved affine sharing and actual sparsemax forward after §2.5. -/
theorem outputTiedSquaredError_strictConvex {N D : ℕ} (floor budget energy : ℝ)
    (offset : Fin N → ℝ) (gain : ℝ) (channel : Fin D)
    (target : Matrix (Fin (N + 1)) (Fin D) ℝ) (hf : 1 / 2 < floor) :
    StrictConvexOn ℝ (outputTiedEnergyDomain N D floor budget energy offset gain channel)
      (periodicEnergySquaredError (fun j : Fin (N + 1) => j) target) := by
  refine ⟨outputTiedEnergyDomain_convex N D floor budget energy offset gain channel, ?_⟩
  intro x hx y hy hxy a b ha hb hab
  have hZ : x.2 ≠ y.2 := fun h => hxy
    (outputTiedEnergy_output_injective floor budget energy offset gain channel x y hx hy h)
  have hg := outputTiedSquaredError_affine_gap floor budget energy offset gain channel target
    x y hf hx hy a b ha.le hb.le hab
  have hn := mul_pos (mul_pos ha hb) (matrixOutputError_pos y.2 x.2 hZ)
  change periodicEnergySquaredError _ _ (a • x + b • y) <
    a * periodicEnergySquaredError _ _ x + b * periodicEnergySquaredError _ _ y
  linarith

/-- The nonsingleton regression domain inhabits the full strict-curvature premise. -/
example : StrictConvexOn ℝ (outputTiedEnergyDomain 2 1 (3 / 4) (1 / 8) 6
    (fun _ => (1 / 24 : ℝ)) (1 / 24) 0)
    (periodicEnergySquaredError (fun j : Fin 3 => j) (taskEnergyTarget 0)) :=
  outputTiedSquaredError_strictConvex _ _ _ _ _ _ _ (by norm_num)

/-- Complete intrinsic distance has a dictionary-size independent lower curvature bound.
Source: exact ordinary-output gap and the adjacent affine readout after sparsemax §2.5.
Multiplication by `1+4*gain^2` avoids a norm convention or hidden scaling in this statement. -/
theorem outputTiedSquaredError_curvature {N D : ℕ} (floor budget energy : ℝ)
    (offset : Fin N → ℝ) (gain : ℝ) (channel : Fin D)
    (target : Matrix (Fin (N + 1)) (Fin D) ℝ)
    (x y : (Fin N → ℝ) × Matrix (Fin (N + 1)) (Fin D) ℝ) (hf : 1 / 2 < floor)
    (hx : x ∈ outputTiedEnergyDomain N D floor budget energy offset gain channel)
    (hy : y ∈ outputTiedEnergyDomain N D floor budget energy offset gain channel)
    (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : a + b = 1) :
    a * b * periodicParameterError y x ≤ (1 + 4 * gain ^ 2) *
      (a * periodicEnergySquaredError (fun j : Fin (N + 1) => j) target x +
       b * periodicEnergySquaredError (fun j : Fin (N + 1) => j) target y -
       periodicEnergySquaredError (fun j : Fin (N + 1) => j) target (a • x + b • y)) := by
  have hp := mul_le_mul_of_nonneg_left
    (outputTiedEnergy_parameter_bound floor budget energy offset gain channel x y hx hy)
    (mul_nonneg ha hb)
  rw [outputTiedSquaredError_affine_gap floor budget energy offset gain channel target
    x y hf hx hy a b ha hb hab]
  nlinarith only [hp]

/-- Distinct real learned endpoints inhabit every complete-coordinate curvature premise. -/
example : (1 / 2 : ℝ) * (1 / 2 : ℝ) *
    periodicParameterError ((taskEnergyParameters 1).1, taskEnergyTarget 1)
      ((taskEnergyParameters 0).1, taskEnergyTarget 0) ≤ (1 + 4 * (1 / 24 : ℝ) ^ 2) *
    ((1 / 2 : ℝ) * periodicEnergySquaredError (fun j : Fin 3 => j) (taskEnergyTarget 0)
      ((taskEnergyParameters 0).1, taskEnergyTarget 0) +
     (1 / 2 : ℝ) * periodicEnergySquaredError (fun j : Fin 3 => j) (taskEnergyTarget 0)
      ((taskEnergyParameters 1).1, taskEnergyTarget 1) -
     periodicEnergySquaredError (fun j : Fin 3 => j) (taskEnergyTarget 0)
       ((1 / 2 : ℝ) • ((taskEnergyParameters 0).1, taskEnergyTarget 0) +
        (1 / 2 : ℝ) • ((taskEnergyParameters 1).1, taskEnergyTarget 1))) :=
  outputTiedSquaredError_curvature (3 / 4) (1 / 8) 6 _ _ _ _ _ _ (by norm_num)
    (taskOutputTiedPair_mem 0) (taskOutputTiedPair_mem 1)
    _ _ (by norm_num) (by norm_num) (by norm_num)

/-- Any attained ordinary-error minimum is unique in all tied intrinsic coordinates.
Source: the proved strict ordinary-loss curvature after arXiv:1602.02068v2, §2.5. -/
theorem outputTiedSquaredError_unique_min {N D : ℕ} (floor budget energy : ℝ)
    (offset : Fin N → ℝ) (gain : ℝ) (channel : Fin D)
    (target : Matrix (Fin (N + 1)) (Fin D) ℝ)
    (x y : (Fin N → ℝ) × Matrix (Fin (N + 1)) (Fin D) ℝ) (hf : 1 / 2 < floor)
    (hx : x ∈ outputTiedEnergyDomain N D floor budget energy offset gain channel)
    (hy : y ∈ outputTiedEnergyDomain N D floor budget energy offset gain channel)
    (hmx : IsMinOn (periodicEnergySquaredError (fun j : Fin (N + 1) => j) target)
      (outputTiedEnergyDomain N D floor budget energy offset gain channel) x)
    (hmy : IsMinOn (periodicEnergySquaredError (fun j : Fin (N + 1) => j) target)
      (outputTiedEnergyDomain N D floor budget energy offset gain channel) y) : x = y := by
  have ht := outputTiedSquaredError_strictConvex floor budget energy offset gain channel target hf
  exact ht.eq_of_isMinOn hmx hmy hx hy

/-- A genuine attained nonidentity fit inhabits every minimizer-uniqueness premise. -/
example : ((taskEnergyParameters 0).1, taskEnergyTarget 0) =
    ((taskEnergyParameters 0).1, taskEnergyTarget 0) :=
  outputTiedSquaredError_unique_min (3 / 4) (1 / 8) 6 _ _ _ _ _ _ (by norm_num)
    (taskOutputTiedPair_mem 0) (taskOutputTiedPair_mem 0)
    (taskOutputTied_isMinOn 0) (taskOutputTied_isMinOn 0)

/-- Every tied constrained local minimum is global for arbitrary ordinary answer tables.
Source: actual convex criterion plus convex extrema after arXiv:1602.02068v2, §2.5. -/
theorem outputTiedSquaredError_local_min_global {N D : ℕ} (floor budget energy : ℝ)
    (offset : Fin N → ℝ) (gain : ℝ) (channel : Fin D)
    (target : Matrix (Fin (N + 1)) (Fin D) ℝ)
    (x : (Fin N → ℝ) × Matrix (Fin (N + 1)) (Fin D) ℝ) (hf : 1 / 2 < floor)
    (hx : x ∈ outputTiedEnergyDomain N D floor budget energy offset gain channel)
    (hm : IsLocalMinOn (periodicEnergySquaredError (fun j : Fin (N + 1) => j) target)
      (outputTiedEnergyDomain N D floor budget energy offset gain channel) x) :
    IsMinOn (periodicEnergySquaredError (fun j : Fin (N + 1) => j) target)
      (outputTiedEnergyDomain N D floor budget energy offset gain channel) x := by
  exact IsMinOn.of_isLocalMinOn_of_convexOn hx hm
    (outputTiedEnergyObjective_convex floor budget energy offset gain channel _
      (matrixOutputError target) hf (matrixOutputError_convex target))

/-- The actual nonconstant fit supplies the local-minimum premise in the same nonsingleton domain. -/
example : IsMinOn (periodicEnergySquaredError (fun j : Fin 3 => j) (taskEnergyTarget 1))
    (outputTiedEnergyDomain 2 1 (3 / 4) (1 / 8) 6
      (fun _ => (1 / 24 : ℝ)) (1 / 24) 0)
    ((taskEnergyParameters 1).1, taskEnergyTarget 1) :=
  outputTiedSquaredError_local_min_global (3 / 4) (1 / 8) 6 _ _ _ _ _ (by norm_num)
    (taskOutputTiedPair_mem 1) (taskOutputTied_isMinOn 1).isLocalMinOn

end Transformer.GPTMini.Sparsemax
