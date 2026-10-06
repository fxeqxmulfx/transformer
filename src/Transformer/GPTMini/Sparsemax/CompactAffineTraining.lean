import Transformer.GPTMini.Sparsemax.CompactAffineMemory

/-!
# Strictly convex ordinary training with constant-size learned state

New training guarantee after arXiv:1602.02068v2, §2.5. The actual
width-three masked sparsemax/common-value forward equals the generated
affine response. Its ordinary sum squared error is strictly convex in all
two-row learned coefficients whenever every prototype is observed.
Both Q/K and original values vary through the fixed scalar readout.

Targets are arbitrary observed vector answers, with no supervised routes
or added geometry criterion. Exact fitting is limited to the generated
response family and coefficient/scalar constraints. The nonconstant finite
witnesses fit with different actual supports on the same compact domain.
Continuity follows from the proved forward identity on the true domain,
not from an assumed globally continuous matrix inverse.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

/-- The ordinary sum squared error of genuine compact sparsemax/common-value predictions.
Source: the new restricted output criterion following arXiv:1602.02068v2, §2.5. -/
def compactAffineSquaredError {N D : ℕ} (offset gain : ℝ) (channel : Fin D)
    (target : Matrix (Fin (N + 2)) (Fin D) ℝ) (W : Matrix (Fin 2) (Fin D) ℝ) : ℝ :=
  matrixOutputError target (compactAffineMemoryForward (fun j : Fin (N + 2) => j) offset gain channel W)

/-- The actual ordinary prediction criterion equals the coefficient-linear response criterion.
Source: the proved compact original value decoder following sparsemax Eq. (1) and §2.5. -/
theorem compactAffineSquaredError_eq {N D : ℕ} (cap floor offset gain : ℝ) (channel : Fin D)
    (target : Matrix (Fin (N + 2)) (Fin D) ℝ) (W : Matrix (Fin 2) (Fin D) ℝ) (hf : 1 / 2 < floor)
    (hW : W ∈ compactAffineMemoryDomain D cap floor offset gain channel) :
    compactAffineSquaredError offset gain channel target W = matrixOutputError target (affineValueOutput N W) := by
  unfold compactAffineSquaredError
  rw [compactAffineMemoryForward_eq cap floor offset gain channel _ W hf hW]
  rfl

/-- A nonconstant four-slot target and true nonidentity forward inhabit all criterion premises. -/
example : compactAffineSquaredError (1 / 16) (1 / 16) 0
    (affineValueOutput 2 (compactAffineExampleCoefficients 0)) (compactAffineExampleCoefficients 1) =
    matrixOutputError (affineValueOutput 2 (compactAffineExampleCoefficients 0))
      (affineValueOutput 2 (compactAffineExampleCoefficients 1)) :=
  compactAffineSquaredError_eq 1 (3 / 4) (1 / 16) (1 / 16) 0 _ _ (by norm_num) (compactAffineExample_mem 1)

/-- The genuine compact joint objective has its exact observed-distance affine gap.
Source: actual forward affinity and ordinary output curvature after sparsemax §2.5. -/
theorem compactAffineSquaredError_affine_gap {N D : ℕ} (cap floor offset gain : ℝ) (channel : Fin D)
    (target : Matrix (Fin (N + 2)) (Fin D) ℝ) (W V : Matrix (Fin 2) (Fin D) ℝ) (hf : 1 / 2 < floor)
    (hW : W ∈ compactAffineMemoryDomain D cap floor offset gain channel)
    (hV : V ∈ compactAffineMemoryDomain D cap floor offset gain channel)
    (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : a + b = 1) :
    compactAffineSquaredError offset gain channel target (a • W + b • V) =
      a * compactAffineSquaredError offset gain channel target W +
        b * compactAffineSquaredError offset gain channel target V -
          a * b * matrixOutputError (affineValueOutput N V) (affineValueOutput N W) := by
  have hm := compactAffineMemoryDomain_convex D cap floor offset gain channel hW hV ha hb hab
  rw [compactAffineSquaredError_eq cap floor offset gain channel target _ hf hm,
    compactAffineSquaredError_eq cap floor offset gain channel target W hf hW,
    compactAffineSquaredError_eq cap floor offset gain channel target V hf hV,
    affineValueOutput_linear, matrixOutputError_affine_gap _ _ _ a b hab]

/-- Distinct compact learned points and their support-changing midpoint inhabit every affine-gap premise. -/
example : compactAffineSquaredError (1 / 16) (1 / 16) 0
    (affineValueOutput 2 (compactAffineExampleCoefficients 0))
    ((1 / 2 : ℝ) • compactAffineExampleCoefficients 0 + (1 / 2 : ℝ) • compactAffineExampleCoefficients 1) =
    (1 / 2 : ℝ) * compactAffineSquaredError (1 / 16) (1 / 16) 0
      (affineValueOutput 2 (compactAffineExampleCoefficients 0)) (compactAffineExampleCoefficients 0) +
    (1 / 2 : ℝ) * compactAffineSquaredError (1 / 16) (1 / 16) 0
      (affineValueOutput 2 (compactAffineExampleCoefficients 0)) (compactAffineExampleCoefficients 1) -
    (1 / 2 : ℝ) * (1 / 2 : ℝ) * matrixOutputError
      (affineValueOutput 2 (compactAffineExampleCoefficients 1)) (affineValueOutput 2 (compactAffineExampleCoefficients 0)) :=
  compactAffineSquaredError_affine_gap 1 (3 / 4) (1 / 16) (1 / 16) 0 _ _ _ (by norm_num)
    (compactAffineExample_mem 0) (compactAffineExample_mem 1) _ _ (by norm_num) (by norm_num) (by norm_num)

/-- Ordinary training has no affine flat directions in the complete compact learned state.
Source: genuine compact forward and injective generated response following arXiv:1602.02068v2, §2.5. -/
theorem compactAffineSquaredError_strictConvex {N D : ℕ} (cap floor offset gain : ℝ) (channel : Fin D)
    (target : Matrix (Fin (N + 2)) (Fin D) ℝ) (hf : 1 / 2 < floor) :
    StrictConvexOn ℝ (compactAffineMemoryDomain D cap floor offset gain channel)
      (compactAffineSquaredError offset gain channel target) := by
  refine ⟨compactAffineMemoryDomain_convex D cap floor offset gain channel, ?_⟩
  intro W hW V hV hWV a b ha hb hab
  have ho : affineValueOutput N W ≠ affineValueOutput N V :=
    fun h => hWV (affineValueOutput_injective N D h)
  have hp := mul_pos (mul_pos ha hb) (matrixOutputError_pos _ _ ho)
  have hg := compactAffineSquaredError_affine_gap cap floor offset gain channel target W V hf hW hV
    a b ha.le hb.le hab
  change compactAffineSquaredError offset gain channel target (a • W + b • V) <
    a * compactAffineSquaredError offset gain channel target W + b * compactAffineSquaredError offset gain channel target V
  linarith

/-- Nonconstant four-slot ordinary answers inhabit the strict-curvature floor hypothesis. -/
example : StrictConvexOn ℝ (compactAffineMemoryDomain 1 1 (3 / 4) (1 / 16) (1 / 16) 0)
    (compactAffineSquaredError (1 / 16) (1 / 16) 0 (affineValueOutput 2 (compactAffineExampleCoefficients 0))) :=
  compactAffineSquaredError_strictConvex _ _ _ _ _ _ (by norm_num)

/-- The actual ordinary objective is continuous on its genuine compact inverse domain.
Source: the exact affine response identity following sparsemax Eq. (1) and §2.5. -/
theorem compactAffineSquaredError_continuousOn {N D : ℕ} (cap floor offset gain : ℝ) (channel : Fin D)
    (target : Matrix (Fin (N + 2)) (Fin D) ℝ) (hf : 1 / 2 < floor) :
    ContinuousOn (compactAffineSquaredError offset gain channel target)
      (compactAffineMemoryDomain D cap floor offset gain channel) := by
  have ho : Continuous (affineValueOutput N : Matrix (Fin 2) (Fin D) ℝ → _) := by
    apply continuous_pi
    intro i
    apply continuous_pi
    intro d
    exact (continuous_id.matrix_elem 0 d).add
      ((continuous_id.matrix_elem 1 d).const_mul (affineValuePosition N i))
  apply ((matrixOutputError_continuous target).comp ho).continuousOn.congr
  intro W hW
  exact compactAffineSquaredError_eq cap floor offset gain channel target W hf hW

/-- Changed sparse supports inhabit every true compact-loss continuity premise. -/
example : ContinuousOn (compactAffineSquaredError (1 / 16) (1 / 16) 0
    (affineValueOutput 2 (compactAffineExampleCoefficients 1)))
    (compactAffineMemoryDomain 1 1 (3 / 4) (1 / 16) (1 / 16) 0) :=
  compactAffineSquaredError_continuousOn _ _ _ _ _ _ (by norm_num)

/-- Both nonconstant targets fit through one generated common value function for every dictionary size.
Source: complete genuine compact-forward witnesses following sparsemax Eq. (1). -/
theorem compactAffineExample_loss_zero (N : ℕ) (edge : Fin 2) :
    compactAffineSquaredError (1 / 16) (1 / 16) 0
      (affineValueOutput N (compactAffineExampleCoefficients edge)) (compactAffineExampleCoefficients edge) = 0 := by
  rw [compactAffineSquaredError_eq 1 (3 / 4) (1 / 16) (1 / 16) 0 _ _
    (by norm_num) (compactAffineExample_mem edge)]
  exact (matrixOutputError_eq_zero _ _).mpr rfl

/-- Both actual nonidentity/identity compact fits are global ordinary-error minima.
Source: exact generated predictions and nonnegativity of ordinary error following sparsemax §2.5. -/
theorem compactAffineExample_min (N : ℕ) (edge : Fin 2) :
    IsMinOn (compactAffineSquaredError (1 / 16) (1 / 16) 0 (affineValueOutput N (compactAffineExampleCoefficients edge)))
      (compactAffineMemoryDomain 1 1 (3 / 4) (1 / 16) (1 / 16) 0) (compactAffineExampleCoefficients edge) := by
  intro W hW
  change compactAffineSquaredError _ _ _ _ _ ≤ compactAffineSquaredError _ _ _ _ _
  rw [compactAffineExample_loss_zero]
  exact matrixOutputError_nonneg _ _

/-- Ordinary output curvature controls all learned coefficients with a size-independent factor two.
Source: generated endpoint observation bounds and the actual affine gap after sparsemax §2.5. -/
theorem compactAffineSquaredError_curvature {N D : ℕ} (cap floor offset gain : ℝ) (channel : Fin D)
    (target : Matrix (Fin (N + 2)) (Fin D) ℝ) (W V : Matrix (Fin 2) (Fin D) ℝ) (hf : 1 / 2 < floor)
    (hW : W ∈ compactAffineMemoryDomain D cap floor offset gain channel)
    (hV : V ∈ compactAffineMemoryDomain D cap floor offset gain channel)
    (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : a + b = 1) :
    2 * a * b * matrixOutputError V W ≤
      a * compactAffineSquaredError offset gain channel target W +
        b * compactAffineSquaredError offset gain channel target V -
          compactAffineSquaredError offset gain channel target (a • W + b • V) := by
  have hg := compactAffineSquaredError_affine_gap cap floor offset gain channel target W V hf hW hV a b ha hb hab
  have hd := mul_le_mul_of_nonneg_left (affineValueOutput_distance N W V) (mul_nonneg ha hb)
  nlinarith only [hg, hd]

/-- Nonconstant learned coefficients and changing supports inhabit every quantitative curvature premise. -/
example : 2 * (1 / 2 : ℝ) * (1 / 2 : ℝ) * matrixOutputError
    (compactAffineExampleCoefficients 1) (compactAffineExampleCoefficients 0) ≤
    (1 / 2 : ℝ) * compactAffineSquaredError (1 / 16) (1 / 16) 0
      (affineValueOutput 2 (compactAffineExampleCoefficients 0)) (compactAffineExampleCoefficients 0) +
    (1 / 2 : ℝ) * compactAffineSquaredError (1 / 16) (1 / 16) 0
      (affineValueOutput 2 (compactAffineExampleCoefficients 0)) (compactAffineExampleCoefficients 1) -
    compactAffineSquaredError (1 / 16) (1 / 16) 0 (affineValueOutput 2 (compactAffineExampleCoefficients 0))
      ((1 / 2 : ℝ) • compactAffineExampleCoefficients 0 + (1 / 2 : ℝ) • compactAffineExampleCoefficients 1) :=
  compactAffineSquaredError_curvature 1 (3 / 4) (1 / 16) (1 / 16) 0 _ _ _ (by norm_num)
    (compactAffineExample_mem 0) (compactAffineExample_mem 1) _ _ (by norm_num) (by norm_num) (by norm_num)

end Transformer.GPTMini.Sparsemax
