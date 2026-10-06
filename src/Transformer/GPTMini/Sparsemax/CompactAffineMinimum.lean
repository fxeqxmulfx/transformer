import Transformer.GPTMini.Sparsemax.CompactAffineTraining

/-!
# Attained compact training without intrinsic flats or bad local minima

New constant-storage training guarantee after arXiv:1602.02068v2, §2.5.
Every nonempty compact coefficient domain has a unique attained ordinary
minimum for arbitrary observed answers. All constrained local minima are
global. Exact fitting and a supplied optimal point are not existence
assumptions. Compact learned geometry is shared with a slope coefficient.

Ordinary suboptimality bounds complete coefficient squared distance with
constant one. A feasible midpoint lowers error by at least half that
distance; every different feasible point has strict descent toward an
attained minimum at arbitrarily short positive segment times. The bounds
are independent of prototype count. They are finite mathematical
guarantees, not an implementation of a numerical optimizer.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

/-- Every nonempty compact model domain has a unique attained minimum for arbitrary ordinary targets.
Source: genuine forward continuity, compact state and strict error curvature after sparsemax §2.5. -/
theorem compactAffineSquaredError_existsUnique {N D : ℕ} (cap floor offset gain : ℝ) (channel : Fin D)
    (target : Matrix (Fin (N + 2)) (Fin D) ℝ) (hf : 1 / 2 < floor)
    (hn : (compactAffineMemoryDomain D cap floor offset gain channel).Nonempty) :
    ∃! W, W ∈ compactAffineMemoryDomain D cap floor offset gain channel ∧
      IsMinOn (compactAffineSquaredError offset gain channel target)
        (compactAffineMemoryDomain D cap floor offset gain channel) W := by
  obtain ⟨W, hW, hm⟩ := (compactAffineMemoryDomain_compact D cap floor offset gain channel).exists_isMinOn hn
    (compactAffineSquaredError_continuousOn cap floor offset gain channel target hf)
  refine ⟨W, ⟨hW, hm⟩, ?_⟩
  intro V hV
  exact (compactAffineSquaredError_strictConvex cap floor offset gain channel target hf).eq_of_isMinOn
    hV.2 hm hV.1 hW

/-- An arbitrary three-slot answer pattern inhabits every unique-attainment premise. -/
example : ∃! W, W ∈ compactAffineMemoryDomain 1 1 (3 / 4) (1 / 16) (1 / 16) 0 ∧
    IsMinOn (compactAffineSquaredError (1 / 16) (1 / 16) 0 !![0; 1; 0])
      (compactAffineMemoryDomain 1 1 (3 / 4) (1 / 16) (1 / 16) 0) W :=
  compactAffineSquaredError_existsUnique _ _ _ _ _ _ (by norm_num) ⟨_, compactAffineExample_mem 0⟩

/-- Every genuine constrained local minimum is global in all compact learned coordinates.
Source: actual strictly convex ordinary training following arXiv:1602.02068v2, §2.5. -/
theorem compactAffineSquaredError_local_min_global {N D : ℕ} (cap floor offset gain : ℝ) (channel : Fin D)
    (target : Matrix (Fin (N + 2)) (Fin D) ℝ) (W : Matrix (Fin 2) (Fin D) ℝ) (hf : 1 / 2 < floor)
    (hW : W ∈ compactAffineMemoryDomain D cap floor offset gain channel)
    (hl : IsLocalMinOn (compactAffineSquaredError offset gain channel target)
      (compactAffineMemoryDomain D cap floor offset gain channel) W) :
    IsMinOn (compactAffineSquaredError offset gain channel target)
      (compactAffineMemoryDomain D cap floor offset gain channel) W :=
  IsMinOn.of_isLocalMinOn_of_convexOn hW hl
    (compactAffineSquaredError_strictConvex cap floor offset gain channel target hf).convexOn

/-- A nonconstant actual compact fit inhabits every local-to-global premise. -/
example : IsMinOn (compactAffineSquaredError (1 / 16) (1 / 16) 0
    (affineValueOutput 2 (compactAffineExampleCoefficients 1)))
    (compactAffineMemoryDomain 1 1 (3 / 4) (1 / 16) (1 / 16) 0) (compactAffineExampleCoefficients 1) :=
  compactAffineSquaredError_local_min_global _ _ _ _ _ _ _ (by norm_num) (compactAffineExample_mem 1)
    (compactAffineExample_min 2 1).isLocalMinOn

/-- Ordinary suboptimality controls every learned coefficient, without an exact fitting assumption.
Source: feasible midpoint comparison and uniform full-state curvature following sparsemax §2.5. -/
theorem compactAffineSquaredError_parameter_growth {N D : ℕ} (cap floor offset gain : ℝ) (channel : Fin D)
    (target : Matrix (Fin (N + 2)) (Fin D) ℝ) (W V : Matrix (Fin 2) (Fin D) ℝ) (hf : 1 / 2 < floor)
    (hW : W ∈ compactAffineMemoryDomain D cap floor offset gain channel)
    (hV : V ∈ compactAffineMemoryDomain D cap floor offset gain channel)
    (hm : IsMinOn (compactAffineSquaredError offset gain channel target)
      (compactAffineMemoryDomain D cap floor offset gain channel) V) :
    matrixOutputError V W ≤ compactAffineSquaredError offset gain channel target W -
      compactAffineSquaredError offset gain channel target V := by
  have hmid := compactAffineMemoryDomain_convex D cap floor offset gain channel hW hV
    (by norm_num : 0 ≤ (1 / 2 : ℝ)) (by norm_num : 0 ≤ (1 / 2 : ℝ)) (by norm_num)
  have hmin := hm hmid
  change compactAffineSquaredError offset gain channel target V ≤
    compactAffineSquaredError offset gain channel target ((1 / 2 : ℝ) • W + (1 / 2 : ℝ) • V) at hmin
  have hg := compactAffineSquaredError_curvature cap floor offset gain channel target W V hf hW hV
    (1 / 2) (1 / 2) (by norm_num) (by norm_num) (by norm_num)
  linarith

/-- Distinct nonconstant compact answers inhabit every complete parameter-growth premise. -/
example : matrixOutputError (compactAffineExampleCoefficients 0) (compactAffineExampleCoefficients 1) ≤
    compactAffineSquaredError (1 / 16) (1 / 16) 0 (affineValueOutput 2 (compactAffineExampleCoefficients 0))
      (compactAffineExampleCoefficients 1) -
    compactAffineSquaredError (1 / 16) (1 / 16) 0 (affineValueOutput 2 (compactAffineExampleCoefficients 0))
      (compactAffineExampleCoefficients 0) :=
  compactAffineSquaredError_parameter_growth _ _ _ _ _ _ _ _ (by norm_num)
    (compactAffineExample_mem 1) (compactAffineExample_mem 0) (compactAffineExample_min 2 0)

/-- A feasible midpoint gives a full learned-state finite loss gain independent of dictionary size.
Source: the compact ordinary-loss affine gap following sparsemax §2.5. -/
theorem compactAffineSquaredError_midpoint_gain {N D : ℕ} (cap floor offset gain : ℝ) (channel : Fin D)
    (target : Matrix (Fin (N + 2)) (Fin D) ℝ) (W V : Matrix (Fin 2) (Fin D) ℝ) (hf : 1 / 2 < floor)
    (hW : W ∈ compactAffineMemoryDomain D cap floor offset gain channel)
    (hV : V ∈ compactAffineMemoryDomain D cap floor offset gain channel)
    (hm : IsMinOn (compactAffineSquaredError offset gain channel target)
      (compactAffineMemoryDomain D cap floor offset gain channel) V) :
    compactAffineSquaredError offset gain channel target ((1 / 2 : ℝ) • W + (1 / 2 : ℝ) • V) +
      (1 / 2 : ℝ) * matrixOutputError V W ≤ compactAffineSquaredError offset gain channel target W := by
  have hg := compactAffineSquaredError_curvature cap floor offset gain channel target W V hf hW hV
    (1 / 2) (1 / 2) (by norm_num) (by norm_num) (by norm_num)
  have hmin := hm hW
  change compactAffineSquaredError offset gain channel target V ≤ compactAffineSquaredError offset gain channel target W at hmin
  linarith

/-- Changed compact geometry and nonconstant original values inhabit every finite-gain premise. -/
example : compactAffineSquaredError (1 / 16) (1 / 16) 0 (affineValueOutput 2 (compactAffineExampleCoefficients 0))
    ((1 / 2 : ℝ) • compactAffineExampleCoefficients 1 + (1 / 2 : ℝ) • compactAffineExampleCoefficients 0) +
    (1 / 2 : ℝ) * matrixOutputError (compactAffineExampleCoefficients 0) (compactAffineExampleCoefficients 1) ≤
    compactAffineSquaredError (1 / 16) (1 / 16) 0 (affineValueOutput 2 (compactAffineExampleCoefficients 0))
      (compactAffineExampleCoefficients 1) :=
  compactAffineSquaredError_midpoint_gain _ _ _ _ _ _ _ _ (by norm_num)
    (compactAffineExample_mem 1) (compactAffineExample_mem 0) (compactAffineExample_min 2 0)

/-- Every different feasible compact learned point strictly descends toward any attained minimum.
Source: complete parameter growth and genuine convex ordinary training following sparsemax §2.5. -/
theorem compactAffineSquaredError_strict_descent {N D : ℕ} (cap floor offset gain : ℝ) (channel : Fin D)
    (target : Matrix (Fin (N + 2)) (Fin D) ℝ) (W V : Matrix (Fin 2) (Fin D) ℝ) (hf : 1 / 2 < floor)
    (hW : W ∈ compactAffineMemoryDomain D cap floor offset gain channel)
    (hV : V ∈ compactAffineMemoryDomain D cap floor offset gain channel)
    (hm : IsMinOn (compactAffineSquaredError offset gain channel target)
      (compactAffineMemoryDomain D cap floor offset gain channel) V)
    (hne : W ≠ V) (t : ℝ) (ht : 0 < t) (hu : t ≤ 1) :
    compactAffineSquaredError offset gain channel target ((1 - t) • W + t • V) <
      compactAffineSquaredError offset gain channel target W := by
  have hg := compactAffineSquaredError_parameter_growth cap floor offset gain channel target W V hf hW hV hm
  have hp := matrixOutputError_pos V W hne
  have hs := (compactAffineSquaredError_strictConvex cap floor offset gain channel target hf).convexOn.2
    hW hV (by linarith : 0 ≤ 1 - t) ht.le (by ring)
  change compactAffineSquaredError offset gain channel target ((1 - t) • W + t • V) ≤
    (1 - t) * compactAffineSquaredError offset gain channel target W +
      t * compactAffineSquaredError offset gain channel target V at hs
  have hd : 0 < t * (compactAffineSquaredError offset gain channel target W -
      compactAffineSquaredError offset gain channel target V) := mul_pos ht (by linarith)
  nlinarith only [hs, hd]

/-- Distinct genuine learned compact points inhabit every arbitrarily short descent premise. -/
example : compactAffineSquaredError (1 / 16) (1 / 16) 0 (affineValueOutput 2 (compactAffineExampleCoefficients 0))
    ((1 - (1 / 2 : ℝ)) • compactAffineExampleCoefficients 1 + (1 / 2 : ℝ) • compactAffineExampleCoefficients 0) <
    compactAffineSquaredError (1 / 16) (1 / 16) 0 (affineValueOutput 2 (compactAffineExampleCoefficients 0))
      (compactAffineExampleCoefficients 1) := by
  apply compactAffineSquaredError_strict_descent _ _ _ _ _ _ _ _ (by norm_num)
    (compactAffineExample_mem 1) (compactAffineExample_mem 0) (compactAffineExample_min 2 0)
  · intro h
    have he := congrArg (fun W : Matrix (Fin 2) (Fin 1) ℝ => W 1 0) h
    norm_num [compactAffineExampleCoefficients] at he
  · norm_num
  · norm_num

end Transformer.GPTMini.Sparsemax
