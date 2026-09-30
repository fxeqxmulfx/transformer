/-
# Muon — correctness of Distributed Muon

arXiv:2502.16982, §2.3, Algorithm 1 (`alg:distribmuon`). An equivalence
between matrix entries and `(device, local entry)` expresses a lossless
partition. The theorem includes local momentum, gathering before the
nonlinear operation, local weight updates, and the final all-gather.
Communication here is exact; the paper's bf16 communication introduces
rounding and is not claimed to equal an exact real-arithmetic execution.
-/

import Transformer.Muon.Section2_Models

open scoped BigOperators Matrix

noncomputable section

namespace Transformer.Muon

variable {a b dp shardSize : ℕ}

/-- Partitioning by entry indices, arXiv:2502.16982, §2.3, Algorithm 1. -/
def scatter (e : (Fin a × Fin b) ≃ (Fin dp × Fin shardSize))
    (X : Matrix (Fin a) (Fin b) ℝ) : Fin dp → Fin shardSize → ℝ :=
  fun p k => X (e.symm (p, k)).1 (e.symm (p, k)).2

/-- Reconstructing a full matrix, arXiv:2502.16982, §2.3, Algorithm 1. -/
def gather (e : (Fin a × Fin b) ≃ (Fin dp × Fin shardSize))
    (x : Fin dp → Fin shardSize → ℝ) : Matrix (Fin a) (Fin b) ℝ :=
  fun i j => x (e (i, j)).1 (e (i, j)).2

/-- Gathering a lossless partition reconstructs the matrix,
arXiv:2502.16982, §2.3, Algorithm 1. -/
theorem gather_scatter (e : (Fin a × Fin b) ≃ (Fin dp × Fin shardSize))
    (X : Matrix (Fin a) (Fin b) ℝ) : gather e (scatter e X) = X := by
  ext i j
  simp [gather, scatter]

/-- Scattering a reconstruction recovers every local entry,
arXiv:2502.16982, §2.3, Algorithm 1. -/
theorem scatter_gather (e : (Fin a × Fin b) ≃ (Fin dp × Fin shardSize))
    (x : Fin dp → Fin shardSize → ℝ) : scatter e (gather e x) = x := by
  funext p k
  simp [gather, scatter]

/-- Gathering commutes with local addition, arXiv:2502.16982, §2.3, Algorithm 1. -/
theorem gather_add (e : (Fin a × Fin b) ≃ (Fin dp × Fin shardSize))
    (x y : Fin dp → Fin shardSize → ℝ) : gather e (x + y) = gather e x + gather e y := rfl

/-- Gathering commutes with local subtraction, arXiv:2502.16982, §2.3, Algorithm 1. -/
theorem gather_sub (e : (Fin a × Fin b) ≃ (Fin dp × Fin shardSize))
    (x y : Fin dp → Fin shardSize → ℝ) : gather e (x - y) = gather e x - gather e y := rfl

/-- Gathering commutes with local scaling, arXiv:2502.16982, §2.3, Algorithm 1. -/
theorem gather_smul (e : (Fin a × Fin b) ≃ (Fin dp × Fin shardSize))
    (c : ℝ) (x : Fin dp → Fin shardSize → ℝ) : gather e (c • x) = c • gather e x := rfl

/-- The gathered sum of reduced local gradient shards is the corresponding
full gradient sum, arXiv:2502.16982, §2.3, Algorithm 1, “Reduce-scatter”. -/
theorem gather_reduced_gradients (e : (Fin a × Fin b) ≃ (Fin dp × Fin shardSize))
    {replicas : ℕ} (G : Fin replicas → Matrix (Fin a) (Fin b) ℝ) :
    gather e (∑ q, scatter e (G q)) = ∑ q, G q := by
  ext i j
  simp [gather, scatter, Matrix.sum_apply]

/-- Distributed Muon with local optimizer states, arXiv:2502.16982, §2.3,
Algorithm 1. `G` is the correctly reduced gradient matrix. -/
def distributedMuonStep (e : (Fin a × Fin b) ≃ (Fin dp × Fin shardSize))
    (μ η wd : ℝ) (m w : Fin dp → Fin shardSize → ℝ) (G : Matrix (Fin a) (Fin b) ℝ) :
    Matrix (Fin a) (Fin b) ℝ × Matrix (Fin a) (Fin b) ℝ :=
  let g := scatter e G
  let m' := μ • m + g
  let nesterov := μ • m' + g
  let u := scatter e (adjustedUpdate (approximatePolar (gather e nesterov)))
  let w' := w - η • (u + wd • w)
  (gather e m', gather e w')

/-- Distributed Muon has exactly the same momentum and weight results as
centralized Muon when its shards are gathered losslessly and arithmetic
precision is the same. This is the mathematical preservation asserted in
arXiv:2502.16982, §2.3, Algorithm 1. -/
theorem distributedMuonStep_eq (e : (Fin a × Fin b) ≃ (Fin dp × Fin shardSize))
    (μ η wd : ℝ) (M W G : Matrix (Fin a) (Fin b) ℝ) :
    distributedMuonStep e μ η wd (scatter e M) (scatter e W) G = muonStep μ η wd M W G := by
  simp only [distributedMuonStep, muonStep, nesterovInput, momentumStep, decayStep,
    gather_add, gather_smul, gather_sub, gather_scatter]

/-- Local momentum also reconstructs correctly for arbitrary sharded states,
arXiv:2502.16982, §2.3, Algorithm 1. -/
theorem distributedMuonStep_arbitrary_states
    (e : (Fin a × Fin b) ≃ (Fin dp × Fin shardSize)) (μ η wd : ℝ)
    (m w : Fin dp → Fin shardSize → ℝ) (G : Matrix (Fin a) (Fin b) ℝ) :
    distributedMuonStep e μ η wd m w G = muonStep μ η wd (gather e m) (gather e w) G := by
  simpa only [scatter_gather] using distributedMuonStep_eq e μ η wd (gather e m) (gather e w) G

/-- Local orthogonalization is not interchangeable with full-matrix
orthogonalization: normalizing `[1,1]` yields entries `1/√2`, whereas
normalizing each one-entry shard gives `1`. This justifies the full gather.
Source: arXiv:2502.16982, §2.1 and §2.3, “Calculate Full Update”. -/
theorem normalization_not_entrywise :
    schulzInitial (fun _ : Fin 1 => fun _ : Fin 2 => (1 : ℝ)) 0 0 ≠
      schulzInitial (1 : Matrix (Fin 1) (Fin 1) ℝ) 0 0 := by
  norm_num [schulzInitial, squaredFrobenius]
  intro h
  change (Real.sqrt 2)⁻¹ * 1 = 1 at h
  simp only [mul_one] at h
  have hroot : Real.sqrt 2 = 1 := inv_eq_one.mp h
  have hs := Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 2)
  rw [hroot] at hs
  norm_num at hs

end Transformer.Muon
