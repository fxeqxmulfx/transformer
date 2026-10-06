import Transformer.GPTMini.Sparsemax.EnergyMemoryExamples
import Mathlib.Analysis.Matrix.Order

/-!
# Meaning of the PSD coupling in the original common values

New restriction of the genuine memory chart after arXiv:1602.02068v2,
Eq. (1). On the coupled domain actual attention B is positive definite.
The block PSD condition is exactly the attention-weighted value-energy
bound `energy I - Vᵀ B V ≥ 0`, with Z=BV. This is not an ordinary bound
on Euclidean value norms. The single global table V remains trainable.

The Schur complement and inverse are justified from the actual attention
and the proved self-weight floor, rather than taken as unproved inputs.
The equivalence applies to arbitrary candidate original value tables at
each feasible geometry and does not fix their entries.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Matrix

/-- The energy coupling and self-weight floor imply actual positive definite attention.
Source: the new block PSD restriction and proved inverse after arXiv:1602.02068v2, Eq. (1). -/
theorem energyCoupledMemoryAttention_posDef {N D : ℕ} (cap floor budget energy : ℝ)
    (x : LocalMemoryParameters N × Matrix (Fin (N + 1)) (Fin D) ℝ)
    (hf : 1 / 2 < floor)
    (hx : x ∈ energyCoupledMemoryDomain N D cap floor budget energy) :
    (memoryGramAttention (localMemoryGram x.1)).PosDef := by
  have hp : (memoryGramScores (localMemoryGram x.1)).PosSemidef := by
    have hs := hx.2.2.submatrix (Sum.inl : Fin (N + 1) → Fin (N + 1) ⊕ Fin D)
    have he : (energyCoupledMemoryMatrix energy x).submatrix Sum.inl Sum.inl =
        memoryGramScores (localMemoryGram x.1) := by
      ext i j
      rfl
    rw [he] at hs
    exact hs
  rw [← memoryGramAttention_normalized cap floor _
    (incidentMemoryGram_mem cap floor x.1 (by linarith) hx.1)] at hp
  exact hp.posDef_iff_det_ne_zero.mpr
    (incidentMemoryAttention_det_unit cap floor x.1 hf hx.1).ne_zero

/-- A genuinely mixed attention matrix satisfies every positive-definiteness premise. -/
example : (memoryGramAttention (localMemoryGram (taskEnergyParameters 0))).PosDef :=
  energyCoupledMemoryAttention_posDef 4 (3 / 4) (1 / 8) 6
    (taskEnergyParameters 0, taskEnergyTarget 0) (by norm_num) (taskEnergyPair_mem 0)

/-- Actual path attention is symmetric throughout the incident parameter domain.
Source: the adjacent permutation embeddings before arXiv:1602.02068v2, Eq. (1). -/
theorem incidentMemoryAttention_transpose {N : ℕ} (cap floor : ℝ) (p : LocalMemoryParameters N)
    (hf : 0 ≤ floor) (hp : p ∈ incidentMemoryParameterDomain N cap floor) :
    (memoryGramAttention (localMemoryGram p)).transpose = memoryGramAttention (localMemoryGram p) := by
  rw [incidentMemoryAttention_normalized cap floor p hf hp]
  ext i j
  exact localMemoryCore_scores_symmetric p.1 j i

/-- Actual nonidentity attention inhabits the symmetry premises. -/
example : (memoryGramAttention (localMemoryGram (taskEnergyParameters 1))).transpose =
    memoryGramAttention (localMemoryGram (taskEnergyParameters 1)) :=
  incidentMemoryAttention_transpose 4 (3 / 4) _ (by norm_num) (taskEnergyParameters_mem 1)

/-- For arbitrary original values, block PSD is equivalent to the attention-weighted energy bound.
Source: the new coupling's Schur complement after arXiv:1602.02068v2, Eq. (1).
Feasible x supplies geometry; the candidate value table is unconstrained by x's own output entries. -/
theorem energyCoupledMemoryValues_iff {N D : ℕ} (cap floor budget energy : ℝ)
    (x : LocalMemoryParameters N × Matrix (Fin (N + 1)) (Fin D) ℝ)
    (values : Matrix (Fin (N + 1)) (Fin D) ℝ)
    (hf : 1 / 2 < floor) (hx : x ∈ energyCoupledMemoryDomain N D cap floor budget energy) :
    (energyCoupledMemoryMatrix energy
      (x.1, memoryValueOutput (localMemoryGram x.1) values)).PosSemidef ↔
    (energy • (1 : Matrix (Fin D) (Fin D) ℝ) - values.transpose *
      memoryGramAttention (localMemoryGram x.1) * values).PosSemidef := by
  let B := memoryGramAttention (localMemoryGram x.1)
  have hB := energyCoupledMemoryAttention_posDef cap floor budget energy x hf hx
  change B.PosDef at hB
  let : Invertible B := Matrix.invertibleOfIsUnitDet B
    (incidentMemoryAttention_det_unit cap floor x.1 hf hx.1)
  have hs := hB.fromBlocks₁₁ (B * values) (energy • (1 : Matrix (Fin D) (Fin D) ℝ))
  have hi : B⁻¹ * (B * values) = values := by
    exact Matrix.nonsing_inv_mul_cancel_left B values
      (incidentMemoryAttention_det_unit cap floor x.1 hf hx.1)
  have ht : (B * values).transpose = values.transpose * B := by
    rw [Matrix.transpose_mul, incidentMemoryAttention_transpose cap floor x.1 (by linarith) hx.1]
  rw [Matrix.conjTranspose_eq_transpose_of_trivial,
    Matrix.mul_assoc (B * values).transpose B⁻¹ (B * values), hi, ht] at hs
  unfold energyCoupledMemoryMatrix memoryValueOutput
  rw [← memoryGramAttention_normalized cap floor _
    (incidentMemoryGram_mem cap floor x.1 (by linarith) hx.1)]
  change (Matrix.fromBlocks B (B * values) (B * values).transpose
    (energy • (1 : Matrix (Fin D) (Fin D) ℝ))).PosSemidef ↔ _
  rw [ht]
  exact hs

/-- Nonconstant original values inhabit the exact energy-equivalence premises. -/
example : (energyCoupledMemoryMatrix 6
    (taskEnergyParameters 0, memoryValueOutput (localMemoryGram (taskEnergyParameters 0))
      (taskEnergyTarget 0))).PosSemidef ↔
    ((6 : ℝ) • (1 : Matrix (Fin 1) (Fin 1) ℝ) - (taskEnergyTarget 0).transpose *
      memoryGramAttention (localMemoryGram (taskEnergyParameters 0)) * taskEnergyTarget 0).PosSemidef :=
  energyCoupledMemoryValues_iff 4 (3 / 4) (1 / 8) 6
    (taskEnergyParameters 0, taskEnergyTarget 0) (taskEnergyTarget 0)
    (by norm_num) (taskEnergyPair_mem 0)

/-- The recovered original common values satisfy the actual attention-weighted energy bound.
Source: the new convex energy restriction after arXiv:1602.02068v2, Eq. (1). -/
theorem energyCoupledMemory_recovered_energy {N D : ℕ} (cap floor budget energy : ℝ)
    (x : LocalMemoryParameters N × Matrix (Fin (N + 1)) (Fin D) ℝ)
    (hf : 1 / 2 < floor) (hx : x ∈ energyCoupledMemoryDomain N D cap floor budget energy) :
    (energy • (1 : Matrix (Fin D) (Fin D) ℝ) -
      (recoverMemoryValues (localMemoryGram x.1) x.2).transpose *
        memoryGramAttention (localMemoryGram x.1) *
          recoverMemoryValues (localMemoryGram x.1) x.2).PosSemidef := by
  apply (energyCoupledMemoryValues_iff cap floor budget energy x _ hf hx).mp
  rw [recoverMemoryValues_exact cap floor _ _ hf
    (incidentMemoryGram_mem cap floor x.1 (by linarith) hx.1)]
  exact hx.2.2

/-- Learned nonconstant shared values satisfy the recovered-energy premises. -/
example : ((6 : ℝ) • (1 : Matrix (Fin 1) (Fin 1) ℝ) -
    (recoverMemoryValues (localMemoryGram (taskEnergyParameters 1)) (taskEnergyTarget 1)).transpose *
      memoryGramAttention (localMemoryGram (taskEnergyParameters 1)) *
        recoverMemoryValues (localMemoryGram (taskEnergyParameters 1)) (taskEnergyTarget 1)).PosSemidef :=
  energyCoupledMemory_recovered_energy 4 (3 / 4) (1 / 8) 6
    (taskEnergyParameters 1, taskEnergyTarget 1) (by norm_num) (taskEnergyPair_mem 1)

/-- The energy bound controls every output direction of the original globally decoded values.
Source: the new attention-weighted value-energy restriction after arXiv:1602.02068v2, Eq. (1). -/
theorem energyCoupledMemory_value_direction_bound {N D : ℕ} (cap floor budget energy : ℝ)
    (x : LocalMemoryParameters N × Matrix (Fin (N + 1)) (Fin D) ℝ)
    (u : Fin D → ℝ) (hf : 1 / 2 < floor)
    (hx : x ∈ energyCoupledMemoryDomain N D cap floor budget energy) :
    u ⬝ᵥ (((recoverMemoryValues (localMemoryGram x.1) x.2).transpose *
      memoryGramAttention (localMemoryGram x.1) *
        recoverMemoryValues (localMemoryGram x.1) x.2) *ᵥ u) ≤ energy * (u ⬝ᵥ u) := by
  have hp := energyCoupledMemory_recovered_energy cap floor budget energy x hf hx
  have h := hp.dotProduct_mulVec_nonneg u
  have he : 0 ≤ energy * (u ⬝ᵥ u) -
      u ⬝ᵥ (((recoverMemoryValues (localMemoryGram x.1) x.2).transpose *
        memoryGramAttention (localMemoryGram x.1) *
          recoverMemoryValues (localMemoryGram x.1) x.2) *ᵥ u) := by
    simpa only [star_trivial, Matrix.sub_mulVec, Matrix.smul_mulVec, Matrix.one_mulVec,
      dotProduct_sub, dotProduct_smul, smul_eq_mul] using h
  exact sub_nonneg.mp he

/-- Nonzero directions and learned nonconstant original values inhabit all bound premises. -/
example : (fun _ : Fin 1 => (1 : ℝ)) ⬝ᵥ
    (((recoverMemoryValues (localMemoryGram (taskEnergyParameters 0)) (taskEnergyTarget 0)).transpose *
      memoryGramAttention (localMemoryGram (taskEnergyParameters 0)) *
        recoverMemoryValues (localMemoryGram (taskEnergyParameters 0)) (taskEnergyTarget 0)) *ᵥ
          (fun _ : Fin 1 => (1 : ℝ))) ≤
    6 * ((fun _ : Fin 1 => (1 : ℝ)) ⬝ᵥ (fun _ : Fin 1 => (1 : ℝ))) :=
  energyCoupledMemory_value_direction_bound 4 (3 / 4) (1 / 8) 6
    (taskEnergyParameters 0, taskEnergyTarget 0) (fun _ => 1) (by norm_num) (taskEnergyPair_mem 0)

end Transformer.GPTMini.Sparsemax
