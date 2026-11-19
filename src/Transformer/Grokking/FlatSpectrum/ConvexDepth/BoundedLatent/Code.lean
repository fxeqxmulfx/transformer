module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.BoundedLatent.Confidence
public import Transformer.Grokking.FlatSpectrum.ConvexDepth.LatentPrograms.Witness
public import Mathlib.Data.Matrix.Block

/-!
# Exact block Fourier sources for arbitrary finite latent banks

Source: `review/convex/moments.md`, CM64, native-code extension. Replicated
four-point Fourier blocks supply the actual unitary premise at every padded
latent size, with entries zero or signed real/imaginary halves.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.BoundedLatent

variable {B S : Type*} [Fintype B] [DecidableEq B] [Fintype S] [DecidableEq S]

/-- Complete block diagonal composition preserves both unitary identities,
CM64. Sparsity of the source is exact, not a dropped learned coefficient. -/
theorem block_unitary (U : B → Matrix S S ℂ)
    (hU : ∀ b, U b ∈ Matrix.unitaryGroup S ℂ) :
    Matrix.blockDiagonal U ∈ Matrix.unitaryGroup (S × B) ℂ := by
  constructor
  · change (Matrix.blockDiagonal U).conjTranspose * Matrix.blockDiagonal U = 1
    rw [Matrix.blockDiagonal_conjTranspose, ← Matrix.blockDiagonal_mul]
    have h : (fun b => (U b).conjTranspose * U b) = 1 := by
      funext b
      exact (hU b).1
    rw [h, Matrix.blockDiagonal_one]
  · change Matrix.blockDiagonal U * (Matrix.blockDiagonal U).conjTranspose = 1
    rw [Matrix.blockDiagonal_conjTranspose, ← Matrix.blockDiagonal_mul]
    have h : (fun b => U b * (U b).conjTranspose) = 1 := by
      funext b
      exact (hU b).2
    rw [h, Matrix.blockDiagonal_one]

/-- Fixed Fourier source for four states per block, CM64. Native flattened
index `4 * block + local` merely enumerates this product type. -/
noncomputable def blockFourier (B : Type*) [DecidableEq B] :
    Matrix (Fin 4 × B) (Fin 4 × B) ℂ := Matrix.blockDiagonal (fun _ : B => fourier4)

omit [Fintype S] [DecidableEq S] in
/-- The complete block source has unit singular values for every finite
block count, CM64; no empirical query-feature flatness is assumed. -/
theorem blockFourier_unitary : blockFourier B ∈ Matrix.unitaryGroup (Fin 4 × B) ℂ := by
  exact block_unitary (fun _ : B => fourier4) (fun _ => fourier4_unitary)

omit [Fintype S] [DecidableEq S] in
/-- CM64's complete column bound for the exact block Fourier source is
eight times the mass cap times the block count. All program blocks are included. -/
theorem block_head_budget {J X C : Type*} [Fintype J] [Fintype C] [DecidableEq C]
    (program : J → X → Fin 4 × B) (order : List J) (samples : List (X × C)) {δ R : ℝ}
    {q : LatentPrograms.Parameters J (Fin 4 × B) C}
    (hq : q ∈ domain program order samples δ R) :
    LatentPrograms.head (blockFourier B) 1 q ∈ headBudget (8 * R * Fintype.card B) := by
  have h := head_budget (blockFourier B) blockFourier_unitary program order samples hq
  have he : (2 : ℝ) * R * Fintype.card (Fin 4 × B) = 8 * R * Fintype.card B := by
    simp only [Fintype.card_prod, Fintype.card_fin, Nat.cast_mul, Nat.cast_ofNat]
    ring
  rwa [he] at h

example := block_unitary (fun _ : Bool => fourier4) (fun _ => fourier4_unitary)

-- One four-state block still has a nonconstant, noninjective two-class decoder.
example : blockFourier Unit ∈ Matrix.unitaryGroup (Fin 4 × Unit) ℂ := blockFourier_unitary

example : ∃ q : LatentPrograms.Parameters Bool (Fin 4 × Unit) Bool,
    q ∈ domain (fun j x => (LatentPrograms.witnessProgram j x, ()))
      [false, true] LatentPrograms.witnessSamples 1 8 ∧
    LatentPrograms.head (blockFourier Unit) 1 q ∈ headBudget 64 := by
  let q : LatentPrograms.Parameters Bool (Fin 4 × Unit) Bool :=
    ((LatentPrograms.witnessPoint true).1,
      fun j s c => (LatentPrograms.witnessPoint true).2 j s.1 c)
  let program := fun j x => (LatentPrograms.witnessProgram j x, ())
  have h := LatentPrograms.witnessPoint_mem true
  have hn : q ∈ LatentPrograms.domain program [false, true] LatentPrograms.witnessSamples :=
    ⟨h.1, (fun j s c => h.2.1 j s.1 c), (fun j s => h.2.2.1 j s.1),
      h.2.2.2.1, h.2.2.2.2⟩
  have hq : q ∈ domain program [false, true] LatentPrograms.witnessSamples 1 8 := by
    simpa only [one_smul] using scaled_mem program [false, true] LatentPrograms.witnessSamples
      hn (by norm_num : (0 : ℝ) ≤ 1) (le_refl 1) (by norm_num : (1 : ℝ) ≤ 8)
  refine ⟨q, hq, ?_⟩
  have hb := block_head_budget program [false, true] LatentPrograms.witnessSamples hq
  norm_num at hb
  exact hb

end Transformer.Grokking.FlatSpectrum.ConvexDepth.BoundedLatent
