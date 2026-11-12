module

public import Transformer.Grokking.FlatSpectrum.SpectralBall.Recovery

/-!
# Tight spectral bounds do not always identify unseen signs

Source: `review/flat_spectrum.md`, SB3. Two `4 × 4` flat sign matrices agree on twelve
entries, with every row and column observed, and disagree at `(0,0)`. Both minimize ordinary
empirical binary CE on the same convex disk-and-spectral-ball domain. This is a separate
counterexample to extending SB2 to arbitrary observation patterns, not a proof about the
particular 97-class stress data or its numerical optimizer.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.SpectralBall

open Matrix Finset Training
open scoped Matrix.Norms.L2Operator

/-- The first flat sign table of SB3. -/
def signTable : Matrix (Fin 4) (Fin 4) ℂ :=
  !![1, 1, 1, 1; 1, -1, 1, -1; 1, 1, -1, -1; 1, -1, -1, 1]

/-- Negate and exchange its first two rows; only four entries change (SB3). -/
def signAlternative : Matrix (Fin 4) (Fin 4) ℂ :=
  !![-1, 1, -1, 1; -1, -1, -1, -1; 1, 1, -1, -1; 1, -1, -1, 1]

/-- All but the block of rows `{0,1}` and columns `{0,2}` are observed (SB3). -/
def twelveCells : Fin 12 → Fin 4 × Fin 4 :=
  ![(0,1), (0,3), (1,1), (1,3), (2,0), (2,1), (2,2), (2,3), (3,0), (3,1), (3,2), (3,3)]

/-- Labels on the twelve observed cells (SB3). -/
def twelveLabels : Fin 12 → Bool :=
  ![true, true, false, false, true, true, false, false, true, false, false, true]

/-- The observations are distinct, and every row and column is represented (SB3). -/
theorem twelveCells_coverage : Function.Injective twelveCells ∧
    (∀ i, ∃ s, (twelveCells s).1 = i) ∧ (∀ j, ∃ s, (twelveCells s).2 = j) := by
  decide

/-- Both tables have exactly the flat squared singular value four (SB3). -/
theorem sign_tables_gram :
    signTableᴴ * signTable = (4 : ℂ) • (1 : Matrix (Fin 4) (Fin 4) ℂ) ∧
    signAlternativeᴴ * signAlternative = (4 : ℂ) • (1 : Matrix (Fin 4) (Fin 4) ℂ) := by
  constructor <;> ext i j <;> fin_cases i <;> fin_cases j <;>
    norm_num [signTable, signAlternative, Matrix.mul_apply, Fin.sum_univ_succ]

/-- Both complex tables lie in the same tight convex domain (SB3). -/
theorem sign_tables_feasible : Feasible 2 signTable ∧ Feasible 2 signAlternative := by
  have hn {X : Matrix (Fin 4) (Fin 4) ℂ}
      (h : Xᴴ * X = (4 : ℂ) • (1 : Matrix (Fin 4) (Fin 4) ℂ)) : ‖X‖ ≤ 2 := by
    have he := l2_opNorm_conjTranspose_mul_self X
    rw [h, norm_smul, norm_one] at he
    norm_num at he
    nlinarith [norm_nonneg X]
  constructor
  · refine ⟨?_, hn sign_tables_gram.1⟩
    intro i j
    fin_cases i <;> fin_cases j <;> norm_num [signTable]
  · refine ⟨?_, hn sign_tables_gram.2⟩
    intro i j
    fin_cases i <;> fin_cases j <;> norm_num [signAlternative]

/-- Either table exactly shows all observed labels (SB3). -/
theorem sign_tables_observed (s : Fin 12) :
    signTable (twelveCells s).1 (twelveCells s).2 = signCode (twelveLabels s) ∧
    signAlternative (twelveCells s).1 (twelveCells s).2 = signCode (twelveLabels s) := by
  fin_cases s <;> norm_num [twelveCells, twelveLabels, signTable, signAlternative, signCode]

variable {m n S : Type*} [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n] [Fintype S]

/-- Any sign table fitting the observed labels minimizes each CE term on the disks (SB3). -/
theorem loss_le_of_unit_signs (cell : S → m × n) (labels : S → Bool) {K X : Matrix m n ℂ}
    (hK : ∀ s, K (cell s).1 (cell s).2 = signCode (labels s))
    (hX : ∀ i j, ‖X i j‖ ≤ 1) :
    ceLoss (codeProbe signCode cell) labels K ≤ ceLoss (codeProbe signCode cell) labels X := by
  rw [ceLoss_signCode_eq, ceLoss_signCode_eq]
  apply Finset.sum_le_sum
  intro s _
  rw [hK s]
  have hs : (signCode (labels s)).re * (signCode (labels s)).re = 1 := by
    cases labels s <;> simp [signCode]
  rw [hs]
  apply Real.log_le_log (by positivity)
  apply add_le_add_right
  apply Real.exp_le_exp.mpr
  linarith [sign_reading_le_one (hX (cell s).1 (cell s).2) (labels s)]

example := loss_le_of_unit_signs twelveCells twelveLabels
  (fun s => (sign_tables_observed s).1) sign_tables_feasible.1.1

/-- Two global CE minima, both flat, disagree on an unseen sign despite 75% observations
covering every row and column (SB3). The loss reads only those twelve cells. -/
theorem ambiguous_flat_minima :
    IsFlat signTable 4 ∧ IsFlat signAlternative 4 ∧
    IsMinOn (ceLoss (codeProbe signCode twelveCells) twelveLabels)
      {X | Feasible 2 X} signTable ∧
    IsMinOn (ceLoss (codeProbe signCode twelveCells) twelveLabels)
      {X | Feasible 2 X} signAlternative ∧ signTable 0 0 ≠ signAlternative 0 0 := by
  refine ⟨Or.inl sign_tables_gram.1, Or.inl sign_tables_gram.2, ?_, ?_, ?_⟩
  · intro X hX
    exact loss_le_of_unit_signs twelveCells twelveLabels
      (fun s => (sign_tables_observed s).1) hX.1
  · intro X hX
    exact loss_le_of_unit_signs twelveCells twelveLabels
      (fun s => (sign_tables_observed s).2) hX.1
  · norm_num [signTable, signAlternative]

end Transformer.Grokking.FlatSpectrum.SpectralBall
