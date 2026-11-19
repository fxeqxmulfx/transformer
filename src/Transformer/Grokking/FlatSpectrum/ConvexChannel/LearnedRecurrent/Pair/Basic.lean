module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Domain
public import Mathlib.Analysis.Normed.Operator.Prod

/-!
# Complete learned two-state complex operators

Source: `review/convex/moments.md`, CM97. All four transition entries, both
readout coefficients and both initial coordinates are parameters. The product
maximum norm and explicit row-sum constraints imply CM95's operator bounds.
The basis identities are complete coordinate representations, not task features.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Pair

/-- Two complex state coordinates, with the product maximum norm, CM97. -/
abbrev State := ℂ × ℂ

/-- A full two-by-two complex matrix stored as two rows, CM97. -/
abbrev Matrix := State × State

/-- Every class readout has both learned complex coefficients, CM97. -/
noncomputable def row (u : State) : State →L[ℂ] ℂ :=
  u.1 • ContinuousLinearMap.fst ℂ ℂ ℂ + u.2 • ContinuousLinearMap.snd ℂ ℂ ℂ

/-- Both learned rows act on both state coordinates, CM97. -/
noncomputable def operator (m : Matrix) : State →L[ℂ] State :=
  (row m.1).prod (row m.2)

/-- The actual class readout is the full complex row product, CM97. -/
theorem row_apply (u z : State) : row u z = u.1 * z.1 + u.2 * z.2 := by
  simp [row]

/-- The actual transition executes all four matrix entries, CM97. -/
theorem operator_apply (m : Matrix) (z : State) :
    operator m z = (m.1.1 * z.1 + m.1.2 * z.2, m.2.1 * z.1 + m.2.2 * z.2) := by
  simp [operator, row_apply]

/-- Every two-state complex linear map is determined by both columns,
CM97. This retains every direction and imaginary entry. -/
theorem basis_expansion {F : Type*} [NormedAddCommGroup F] [NormedSpace ℂ F]
    (L : State →L[ℂ] F) (z : State) :
    L z = z.1 • L (1, 0) + z.2 • L (0, 1) := by
  calc
    L z = L (z.1 • (1, 0) + z.2 • (0, 1)) := by congr 1; ext <;> simp
    _ = z.1 • L (1, 0) + z.2 • L (0, 1) := by rw [map_add, map_smul, map_smul]

/-- Every complex linear class readout has the declared full row
representation, CM97. -/
theorem row_representation (L : State →L[ℂ] ℂ) : row (L (1, 0), L (0, 1)) = L := by
  apply ContinuousLinearMap.ext
  intro z
  rw [row_apply]
  simpa [smul_eq_mul, mul_comm] using (basis_expansion L z).symm

/-- Every complex linear state transition has the full matrix
representation, CM97; no diagonal or low-rank restriction is imposed. -/
theorem operator_representation (L : State →L[ℂ] State) :
    operator (((L (1, 0)).1, (L (0, 1)).1), ((L (1, 0)).2, (L (0, 1)).2)) = L := by
  apply ContinuousLinearMap.ext
  intro z
  rw [operator_apply]
  apply Prod.ext
  · simpa [smul_eq_mul, mul_comm] using congrArg Prod.fst (basis_expansion L z).symm
  · simpa [smul_eq_mul, mul_comm] using congrArg Prod.snd (basis_expansion L z).symm

/-- All numerical paths in the two-state recurrence are learned, CM97. -/
noncomputable def parameters {O : Type*} (a : State) (t : O → Matrix) (u : O → State) :
    Parameters O State := ⟨a, fun o => operator (t o), fun c => row (u c)⟩

/-- Full matrix and row coordinates reconstruct every two-state recurrent
parameter point, CM97. This is independent of the later norm constraints. -/
theorem reconstruction {O : Type*} (f : Parameters O State) :
    parameters f.initial
      (fun o => (((f.transition o (1, 0)).1, (f.transition o (0, 1)).1),
        ((f.transition o (1, 0)).2, (f.transition o (0, 1)).2)))
      (fun c => (f.readout c (1, 0), f.readout c (0, 1))) = f := by
  cases f with
  | mk a t u =>
    have ht : (fun o => operator (((t o (1, 0)).1, (t o (0, 1)).1),
        ((t o (1, 0)).2, (t o (0, 1)).2))) = t := by
      funext o
      exact operator_representation (t o)
    have hu : (fun c => row (u c (1, 0), u c (0, 1))) = u := by
      funext c
      exact row_representation (u c)
    change Parameters.mk a _ _ = Parameters.mk a t u
    rw [ht, hu]

/-- The explicit learned row-sum budget bounds its actual operator norm,
CM97, for the declared maximum state norm. -/
theorem row_norm_bound (u : State) (hu : ‖u.1‖ + ‖u.2‖ ≤ 1) : ‖row u‖ ≤ 1 := by
  apply ContinuousLinearMap.opNorm_le_bound _ (by norm_num)
  intro z
  rw [row_apply]
  calc
    ‖u.1 * z.1 + u.2 * z.2‖ ≤ ‖u.1 * z.1‖ + ‖u.2 * z.2‖ := norm_add_le _ _
    _ = ‖u.1‖ * ‖z.1‖ + ‖u.2‖ * ‖z.2‖ := by rw [norm_mul, norm_mul]
    _ ≤ ‖u.1‖ * ‖z‖ + ‖u.2‖ * ‖z‖ := add_le_add
      (mul_le_mul_of_nonneg_left (norm_fst_le z) (norm_nonneg _))
      (mul_le_mul_of_nonneg_left (norm_snd_le z) (norm_nonneg _))
    _ = (‖u.1‖ + ‖u.2‖) * ‖z‖ := by ring
    _ ≤ 1 * ‖z‖ := mul_le_mul_of_nonneg_right hu (norm_nonneg _)

/-- Both learned matrix row budgets bound the complete operator, CM97. -/
theorem operator_norm_bound (m : Matrix) (h₁ : ‖m.1.1‖ + ‖m.1.2‖ ≤ 1)
    (h₂ : ‖m.2.1‖ + ‖m.2.2‖ ≤ 1) : ‖operator m‖ ≤ 1 := by
  rw [operator, ContinuousLinearMap.opNorm_prod, Prod.norm_def]
  exact max_le (row_norm_bound m.1 h₁) (row_norm_bound m.2 h₂)

/-- Explicit numerical budgets imply all CM95 Unit hypotheses, CM97.
They do not supply unseen labels or an optimal component-search algorithm. -/
theorem unit {O : Type*} (a : State) (t : O → Matrix) (u : O → State) (ha : ‖a‖ ≤ 1)
    (ht : ∀ o, (‖(t o).1.1‖ + ‖(t o).1.2‖ ≤ 1) ∧
      (‖(t o).2.1‖ + ‖(t o).2.2‖ ≤ 1))
    (hu : ∀ c, ‖(u c).1‖ + ‖(u c).2‖ ≤ 1) : Unit (parameters a t u) :=
  ⟨ha, fun o => operator_norm_bound (t o) (ht o).1 (ht o).2,
    fun c => row_norm_bound (u c) (hu c)⟩

example : ‖row (Complex.I, 0)‖ ≤ 1 := row_norm_bound _ (by simp)

example : ‖operator ((0, Complex.I), (Complex.I, 0))‖ ≤ 1 :=
  operator_norm_bound _ (by simp) (by simp)

example : Unit (parameters (Complex.I, Complex.I)
    (fun _ : Bool => ((0, Complex.I), (Complex.I, 0))) (fun c => if c then (Complex.I, 0)
      else (0, Complex.I))) :=
  unit _ _ _ (by simp [Prod.norm_def]) (by intro o; simp)
    (by intro c; cases c <;> simp)

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Pair
