module

public import Transformer.Grokking.Convex.CoupledMoments.Schatten.Basic
public import Transformer.Grokking.FlatSpectrum.Roots

/-!
# A class shift in every cell is a symmetry of the coupled-moment model

Source: `review/convex/moments.md`, CM121. Let the classes form a finite additive group `C` and
move the classes of the cell `(i, j)` by its own offset, `c ↦ c + u i + v j` (`shift`).
Probability rows go to probability rows (`rows_shift_iff`). For a code that is an additive
character `χ`, the moment only gains unit phases on its rows and columns,
`M_χ(shift P) = diag(χ ∘ u) · M_χ(P) · diag(χ ∘ v)` (`moment_shift`), so its operator and
nuclear norms stay (`norm_moment_shift`, `nuclearNorm_moment_shift`) and every budget set is
mapped onto itself (`budgetFeasible_shift_iff`, `shift_image_budgetFeasible`). Observed CE with
the labels shifted in the same way does not change (`loss_shift`), nor does an exact fit
(`shift_fit_iff`). The shift is linear and fixes the uniform table (`shift_uniform`), so it
commutes with the common mixture of CM4.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.CoupledMoments.Schatten

open Matrix Finset FlatSpectrum ConvexRecall
open Landscape.Neural (crossEntropy)
open scoped Matrix.Norms.L2Operator

variable {m n C S : Type*} [AddCommGroup C]

/-- The class shift `c ↦ c + u i + v j` of the cell `(i, j)` (CM121). -/
def shift (u : m → C) (v : n → C) (P : Table m n C) : Table m n C :=
  fun i j c => P i j (c - u i - v j)

/-- The shift by `-u, -v` undoes the shift by `u, v`. -/
theorem shift_neg_shift (u : m → C) (v : n → C) (P : Table m n C) :
    shift (-u) (-v) (shift u v P) = P := by
  funext i j c
  simp only [shift, Pi.neg_apply, sub_neg_eq_add]
  congr 1
  abel

/-- The shift is linear in the table. -/
theorem shift_add_smul (u : m → C) (v : n → C) (a b : ℝ) (P Q : Table m n C) :
    shift u v (a • P + b • Q) = a • shift u v P + b • shift u v Q := rfl

/-- An exact fit of the labels `y` shifts to an exact fit of `y + u i + v j`. -/
theorem shift_fit_iff [DecidableEq C] (cell : S → m × n) (labels : S → C) (u : m → C)
    (v : n → C) (P : Table m n C) :
    (∀ s, shift u v P (cell s).1 (cell s).2 =
        targetLogits (labels s + u (cell s).1 + v (cell s).2)) ↔
      ∀ s, P (cell s).1 (cell s).2 = targetLogits (labels s) := by
  have key (s : S) (c : C) : (c - (u (cell s).1 + v (cell s).2) =
      labels s) ↔ c = labels s + u (cell s).1 + v (cell s).2 := by
    rw [sub_eq_iff_eq_add, add_assoc]
  refine forall_congr' fun s => ⟨fun h => funext fun c => ?_, fun h => funext fun c => ?_⟩
  · have := congrFun h (c + (u (cell s).1 + v (cell s).2))
    simpa [shift, targetLogits, sub_sub, add_assoc] using this
  · have := congrFun h (c - (u (cell s).1 + v (cell s).2))
    simpa [shift, targetLogits, sub_sub, key] using this

variable [Fintype C]

/-- The shift fixes the uniform table. -/
theorem shift_uniform (u : m → C) (v : n → C) : shift u v (uniform : Table m n C) = uniform :=
  rfl

/-- Reindexing the classes of one cell by its offset keeps every sum over the classes. -/
lemma sum_shift {M : Type*} [AddCommMonoid M] (f : C → M) (s : C) :
    ∑ c, f (c - s) = ∑ c, f c :=
  Fintype.sum_equiv (Equiv.subRight s) _ _ fun _ => rfl

lemma rows_shift {u : m → C} {v : n → C} {P : Table m n C} (hP : Rows P) :
    Rows (shift u v P) := by
  refine ⟨fun i j c => hP.1 i j _, fun i j => ?_⟩
  simp only [shift, sub_sub]
  rw [sum_shift (P i j), hP.2 i j]

/-- **The shift maps the product of simplexes onto itself.** -/
theorem rows_shift_iff (u : m → C) (v : n → C) (P : Table m n C) :
    Rows (shift u v P) ↔ Rows P :=
  ⟨fun h => shift_neg_shift u v P ▸ rows_shift h, rows_shift⟩

example := rows_shift (u := fun _ : Fin 2 => (1 : ZMod 3)) (v := fun _ : Fin 2 => 2)
  (rows_uniform (C := ZMod 3))

/-- **Observed CE with shifted labels**: the shifted table on the labels
`y + u i + v j` has the loss of the table on `y`. -/
theorem loss_shift [Fintype S] (cell : S → m × n) (labels : S → C) (u : m → C) (v : n → C)
    (P : Table m n C) :
    loss cell (fun s => labels s + u (cell s).1 + v (cell s).2) (shift u v P) =
      loss cell labels P := by
  refine sum_congr rfl fun s _ => ?_
  simp only [crossEntropy, shift, sub_sub]
  rw [sum_shift (fun c => Real.exp (P (cell s).1 (cell s).2 c)), add_assoc, add_sub_cancel_right]

/-- **A character moment gains unit phases**:
`M_χ(shift P) = diag(χ ∘ u) · M_χ(P) · diag(χ ∘ v)`. -/
theorem moment_shift [Fintype m] [DecidableEq m] [Fintype n] [DecidableEq n]
    (χ : AddChar C ℂ) (u : m → C) (v : n → C) (P : Table m n C) :
    moment χ (shift u v P) =
      diagonal (fun i => χ (u i)) * moment χ P * diagonal (fun j => χ (v j)) := by
  ext i j
  rw [mul_diagonal, diagonal_mul]
  simp only [moment, shift, sub_sub]
  rw [Fintype.sum_equiv (Equiv.subRight (u i + v j))
    (fun x => (P i j (x - (u i + v j)) : ℂ) * χ x)
    (fun c => (P i j c : ℂ) * χ (c + (u i + v j))) fun x => by simp]
  simp only [AddChar.map_add_eq_mul, mul_sum, sum_mul]
  exact sum_congr rfl fun _ _ => by ring

variable [Fintype m] [DecidableEq m] [Fintype n] [DecidableEq n]

lemma norm_diagonal_mul_mul_diagonal_le {a : m → ℂ} {b : n → ℂ} (ha : ∀ i, ‖a i‖ ≤ 1)
    (hb : ∀ j, ‖b j‖ ≤ 1) (M : Matrix m n ℂ) : ‖diagonal a * M * diagonal b‖ ≤ ‖M‖ := by
  have ha' : ‖a‖ ≤ 1 := (pi_norm_le_iff_of_nonneg zero_le_one).2 ha
  have hb' : ‖b‖ ≤ 1 := (pi_norm_le_iff_of_nonneg zero_le_one).2 hb
  calc ‖diagonal a * M * diagonal b‖ ≤ ‖diagonal a‖ * ‖M‖ * ‖diagonal b‖ :=
        (l2_opNorm_mul _ _).trans (mul_le_mul_of_nonneg_right (l2_opNorm_mul _ _) (norm_nonneg _))
    _ ≤ 1 * ‖M‖ * 1 := by
        rw [l2_opNorm_diagonal, l2_opNorm_diagonal]
        gcongr
    _ = ‖M‖ := by ring

example := norm_diagonal_mul_mul_diagonal_le (a := fun _ : Fin 2 => Complex.I)
  (b := fun _ : Fin 3 => -1) (by simp) (by simp) 0

/-- `diag(star a) · diag(a) = 1` when every `|a i| = 1`. -/
lemma diagonal_star_mul_diagonal {k : Type*} [Fintype k] [DecidableEq k] {a : k → ℂ}
    (ha : ∀ i, ‖a i‖ = 1) : diagonal (star a) * diagonal a = 1 := by
  rw [diagonal_mul_diagonal, ← diagonal_one]
  congr 1
  funext i
  rw [Pi.star_apply, Complex.star_def, Complex.conj_mul', ha i]
  norm_num

example := diagonal_star_mul_diagonal (a := fun _ : Fin 2 => Complex.I) (by simp)

/-- Unit phases on the rows and columns keep the operator norm. -/
theorem norm_diagonal_mul_mul_diagonal {a : m → ℂ} {b : n → ℂ} (ha : ∀ i, ‖a i‖ = 1)
    (hb : ∀ j, ‖b j‖ = 1) (M : Matrix m n ℂ) : ‖diagonal a * M * diagonal b‖ = ‖M‖ := by
  refine le_antisymm (norm_diagonal_mul_mul_diagonal_le (fun i => (ha i).le)
    (fun j => (hb j).le) M) ?_
  have hM : diagonal (star a) * (diagonal a * M * diagonal b) * diagonal (star b) = M := by
    have hb' : diagonal b * diagonal (star b) = 1 := by
      rw [diagonal_mul_diagonal, ← diagonal_one]
      congr 1
      funext j
      rw [Pi.star_apply, Complex.star_def, mul_comm, Complex.conj_mul', hb j]
      norm_num
    rw [← Matrix.mul_assoc, ← Matrix.mul_assoc, diagonal_star_mul_diagonal ha, Matrix.one_mul,
      Matrix.mul_assoc, hb', Matrix.mul_one]
  conv_lhs => rw [← hM]
  exact norm_diagonal_mul_mul_diagonal_le (fun i => by simp [ha i]) (fun j => by simp [hb j]) _

example := norm_diagonal_mul_mul_diagonal (a := fun _ : Fin 2 => Complex.I)
  (b := fun _ : Fin 3 => -1) (by simp) (by simp) 0

/-- **The shift keeps the operator norm of every character moment.** -/
theorem norm_moment_shift (χ : AddChar C ℂ) (u : m → C) (v : n → C) (P : Table m n C) :
    ‖moment χ (shift u v P)‖ = ‖moment χ P‖ := by
  rw [moment_shift, norm_diagonal_mul_mul_diagonal (fun _ => AddChar.norm_apply _ _)
    (fun _ => AddChar.norm_apply _ _)]

/-- **The shift keeps the nuclear norm of every character moment.** -/
theorem nuclearNorm_moment_shift (χ : AddChar C ℂ) (u : m → C) (v : n → C) (P : Table m n C) :
    nuclearNorm (moment χ (shift u v P)) = nuclearNorm (moment χ P) := by
  have hu := diagonal_star_mul_diagonal (a := fun i => χ (u i)) fun _ => AddChar.norm_apply _ _
  have hv := diagonal_star_mul_diagonal (a := star fun j => χ (v j)) fun _ => by
    rw [Pi.star_apply, norm_star, AddChar.norm_apply]
  rw [← diagonal_conjTranspose] at hu hv
  have h := nuclearNorm_isometry hu hv (moment χ P)
  rwa [diagonal_conjTranspose, star_star, ← moment_shift] at h

/-- **The shift maps every budget set of character moments onto itself.** -/
theorem budgetFeasible_shift_iff {F : Type*} (χ : F → AddChar C ℂ) (R B : F → ℝ) (u : m → C)
    (v : n → C) (P : Table m n C) :
    BudgetFeasible (fun f => ⇑(χ f)) R B (shift u v P) ↔
      BudgetFeasible (fun f => ⇑(χ f)) R B P := by
  simp only [BudgetFeasible, Feasible, rows_shift_iff, norm_moment_shift,
    nuclearNorm_moment_shift]

theorem shift_image_budgetFeasible {F : Type*} (χ : F → AddChar C ℂ) (R B : F → ℝ) (u : m → C)
    (v : n → C) :
    shift u v '' {P | BudgetFeasible (fun f => ⇑(χ f)) R B P} =
      {P | BudgetFeasible (fun f => ⇑(χ f)) R B P} := by
  ext Q
  refine ⟨fun ⟨P, hP, hQ⟩ => hQ ▸ (budgetFeasible_shift_iff χ R B u v P).2 hP, fun hQ => ?_⟩
  refine ⟨shift (-u) (-v) Q, (budgetFeasible_shift_iff χ R B (-u) (-v) Q).2 hQ, ?_⟩
  simpa using shift_neg_shift (-u) (-v) Q

/-- The Fourier codes of `convex_stress_schatten` are `c ↦ ψ(f c)` for one character `ψ`. -/
example {K : Type*} [CommRing K] [Fintype K] [DecidableEq K] (ψ : AddChar K ℂ) (R B : K → ℝ)
    (u v : K → K) (P : Table K K K) :
    BudgetFeasible (fun f => ⇑(ψ.mulShift f)) R B (shift u v P) ↔
      BudgetFeasible (fun f => ⇑(ψ.mulShift f)) R B P :=
  budgetFeasible_shift_iff ψ.mulShift R B u v P

end Transformer.Grokking.Convex.CoupledMoments.Schatten
