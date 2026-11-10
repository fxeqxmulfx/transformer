module

public import Mathlib.Analysis.SpecificLimits.Basic
public import Transformer.Grokking.FlatSpectrum.Training.Certificate
public import Transformer.Grokking.FlatSpectrum.Training.Continuity

/-!
# The squared loss on the seen cells

Source: `review/flat_spectrum.md`, *Training reaches the optimal fit* (added), item WD7, its
loss. `mask Ω` keeps the entries on `Ω` and zeroes the others, the projection `P_Ω`. The
squared loss on the seen cells is `fitLoss Ω K X = ½ ‖P_Ω (X − K)‖² = ½ Σ_Ω |X_ij − K_ij|²`. It
is nonnegative, continuous, vanishes exactly at the matrices equal to `K` on `Ω`
(`fitLoss_eq_zero_iff`), and satisfies the first-order inequality with the continuous gradient
`P_Ω (X − K)` (`fitLoss_first_order`), as
`L Y − L X − ⟨P_Ω (X − K), Y − X⟩ = ½ ‖P_Ω (Y − X)‖²`.

For a `1 × 1` kernel `K = 1` and `0 ≤ s ≤ 1` the factorization `(1 − s) · (1 − s)` is a least
one for the weight decay `λ = s (2 − s)` (`isDecayedMin_one_sub`): weight decay shrinks the
product from `1` to `1 − λ`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix Filter Topology
open scoped ComplexOrder

variable {m n : Type*} [Fintype m] [Fintype n]

open Classical in
/-- The entries of `X` on `Ω`, zero elsewhere: the projection `P_Ω`. -/
noncomputable def mask (Ω : Set (m × n)) (X : Matrix m n ℂ) : Matrix m n ℂ :=
  of fun i j => if (i, j) ∈ Ω then X i j else 0

/-- The squared loss `½ Σ_Ω |X_ij − K_ij|²` on the seen cells `Ω`, item WD7. -/
noncomputable def fitLoss (Ω : Set (m × n)) (K X : Matrix m n ℂ) : ℝ :=
  frobeniusSq (mask Ω (X - K)) / 2

omit [Fintype m] [Fintype n] in
lemma mask_sub (Ω : Set (m × n)) (X Y : Matrix m n ℂ) :
    mask Ω (X - Y) = mask Ω X - mask Ω Y := by
  ext i j
  simp only [mask, of_apply, Matrix.sub_apply]
  split_ifs <;> simp

/-- `⟨P_Ω A, B⟩ = ⟨P_Ω A, P_Ω B⟩`. -/
lemma pairing_mask_left (Ω : Set (m × n)) (A B : Matrix m n ℂ) :
    pairing (mask Ω A) B = pairing (mask Ω A) (mask Ω B) := by
  simp only [pairing, trace_conjTranspose_mul]
  congr 1
  refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
  simp only [mask, of_apply]
  split_ifs <;> simp

lemma frobeniusSq_mask_le (Ω : Set (m × n)) (A : Matrix m n ℂ) :
    frobeniusSq (mask Ω A) ≤ frobeniusSq A := by
  unfold frobeniusSq
  refine Finset.sum_le_sum fun i _ => Finset.sum_le_sum fun j _ => ?_
  simp only [mask, of_apply]
  split_ifs
  · exact le_rfl
  · rw [norm_zero, zero_pow two_ne_zero]
    exact sq_nonneg _

omit [Fintype m] [Fintype n] in
lemma continuous_mask (Ω : Set (m × n)) : Continuous (mask Ω : Matrix m n ℂ → Matrix m n ℂ) := by
  refine continuous_matrix fun i j => ?_
  simp only [mask, of_apply]
  split_ifs
  · exact continuous_id.matrix_elem i j
  · exact continuous_const

omit [Fintype m] [Fintype n] in
/-- The gradient `P_Ω (X − K)` of the squared loss is continuous. -/
lemma continuous_mask_sub (Ω : Set (m × n)) (K : Matrix m n ℂ) :
    Continuous fun X : Matrix m n ℂ => mask Ω (X - K) :=
  (continuous_mask Ω).comp (continuous_id.sub continuous_const)

lemma continuous_fitLoss (Ω : Set (m × n)) (K : Matrix m n ℂ) : Continuous (fitLoss Ω K) :=
  (continuous_frobeniusSq.comp (continuous_mask_sub Ω K)).div_const 2

lemma fitLoss_nonneg (Ω : Set (m × n)) (K X : Matrix m n ℂ) : 0 ≤ fitLoss Ω K X :=
  div_nonneg (frobeniusSq_nonneg _) zero_le_two

/-- **The squared loss satisfies the first-order inequality** with the gradient `P_Ω (X − K)`
(`review/flat_spectrum.md`, item WD7): the difference of the two sides is `½ ‖P_Ω (Y − X)‖²`. -/
theorem fitLoss_first_order (Ω : Set (m × n)) (K X Y : Matrix m n ℂ) :
    fitLoss Ω K X + pairing (mask Ω (X - K)) (Y - X) ≤ fitLoss Ω K Y := by
  have e : mask Ω (Y - X) = mask Ω (Y - K) - mask Ω (X - K) := by
    rw [← mask_sub, sub_sub_sub_cancel_right]
  rw [pairing_mask_left, e, pairing_sub_right, pairing_self]
  unfold fitLoss
  have h := frobeniusSq_sub (mask Ω (X - K)) (mask Ω (Y - K))
  have h0 := frobeniusSq_nonneg (mask Ω (X - K) - mask Ω (Y - K))
  linarith

/-- `⟨P_Ω (X − K), X⟩ ≥ −¼ ‖K‖²`: the first-order term of the squared loss is bounded below. -/
lemma neg_le_pairing_mask_sub (Ω : Set (m × n)) (K X : Matrix m n ℂ) :
    -(frobeniusSq K / 4) ≤ pairing (mask Ω (X - K)) X := by
  have h1 : pairing (mask Ω (X - K)) (X - K + K) =
      frobeniusSq (mask Ω (X - K)) + pairing (mask Ω (X - K)) K := by
    rw [pairing_add_right, pairing_mask_left, pairing_self]
  rw [sub_add_cancel] at h1
  have h2 := frobeniusSq_add (mask Ω (X - K)) ((1 / 2 : ℝ) • K)
  rw [pairing_smul_right, frobeniusSq_smul] at h2
  have h3 := frobeniusSq_nonneg (mask Ω (X - K) + (1 / 2 : ℝ) • K)
  linarith

/-- **The squared loss vanishes exactly at the fits** (`review/flat_spectrum.md`, item WD7):
at the matrices equal to `K` on `Ω`. -/
theorem fitLoss_eq_zero_iff {Ω : Set (m × n)} {K X : Matrix m n ℂ} :
    fitLoss Ω K X = 0 ↔ X ∈ {X' : Matrix m n ℂ | ∀ ij ∈ Ω, X' ij.1 ij.2 = K ij.1 ij.2} := by
  rw [fitLoss, div_eq_zero_iff, or_iff_left two_ne_zero]
  constructor
  · rintro h ⟨i, j⟩ hij
    have h0 := congrFun (congrFun (eq_zero_of_frobeniusSq_eq_zero h) i) j
    simp only [mask, of_apply, Matrix.zero_apply, Matrix.sub_apply, hij, ↓reduceIte] at h0
    exact sub_eq_zero.mp h0
  · intro h
    have h0 : mask Ω (X - K) = 0 := by
      ext i j
      simp only [mask, of_apply, Matrix.zero_apply, Matrix.sub_apply]
      split_ifs with hij
      · exact sub_eq_zero.mpr (h (i, j) hij)
      · rfl
    rw [h0]
    simp [frobeniusSq]

example := (fitLoss_eq_zero_iff (Ω := Set.univ) (K := (1 : Matrix (Fin 1) (Fin 1) ℂ))
  (X := 1)).mpr fun _ _ => rfl

/-- **Weight decay shrinks a `1 × 1` kernel** (`review/flat_spectrum.md`, item WD7, its
example): for `K = 1` and `0 ≤ s ≤ 1`, the factorization `(1 − s) · (1 − s)` is a least one for
the weight decay `λ = s (2 − s)`, with product `(1 − s)² = 1 − λ`. -/
theorem isDecayedMin_one_sub {s : ℝ} (h0 : 0 ≤ s) (h1 : s ≤ 1) :
    IsDecayedMin (fitLoss Set.univ (1 : Matrix (Fin 1) (Fin 1) ℂ)) (s * (2 - s))
      ((1 - s) • (1 : Matrix (Fin 1) (Fin 1) ℂ)) ((1 - s) • (1 : Matrix (Fin 1) (Fin 1) ℂ)) := by
  have hG : mask Set.univ (((1 - s) • (1 : Matrix (Fin 1) (Fin 1) ℂ)) * ((1 - s) • 1)ᴴ - 1) =
      (-(s * (2 - s))) • (1 : Matrix (Fin 1) (Fin 1) ℂ) := by
    rw [conjTranspose_smul, conjTranspose_one, star_trivial, Matrix.smul_mul, Matrix.mul_smul,
      Matrix.one_mul]
    ext i j
    simp only [mask, of_apply, Set.mem_univ, ↓reduceIte]
    rw [show (1 - s) • (1 - s) • (1 : Matrix (Fin 1) (Fin 1) ℂ) - 1 = (-(s * (2 - s))) • 1 by
      module]
  refine isDecayedMin_of_posSemidef (G := fun X => mask Set.univ (X - 1))
    (fitLoss_first_order Set.univ 1) ?_ ?_
  · rw [hG, lift_mul_stack, ← stack_zero]
    congr 1
    · rw [Matrix.smul_mul, Matrix.one_mul]
      module
    · rw [conjTranspose_smul, conjTranspose_one, star_trivial, Matrix.smul_mul, Matrix.one_mul]
      module
  · rw [hG]
    refine lift_posSemidef_of_frobeniusSq_le (mul_nonneg h0 (by linarith)) ?_
    rw [frobeniusSq_smul]
    simp [frobeniusSq]

example := isDecayedMin_one_sub (s := 1 / 2) (by norm_num) (by norm_num)

/-! The shrinking family: `s_j = 1/(j + 1)`, weight decays `λ_j = s_j (2 − s_j) → 0⁺`, and the
trained fits `(1 − s_j) · (1 − s_j) → 1 · 1` of the `1 × 1` kernel `1`. -/

lemma shrink_pos (j : ℕ) : 0 < 1 / ((j : ℝ) + 1) * (2 - 1 / ((j : ℝ) + 1)) := by
  have h1 : 1 / ((j : ℝ) + 1) ≤ 1 := by
    rw [div_le_one (by positivity)]
    linarith [Nat.cast_nonneg (α := ℝ) j]
  exact mul_pos (by positivity) (by linarith)

lemma tendsto_shrink :
    Tendsto (fun j : ℕ => 1 / ((j : ℝ) + 1) * (2 - 1 / ((j : ℝ) + 1))) atTop (𝓝 0) := by
  have h := tendsto_one_div_add_atTop_nhds_zero_nat (𝕜 := ℝ)
  simpa using h.mul (h.const_sub 2)

lemma tendsto_shrink_factor : Tendsto
    (fun j : ℕ => (1 - 1 / ((j : ℝ) + 1)) • (1 : Matrix (Fin 1) (Fin 1) ℂ)) atTop (𝓝 1) := by
  have h := ((tendsto_one_div_add_atTop_nhds_zero_nat (𝕜 := ℝ)).const_sub 1).smul_const
    (1 : Matrix (Fin 1) (Fin 1) ℂ)
  simpa using h

lemma isDecayedMin_shrink (j : ℕ) :
    IsDecayedMin (fitLoss Set.univ (1 : Matrix (Fin 1) (Fin 1) ℂ))
      (1 / ((j : ℝ) + 1) * (2 - 1 / ((j : ℝ) + 1)))
      ((1 - 1 / ((j : ℝ) + 1)) • (1 : Matrix (Fin 1) (Fin 1) ℂ))
      ((1 - 1 / ((j : ℝ) + 1)) • (1 : Matrix (Fin 1) (Fin 1) ℂ)) := by
  refine isDecayedMin_one_sub (by positivity) ?_
  rw [div_le_one (by positivity)]
  linarith [Nat.cast_nonneg (α := ℝ) j]

end Transformer.Grokking.FlatSpectrum.Training
