module

public import Transformer.Grokking.FlatSpectrum.Training.Scope.IdentityLimit

/-!
# The identity kernel: one step moves the sum of the factors

Source: `review/flat_spectrum.md`, *Training reaches the optimal fit* (added), item WD16,
statement 6; the item "Width" of WD11.

For the squared loss on all cells of the identity of any size, with factors `U`, `V` of any
width, put `P = U + V` and `Q = U − V`:

* a step gives `P' = (1 + η (1 − λ)) P − η (U Vᴴ V + V Uᴴ U)` (`decayStep_identity_add`);
* `4 (U Vᴴ V + V Uᴴ U) = P (Pᴴ P + Qᴴ Q) − Q (Pᴴ Q + Qᴴ P)` (`four_smul_cubic`), so
  `4 ⟨P, U Vᴴ V + V Uᴴ U⟩ ≤ ‖P‖² (‖P‖² + ‖Q‖²)` (`four_mul_pairing_add_cubic_le`): the terms
  with `Q` on the left sum to `⟨Pᴴ Q, Qᴴ P⟩ + ‖Qᴴ P‖² ≥ 0`;
* so with `a = 1 + η (1 − λ) ≥ 0`, `‖P'‖² ≥ a (a − η (‖U‖² + ‖V‖²)) ‖P‖²`
  (`le_frobeniusSq_decayStep_identity_add`);
* in a ball `‖U‖² + ‖V‖² ≤ ρ` with `η (λ + ρ/2 + κ + 2 ρ) ≤ 1` this gives
  `‖P'‖² ≥ η² ‖P‖²`, and where `‖U‖² + ‖V‖² ≤ 1 − λ` it gives `‖P'‖² ≥ ‖P‖²`
  (`frobeniusSq_add_decayStep_identity`): the sum stays nonzero, and near the origin it does
  not decrease.

For the `1 × 1` identity this is WD15, with other constants.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training.Scope

open Matrix Filter Topology

variable {m k : Type*} [Fintype m] [Fintype k] [DecidableEq m] {η lam κ ρ : ℝ}

local notation "G₁" => fun X : Matrix m m ℂ => mask Set.univ (X - 1)

/-- **The sum of the factors after a step on the identity** (`review/flat_spectrum.md`, item
WD16, statement 6): `P' = (1 + η (1 − λ)) P − η (U Vᴴ V + V Uᴴ U)`. -/
lemma decayStep_identity_add (η lam : ℝ) (U V : Matrix m k ℂ) :
    (decayStep G₁ η lam (U, V)).1 + (decayStep G₁ η lam (U, V)).2 =
      (1 + η * (1 - lam)) • (U + V) - η • (U * (Vᴴ * V) + V * (Uᴴ * U)) := by
  simp only [decayStep, mask_univ]
  rw [conjTranspose_sub, conjTranspose_mul, conjTranspose_conjTranspose, conjTranspose_one,
    Matrix.sub_mul, Matrix.sub_mul, Matrix.one_mul, Matrix.one_mul, Matrix.mul_assoc,
    Matrix.mul_assoc]
  module

omit [DecidableEq m] in
/-- `4 (U Vᴴ V + V Uᴴ U) = P (Pᴴ P + Qᴴ Q) − Q (Pᴴ Q + Qᴴ P)` for `P = U + V`, `Q = U − V`. -/
lemma four_smul_cubic (U V : Matrix m k ℂ) :
    (4 : ℝ) • (U * (Vᴴ * V) + V * (Uᴴ * U)) =
      (U + V) * ((U + V)ᴴ * (U + V) + (U - V)ᴴ * (U - V)) -
        (U - V) * ((U + V)ᴴ * (U - V) + (U - V)ᴴ * (U + V)) := by
  simp only [conjTranspose_add, conjTranspose_sub, Matrix.mul_add, Matrix.add_mul,
    Matrix.mul_sub, Matrix.sub_mul]
  module

omit [DecidableEq m] in
/-- `4 ⟨P, U Vᴴ V + V Uᴴ U⟩ ≤ ‖P‖² (‖P‖² + ‖Q‖²)` for `P = U + V`, `Q = U − V`. -/
lemma four_mul_pairing_add_cubic_le (U V : Matrix m k ℂ) :
    4 * pairing (U + V) (U * (Vᴴ * V) + V * (Uᴴ * U)) ≤
      frobeniusSq (U + V) * (frobeniusSq (U + V) + frobeniusSq (U - V)) := by
  have key : 4 * pairing (U + V) (U * (Vᴴ * V) + V * (Uᴴ * U)) =
      pairing (U + V) ((U + V) * ((U + V)ᴴ * (U + V) + (U - V)ᴴ * (U - V)) -
        (U - V) * ((U + V)ᴴ * (U - V) + (U - V)ᴴ * (U + V))) := by
    rw [← four_smul_cubic, pairing_smul_right]
  rw [key]
  generalize U + V = P
  generalize U - V = Q
  rw [pairing_sub_right, Matrix.mul_add, Matrix.mul_add, pairing_add_right, pairing_add_right]
  have hp := frobeniusSq_nonneg P
  have hq := frobeniusSq_nonneg Q
  have t1 : pairing P (P * (Pᴴ * P)) ≤ frobeniusSq P * frobeniusSq P := by
    rw [pairing_comm, pairing_mul_left, pairing_self]
    have a := frobeniusSq_mul_le Pᴴ P
    rwa [frobeniusSq_conjTranspose] at a
  have t2 : pairing P (P * (Qᴴ * Q)) ≤ frobeniusSq P * frobeniusSq Q := by
    rw [pairing_comm, pairing_mul_left]
    have a := frobeniusSq_mul_le Qᴴ Q
    have b := frobeniusSq_mul_le Pᴴ P
    rw [frobeniusSq_conjTranspose] at a b
    refine (le_abs_self _).trans (abs_pairing_le (mul_nonneg hp hq) ?_)
    calc frobeniusSq (Qᴴ * Q) * frobeniusSq (Pᴴ * P)
        ≤ frobeniusSq Q * frobeniusSq Q * (frobeniusSq P * frobeniusSq P) :=
          mul_le_mul a b (frobeniusSq_nonneg _) (mul_nonneg hq hq)
      _ = (frobeniusSq P * frobeniusSq Q) ^ 2 := by ring
  have t34 : 0 ≤ pairing P (Q * (Pᴴ * Q)) + pairing P (Q * (Qᴴ * P)) := by
    have e1 : pairing P (Q * (Pᴴ * Q)) = pairing (Pᴴ * Q) (Qᴴ * P) := by
      rw [pairing_comm, pairing_mul_left]
    have e2 : pairing P (Q * (Qᴴ * P)) = frobeniusSq (Qᴴ * P) := by
      rw [pairing_comm, pairing_mul_left, pairing_self]
    rw [e1, e2]
    have h := abs_pairing_le (A := Pᴴ * Q) (B := Qᴴ * P) (c := frobeniusSq (Qᴴ * P))
      (frobeniusSq_nonneg _) (by
        rw [show Pᴴ * Q = (Qᴴ * P)ᴴ by rw [conjTranspose_mul, conjTranspose_conjTranspose],
          frobeniusSq_conjTranspose, sq])
    linarith [neg_abs_le (pairing (Pᴴ * Q) (Qᴴ * P))]
  linarith

/-- **The sum of the factors after a step on the identity, from below**
(`review/flat_spectrum.md`, item WD16, statement 6): with
`a = 1 + η (1 − λ) ≥ 0`, `‖P'‖² ≥ a (a − η (‖U‖² + ‖V‖²)) ‖P‖²`. -/
lemma le_frobeniusSq_decayStep_identity_add (hη : 0 ≤ η) (U V : Matrix m k ℂ)
    (ha : 0 ≤ 1 + η * (1 - lam)) :
    (1 + η * (1 - lam)) * (1 + η * (1 - lam) - η * (frobeniusSq U + frobeniusSq V)) *
        frobeniusSq (U + V) ≤
      frobeniusSq ((decayStep G₁ η lam (U, V)).1 + (decayStep G₁ η lam (U, V)).2) := by
  rw [decayStep_identity_add, frobeniusSq_sub, frobeniusSq_smul, frobeniusSq_smul,
    pairing_smul_left, pairing_smul_right, frobeniusSq_add_frobeniusSq]
  have h := four_mul_pairing_add_cubic_le U V
  have hE := frobeniusSq_nonneg (U * (Vᴴ * V) + V * (Uᴴ * U))
  nlinarith [mul_nonneg (mul_nonneg ha hη) (sub_nonneg.2 h), mul_nonneg (sq_nonneg η) hE]

/-- **In the ball the sum stays nonzero, and near the origin it does not decrease**
(`review/flat_spectrum.md`, item WD16, statement 6). -/
lemma frobeniusSq_add_decayStep_identity (hη : 0 < η) (hκ0 : 0 ≤ κ)
    (hdesc : η * (lam + (ρ / 2 + κ) + 2 * ρ) ≤ 1) {U V : Matrix m k ℂ}
    (hw : frobeniusSq U + frobeniusSq V ≤ ρ) :
    η ^ 2 * frobeniusSq (U + V) ≤
        frobeniusSq ((decayStep G₁ η lam (U, V)).1 + (decayStep G₁ η lam (U, V)).2) ∧
      (frobeniusSq U + frobeniusSq V ≤ 1 - lam → frobeniusSq (U + V) ≤
        frobeniusSq ((decayStep G₁ η lam (U, V)).1 + (decayStep G₁ η lam (U, V)).2)) := by
  have hp := frobeniusSq_nonneg (U + V)
  have hs : 0 ≤ frobeniusSq U + frobeniusSq V :=
    add_nonneg (frobeniusSq_nonneg U) (frobeniusSq_nonneg V)
  have hρ0 : 0 ≤ ρ := hs.trans hw
  have ha2 : η ≤ 1 + η * (1 - lam) - η * (frobeniusSq U + frobeniusSq V) := by nlinarith
  have ha1 : 1 + η * (1 - lam) - η * (frobeniusSq U + frobeniusSq V) ≤ 1 + η * (1 - lam) := by
    nlinarith
  have hstep := le_frobeniusSq_decayStep_identity_add (lam := lam) hη.le U V (by linarith)
  refine ⟨le_trans ?_ hstep, fun hsmall => le_trans ?_ hstep⟩
  · exact mul_le_mul_of_nonneg_right (by nlinarith) hp
  · have hb2 : 1 ≤ 1 + η * (1 - lam) - η * (frobeniusSq U + frobeniusSq V) := by nlinarith
    exact le_mul_of_one_le_left hp (by nlinarith)

/-! The witness: the pair `U = V = 1` of the `2 × 2` identity, with `λ = ½`, `κ = 2`, `ρ = 8`
and `η = 1/200`. -/

example := le_frobeniusSq_decayStep_identity_add (lam := 1 / 2) (η := 1 / 200) (by norm_num)
  (1 : Matrix (Fin 2) (Fin 2) ℂ) 1 (by norm_num)

example := frobeniusSq_add_decayStep_identity (lam := 1 / 2) (η := 1 / 200) (κ := 2) (ρ := 8)
  (by norm_num) (by norm_num) (by norm_num) (U := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (V := 1)
  (by rw [frobeniusSq_one_fin_two]; norm_num)

end Transformer.Grokking.FlatSpectrum.Training.Scope
