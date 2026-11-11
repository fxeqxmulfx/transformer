module

public import Transformer.Grokking.FlatSpectrum.Training.SquaredLoss
public import Transformer.Grokking.FlatSpectrum.Training.Scope.SignRun

/-!
# The `1 × 1` identity: one step moves the sum of the factors

Source: `review/flat_spectrum.md`, *Training reaches the optimal fit* (added), item WD15,
statements 1 and 2; the item "Width" of WD11.

For the squared loss on the `1 × 1` identity the factors `U`, `V` are rows of any width and
the product is the number `x = U Vᴴ`. With `P = U + V` and `Q = U − V`:

* `Re x = (‖P‖² − ‖Q‖²) / 4` (`re_mul_conjTranspose_apply`) and `Im x = −½ ⟨P, i Q⟩`
  (`im_mul_conjTranspose_apply`);
* a step gives `P' = (1 − η (λ − 1 + Re x)) P + η Im x · i Q` (`decayStep_one_add`);
* so with `a = 1 − η (λ − 1 + Re x)`, for `a ≥ 0` and `η ≥ 0`,
  `‖P'‖² ≥ a (a − η ‖Q‖²) ‖P‖²` (`le_frobeniusSq_decayStep_one_add`), by
  `⟨P, i Q⟩² ≤ ‖P‖² ‖Q‖²`;
* in a ball `‖U‖² + ‖V‖² ≤ ρ` with `η (λ + ρ/2 + κ + 2 ρ) ≤ 1` this gives
  `‖P'‖² ≥ η² ‖P‖²`, and where `‖U‖² + ‖V‖² ≤ ⅔ (1 − λ)` it gives `‖P'‖² ≥ ‖P‖²`
  (`frobeniusSq_add_decayStep_one`): the sum stays nonzero, and near the origin it does not
  decrease.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training.Scope

open Matrix Filter Topology

variable {k : Type*} [Fintype k] {η lam κ ρ : ℝ}

lemma frobeniusSq_one_fin_one : frobeniusSq (1 : Matrix (Fin 1) (Fin 1) ℂ) = 1 := by
  simp [frobeniusSq]

/-- `‖c A‖² = |c|² ‖A‖²` for a complex number `c`. -/
lemma frobeniusSq_complex_smul {m n : Type*} [Fintype m] [Fintype n] (c : ℂ)
    (A : Matrix m n ℂ) : frobeniusSq (c • A) = ‖c‖ ^ 2 * frobeniusSq A := by
  simp [frobeniusSq, mul_pow, Finset.mul_sum]

omit [Fintype k] in
/-- A `1 × 1` matrix multiplies a row as its entry. -/
lemma fin_one_mul (A : Matrix (Fin 1) (Fin 1) ℂ) (V : Matrix (Fin 1) k ℂ) :
    A * V = A 0 0 • V := by
  ext i j
  rw [Subsingleton.elim i 0]
  simp [Matrix.mul_apply]

local notation "G₁" => fun X : Matrix (Fin 1) (Fin 1) ℂ => mask Set.univ (X - 1)

/-- **The sum of the factors after a step on the `1 × 1` identity** (`review/flat_spectrum.md`,
item WD15, statement 1): `P' = (1 − η (λ − 1 + Re x)) P + η Im x · i Q`. -/
lemma decayStep_one_add (η lam : ℝ) (U V : Matrix (Fin 1) k ℂ) :
    (decayStep G₁ η lam (U, V)).1 + (decayStep G₁ η lam (U, V)).2 =
      (1 - η * (lam + (((U * Vᴴ) 0 0).re - 1))) • (U + V) +
        (η * ((U * Vᴴ) 0 0).im) • (Complex.I • (U - V)) := by
  have hmask : mask Set.univ (U * Vᴴ - 1) = U * Vᴴ - 1 := by
    ext i j
    simp [mask]
  simp only [decayStep, hmask]
  rw [fin_one_mul, fin_one_mul]
  ext i j
  simp only [Matrix.add_apply, Matrix.sub_apply, Matrix.smul_apply, conjTranspose_apply,
    Matrix.one_apply_eq, Complex.real_smul, smul_eq_mul, star_sub, star_one]
  apply Complex.ext <;> simp <;> ring

/-- `Im (U Vᴴ) = −½ ⟨U + V, i (U − V)⟩`. -/
lemma im_mul_conjTranspose_apply (U V : Matrix (Fin 1) k ℂ) :
    ((U * Vᴴ) 0 0).im = -(pairing (U + V) (Complex.I • (U - V))) / 2 := by
  simp only [pairing, Matrix.trace, Matrix.diag_apply, Matrix.mul_apply, conjTranspose_apply,
    Fin.sum_univ_one, Complex.im_sum, Complex.re_sum, Matrix.add_apply, Matrix.sub_apply,
    Matrix.smul_apply, smul_eq_mul]
  rw [← Finset.sum_neg_distrib, Finset.sum_div]
  refine Finset.sum_congr rfl fun j _ => ?_
  simp [Complex.mul_re, Complex.mul_im]
  ring

lemma frobeniusSq_I_smul (Q : Matrix (Fin 1) k ℂ) :
    frobeniusSq (Complex.I • Q) = frobeniusSq Q := by
  simp [frobeniusSq]

/-- `Re (U Vᴴ) = (‖U + V‖² − ‖U − V‖²) / 4`. -/
lemma re_mul_conjTranspose_apply (U V : Matrix (Fin 1) k ℂ) :
    ((U * Vᴴ) 0 0).re = (frobeniusSq (U + V) - frobeniusSq (U - V)) / 4 := by
  rw [← pairing_one_mul_conjTranspose]
  simp [pairing, Matrix.trace_fin_one]

/-- **The sum of the factors after a step, from below** (`review/flat_spectrum.md`, item WD15,
statement 1): `‖P'‖² ≥ a (a − η ‖Q‖²) ‖P‖²` for `a = 1 − η (λ − 1 + Re x) ≥ 0`. -/
lemma le_frobeniusSq_decayStep_one_add (hη : 0 ≤ η) (U V : Matrix (Fin 1) k ℂ)
    (ha : 0 ≤ 1 - η * (lam - 1 + (frobeniusSq (U + V) - frobeniusSq (U - V)) / 4)) :
    (1 - η * (lam - 1 + (frobeniusSq (U + V) - frobeniusSq (U - V)) / 4)) *
        (1 - η * (lam - 1 + (frobeniusSq (U + V) + 3 * frobeniusSq (U - V)) / 4)) *
        frobeniusSq (U + V) ≤
      frobeniusSq ((decayStep G₁ η lam (U, V)).1 + (decayStep G₁ η lam (U, V)).2) := by
  have hc := pairing_sq_le (U + V) (Complex.I • (U - V))
  have hq := frobeniusSq_nonneg (U - V)
  have hR : ∀ a b : ℝ, frobeniusSq (a • (U + V) + b • (Complex.I • (U - V))) =
      a ^ 2 * frobeniusSq (U + V) + 2 * (a * (b * pairing (U + V) (Complex.I • (U - V)))) +
        b ^ 2 * frobeniusSq (U - V) := fun a b => by
    rw [frobeniusSq_add, frobeniusSq_smul, frobeniusSq_smul, pairing_smul_left,
      pairing_smul_right, frobeniusSq_I_smul]
  rw [decayStep_one_add, hR, im_mul_conjTranspose_apply, re_mul_conjTranspose_apply]
  rw [frobeniusSq_I_smul] at hc
  generalize pairing (U + V) (Complex.I • (U - V)) = c at hc ⊢
  generalize frobeniusSq (U + V) = p at *
  generalize frobeniusSq (U - V) = q at *
  have h1 := mul_nonneg (mul_nonneg ha hη) (sub_nonneg.2 hc)
  have h2 := mul_nonneg (mul_nonneg (sq_nonneg η) (sq_nonneg c)) hq
  nlinarith [h1, h2]

/-- **In the ball the sum stays nonzero, and near the origin it does not decrease**
(`review/flat_spectrum.md`, item WD15, statement 2). -/
lemma frobeniusSq_add_decayStep_one (hη : 0 < η) (hκ0 : 0 ≤ κ)
    (hdesc : η * (lam + (ρ / 2 + κ) + 2 * ρ) ≤ 1) {U V : Matrix (Fin 1) k ℂ}
    (hw : frobeniusSq U + frobeniusSq V ≤ ρ) :
    η ^ 2 * frobeniusSq (U + V) ≤
        frobeniusSq ((decayStep G₁ η lam (U, V)).1 + (decayStep G₁ η lam (U, V)).2) ∧
      (frobeniusSq U + frobeniusSq V ≤ 2 * (1 - lam) / 3 → frobeniusSq (U + V) ≤
        frobeniusSq ((decayStep G₁ η lam (U, V)).1 + (decayStep G₁ η lam (U, V)).2)) := by
  have hp := frobeniusSq_nonneg (U + V)
  have hq := frobeniusSq_nonneg (U - V)
  have hn := frobeniusSq_add_frobeniusSq U V
  have hρ0 : 0 ≤ ρ := (add_nonneg (frobeniusSq_nonneg U) (frobeniusSq_nonneg V)).trans hw
  have ha2 : η ≤ 1 - η * (lam - 1 + (frobeniusSq (U + V) + 3 * frobeniusSq (U - V)) / 4) := by
    nlinarith
  have ha1 : 1 - η * (lam - 1 + (frobeniusSq (U + V) + 3 * frobeniusSq (U - V)) / 4) ≤
      1 - η * (lam - 1 + (frobeniusSq (U + V) - frobeniusSq (U - V)) / 4) := by
    nlinarith
  have hstep := le_frobeniusSq_decayStep_one_add (lam := lam) hη.le U V (by linarith)
  refine ⟨le_trans ?_ hstep, fun hsmall => le_trans ?_ hstep⟩
  · exact mul_le_mul_of_nonneg_right (by nlinarith) hp
  · have hb2 : 1 ≤ 1 - η * (lam - 1 + (frobeniusSq (U + V) + 3 * frobeniusSq (U - V)) / 4) := by
      nlinarith
    exact le_mul_of_one_le_left hp (by nlinarith)

/-! The witness: the pair `U = V = 0` of width one, with `λ = ½`, `κ = 1`, `ρ = 4` and
`η = 1/100`. -/

example := le_frobeniusSq_decayStep_one_add (lam := 1 / 2) (η := 1 / 100) (by norm_num)
  (0 : Matrix (Fin 1) (Fin 1) ℂ) 0 (by simp [frobeniusSq]; norm_num)

example := frobeniusSq_add_decayStep_one (lam := 1 / 2) (η := 1 / 100) (κ := 1) (ρ := 4)
  (by norm_num) zero_le_one (by norm_num) (U := (0 : Matrix (Fin 1) (Fin 1) ℂ)) (V := 0)
  (by simp [frobeniusSq])

end Transformer.Grokking.FlatSpectrum.Training.Scope
