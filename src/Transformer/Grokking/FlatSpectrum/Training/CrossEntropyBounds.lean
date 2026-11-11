module

public import Transformer.Grokking.FlatSpectrum.Training.CrossEntropy

/-!
# The three constants of the cross-entropy

Source: `review/flat_spectrum.md`, *Training with cross-entropy* (added), item XE2.
The run of gradient descent with weight decay is controlled through three bounds on the loss
(items WD6 and WD8). The squared loss has them in a ball. The cross-entropy `ceLoss A y` of
`|S|` samples whose probes have norm at most `a` has them at every matrix. A logit moves by at
most `a ‖H‖` when the product moves by `H` (`abs_pairing_le_mul_sqrt`), and

* `‖G X‖ ≤ 2 a |S|` (`frobeniusSq_ceGrad_le`): the gradient pairs with `H` to a softmax mean of
  logit increments minus one increment, at most `2 a ‖H‖` for each sample
  (`pairing_ceGrad_le`);
* `⟨G X, X⟩ ≥ −|S| log |C|` (`neg_le_pairing_ceGrad`): a convex loss bounded below by `l₀` has
  `⟨G X, X⟩ ≥ −(L 0 − l₀)` (`neg_le_pairing_of_first_order`), and here `L 0 = |S| log |C|`;
* `L Y ≤ L X + ⟨G X, Y − X⟩ + (|S| a² / 2) ‖Y − X‖²` (`ceLoss_le_add`), by Hoeffding's lemma
  for each sample (`crossEntropy_le_add`).

So `γ = 2 a |S|`, `b = |S| log |C|` and `s = |S| a²`, with no radius.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix
open Landscape.Neural (crossEntropy)
open Geometry (probability probability_pos sum_probability)

variable {m n S C : Type*} [Fintype m] [Fintype n] [Fintype S] [Fintype C]
  {A : S → C → Matrix m n ℂ} {y : S → C} {a : ℝ}

/-- A logit moves by at most `a ‖H‖` when the product moves by `H`: `|⟨B, H⟩| ≤ a ‖H‖` for a
probe with `‖B‖ ≤ a`. -/
lemma abs_pairing_le_mul_sqrt {B : Matrix m n ℂ} (ha0 : 0 ≤ a) (ha : frobeniusSq B ≤ a ^ 2)
    (H : Matrix m n ℂ) : |pairing B H| ≤ a * √(frobeniusSq H) :=
  abs_pairing_le (mul_nonneg ha0 (Real.sqrt_nonneg _)) (by
    rw [mul_pow, Real.sq_sqrt (frobeniusSq_nonneg H)]
    exact mul_le_mul_of_nonneg_right ha (frobeniusSq_nonneg H))

example := abs_pairing_le_mul_sqrt zero_le_one (frobeniusSq_signProbe_le 0 true) 1

/-- Finitely many probes have a common bound `a` on their norms. -/
lemma exists_forall_frobeniusSq_le (A : S → C → Matrix m n ℂ) :
    ∃ a : ℝ, 0 ≤ a ∧ ∀ s c, frobeniusSq (A s c) ≤ a ^ 2 := by
  refine ⟨√(∑ s, ∑ c, frobeniusSq (A s c)), Real.sqrt_nonneg _, fun s c => ?_⟩
  rw [Real.sq_sqrt (Finset.sum_nonneg fun s _ => Finset.sum_nonneg fun c _ =>
    frobeniusSq_nonneg _)]
  exact (Finset.single_le_sum (f := fun c => frobeniusSq (A s c))
    (fun c _ => frobeniusSq_nonneg _) (Finset.mem_univ c)).trans
    (Finset.single_le_sum (f := fun s => ∑ c, frobeniusSq (A s c))
      (fun s _ => Finset.sum_nonneg fun c _ => frobeniusSq_nonneg _) (Finset.mem_univ s))

/-- A matrix with `‖T‖² ≤ c ‖T‖` has `‖T‖ ≤ c`. -/
lemma frobeniusSq_le_sq_of_le_mul_sqrt {T : Matrix m n ℂ} {c : ℝ}
    (h : frobeniusSq T ≤ c * √(frobeniusSq T)) : frobeniusSq T ≤ c ^ 2 := by
  have h0 := Real.sqrt_nonneg (frobeniusSq T)
  have h1 := Real.sq_sqrt (frobeniusSq_nonneg T)
  rcases h0.eq_or_lt with h0' | h0'
  · have hF : frobeniusSq T = 0 := by
      rw [← h1, ← h0']
      norm_num
    rw [hF]
    positivity
  · have h2 : √(frobeniusSq T) ≤ c := by
      by_contra hcon
      rw [not_le] at hcon
      nlinarith [mul_pos (sub_pos.mpr hcon) h0']
    nlinarith

example := frobeniusSq_le_sq_of_le_mul_sqrt (T := (0 : Matrix (Fin 1) (Fin 1) ℂ)) (c := 1)
  (by simp [frobeniusSq])

/-- The gradient of the cross-entropy pairs with `H` to at most `2 a |S| ‖H‖`: for each sample
a softmax mean of logit increments minus the increment of the label. -/
lemma pairing_ceGrad_le (ha0 : 0 ≤ a) (ha : ∀ s c, frobeniusSq (A s c) ≤ a ^ 2)
    (X H : Matrix m n ℂ) :
    pairing (ceGrad A y X) H ≤ 2 * a * Fintype.card S * √(frobeniusSq H) := by
  have hs : ∀ s ∈ Finset.univ, (∑ c, probability (fun c => pairing (A s c) X) c *
      pairing (A s c) H - pairing (A s (y s)) H) ≤ 2 * (a * √(frobeniusSq H)) := by
    intro s _
    have hd := fun c => abs_pairing_le_mul_sqrt ha0 (ha s c) H
    have h1 : ∑ c, probability (fun c => pairing (A s c) X) c * pairing (A s c) H ≤
        ∑ c, probability (fun c => pairing (A s c) X) c * (a * √(frobeniusSq H)) :=
      Finset.sum_le_sum fun c _ =>
        mul_le_mul_of_nonneg_left (le_of_abs_le (hd c)) (probability_pos _ c).le
    rw [← Finset.sum_mul, sum_probability _ (y s), one_mul] at h1
    linarith [neg_le_of_abs_le (hd (y s))]
  rw [pairing_ceGrad]
  refine (Finset.sum_le_sum hs).trans (le_of_eq ?_)
  rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
  ring

/-- **The gradient of the cross-entropy is bounded** (`review/flat_spectrum.md`, item XE2):
`‖G X‖ ≤ 2 a |S|` at every `X`, for probes of norm at most `a`. -/
theorem frobeniusSq_ceGrad_le (ha0 : 0 ≤ a) (ha : ∀ s c, frobeniusSq (A s c) ≤ a ^ 2)
    (X : Matrix m n ℂ) : frobeniusSq (ceGrad A y X) ≤ (2 * a * Fintype.card S) ^ 2 :=
  frobeniusSq_le_sq_of_le_mul_sqrt
    ((pairing_self _).ge.trans (pairing_ceGrad_le ha0 ha X (ceGrad A y X)))

/-- A convex loss bounded below by `l₀` has `⟨G X, X⟩ ≥ −(L 0 − l₀)`: the first-order
inequality from `X` to `0`. -/
lemma neg_le_pairing_of_first_order {L : Matrix m n ℂ → ℝ} {G : Matrix m n ℂ → Matrix m n ℂ}
    (hconv : ∀ X Y, L X + pairing (G X) (Y - X) ≤ L Y) {l₀ : ℝ} (hlow : ∀ X, l₀ ≤ L X)
    (X : Matrix m n ℂ) : -(L 0 - l₀) ≤ pairing (G X) X := by
  have h := hconv X 0
  rw [pairing_sub_right, pairing_zero_right] at h
  linarith [hlow X]

variable (A y) in
/-- **The first-order term of the cross-entropy is bounded below**
(`review/flat_spectrum.md`, item XE2): `⟨G X, X⟩ ≥ −|S| log |C|` at every `X`. -/
theorem neg_le_pairing_ceGrad (X : Matrix m n ℂ) :
    -(Fintype.card S * Real.log (Fintype.card C)) ≤ pairing (ceGrad A y X) X := by
  have h := neg_le_pairing_of_first_order (ceLoss_first_order A y) (ceLoss_nonneg A y) X
  rwa [ceLoss_zero, sub_zero] at h

/-- **The cross-entropy lies below its quadratic model** (`review/flat_spectrum.md`, item
XE2), with the constant `s = |S| a²` at every pair of matrices:
`L Y ≤ L X + ⟨G X, Y − X⟩ + (s/2) ‖Y − X‖²`. -/
theorem ceLoss_le_add (ha0 : 0 ≤ a) (ha : ∀ s c, frobeniusSq (A s c) ≤ a ^ 2)
    (X Y : Matrix m n ℂ) :
    ceLoss A y Y ≤ ceLoss A y X + pairing (ceGrad A y X) (Y - X) +
      Fintype.card S * a ^ 2 / 2 * frobeniusSq (Y - X) := by
  have key : ∀ s ∈ Finset.univ, crossEntropy (fun c => pairing (A s c) Y) (y s) ≤
      crossEntropy (fun c => pairing (A s c) X) (y s) +
        (∑ c, probability (fun c => pairing (A s c) X) c * pairing (A s c) (Y - X) -
          pairing (A s (y s)) (Y - X)) + a ^ 2 / 2 * frobeniusSq (Y - X) := by
    intro s _
    have h := crossEntropy_le_add (fun c => pairing (A s c) X) (fun c => pairing (A s c) Y)
      (y s) (r := a * √(frobeniusSq (Y - X))) fun c => by
        rw [← pairing_sub_right]
        exact abs_pairing_le_mul_sqrt ha0 (ha s c) _
    rw [mul_pow, Real.sq_sqrt (frobeniusSq_nonneg _)] at h
    simp only [pairing_sub_right]
    linarith
  rw [pairing_ceGrad, ceLoss, ceLoss]
  refine (Finset.sum_le_sum key).trans (le_of_eq ?_)
  rw [Finset.sum_add_distrib, Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ,
    nsmul_eq_mul]
  ring

/-! The witness: one sample of a `1 × 1` product read with the code `±1`, so `a = 1`,
`γ = 2`, `b = log 2` and `s = 1`. -/

example := pairing_ceGrad_le (y := fun _ => true) zero_le_one frobeniusSq_signProbe_le 0 1

example := frobeniusSq_ceGrad_le (y := fun _ => true) zero_le_one frobeniusSq_signProbe_le 0

example := neg_le_pairing_of_first_order (ceLoss_first_order signProbe fun _ => true)
  (ceLoss_nonneg signProbe fun _ => true) 1

example := neg_le_pairing_ceGrad signProbe (fun _ => true) 1

example := ceLoss_le_add (y := fun _ => true) zero_le_one frobeniusSq_signProbe_le 0 1

end Transformer.Grokking.FlatSpectrum.Training
