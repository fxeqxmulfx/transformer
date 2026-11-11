module

public import Mathlib.Analysis.Calculus.Deriv.Add
public import Mathlib.Analysis.Calculus.Deriv.Mul
public import Mathlib.Analysis.Calculus.Deriv.Pow
public import Transformer.Grokking.Protocol.AdamLimit

/-!
# AdamW does not end at the minimizer of the weight-decayed loss: two witnesses

Source: `review/flat_spectrum.md`, *Training reaches the optimal fit* (added), item WD13;
Lemma 1 of that note, "Training with weight decay then behaves like minimum-nuclear-norm
interpolation", and the item "The model and the optimizer" of WD11.

By `AdamLimit` a limit `θ` of the author's AdamW run on one gradient field `g` satisfies
`wd · θ = − g θ / (|g θ| + ε)`, so it is stationary for the loss with the penalty
`(wd/2) ‖θ‖²` only where the gradient is `0` or has the absolute value `1 − ε`. With the
source's constants and weight decay `1`:

* the loss `½ (p − 3)²` of one parameter, with the gradient `scalarField`
  (`hasDerivAt_scalarLoss`): the penalized loss has the minimizer `p = 3/2`, where the
  gradient is `−3/2`, and no run converges to it (`not_tendsto_adamRun_scalar`);
* the pure factorization `x = u v` of a `1 × 1` product with the squared loss `½ (u v − 3)²`,
  with the gradient `factorField` (`hasDerivAt_factorLoss_left`, `hasDerivAt_factorLoss_right`):
  the weight-decayed loss `½ (u v − 3)² + ½ (u² + v²)` of WD2 (`factorLoss`) has the least
  value `5/2`, attained exactly at `u = v = ±√2` (`factorLoss_le_iff`), where the gradients of
  the loss are `∓√2`, and no run converges to a minimizer, whatever its start
  (`not_tendsto_adamRun_factor`).

So for AdamW the modeling step of Lemma 1 fails already for the pure factorization, here with
real factors of width one: the limits of gradient descent with weight decay that WD5 and WD8
describe minimize the weight-decayed loss, and a limit of AdamW never does. A run that stays
at its start is the witness of the hypotheses (`adamRun_zero`).
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

open Filter Topology

variable {ι : Type*}

/-- On the zero field the run from zero stays there, for every weight decay. -/
lemma adamRun_zero (h : Hyper) (t : ℕ) :
    adamRun h (fun _ _ => 0) (0 : ι → ℝ) t = ⟨0, 0, 0⟩ := by
  induction t with
  | zero => rfl
  | succ t ih =>
    rw [adamRun, ih]
    simp only [adamStep]
    congr 1 <;> funext i <;> simp

/-- The gradient of `½ (p − 3)²`. -/
def scalarField : (Unit → ℝ) → Unit → ℝ := fun θ _ => θ () - 3

lemma continuous_scalarField : Continuous scalarField :=
  continuous_pi fun _ => (continuous_apply ()).sub continuous_const

/-- `scalarField` is the derivative of `½ (p − 3)²`. -/
lemma hasDerivAt_scalarLoss (θ : Unit → ℝ) :
    HasDerivAt (fun p : ℝ => (p - 3) ^ 2 / 2) (scalarField θ ()) (θ ()) := by
  have h0 : HasDerivAt (fun p : ℝ => p - 3) 1 (θ ()) := HasDerivAt.sub_const 3 (hasDerivAt_id' _)
  refine (HasDerivAt.div_const (HasDerivAt.fun_pow h0 2) 2).congr_deriv ?_
  simp only [scalarField]
  ring

/-- **No AdamW run on `½ (p − 3)²` converges to the minimizer `3/2` of the loss with the
penalty `½ p²`** (`review/flat_spectrum.md`, item WD13), for the source's constants with
weight decay `1` and every start. -/
theorem not_tendsto_adamRun_scalar (θ₀ : Unit → ℝ) :
    ¬ Tendsto (fun t => (adamRun (Hyper.source 1) (fun _ => scalarField) θ₀ t).params) atTop
      (𝓝 fun _ => 3 / 2) := by
  intro hlim
  rcases eq_zero_or_abs_eq_of_tendsto_adamRun (Hyper.source_regular 1) continuous_scalarField
    hlim () (by norm_num [scalarField, Hyper.source]) with h0 | h0
  · norm_num [scalarField] at h0
  · rw [show scalarField (fun _ => 3 / 2) () = -(3 / 2) by norm_num [scalarField], abs_neg,
      abs_of_pos (by norm_num)] at h0
    norm_num [Hyper.source] at h0

/-- The gradient of `½ (u v − 3)²` in the factors `u = θ false`, `v = θ true`. -/
def factorField : (Bool → ℝ) → Bool → ℝ := fun θ b => (θ false * θ true - 3) * θ (!b)

lemma continuous_factorField : Continuous factorField :=
  continuous_pi fun b => (((continuous_apply false).mul (continuous_apply true)).sub
    continuous_const).mul (continuous_apply (!b))

/-- `factorField` at `u` is the derivative of `½ (u v − 3)²` in `u`. -/
lemma hasDerivAt_factorLoss_left (θ : Bool → ℝ) :
    HasDerivAt (fun u : ℝ => (u * θ true - 3) ^ 2 / 2) (factorField θ false) (θ false) := by
  have h0 : HasDerivAt (fun u : ℝ => u * θ true - 3) (1 * θ true) (θ false) :=
    HasDerivAt.sub_const 3 (HasDerivAt.mul_const (hasDerivAt_id' _) _)
  refine (HasDerivAt.div_const (HasDerivAt.fun_pow h0 2) 2).congr_deriv ?_
  simp only [factorField, Bool.not_false]
  ring

/-- `factorField` at `v` is the derivative of `½ (u v − 3)²` in `v`. -/
lemma hasDerivAt_factorLoss_right (θ : Bool → ℝ) :
    HasDerivAt (fun v : ℝ => (θ false * v - 3) ^ 2 / 2) (factorField θ true) (θ true) := by
  have h0 : HasDerivAt (fun v : ℝ => θ false * v - 3) (θ false * 1) (θ true) :=
    HasDerivAt.sub_const 3 (HasDerivAt.const_mul _ (hasDerivAt_id' _))
  refine (HasDerivAt.div_const (HasDerivAt.fun_pow h0 2) 2).congr_deriv ?_
  simp only [factorField, Bool.not_true]
  ring

/-- The weight-decayed loss `½ (u v − 3)² + ½ (u² + v²)` of the `1 × 1` factorization
(`review/flat_spectrum.md`, item WD2, with the kernel `3` and `λ = 1`). -/
noncomputable def factorLoss (u v : ℝ) : ℝ := (u * v - 3) ^ 2 / 2 + (u ^ 2 + v ^ 2) / 2

/-- The weight-decayed loss is `5/2 + ½ (u v − 2)² + ½ (u − v)²`. -/
lemma factorLoss_eq (u v : ℝ) :
    factorLoss u v = 5 / 2 + (u * v - 2) ^ 2 / 2 + (u - v) ^ 2 / 2 := by
  unfold factorLoss
  ring

/-- Its least value is `5/2`. -/
lemma five_halves_le_factorLoss (u v : ℝ) : 5 / 2 ≤ factorLoss u v := by
  rw [factorLoss_eq]
  nlinarith [sq_nonneg (u * v - 2), sq_nonneg (u - v)]

/-- It is attained exactly at `u = v` with `u² = 2`. -/
lemma factorLoss_le_iff {u v : ℝ} : factorLoss u v ≤ 5 / 2 ↔ u = v ∧ u ^ 2 = 2 := by
  rw [factorLoss_eq]
  constructor
  · intro h
    have h1 : (u - v) ^ 2 = 0 := by nlinarith [sq_nonneg (u * v - 2), sq_nonneg (u - v)]
    have h2 : (u * v - 2) ^ 2 = 0 := by nlinarith [sq_nonneg (u * v - 2), sq_nonneg (u - v)]
    have huv : u = v := sub_eq_zero.1 (pow_eq_zero_iff two_ne_zero |>.1 h1)
    refine ⟨huv, ?_⟩
    have h3 := pow_eq_zero_iff two_ne_zero |>.1 h2
    rw [← huv] at h3
    linarith
  · rintro ⟨rfl, hu⟩
    have h3 : u * u = 2 := by linarith [sq u]
    rw [h3]
    norm_num

/-- **No AdamW run on the `1 × 1` factorization converges to a minimizer of its
weight-decayed loss** (`review/flat_spectrum.md`, item WD13): for the squared loss
`½ (u v − 3)²`, the source's constants with weight decay `1` and every start, the limit of the
parameters, if there is one, does not minimize `½ (u v − 3)² + ½ (u² + v²)`. -/
theorem not_tendsto_adamRun_factor (θ₀ : Bool → ℝ) {θ : Bool → ℝ}
    (hmin : ∀ u v, factorLoss (θ false) (θ true) ≤ factorLoss u v) :
    ¬ Tendsto (fun t => (adamRun (Hyper.source 1) (fun _ => factorField) θ₀ t).params) atTop
      (𝓝 θ) := by
  intro hlim
  obtain ⟨huv, hs⟩ := factorLoss_le_iff.1
    ((hmin √2 √2).trans (factorLoss_le_iff.2 ⟨rfl, Real.sq_sqrt zero_le_two⟩))
  have hf : factorField θ false = -θ false := by
    simp only [factorField, Bool.not_false]
    rw [← huv]
    linear_combination θ false * hs
  have he : (Hyper.source 1).eps = 1 / 10 ^ 8 := rfl
  rcases eq_zero_or_abs_eq_of_tendsto_adamRun (Hyper.source_regular 1) continuous_factorField
    hlim false (by rw [hf]; simp [Hyper.source]) with h0 | h0
  · rw [hf, neg_eq_zero] at h0
    rw [h0] at hs
    norm_num at hs
  · rw [hf, abs_neg, he] at h0
    have h4 : θ false ^ 2 = (1 - 1 / 10 ^ 8) ^ 2 := by rw [← sq_abs, h0]
    rw [hs] at h4
    norm_num at h4

/-! The witnesses of the hypotheses: the zero field from the zero start, where the run does
not move, with the source's constants; and the minimizer `u = v = √2`. -/

private lemma zero_tendsto (wd : ℝ) : Tendsto (fun t =>
    (adamRun (Hyper.source wd) (fun _ _ => 0) (0 : Unit → ℝ) t).params) atTop (𝓝 0) :=
  tendsto_const_nhds.congr fun t => by rw [adamRun_zero]

example := tendsto_movingAverage (β := 1 / 2) (a := 1) (x := fun _ => 1) (g := fun _ => 1)
  (by norm_num) (by norm_num) (fun _ => by norm_num) tendsto_const_nhds

example := tendsto_adamRun_first (Hyper.source_regular 1).2.1 (Hyper.source_regular 1).2.2.1
  continuous_const (zero_tendsto 1) ()

example := tendsto_adamRun_second (Hyper.source_regular 1).2.2.2.1
  (Hyper.source_regular 1).2.2.2.2.1 continuous_const (zero_tendsto 1) ()

example := decay_mul_eq_of_tendsto_adamRun (Hyper.source_regular 1) continuous_const
  (zero_tendsto 1) ()

example := abs_decay_mul_lt_one_of_tendsto_adamRun (Hyper.source_regular 1) continuous_const
  (zero_tendsto 1) ()

example := eq_zero_or_abs_eq_of_tendsto_adamRun (Hyper.source_regular 1) continuous_const
  (zero_tendsto 1) () (by simp)

example := eq_zero_of_tendsto_adamRun (Hyper.source_regular 0) rfl continuous_const
  (zero_tendsto 0) ()

example (θ₀ : Bool → ℝ) := not_tendsto_adamRun_factor θ₀ (θ := fun _ => √2) fun u v =>
  (factorLoss_le_iff.2 ⟨rfl, Real.sq_sqrt zero_le_two⟩).trans (five_halves_le_factorLoss u v)

end Transformer.Grokking.Protocol
