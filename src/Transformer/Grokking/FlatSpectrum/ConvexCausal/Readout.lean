module

public import Transformer.Grokking.FlatSpectrum.ConvexCausal.Generation
public import Transformer.Grokking.FlatSpectrum.ConvexCausal.Calibration

/-!
# Numeric readouts and task-specific sequence scores

Source: `review/convex_causal.md`, CCa10, and the numeric probes, field readouts
and forecast metrics in `diff-gpt/stress_tests_for_gpt.md`. A fixed decoder does
not change categorical training. Recovering generated tokens preserves any
fixed original scorer. A conditional distribution also admits the usual mean
readout with its exact squared-error decomposition; intrinsic variance and
codec errors remain and are not replaced by a universal accuracy claim.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexCausal

open scoped BigOperators

variable {C : Type*} [Fintype C]

/-- Fixed numeric decoding followed by a conditional mean (CCa10). -/
noncomputable def mean (p : C → ℝ) (decode : C → ℝ) : ℝ := ∑ c, p c * decode c

/-- Conditional squared error in the decoder's original numeric units (CCa10). -/
noncomputable def risk (p : C → ℝ) (decode : C → ℝ) (a : ℝ) : ℝ :=
  ∑ c, p c * (decode c - a) ^ 2

/-- Exact variance/bias decomposition, retaining the irreducible conditional error (CCa10). -/
theorem risk_decomposition (p : C → ℝ) (decode : C → ℝ) (hsum : ∑ c, p c = 1) (a : ℝ) :
    risk p decode a = risk p decode (mean p decode) + (mean p decode - a) ^ 2 := by
  let μ := mean p decode
  calc
    risk p decode a = ∑ c, (p c * (decode c - μ) ^ 2 +
        2 * (μ - a) * (p c * decode c) + (a ^ 2 - μ ^ 2) * p c) := by
      apply Finset.sum_congr rfl
      intro c _
      ring
    _ = risk p decode μ + 2 * (μ - a) * μ + (a ^ 2 - μ ^ 2) := by
      simp only [Finset.sum_add_distrib, ← Finset.mul_sum, hsum, mul_one]
      rfl
    _ = risk p decode (mean p decode) + (mean p decode - a) ^ 2 := by
      dsimp [μ]
      ring

/-- The decoded conditional mean minimizes conditional squared error (CCa10). -/
theorem mean_minimizes_risk (p : C → ℝ) (decode : C → ℝ) (hsum : ∑ c, p c = 1) (a : ℝ) :
    risk p decode (mean p decode) ≤ risk p decode a := by
  rw [risk_decomposition p decode hsum a]
  exact le_add_of_nonneg_right (sq_nonneg _)

/-- Probability mass makes the remaining numeric risk nonnegative (CCa10). -/
theorem risk_nonneg (p : C → ℝ) (hp : ∀ c, 0 ≤ p c) (decode : C → ℝ) (a : ℝ) :
    0 ≤ risk p decode a :=
  Finset.sum_nonneg fun c _ => mul_nonneg (hp c) (sq_nonneg _)

variable [Nonempty C] [DecidableEq C]

/-- Exact categorical recovery leaves precisely the codec error against a real teacher (CCa10). -/
theorem decoded_error {X : Type*} (f : X → C) (decode : C → ℝ) {ε : ℝ}
    (hε : ε < 1) (x : X) (teacher : ℝ) :
    (decode (best (smooth ε (point f) x)) - teacher) ^ 2 = (decode (f x) - teacher) ^ 2 := by
  rw [best_point f hε x]

/-- Free-generation equality preserves the original task scorer, including EOS and numeric decoding
(CCa10). The score is fixed, not a substituted training metric. -/
theorem recovered_score {H S : Type*} [Fintype H] [DecidableEq H] [Fintype S]
    (rule : H → List C → C) (input : S → List C) (label : S → C) (g : H)
    (hfit : ∀ s, rule g (input s) = label s) (hsep : Separates rule input)
    {ε : ℝ} (hε : ε < 1) (eos : C) (n : ℕ) (history : List C) (score : List C → ℝ) :
    score (generate (fun xs => best (smooth ε (trained rule input label) xs)) eos n history) =
      score (generate (rule g) eos n history) := by
  apply congrArg score
  exact trained_generation rule input label g hfit hsep hε eos n history

example := risk_decomposition (fun _ : Bool => (1 / 2 : ℝ)) (fun c => if c then 2 else 0)
  (by norm_num [Fintype.sum_bool]) 3
example := mean_minimizes_risk (fun _ : Bool => (1 / 2 : ℝ)) (fun c => if c then 2 else 0)
  (by norm_num [Fintype.sum_bool]) 3
example := risk_nonneg (fun _ : Bool => (1 / 2 : ℝ)) (by intro c; norm_num)
  (fun c => if c then 2 else 0) 1
example := decoded_error (id : Bool → Bool) (fun c => if c then 2 else 0)
  (ε := 1 / 4) (by norm_num) true (9 / 4)
example := recovered_score alternating (fun _ : Unit => []) (fun _ => false) false
  (by intro s; rfl) alternating_separates (ε := 1 / 4) (by norm_num) true 4 []
  (fun xs => xs.length)

end Transformer.Grokking.FlatSpectrum.ConvexCausal
