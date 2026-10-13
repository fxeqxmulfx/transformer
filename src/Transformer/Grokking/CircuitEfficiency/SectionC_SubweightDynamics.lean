import Transformer.Grokking.CircuitEfficiency.SectionC_SubweightGradient

/-!
# Finite gradient descent on the actual product subweights

Source: Varma et al., arXiv:2309.02390v1, section 3, Slow vs fast
learning, and appendix C, eq:dynamics and Table simulation parameters.
Use the four partial derivatives proved from actual multiclass CE.
Updates are simultaneous ordinary gradient descent with a coupled
power penalty, precisely the source's stated minimal-model algorithm.

Derive product growth and imbalance changes at finite rate. At the
one-zero-factor initialization, product growth is proportional to the
square of the seeded factor. Unequal seeds give correct train and
incorrect test decisions after one step. An entirely absent factor
pair stays absent forever; the correct effective minimum alone cannot
guarantee its discovery. Native AdamW and rounding remain separate.
-/

namespace Transformer.Grokking.CircuitEfficiency

/-- Simultaneous gradient descent on all four source subweights.
Source: arXiv:2309.02390v1, appendix C, eq:dynamics; the gradient
candidate is proved to equal actual partial derivatives for r>=1. -/
noncomputable def subweightGDStep (remaining : ℕ)
    (multiplier firstCost secondCost exponent rate : ℝ) (z : (ℝ × ℝ) × (ℝ × ℝ)) :
    (ℝ × ℝ) × (ℝ × ℝ) :=
  let g := subweightGradient remaining multiplier firstCost secondCost exponent z.1.1 z.1.2 z.2.1 z.2.2
  ((z.1.1 - rate * g.1.1, z.1.2 - rate * g.1.2),
   (z.2.1 - rate * g.2.1, z.2.2 - rate * g.2.2))

/-- Repeated source updates without changing the rate or objective.
Source: arXiv:2309.02390v1, appendix C, eq:dynamics. This defines an
actual iteration, rather than prescribing its success or delay. -/
noncomputable def subweightGDPath (remaining : ℕ)
    (multiplier firstCost secondCost exponent rate : ℝ) (initial : (ℝ × ℝ) × (ℝ × ℝ)) :
    ℕ → (ℝ × ℝ) × (ℝ × ℝ)
  | 0 => initial
  | n + 1 => subweightGDStep remaining multiplier firstCost secondCost exponent rate
      (subweightGDPath remaining multiplier firstCost secondCost exponent rate initial n)

/-- The actual first circuit's product update includes a finite quadratic
rate term and state-dependent factor energy. Source: arXiv:2309.02390v1,
appendix C, eq:dynamics; it is not gradient descent directly on the product. -/
theorem subweight_gd_product_update (remaining : ℕ)
    (multiplier firstCost secondCost exponent rate a b c d : ℝ) :
    let next := subweightGDStep remaining multiplier firstCost secondCost exponent rate ((a, b), (c, d))
    let marginal := circuitMarginal remaining multiplier firstCost exponent (a * b) (c * d)
    next.1.1 * next.1.2 = a * b * (1 + rate ^ 2 * marginal ^ 2) -
      rate * marginal * (a ^ 2 + b ^ 2) := by
  dsimp [subweightGDStep, subweightGradient]
  ring

/-- Unlike continuous gradient flow, a finite GD update multiplies the
factor-square imbalance by a rate-dependent coefficient. Source:
arXiv:2309.02390v1, appendix C, eq:dynamics, derived without an ODE transfer. -/
theorem subweight_gd_imbalance_update (remaining : ℕ)
    (multiplier firstCost secondCost exponent rate a b c d : ℝ) :
    let next := subweightGDStep remaining multiplier firstCost secondCost exponent rate ((a, b), (c, d))
    let marginal := circuitMarginal remaining multiplier firstCost exponent (a * b) (c * d)
    next.1.1 ^ 2 - next.1.2 ^ 2 = (1 - rate ^ 2 * marginal ^ 2) * (a ^ 2 - b ^ 2) := by
  dsimp [subweightGDStep, subweightGradient]
  ring

/-- At the source's tied initialization only the first factors change.
Source: arXiv:2309.02390v1, appendix C's eq:dynamics and Table; no
stationary-gradient or learning-speed assumption replaces actual CE. -/
theorem subweight_gd_initial_step (remaining : ℕ)
    (multiplier firstCost secondCost exponent rate genSeed memSeed : ℝ) (hr : 1 < exponent) :
    subweightGDStep remaining multiplier firstCost secondCost exponent rate ((0, genSeed), (0, memSeed)) =
      ((rate * (((remaining : ℝ) + 1) / ((remaining : ℝ) + 2)) * genSeed, genSeed),
       (rate * (((remaining : ℝ) + 1) / ((remaining : ℝ) + 2)) * memSeed, memSeed)) := by
  unfold subweightGDStep
  rw [subweight_gradient_initial remaining multiplier firstCost secondCost exponent genSeed memSeed hr]
  dsimp
  congr 2 <;> ring

example : (1 : ℝ) < 5 / 3 := by norm_num

/-- The first product weights scale as the squared seed values. Source:
arXiv:2309.02390v1, section 3, Slow vs fast learning; this finite-step
identity is a consequence of its actual multiclass CE gradient. -/
theorem subweight_gd_initial_products (remaining : ℕ)
    (multiplier firstCost secondCost exponent rate genSeed memSeed : ℝ) (hr : 1 < exponent) :
    let next := subweightGDStep remaining multiplier firstCost secondCost exponent rate ((0, genSeed), (0, memSeed))
    (next.1.1 * next.1.2, next.2.1 * next.2.2) =
      (rate * (((remaining : ℝ) + 1) / ((remaining : ℝ) + 2)) * genSeed ^ 2,
       rate * (((remaining : ℝ) + 1) / ((remaining : ℝ) + 2)) * memSeed ^ 2) := by
  rw [subweight_gd_initial_step remaining multiplier firstCost secondCost exponent rate genSeed memSeed hr]
  dsimp
  congr 1 <;> ring

example : (1 : ℝ) < 5 / 3 := by norm_num

/-- An unequal nonnegative seeding produces perfect train decisions but
wrong test decisions after one finite positive-rate update. Source:
arXiv:2309.02390v1, section 3 and appendix C, the initial memorization
phase. This says neither that the objective decreases at any rate nor
that a later delayed transition necessarily occurs. -/
theorem subweight_gd_initial_train_fit_test_failure (remaining : ℕ)
    (multiplier firstCost secondCost exponent rate genSeed memSeed : ℝ)
    (hr : 1 < exponent) (heta : 0 < rate) (hg : 0 ≤ genSeed) (hgm : genSeed < memSeed) :
    let next := subweightGDStep remaining multiplier firstCost secondCost exponent rate ((0, genSeed), (0, memSeed))
    Transformer.Grokking.NaiveLoss.StrictCorrect
      (trainTableLogits remaining (next.1.1 * next.1.2) (next.2.1 * next.2.2)) 0 ∧
    ¬ Transformer.Grokking.NaiveLoss.StrictCorrect
      (heldoutTableLogits remaining (next.1.1 * next.1.2) (next.2.1 * next.2.2)) 0 := by
  rw [subweight_gd_initial_step remaining multiplier firstCost secondCost exponent rate genSeed memSeed hr]
  dsimp
  have hm : 0 < memSeed := lt_of_le_of_lt hg hgm
  have hc : 0 < rate * (((remaining : ℝ) + 1) / ((remaining : ℝ) + 2)) := by positivity
  have hs : genSeed ^ 2 < memSeed ^ 2 := by nlinarith
  have horder := mul_lt_mul_of_pos_left hs hc
  constructor
  · apply (train_table_strict_correct_iff remaining _ _).mpr
    nlinarith [sq_nonneg genSeed, mul_pos hc (sq_pos_of_pos hm)]
  · apply heldout_table_incorrect
    nlinarith [horder]

example : (1 : ℝ) < 5 / 3 ∧ 0 < (1 / 100 : ℝ) ∧ 0 ≤ (1 / 200 : ℝ) ∧
    (1 / 200 : ℝ) < 1 := by norm_num

/-- A completely absent first factor pair cannot be activated by its
coordinate gradients. Source: arXiv:2309.02390v1, section 3's product
parameterization; this holds independently of the other circuit state. -/
theorem subweight_gd_preserves_absent_gen (remaining : ℕ)
    (multiplier firstCost secondCost exponent rate : ℝ) (z : (ℝ × ℝ) × (ℝ × ℝ))
    (hz : z.1 = (0, 0)) :
    (subweightGDStep remaining multiplier firstCost secondCost exponent rate z).1 = (0, 0) := by
  rcases z with ⟨⟨a, b⟩, ⟨c, d⟩⟩
  have ha := congrArg Prod.fst hz
  have hb := congrArg Prod.snd hz
  dsimp at ha hb
  subst a
  subst b
  simp [subweightGDStep, subweightGradient]

example : (((0 : ℝ), (0 : ℝ)), ((0 : ℝ), (1 : ℝ))).1 = (0, 0) := by rfl

/-- The entirely absent Gen circuit remains absent at every iteration,
even when its effective product weight has a profitable feasible direction.
Source: arXiv:2309.02390v1, section 3's product learning, contrasted with
appendix D's allocation argument. No rate choice or finite horizon cures it. -/
theorem subweight_gd_absent_gen_forever (remaining : ℕ)
    (multiplier firstCost secondCost exponent rate : ℝ) (initial : (ℝ × ℝ) × (ℝ × ℝ))
    (h0 : initial.1 = (0, 0)) (n : ℕ) :
    (subweightGDPath remaining multiplier firstCost secondCost exponent rate initial n).1 = (0, 0) := by
  induction n with
  | zero => exact h0
  | succ n ih =>
    exact subweight_gd_preserves_absent_gen remaining multiplier firstCost secondCost exponent rate _ ih

example : (((0 : ℝ), (0 : ℝ)), ((0 : ℝ), (1 : ℝ))).1 = (0, 0) := by rfl

/-- The paper's 113-class seeds and rate give an exact factor 40000
between the first products. Source: arXiv:2309.02390v1, appendix C,
Table simulation hyperparameters, independent of its penalty normalization. -/
theorem source_initial_gd_products (multiplier firstCost secondCost : ℝ) :
    let next := subweightGDStep 111 multiplier firstCost secondCost (5 / 3) (1 / 100)
      ((0, 1 / 200), (0, 1))
    (next.1.1 * next.1.2, next.2.1 * next.2.2) = (7 / 28250000, 28 / 2825) := by
  dsimp only
  rw [subweight_gd_initial_products 111 multiplier firstCost secondCost (5 / 3) (1 / 100)
    (1 / 200) 1 (by norm_num)]
  norm_num

end Transformer.Grokking.CircuitEfficiency
