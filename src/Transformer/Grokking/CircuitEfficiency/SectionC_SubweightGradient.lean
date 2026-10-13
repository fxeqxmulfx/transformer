import Transformer.Grokking.CircuitEfficiency.SectionD_CEBudgetBasic

/-!
# Gradients of the actual CE budget in product subweights

Source: Varma et al., arXiv:2309.02390v1, section 3, Slow vs fast
learning, and appendix C, sim-overall-logits and eq:dynamics. Each
circuit weight is a product of two trainable subweights. Derive all
four partial derivatives of the actual finite-class CE budget, rather
than assigning a stationary equation in product-weight coordinates.

The penalty coefficient remains explicit; appendix D uses alpha/p,
whereas appendix C's displayed LossWD suppresses that normalization.
The gradient formula is valid for r>=1, including zero products. At
r>1 and the source's one-zero-factor initialization, only the first
factor of each circuit receives a nonzero CE gradient. This models
fixed lookup tables and coupled regularization, not learned GPTMini
circuits or native AdamW's persistent moments.
-/

namespace Transformer.Grokking.CircuitEfficiency

/-- Marginal objective derivative in a circuit's product-weight
coordinate. Source: arXiv:2309.02390v1, appendices C–D; its derivative
status is proved below, with own and other both used in actual CE. -/
noncomputable def circuitMarginal (remaining : ℕ)
    (multiplier cost exponent own other : ℝ) : ℝ :=
  -((remaining : ℝ) + 1) / (Real.exp (own + other) + (remaining : ℝ) + 1) +
    multiplier * cost * exponent * own ^ (exponent - 1)

/-- The stated source objective in its four independently trainable
subweights. Source: arXiv:2309.02390v1, appendix C, sim-overall-logits. -/
noncomputable def subweightLoss (remaining : ℕ)
    (multiplier firstCost secondCost exponent a b c d : ℝ) : ℝ :=
  tableBudgetLoss remaining multiplier firstCost secondCost exponent (a * b) (c * d)

/-- Algebraic candidate for all four coordinate derivatives, justified
by the actual chain rule below. Source: arXiv:2309.02390v1, appendix C,
eq:dynamics; pairs group the Gen and Mem factors respectively. -/
noncomputable def subweightGradient (remaining : ℕ)
    (multiplier firstCost secondCost exponent a b c d : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  let gen := circuitMarginal remaining multiplier firstCost exponent (a * b) (c * d)
  let mem := circuitMarginal remaining multiplier secondCost exponent (c * d) (a * b)
  ((b * gen, a * gen), (d * mem, c * mem))

/-- Differentiate the actual objective in its first circuit weight.
Source: arXiv:2309.02390v1, appendix D's budget and appendix C's CE;
r>=1 permits this derivative also at a zero weight. -/
theorem table_budget_first_weight_deriv (remaining : ℕ)
    (multiplier firstCost secondCost exponent x y : ℝ) (hr : 1 ≤ exponent) :
    HasDerivAt (fun t : ℝ => tableBudgetLoss remaining multiplier firstCost secondCost exponent t y)
      (circuitMarginal remaining multiplier firstCost exponent x y) x := by
  have ht := (table_train_ce_deriv remaining (x + y)).comp x ((hasDerivAt_id x).add_const y)
  have hp := Real.hasDerivAt_rpow_const (x := x) (p := exponent) (Or.inr hr)
  have hc := ((hp.const_mul firstCost).add_const (secondCost * y ^ exponent)).const_mul multiplier
  convert ht.add hc using 1
  · funext t
    rfl
  · unfold circuitMarginal
    ring

example : (1 : ℝ) ≤ 5 / 3 := by norm_num

/-- The second marginal follows by swapping the actual costs and
weights. Source: arXiv:2309.02390v1, appendix D's symmetric objective. -/
theorem table_budget_second_weight_deriv (remaining : ℕ)
    (multiplier firstCost secondCost exponent x y : ℝ) (hr : 1 ≤ exponent) :
    HasDerivAt (fun t : ℝ => tableBudgetLoss remaining multiplier firstCost secondCost exponent x t)
      (circuitMarginal remaining multiplier secondCost exponent y x) y := by
  have hd := table_budget_first_weight_deriv remaining multiplier secondCost firstCost exponent y x hr
  convert hd using 1
  funext t
  exact table_budget_swap remaining multiplier firstCost secondCost exponent x t

example : (1 : ℝ) ≤ 5 / 3 := by norm_num

/-- The candidate is exactly the tuple of actual partial derivatives
of the source's product-parameter objective. Source: arXiv:2309.02390v1,
appendix C, eq:dynamics. No derivative or circuit-learning premise is
inserted into the definition of subweightLoss. -/
theorem subweight_gradient_eq_derivatives (remaining : ℕ)
    (multiplier firstCost secondCost exponent a b c d : ℝ) (hr : 1 ≤ exponent) :
    subweightGradient remaining multiplier firstCost secondCost exponent a b c d =
      ((deriv (fun t => subweightLoss remaining multiplier firstCost secondCost exponent t b c d) a,
        deriv (fun t => subweightLoss remaining multiplier firstCost secondCost exponent a t c d) b),
       (deriv (fun t => subweightLoss remaining multiplier firstCost secondCost exponent a b t d) c,
        deriv (fun t => subweightLoss remaining multiplier firstCost secondCost exponent a b c t) d)) := by
  have hg := table_budget_first_weight_deriv remaining multiplier firstCost secondCost exponent (a * b) (c * d) hr
  have hm := table_budget_second_weight_deriv remaining multiplier firstCost secondCost exponent (a * b) (c * d) hr
  have ha : HasDerivAt (fun t => subweightLoss remaining multiplier firstCost secondCost exponent t b c d)
      (b * circuitMarginal remaining multiplier firstCost exponent (a * b) (c * d)) a := by
    convert hg.comp a ((hasDerivAt_id a).mul_const b) using 1
    · funext t
      rfl
    · ring
  have hb : HasDerivAt (fun t => subweightLoss remaining multiplier firstCost secondCost exponent a t c d)
      (a * circuitMarginal remaining multiplier firstCost exponent (a * b) (c * d)) b := by
    convert hg.comp b ((hasDerivAt_id b).const_mul a) using 1
    · funext t
      rfl
    · ring
  have hc : HasDerivAt (fun t => subweightLoss remaining multiplier firstCost secondCost exponent a b t d)
      (d * circuitMarginal remaining multiplier secondCost exponent (c * d) (a * b)) c := by
    convert hm.comp c ((hasDerivAt_id c).mul_const d) using 1
    · funext t
      rfl
    · ring
  have hd : HasDerivAt (fun t => subweightLoss remaining multiplier firstCost secondCost exponent a b c t)
      (c * circuitMarginal remaining multiplier secondCost exponent (c * d) (a * b)) d := by
    convert hm.comp d ((hasDerivAt_id d).const_mul c) using 1
    · funext t
      rfl
    · ring
  rw [ha.deriv, hb.deriv, hc.deriv, hd.deriv]
  rfl

example : (1 : ℝ) ≤ 5 / 3 := by norm_num

/-- The zero-product marginal is the negative initial CE slope for
r>1. Source: arXiv:2309.02390v1, appendix C's tied initialization;
finite costs have zero first-order penalty here. -/
theorem circuit_marginal_initial (remaining : ℕ)
    (multiplier cost exponent : ℝ) (hr : 1 < exponent) :
    circuitMarginal remaining multiplier cost exponent 0 0 =
      -((remaining : ℝ) + 1) / ((remaining : ℝ) + 2) := by
  have he : exponent - 1 ≠ 0 := by linarith
  simp only [circuitMarginal, add_zero, Real.exp_zero, Real.zero_rpow he, mul_zero]
  congr 1
  ring

example : (1 : ℝ) < 5 / 3 := by norm_num

/-- Initial gradients are linear in the two seeded second factors.
Source: arXiv:2309.02390v1, section 3, Slow vs fast learning, and
appendix C's Table. This is derived from actual CE, not an assigned
learning-speed parameter. The two unseeded first factors receive it. -/
theorem subweight_gradient_initial (remaining : ℕ)
    (multiplier firstCost secondCost exponent genSeed memSeed : ℝ) (hr : 1 < exponent) :
    subweightGradient remaining multiplier firstCost secondCost exponent 0 genSeed 0 memSeed =
      ((-genSeed * (((remaining : ℝ) + 1) / ((remaining : ℝ) + 2)), 0),
       (-memSeed * (((remaining : ℝ) + 1) / ((remaining : ℝ) + 2)), 0)) := by
  simp only [subweightGradient, zero_mul, circuit_marginal_initial remaining multiplier firstCost exponent hr,
    circuit_marginal_initial remaining multiplier secondCost exponent hr]
  congr 2 <;> ring

example : (1 : ℝ) < 5 / 3 := by norm_num

/-- At four zero factors all actual coordinate derivatives vanish,
despite the negative CE derivative in a product-weight coordinate.
Source: arXiv:2309.02390v1, section 3's product parameterization;
this diagnoses why an equilibrium result does not imply learning. -/
theorem all_zero_subweight_derivatives (remaining : ℕ)
    (multiplier firstCost secondCost exponent : ℝ) (hr : 1 ≤ exponent) :
    ((deriv (fun t => subweightLoss remaining multiplier firstCost secondCost exponent t 0 0 0) 0,
      deriv (fun t => subweightLoss remaining multiplier firstCost secondCost exponent 0 t 0 0) 0),
     (deriv (fun t => subweightLoss remaining multiplier firstCost secondCost exponent 0 0 t 0) 0,
      deriv (fun t => subweightLoss remaining multiplier firstCost secondCost exponent 0 0 0 t) 0)) =
      ((0, 0), (0, 0)) := by
  rw [← subweight_gradient_eq_derivatives remaining multiplier firstCost secondCost exponent 0 0 0 0 hr]
  simp [subweightGradient]

example : (1 : ℝ) ≤ 5 / 3 := by norm_num

end Transformer.Grokking.CircuitEfficiency
