module

public import Transformer.Grokking.Protocol.AdamBox
public import Transformer.Grokking.Protocol.Division
public import Transformer.Grokking.Protocol.Target
public import Mathlib.MeasureTheory.Constructions.Pi
public import Mathlib.MeasureTheory.Measure.Lebesgue.Basic
public import Mathlib.Probability.ConditionalProbability
public import Mathlib.Probability.Distributions.Gaussian.Real

/-!
# The target for division modulo 97: what Figure 1 would mean as a theorem

Source: arXiv:2201.02177v1 (Power et al.), Figure 1 (left): "division mod 97 with 50% of the
data in the training set", where training accuracy is close to perfect before `10³` steps
and "it takes close to `10⁶` steps for validation accuracy to reach that level"; the appendix
*Additional experimental details*, by which that section has an optimization budget of `10⁶`
and "Adam optimizer with no weight decay"; openai/grok at 3d64b1d8;
`experiments/grok_division/PROTOCOL.md`. The paper reports curves of finite runs and states no
theorem. This file writes down the statement that an asymptotic reading of the figure would
be, with every choice of the protocol fixed, and proves only what that statement means. It
asserts nothing about its truth; `Refutation` proves it false as long as nothing is asked of
the split.

The initial parameters are independent (`initMeasure`): the embedding standard normal, every
linear map uniform on `[-1/√n, 1/√n]` for its `n` inputs (`uniformInit`), the gains `1` and
the biases `0` (`initLaw`); these are PyTorch's defaults, on which the paper is silent. For
the paper's sizes this is a probability law, and so is the law of a run
(`isProbabilityMeasure_initLaw` and the instances after it): "almost every run" is not empty.

A run draws a shuffle of the `9312` equations, of which the first `4656` are trained on, the
initial parameters, and a shuffle of the training equations for every epoch; update `t` sees
`divisionGrad` on batch `t` of at most `512` equations, under `Hyper.source decay` (`decay = 0`
is the Adam of Figure 1, `decay = 1` the AdamW of the paper's other experiments).

`division_generalizes_iff`: the held-out accuracy of the answer tends to `1` for almost every
run exactly when, for almost every run, from some update on the first of the largest of the
239 logits after `=` is the answer's token for every held-out equation.
`division_generalizes_source_iff`: the same for the source's accuracy, which also asks for the
closing token.

What is proved about the run itself is only how far it can move a parameter: under Adam less
than `0.017 t` in `t` updates (`abs_trajectory_division_sub_le`), and under AdamW with weight
decay `1` never out of `[-17, 17]` once inside (`abs_trajectory_division_le`).

Deviations of this statement from the runs behind the figure: exact reals in place of `f32`;
a uniformly random split in place of the source's constant seed; an unbounded number of
updates in place of the budget.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

open Filter MeasureTheory ProbabilityTheory

/-- The uniform law on `[-1/√n, 1/√n]`: PyTorch's default for a linear map with `n` inputs. -/
noncomputable def uniformInit (n : ℕ) : Measure ℝ := cond volume (Set.Icc (-(1 / √n)) (1 / √n))

lemma isProbabilityMeasure_uniformInit {n : ℕ} (hn : 0 < n) :
    IsProbabilityMeasure (uniformInit n) := by
  have hpos : (0 : ℝ) < 1 / √n := by positivity
  refine cond_isProbabilityMeasure_of_finite ?_ ?_
  · rw [Real.volume_Icc]
    exact (ENNReal.ofReal_pos.2 (by linarith)).ne'
  · rw [Real.volume_Icc]
    exact ENNReal.ofReal_ne_top

/-- **The law of one initial parameter.** -/
noncomputable def initLaw (S : Sizes) : Param S → Measure ℝ
  | .emb _ _ => gaussianReal 0 1
  | .query _ _ _ _ => uniformInit S.width
  | .key _ _ _ _ => uniformInit S.width
  | .value _ _ _ _ => uniformInit S.width
  | .proj _ _ _ _ => uniformInit (S.heads * S.headWidth)
  | .gain1 _ _ => Measure.dirac 1
  | .bias1 _ _ => Measure.dirac 0
  | .up _ _ _ => uniformInit S.width
  | .down _ _ _ => uniformInit S.hidden
  | .gain2 _ _ => Measure.dirac 1
  | .bias2 _ _ => Measure.dirac 0
  | .out _ _ => uniformInit S.width

/-- **The law of the initial parameters**: independent coordinates. -/
noncomputable def initMeasure (S : Sizes) : Measure (Param S → ℝ) := Measure.pi (initLaw S)

/-- **Every initial parameter has a probability law** when no linear map has zero inputs. -/
theorem isProbabilityMeasure_initLaw {S : Sizes} (hw : 0 < S.width)
    (hp : 0 < S.heads * S.headWidth) (hh : 0 < S.hidden) (p : Param S) :
    IsProbabilityMeasure (initLaw S p) := by
  cases p <;> simp only [initLaw] <;>
    first
      | infer_instance
      | exact isProbabilityMeasure_uniformInit hw
      | exact isProbabilityMeasure_uniformInit hp
      | exact isProbabilityMeasure_uniformInit hh

instance (p : Param paperSizes) : IsProbabilityMeasure (initLaw paperSizes p) :=
  isProbabilityMeasure_initLaw (by decide) (by decide) (by decide) p

instance : IsProbabilityMeasure (initMeasure paperSizes) := by
  unfold initMeasure
  infer_instance

/-- **The law of a run of the division protocol is a probability law.** -/
example : IsProbabilityMeasure (law (97 * 96) 4656 (initMeasure paperSizes)) := inferInstance

/-- **What the target of Figure 1 says.** With weight decay `decay` (`0` in the figure), the
held-out accuracy of the answer tends to `1` for almost every run of the protocol if and only
if, for almost every run, from some update on every held-out equation is answered. -/
theorem division_generalizes_iff (decay : ℝ) :
    Generalizes (initMeasure paperSizes) answerLogits (fun x => residueToken x.1)
        (Hyper.source decay) divisionGrad equations (by norm_num : 4656 ≤ 97 * 96) 512 ↔
      ∀ᵐ ω ∂law (97 * 96) 4656 (initMeasure paperSizes), ∀ᶠ t in atTop,
        ∀ x ∈ heldout equations 4656 ω.1,
          predict (answerLogits (trajectory (Hyper.source decay) divisionGrad equations
            (by norm_num : 4656 ≤ 97 * 96) 512 ω t) x) = residueToken x.1 :=
  generalizes_iff (by norm_num)

/-- **The target for the source's accuracy**, which counts an equation only if the answer and
the closing token are both predicted: both accuracies tend to `1` for almost every run if and
only if, for almost every run, from some update on both tokens of every held-out equation are
predicted. -/
theorem division_generalizes_source_iff (decay : ℝ) :
    (Generalizes (initMeasure paperSizes) answerLogits (fun x => residueToken x.1)
        (Hyper.source decay) divisionGrad equations (by norm_num : 4656 ≤ 97 * 96) 512 ∧
      Generalizes (initMeasure paperSizes) closingLogits (fun _ => eosToken)
        (Hyper.source decay) divisionGrad equations (by norm_num : 4656 ≤ 97 * 96) 512) ↔
      ∀ᵐ ω ∂law (97 * 96) 4656 (initMeasure paperSizes), ∀ᶠ t in atTop,
        ∀ x ∈ heldout equations 4656 ω.1,
          predict (answerLogits (trajectory (Hyper.source decay) divisionGrad equations
            (by norm_num : 4656 ≤ 97 * 96) 512 ω t) x) = residueToken x.1 ∧
          predict (closingLogits (trajectory (Hyper.source decay) divisionGrad equations
            (by norm_num : 4656 ≤ 97 * 96) 512 ω t) x) = eosToken := by
  rw [generalizes_iff (by norm_num), generalizes_iff (by norm_num), ← eventually_and]
  refine eventually_congr (Eventually.of_forall fun ω => ?_)
  rw [← eventually_and]
  exact eventually_congr (Eventually.of_forall fun t => forall₂_and.symm)

/-- **The Adam run of Figure 1 moves every parameter at most linearly**: after `t` updates of
any run, a parameter is at most `0.017 t` from its initial value. -/
theorem abs_trajectory_division_sub_le (ω : Sample (97 * 96) 4656 (Param paperSizes)) (t : ℕ)
    (p : Param paperSizes) :
    |trajectory (Hyper.source 0) divisionGrad equations (by norm_num : 4656 ≤ 97 * 96) 512 ω t p -
        ω.2.1 p| ≤ 17 / 1000 * t := by
  have h := abs_adamRun_params_sub_le (h := Hyper.source 0)
    (grad := fun s θ => divisionGrad ((batchSlots 4656 512 ω.2.2 s).map
      (trainAt equations (by norm_num : 4656 ≤ 97 * 96) ω.1)) θ) (θ₀ := ω.2.1)
    (by norm_num [Hyper.source]) (by norm_num [Hyper.source]) (by norm_num [Hyper.source])
    (by norm_num [Hyper.source]) (by norm_num [Hyper.source]) (by norm_num [Hyper.source])
    (by norm_num [Hyper.source]) rfl p t
  have hs := (stepBound_source_lt 0).le
  have hr : (Hyper.source 0).rate = 1 / 1000 := rfl
  rw [hr] at h
  refine h.trans ?_
  nlinarith [Nat.cast_nonneg (α := ℝ) t]

/-- **The AdamW run keeps every parameter in a fixed interval**: with weight decay `1`, within
`max(|θ₀|, 17)` of `0` at every update of every run. -/
theorem abs_trajectory_division_le (ω : Sample (97 * 96) 4656 (Param paperSizes)) (t : ℕ)
    (p : Param paperSizes) :
    |trajectory (Hyper.source 1) divisionGrad equations (by norm_num : 4656 ≤ 97 * 96) 512 ω t p| ≤
      max |ω.2.1 p| 17 := by
  have h := abs_adamRun_params_le (h := Hyper.source 1)
    (grad := fun s θ => divisionGrad ((batchSlots 4656 512 ω.2.2 s).map
      (trainAt equations (by norm_num : 4656 ≤ 97 * 96) ω.1)) θ) (θ₀ := ω.2.1)
    (by norm_num [Hyper.source]) (by norm_num [Hyper.source]) (by norm_num [Hyper.source])
    (by norm_num [Hyper.source]) (by norm_num [Hyper.source]) (by norm_num [Hyper.source])
    (by norm_num [Hyper.source]) (by norm_num [Hyper.source]) (by norm_num [Hyper.source]) p t
  have hd : (Hyper.source 1).decay = 1 := rfl
  rw [hd, div_one] at h
  exact h.trans (max_le_max le_rfl (stepBound_source_lt 1).le)

/-! The witnesses: the paper's sizes for the hypotheses of `isProbabilityMeasure_initLaw`, and
the law of a linear map with `128` inputs. -/

example := isProbabilityMeasure_initLaw (S := paperSizes) (by decide) (by decide) (by decide)
  (.emb 0 0)

example : IsProbabilityMeasure (uniformInit 128) := isProbabilityMeasure_uniformInit (by norm_num)

end Transformer.Grokking.Protocol
