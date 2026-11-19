module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Estimate.Witness
public import Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Examples

/-!
# A feasible minibatch improvement can worsen the complete training mean

Source: `review/convex/moments.md`, CM105. The same full learned ordered family,
two distinct prefixes and capacity sixteen remain. A halfway step retains
imaginary coordinates, lowers actual CE on training index zero and raises the
actual complete mean. The conclusion is an optimizer inference counterexample.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Estimate

open Pair Optimality

/-- Complete capacity-sixteen bias toward true on the retained constant state, CM105. -/
noncomputable def biasTable (j c : Bool) : State :=
  (if j then (0 : ℝ) else 16) • (if c then ((1, 0) : State) else (-1, 0))

/-- The complete bias table belongs to the same shared-capacity domain, CM105. -/
theorem bias_mem : biasTable ∈ jointDomain 16 :=
  weighted_mem _ _ _ (by intro j; cases j <;> norm_num) (by norm_num)
    (by intro j c; cases c <;> simp [rowSize])

/-- Actual full class logits of the feasible bias table on every prefix, CM105. -/
theorem bias_scores (xs : List Bool) (c : Bool) :
    jointScore allocationModel biasTable xs c = if c then 16 else -16 := by
  rw [jointScore, scores_pairing, Fintype.sum_bool]
  cases c <;> norm_num [biasTable, allocation_constant, pairing, Pair.row_apply]

/-- A genuine finite convex step from the full imaginary parked table, CM105. -/
noncomputable def halfway : Bool → Bool → State :=
  (1 / 2 : ℝ) • parked + (1 / 2 : ℝ) • biasTable

/-- Convexity proves feasibility of the actual finite update, CM105. -/
theorem halfway_mem : halfway ∈ jointDomain 16 :=
  joint_convex 16 parked_mem bias_mem (by norm_num) (by norm_num) (by norm_num)

/-- Both original class rows retain a nonzero imaginary coordinate after the
finite update, CM105; feasibility does not come from replacing them by real rows. -/
theorem halfway_imaginary (c : Bool) : (halfway false c).2.im = 8 := by
  cases c <;> norm_num [halfway, parked, allocationRows, biasTable]

/-- The interpolated complete table has actual scores eight and minus eight,
CM105; its nonzero imaginary rows are retained. -/
theorem halfway_scores (xs : List Bool) (c : Bool) :
    jointScore allocationModel halfway xs c = if c then 8 else -8 := by
  rw [halfway, map_add, map_smul, map_smul]
  cases c <;> norm_num [parked_scores, bias_scores]

/-- The actual true-class probability increases on the selected minibatch,
retaining the complete two-class denominator, CM105. -/
theorem halfway_probability (xs : List Bool) :
    Geometry.probability (jointScore allocationModel parked xs) true <
      Geometry.probability (jointScore allocationModel halfway xs) true := by
  have he : Real.exp (-8) < Real.exp 8 := Real.exp_lt_exp.mpr (by norm_num)
  have hp : (1 / 2 : ℝ) < Real.exp 8 / (Real.exp 8 + Real.exp (-8)) := by
    apply (div_lt_div_iff₀ (by norm_num) (by positivity)).mpr
    linarith
  simpa [Geometry.probability, parked_scores, halfway_scores,
    one_add_one_eq_two, add_comm] using hp

/-- Actual ordinary CE decreases on each true-target prefix, CM105. -/
theorem halfway_true_ce (xs : List Bool) :
    Landscape.Neural.crossEntropy (jointScore allocationModel halfway xs) true <
      Landscape.Neural.crossEntropy (jointScore allocationModel parked xs) true := by
  rw [Geometry.crossEntropy_eq_neg_log_probability,
    Geometry.crossEntropy_eq_neg_log_probability]
  have h := Real.log_lt_log (Geometry.probability_pos _ _) (halfway_probability xs)
  linarith

/-- Expand the actual two-record training mean without dropping either record, CM105. -/
theorem two_mean (u : Bool → Bool → State) :
    jointLoss 1 allocationModel allocationPrefix allocationTarget u = (1 / 2 : ℝ) *
      (Landscape.Neural.crossEntropy (jointScore allocationModel u [true, false]) true +
        Landscape.Neural.crossEntropy (jointScore allocationModel u [false, true]) false) := by
  unfold jointLoss meanLoss
  rw [Fin.sum_univ_two]
  norm_num only [Nat.cast_one]
  rfl

/-- Complete training CE at the original zero-logit table is log two, CM105. -/
theorem parked_mean : jointLoss 1 allocationModel allocationPrefix allocationTarget parked =
    Real.log 2 := by
  rw [two_mean]
  norm_num [Landscape.Neural.crossEntropy, parked_scores]
  ring

/-- The complete mean after the minibatch-improving step retains both
opposite targets and equals the symmetric log-sum-exp, CM105. -/
theorem halfway_mean : jointLoss 1 allocationModel allocationPrefix allocationTarget halfway =
    Real.log (Real.exp 8 + Real.exp (-8)) := by
  rw [two_mean]
  simp only [Landscape.Neural.crossEntropy, halfway_scores, Fintype.sum_bool]
  norm_num
  ring

/-- Every strictly positive contrast raises the symmetric exponential sum
above its zero-contrast value, CM105's full-mean comparison. -/
theorem exponential_separation (t : ℝ) (ht : 0 < t) : 2 < Real.exp t + Real.exp (-t) := by
  have h₁ : 1 < Real.exp t := Real.one_lt_exp_iff.mpr ht
  have hs := sq_pos_of_pos (sub_pos.mpr h₁)
  have hm : Real.exp t * Real.exp (-t) = 1 := by rw [← Real.exp_add]; simp
  by_contra hn
  have hle : Real.exp t + Real.exp (-t) ≤ 2 := le_of_not_gt hn
  have hp := mul_le_mul_of_nonneg_left hle (Real.exp_pos t).le
  nlinarith

/-- The halfway step increases actual complete training CE, despite its
actual CE improvement on the selected original minibatch, CM105. -/
theorem halfway_training_worsens :
    jointLoss 1 allocationModel allocationPrefix allocationTarget parked <
      jointLoss 1 allocationModel allocationPrefix allocationTarget halfway := by
  rw [parked_mean, halfway_mean]
  exact Real.log_lt_log (by norm_num) (exponential_separation 8 (by norm_num))

/-- The minibatch is explicitly training index zero; its finite feasible
step lowers its actual mean CE, CM105. -/
theorem halfway_minibatch_improves : jointLoss 0 allocationModel
    (fun _ => allocationPrefix 0) (fun _ => allocationTarget 0) halfway <
      jointLoss 0 allocationModel (fun _ => allocationPrefix 0)
        (fun _ => allocationTarget 0) parked := by
  simpa [jointLoss, meanLoss, jointScore, allocationPrefix, allocationTarget] using
    halfway_true_ce [true, false]

/-- CM105 counterexample: a genuine finite feasible minibatch improvement
does not imply improvement of the complete indexed training mean. -/
theorem minibatch_improvement_not_training_improvement : parked ∈ jointDomain 16 ∧
    halfway ∈ jointDomain 16 ∧
    jointLoss 0 allocationModel (fun _ => allocationPrefix 0) (fun _ => allocationTarget 0) halfway <
      jointLoss 0 allocationModel (fun _ => allocationPrefix 0) (fun _ => allocationTarget 0) parked ∧
    jointLoss 1 allocationModel allocationPrefix allocationTarget parked <
      jointLoss 1 allocationModel allocationPrefix allocationTarget halfway :=
  ⟨parked_mem, halfway_mem, halfway_minibatch_improves, halfway_training_worsens⟩

example : 2 < Real.exp 8 + Real.exp (-8) := exponential_separation _ (by norm_num)

end Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Estimate
