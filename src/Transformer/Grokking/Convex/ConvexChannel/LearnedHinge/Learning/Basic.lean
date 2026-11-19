module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Projection
public import Transformer.Grokking.Convex.ConvexDepth.RowCE

/-!
# Ordinary CE slopes of the actual learned complex sequence path

Source: `review/convex/moments.md`, CM82. On the complete activation domain,
the change of actual recovered complex logits has a derived linear expression.
The ordinary full-class CE first-order inequality then justifies its slope.
This training algebra does not replace the actual forward outside the domain.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Learning

open scoped BigOperators
open Landscape.Neural (crossEntropy)
open FlatSpectrum

variable {E X J C : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] [Fintype J]

/-- The complete direction of the two actual logit components on their
declared branches, CM82. A direction itself need not be a feasible model. -/
noncomputable def logitDirection (input : E →ₗ[ℝ] ℂ) (mask : J → Bool → Bool)
    (v : Parameters J C E) : C → ℝ :=
  gatedForward (fun j => mask j true) (Complex.reLm.comp input) v -
    gatedForward (fun j => mask j false) (Complex.imLm.comp input) v

/-- The derived direction is linear in all joint coordinates, CM82.
Both readout signs and both complex components remain present. -/
theorem logitDirection_combination (input : E →ₗ[ℝ] ℂ) (mask : J → Bool → Bool)
    (p q : Parameters J C E) (a b : ℝ) :
    logitDirection input mask (a • p + b • q) =
      a • logitDirection input mask p + b • logitDirection input mask q := by
  unfold logitDirection
  rw [gatedForward_combination, gatedForward_combination]
  funext c
  simp only [Pi.sub_apply, Pi.add_apply, Pi.smul_apply, smul_eq_mul]
  ring

/-- Actual complex logits equal the full derived expression only with
the original input-region and parameter-feasibility premises, CM82. -/
theorem logits_on_domain (input : X → E →ₗ[ℝ] ℂ) (region : Set X)
    (mask : J → X × Bool → Bool) {p : Parameters J C E}
    (hp : p ∈ complexDomain input region mask) {x : X} (hx : x ∈ region) :
    complexLogits p (input x) = logitDirection (input x) (fun j k => mask j (x, k)) p := by
  exact complexLogits_on_cone input region mask hp hx

/-- The change between any two feasible actual models is given by the
derived direction on their difference, CM82. No free-factor gradient is assumed. -/
theorem logits_difference (input : X → E →ₗ[ℝ] ℂ) (region : Set X)
    (mask : J → X × Bool → Bool) {p q : Parameters J C E}
    (hp : p ∈ complexDomain input region mask) (hq : q ∈ complexDomain input region mask)
    {x : X} (hx : x ∈ region) :
    complexLogits q (input x) - complexLogits p (input x) =
      logitDirection (input x) (fun j k => mask j (x, k)) (q - p) := by
  rw [logits_on_domain input region mask hp hx, logits_on_domain input region mask hq hx]
  have h := logitDirection_combination (input x) (fun j k => mask j (x, k)) q p 1 (-1)
  simpa only [one_smul, neg_one_smul, sub_eq_add_neg] using h.symm

/-- An actually feasible query has the expected actual complex logit
increment, CM82. Both points satisfy the unchanged complete domain. -/
theorem logits_shift (input : X → E →ₗ[ℝ] ℂ) (region : Set X)
    (mask : J → X × Bool → Bool) {p v : Parameters J C E} (amount : ℝ)
    (hp : p ∈ complexDomain input region mask)
    (hv : p + amount • v ∈ complexDomain input region mask)
    {x : X} (hx : x ∈ region) :
    complexLogits (p + amount • v) (input x) = complexLogits p (input x) +
      amount • logitDirection (input x) (fun j k => mask j (x, k)) v := by
  rw [logits_on_domain input region mask hv hx, logits_on_domain input region mask hp hx]
  simpa only [one_smul] using
    logitDirection_combination (input x) (fun j k => mask j (x, k)) p v 1 amount

variable [Fintype C]

/-- The ordinary CE slope in a full-class logit direction, CM82. The
target appears only in this loss derivative, never in model inference. -/
noncomputable def ceSlope (logits direction : C → ℝ) (target : C) : ℝ :=
  (∑ c, Geometry.probability logits c * direction c) - direction target

/-- The full softmax-minus-target coordinate formula equals the actual
ordinary CE slope, CM82, including every competing class. -/
theorem ceSlope_eq [DecidableEq C] (logits direction : C → ℝ) (target : C) :
    ceSlope logits direction target =
      ∑ c, ConvexDepth.rowCEGradient logits target c * direction c := by
  simp [ceSlope, ConvexDepth.rowCEGradient, sub_mul, Finset.sum_sub_distrib, ite_mul]

/-- The actual learned-model CE slope in every supplied joint direction,
CM82. Its use as a derivative will require a feasible two-sided neighbourhood. -/
noncomputable def complexSlope (input : E →ₗ[ℝ] ℂ) (mask : J → Bool → Bool)
    (target : C) (p v : Parameters J C E) : ℝ :=
  ceSlope (complexLogits p input) (logitDirection input mask v) target

/-- The derived slope supports the actual ordinary CE at every other
feasible point, CM82. This includes boundary points and is not a smoothness
or optimizer-convergence claim. -/
theorem complexLoss_first_order (input : X → E →ₗ[ℝ] ℂ) (region : Set X)
    (mask : J → X × Bool → Bool) {p q : Parameters J C E}
    (hp : p ∈ complexDomain input region mask) (hq : q ∈ complexDomain input region mask)
    {x : X} (hx : x ∈ region) (target : C) :
    complexLoss (input x) target p +
      complexSlope (input x) (fun j k => mask j (x, k)) target p (q - p) ≤
        complexLoss (input x) target q := by
  unfold complexLoss complexSlope ceSlope
  rw [← logits_difference input region mask hp hq hx]
  exact Training.crossEntropy_first_order _ _ target

end Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Learning
