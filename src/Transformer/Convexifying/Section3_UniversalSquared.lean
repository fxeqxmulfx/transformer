/-
# Squared-loss targets force an affine decoder

Ergen, Neyshabur, Mehta, arXiv:2211.11052v1, §2–§3.1, equation
`eq:attention_only_obj`.  For a convex regularizer, convexity of the
training objective for every squared-loss target forces the decoder to
preserve convex combinations.  This closes the nonlinear-decoder route to
an exact universal reformulation under the prediction-budget criterion.
-/

import Transformer.Convexifying.Section3_UniversalLoss
import Transformer.Convexifying.Section3_ScalarCounterexample

namespace Transformer.Convexifying

/-- The same decoder-affinity conclusion follows if every squared-loss
target, rather than every affine loss, gives a convex objective.  Squared
losses are explicitly included in the paper's allowed class.  Varying the
target produces arbitrary linear slopes after expanding the square.
Source: arXiv:2211.11052v1, `eq:attention_only_obj`. -/
theorem decoder_jensen_of_all_square_targets {E : Type*} [AddCommGroup E] [Module ℝ E]
    (C : Set E) (decode penalty : E → ℝ)
    (hall : ∀ target : ℝ,
      ConvexOn ℝ C (fun z => penalty z + (decode z - target) ^ 2))
    {x y : E} (hx : x ∈ C) (hy : y ∈ C)
    {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : a + b = 1) :
    decode (a • x + b • y) = a * decode x + b * decode y := by
  let r' : E → ℝ := fun z => penalty z + decode z ^ 2
  have hall' : ∀ M : ℝ, ConvexOn ℝ C (fun z => r' z + M * decode z) := by
    intro M
    let target : ℝ := -M / 2
    have h := hall target
    constructor
    · exact h.1
    · intro u hu v hv a b ha hb hab
      have hineq := h.2 hu hv ha hb hab
      change penalty (a • u + b • v) +
          (decode (a • u + b • v) - target) ^ 2 ≤
        a * (penalty u + (decode u - target) ^ 2) +
        b * (penalty v + (decode v - target) ^ 2) at hineq
      have hrewrite (w : E) :
          penalty w + (decode w - target) ^ 2 =
            r' w + M * decode w + target ^ 2 := by
        dsimp [r', target]
        ring
      rw [hrewrite (a • u + b • v), hrewrite u, hrewrite v] at hineq
      have hconst : a * target ^ 2 + b * target ^ 2 = target ^ 2 := by
        rw [← add_mul, hab, one_mul]
      change r' (a • u + b • v) + M * decode (a • u + b • v) ≤
        a * (r' u + M * decode u) + b * (r' v + M * decode v)
      nlinarith [hconst]
  exact decoder_jensen_of_all_affine_losses C decode r' hall' hx hy ha hb hab

/-- If a changed regularizer is convex and the decoded squared-loss
training objective is convex for every target, its prediction-budget
epigraph is convex.  Source: `eq:attention_only_obj`, abstract candidate. -/
theorem decodedEpigraph_convex_of_all_square_targets {E : Type*} [AddCommGroup E] [Module ℝ E]
    (C : Set E) (decode penalty : E → ℝ)
    (hpen : ConvexOn ℝ C penalty)
    (hall : ∀ target : ℝ,
      ConvexOn ℝ C (fun z => penalty z + (decode z - target) ^ 2)) :
    Convex ℝ (decodedEpigraph C decode penalty) := by
  apply convex_iff_forall_pos.mpr
  intro x hx y hy a b ha hb hab
  change (∃ z ∈ C, decode z = x.1 ∧ penalty z ≤ x.2) at hx
  change (∃ z ∈ C, decode z = y.1 ∧ penalty z ≤ y.2) at hy
  obtain ⟨u, hu, huout, hucost⟩ := hx
  obtain ⟨v, hv, hvout, hvcost⟩ := hy
  have hz : a • u + b • v ∈ C :=
    hpen.1 hu hv (le_of_lt ha) (le_of_lt hb) hab
  refine ⟨a • u + b • v, hz, ?_, ?_⟩
  · rw [decoder_jensen_of_all_square_targets C decode penalty hall
      hu hv (le_of_lt ha) (le_of_lt hb) hab, huout, hvout]
    simp
  · have hcost := hpen.2 hu hv (le_of_lt ha) (le_of_lt hb) hab
    have h₁ : 0 ≤ a * (x.2 - penalty u) :=
      mul_nonneg (le_of_lt ha) (sub_nonneg.mpr hucost)
    have h₂ : 0 ≤ b * (y.2 - penalty v) :=
      mul_nonneg (le_of_lt hb) (sub_nonneg.mpr hvcost)
    change penalty (a • u + b • v) ≤ a * x.2 + b * y.2
    have hcost' : penalty (a • u + b • v) ≤
        a * penalty u + b * penalty v := by simpa using hcost
    linarith

/-- **No exact convex training reformulation for all squared targets.**
The parameter regularizer may differ from the paper's four-matrix penalty
and the decoder may initially be nonlinear.  If the candidate regularizer
and every squared-target training objective are convex, exact equality of
the prediction-budget epigraphs is impossible.  Source:
arXiv:2211.11052v1, §2–§3.1, `eq:attention_only_obj`. -/
theorem no_universal_square_loss_lift {E : Type*} [AddCommGroup E] [Module ℝ E]
    (C : Set E) (decode penalty : E → ℝ)
    (hpen : ConvexOn ℝ C penalty)
    (hall : ∀ target : ℝ,
      ConvexOn ℝ C (fun z => penalty z + (decode z - target) ^ 2)) :
    decodedEpigraph C decode penalty ≠ originalMixtureEpigraph := by
  intro hEq
  have hconv := decodedEpigraph_convex_of_all_square_targets
    C decode penalty hpen hall
  rw [hEq] at hconv
  exact originalMixtureEpigraph_not_convex hconv

/-- Requiring convexity for every nonnegative weight on squared loss,
including weight zero, automatically gives convexity of the changed
regularizer.  Thus even allowing an arbitrary parameter penalty and a
nonlinear decoder cannot yield an exact epigraph reformulation valid for
this full family of convex losses.  Source: arXiv:2211.11052v1,
`eq:attention_only_obj`, arbitrary convex-loss hypothesis. -/
theorem no_universal_scaled_square_loss_lift {E : Type*} [AddCommGroup E] [Module ℝ E]
    (C : Set E) (decode penalty : E → ℝ)
    (hall : ∀ K : ℝ, 0 ≤ K → ∀ target : ℝ,
      ConvexOn ℝ C
        (fun z => penalty z + K * (decode z - target) ^ 2)) :
    decodedEpigraph C decode penalty ≠ originalMixtureEpigraph := by
  have hpen : ConvexOn ℝ C penalty := by
    simpa using hall 0 (by norm_num) 0
  have hsq : ∀ target : ℝ,
      ConvexOn ℝ C (fun z => penalty z + (decode z - target) ^ 2) := by
    intro target
    simpa using hall 1 (by norm_num) target
  exact no_universal_square_loss_lift C decode penalty hpen hsq

/-- The convex-regularizer and universal squared-loss hypotheses are
satisfiable with a linear decoder on a nonempty parameter space.
Source: arXiv:2211.11052v1, `eq:attention_only_obj`, squared-loss case. -/
example :
    ConvexOn ℝ (Set.univ : Set (Fin 1 → ℝ)) (fun _ => (0 : ℝ)) ∧
    (∀ target : ℝ, ConvexOn ℝ (Set.univ : Set (Fin 1 → ℝ))
      (fun z => (0 : ℝ) + (z 0 - target) ^ 2)) := by
  constructor
  · constructor
    · exact convex_univ
    · intro x _ y _ a b _ _ _
      simp
  · intro target
    constructor
    · exact convex_univ
    · intro x _ y _ a b ha hb hab
      have hb1 : b ≤ 1 := by linarith
      have haeq : a = 1 - b := by linarith
      rw [haeq]
      have h := squareLoss_convex (x 0) (y 0) target b hb hb1
      simpa [squareLoss, Pi.add_apply, Pi.smul_apply, smul_eq_mul] using h

/-- The scaled squared-loss family used in the no-go theorem is
satisfiable for a linear decoder on the whole parameter space.
Source: arXiv:2211.11052v1, `eq:attention_only_obj`, squared-loss case. -/
example : ∀ K : ℝ, 0 ≤ K → ∀ target : ℝ,
    ConvexOn ℝ (Set.univ : Set (Fin 1 → ℝ))
      (fun z => (0 : ℝ) + K * (z 0 - target) ^ 2) := by
  intro K hK target
  constructor
  · exact convex_univ
  · intro x _ y _ a b ha hb hab
    have hb1 : b ≤ 1 := by linarith
    have haeq : a = 1 - b := by linarith
    rw [haeq]
    have h := squareLoss_convex (x 0) (y 0) target b hb hb1
    have hscaled := mul_le_mul_of_nonneg_left h hK
    simpa [squareLoss, Pi.add_apply, Pi.smul_apply, smul_eq_mul,
      mul_add, mul_comm, mul_left_comm, mul_assoc] using hscaled

end Transformer.Convexifying
