/-
# Exchange a row bit with a feature bit without changing dimensions

Arora et al., arXiv:2312.04927v1, Appendix `prop: butterfly-hyena`.
Two opposite channel shifts around a conditional feature swap exchange
one row bit and one feature bit. The resulting five-layer permutation is
its own inverse and uses exactly the original n × d coordinates.
-/

import Transformer.Zoology.Appendix_ControlledFeatureSwap

noncomputable section

namespace Transformer.Zoology

/-- A negative half-block shift from an upper row reaches a lower row.
Source: Appendix `eq: butterfly-split`, cyclic half-block offsets. -/
theorem butterflyToggleUpper_sub_offset (p : ℕ) (t : Fin p)
    (i : Fin (butterflyWidth p)) (hi : butterflyToggleUpper p t.val i = true) :
    butterflyToggleUpper p t.val (i - butterflyToggleOffset p t) = false := by
  cases h : butterflyToggleUpper p t.val (i - butterflyToggleOffset p t)
  · rfl
  · have hp := butterflyToggleIndex_shift p t (i - butterflyToggleOffset p t)
    simp only [h, ite_true, sub_add_cancel] at hp
    have ho := butterflyToggleUpper_toggle p t (i - butterflyToggleOffset p t)
    rw [hp, hi, h] at ho
    cases ho

/-- Exchange two binary digits by toggling both exactly when they differ.
Source: Appendix `prop: butterfly-hyena`, in-place layout permutation. -/
def rowFeatureExchange {p k : ℕ} (r : Fin p) (f : Fin k)
    (u : RealSequence (butterflyWidth p) (butterflyWidth k)) :
    RealSequence (butterflyWidth p) (butterflyWidth k) :=
  fun i q => if butterflyToggleUpper p r.val i = butterflyToggleUpper k f.val q then
    u i q else u (butterflyToggleIndex p r.val i) (butterflyToggleIndex k f.val q)

/-- The bit exchange is an exact involutive permutation.
Source: Appendix `prop: butterfly-hyena`, reversible layout change. -/
theorem rowFeatureExchange_involutive {p k : ℕ} (r : Fin p) (f : Fin k)
    (u : RealSequence (butterflyWidth p) (butterflyWidth k)) :
    rowFeatureExchange r f (rowFeatureExchange r f u) = u := by
  funext i q
  cases hr : butterflyToggleUpper p r.val i <;>
    cases hf : butterflyToggleUpper k f.val q <;>
      simp [rowFeatureExchange, hr, hf, butterflyToggleUpper_toggle,
        butterflyToggleIndex_involutive p r.val i,
        butterflyToggleIndex_involutive k f.val q]

/-- Five actual K layers for the reversible bit exchange.
Source: Appendix `lmm:primitives`, shifts and conditional feature mixing. -/
def rowFeatureExchangeNetwork {p k : ℕ} (r : Fin p) (f : Fin k) :
    CyclicKCoyoteNetwork (butterflyWidth p) k 1 :=
  (({layers := [inPlaceShiftParameters (fun q =>
      if butterflyToggleUpper k f.val q then -butterflyToggleOffset p r else 0)]} :
        CyclicKCoyoteNetwork (butterflyWidth p) k 1).append
    (controlledFeatureSwapNetwork f (butterflyToggleUpper p r.val))).append
    {layers := [inPlaceShiftParameters (fun q =>
      if butterflyToggleUpper k f.val q then butterflyToggleOffset p r else 0)]}

/-- The two shifts and selected swaps realize the original bit exchange.
Source: Appendix `prop: butterfly-hyena`, exact routing without workspace. -/
theorem rowFeatureExchangeNetwork_correct {p k : ℕ} (r : Fin p) (f : Fin k)
    (u : RealSequence (butterflyWidth p) (butterflyWidth k)) :
    (rowFeatureExchangeNetwork r f).run u = rowFeatureExchange r f u := by
  change coyoteLayerCyclic (inPlaceShiftParameters (fun q =>
    if butterflyToggleUpper k f.val q then butterflyToggleOffset p r else 0)).toParameters
      ((controlledFeatureSwapNetwork f (butterflyToggleUpper p r.val)).run
        (coyoteLayerCyclic (inPlaceShiftParameters (fun q =>
          if butterflyToggleUpper k f.val q then -butterflyToggleOffset p r else 0)).toParameters u)) = _
  rw [inPlaceShift_correct, inPlaceShift_correct, controlledFeatureSwapNetwork_correct]
  funext i q
  cases hf : butterflyToggleUpper k f.val q <;>
    cases hr : butterflyToggleUpper p r.val i
  · simp [rowFeatureExchange, hf, hr]
  · have hp := butterflyToggleIndex_shift p r i
    simp [hr] at hp
    simp [rowFeatureExchange, hf, hr, butterflyToggleUpper_toggle, hp, sub_neg_eq_add]
  · have hp := butterflyToggleIndex_shift p r i
    simp [hr] at hp
    have ho := butterflyToggleUpper_toggle p r i
    simp only [hr, Bool.not_false] at ho
    rw [hp] at ho
    simp [rowFeatureExchange, hf, hr, butterflyToggleUpper_toggle, ho, hp]
  · have ho := butterflyToggleUpper_sub_offset p r i hr
    simp [rowFeatureExchange, hf, hr, ho, sub_neg_eq_add]

/-- The in-place exchange always takes exactly five layers.
Source: Appendix `prop: butterfly-hyena`, constant-depth layout routing. -/
theorem rowFeatureExchangeNetwork_layerCount {p k : ℕ} (r : Fin p) (f : Fin k) :
    (rowFeatureExchangeNetwork r f).layerCount = 5 := by
  rw [rowFeatureExchangeNetwork, CyclicKCoyoteNetwork.layerCount_append,
    CyclicKCoyoteNetwork.layerCount_append, controlledFeatureSwapNetwork_layerCount]
  rfl

/-- The upper-row premise has an actual two-row example.
Source: Appendix `eq: butterfly-split`, upper endpoint of the smallest block. -/
example : butterflyToggleUpper 1 0 0 = true := by decide

end Transformer.Zoology
