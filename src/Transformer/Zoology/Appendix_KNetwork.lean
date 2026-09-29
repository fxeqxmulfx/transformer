/-
# Stacking K-constrained Coyote layers

Arora et al., arXiv:2312.04927v1, Appendix `def: W-kmat`,
Lemma `lem: stacking-layers`, and Theorem `thm: gen-ac`. The theoretical
model uses K-matrix projections in every layer. This module gives stacks
of such layers an exact functional and parameter-count semantics.
-/

import Transformer.Zoology.Appendix_KCoyoteGates
import Transformer.Zoology.Appendix_Network

namespace Transformer.Zoology

/-- A stack of Coyote layers all using the same feature width and
expanded K-matrix class. Source: Appendix `def: W-kmat` and
Lemma `lem: stacking-layers`. -/
structure KCoyoteNetwork (n k e : ℕ) where
  layers : List (KCoyoteParameters n k e)

/-- Execute each K-constrained layer in list order.
Source: Appendix Lemma `lem: stacking-layers`. -/
def KCoyoteNetwork.run {n k e : ℕ} (net : KCoyoteNetwork n k e)
    (u : RealSequence n (butterflyWidth k)) :
    RealSequence n (butterflyWidth k) :=
  net.layers.foldl (fun state p => coyoteLayer p.toParameters state) u

/-- Exact depth of a K-constrained Coyote stack.
Source: Appendix Lemma `lem: stacking-layers`. -/
def KCoyoteNetwork.layerCount {n k e : ℕ} (net : KCoyoteNetwork n k e) : ℕ :=
  net.layers.length

/-- Sum of the stored K-factor, filter, and bias coefficients across all
layers. Source: Appendix Proposition `prop: single-baseconv`. -/
def KCoyoteNetwork.parameterCount {n k e : ℕ}
    (net : KCoyoteNetwork n k e) : ℕ :=
  (net.layers.map KCoyoteParameters.parameterCount).sum

/-- Concatenating K-constrained stacks composes their functions and adds
both layer count and parameter count exactly. Source: Appendix Lemma
`lem: stacking-layers`, K-constrained specialization. -/
theorem kCoyote_stacking {n k e : ℕ}
    (first second : KCoyoteNetwork n k e)
    (u : RealSequence n (butterflyWidth k)) :
    ({layers := first.layers ++ second.layers} : KCoyoteNetwork n k e).run u =
        second.run (first.run u) ∧
      ({layers := first.layers ++ second.layers} : KCoyoteNetwork n k e).layerCount =
        first.layerCount + second.layerCount ∧
      ({layers := first.layers ++ second.layers} : KCoyoteNetwork n k e).parameterCount =
        first.parameterCount + second.parameterCount := by
  constructor
  · simp [KCoyoteNetwork.run, List.foldl_append]
  constructor
  · simp [KCoyoteNetwork.layerCount]
  · simp [KCoyoteNetwork.parameterCount, List.map_append,
      List.sum_append]

/-- Forget the K representations while preserving the layer sequence.
Source: Appendix `def: W-kmat`, Coyote model semantics. -/
def KCoyoteNetwork.toNetwork {n k e : ℕ} (net : KCoyoteNetwork n k e) :
    CoyoteNetwork n (butterflyWidth k) :=
  {layers := net.layers.map KCoyoteParameters.toParameters}

/-- Decoding the K weights does not change the stack's output.
Source: Appendix `def: W-kmat` and Lemma `lem: stacking-layers`. -/
theorem KCoyoteNetwork.run_toNetwork {n k e : ℕ}
    (net : KCoyoteNetwork n k e)
    (u : RealSequence n (butterflyWidth k)) :
    net.toNetwork.run u = net.run u := by
  simp [KCoyoteNetwork.toNetwork, CoyoteNetwork.run, KCoyoteNetwork.run,
    List.foldl_map]

/-- Every two-feature ordinary Coyote stack has a K-constrained encoding
with identical behavior. Source: Appendix `def: W-kmat`, size-two case. -/
def liftTwoFeatureNetwork {n : ℕ} (net : CoyoteNetwork n 2) :
    KCoyoteNetwork n 1 0 :=
  {layers := net.layers.map twoFeatureKParameters}

/-- Encoding an entire two-feature stack preserves its output, layer count,
and uses one `BB*` factor per layer. Source: Appendix
Lemma `lem: stacking-layers` and `def: W-kmat`. -/
theorem liftTwoFeatureNetwork_correct {n : ℕ}
    (net : CoyoteNetwork n 2) (u : RealSequence n 2) :
    (liftTwoFeatureNetwork net).run u = net.run u ∧
      (liftTwoFeatureNetwork net).layerCount = net.layerCount := by
  constructor
  · rcases net with ⟨layers⟩
    change (layers.map twoFeatureKParameters).foldl
      (fun state p => coyoteLayer p.toParameters state) u =
      layers.foldl (fun state p => coyoteLayer p state) u
    induction layers generalizing u with
    | nil => rfl
    | cons p rest ih =>
        simp only [List.map_cons, List.foldl_cons]
        rw [twoFeatureKParameters_toParameters]
        exact ih _
  · simp [liftTwoFeatureNetwork, KCoyoteNetwork.layerCount,
      CoyoteNetwork.layerCount]

end Transformer.Zoology
