/-
# Cyclic Coyote networks with genuine K-matrix weights

Arora et al., arXiv:2312.04927v1, Appendix `def: W-kmat`,
`def: kaleidoscope`, and `lem: stacking-layers`. Zero-weight convolution
and position-gating layers admit constant-width, constant-expansion
K representations. The cyclic option does not change parameter storage.
-/

import Transformer.Zoology.Appendix_CyclicCircuitNetwork
import Transformer.Zoology.Appendix_KCoyoteZero

namespace Transformer.Zoology

/-- A cyclic-convolution stack whose weights are stored as expanded
K-matrices. Source: Appendix `def: W-kmat` and `lem: stacking-layers`. -/
structure CyclicKCoyoteNetwork (n k e : ℕ) where
  layers : List (KCoyoteParameters n k e)

/-- Execute a K-constrained cyclic stack.
Source: Appendix `lem: stacking-layers`, cyclic option. -/
def CyclicKCoyoteNetwork.run {n k e : ℕ}
    (net : CyclicKCoyoteNetwork n k e)
    (u : RealSequence n (butterflyWidth k)) :
    RealSequence n (butterflyWidth k) :=
  net.layers.foldl (fun state p => coyoteLayerCyclic p.toParameters state) u

/-- Depth of a K-constrained cyclic stack.
Source: Appendix `lem: stacking-layers`. -/
def CyclicKCoyoteNetwork.layerCount {n k e : ℕ}
    (net : CyclicKCoyoteNetwork n k e) : ℕ := net.layers.length

/-- Count all stored K-factor coefficients, filters, and bias entries.
Source: Appendix `prop: single-baseconv`, actual stored parameters. -/
def CyclicKCoyoteNetwork.parameterCount {n k e : ℕ}
    (net : CyclicKCoyoteNetwork n k e) : ℕ :=
  (net.layers.map KCoyoteParameters.parameterCount).sum

/-- Decode every K-matrix weight to the corresponding ordinary cyclic
Coyote layer. Source: Appendix `def: W-kmat`. -/
def CyclicKCoyoteNetwork.toNetwork {n k e : ℕ}
    (net : CyclicKCoyoteNetwork n k e) :
    CyclicCoyoteNetwork n (butterflyWidth k) :=
  { layers := net.layers.map KCoyoteParameters.toParameters }

/-- Decoding K weights preserves both the function and depth.
Source: Appendix `def: W-kmat` and `lem: stacking-layers`. -/
theorem CyclicKCoyoteNetwork.toNetwork_correct {n k e : ℕ}
    (net : CyclicKCoyoteNetwork n k e)
    (u : RealSequence n (butterflyWidth k)) :
    net.toNetwork.run u = net.run u ∧
      net.toNetwork.layerCount = net.layerCount := by
  constructor
  · simp [CyclicKCoyoteNetwork.toNetwork, CyclicCoyoteNetwork.run,
      CyclicKCoyoteNetwork.run, List.foldl_map]
  · simp [CyclicKCoyoteNetwork.toNetwork, CyclicCoyoteNetwork.layerCount,
      CyclicKCoyoteNetwork.layerCount]

/-- Stack two K-constrained networks of the same workspace shape.
Source: Appendix `lem: stacking-layers`. -/
def CyclicKCoyoteNetwork.append {n k e : ℕ}
    (first second : CyclicKCoyoteNetwork n k e) : CyclicKCoyoteNetwork n k e :=
  {layers := first.layers ++ second.layers}

/-- Stacking executes the first network and then the second.
Source: Appendix `lem: stacking-layers`, cyclic option. -/
theorem CyclicKCoyoteNetwork.run_append {n k e : ℕ}
    (first second : CyclicKCoyoteNetwork n k e)
    (u : RealSequence n (butterflyWidth k)) :
    (first.append second).run u = second.run (first.run u) := by
  simp [CyclicKCoyoteNetwork.append, CyclicKCoyoteNetwork.run, List.foldl_append]

/-- Depths add when networks are stacked.
Source: Appendix `lem: stacking-layers`. -/
theorem CyclicKCoyoteNetwork.layerCount_append {n k e : ℕ}
    (first second : CyclicKCoyoteNetwork n k e) :
    (first.append second).layerCount = first.layerCount + second.layerCount := by
  simp [CyclicKCoyoteNetwork.append, CyclicKCoyoteNetwork.layerCount]

/-- Stacking counts all parameters in both networks.
Source: Appendix `prop: single-baseconv` and `lem: stacking-layers`. -/
theorem CyclicKCoyoteNetwork.parameterCount_append {n k e : ℕ}
    (first second : CyclicKCoyoteNetwork n k e) :
    (first.append second).parameterCount =
      first.parameterCount + second.parameterCount := by
  simp [CyclicKCoyoteNetwork.append, CyclicKCoyoteNetwork.parameterCount]

/-- Replace the projection of a layer by the explicit zero K-matrix,
preserving its convolution and biases. Source: Appendix `lmm:primitives`,
zero-weight convolution/gating constructions. -/
def zeroWeightCyclicKParameters {n k : ℕ}
    (p : CoyoteParameters n (butterflyWidth k)) : KCoyoteParameters n k 1 := {
  weight := zeroExpandedKaleidoscope k
  filter := p.filter
  bias₁ := p.bias₁
  bias₂ := p.bias₂
}

/-- A layer with zero ordinary projection is unchanged by storing that
projection as the zero K-matrix. Source: Appendix `lmm:primitives` and
`def: W-kmat`. -/
theorem zeroWeightCyclicKParameters_correct {n k : ℕ}
    (p : CoyoteParameters n (butterflyWidth k))
    (hweight : p.weight = fun _ _ => 0) :
    (zeroWeightCyclicKParameters p).toParameters = p := by
  have hzero : kaleidoscopeWeight (zeroExpandedKaleidoscope k) =
      fun _ _ => 0 := by
    funext i j
    simp [kaleidoscopeWeight, ExpandedKaleidoscope.matrix,
      zeroExpandedKaleidoscope_apply]
  cases p
  simp_all [zeroWeightCyclicKParameters, KCoyoteParameters.toParameters]

/-- The zero K-matrix uses one `BB*` factor at one extra butterfly level;
its coefficient count is independent of the original dense weight.
Source: Appendix `prop: single-baseconv`, constant K expansion. -/
theorem zeroWeightCyclicKParameters_parameterCount {n k : ℕ}
    (p : CoyoteParameters n (butterflyWidth k)) :
    (zeroWeightCyclicKParameters p).parameterCount =
      3 * n * butterflyWidth k +
        4 * (k + 1) * butterflyWidth (k + 1) := by
  rw [kCoyote_parameterCount_eq]
  simp [zeroWeightCyclicKParameters, zeroExpandedKaleidoscope,
    Kaleidoscope.width]

/-- The zero-weight hypothesis holds for a genuine convolution layer.
Source: Appendix `lmm:primitives`, convolution construction. -/
example (h : RealSequence 2 (butterflyWidth 1)) :
    (convolutionCoyoteParameters h).weight = fun _ _ => 0 := rfl

end Transformer.Zoology
