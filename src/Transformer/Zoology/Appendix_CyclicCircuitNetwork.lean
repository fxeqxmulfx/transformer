/-
# Cyclic Coyote networks for arithmetic-circuit simulation

Arora et al., arXiv:2312.04927v1, §4 and Appendix Theorem `thm: gen-ac`.
The appendix permits cyclic convolution. These lemmas lift the already
verified one-token circuit compiler to independent rows of an arbitrary
sequence, retaining the precise layer count.
-/

import Transformer.Zoology.Appendix_CircuitParallel
import Transformer.Zoology.Appendix_ShiftUp
import Transformer.Zoology.Appendix_PaddedModel

namespace Transformer.Zoology

/-- A stack of Coyote layers using the cyclic convolution option in the
paper. Source: §4 equation `eq: coyote-recursion` and Appendix
`sec:simplified-hyena-append`. -/
structure CyclicCoyoteNetwork (n d : ℕ) where
  layers : List (CoyoteParameters n d)

/-- Execute a cyclic Coyote stack in list order. Source: Appendix Lemma
`lem: stacking-layers`, with cyclic convolution. -/
def CyclicCoyoteNetwork.run {n d : ℕ} (net : CyclicCoyoteNetwork n d)
    (u : RealSequence n d) : RealSequence n d :=
  net.layers.foldl (fun state p => coyoteLayerCyclic p state) u

/-- Number of layers in a cyclic Coyote stack. Source: Appendix Lemma
`lem: stacking-layers`. -/
def CyclicCoyoteNetwork.layerCount {n d : ℕ}
    (net : CyclicCoyoteNetwork n d) : ℕ := net.layers.length

/-- Dense scalar storage for all cyclic Coyote layers: one `d × d`
matrix and three `n × d` arrays per layer. Source: §4 equation
`eq: coyote-recursion`; this is not the K-matrix storage bound. -/
def CyclicCoyoteNetwork.denseParameterCount {n d : ℕ}
    (net : CyclicCoyoteNetwork n d) : ℕ :=
  net.layerCount * (d * d + 3 * n * d)

/-- External and inner dimensions of a cyclic-convolution Coyote stack.
Source: Appendix `def: gated-conv`, using its permitted cyclic variant. -/
structure PaddedCyclicCoyoteModel (n d : ℕ) where
  innerLength : ℕ
  innerWidth : ℕ
  lengthBound : n ≤ innerLength
  widthBound : d ≤ innerWidth
  network : CyclicCoyoteNetwork innerLength innerWidth

/-- Zero-pad the input, execute the cyclic stack, and crop its output.
Source: Appendix `def: gated-conv`, input/output convention. -/
def PaddedCyclicCoyoteModel.run {n d : ℕ}
    (model : PaddedCyclicCoyoteModel n d)
    (u : RealSequence n d) : RealSequence n d :=
  cropSequence model.lengthBound model.widthBound
    (model.network.run (padSequence u))

/-- A feature-dependent cyclic impulse reads the row at its own offset.
Source: Appendix Proposition `prop: prim-shift`, cyclic variant. -/
theorem cyclicConvolution_columnImpulse {n d : ℕ}
    (u : RealSequence n d) (shift : Fin d → Fin n) :
    cyclicConvolution u (fun k q => if k = shift q then 1 else 0) =
      fun i q => u (i - shift q) q := by
  classical
  funext i q
  unfold cyclicConvolution
  rw [Finset.sum_eq_single (shift q)]
  · simp
  · intro k _ hk
    simp [hk]
  · intro h
    exact (h (Finset.mem_univ (shift q))).elim

/-- A filter supported at zero acts independently on each sequence row.
Source: §4 equation `eq: coyote-recursion`, cyclic convolution. -/
theorem cyclicConvolution_zeroSupport {n d : ℕ} [NeZero n]
    (u : RealSequence n d) (scale : Fin d → ℝ) (i : Fin n) (q : Fin d) :
    cyclicConvolution u (fun k q => if k = 0 then scale q else 0) i q =
      scale q * u i q := by
  classical
  unfold cyclicConvolution
  rw [Finset.sum_eq_single (0 : Fin n)]
  · simp
  · intro k _ hk
    simp [hk]
  · intro h
    exact (h (Finset.mem_univ (0 : Fin n))).elim

/-- A cyclic Coyote layer with zero linear weight computes columnwise
cyclic convolution. Source: Appendix `lmm:primitives`, cyclic option. -/
theorem coyote_cyclic_realizes_convolution {n d : ℕ}
    (u h : RealSequence n d) :
    coyoteLayerCyclic (convolutionCoyoteParameters h) u =
      cyclicConvolution u h := by
  funext i q
  simp [coyoteLayerCyclic, convolutionCoyoteParameters,
    linearProjection]

/-- A cyclic Coyote layer with zero convolution filter computes a shared
linear projection. Source: Appendix `lmm:primitives`, linear case. -/
theorem coyote_cyclic_realizes_linear {n d : ℕ}
    (u : RealSequence n d) (W : Fin d → Fin d → ℝ) :
    coyoteLayerCyclic (linearCoyoteParameters W) u =
      linearProjection u W := by
  funext i q
  simp [coyoteLayerCyclic, linearCoyoteParameters,
    cyclicConvolution]

/-- Lift a one-token Coyote layer so that it acts independently at every
row of a longer sequence. Source: Appendix Theorem `thm: gen-ac`,
layerwise simulation using zero-offset convolution. -/
def liftOneTokenCyclic {n d : ℕ} [NeZero n]
    (p : CoyoteParameters 1 d) : CoyoteParameters n d := {
  weight := p.weight
  filter := fun k q => if k = 0 then p.filter 0 q else 0
  bias₁ := fun _ q => p.bias₁ 0 q
  bias₂ := fun _ q => p.bias₂ 0 q
}

/-- The lifted layer at a row equals the original one-token layer on that
row. Source: Appendix Theorem `thm: gen-ac`, gate stages. -/
theorem liftOneTokenCyclic_apply {n d : ℕ} [NeZero n]
    (p : CoyoteParameters 1 d) (u : RealSequence n d)
    (i : Fin n) (q : Fin d) :
    coyoteLayerCyclic (liftOneTokenCyclic p) u i q =
      coyoteLayer p (fun _ => u i) 0 q := by
  rw [coyoteLayerCyclic, coyoteLayer]
  simp only [liftOneTokenCyclic]
  rw [cyclicConvolution_zeroSupport]
  simp [linearProjection, causalConvolution]

/-- Lifting every compiled layer makes the circuit compiler act on each
row independently. Source: Appendix Theorem `thm: gen-ac`, composition
of the levelwise simulations. -/
theorem liftOneTokenCyclic_foldl {n d : ℕ} [NeZero n]
    (ps : List (CoyoteParameters 1 d)) (u : RealSequence n d)
    (i : Fin n) :
    ((ps.map liftOneTokenCyclic).foldl
      (fun state p => coyoteLayerCyclic p state) u) i =
      (ps.foldl (fun state p => coyoteLayer p state)
        (fun _ => u i)) 0 := by
  induction ps generalizing u with
  | nil => rfl
  | cons p ps ih =>
      simp only [List.map_cons, List.foldl_cons]
      rw [ih]
      have hrow : (fun _ : Fin 1 =>
          (coyoteLayerCyclic (liftOneTokenCyclic p) u) i) =
          coyoteLayer p (fun _ => u i) := by
        funext j q
        fin_cases j
        exact liftOneTokenCyclic_apply p u i q
      rw [hrow]

/-- The lifted depth-sensitive compiler still uses exactly two layers
per arithmetic-circuit level. Source: Appendix Theorem `thm: gen-ac`. -/
theorem liftedCircuitLevel_layerCount {n inputs outputs : ℕ} [NeZero n]
    (c : ArithmeticCircuit inputs outputs) (level : Fin c.size → ℕ)
    (depth : ℕ) :
    ((compiledCircuitLevelPrefix c level depth).map
      (liftOneTokenCyclic (n := n))).length = 2 * depth := by
  simpa using compiledCircuitLevelPrefix_length c level depth

end Transformer.Zoology
