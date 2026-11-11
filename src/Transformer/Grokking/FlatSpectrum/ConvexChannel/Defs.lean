module

public import Transformer.Grokking.FlatSpectrum.ConvexRecall.Loss

/-!
# Jointly learned routing and output channels

Source: `review/flat_spectrum.md`, *Convex channels and capacity*, CC1.
All entries of `W(q,k,v,c)` are directly learned. They give the joint mass
of selecting input key `k` and generating output class `c` when that key
contains value `v`. Summing over output classes gives a routing mass that
is independent of `v`; routing masses sum to one. These are linear
constraints and nonnegativity. The forward reads categorical input
bindings only. It has no fixed answer-copy decoder or arithmetic features.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel

open scoped BigOperators

/-- The complete, unfactorized trainable interaction tensor (CC1). -/
abbrev Channel (Q K V C : Type*) := Q → K → V → C → ℝ

variable {Q K V C S : Type*} [Fintype K] [Fintype C] [Fintype S]

/-- Nonnegative joint weights, value-independent routing mass, and unit
total mass. Every constraint is affine or a half-space (CC1). -/
def Feasible (W : Channel Q K V C) : Prop :=
  (∀ q k v c, 0 ≤ W q k v c) ∧
  (∀ q k v w, ∑ c, W q k v c = ∑ c, W q k w c) ∧
  ∀ q v, ∑ k, ∑ c, W q k v c = 1

/-- Generated class logits from the supplied categorical bindings (CC1).
No target class is an argument of this forward map. -/
def logits (W : Channel Q K V C) (q : Q) (D : K → V) (c : C) : ℝ :=
  ∑ k, W q k (D k) c

/-- Ordinary answer CE; labels are used only in the loss (CC1). -/
noncomputable def answerLoss (query : S → Q) (data : S → K → V) (label : S → C)
    (W : Channel Q K V C) : ℝ :=
  ∑ s, Landscape.Neural.crossEntropy (logits W (query s) (data s)) (label s)

end Transformer.Grokking.FlatSpectrum.ConvexChannel
