import Transformer.GPTMini.Convex.Structured.Basic
import Mathlib.Algebra.BigOperators.Ring.Finset

/-!
# Compact contraction of learned matching and value channels

New proposed structured head following Structured.Basic at 19ff013.
For each matching group, free token-local query and key log potentials
are added in a shared channel. Summing latent channel assignments gives
a product of small positive dot products of exponential embeddings.
Both input tables remain trainable; there is no fixed input feature bank.
Free value log potentials add independent output-channel factors to the
same complete configuration. Finite distributivity computes the exact
joint normalizer without materializing every channel combination.

With four four-channel matching groups and five four-channel value
groups, each memory position needs 16+20 channel exponentials, its bias and small
reductions. The implicit 4^9 assignments are a mathematical probability
space, not a stored parameter/feature table or an inference enumeration.
Raw adjacency, role selection, parameter convexity, value readout and
complete Basis capability remain separate construction obligations.
This module proves an actual contraction identity, not FLOP measurements.
-/

namespace Transformer.GPTMini.Convex.Structured

open scoped BigOperators Classical
noncomputable section

variable {G C : Type*} [Fintype G] [Fintype C]

/-- Additive log energy of one channel assignment, with no learned-parameter products.
Source: the new factorial structured head's free per-group log potentials. -/
def channelEnergy (potential : G → C → ℝ) (assignment : G → C) : ℝ :=
  ∑ g, potential g (assignment g)

/-- Compact channel partition evaluated by one small categorical sum per group.
Source: the proposed head's product contraction, rather than an enumerated latent table. -/
def channelPartition (potential : G → C → ℝ) : ℝ :=
  ∏ g, ∑ c, Real.exp (potential g c)

omit [Fintype C] in
/-- Actual additive channel energy exponentiates into independent local factors.
Source: the finite exponential-sum identity; the input log potentials remain arbitrary. -/
theorem channelEnergy_exp (potential : G → C → ℝ) (assignment : G → C) :
    Real.exp (channelEnergy potential assignment) = ∏ g, Real.exp (potential g (assignment g)) :=
  Real.exp_sum _ _

/-- The compact product of small sums equals the full exponential-size configuration normalizer exactly.
Source: finite distributivity (Fintype.prod_sum), applied to actual exponential factors. -/
theorem channelPartition_eq (potential : G → C → ℝ) :
    channelPartition potential = ∑ assignment : G → C, Real.exp (channelEnergy potential assignment) := by
  rw [channelPartition, Fintype.prod_sum]
  apply Finset.sum_congr rfl
  intro assignment _
  exact (channelEnergy_exp potential assignment).symm

/-- Compact contraction is positive at every finite free potential assignment.
Source: nonempty local categorical domains and the true positive exponential factors. -/
theorem channelPartition_pos [Nonempty C] (potential : G → C → ℝ) :
    0 < channelPartition potential :=
  Finset.prod_pos (fun _ _ => Finset.sum_pos (fun _ _ => Real.exp_pos _) Finset.univ_nonempty)

/-- The actual normalized distribution over implicit channel configurations.
Source: the exact compact joint partition above, without a prepared chosen channel at inference. -/
def channelProbability (potential : G → C → ℝ) (assignment : G → C) : ℝ :=
  Real.exp (channelEnergy potential assignment) / channelPartition potential

/-- Every complete channel assignment has positive genuine mass at finite parameters.
Source: the true factorial distribution and its compact positive normalizer. -/
theorem channelProbability_pos [Nonempty C] (potential : G → C → ℝ) (assignment : G → C) :
    0 < channelProbability potential assignment :=
  div_pos (Real.exp_pos _) (channelPartition_pos potential)

/-- The compact implementation gives exactly unit total mass over the implicit assignments.
Source: the exact distributive contraction, not an independent normalization assumption. -/
theorem channelProbability_sum [Nonempty C] (potential : G → C → ℝ) :
    ∑ assignment : G → C, channelProbability potential assignment = 1 := by
  unfold channelProbability
  rw [← Finset.sum_div, ← channelPartition_eq]
  exact div_self (channelPartition_pos potential).ne'

variable {H D J : Type*} [Fintype H] [Fintype D] [Fintype J]

/-- One physical memory route and its implicit matching/value channels.
Source: the proposed compact Gibbs attention configuration, shared across the memory positions. -/
abbrev PointerConfiguration := J × ((G → C) × (H → D))

/-- Complete route energy adds jointly free Q/K and value log potentials to a positional route bias.
Source: the new head formula; Q/K matching is changed from a product of raw trained weights. -/
def pointerEnergy (query : G → C → ℝ) (key : J → G → C → ℝ)
    (value : J → H → D → ℝ) (bias : J → ℝ) (z : PointerConfiguration (G := G) (C := C) (H := H) (D := D) (J := J)) : ℝ :=
  bias z.1 + channelEnergy (fun g c => query g c + key z.1 g c) z.2.1 + channelEnergy (value z.1) z.2.2

/-- Actual compact joint normalizer: one scalar contribution per causal memory position.
Source: the proposed pointer contraction, including both trainable matching and value factors. -/
def pointerPartition (query : G → C → ℝ) (key : J → G → C → ℝ)
    (value : J → H → D → ℝ) (bias : J → ℝ) : ℝ :=
  ∑ j, Real.exp (bias j) * channelPartition (fun g c => query g c + key j g c) * channelPartition (value j)

/-- The computed memory/group contraction equals the full joint Gibbs partition over routes and all channels.
Source: exact finite product-type sums and the proved per-group distributivity. -/
theorem pointerPartition_eq (query : G → C → ℝ) (key : J → G → C → ℝ)
    (value : J → H → D → ℝ) (bias : J → ℝ) :
    pointerPartition query key value bias = ∑ z, Real.exp (pointerEnergy query key value bias z) := by
  rw [Fintype.sum_prod_type]
  simp_rw [Fintype.sum_prod_type, pointerEnergy, Real.exp_add]
  unfold pointerPartition
  simp_rw [channelPartition_eq, Finset.mul_sum, Finset.sum_mul]
  apply Finset.sum_congr rfl
  intro j _
  exact Finset.sum_comm

/-- Joint route/value normalization is strictly positive whenever each local choice domain is nonempty.
Source: positive physical route factors and compact matching/value partitions. -/
theorem pointerPartition_pos [Nonempty J] [Nonempty C] [Nonempty D]
    (query : G → C → ℝ) (key : J → G → C → ℝ) (value : J → H → D → ℝ) (bias : J → ℝ) :
    0 < pointerPartition query key value bias :=
  Finset.sum_pos (fun _ _ => mul_pos (mul_pos (Real.exp_pos _) (channelPartition_pos _))
    (channelPartition_pos _)) Finset.univ_nonempty

/-- The compact attention weight on one memory position after summing its matching and value channels.
Source: the actual contracted joint Gibbs normalizer; value parameters can affect routing. -/
def pointerWeight (query : G → C → ℝ) (key : J → G → C → ℝ)
    (value : J → H → D → ℝ) (bias : J → ℝ) (j : J) : ℝ :=
  Real.exp (bias j) * channelPartition (fun g c => query g c + key j g c) * channelPartition (value j) /
    pointerPartition query key value bias

/-- Actual contracted attention weights normalize across physical memory positions.
Source: the joint route/value distribution and the computed scalar normalizer. -/
theorem pointerWeight_sum [Nonempty J] [Nonempty C] [Nonempty D]
    (query : G → C → ℝ) (key : J → G → C → ℝ) (value : J → H → D → ℝ) (bias : J → ℝ) :
    ∑ j, pointerWeight query key value bias j = 1 := by
  unfold pointerWeight
  rw [← Finset.sum_div]
  exact div_self (pointerPartition_pos query key value bias).ne'

/-- Every actual contracted causal memory weight remains positive at finite trainable potentials.
Source: all local factors and the real shared joint normalizer are positive. -/
theorem pointerWeight_pos [Nonempty J] [Nonempty C] [Nonempty D]
    (query : G → C → ℝ) (key : J → G → C → ℝ) (value : J → H → D → ℝ) (bias : J → ℝ) (j : J) :
    0 < pointerWeight query key value bias j :=
  div_pos (mul_pos (mul_pos (Real.exp_pos _) (channelPartition_pos _)) (channelPartition_pos _))
    (pointerPartition_pos query key value bias)

/-- A concrete control simultaneously includes learned matching groups and independently learned value groups.
Source: the compact pointer normalization, evaluated on two positions and two channels in each of two groups. -/
example : ∑ j : Fin 2, pointerWeight (fun _ : Fin 2 => fun _ : Fin 2 => (0 : ℝ))
    (fun _ _ _ => 0) (fun _ : Fin 2 => fun _ : Fin 2 => fun _ : Fin 2 => 0) (fun _ => 0) j = 1 :=
  pointerWeight_sum _ _ _ _

end
end Transformer.GPTMini.Convex.Structured
