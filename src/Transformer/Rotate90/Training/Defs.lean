module

public import Transformer.Rotate90.Power
public import Mathlib.Analysis.Calculus.Gradient.Basic
public import Mathlib.Topology.Algebra.Module.FiniteDimension

/-!
# Actual Rotate90 loss with the three uses of its embedding exposed

Source: `review/flat_spectrum.md`, *Alias symmetry and training*. The auxiliary
untied function exposes the two lookup occurrences and the output occurrence of
E only for differentiation. Restricting to the diagonal recovers exactly the
current tied network, with every residual block and normalization retained.
The full-table loss includes zero numerators and excludes zero denominators.
-/

@[expose] public section

namespace Transformer.Rotate90.Training

open Landscape.Neural
open scoped BigOperators

/-- The Euclidean coordinates of the actual embedding tensor. -/
abbrev Table (V C : Type*) := EuclideanSpace ℝ (V × C)

/-- Three occurrences of the same tensor, temporarily varied independently. -/
abbrev Tables (V C : Type*) := Table V C × Table V C × Table V C

variable {V K H C : Type*} [Fintype V] [Fintype K] [Fintype H] [Fintype C]

/-- Read one ordinary table entry. -/
def entry (E : Table V C) (q : V) (i : C) : ℝ := E.ofLp (q, i)

/-- Unit perturbation of one actual embedding coordinate. -/
noncomputable def basis (q : V) (i : C) : Table V C := by
  classical
  exact EuclideanSpace.single (q, i) 1

/-- All source blocks and both outer normalizers, given the two embedding rows. -/
noncomputable def hidden (eps : ℝ) (blocks : List (Power.BlockWeights K H))
    (x y : Channels K → ℝ) : Channels K → ℝ :=
  rmsNorm eps (blocks.foldl (fun z B => Power.block eps B z)
    (rmsNorm eps (fun i => x i + rotate90 y i)))

/-- The actual readout with separate names for the three occurrences of E. -/
noncomputable def untiedLogit (eps : ℝ) (blocks : List (Power.BlockWeights K H))
    (tables : Tables V (Channels K)) (a b q : V) : ℝ :=
  tiedReadout (entry tables.2.2)
    (hidden eps blocks (entry tables.1 a) (entry tables.2.1 b)) q

/-- Reusing E in all three positions is a continuous linear diagonal map. -/
noncomputable def diagonal : Table V C →L[ℝ] Tables V C :=
  (ContinuousLinearMap.id ℝ _).prod
    ((ContinuousLinearMap.id ℝ _).prod (ContinuousLinearMap.id ℝ _))

omit [Fintype V] in
/-- The diagonal restriction is precisely the existing arbitrary-depth forward. -/
theorem untiedLogit_diagonal (eps : ℝ) (blocks : List (Power.BlockWeights K H))
    (E : Table V (Channels K)) (a b q : V) :
    untiedLogit eps blocks (diagonal E) a b q =
      Power.stackLogit eps (entry E) blocks a b q := by
  rfl

/-- Pull back the table's rows by a finite permutation, leaving channels fixed. -/
noncomputable def permute (σ : Equiv.Perm V) : Table V C →L[ℝ] Table V C :=
  LinearMap.toContinuousLinearMap
    { toFun := fun E => WithLp.toLp 2 (fun j => E.ofLp (σ j.1, j.2))
      map_add' := by intro E F; rfl
      map_smul' := by intro r E; rfl }

/-- The reindexing has the stated action on actual scalar parameters. -/
theorem entry_permute (σ : Equiv.Perm V) (E : Table V C) (q : V) (i : C) :
    entry (permute σ E) q i = entry E (σ q) i := by
  rfl

/-- Reindex the two lookups and the output independently in the auxiliary function. -/
noncomputable def permuteThree (α β γ : Equiv.Perm V) : Tables V C →L[ℝ] Tables V C :=
  (permute α).prodMap ((permute β).prodMap (permute γ))

/-- Hidden computations depend on the selected rows, so the auxiliary logits
transform by the three corresponding operand and class permutations. -/
theorem untiedLogit_permute (eps : ℝ) (blocks : List (Power.BlockWeights K H))
    (tables : Tables V (Channels K)) (α β γ : Equiv.Perm V) (a b q : V) :
    untiedLogit eps blocks (permuteThree α β γ tables) a b q =
      untiedLogit eps blocks tables (α a) (β b) (γ q) := by
  rfl

/-- Softmax includes every class, so permuting classes and their target preserves
cross-entropy. This keeps the competing-answer term in the actual objective. -/
theorem crossEntropy_permute (σ : Equiv.Perm V) (L : V → ℝ) (q : V) :
    crossEntropy (fun c => L (σ c)) q = crossEntropy L (σ q) := by
  unfold crossEntropy
  rw [Equiv.sum_comp σ (fun c => Real.exp (L c))]

variable {p : ℕ} [Fact p.Prime]

/-- Full-table mean cross-entropy with the three embedding occurrences exposed. -/
noncomputable def fullLoss (eps : ℝ) (blocks : List (Power.BlockWeights K H)) (k : ℕ)
    (tables : Tables (ZMod p) (Channels K)) : ℝ :=
  (∑ a : ZMod p, ∑ b : (ZMod p)ˣ,
    crossEntropy (untiedLogit eps blocks tables a b) ((a / (b : ZMod p)) ^ k)) /
      (Fintype.card (ZMod p) * Fintype.card (ZMod p)ˣ)

/-- The actual tied-model training objective, with branch matrices held fixed
when taking its embedding partial gradient. Branch matrices may change each step. -/
noncomputable def sharedLoss (eps : ℝ) (blocks : List (Power.BlockWeights K H)) (k : ℕ)
    (E : Table (ZMod p) (Channels K)) : ℝ :=
  fullLoss eps blocks k (diagonal E)

/-- An embedding merges the multiplicative cosets of a subgroup. Zero remains a
separate fixed residue; no restriction is placed on its embedding row. -/
def Aliased (S : Subgroup (ZMod p)ˣ) (E : Table (ZMod p) C) : Prop :=
  ∀ h ∈ S, ∀ q i, entry E ((h : ZMod p) * q) i = entry E q i

/-- The predicate is exactly fixedness under each subgroup row permutation. -/
theorem aliased_iff (S : Subgroup (ZMod p)ˣ) (E : Table (ZMod p) C) :
    Aliased S E ↔ ∀ h ∈ S, permute (C := C) (Units.mulLeft h) E = E := by
  constructor
  · intro hE h hh
    ext ⟨q, i⟩
    exact hE h hh q i
  · intro hE h hh q i
    exact congrArg (fun F => entry F q i) (hE h hh)

omit [Fintype C] in
/-- A concrete nonzero alias family keeps zero separate and merges the nonzero
rows. It witnesses the symmetry hypotheses at every prime and channel count;
neither the zero row nor the common nonzero row is required to vanish. -/
theorem aliased_zero_nonzero (S : Subgroup (ZMod p)ˣ) (zeroRow nonzeroRow : C → ℝ) :
    Aliased S (WithLp.toLp 2 fun j => if j.1 = 0 then zeroRow j.2 else nonzeroRow j.2) := by
  intro h hh q i
  simp only [entry]
  by_cases hq : q = 0
  · simp only [hq, mul_zero, ite_true]
  · simp only [hq, mul_ne_zero h.ne_zero hq, ite_false]

example := aliased_zero_nonzero (p := 3) (C := Fin 16) ⊤
  (fun i => (i.val + 2 : ℝ)) (fun i => (i.val + 1 : ℝ))

example := untiedLogit_diagonal (V := Fin 3) (K := Fin 1) (H := Fin 2)
  (1 / 8388608) [Power.zeroBlock] (WithLp.toLp 2 fun j => (j.1.val + 1 : ℝ)) 0 1 2
example := entry_permute (V := Fin 3) (C := Fin 2) (Equiv.refl _)
  (WithLp.toLp 2 fun j => (j.1.val + j.2.val : ℝ)) 1 0
example := untiedLogit_permute (V := Fin 3) (K := Fin 1) (H := Fin 2)
  (1 / 8388608) [Power.zeroBlock] (0, 0, 0)
  (Equiv.refl _) (Equiv.refl _) (Equiv.refl _) 0 1 2
example := crossEntropy_permute (Equiv.refl (Fin 3)) (fun q => (q.val : ℝ)) 0
example := aliased_iff (p := 3) (C := Fin 2) ⊤ (WithLp.toLp 2 fun _ => 1)

end Transformer.Rotate90.Training
