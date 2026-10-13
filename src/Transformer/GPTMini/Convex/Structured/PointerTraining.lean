import Transformer.GPTMini.Convex.Structured.Factorial
import Mathlib.LinearAlgebra.Pi
import Mathlib.Data.Fintype.Sum

/-!
# Jointly trainable compact Q/K/value log-potential tables

New structured-pointer proposal based on the actual Gibbs contraction
in Factorial. A token's embedding contains separate free query, key and
value log-potential fields. Every physical query/memory occurrence reads
that same shared token table. One free chronology coefficient multiplies
the supplied physical position. Complete configuration energies are
proved linear in all these simultaneous raw parameters; the actually
contracted likelihood is globally convex on their unrestricted domain.

Four four-channel matching groups and five four-channel value groups
give 52 free fields per vocabulary entry and one chronology parameter.
This storage does not grow with prefixes, records or latent combinations.
All Q/K/value fields are trained, with no fixed input interaction bank.
Complete route/channel targets are additional training supervision.
The proof does not convexify ordinary output-label-only cross entropy,
prove successful AdamW training, or yet realize the complete raw Basis
encoder/readout inside a genuine replacement residual block.
-/

namespace Transformer.GPTMini.Convex.Structured

open scoped BigOperators Classical
noncomputable section

variable {G C H D J : Type*} [Fintype G] [Fintype C] [Fintype H] [Fintype D] [Fintype J]
variable {V : ℕ}

/-- Shared token-local query/key fields and separate jointly trained value fields.
Source: the proposed compact log-potential embedding layout. -/
abbrev PointerField (G C H D : Type*) := (Fin 2 × (G × C)) ⊕ (H × D)

/-- Unrestricted raw embedding parameters and the ordinary learned chronology scalar.
Source: the proposed shared tables, with neither per-prefix weights nor fixed prototype values. -/
abbrev PointerParameters (V : ℕ) (G C H D : Type*) := (Fin V → PointerField G C H D → ℝ) × ℝ

/-- Query log potentials are actual fields of the shared raw token embedding.
Source: the proposed query slot, distinct from the freely trained key and value slots. -/
def pointerQuery (θ : PointerParameters V G C H D) (token : Fin V) (g : G) (c : C) : ℝ :=
  θ.1 token (.inl (0, (g, c)))

/-- Key log potentials read the same vocabulary table at the physical memory key token.
Source: the proposed separate key slot, not a predefined query-key match function. -/
def pointerKey (θ : PointerParameters V G C H D) (token : Fin V) (g : G) (c : C) : ℝ :=
  θ.1 token (.inl (1, (g, c)))

/-- Value channel log potentials are trained jointly with the shared matching tables.
Source: the proposed value slot, rather than a fixed table of output values for input records. -/
def pointerValue (θ : PointerParameters V G C H D) (token : Fin V) (h : H) (d : D) : ℝ :=
  θ.1 token (.inr (h, d))

/-- The full actual pointer score as a linear map of every raw embedding/value/chronology parameter.
Source: the proposed finite log-potential energy, evaluated at shared token lookups and physical positions. -/
def pointerLinear (query : Fin V) (keys values : J → Fin V) (positions : J → ℝ)
    (z : PointerConfiguration (G := G) (C := C) (H := H) (D := D) (J := J)) :
    PointerParameters V G C H D →ₗ[ℝ] ℝ where
  toFun θ := pointerEnergy (pointerQuery θ query) (fun j => pointerKey θ (keys j))
    (fun j => pointerValue θ (values j)) (fun j => θ.2 * positions j) z
  map_add' x y := by
    simp only [pointerEnergy, channelEnergy, pointerQuery, pointerKey, pointerValue,
      Prod.fst_add, Prod.snd_add, Pi.add_apply, Finset.sum_add_distrib]
    ring
  map_smul' a x := by
    simp only [pointerEnergy, channelEnergy, pointerQuery, pointerKey, pointerValue,
      Prod.smul_fst, Prod.smul_snd, Pi.smul_apply, smul_eq_mul, RingHom.id_apply]
    simp only [← mul_add, ← Finset.mul_sum]
    ring

/-- Actual compact training loss from the computed route/value normalizer and observed complete configuration.
Source: Factorial's small-sum contraction; the observed configuration is not passed into inference. -/
def contractedPointerNLL (query : Fin V) (keys values : J → Fin V) (positions : J → ℝ)
    (observed : PointerConfiguration (G := G) (C := C) (H := H) (D := D) (J := J))
    (θ : PointerParameters V G C H D) : ℝ :=
  Real.log (pointerPartition (pointerQuery θ query) (fun j => pointerKey θ (keys j))
    (fun j => pointerValue θ (values j)) (fun j => θ.2 * positions j)) -
  pointerEnergy (pointerQuery θ query) (fun j => pointerKey θ (keys j))
    (fun j => pointerValue θ (values j)) (fun j => θ.2 * positions j) observed

/-- The actually contracted objective equals the full affine-energy joint Gibbs NLL for every simultaneous parameter assignment.
Source: the proved exact exponential-size/compact partition identity and the actual shared token-table score. -/
theorem contractedPointerNLL_eq (query : Fin V) (keys values : J → Fin V) (positions : J → ℝ)
    (observed : PointerConfiguration (G := G) (C := C) (H := H) (D := D) (J := J))
    (θ : PointerParameters V G C H D) :
    contractedPointerNLL query keys values positions observed θ =
      jointNLL (pointerLinear query keys values positions) (fun _ => 0) observed θ := by
  unfold contractedPointerNLL jointNLL partition
  rw [pointerPartition_eq]
  simp only [energy, pointerLinear, LinearMap.coe_mk, AddHom.coe_mk, add_zero]

/-- The actual compact pointer loss is globally convex jointly in learned embeddings, Q/K, values and chronology.
Source: its exact joint Gibbs identity and Structured.Basic's unrestricted affine-energy convexity theorem. -/
theorem contractedPointerNLL_convex [Nonempty J] [Nonempty C] [Nonempty D]
    (query : Fin V) (keys values : J → Fin V) (positions : J → ℝ)
    (observed : PointerConfiguration (G := G) (C := C) (H := H) (D := D) (J := J)) :
    ConvexOn ℝ Set.univ (contractedPointerNLL query keys values positions observed) := by
  have hfun : contractedPointerNLL query keys values positions observed =
      jointNLL (pointerLinear query keys values positions) (fun _ => 0) observed := by
    funext θ
    exact contractedPointerNLL_eq query keys values positions observed θ
  rw [hfun]
  exact jointNLL_convex _ _ _

/-- A finite training minibatch shares exactly the same raw token tables and chronology weight.
Source: the proposed ordinary-gradient training objective; supervision remains outside every inference normalizer. -/
def contractedPointerBatchNLL {B : Type*} [Fintype B] (queries : B → Fin V)
    (keys values : B → J → Fin V) (positions : B → J → ℝ)
    (observed : B → PointerConfiguration (G := G) (C := C) (H := H) (D := D) (J := J))
    (θ : PointerParameters V G C H D) : ℝ :=
  ∑ b, contractedPointerNLL (queries b) (keys b) (values b) (positions b) (observed b) θ

/-- The actual finite minibatch objective is globally convex in all jointly shared Q/K/value/chronology parameters.
Source: the complete contracted likelihood at each example and finite sums of its genuine convex losses. -/
theorem contractedPointerBatchNLL_convex {B : Type*} [Fintype B]
    [Nonempty J] [Nonempty C] [Nonempty D] (queries : B → Fin V)
    (keys values : B → J → Fin V) (positions : B → J → ℝ)
    (observed : B → PointerConfiguration (G := G) (C := C) (H := H) (D := D) (J := J)) :
    ConvexOn ℝ Set.univ (contractedPointerBatchNLL queries keys values positions observed) := by
  refine ⟨convex_univ, ?_⟩
  intro x hx y hy a b ha hb hab
  simp only [contractedPointerBatchNLL, smul_eq_mul]
  rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
  apply Finset.sum_le_sum
  intro sample _
  exact (contractedPointerNLL_convex (queries sample) (keys sample) (values sample)
    (positions sample) (observed sample)).2 hx hy ha hb hab

/-- The exact trainable field count is linear in group/channel counts rather than their latent Cartesian product.
Source: the actual disjoint Q/K/value slot type, shared by every vocabulary occurrence. -/
theorem pointerField_card :
    Fintype.card (PointerField G C H D) = 2 * (Fintype.card G * Fintype.card C) + Fintype.card H * Fintype.card D := by
  simp only [PointerField, Fintype.card_sum, Fintype.card_prod, Fintype.card_fin]

/-- The proposed Basis pointer uses exactly 52 trained token-local fields.
Source: four four-channel Q/K groups and five four-channel value groups, independent of context length. -/
theorem pointerField_basis_card : Fintype.card (PointerField (Fin 4) (Fin 4) (Fin 5) (Fin 4)) = 52 := by
  rw [pointerField_card]
  norm_num

/-- Fifty-two free fields plus ten fixed label-code axes and one constant fit the original width 64.
Source: the proposed complete residual layout; this count alone does not realize its original tied readout. -/
theorem pointerField_basis_width :
    Fintype.card (PointerField (Fin 4) (Fin 4) (Fin 5) (Fin 4)) + 10 + 1 ≤ 64 := by
  rw [pointerField_basis_card]
  norm_num

/-- A real two-token/two-channel shared-table control satisfies the convex training statement simultaneously.
Source: unrestricted learned Q/K/value slots and physical positions, without a pre-solved routing premise. -/
example : ConvexOn ℝ Set.univ
    (contractedPointerNLL (G := Fin 1) (C := Fin 2) (H := Fin 1) (D := Fin 2)
      (0 : Fin 2) (fun j : Fin 2 => j) (fun j : Fin 2 => j) (fun j : Fin 2 => (j.val : ℝ))
      (0, (fun _ => 0), (fun _ => 0))) :=
  contractedPointerNLL_convex _ _ _ _ _

/-- A concrete nonempty minibatch has simultaneous unrestricted matching/value variables and a convex actual objective.
Source: the two-position control repeated under a shared vocabulary table, not separately fitted per-example parameters. -/
example : ConvexOn ℝ Set.univ
    (contractedPointerBatchNLL (G := Fin 1) (C := Fin 2) (H := Fin 1) (D := Fin 2)
      (fun _ : Fin 2 => (0 : Fin 2)) (fun _ : Fin 2 => fun j : Fin 2 => j)
      (fun _ : Fin 2 => fun j : Fin 2 => j) (fun _ : Fin 2 => fun j : Fin 2 => (j.val : ℝ))
      (fun _ : Fin 2 => (0, (fun _ => 0), (fun _ => 0)))) :=
  contractedPointerBatchNLL_convex _ _ _ _ _

end
end Transformer.GPTMini.Convex.Structured
