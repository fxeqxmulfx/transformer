module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.SharedAtoms.Model
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.SharedAtoms.Queries

/-!
# Positive input-coordinate geometry for the shared complex factors

Source: `review/flat_spectrum.md`, CM93. The first CM92 experiment concentrates
almost all input-factor energy on always-present bias addresses. A separable
positive coordinate change retains every original object and trainable complex
direction, but changes the fixed-budget capacity region. No equality between
the old and new bounded domains, or convergence improvement, is asserted.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.SharedAtoms.Scaling

open scoped BigOperators
open Identification
open LearnedHinge (Prefix)

variable {O : Type*} [Fintype O] {T : ℕ}

/-- Positive factors can be chosen from original address frequencies or
dimensions. These are coordinate scales, not object semantics, CM93. -/
def weight (a : QueryCoordinate O → ℝ) (b : ContextCoordinate O T → ℝ)
    (z : RealCoordinate O T) : ℝ := a z.2.2.1 * b z.2.2.2

/-- The actual learned input vectors after a separable geometry change,
CM93. Every output, query and context parameter remains trainable. -/
def atom (a : QueryCoordinate O → ℝ) (b : ContextCoordinate O T → ℝ)
    (f : Atom O T) : Atom O T where
  output := f.output
  query q := (a q : ℂ) * f.query q
  context d := (b d : ℂ) * f.context d

omit [Fintype O] in
/-- Every real/imaginary tensor coordinate agrees with the executed scaled
factors, CM93. The scale is shared across classes and complex components. -/
theorem atom_encode (a : QueryCoordinate O → ℝ) (b : ContextCoordinate O T → ℝ)
    (f : Atom O T) : encode (joint (atom a b f)) = Updates.diagonal (weight a b) (encode (joint f)) := by
  funext z
  rcases z with ⟨flag, c, q, d⟩
  cases flag <;> simp [encode, joint, atom, Updates.diagonal, weight, Complex.mul_re,
    Complex.mul_im] <;> ring

omit [Fintype O] in
/-- Nonzero separable scaling has an exact inverse on the complete joint
space, CM93. Thus it removes no original class or trainable direction. -/
theorem inverse (a : QueryCoordinate O → ℝ) (b : ContextCoordinate O T → ℝ)
    (ha : ∀ q, a q ≠ 0) (hb : ∀ d, b d ≠ 0) (r : RealCoordinate O T → ℝ) :
    Updates.diagonal (fun z => (weight a b z)⁻¹) (Updates.diagonal (weight a b) r) = r :=
  Updates.diagonal_inverse _ r (fun z => mul_ne_zero (ha z.2.2.1) (hb z.2.2.2))

/-- The changed bounded geometry, CM93. The image is not silently
identified with the old fixed-budget domain. -/
def domain (a : QueryCoordinate O → ℝ) (b : ContextCoordinate O T → ℝ) (B : ℝ) :
    Set (RealCoordinate O T → ℝ) := Updates.diagonal (weight a b) '' SharedAtoms.domain B

/-- A fixed separable coordinate geometry preserves joint convexity, CM93. -/
theorem domain_convex (a : QueryCoordinate O → ℝ) (b : ContextCoordinate O T → ℝ) (B : ℝ) :
    Convex ℝ (domain a b B) := (SharedAtoms.domain_convex B).linear_image _

/-- Ordinary CE remains convex on the actual scaled joint parameters,
CM93. Independent factor coordinates are not claimed convex. -/
theorem loss_convex [DecidableEq O] (a : QueryCoordinate O → ℝ)
    (b : ContextCoordinate O T → ℝ) (B : ℝ) (tokens : Prefix O (T + 1)) (target : O) :
    ConvexOn ℝ (domain a b B) (fun r => sequenceLoss tokens target (decode r)) := by
  have h := (sequenceLoss_convex tokens target).comp_linearMap Updates.decodeLinear
  have hu : ConvexOn ℝ Set.univ
      (fun r : RealCoordinate O T → ℝ => sequenceLoss tokens target (decode r)) := by
    simpa only [Set.preimage_univ, Function.comp_def, Updates.decodeLinear,
      LinearMap.coe_mk, AddHom.coe_mk] using h
  exact hu.subset (Set.subset_univ _) (domain_convex a b B)

/-- Every scaled domain point is still a finite executed factor mixture,
CM93. Finite existence does not supply an efficient component optimizer. -/
theorem finite_representation (a : QueryCoordinate O → ℝ) (b : ContextCoordinate O T → ℝ)
    (B : ℝ) (r : RealCoordinate O T → ℝ) (hr : r ∈ domain a b B) :
    ∃ (J : Type) (_ : Fintype J) (w : J → ℝ) (f : J → Atom O T),
      (∀ j, 0 ≤ w j) ∧ (∑ j, w j = 1) ∧ (∀ j, Unit (f j)) ∧
      decode r = mixture (fun j => w j * B) (fun j => atom a b (f j)) := by
  obtain ⟨s, hs, rfl⟩ := hr
  obtain ⟨J, hJ, w, f, hw, hm, hf, he⟩ := SharedAtoms.finite_representation B s hs
  refine ⟨J, hJ, w, f, hw, hm, hf, ?_⟩
  have hcoord : Updates.diagonal (weight a b) s =
      encode (mixture (fun j => w j * B) (fun j => atom a b (f j))) := by
    rw [mixture_encode, he, map_sum]
    apply Finset.sum_congr rfl
    intro j hj
    rw [map_smul, map_smul, ← atom_encode, smul_smul]
  rw [hcoord, decode_encode]

example : Updates.diagonal (fun z => (weight (fun _ : QueryCoordinate Bool => 1 / 2)
    (fun _ : ContextCoordinate Bool 1 => 1 / 2) z)⁻¹)
    (Updates.diagonal (weight (fun _ => 1 / 2) (fun _ => 1 / 2))
      (fun _ : RealCoordinate Bool 1 => 1)) = (fun _ => 1) :=
  inverse _ _ (by norm_num) (by norm_num) _

example : ∃ (J : Type) (_ : Fintype J) (w : J → ℝ) (f : J → Atom Bool 1),
    (∀ j, 0 ≤ w j) ∧ (∑ j, w j = 1) ∧ (∀ j, Unit (f j)) ∧
      decode (Updates.diagonal (weight (fun _ => 1 / 2) (fun _ => 1 / 2)) 0) =
        mixture (fun j => w j * 1) (fun j => atom (fun _ => 1 / 2) (fun _ => 1 / 2) (f j)) :=
  finite_representation _ _ 1 _ ⟨0, SharedAtoms.zero_mem 1, rfl⟩

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.SharedAtoms.Scaling
