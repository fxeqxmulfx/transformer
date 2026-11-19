module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Capacity.Sequence
public import Mathlib.Analysis.Convex.Combination

/-!
# Learned complex factors shared across original output objects

Source: `review/convex/moments.md`, CM92. Each component learns an output
vector and two input encoders. Their product is a complete joint coefficient
tensor. The class factor is shared across both input factors, rather than
recovering independent factors per output class. No task program or semantic
meaning for an object ID is supplied.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.SharedAtoms

open scoped BigOperators
open Identification
open LearnedHinge (Prefix)

variable {O : Type*} [Fintype O] {T : ℕ}

/-- Three learned complex vectors, CM92. This structure holds parameters,
not an assumed theorem or an unseen-answer decoder. -/
structure Atom (O : Type*) (T : ℕ) where
  output : O → ℂ
  query : QueryCoordinate O → ℂ
  context : ContextCoordinate O T → ℂ

/-- Complete complex coefficient tensor of one learned component, CM92. -/
def joint (f : Atom O T) : SequenceParameters O T :=
  fun c q d => f.output c * f.query q * f.context d

/-- Squared norm of every entry of a complex vector, CM92. -/
def size {I : Type*} [Fintype I] (v : I → ℂ) : ℝ := ∑ i, Complex.normSq (v i)

/-- A unit component bound on all three factors, CM92. Directions remain
freely learned; this is not a finite precomputed feature dictionary. -/
def Unit (f : Atom O T) : Prop :=
  size f.output ≤ 1 ∧ size f.query ≤ 1 ∧ size f.context ≤ 1

/-- Actual factor execution on an object sequence, CM92. The output vector
and both encoders are trainable. The final result is a complex class score. -/
def forward (f : Atom O T) (tokens : Prefix O (T + 1)) (c : O) : ℂ :=
  (1 + Complex.I) * f.output c * readQuery tokens f.query * readContext (past tokens) f.context

/-- Each factor path agrees with the complete coefficient model, CM92.
This identity certifies the executed path, rather than defining it by a table. -/
theorem forward_eq [DecidableEq O] (f : Atom O T) (tokens : Prefix O (T + 1)) (c : O) :
    forward f tokens c = sequenceBranch (joint f c) tokens := by
  rw [sequenceBranch_eq, LearnedBilinear.forward_eq]
  unfold forward
  rw [← queryRow_read tokens f.query, ← contextRow_read (past tokens) f.context]
  simp only [Matrix.mulVec, dotProduct, joint, Finset.mul_sum, Finset.sum_mul]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro q hq
  apply Finset.sum_congr rfl
  intro d hd
  have hqreal : star (queryRow tokens q) = queryRow tokens q := by
    rw [← query_cast tokens q]
    simp
  simp only [Pi.star_apply, hqreal]
  ring

/-- The complete real size factorizes over all learned factors, CM92.
Both real and imaginary coefficient directions are included. -/
theorem joint_energy (f : Atom O T) :
    Capacity.energy (encode (joint f)) = size f.output * size f.query * size f.context := by
  simp only [Capacity.energy, encode, Fintype.sum_prod_type, Fintype.sum_bool,
    Bool.false_eq_true, ↓reduceIte]
  rw [← Finset.sum_add_distrib]
  simp only [← Finset.sum_add_distrib]
  have hc (c : O) (q : QueryCoordinate O) (d : ContextCoordinate O T) :
      (joint f c q d).im ^ 2 + (joint f c q d).re ^ 2 =
        Complex.normSq (f.output c) * Complex.normSq (f.query q) * Complex.normSq (f.context d) := by
    rw [← Complex.normSq_mul, ← Complex.normSq_mul, Complex.normSq_apply]
    unfold joint
    ring
  simp only [hc, size, ← Finset.mul_sum, ← Finset.sum_mul, mul_assoc]

/-- Every unit learned component fits the complete unit energy ball, CM92.
This bound is sufficient and may be loose for sums of shared components. -/
theorem unit_energy (f : Atom O T) (hf : Unit f) : Capacity.energy (encode (joint f)) ≤ 1 := by
  rw [joint_energy]
  have hq : 0 ≤ size f.query := Finset.sum_nonneg (fun i _ => Complex.normSq_nonneg _)
  have hd : 0 ≤ size f.context := Finset.sum_nonneg (fun i _ => Complex.normSq_nonneg _)
  have h1 := mul_le_mul_of_nonneg_right hf.1 hq
  have h2 := mul_le_mul_of_nonneg_right (h1.trans (by simpa using hf.2.1)) hd
  simpa using h2.trans (by simpa using hf.2.2)

/-- A valid zero component, CM92, also supplies the unused mass in a
finite mixture with total active mass below its declared budget. -/
def zeroAtom : Atom O T := ⟨0, 0, 0⟩

theorem zero_unit : Unit (zeroAtom : Atom O T) := by
  norm_num [Unit, size, zeroAtom]

example : Capacity.energy (encode (joint (zeroAtom : Atom Bool 1))) ≤ 1 :=
  unit_energy _ zero_unit

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.SharedAtoms
