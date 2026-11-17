module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.SharedAtoms.Domain

/-!
# Finite learned component execution and complete coordinate coverage

Source: `review/flat_spectrum.md`, CM92. A learner stores finite learned
complex factors and real mixture weights, executes all components and scores
every original object. These identities permit CM88's complete per-record
forward-query construction; they do not assert that factor discovery is convex
or that a finite component count solves every original task.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.SharedAtoms

open scoped BigOperators
open Identification
open LearnedHinge (Prefix)

variable {O J : Type*} [Fintype O] [Fintype J] {T : ℕ}

/-- Complete coefficient mixture, CM92. No original class is pooled or
removed; every component contributes to its full coefficient tensor. -/
noncomputable def mixture (w : J → ℝ) (f : J → Atom O T) : SequenceParameters O T :=
  ∑ j, w j • joint (f j)

/-- Actual finite learned-factor execution, CM92. Weights are active mass,
including the global atomic budget if the state is in the bounded domain. -/
noncomputable def run (w : J → ℝ) (f : J → Atom O T) (tokens : Prefix O (T + 1)) (c : O) : ℂ :=
  ∑ j, w j • forward (f j) tokens c

omit [Fintype O] in
/-- Complete real coordinates of the factor mixture, CM92. -/
theorem mixture_encode (w : J → ℝ) (f : J → Atom O T) :
    encode (mixture w f) = ∑ j, w j • encode (joint (f j)) := by
  funext z
  rcases z with ⟨b, c, q, d⟩
  cases b <;> simp [encode, mixture, Matrix.sum_apply, Matrix.smul_apply]

/-- Every complex class score equals the original joint model's executed
score, CM92. This holds before taking real parts and on unobserved prefixes. -/
theorem run_eq [DecidableEq O] (w : J → ℝ) (f : J → Atom O T)
    (tokens : Prefix O (T + 1)) (c : O) :
    run w f tokens c = sequenceBranch (mixture w f c) tokens := by
  simp only [run, forward_eq, sequenceBranch_eq, LearnedBilinear.forward_eq,
    mixture, Finset.sum_apply, Pi.smul_apply, Matrix.sum_mulVec, Matrix.smul_mulVec,
    dotProduct_sum, dotProduct_smul, Finset.mul_sum, Complex.real_smul]
  apply Finset.sum_congr rfl
  intro j hj
  ring

/-- Real scores preserve the same full-vocabulary next-object interface,
CM92. The equality does not assume that the prefix appeared in training. -/
theorem logits_eq [DecidableEq O] (w : J → ℝ) (f : J → Atom O T)
    (tokens : Prefix O (T + 1)) (c : O) :
    (run w f tokens c).re = sequenceLogits (mixture w f) tokens c := by
  rw [run_eq]
  rfl

/-- A stored normalized finite mixture has the declared joint capacity,
CM92. All components, including zero mass and the zero atom, may remain stored. -/
theorem mixture_feasible (B : ℝ) (w : J → ℝ) (f : J → Atom O T)
    (hw : ∀ j, 0 ≤ w j) (hmass : ∑ j, w j = 1) (hf : ∀ j, Unit (f j)) :
    encode (mixture (fun j => w j * B) f) ∈ domain B := by
  rw [mixture_encode]
  have he : (∑ j, (w j * B) • encode (joint (f j))) =
      ∑ j, w j • (B • encode (joint (f j))) := by
    simp only [smul_smul]
  rw [he]
  exact finite_feasible B w f hw hmass hf

/-- Every allowed complete joint point has an actual finite executed
mixture, CM92. This is not merely a formal convex domain with no forward path.
It provides no efficient algorithm for discovering the representation. -/
theorem executable_representation (B : ℝ) (r : RealCoordinate O T → ℝ) (hr : r ∈ domain B) :
    ∃ (K : Type) (_ : Fintype K) (w : K → ℝ) (f : K → Atom O T),
      (∀ k, 0 ≤ w k) ∧ (∑ k, w k = 1) ∧ (∀ k, Unit (f k)) ∧
        decode r = mixture (fun k => w k * B) f := by
  obtain ⟨K, hK, w, f, hw, hm, hf, he⟩ := finite_representation B r hr
  refine ⟨K, hK, w, f, hw, hm, hf, ?_⟩
  have hc : r = encode (mixture (fun k => w k * B) f) := by
    rw [mixture_encode, he]
    simp only [smul_smul]
  rw [hc, decode_encode]

example (B : ℝ) : encode (mixture (fun _ : Fin 1 => (1 : ℝ) * B)
    (fun _ => (zeroAtom : Atom Bool 1))) ∈ domain B :=
  mixture_feasible B (fun _ => 1) (fun _ : Fin 1 => zeroAtom)
    (by norm_num) (by simp) (fun _ => zero_unit)

example : ∃ (K : Type) (_ : Fintype K) (w : K → ℝ) (f : K → Atom Bool 1),
    (∀ k, 0 ≤ w k) ∧ (∑ k, w k = 1) ∧ (∀ k, Unit (f k)) ∧
      decode (0 : RealCoordinate Bool 1 → ℝ) = mixture (fun k => w k * 1) f :=
  executable_representation 1 0 (zero_mem 1)

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.SharedAtoms
