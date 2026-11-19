module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Lifting

/-!
# A complete activation-pattern expansion and its explicit size

Source: `review/convex/moments.md`, CM75. Retaining every Boolean pattern
gives one label-independent convex domain which contains the exact lift of
every original finite-width phased complex ReLU network. The embeddings and
readout amplitudes remain learned. Reconstruction preserves predictions on
every input, not only the finite region defining the masks.

This construction widens the network and is exponentially large in the
number of region inputs. It proves no efficient pattern-discovery oracle,
compact exact storage, unseen-answer identification, or feasible benchmark
run. A sampled pattern family cannot be substituted in its coverage claim.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge

open scoped BigOperators

variable {E X J C : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- A full assignment of real and imaginary signs on an input region,
CM75. This indexes constraints, not a frozen embedding vector. -/
abbrev Pattern (X : Type*) := X × Bool → Bool

/-- The actual activation pattern of a physical representation, CM75.
No target or task identity is read. -/
noncomputable def signMask (input : X → E →ₗ[ℝ] ℂ) (w : E) : Pattern X := by
  classical
  exact fun x => decide (0 ≤ components input x w)

/-- The actual representation satisfies every sign selected from it,
CM75, including zero preactivations. -/
theorem signMask_admits (input : X → E →ₗ[ℝ] ℂ) (w : E) (x : X × Bool) :
    0 ≤ orient (signMask input w x) (components input x w) := by
  classical
  by_cases h : 0 ≤ components input x w
  · simp [signMask, orient, h]
  · have hn : components input x w ≤ 0 := le_of_lt (lt_of_not_ge h)
    simpa [signMask, orient, h] using neg_nonneg.mpr hn

/-- Keep each original neuron in its actual pattern slot while retaining
every other pattern coordinate at zero, CM75. This is a coverage witness,
not a learner supplied with a solution. -/
noncomputable def expanded (input : X → E →ₗ[ℝ] ℂ) (w : J → E) (a : J → C → ℝ) :
    Parameters (J × Pattern X) C E := by
  classical
  exact fun jm c s => if jm.2 = signMask input (w jm.1) then lift w a jm.1 c s else 0

/-- Every physical model has a feasible point in the same full pattern
domain, CM75. The domain itself does not depend on that model or its labels. -/
theorem expanded_mem (input : X → E →ₗ[ℝ] ℂ) (w : J → E) (a : J → C → ℝ) :
    expanded input w a ∈ complexDomain input Set.univ (fun jm x => jm.2 x) := by
  classical
  intro ⟨j, m⟩ c s x hx
  by_cases hm : m = signMask input (w j)
  · subst m
    change 0 ≤ orient (signMask input (w j) x)
      (components input x (expanded input w a (j, signMask input (w j)) c s))
    simp only [expanded, ↓reduceIte, lift, map_smul, smul_eq_mul, orient_mul]
    exact mul_nonneg (part_nonneg _ _) (signMask_admits input (w j) x)
  · change 0 ≤ orient (m x) (components input x (expanded input w a (j, m) c s))
    simp [expanded, hm, orient]

/-- All-pattern expansion reproduces the whole complex function, including
inputs not used to define the patterns, CM75. Width and parameter growth are
charged separately; no original compact-width equality is asserted. -/
theorem expanded_forward [Fintype X] [DecidableEq X] [Fintype J]
    (input : X → E →ₗ[ℝ] ℂ) (w : J → E) (a : J → C → ℝ) (read : E →ₗ[ℝ] ℂ) :
    complexForward (expanded input w a) read = factoredForward w a read := by
  classical
  funext c
  unfold complexForward factoredForward
  rw [Fintype.sum_prod_type]
  apply Finset.sum_congr rfl
  intro j hj
  have he (m : Pattern X) :
      complexNeuron (expanded input w a (j, m) c true) read -
        complexNeuron (expanded input w a (j, m) c false) read =
      if m = signMask input (w j) then
        (a j c : ℂ) * (1 + Complex.I) * splitHinge (read (w j)) else 0 := by
    by_cases hm : m = signMask input (w j)
    · simp only [expanded, hm, ↓reduceIte, lift,
        complexNeuron_of_scaled _ (part_nonneg _ _)]
      rw [← sub_mul, ← sub_mul, ← Complex.ofReal_sub, part_difference]
    · simp [expanded, hm, complexNeuron_eq, splitHinge, hinge]
  simp_rw [he]
  exact Fintype.sum_ite_eq' _ _

/-- The complete Boolean pattern index has exponential size, CM75.
This counts even redundant or unrealizable sign assignments; removing them
requires a separate exact proof rather than a sampled replacement. -/
theorem pattern_count (X : Type*) [Fintype X] [DecidableEq X] :
    Fintype.card (Pattern X) = 2 ^ (2 * Fintype.card X) := by
  simp only [Pattern, Fintype.card_fun, Fintype.card_prod, Fintype.card_bool]
  rw [Nat.mul_comm]

/-- The recovered full-pattern width includes both signs and every class,
CM75. This count is not a compactness guarantee. -/
theorem expanded_width (X J C : Type*) [Fintype X] [DecidableEq X] [Fintype J] [Fintype C] :
    Fintype.card ((J × Pattern X) × C × Bool) =
      2 * Fintype.card J * Fintype.card C * 2 ^ (2 * Fintype.card X) := by
  simp only [Fintype.card_prod, Fintype.card_bool, pattern_count]
  ring

/-- Even one binary64 complex coordinate in every full Boolean pattern
slot for twenty inputs exceeds the user's 110 GB RAM cap, CM75. This is a
lower bound for naive enumeration, not for every possible exact algorithm. -/
theorem twenty_inputs_exceed_ram : (110000000000 : ℕ) < 16 * 2 ^ (2 * 20) := by
  norm_num

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge
