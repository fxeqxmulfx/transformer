module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.ComplexLoss

/-!
# Exact prediction inclusion for independently specified neuron factors

Source: `review/convex/moments.md`, CM75. A complex representation per hidden
unit and arbitrary signed real amplitudes per output class have an exact
lift on every input. Recovery uses two class-specific neurons per original
unit/class pair. Thus the recovered class is wider, and does not impose the
original low-width shared-embedding constraint. No such constraint is silently
called convex. The original network belongs to some input-only activation
cone; the union of those cones is not claimed to be convex or cheap to search.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedHinge

open scoped BigOperators

variable {E X J C : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- Positive and negative output amplitudes, CM75. -/
def part (a : ℝ) (positive : Bool) : ℝ := max (if positive then a else -a) 0

/-- Both readout parts are nonnegative, CM75. -/
theorem part_nonneg (a : ℝ) (positive : Bool) : 0 ≤ part a positive := le_max_right _ _

/-- Splitting a learned readout preserves its exact signed value, CM75. -/
theorem part_difference (a : ℝ) : part a true - part a false = a := by
  by_cases ha : 0 ≤ a
  · simp [part, max_eq_left ha, max_eq_right (neg_nonpos.mpr ha)]
  · have ha' : a ≤ 0 := le_of_lt (lt_of_not_ge ha)
    simp [part, max_eq_right ha', max_eq_left (neg_nonneg.mpr ha')]

/-- Joint coordinates absorb all learned representation/readout products,
CM75. Both signs and every output class are retained. -/
def lift (w : J → E) (a : J → C → ℝ) : Parameters J C E :=
  fun j c s => part (a j c) s • w j

/-- The independently specified two-layer complex model, CM75. Its fixed
readout phase is the same phase as the recovered model. -/
noncomputable def factoredForward [Fintype J] (w : J → E) (a : J → C → ℝ)
    (input : E →ₗ[ℝ] ℂ) (c : C) : ℂ :=
  ∑ j, (a j c : ℂ) * (1 + Complex.I) * splitHinge (input (w j))

/-- Lifting preserves the entire complex prediction at every possible
input, CM75. This is stronger than an identity only on training logits. -/
theorem lift_forward [Fintype J] (w : J → E) (a : J → C → ℝ) (input : E →ₗ[ℝ] ℂ) :
    complexForward (lift w a) input = factoredForward w a input := by
  funext c
  unfold complexForward factoredForward
  apply Finset.sum_congr rfl
  intro j hj
  simp only [lift, complexNeuron_of_scaled _ (part_nonneg _ _)]
  rw [← sub_mul, ← sub_mul, ← Complex.ofReal_sub, part_difference]

/-- Orienting a sign constraint commutes with every scalar product, CM75. -/
theorem orient_mul (active : Bool) (a t : ℝ) : orient active (a * t) = a * orient active t := by
  cases active <;> simp [orient]

/-- An actual network whose representations obey the stated signs lifts
into that same complete input cone, CM75. The readouts may have either sign. -/
theorem lift_mem (input : X → E →ₗ[ℝ] ℂ) (region : Set X)
    (mask : J → X × Bool → Bool) (w : J → E) (a : J → C → ℝ)
    (hw : ∀ j x, x.1 ∈ region → 0 ≤ orient (mask j x) (components input x (w j))) :
    lift w a ∈ complexDomain input region mask := by
  intro j c s x hx
  change 0 ≤ orient (mask j x) (components input x (part (a j c) s • w j))
  rw [map_smul, smul_eq_mul, orient_mul]
  exact mul_nonneg (part_nonneg _ _) (hw j x hx)

/-- Every physical representation has an input-only activation mask,
CM75. This existential statement is not a convex algorithm for moving
between masks, nor an assertion that finitely sampled masks cover them all. -/
theorem representations_admit_mask (input : X → E →ₗ[ℝ] ℂ) (region : Set X) (w : J → E) :
    ∃ mask : J → X × Bool → Bool,
      ∀ j x, x.1 ∈ region → 0 ≤ orient (mask j x) (components input x (w j)) := by
  classical
  refine ⟨fun j x => decide (0 ≤ components input x (w j)), ?_⟩
  intro j x hx
  by_cases h : 0 ≤ components input x (w j)
  · simp [orient, h]
  · have hn : components input x (w j) ≤ 0 := le_of_lt (lt_of_not_ge h)
    simpa [orient, h] using neg_nonneg.mpr hn

/-- Recovering class-specific signed neurons has an explicit width cost,
CM75. The original per-unit representation sharing is enlarged. -/
theorem recovered_width (J C : Type*) [Fintype J] [Fintype C] :
    Fintype.card (J × C × Bool) = 2 * Fintype.card J * Fintype.card C := by
  simp only [Fintype.card_prod, Fintype.card_bool]
  ring

example := lift_mem (J := Unit) (C := Bool)
  (fun _ : Bool => (LinearMap.id : ℂ →ₗ[ℝ] ℂ)) Set.univ (fun _ _ => true)
  (fun _ => 1 + Complex.I) (fun _ c => if c then 2 else -3)
  (by intro j x hx; cases h : x.2 <;> simp [components, h, orient])

end Transformer.Grokking.Convex.ConvexChannel.LearnedHinge
