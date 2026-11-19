module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Learning.Queries

/-!
# Complete raw-coordinate ordinary CE update direction

Source: `review/convex/moments.md`, CM82. The full-class CE residual multiplies
the original input row, readout sign, component sign and activation branch.
Every slot/class/sign/component/coordinate is present. Its complete pairing
is proved equal to the actual complex loss slope on the declared domain.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Learning

open scoped BigOperators
open PatternReduction

/-- The native positive sign and real component both correspond to true,
CM82. The other readout sign or component reverses its logit contribution. -/
def polarity (positive : Bool) : ℝ := if positive then 1 else -1

/-- Exact full-row pairing with a weighted fixed branch, CM82. This is
linear training algebra, not evaluation of an infeasible actual neuron. -/
theorem gated_dot {D : Type*} [Fintype D] (weight : ℝ) (active : Bool) (row v : D → ℝ) :
    Projection.dot (fun d => weight * gated active (row d)) v =
      weight * gated active (Projection.dot row v) := by
  cases active <;> simp [Projection.dot, gated, ← Finset.mul_sum, mul_assoc]

variable {O J : Type*} [Fintype O] [Fintype J] {T : ℕ}

/-- Expand a direction through every raw component and sign, CM82. -/
noncomputable def rawDirection (tokens : Prefix O T) (mask : J → Bool → Bool)
    (v : Parameters J O (ObjectWeights O T)) (c : O) : ℝ :=
  ∑ j, ∑ s : Bool, ∑ k : Bool, polarity s * polarity k *
    gated (mask j k) (realInput tokens (project k (v j c s)))

/-- The raw coordinate direction is exactly the complete complex logit
direction, CM82; the imaginary and negative-readout terms are both necessary. -/
theorem rawDirection_eq (tokens : Prefix O T) (mask : J → Bool → Bool)
    (v : Parameters J O (ObjectWeights O T)) :
    rawDirection tokens mask v = logitDirection (rawInput tokens) mask v := by
  have hre (w : ObjectWeights O T) :
      (Complex.reLm.comp (rawInput tokens)) w = realInput tokens (project true w) :=
    Projection.raw_components tokens true w
  have him (w : ObjectWeights O T) :
      (Complex.imLm.comp (rawInput tokens)) w = realInput tokens (project false w) :=
    Projection.raw_components tokens false w
  funext c
  unfold rawDirection logitDirection gatedForward
  simp only [Pi.sub_apply, ← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro j hj
  simp only [Fintype.sum_bool, polarity, Bool.false_eq_true, ↓reduceIte, hre, him]
  ring

/-- Full real-coordinate work, using all learned complex coordinates,
CM82. No factor norm or complex component replaces this joint update metric. -/
def rawWork (g : J → O → Bool → Bool → Coordinate O T → ℝ)
    (v : Parameters J O (ObjectWeights O T)) : ℝ :=
  ∑ j, ∑ c, ∑ s : Bool, ∑ k : Bool, Projection.dot (g j c s k) (project k (v j c s))

variable [DecidableEq O]

/-- Pull a supplied full-class logit residual back to every raw joint
coordinate, CM82. Its source as a CE derivative is proved separately. -/
noncomputable def rawPullback (tokens : Prefix O T) (mask : J → Bool → Bool)
    (residual : O → ℝ) (j : J) (c : O) (s k : Bool)
    (d : Coordinate O T) : ℝ :=
  (residual c * polarity s * polarity k) *
    gated (mask j k) (Projection.rawRow tokens d)

/-- The complete raw pullback preserves the residual's work against the
entire complex logit direction, CM82. Every coordinate is included. -/
theorem rawPullback_work (tokens : Prefix O T) (mask : J → Bool → Bool) (residual : O → ℝ)
    (v : Parameters J O (ObjectWeights O T)) :
    rawWork (rawPullback tokens mask residual) v = ∑ c, residual c * rawDirection tokens mask v c := by
  unfold rawWork rawPullback
  simp_rw [gated_dot, Projection.rawRow_dot]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro c hc
  unfold rawDirection
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro j hj
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro s hs
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro k hk
  ring

/-- The actual full-vocabulary ordinary CE update direction in every raw
joint coordinate, CM82. Targets appear only in the CE residual. -/
noncomputable def rawGradient (tokens : Prefix O T) (mask : J → Bool → Bool) (target : O)
    (p : Parameters J O (ObjectWeights O T)) :=
  rawPullback tokens mask (fun c => ConvexDepth.rowCEGradient (sequenceLogits p tokens) target c)

/-- The complete coordinate direction has exactly the actual ordinary CE
slope against every supplied direction, CM82. Feasibility is required when
using that slope as a derivative or supporting inequality, as proved below. -/
theorem rawGradient_work (tokens : Prefix O T) (mask : J → Bool → Bool) (target : O)
    (p v : Parameters J O (ObjectWeights O T)) :
    rawWork (rawGradient tokens mask target p) v =
      complexSlope (rawInput tokens) mask target p v := by
  rw [rawGradient, rawPullback_work, rawDirection_eq, complexSlope, ceSlope_eq]
  rfl

/-- The raw-coordinate proposal uses a genuine supporting direction of
the actual learned next-object CE over the entire original cone, CM82. -/
theorem sequenceLoss_first_order (mask : J → Pattern (Prefix O T))
    {p q : Parameters J O (ObjectWeights O T)}
    (hp : p ∈ complexDomain rawInput Set.univ mask)
    (hq : q ∈ complexDomain rawInput Set.univ mask) (tokens : Prefix O T) (target : O) :
    sequenceLoss tokens target p +
      rawWork (rawGradient tokens (fun j k => mask j (tokens, k)) target p) (q - p) ≤
        sequenceLoss tokens target q := by
  rw [rawGradient_work]
  exact complexLoss_first_order (E := ObjectWeights O T) (X := Prefix O T) (J := J) (C := O)
    (rawInput (O := O) (T := T)) Set.univ mask (p := p) (q := q)
    hp hq (Set.mem_univ tokens) target

/-- Along certified feasible two-sided raw queries, the full-coordinate
pairing is the derivative of the actual recovered complex CE, CM82. -/
theorem sequenceLoss_hasDerivAt (mask : J → Pattern (Prefix O T))
    {p v : Parameters J O (ObjectWeights O T)} {radius : ℝ} (hr : 0 < radius)
    (hp : p ∈ complexDomain rawInput Set.univ mask)
    (hplus : p + radius • v ∈ complexDomain rawInput Set.univ mask)
    (hminus : p + (-radius) • v ∈ complexDomain rawInput Set.univ mask)
    (tokens : Prefix O T) (target : O) :
    HasDerivAt (fun amount : ℝ => sequenceLoss tokens target (p + amount • v))
      (rawWork (rawGradient tokens (fun j k => mask j (tokens, k)) target p) v) 0 := by
  rw [rawGradient_work]
  exact complexLoss_hasDerivAt (E := ObjectWeights O T) (X := Prefix O T) (J := J) (C := O)
    (rawInput (O := O) (T := T)) Set.univ mask (p := p) (v := v)
    hr hp hplus hminus (Set.mem_univ tokens) target

end Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Learning
