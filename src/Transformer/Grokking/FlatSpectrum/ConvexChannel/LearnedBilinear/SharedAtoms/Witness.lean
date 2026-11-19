module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.SharedAtoms.Model
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Updates.Unobserved

/-!
# Nonconstant complex shared-factor execution controls

Source: `review/convex/moments.md`, CM92. Both object names and every output
remain present. A nonzero unit component changes predictions with its learned
query/context entries. Its imaginary entries affect real scores. These are
representation controls, not supplied features or stress-test accuracy credit.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.SharedAtoms.Witness

open scoped BigOperators
open Identification
open Updates (pairInput)

/-- One learned unit component with nonzero complex input encoders, CM92. -/
noncomputable def component : Atom Bool 1 where
  output c := if c then 1 / 2 else -1 / 2
  query q := match q with
    | none => 0
    | some b => if b then Complex.I / 2 else 1 / 2
  context d := match d with
    | some (.inl (_, b)) => if b then Complex.I / 2 else 1 / 2
    | _ => 0

/-- All three complete squared factor norms are one half, CM92. -/
theorem sizes : size component.output = 1 / 2 ∧
    size component.query = 1 / 2 ∧ size component.context = 1 / 2 := by
  norm_num [size, component, Complex.normSq_apply, Fintype.sum_option,
    Fintype.sum_sum_type, Fintype.sum_prod_type, Fin.sum_univ_succ]

/-- A nonzero feasible component, independently of domain convexity, CM92. -/
theorem unit : Unit component := by
  unfold Unit
  rw [sizes.1, sizes.2.1, sizes.2.2]
  norm_num

/-- Complex multiplication changes real next-object scores, CM92. All four
input pairs and both output objects are represented. -/
theorem scores (a b c : Bool) :
    (forward component (pairInput a b) c).re =
      (if c then (1 : ℝ) else -1) * (if a || b then -1 / 8 else 1 / 8) := by
  cases a <;> cases b <;> cases c <;>
    norm_num [forward, component, readQuery, readContext, pairInput, past,
      Fin.sum_univ_succ]

/-- Both original labels occur among the component's unique predictions,
CM92. The Boolean values are simply two original object identities. -/
theorem nonconstant :
    (forward component (pairInput false false) false).re <
      (forward component (pairInput false false) true).re ∧
    (forward component (pairInput false true) true).re <
      (forward component (pairInput false true) false).re := by
  norm_num [scores]

example : Capacity.energy (encode (joint component)) ≤ 1 := unit_energy component unit

example : (8 : ℝ) • encode (joint component) ∈ domain 8 := generator_mem 8 component unit

example : encode (mixture (fun _ : Fin 1 => (1 : ℝ) * 8) (fun _ => component)) ∈ domain 8 :=
  mixture_feasible 8 (fun _ => 1) (fun _ : Fin 1 => component)
    (by norm_num) (by simp) (fun _ => unit)

example : (1 - (1 / 2 : ℝ)) • ((8 : ℝ) • encode (joint component)) +
    (1 / 2 : ℝ) • ((8 : ℝ) • encode (joint component)) ∈ domain 8 :=
  step_feasible 8 (1 / 2) _ component (generator_mem 8 component unit)
    unit (by norm_num) (by norm_num)

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.SharedAtoms.Witness
