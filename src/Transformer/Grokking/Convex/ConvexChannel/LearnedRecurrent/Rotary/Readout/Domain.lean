module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Readout.Execution
public import Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Queries

/-!
# Joint conditional readout optimization and actual feasible queries

Source: `review/convex/moments.md`, CM100. This subproblem varies every readout
on already learned states. It neither freezes the full architecture permanently
nor asserts convexity in independent transition factors. Every finite query is
an actual point of CM98's original learned model family.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Readout

open Pair
open scoped BigOperators
variable {J O : Type*} [Fintype J]

/-- Complete executed score map from all learned readout coordinates, CM100. -/
noncomputable def scoreMap (B : ℝ) (w : J → ℝ) (f : J → Parameters O) :
    (J → O → State) →ₗ[ℝ] (List O → O → ℝ) where
  toFun u := ∑ j, w j • (B • score (replace (f j) (u j)))
  map_add' u v := by
    funext xs c
    simp only [Finset.sum_apply, Pi.add_apply, Pi.smul_apply, smul_eq_mul]
    change (∑ j, w j * (B * score (replace (f j) (u j + v j)) xs c)) =
      (∑ j, w j * (B * score (replace (f j) (u j)) xs c)) +
      (∑ j, w j * (B * score (replace (f j) (v j)) xs c))
    simp only [score_add, mul_add, Finset.sum_add_distrib]
  map_smul' a u := by
    funext xs c
    simp only [Finset.sum_apply, Pi.smul_apply, smul_eq_mul]
    change (∑ j, w j * (B * score (replace (f j) (a • u j)) xs c)) =
      a * ∑ j, w j * (B * score (replace (f j) (u j)) xs c)
    simp only [score_smul]
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro j hj
    ring

/-- Complete conditional readouts remain inside the original learned domain, CM100. -/
theorem score_feasible (B : ℝ) (w : J → ℝ) (f : J → Parameters O)
    (hw : ∀ j, 0 ≤ w j) (hm : ∑ j, w j = 1) (hf : ∀ j, Unit (f j))
    (u : J → O → State) (hu : u ∈ rowDomain J O) :
    scoreMap B w f u ∈ domain O B :=
  finite_feasible B w (fun j => replace (f j) (u j)) hw hm
    (fun j => replace_unit (f j) (hf j) (u j) (hu j))

/-- Actual full-class ordinary CE is jointly convex in all readouts, CM100. -/
theorem loss_convex [Fintype O] (B : ℝ) (w : J → ℝ) (f : J → Parameters O)
    (xs : List O) (target : O) :
    ConvexOn ℝ (rowDomain J O)
      (fun u => Landscape.Neural.crossEntropy (scoreMap B w f u xs) target) := by
  have h := (ConvexRecall.convex_crossEntropy target).comp_linearMap
    ((evaluation xs).comp (scoreMap B w f))
  have hu : ConvexOn ℝ Set.univ
      (fun u => Landscape.Neural.crossEntropy (scoreMap B w f u xs) target) := by
    simpa only [Set.preimage_univ, Function.comp_def, LinearMap.comp_apply,
      evaluation, LinearMap.coe_mk, AddHom.coe_mk] using h
  exact hu.subset (Set.subset_univ _) (rowDomain_convex J O)

/-- Every row may receive an arbitrary numerical displacement before the
concrete radial feasibility step, CM100. No descent claim is implicit. -/
theorem projected_feasible (B : ℝ) (w : J → ℝ) (f : J → Parameters O)
    (hw : ∀ j, 0 ≤ w j) (hm : ∑ j, w j = 1) (hf : ∀ j, Unit (f j))
    (u : J → O → State) :
    scoreMap B w f (fun j c => projectRow (u j c)) ∈ domain O B :=
  score_feasible B w f hw hm hf _ (project_table_mem J O u)

/-- Both signed class probes remain feasible in the full learned family,
CM100. A bias query need not stay in the conditional readout slice. -/
theorem readout_query_feasible [DecidableEq O] (B eta : ℝ) (w : J → ℝ)
    (f : J → Parameters O) (hw : ∀ j, 0 ≤ w j) (hm : ∑ j, w j = 1)
    (hf : ∀ j, Unit (f j)) (u : J → O → State) (hu : u ∈ rowDomain J O)
    (c : O) (s : ℝ) (hs : |s| ≤ 1) (he : 0 ≤ eta) (he1 : eta ≤ 1) :
    query B eta (scoreMap B w f u) c s ∈ domain O B :=
  query_feasible B eta _ c s (score_feasible B w f hw hm hf u hu) hs he he1

example : scoreMap 16 (fun _ : Fin 1 => 1) (fun _ => positionWitness)
    (fun _ c => if c then (Complex.I, 0) else (0, Complex.I)) ∈ domain Bool 16 :=
  score_feasible _ _ _ (by norm_num) (by simp) (fun _ => position_unit) _
    (by intro j c; cases c <;> simp [rowSize])

example : scoreMap 16 (fun _ : Fin 1 => 1) (fun _ => positionWitness)
    (fun _ c => projectRow (if c then (2 * Complex.I, 0) else (0, 2 * Complex.I))) ∈
    domain Bool 16 :=
  projected_feasible _ _ _ (by norm_num) (by simp) (fun _ => position_unit) _

example : query 16 (1 / 160000) (scoreMap 16 (fun _ : Fin 1 => 1)
    (fun _ => positionWitness) (fun _ c => if c then (Complex.I, 0)
      else (0, Complex.I))) true (-1) ∈ domain Bool 16 :=
  readout_query_feasible _ _ _ _ (by norm_num) (by simp) (fun _ => position_unit) _
    (by intro j c; cases c <;> simp [rowSize]) _ _ (by norm_num)
    (by norm_num) (by norm_num)

end Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Readout
