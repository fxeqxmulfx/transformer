module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Readout.Domain

/-!
# Complete indexed minibatches for conditional readout refitting

Source: `review/convex/moments.md`, CM100. Every observation has its own index;
duplicate prefixes and labels keep their multiplicity. A nonempty short tail is
another indexed batch. Convexity and feasible projected steps imply no
generalization guarantee or descent on a different batch.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Readout

open Pair
open scoped BigOperators
variable {J O : Type*}

/-- Ordinary indexed mean CE, with every original record retained, CM100. -/
noncomputable def meanLoss [Fintype J] [Fintype O]
    (n : ℕ) (B : ℝ) (w : J → ℝ) (f : J → Parameters O)
    (xs : Fin (n + 1) → List O) (target : Fin (n + 1) → O) (u : J → O → State) : ℝ :=
  ((n : ℝ) + 1)⁻¹ * ∑ i, Landscape.Neural.crossEntropy (scoreMap B w f u (xs i)) (target i)

/-- Full ordinary minibatch CE is jointly convex in every readout, CM100. -/
theorem meanLoss_convex [Fintype J] [Fintype O]
    (n : ℕ) (B : ℝ) (w : J → ℝ) (f : J → Parameters O)
    (xs : Fin (n + 1) → List O) (target : Fin (n + 1) → O) :
    ConvexOn ℝ (rowDomain J O) (meanLoss n B w f xs target) := by
  refine ⟨rowDomain_convex J O, ?_⟩
  intro u hu v hv a b ha hb hab
  have hs := Finset.sum_le_sum (s := Finset.univ) (fun (i : Fin (n + 1)) _ =>
    (loss_convex B w f (xs i) (target i)).2 hu hv ha hb hab)
  have hscale := mul_le_mul_of_nonneg_left hs (by positivity : 0 ≤ ((n : ℝ) + 1)⁻¹)
  simpa only [meanLoss, smul_eq_mul, Finset.sum_add_distrib, ← Finset.mul_sum,
    mul_add, mul_left_comm] using hscale

/-- Every row of an accepted interpolation remains feasible, CM100. -/
theorem update_rows (u v : J → O → State) (hu : u ∈ rowDomain J O)
    (hv : v ∈ rowDomain J O) (eta : ℝ) (he : 0 ≤ eta) (he1 : eta ≤ 1) :
    (1 - eta) • u + eta • v ∈ rowDomain J O :=
  rowDomain_convex J O hu hv (by linarith) he (by ring)

/-- A concrete displaced/projected readout proposal and its accepted
interpolation are actual admissible models, CM100. -/
theorem projected_update_feasible [Fintype J]
    (B : ℝ) (w : J → ℝ) (f : J → Parameters O)
    (hw : ∀ j, 0 ≤ w j) (hm : ∑ j, w j = 1) (hf : ∀ j, Unit (f j))
    (u v : J → O → State) (hu : u ∈ rowDomain J O)
    (eta : ℝ) (he : 0 ≤ eta) (he1 : eta ≤ 1) :
    scoreMap B w f ((1 - eta) • u + eta • (fun j c => projectRow (v j c))) ∈ domain O B :=
  score_feasible B w f hw hm hf _
    (update_rows u _ hu (project_table_mem J O v) eta he he1)

example : ConvexOn ℝ (rowDomain (Fin 1) Bool)
    (meanLoss 1 16 (fun _ => 1) (fun _ => positionWitness)
      (fun _ => [true, false]) (fun i => i.val == 0)) :=
  meanLoss_convex _ _ _ _ _ _

example : (1 - (1 / 2 : ℝ)) • (fun (_ : Fin 1) (_ : Bool) => (Complex.I, 0)) +
    (1 / 2 : ℝ) • (fun (_ : Fin 1) (_ : Bool) => (0, Complex.I)) ∈
    rowDomain (Fin 1) Bool :=
  update_rows _ _ (by intro j c; simp [rowSize]) (by intro j c; simp [rowSize])
    _ (by norm_num) (by norm_num)

example : scoreMap 16 (fun _ : Fin 1 => 1) (fun _ => positionWitness)
    ((1 - (1 / 2 : ℝ)) • (fun (_ : Fin 1) (_ : Bool) => (Complex.I, 0)) +
      (1 / 2 : ℝ) • (fun (_ : Fin 1) (_ : Bool) => projectRow (0, 2 * Complex.I))) ∈
    domain Bool 16 :=
  projected_update_feasible _ _ _ (by norm_num) (by simp) (fun _ => position_unit)
    _ _ (by intro j c; simp [rowSize]) _ (by norm_num) (by norm_num)

end Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Readout
