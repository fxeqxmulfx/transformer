module

public import Transformer.Grokking.FiniteRecovery.ArgandGaussianRecovery
public import Transformer.Grokking.FiniteRecovery.ArgandPhaseNorm

/-! A nuclear norm estimate proved from observed training loss for the
binary copy and constant tasks. At every weight vector the two omitted
words have the same logits as their respective marked training sites. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open Filter MeasureTheory Finset Matrix FlatSpectrum
open scoped Topology BigOperators

variable {H d layers : ℕ} {U V : Type*}

/-- Reusing the graph labels of two mixed words gives exactly the full
sum of final-position losses for every weight vector. -/
theorem mixedPositionBatch_totalLoss (n repeats : ℕ) (mode : Option Bool) (w : Weights Bool H d layers) :
    ∑ word : Word Bool (n + 3), (Model.fromWeights w).labelLoss repeats word (lastTokenTask mode word) =
      (mixedPositionBatch n mode).length * batchLoss repeats (mixedPositionBatch n mode) w := by
  classical
  let model := Model.fromWeights w
  let F (x : Word Bool (n + 3)) := model.labelLoss repeats x (lastTokenTask mode x)
  let L (p : LabeledPosition Bool) := -Real.log (model.probability repeats p.2.1 p.2.2.1 p.2.2.2)
  let const (b : Bool) : Word Bool (n + 3) := fun _ => b
  have hne : const false ≠ const true := fun h => Bool.false_ne_true (congrFun h 0)
  have hmarked (b : Bool) : L ⟨n + 2, markedWord n b, 0, mode.getD b⟩ = F (const b) := by
    have hlog (a : Bool) : model.logits repeats (const b) (Fin.last (n + 2)) a =
        model.logits repeats (markedWord n b) 0 a :=
      (model.logits_const_eq_first repeats b _ a).trans
        (model.logits_first_eq_of_token_eq _ (u := const b) (v := markedWord n b) (marked_first n b).symm a)
    simp only [L, F, Model.labelLoss, Model.probability, lastTokenTask, const, hlog]
  let batch := mixedPositionBatch n mode
  have hlen : (batch.length : ℝ) ≠ 0 := by
    apply Nat.cast_ne_zero.mpr
    simp [batch, mixedPositionBatch]
  change (∑ x, F x) = (batch.length : ℝ) * ((batch.map L).sum / batch.length)
  rw [mul_div_cancel₀ _ hlen]
  have hm : ((observedLastPositions (lastTokenTask mode) (mixedInputs n)).map L).sum =
      ∑ x ∈ mixedInputs n, F x := by
    rw [observedLastPositions, List.map_map, Finset.sum_map_toList]
    rfl
  change _ = (((observedLastPositions _ _ ++ _) : List (LabeledPosition Bool)).map L).sum
  rw [List.map_append, List.sum_append, hm]
  simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero, hmarked]
  have hs := Finset.sum_sdiff (f := F) (Finset.subset_univ ({const false, const true} : Finset _))
  simpa only [mixedInputs, Finset.sum_pair hne, add_assoc] using hs.symm

variable [Fintype U] [Fintype V] [DecidableEq V]

/-- Entrywise absolute mass bounds the nuclear norm by decomposing the
matrix into its individual rank-one cells. -/
private theorem nuclearNorm_le_entry_sum (X : Matrix U V ℂ) :
    nuclearNorm X ≤ ∑ u, ∑ v, ‖X u v‖ := by
  classical
  have hsingle (u : U) (v : V) (c : ℂ) : nuclearNorm (single u v c : Matrix U V ℂ) = ‖c‖ := by
    have h : single u v c = vecMulVec (Pi.single u c) (Pi.single v (1 : ℂ)) := by
      ext i j
      by_cases hu : i = u <;> by_cases hv : j = v <;>
        simp_all [single_apply, vecMulVec_apply] <;> grind
    rw [h, nuclearNorm_vecMulVec]
    simp [Pi.single_apply, apply_ite]
  calc
    _ = nuclearNorm (∑ u, ∑ v, single u v (X u v)) := congrArg _ (matrix_eq_sum_single X)
    _ ≤ ∑ u, ∑ v, nuclearNorm (single u v (X u v)) :=
      (le_sum_of_subadditive nuclearNorm nuclearNorm_zero.le nuclearNorm_add_le _ _).trans
        (sum_le_sum fun _ _ => le_sum_of_subadditive nuclearNorm nuclearNorm_zero.le nuclearNorm_add_le _ _)
    _ = _ := by simp only [hsingle]

/-- The observed mean cross-entropy bounds the nuclear distance of the
entire trained phase table from the target, including both hidden words. -/
theorem nuclearDistance_mixedWords_le_batchLoss (n repeats : ℕ) (mode : Option Bool)
    (cut : U × V ≃ Word Bool (n + 3)) (code : Bool → ℂ) (hunit : ∀ a, ‖code a‖ = 1)
    (w : Weights Bool H d layers) :
    nuclearNorm ((Model.fromWeights w).phaseKernel repeats cut code -
      targetKernel cut code (lastTokenTask mode)) ≤
        2 * (mixedPositionBatch n mode).length * batchLoss repeats (mixedPositionBatch n mode) w := by
  let model := Model.fromWeights w
  let target := lastTokenTask (T := n + 2) mode
  let F (word : Word Bool (n + 3)) := model.labelLoss repeats word (target word)
  have h := nuclearNorm_le_entry_sum (model.phaseKernel repeats cut code - targetKernel cut code target)
  refine h.trans ((sum_le_sum fun u _ => sum_le_sum fun v _ =>
    model.phaseError_le_labelLoss repeats cut code hunit target u v).trans_eq ?_)
  change (∑ u, ∑ v, 2 * F (cut (u,v))) = _
  rw [← Fintype.sum_prod_type (fun p : U × V => 2 * F (cut p)), cut.sum_comp (fun word => 2 * F word), ← Finset.mul_sum]
  change 2 * (∑ word, model.labelLoss repeats word (lastTokenTask mode word)) = _
  rw [mixedPositionBatch_totalLoss, mul_assoc]

/-- The true nuclear norm plus a multiple of the observed mean loss
is an upper bound on the learned norm at every step of any optimizer. -/
theorem nuclearNorm_mixedWords_le_batchLoss (n repeats : ℕ) (mode : Option Bool)
    (cut : U × V ≃ Word Bool (n + 3)) (code : Bool → ℂ) (hunit : ∀ a, ‖code a‖ = 1)
    (w : Weights Bool H d layers) :
    nuclearNorm ((Model.fromWeights w).phaseKernel repeats cut code) ≤
      nuclearNorm (targetKernel cut code (lastTokenTask mode)) +
        2 * (mixedPositionBatch n mode).length * batchLoss repeats (mixedPositionBatch n mode) w := by
  let X := (Model.fromWeights w).phaseKernel repeats cut code
  let K := targetKernel cut code (lastTokenTask mode)
  have h := nuclearNorm_add_le K (X - K)
  rw [add_sub_cancel] at h
  exact h.trans (add_le_add le_rfl (nuclearDistance_mixedWords_le_batchLoss n repeats mode cut code hunit w))

/-- Given any positive tolerance, a full-support draw has positive
probability of converging in weights to a model whose phase table is
within that nuclear distance of the target. -/
theorem mixedWords_momentum_nuclearDistance {layers : ℕ}
    (mu : Measure (ComplexParameter Bool 1 1 layers → ℂ)) [mu.IsOpenPosMeasure]
    (n repeats : ℕ) (mode : Option Bool) (cut : U × V ≃ Word Bool (n + 3))
    (code : Bool → ℂ) (hunit : ∀ a, ‖code a‖ = 1) {ε : ℝ} (hε : 0 < ε) :
    ∃ lam₀ > 0, ∀ lam, 0 < lam → lam ≤ lam₀ →
      ∃ S : Set (ComplexParameter Bool 1 1 layers → ℂ), IsOpen S ∧ 0 < mu S ∧
        ∃ η₀ > 0, ∀ η, 0 < η → η ≤ η₀ → ∀ draw ∈ S,
          ∃ star : Weights Bool 1 1 layers,
            Tendsto (fun t => (fullGradientMomentumRun true repeats (mixedPositionBatch n mode) lam
              (fun _ => η) draw t).weights) atTop (𝓝 star) ∧
            nuclearNorm ((Model.fromWeights star).phaseKernel repeats cut code -
              targetKernel cut code (lastTokenTask mode)) ≤ ε := by
  let batch := mixedPositionBatch n mode
  have hn : 0 < 2 * (batch.length : ℝ) := by simp [batch, mixedPositionBatch]; positivity
  obtain ⟨lam₀, hlam₀, hr⟩ := binaryBatch_momentum_lowLoss mu repeats batch
    (mixedPositionBatch_nativeMargins n layers repeats mode) (div_pos hε hn)
  refine ⟨lam₀, hlam₀, fun lam hlam hle => ?_⟩
  obtain ⟨S, hS, hp, η₀, hη₀, ht⟩ := hr lam hlam hle
  refine ⟨S, hS, hp, η₀, hη₀, fun η hη hηle draw hd => ?_⟩
  obtain ⟨star, hs, hcost⟩ := ht η hη hηle draw hd
  refine ⟨star, hs, (nuclearDistance_mixedWords_le_batchLoss n repeats mode cut code hunit star).trans ?_⟩
  exact (mul_comm _ (batchLoss repeats batch star)) ▸ (le_div_iff₀ hn).mp hcost

example {ε : ℝ} (hε : 0 < ε) (mode : Option Bool) :=
  let mu := independentGaussianDraws (ι := ComplexParameter Bool 1 1 2) (fun _ => 1)
  let _ := independentGaussianDraws_openPos (ι := ComplexParameter Bool 1 1 2) (fun _ => 1) (by simp)
  mixedWords_momentum_nuclearDistance mu 29 2 mode (appendCut Bool 16 16)
    (fun b => if b then (1 : ℂ) else -1) (by rintro (_ | _) <;> simp) hε

end Transformer.Grokking.FiniteRecovery.Argand
