module

public import Transformer.Grokking.FlatSpectrum.ConvexRecall.Training
public import Mathlib.Topology.Sequences

/-!
# Convergent answer loss forces convergent matching

Source: `review/flat_spectrum.md`, *Convex answer-loss recall*, CR7.
On the compact stochastic domain, continuity of ordinary answer CE and
uniqueness of its minimizer imply that every feasible sequence whose loss
approaches the optimum converges to identity matching. This is a stability
theorem for minimizing sequences, not an assumed optimizer trajectory.
Convergence of a particular algorithm's loss remains a separate obligation.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexRecall

open scoped BigOperators Topology
open Filter Landscape.Neural

variable {κ ν S : Type*} [Fintype κ] [DecidableEq κ] [Fintype ν] [DecidableEq ν]
  [Fintype S]

omit [DecidableEq κ] in
/-- Stochastic matching matrices form a closed set (CR7). -/
theorem isClosed_stochastic : IsClosed {A : Matrix κ κ ℝ | IsStochastic A} := by
  have he : {A : Matrix κ κ ℝ | IsStochastic A} =
      (⋂ q, ⋂ k, {A | 0 ≤ A q k}) ∩ (⋂ q, {A | ∑ k, A q k = 1}) := by
    ext A
    simp [IsStochastic]
  rw [he]
  apply IsClosed.inter
  · exact isClosed_iInter fun q => isClosed_iInter fun k =>
      isClosed_le continuous_const (by fun_prop)
  · exact isClosed_iInter fun q => isClosed_eq (by fun_prop) continuous_const

omit [DecidableEq κ] in
/-- No matching mass can escape to infinity: every feasible coordinate
lies in `[0,1]`, and the finite stochastic domain is compact (CR7). -/
theorem isCompact_stochastic : IsCompact {A : Matrix κ κ ℝ | IsStochastic A} := by
  have hc : IsCompact {A : Matrix κ κ ℝ | ∀ q k, A q k ∈ Set.Icc (0 : ℝ) 1} :=
    isCompact_pi_infinite fun _ => isCompact_pi_infinite fun _ => isCompact_Icc
  apply hc.of_isClosed_subset isClosed_stochastic
  intro A hA q k
  refine ⟨hA.1 q k, ?_⟩
  have h := Finset.single_le_sum (s := Finset.univ) (f := A q)
    (fun j _ => hA.1 q j) (Finset.mem_univ k)
  simpa [hA.2 q] using h

omit [DecidableEq κ] [Fintype ν] in
/-- Dictionary logits are continuous in every matching parameter (CR7). -/
theorem continuous_logits (q : κ) (D : κ → ν) (c : ν) :
    Continuous fun A : Matrix κ κ ℝ => logits A q D c := by
  unfold logits
  apply continuous_finsetSum
  intro k _
  by_cases h : D k = c <;> simp only [h, ↓reduceIte] <;> fun_prop

omit [DecidableEq κ] in
/-- The actual finite-logit cross-entropy is continuous everywhere;
the exponential denominator is strictly positive (CR7). -/
theorem continuous_answerLoss (query : S → κ) (data : S → κ → ν) :
    Continuous (answerLoss query data) := by
  unfold answerLoss
  apply continuous_finsetSum
  intro s _
  unfold crossEntropy
  apply Continuous.sub
  · apply Continuous.log
    · apply continuous_finsetSum
      intro c _
      exact Real.continuous_exp.comp (continuous_logits (query s) (data s) c)
    · intro A
      exact (sum_exp_pos (logits A (query s) (data s)) (data s (query s))).ne'
  · exact continuous_logits (query s) (data s) (data s (query s))

/-- **Loss convergence implies convergence to correct matching.** For
separating input dictionaries, every feasible minimizing sequence tends
to the identity matrix (`review/flat_spectrum.md`, CR7). No loss-to-answer
or parameter-convergence premise is hidden in the separation condition. -/
theorem tendsto_one_of_loss_tendsto [Nontrivial ν] (query : S → κ) (data : S → κ → ν)
    (hsep : Separates query data) (A : ℕ → Matrix κ κ ℝ) (hA : ∀ n, IsStochastic (A n))
    (hloss : Tendsto (fun n => answerLoss query data (A n)) atTop
      (𝓝 (answerLoss query data 1))) :
    Tendsto A atTop (𝓝 1) := by
  apply isCompact_stochastic.tendsto_nhds_of_unique_mapClusterPt
    (Eventually.of_forall hA)
  intro B hB hcluster
  obtain ⟨U, hU, hsub⟩ := mapClusterPt_iff_ultrafilter.mp hcluster
  have hc := (continuous_answerLoss query data).tendsto B |>.comp hsub
  have hl := hloss.mono_left hU
  have he : answerLoss query data B = answerLoss query data 1 :=
    tendsto_nhds_unique hc hl
  exact eq_one_of_answerLoss_le query data hsep hB he.le

/-- A feasible minimizing sequence eventually answers **all** dictionaries
correctly, with a strict argmax. Finiteness makes the eventual statement
uniform over test queries, dictionaries, and competitor classes (CR7). -/
theorem eventually_all_strict_argmax [Nontrivial ν] (query : S → κ) (data : S → κ → ν)
    (hsep : Separates query data) (A : ℕ → Matrix κ κ ℝ) (hA : ∀ n, IsStochastic (A n))
    (hloss : Tendsto (fun n => answerLoss query data (A n)) atTop
      (𝓝 (answerLoss query data 1))) :
    ∀ᶠ n in atTop, ∀ (q : κ) (D : κ → ν) (c : ν),
      c ≠ D q → logits (A n) q D c < logits (A n) q D (D q) := by
  have ht := tendsto_one_of_loss_tendsto query data hsep A hA hloss
  apply eventually_all.mpr
  intro q
  apply eventually_all.mpr
  intro D
  apply eventually_all.mpr
  intro c
  by_cases hc : c = D q
  · exact Eventually.of_forall (fun _ h => (h hc).elim)
  · have hl := (continuous_logits q D c).tendsto (1 : Matrix κ κ ℝ) |>.comp ht
    have hr := (continuous_logits q D (D q)).tendsto (1 : Matrix κ κ ℝ) |>.comp ht
    have hlt : logits (1 : Matrix κ κ ℝ) q D c < logits 1 q D (D q) := by
      simp [logits_one, Ne.symm hc]
    exact (hl.eventually_lt hr hlt).mono fun _ h _ => h

/-- The hypotheses are simultaneously realized by a feasible minimizing
sequence on separating Boolean input dictionaries (CR7). -/
example : Tendsto (fun _ : ℕ => (1 : Matrix Bool Bool ℝ)) atTop (𝓝 1) :=
  tendsto_one_of_loss_tendsto (id : Bool → Bool) (fun _ => id) separates_bool_id
    (fun _ => 1) (fun _ => stochastic_one) tendsto_const_nhds

example := eventually_all_strict_argmax (id : Bool → Bool) (fun _ => id) separates_bool_id
  (fun _ : ℕ => 1) (fun _ => stochastic_one) tendsto_const_nhds

end Transformer.Grokking.FlatSpectrum.ConvexRecall
