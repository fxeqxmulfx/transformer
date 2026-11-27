module

public import Transformer.Grokking.FiniteRecovery.ArgandPhaseNorm
public import Transformer.Grokking.FiniteRecovery.ArgandFullGradientConvergence

/-! An a priori nuclear norm bound for the complete trained Argand,
including the stationary limit, from the energy of a finite prefix.
No logits, labels or spectral measurements on hidden inputs are used. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open Filter Finset FlatSpectrum Transformer.Optimization
open scoped Topology

variable {A U V : Type*} [Fintype A] {T H d layers : ℕ}

/-- Each tied embedding uses a subset of the optimizer's independent
real coordinates, counted once each in its Euclidean norm. -/
theorem norm_embedding_le_weightSpace (w : WeightSpace A H d layers) (a : A) :
    ‖(Model.fromWeights (WithLp.ofLp w)).embedding a‖ ≤ ‖w‖ := by
  classical
  let e : Fin (H * d) × Bool ↪ Parameter A H d layers :=
    ⟨fun p => .inl (.embedding a p.1, p.2), by rintro ⟨i,b⟩ ⟨j,c⟩ h; simpa using h⟩
  have hs := sum_le_sum_of_subset_of_nonneg
    (s := univ.map e) (t := univ) (f := fun k => ‖w k‖ ^ 2) (subset_univ _)
    (fun k _ _ => sq_nonneg _)
  rw [Finset.sum_map, Fintype.sum_prod_type] at hs
  apply le_of_sq_le_sq _ (norm_nonneg _)
  rw [PiLp.norm_sq_eq_of_L2, PiLp.norm_sq_eq_of_L2]
  refine le_trans (le_of_eq (sum_congr rfl fun j _ => ?_)) hs
  rw [Fintype.sum_bool, Complex.sq_norm, Complex.normSq_apply]
  change w (.inl (.embedding a j, false)) * w (.inl (.embedding a j, false)) +
    w (.inl (.embedding a j, true)) * w (.inl (.embedding a j, true)) =
      ‖w (.inl (.embedding a j, true))‖ ^ 2 + ‖w (.inl (.embedding a j, false))‖ ^ 2
  simp [Real.norm_eq_abs, pow_two, add_comm]

variable [Nonempty A] [Fintype U] [Fintype V] [DecidableEq V]

/-- On an objective sublevel the penalty bounds all embeddings, and
the tied readout bounds the surviving mass of the centered phase. -/
theorem nuclearNorm_phaseKernel_le_of_objective
    (repeats : ℕ) (batch : List (LabeledPosition A)) {lam E : ℝ} (hlam : 0 < lam)
    (cut : U × V ≃ Word A (T + 1)) (code : A → ℂ)
    (hunit : ∀ a, ‖code a‖ = 1) (hzero : ∑ a, code a = 0)
    (w : WeightSpace A H d layers) (hw : fullGradientObjective repeats batch lam w ≤ E) :
    nuclearNorm ((Model.fromWeights (WithLp.ofLp w)).phaseKernel repeats cut code) ≤
      phaseTableSizeFactor (Fintype.card U) (Fintype.card V) *
        (1 - Real.exp (-(Real.sqrt (2 * E / lam) * Real.sqrt (H * d : ℕ)))) := by
  let R := Real.sqrt (2 * E / lam)
  let D := R * Real.sqrt (H * d : ℕ)
  have hcoer := (fullGradientObjective_coercive repeats batch lam w).trans hw
  have hr : ‖w‖ ≤ R := Real.le_sqrt_of_sq_le ((le_div_iff₀ hlam).mpr (by linarith only [hcoer]))
  have hD : 0 ≤ D := mul_nonneg (Real.sqrt_nonneg _) (Real.sqrt_nonneg _)
  apply nuclearNorm_le_of_entryBound _ (sub_nonneg.mpr (Real.exp_le_one_iff.mpr (neg_nonpos.mpr hD)))
  intro u v
  apply Model.phaseEntry_le_of_embeddingBound _ (by norm_num [Model.fromWeights])
    (fun a => (norm_embedding_le_weightSpace w a).trans hr) repeats (cut (u,v)) code hunit hzero

/-- The same explicit bound applies at the optimizer limit using only
the prefix energy cap E, with a common positive rate bound for its tail. -/
theorem fullGradientMomentumRun_nuclearNorm_le
    (realScoreInit : Bool) (repeats : ℕ) (batch : List (LabeledPosition A))
    {lam : ℝ} (hlam : 0 < lam) (prefixRate : ℕ → ℝ) (N : ℕ) (E : ℝ)
    (cut : U × V ≃ Word A (T + 1)) (code : A → ℂ)
    (hunit : ∀ a, ‖code a‖ = 1) (hzero : ∑ a, code a = 0) :
    ∃ η₀ > 0, ∀ η, 0 < η → η ≤ η₀ → ∀ draw : ComplexParameter A H d layers → ℂ,
      let f := fullGradientObjective repeats batch lam
      let run := fullGradientMomentumRun realScoreInit repeats batch lam
        (fun t => if t < N then prefixRate t else η) draw
      let x := fun t => WithLp.toLp 2 (run t).weights
      fullMomentumEnergy f η (x N) (WithLp.toLp 2 (run N).velocity) ≤ E →
        ∃ star : WeightSpace A H d layers, Tendsto x atTop (𝓝 star) ∧ gradient f star = 0 ∧
          nuclearNorm ((Model.fromWeights (WithLp.ofLp star)).phaseKernel repeats cut code) ≤
            phaseTableSizeFactor (Fintype.card U) (Fintype.card V) *
              (1 - Real.exp (-(Real.sqrt (2 * E / lam) * Real.sqrt (H * d : ℕ)))) := by
  obtain ⟨η₀, hη₀, hr⟩ := fullGradientMomentumRun_converges_below
    (H := H) (d := d) (layers := layers) realScoreInit repeats batch hlam prefixRate N E
  refine ⟨η₀, hη₀, fun η hη hle draw hcap => ?_⟩
  obtain ⟨star, hs, hg, he⟩ := hr η hη hle draw hcap
  exact ⟨star, hs, hg, nuclearNorm_phaseKernel_le_of_objective repeats batch hlam cut code hunit hzero star he⟩

example (E : ℝ) :=
  let e := wordEnumeration 2 4
  fullGradientMomentumRun_nuclearNorm_le (H := 4) (d := 16) (layers := 1) true 1
    (observedLastPositions (T := 7) (modularSum e) (inputsAtCells (enumeratedCut e) (staircase 15)))
    (show 0 < (1 / 10 : ℝ) by norm_num) (wsdRate 20000) 20000 E (T := 7)
    (enumeratedCut e) ZMod.stdAddChar (by simp)
    (by simpa using sum_stdAddChar_mul 2 (1 : ZMod 2))

/-- On the 16 by 16 parity cut, forcing the radius-only estimate down
to the truth's norm gives a cross-entropy floor of log(7/4) at every
single position. This quantifies the loss of information in that estimate. -/
theorem loss_floor_of_eight_variable_radius_bound (model : Model (ZMod 2) H d layers)
    (heps : 0 < model.eps) {B : ℝ} (hB : ∀ a, ‖model.embedding a‖ ≤ B)
    (hbound : phaseTableSizeFactor 16 16 * (1 - Real.exp (-(B * Real.sqrt (H * d : ℕ)))) ≤ 16)
    (repeats : ℕ) (word : Word (ZMod 2) 8) (a : ZMod 2) :
    Real.log (7 / 4) ≤ -Real.log (model.probability repeats word (Fin.last 7) a) := by
  have hsize : phaseTableSizeFactor 16 16 = 64 := by norm_num [phaseTableSizeFactor]
  rw [hsize] at hbound
  apply le_trans _ (model.lossGe_of_embeddingBound heps hB repeats word _ a)
  apply Real.log_le_log (by norm_num)
  rw [neg_mul]
  linarith only [hbound]

end Transformer.Grokking.FiniteRecovery.Argand
