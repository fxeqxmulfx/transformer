module

public import Transformer.Grokking.FiniteRecovery.VerifiedRestarts
public import Mathlib.MeasureTheory.Measure.OpenPos

/-! Test a trained checkpoint first, then test one perturbed copy if the
checkpoint fails. Rejection leaves the underlying optimizer trajectory
intact. The independent draws here are proposals along one trajectory. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery

open MeasureTheory Filter Metric Set
open scoped ENNReal Classical

variable {E : Type*} [NormedAddCommGroup E]

/-- The ordinary checkpoint has priority. The caller retains the
returned candidate only if it passes the same verifier. -/
noncomputable def checkpointCandidate (ok : E → Prop) (center noise : E) : E :=
  if ok center then center else center + noise

theorem checkpointCandidate_pass (ok : E → Prop) (center noise : E) :
    ok (checkpointCandidate ok center noise) ↔ ok center ∨ ok (center+noise) := by
  by_cases h : ok center <;> simp [checkpointCandidate,h]

/-- Every original success by the cap remains a success on that seed.
This implication is pointwise and places no assumptions on the noise. -/
theorem checkpoint_preserves_hit (ok : E → Prop) (center noise : ℕ → E) (m : ℕ)
    (h : ∃ j ∈ Finset.range m, ok (center j)) :
    retained ok (fun j => checkpointCandidate ok (center j) (noise j)) m ≠ none := by
  obtain ⟨j,hj,hc⟩ := h
  exact fun hn => (retained_none_iff _ _ _).mp hn j hj ((checkpointCandidate_pass ..).2 (Or.inl hc))

variable [MeasurableSpace E] [BorelSpace E]

omit [BorelSpace E] in
/-- A fixed full-support noise law can hit any nonempty open target
from any center in a compact set with a common positive probability. -/
theorem compact_shift_mass (μ : Measure E) [μ.IsOpenPosMeasure]
    {C U : Set E} (hc : IsCompact C) (hu : IsOpen U) (hne : U.Nonempty) :
    ∃ p > (0 : ℝ≥0∞), ∀ c ∈ C, p ≤ μ {z | c+z ∈ U} := by
  obtain ⟨w,hw⟩ := hne
  obtain ⟨ε,hε,hball⟩ := isOpen_iff.mp hu w hw
  have hd : 0 < ε/2 := by positivity
  obtain ⟨F,hcover⟩ := hc.elim_finite_subcover (fun v : E => ball v (ε/2)) (fun _ => isOpen_ball)
    (fun c _ => mem_iUnion.mpr ⟨c,mem_ball_self hd⟩)
  let q (v : E) := μ (ball (w-v) (ε/2))
  refine ⟨F.inf q,(Finset.lt_inf_iff (by simp)).mpr (fun v _ =>
    isOpen_ball.measure_pos μ ⟨w-v,mem_ball_self hd⟩),?_⟩
  intro c hC
  obtain ⟨v,hv⟩ := mem_iUnion.mp (hcover hC)
  obtain ⟨hF,hcv⟩ := mem_iUnion.mp hv
  refine (Finset.inf_le (f := q) hF).trans (measure_mono fun z hz => hball ?_)
  have h := dist_add_add_le c z v (w-v)
  rw [add_sub_cancel] at h
  exact h.trans_lt (by have h₁ := mem_ball.mp hcv; have h₂ := mem_ball.mp hz; linarith)

/-- The exact miss law for independent proposals at possibly different
checkpoints. Checking a correct center sets its miss factor to zero. -/
theorem checkpoint_miss_product (μ : Measure E) [IsProbabilityMeasure μ] {U : Set E}
    (hu : MeasurableSet U) (center : ℕ → E) (m : ℕ) :
    Measure.infinitePi (fun _ : ℕ => μ) {z | retained (· ∈ U)
      (fun j => checkpointCandidate (· ∈ U) (center j) (z j)) m = none} =
      ∏ j ∈ Finset.range m, if center j ∈ U then 0 else 1-μ {z | center j+z ∈ U} := by
  let V (j : ℕ) : Set E := {z | checkpointCandidate (· ∈ U) (center j) z ∈ U}
  have hV (j : ℕ) : V j = if center j ∈ U then univ else {z | center j+z ∈ U} := by
    ext z; by_cases h : center j ∈ U <;> simp [V,checkpointCandidate,h]
  have hm (j : ℕ) : MeasurableSet (V j) := by
    rw [hV]; split
    · exact MeasurableSet.univ
    · exact hu.preimage ((continuous_const.add continuous_id).measurable)
  rw [show {z | retained (· ∈ U) (fun j => checkpointCandidate (· ∈ U) (center j) (z j)) m = none} =
    Set.pi (Finset.range m : Set ℕ) (fun j => (V j)ᶜ) from
      Set.ext (fun _ => retained_none_iff ..),Measure.infinitePi_pi _ (fun j _ => (hm j).compl)]
  apply Finset.prod_congr rfl
  intro j _
  rw [prob_compl_eq_one_sub (hm j),hV]
  by_cases h : center j ∈ U <;> simp [h]

/-- From one bounded path, finitely many forward proposals reach any
specified confidence, and the accepted weights eventually remain fixed.
The open target and compact set determine one bound for every such path. -/
theorem compact_checkpoint_convergence (μ : Measure E) [IsProbabilityMeasure μ] [μ.IsOpenPosMeasure]
    {C U : Set E} (hc : IsCompact C) (hu : IsOpen U) (hne : U.Nonempty) :
    ∃ p > (0 : ℝ≥0∞), ∀ center : ℕ → E, (∀ j, center j ∈ C) →
      let P := Measure.infinitePi (fun _ : ℕ => μ)
      let trial (z : ℕ → E) j := checkpointCandidate (· ∈ U) (center j) (z j)
      (∀ m, P {z | retained (· ∈ U) (trial z) m = none} ≤ (1-p)^m) ∧
        (∀ q > (0 : ℝ≥0∞), ∃ m, P {z | retained (· ∈ U) (trial z) m = none} < q) ∧
        ∀ᵐ z ∂P, ∃ j, trial z j ∈ U ∧ ∀ k, retained (· ∈ U) (trial z) (j+1+k) = some (trial z j) := by
  obtain ⟨p,hp,hshift⟩ := compact_shift_mass μ hc hu hne
  refine ⟨p,hp,?_⟩
  intro center hC P trial
  have hb (m : ℕ) : P {z | retained (· ∈ U) (trial z) m = none} ≤ (1-p)^m := by
    rw [checkpoint_miss_product μ hu.measurableSet center m]
    calc
      _ ≤ ∏ _j ∈ Finset.range m, (1-p) := Finset.prod_le_prod fun j _ => by
        split
        · exact bot_le
        · exact tsub_le_tsub_left (hshift _ (hC j)) _
      _ = _ := by simp
  have hlim := ENNReal.tendsto_pow_atTop_nhds_zero_of_lt_one
    (ENNReal.sub_lt_self ENNReal.one_ne_top one_ne_zero hp.ne')
  refine ⟨hb,fun q hq => ?_,?_⟩
  · obtain ⟨m,hm⟩ := (hlim.eventually (gt_mem_nhds hq)).exists
    exact ⟨m,(hb m).trans_lt hm⟩
  · have hah : ∀ᵐ z ∂P, ∃ j, trial z j ∈ U := by
      rw [ae_iff]
      apply le_antisymm _ bot_le
      apply le_of_tendsto_of_tendsto' tendsto_const_nhds hlim
      intro m
      apply le_trans _ (hb m)
      apply measure_mono
      intro z hz
      exact (retained_none_iff _ _ m).2 (fun j _ hj => hz ⟨j,hj⟩)
    filter_upwards [hah] with z hz
    obtain ⟨j,hj,_,hkeep⟩ := retained_settles (· ∈ U) (trial z) hz
    exact ⟨j,hj,hkeep⟩

example (μ : Measure ℝ) [IsProbabilityMeasure μ] [μ.IsOpenPosMeasure] :=
  compact_checkpoint_convergence μ (isCompact_closedBall (0 : ℝ) 120) (U := Ioo 1 2)
    isOpen_Ioo (Set.nonempty_Ioo.mpr (by norm_num))

example (z : ℕ → ℝ) : retained (fun x : ℝ => x > 0)
    (fun j => checkpointCandidate (fun x : ℝ => x > 0) 1 (z j)) 1 ≠ none :=
  checkpoint_preserves_hit _ (fun _ => 1) z 1 ⟨0,by decide,by norm_num⟩

end Transformer.Grokking.FiniteRecovery
