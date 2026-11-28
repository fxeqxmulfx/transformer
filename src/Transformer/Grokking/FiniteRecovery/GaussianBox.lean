module

public import Transformer.Grokking.FiniteRecovery.DelayedCheckpoints

/-! An explicit Gaussian lower bound for the delayed-checkpoint theorem
given the radius of a box of strictly successful weights. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery

open Set MeasureTheory ProbabilityTheory
open scoped NNReal ENNReal

/-- The largest lower bound for the mass of one perturbed copy over the
specified nonempty set of centers. The center set is part of the quantity. -/
noncomputable def uniformProposalMass {E : Type*} [Add E] [MeasurableSpace E]
    (μ : Measure E) (C U : Set E) : ℝ≥0∞ := ⨅ c : C, μ {z | c.val+z ∈ U}

theorem uniformProposalMass_le {E : Type*} [Add E] [MeasurableSpace E] (μ : Measure E)
    (C U : Set E) {c : E} (hc : c ∈ C) : uniformProposalMass μ C U ≤ μ {z | c+z ∈ U} :=
  iInf_le _ (⟨c,hc⟩ : C)

theorem uniformProposalMass_pos {E : Type*} [NormedAddCommGroup E] [MeasurableSpace E]
    (μ : Measure E) [μ.IsOpenPosMeasure] {C U : Set E} (hc : IsCompact C)
    (hu : IsOpen U) (hne : U.Nonempty) : 0 < uniformProposalMass μ C U := by
  obtain ⟨p,hp,lo⟩ := compact_shift_mass μ hc hu hne
  exact hp.trans_le (le_iInf fun c => lo c.val c.property)

/-- The interval length times the smallest normal density on that interval. -/
theorem gaussian_interval_floor (v : ℝ≥0) (hv : 0 < v) (s D ρ : ℝ) (hs : |s| ≤ D) :
    ENNReal.ofReal (2*ρ) * gaussianPDF 0 v (D+ρ) ≤ (gaussianReal 0 v) (Ioo (s-ρ) (s+ρ)) := by
  rw [gaussianReal_apply _ hv.ne']
  calc
    _ = ∫⁻ _ in Ioo (s-ρ) (s+ρ), gaussianPDF 0 v (D+ρ) := by
      simp [Real.volume_Ioo,show s+ρ-(s-ρ) = 2*ρ by ring,mul_comm]
    _ ≤ _ := by
      apply setLIntegral_mono' measurableSet_Ioo
      intro x hx
      obtain ⟨hl,hu⟩ := abs_le.mp hs
      obtain ⟨h1,h2⟩ := hx
      have hsq : x^2 ≤ (D+ρ)^2 := by
        nlinarith [mul_nonneg (show 0 ≤ D+ρ-x by linarith) (show 0 ≤ D+ρ+x by linarith)]
      apply ENNReal.ofReal_le_ofReal
      simp only [gaussianPDFReal_def,sub_zero]
      exact mul_le_mul_of_nonneg_left (Real.exp_le_exp.mpr
        (div_le_div_of_nonneg_right (neg_le_neg hsq) (by positivity))) (by positivity)

/-- In d dimensions each coordinate independently reaches its required
interval with this explicit probability floor. D bounds the coordinate
distance between the reference and the proposal center. -/
noncomputable def gaussianBoxFloor (v : ℝ≥0) (D ρ : ℝ) (d : ℕ) : ℝ≥0∞ :=
  (ENNReal.ofReal (2*ρ) * gaussianPDF 0 v (D+ρ))^d

theorem gaussianBoxFloor_pos (v : ℝ≥0) (hv : 0 < v) (D ρ : ℝ) (hρ : 0 < ρ) (d : ℕ) :
    0 < gaussianBoxFloor v D ρ d := by
  unfold gaussianBoxFloor
  exact pos_iff_ne_zero.mpr (pow_ne_zero _ (mul_ne_zero
    (ENNReal.ofReal_pos.mpr (show 0 < 2*ρ by positivity)).ne' (gaussianPDF_pos _ hv.ne' _).ne'))

variable {ι : Type*} [Fintype ι]

theorem gaussian_box_shift_mass (v : ℝ≥0) (hv : 0 < v) (D ρ : ℝ) (ref center : ι → ℝ)
    (dist : ∀ i, |ref i-center i| ≤ D) (U : Set (ι → ℝ))
    (box : ∀ w, (∀ i, |w i-ref i| < ρ) → w ∈ U) :
    gaussianBoxFloor v D ρ (Fintype.card ι) ≤
      Measure.pi (fun _ : ι => gaussianReal 0 v) {z | center+z ∈ U} := by
  let coord (i : ι) := Ioo (ref i-center i-ρ) (ref i-center i+ρ)
  calc
    _ = ∏ _ : ι, (ENNReal.ofReal (2*ρ) * gaussianPDF 0 v (D+ρ)) := by simp [gaussianBoxFloor]
    _ ≤ ∏ i, (gaussianReal 0 v) (coord i) :=
      Finset.prod_le_prod fun i _ => gaussian_interval_floor v hv _ _ _ (dist i)
    _ = _ := (Measure.pi_pi (fun _ : ι => gaussianReal 0 v) coord).symm
    _ ≤ _ := by
      apply measure_mono
      intro z hz
      apply box
      intro i
      obtain ⟨lo,hi⟩ := hz i (mem_univ i)
      change |center i+z i-ref i| < ρ
      exact abs_lt.mpr ⟨by linarith,by linarith⟩

theorem gaussian_box_uniform_floor (v : ℝ≥0) (hv : 0 < v) (D ρ : ℝ) (ref : ι → ℝ)
    (C U : Set (ι → ℝ)) (dist : ∀ c ∈ C, ∀ i, |ref i-c i| ≤ D)
    (box : ∀ w, (∀ i, |w i-ref i| < ρ) → w ∈ U) :
    gaussianBoxFloor v D ρ (Fintype.card ι) ≤ uniformProposalMass (Measure.pi fun _ : ι =>
      gaussianReal 0 v) C U :=
  le_iInf fun c => gaussian_box_shift_mass v hv D ρ ref c.val (dist c.val c.property) U box

/-- This is the miss event of the actual delayed verifier. The ordinary
prefix can only remove failures. A known box replaces the existential p
by a number for the m independent eligible checks. -/
theorem delayed_gaussian_box_miss (v : ℝ≥0) (hv : 0 < v) (D ρ : ℝ) (ref : ι → ℝ)
    (center : ℕ → ι → ℝ) (U : Set (ι → ℝ)) (hU : MeasurableSet U)
    (box : ∀ w, (∀ i, |w i-ref i| < ρ) → w ∈ U) (B m : ℕ)
    (dist : ∀ j i, |ref i-center (B+j) i| ≤ D) :
    let law := Measure.pi (fun _ : ι => gaussianReal 0 v)
    Measure.infinitePi (fun _ : ℕ => law) {z | retained (· ∈ U)
      (delayedCheckpoint B (· ∈ U) center z) (B+m) = none} ≤
      (1-gaussianBoxFloor v D ρ (Fintype.card ι))^m := by
  intro law
  let tail (z : ℕ → ι → ℝ) j := checkpointCandidate (· ∈ U) (center (B+j)) (z j)
  have sub : {z | retained (· ∈ U) (delayedCheckpoint B (· ∈ U) center z) (B+m) = none} ⊆
      {z | retained (· ∈ U) (tail z) m = none} := by
    intro z miss
    apply (retained_none_iff ..).2
    intro j hj
    simpa [tail,delayedCheckpoint] using (retained_none_iff ..).1 miss (B+j)
      (Finset.mem_range.mpr (Nat.add_lt_add_left (Finset.mem_range.mp hj) B))
  apply (measure_mono sub).trans
  change Measure.infinitePi _ {z | retained (· ∈ U)
    (fun j => checkpointCandidate (· ∈ U) (center (B+j)) (z j)) m = none} ≤ _
  rw [checkpoint_miss_product law hU (fun j => center (B+j)) m]
  calc
    _ ≤ ∏ _j ∈ Finset.range m, (1-gaussianBoxFloor v D ρ (Fintype.card ι)) :=
      Finset.prod_le_prod fun j _ => by
        split
        · exact bot_le
        · exact tsub_le_tsub_left (gaussian_box_shift_mass v hv D ρ ref _ (dist j) U box) _
    _ = _ := by simp

example (m : ℕ) := delayed_gaussian_box_miss (ι := Fin 1) 1 (by norm_num) 0 1 0 (fun _ => 0)
  {w | |w 0| < 1} (isOpen_lt (continuous_apply 0).abs continuous_const).measurableSet
  (fun _ h => by simpa using h 0) 32 m (by simp)

end Transformer.Grokking.FiniteRecovery
