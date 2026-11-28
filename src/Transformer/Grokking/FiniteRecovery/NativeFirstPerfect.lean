module

public import Transformer.Grokking.FiniteRecovery.FirstPerfect
public import Transformer.Grokking.FiniteRecovery.AdamWNativeBasin
public import Transformer.Grokking.FiniteRecovery.AdamWRelabeling

/-! First perfect hidden accuracy for ordinary AdamW with native random
initialization. Initial errors can be required explicitly when studying
what training learns. The WSD prefix bound uses the unchanged schedule. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open MeasureTheory Set ProbabilityTheory
open scoped NNReal

variable {A X : Type*} [Fintype A] [Finite X] {H d L T : ℕ}

/-- A scheduled strict hit in an open set of starts has positive probability
exactly when one real training trajectory in that set hits. The event can
restrict the initial logits and all checkpoints come from the same draw. -/
theorem adamW_hit_mass_iff (r : ℕ) (grid : Finset ℕ) (rate : ℕ → ℝ)
    (batch : ℕ → List (LabeledPosition A)) (prompt : X → Word A (T+1)) (f : X → A)
    (I : Set (ComplexParameter A H d L → ℂ)) (hI : IsOpen I)
    (variance : ComplexParameter A H d L → ℝ≥0) (hv : ∀ i, 0 < variance i) :
    let ok z t := z ∈ I ∧ z ∈ terminalStrictDraws r t rate (fun _ => batch) prompt f
    let μ := independentGaussianDraws variance
    let p := μ {z | ∃ t ∈ grid, ok z t}
    (0 < p ↔ ∃ z, ∃ t ∈ grid, ok z t) ∧ ∀ m, Measure.infinitePi (fun _ : ℕ => μ)
      {z | ∀ j ∈ Finset.range m, firstPerfect grid (ok (z j)) = none} = (1-p)^m := by
  intro ok μ p
  let := independentGaussianDraws_openPos variance hv
  have hu (t : ℕ) : IsOpen {z | ok z t} := hI.inter (adamW_strict_open r t rate batch prompt f)
  have hp := firstPerfect_probability μ grid (fun t => {z | ok z t}) (fun t _ => (hu t).measurableSet)
  refine ⟨⟨fun h => ?_,fun ⟨z,t,ht,hz⟩ => ((hu t).measure_pos μ ⟨z,hz⟩).trans_le (hp.2.1 t ht)⟩,hp.2.2⟩
  by_contra! he
  have hempty : {z | ∃ t ∈ grid, ok z t} = ∅ := by
    ext z
    exact ⟨fun ⟨t,ht,hz⟩ => he z t ht hz,False.elim⟩
  simp [p,hempty] at h

omit [Fintype A] [Finite X] in
/-- Requiring a rival to beat the correct token strictly on at least one
initial input is an open condition. Intersecting with it excludes seeds
that already solve the held-out task at initialization. -/
theorem isOpen_wrong_native_start (r : ℕ) (prompt : X → Word A (T+1)) (f : X → A) :
    let score (z : ComplexParameter A H d L → ℂ) x :=
      (Model.fromWeights (sgdInitialWeights true z)).logits r (prompt x) (Fin.last T)
    IsOpen {z | ∃ x b, score z x (f x) < score z x b} := by
  intro score
  rw [show {z | ∃ x b, score z x (f x) < score z x b} =
    ⋃ x, ⋃ b, {z | score z x (f x) < score z x b} by ext; simp]
  exact isOpen_iUnion fun x => isOpen_iUnion fun b => isOpen_lt
    ((continuous_logits_fromWeights r (prompt x) _ (f x)).comp (continuous_sgdInitialWeights true))
    ((continuous_logits_fromWeights r (prompt x) _ b).comp (continuous_sgdInitialWeights true))

/-- A native realizing margin gives positive first-hit mass at every WSD
budget, because initialization itself is an evaluation. -/
theorem adamW_native_hit_probability (r interval cap : ℕ) (batch : ℕ → List (LabeledPosition A))
    (prompt : X → Word A (T+1)) (f : X → A) (variance : ComplexParameter A H d L → ℝ≥0)
    (hv : ∀ i, 0 < variance i) (w : ComplexParameter A H d L → ℂ)
    (hw : StrictOn (fun x => (Model.fromWeights (sgdInitialWeights true w)).logits r
      (prompt x) (Fin.last T)) f) :
    let grid := evaluationGrid interval cap
    let ok z t := z ∈ terminalStrictDraws r t (wsdRate cap) (fun _ => batch) prompt f
    let μ := independentGaussianDraws variance
    let p := μ {z | ∃ t ∈ grid, ok z t}
    0 < p ∧ ∀ m, Measure.infinitePi (fun _ : ℕ => μ)
      {z | ∀ j ∈ Finset.range m, firstPerfect grid (ok (z j)) = none} = (1-p)^m := by
  intro grid ok μ p
  have h := adamW_hit_mass_iff r grid (wsdRate cap) batch prompt f univ isOpen_univ variance hv
  simp only [mem_univ,true_and] at h
  exact ⟨h.1.mpr ⟨w,0,by simp [grid,evaluationGrid],hw⟩,h.2⟩

/-- The exact total learning rate paid by the first K native updates
within warmup, including the zero-rate first update. -/
theorem wsd_warmup_sum (K B : ℕ) (hk : K ≤ B*2/100) :
    ∑ t ∈ Finset.range K, wsdRate B t =
      ((K*(K-1)/2 : ℕ) : ℝ) / (1000 * ((B*2/100 : ℕ) : ℝ)) := by
  rw [← Finset.sum_range_id, Nat.cast_sum, Finset.sum_div]
  apply Finset.sum_congr rfl
  intro t ht
  have ht' := Finset.mem_range.mp ht
  have hw : t < B*2/100 := lt_of_lt_of_le ht' hk
  have hb : t ≤ B-1 := by omega
  simp only [wsdRate,min_eq_left hb,ite_eq_left hw]
  ring

/-- With fixed K, increasing the WSD budget eventually makes the total
learning rate through K smaller than any prescribed positive bound. -/
theorem wsd_warmup_sum_small (K : ℕ) {δ : ℝ} (hδ : 0 < δ) :
    ∃ B₀, ∀ B ≥ B₀, (∑ t ∈ Finset.range K, wsdRate B t) ≤ δ := by
  let s : ℝ := ((K*(K-1)/2 : ℕ) : ℝ)
  obtain ⟨M,hM⟩ := exists_nat_gt (s/δ)
  refine ⟨50 * max K (M+1),fun B hB => ?_⟩
  have hk : K ≤ B*2/100 := by omega
  have hm : M < B*2/100 := by omega
  have hw : 0 < ((B*2/100 : ℕ) : ℝ) := by exact_mod_cast (Nat.zero_le M).trans_lt hm
  rw [wsd_warmup_sum K B hk,div_le_iff₀ (mul_pos (by norm_num) hw)]
  have hs := (div_lt_iff₀ hδ).mp (hM.trans (show (M:ℝ) < ((B*2/100:ℕ):ℝ) by exact_mod_cast hm))
  change s ≤ _
  nlinarith [mul_pos hw hδ]

/-- A positive-mass open basin stays strictly correct through all K
trained updates on every sufficiently long unscaled WSD schedule. The
threshold and the basin are common to all minibatch streams. -/
theorem adamW_native_warmup_basin (r K : ℕ) (prompt : X → Word A (T+1)) (f : X → A)
    (variance : ComplexParameter A H d L → ℝ≥0) (hv : ∀ i, 0 < variance i)
    (w : ComplexParameter A H d L → ℂ) (hw : StrictOn (fun x =>
      (Model.fromWeights (sgdInitialWeights true w)).logits r (prompt x) (Fin.last T)) f) :
    ∃ B₀ U, IsOpen U ∧ 0 < independentGaussianDraws variance U ∧ ∀ B ≥ B₀,
      ∀ z ∈ U, ∀ batch : ℕ → List (LabeledPosition A), ∀ t ≤ K,
        z ∈ terminalStrictDraws r t (wsdRate B) (fun _ => batch) prompt f := by
  obtain ⟨δ,hδ,U,hU,hpos,basin⟩ := adamW_native_basin r K prompt f variance hv w hw
  obtain ⟨B₀,hsmall⟩ := wsd_warmup_sum_small K hδ
  exact ⟨B₀,U,hU,hpos,fun B hB z hz batch t ht => basin z hz _ (fun k =>
    ⟨(wsdRate_bounds B k).1,(wsdRate_bounds B k).2.trans (by norm_num)⟩) (hsmall B hB) batch t ht⟩

open Division.Factors

/-- Every member of the unknown division-factor class uses the native
width 64, six layers and a single pass. The hidden inputs may be any
subset, including the complement of the numerical training mask. -/
theorem factor_native_hit_probability {Z : Type*} [Finite Z] (cap : ℕ)
    (hidden : Z → DivisionInput) (f : DivisionInput → Fin 239)
    (hf : f ∈ factorClass unitToken) (batch : ℕ → List (LabeledPosition (Fin 239)))
    (variance : ComplexParameter (Fin 239) 4 16 6 → ℝ≥0) (hv : ∀ i, 0 < variance i) :
    0 < independentGaussianDraws variance {z | ∃ t ∈ evaluationGrid 256 cap,
      z ∈ terminalStrictDraws 1 t (wsdRate cap) (fun _ => batch) (divisionPrompt ∘ hidden) (f ∘ hidden)} := by
  obtain ⟨w,hw⟩ := class_native_realization f hf
  exact (adamW_native_hit_probability 1 256 cap batch (divisionPrompt ∘ hidden) (f ∘ hidden)
    variance hv w (fun z => hw (hidden z))).1

/-- The same positive first-hit mass holds for binary modular sums of
up to 32 variables on any hidden subset, with no scaling of WSD rates. -/
theorem modularSum_native_hit_probability {k s : ℕ} {Z : Type*} [Finite Z] (cap : ℕ)
    (e : Fin (s+1) ≃ Word (ZMod 2) (k+1)) (hk : (k+1)+(k+1) ≤ 32)
    (hidden : Z → Word (ZMod 2) ((k+1)+k+1)) (batch : ℕ → List (LabeledPosition (ZMod 2)))
    (variance : ComplexParameter (ZMod 2) 4 16 6 → ℝ≥0) (hv : ∀ i, 0 < variance i) :
    0 < independentGaussianDraws variance {z | ∃ t ∈ evaluationGrid 256 cap,
      z ∈ terminalStrictDraws 1 t (wsdRate cap) (fun _ => batch) hidden (modularSum e ∘ hidden)} := by
  obtain ⟨w,hw⟩ := modularSum_initial_or_complement e hk 0
  simp only [add_zero] at hw
  exact (adamW_native_hit_probability 1 256 cap batch hidden (modularSum e ∘ hidden)
    variance hv w (fun z => hw (hidden z))).1

example (batch : ℕ → List (LabeledPosition (Fin 239))) :=
  factor_native_hit_probability 100000 id divisionAnswer divisionAnswer_mem batch (fun _ => 1) (by intro; norm_num)

example (batch : ℕ → List (LabeledPosition (ZMod 2))) :=
  modularSum_native_hit_probability 20480 (wordEnumeration 2 4) (by decide) id batch (fun _ => 1) (by intro; norm_num)

example (batch : ℕ → List (LabeledPosition (ZMod 2))) :=
  modularSum_native_hit_probability 100000 (wordEnumeration 2 8) (by decide) id batch (fun _ => 1) (by intro; norm_num)

example (batch : ℕ → List (LabeledPosition (ZMod 2))) :=
  modularSum_native_hit_probability 200000 (wordEnumeration 2 16) (by decide) id batch (fun _ => 1) (by intro; norm_num)

example (batch : ℕ → List (LabeledPosition (Fin 239)))
    (variance : ComplexParameter (Fin 239) 4 16 6 → ℝ≥0) (hv : ∀ i, 0 < variance i) :=
  adamW_hit_mass_iff 1 (evaluationGrid 256 100000) (wsdRate 100000) batch divisionPrompt divisionAnswer
    _ (isOpen_wrong_native_start 1 divisionPrompt divisionAnswer) variance hv

example := adamW_native_warmup_basin 1 256 divisionPrompt divisionAnswer (fun _ => 1) (by intro; norm_num)
  Division.division_native_realization.choose Division.division_native_realization.choose_spec

example : (∑ t ∈ Finset.range 256, wsdRate 100000 t) = 0.01632 := by
  rw [wsd_warmup_sum 256 100000 (by decide)]
  norm_num

example : (∑ t ∈ Finset.range 256, wsdRate 200000 t) = 0.00816 := by
  rw [wsd_warmup_sum 256 200000 (by decide)]
  norm_num

end Transformer.Grokking.FiniteRecovery.Argand
