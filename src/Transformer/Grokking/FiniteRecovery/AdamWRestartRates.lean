module

public import Transformer.Grokking.FiniteRecovery.AdamWNativeBasin

/-! A universal independent-restart schedule that needs no estimate of
the native realizing margin. Every trial uses ordinary AdamW and WSD.
An arbitrary finite prefix may use the stock schedule. Subsequent trials
shrink their total rate and may have different training horizons. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open MeasureTheory Filter Set
open scoped NNReal ENNReal Topology

variable {A X : Type*} [Fintype A] [Finite X] {T H d L : ℕ}

/-- Start with the unscaled WSD runs. The later peak scale decreases with
both the restart index and the chosen horizon, including growing horizons. -/
noncomputable def restartWsdRate (stockRuns : ℕ) (K : ℕ → ℕ) (j t : ℕ) : ℝ :=
  if j < stockRuns then wsdRate (K j) t else
    wsdRate (K j) t * min 1 (((K 0:ℝ)+1)/((K j:ℝ)+1)) * (((stockRuns:ℝ)+1)/((j:ℝ)+1))

theorem restartWsdRate_prefix {stockRuns j : ℕ} (K : ℕ → ℕ) (hj : j < stockRuns) :
    restartWsdRate stockRuns K j = wsdRate (K j) := funext fun _ => ite_eq_left hj

theorem restartWsdRate_bounds (stockRuns : ℕ) (K : ℕ → ℕ) (j t : ℕ) :
    0 ≤ restartWsdRate stockRuns K j t ∧ restartWsdRate stockRuns K j t ≤ 10 := by
  obtain ⟨h0,h1⟩ := wsdRate_bounds (K j) t
  unfold restartWsdRate
  split
  · exact ⟨h0,by linarith⟩
  · rename_i hj
    have hj' : (stockRuns:ℝ) ≤ j := by exact_mod_cast Nat.le_of_not_gt hj
    have hfactor : ((stockRuns:ℝ)+1)/((j:ℝ)+1) ≤ 1 := (div_le_one (by positivity)).2 (by linarith)
    have hfirst := mul_le_of_le_one_right h0 (min_le_left 1 (((K 0:ℝ)+1)/((K j:ℝ)+1)))
    exact ⟨by positivity,(mul_le_of_le_one_right (by positivity) hfactor).trans (hfirst.trans (by linarith))⟩

/-- Total permitted movement tends to zero along the restarted trials
regardless of how their finite training horizons grow. -/
theorem restartWsdRate_total (stockRuns : ℕ) (K : ℕ → ℕ) :
    Tendsto (fun j => ∑ t ∈ Finset.range (K j), restartWsdRate stockRuns K j t) atTop (𝓝 0) := by
  let C : ℝ := ((K 0:ℝ)+1)*((stockRuns:ℝ)+1)
  have ht : Tendsto (fun j : ℕ => C*(1/((j:ℝ)+1))) atTop (𝓝 0) := by
    simpa using tendsto_const_nhds.mul (tendsto_one_div_add_atTop_nhds_zero_nat (𝕜 := ℝ)) (a := C)
  apply squeeze_zero' (Eventually.of_forall fun j => Finset.sum_nonneg fun t _ => (restartWsdRate_bounds stockRuns K j t).1) _ ht
  filter_upwards [eventually_ge_atTop stockRuns] with j hj
  have hs : (∑ t ∈ Finset.range (K j), wsdRate (K j) t) ≤ (K j : ℝ) := by
    calc
      _ ≤ ∑ _ ∈ Finset.range (K j), (1:ℝ) := Finset.sum_le_sum fun t _ => by linarith [(wsdRate_bounds (K j) t).2]
      _ = _ := by simp
  let a : ℝ := (K 0:ℝ)+1
  let b : ℝ := (K j:ℝ)+1
  have hb : 0 < b := by dsimp [b]; positivity
  have ha : 0 ≤ a/b := by dsimp [a,b]; positivity
  have hcap : (K j:ℝ)*min 1 (a/b) ≤ a := calc
    _ ≤ b*(a/b) := mul_le_mul (by dsimp [b]; linarith) (min_le_right _ _) (le_min zero_le_one ha) hb.le
    _ = a := by field_simp
  simp only [restartWsdRate,ite_eq_right (Nat.not_lt.mpr hj),← Finset.sum_mul]
  change _ * min 1 (a/b) * _ ≤ _
  apply le_trans (mul_le_mul_of_nonneg_right ((mul_le_mul_of_nonneg_right hs (le_min zero_le_one ha)).trans hcap)
    (show 0 ≤ ((stockRuns:ℝ)+1)/((j:ℝ)+1) by positivity))
  dsimp [a,C]
  exact le_of_eq (by ring)

abbrev TrialDraw (A : Type*) (H d L : ℕ) := ComplexParameter A H d L → ℂ

noncomputable def restartScores (r : ℕ) (K : ℕ → ℕ) (rate : ℕ → ℕ → ℝ)
    (prompt : X → Word A (T+1)) (draws : ℕ → TrialDraw A H d L)
    (batch : ℕ → ℕ → List (LabeledPosition A)) (j : ℕ) (x : X) : A → ℝ :=
  (Model.fromWeights (adamWRun r (batch j) (rate j) (draws j) (K j)).weights).logits r (prompt x) (Fin.last T)

/-- The schedule and the selected minibatches can vary between trials.
The probability of missing all readable endpoints in a finite prefix is
bounded by a geometric tail, uniformly in the unknown member. The bound
allows looking at all the draws when choosing minibatches. -/
theorem adamW_restart_rate_bound (r : ℕ) (K : ℕ → ℕ) (rate : ℕ → ℕ → ℝ)
    (hr : ∀ j t, 0 ≤ rate j t ∧ rate j t ≤ 10)
    (hlim : Tendsto (fun j => ∑ t ∈ Finset.range (K j),rate j t) atTop (𝓝 0))
    (prompt : X → Word A (T+1)) (S : Set X) (family : Set (X → A))
    (hid : InjOn (fun g : X → A => fun x : S => g x) family)
    (variance : ComplexParameter A H d L → ℝ≥0) (hv : ∀ i, 0 < variance i)
    (hrep : ∀ f ∈ family, ∃ z : TrialDraw A H d L, StrictOn
      (fun x => (Model.fromWeights (sgdInitialWeights true z)).logits r (prompt x) (Fin.last T)) f) :
    let μ := independentGaussianDraws variance
    let P := Measure.infinitePi (fun _ : ℕ => μ)
    ∃ J, ∃ p : family → ℝ≥0∞, (∀ f, 0 < p f) ∧ ∀ f : family, ∀ m,
      P {z | ∃ batch : ℕ → ℕ → List (LabeledPosition A), ∀ j ∈ Finset.range (J+m),
        ¬classCertificate S family (fun x => f.val x) (restartScores r K rate prompt z batch j)}
        ≤ (1-p f)^m := by
  classical
  let μ := independentGaussianDraws variance
  choose w hw using fun f : family => hrep f f.property
  choose δ hδ U hU hp basin using fun f : family =>
    adamW_native_rate_basin r prompt f.val variance hv (w f) (hw f)
  obtain ⟨a,ha,hle⟩ := exists_pos_le_all δ hδ
  obtain ⟨J,hJ⟩ := eventually_atTop.mp (hlim.eventually (gt_mem_nhds ha))
  refine ⟨J,fun f => μ (U f),hp,fun f m => ?_⟩
  rw [← iid_miss_window μ (hU f).measurableSet J m]
  apply measure_mono
  rintro z ⟨batch,h⟩ j hj hu
  obtain ⟨hjJ,hjm⟩ := Finset.mem_Ico.mp hj
  exact h j (Finset.mem_range.mpr hjm) <| (classCertificate_iff S family hid f.val f.property _).2
    (basin f _ hu _ (hr j) (batch j) (K j) ((hJ j hjJ).le.trans (hle f)))

/-- The executable schedule needs only a prefix length and finite
budgets. At any prescribed confidence a common restart count suffices
for all targets. No step-size or basin threshold is supplied by the user. -/
theorem adamW_restartWsd_confidence (r stockRuns : ℕ) (K : ℕ → ℕ)
    (prompt : X → Word A (T+1)) (S : Set X) (family : Set (X → A))
    (hid : InjOn (fun g : X → A => fun x : S => g x) family)
    (variance : ComplexParameter A H d L → ℝ≥0) (hv : ∀ i, 0 < variance i)
    (hrep : ∀ f ∈ family, ∃ z : TrialDraw A H d L, StrictOn
      (fun x => (Model.fromWeights (sgdInitialWeights true z)).logits r (prompt x) (Fin.last T)) f)
    {failure : ℝ≥0∞} (hf : 0 < failure) :
    let P := Measure.infinitePi (fun _ : ℕ => independentGaussianDraws variance)
    ∃ N, ∀ n ≥ N, ∀ f : family,
      P {z | ∃ batch : ℕ → ℕ → List (LabeledPosition A), ∀ j ∈ Finset.range n,
        ¬classCertificate S family (fun x => f.val x) (restartScores r K (restartWsdRate stockRuns K) prompt z batch j)}
        < failure := by
  obtain ⟨J,p,hp,bound⟩ := adamW_restart_rate_bound r K _ (restartWsdRate_bounds stockRuns K)
    (restartWsdRate_total stockRuns K) prompt S family hid variance hv hrep
  obtain ⟨M,hM⟩ := uniform_restart_confidence p hp hf
  refine ⟨J+M,fun n hn f => ?_⟩
  have he : J+(n-J) = n := by omega
  exact he ▸ (bound f (n-J)).trans_lt (hM (n-J) (by omega) f)

/-- With probability one, for every target and every minibatch stream,
the verifier eventually retains a fixed trial whose strict readout is
correct on the whole domain. The initial stock runs may have any finite
budgets, and the subsequent WSD rescaling is independent of the task. -/
theorem adamW_restartWsd_stabilizes (r stockRuns : ℕ) (K : ℕ → ℕ)
    (prompt : X → Word A (T+1)) (S : Set X) (family : Set (X → A))
    (hid : InjOn (fun g : X → A => fun x : S => g x) family)
    (variance : ComplexParameter A H d L → ℝ≥0) (hv : ∀ i, 0 < variance i)
    (hrep : ∀ f ∈ family, ∃ z : TrialDraw A H d L, StrictOn
      (fun x => (Model.fromWeights (sgdInitialWeights true z)).logits r (prompt x) (Fin.last T)) f) :
    let P := Measure.infinitePi (fun _ : ℕ => independentGaussianDraws variance)
    ∀ᵐ z ∂P, ∀ f : family, ∀ batch : ℕ → ℕ → List (LabeledPosition A),
      let score := restartScores r K (restartWsdRate stockRuns K) prompt z batch
      let ok := fun j => classCertificate S family (fun x => f.val x) (score j)
      ∃ j, StrictOn (score j) f.val ∧ ∀ k, retained ok id (j+1+k) = some j := by
  let P := Measure.infinitePi (fun _ : ℕ => independentGaussianDraws variance)
  let score := restartScores r K (restartWsdRate stockRuns K) prompt (H := H) (d := d) (L := L)
  let ok (f : family) z batch j := classCertificate S family (fun x => f.val x) (score z batch j)
  obtain ⟨J,p,hp,bound⟩ := adamW_restart_rate_bound r K _ (restartWsdRate_bounds stockRuns K)
    (restartWsdRate_total stockRuns K) prompt S family hid variance hv hrep
  have hit (f : family) : ∀ᵐ z ∂P, ∀ batch, ∃ j, ok f z batch j := by
    rw [ae_iff]
    apply le_antisymm _ bot_le
    have hlt := ENNReal.sub_lt_self ENNReal.one_ne_top one_ne_zero (hp f).ne'
    apply le_of_tendsto_of_tendsto' tendsto_const_nhds (ENNReal.tendsto_pow_atTop_nhds_zero_of_lt_one hlt)
    intro m
    apply le_trans _ (bound f m)
    apply measure_mono
    intro z hz
    change ¬ (∀ batch, ∃ j, ok f z batch j) at hz
    push Not at hz
    obtain ⟨batch,h⟩ := hz
    exact ⟨batch,fun j _ => h j⟩
  filter_upwards [ae_all_iff.mpr hit] with z hz
  intro f batch
  obtain ⟨j,hj,_,hkeep⟩ := retained_settles (ok := ok f z batch) (z := id) (hz f batch)
  exact ⟨j,(classCertificate_iff S family hid f.val f.property _).mp hj,hkeep⟩

end Transformer.Grokking.FiniteRecovery.Argand
