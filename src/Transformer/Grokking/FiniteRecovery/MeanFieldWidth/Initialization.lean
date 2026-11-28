module

public import Transformer.Grokking.FiniteRecovery.MeanFieldWidth.Recovery
public import Transformer.Grokking.FiniteRecovery.MeanFieldWidth.Certificate

/-! Unbounded initializations. The certificate identity
`certificate_residual_of_init` needs every initial weight within a bound C₀,
which no Gaussian start satisfies surely. The width transfer is therefore
applied with its slope hypothesis only on the box of starts within C₀, and
the starts outside the box are lost from the probability of recovery. By
the union bound that loss is at most N · |P| · δ when every coordinate of
the initialization law leaves [-C₀, C₀] with probability at most δ. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.MeanFieldWidth

open MeasureTheory Finset Filter Transformer.Grokking.FiniteRecovery.ObservedConstraints

variable {P K J : Type*}

/-- Permanent recovery from the width transfer when the slope bound holds only
for starts in a measurable set E. The recovered set lies inside E. -/
theorem permanent_of_width_transfer_on {X Act R : Type*} [Fintype X] [Fintype Act] [Nonempty Act]
    [MeasurableSpace Act] [MeasurableSingletonClass Act] [Fintype R] [Fintype P] [Fintype K]
    [Fintype J] (A : Training P K) (μ : Measure (P → ℝ)) [IsProbabilityMeasure μ]
    {Γ : (J → ℝ) → ℝ} {χ : ℝ → (P → ℝ) → (P → ℝ) → J → ℝ} {η LG Lφ Cφ LΓ Lχ Mχ Cχ : ℝ}
    (h : Regular A Γ χ η LG Lφ Cφ LΓ Lχ Mχ Cχ) {N : ℕ} (hN : 0 < N) {B H n : ℕ} (hn : 0 < n)
    (score : (Fin N → P → ℝ) → X → Act → ℝ) (S : Finset X) (rule : R → Set (X → Act))
    (f : X → Act) (hf : ∀ j, f ∈ rule j)
    (hid : Set.InjOn (fun g : X → Act => fun x : S => g x) (⋂ j, rule j))
    (V : (Fin N → P → ℝ) → ℝ) (above : ∀ v, risk (tableLaw (score v)) S (fun x => f x) rule ≤ V v)
    {ρ r q rbar : ℝ} (hρ : 0 ≤ ρ) (hlt : ρ < 1) (hr : r < 1/2)
    (hlim : ∀ t ∈ Icc B H, ∀ u ∈ Set.Ioo (0:ℝ) 1, limitResidual A μ Γ χ t u ≤ rbar)
    (tail : (Fin N → P → ℝ) → ℝ) (htm : Measurable tail)
    (hti : Integrable tail (Measure.pi fun _ : Fin N => μ)) (htn : ∀ z, 0 ≤ tail z)
    (htail : ∀ z t, H < t → ∀ u ∈ Set.Ioo (0:ℝ) 1, residual A Γ χ z t u ≤ tail z)
    (small : max rbar 0 +
        widthError A.eps η LG Lφ Cφ LΓ Lχ Mχ Cχ (Fintype.card K) (Fintype.card J) N B H n +
        ∫ z, tail z ∂(Measure.pi fun _ : Fin N => μ) < (1-q)*((1-ρ)*(1/2-r)))
    {E : Set (Fin N → P → ℝ)} (hE : MeasurableSet E)
    (slope : ∀ z ∈ E, ∀ t, B ≤ t →
      let line := segmentRisk V (weightsAt A z t) (weightsAt A z (t+1) - weightsAt A z t)
      ContinuousOn line (Set.Icc 0 1) ∧ DifferentiableOn ℝ line (Set.Ioo 0 1) ∧
      ∀ u ∈ Set.Ioo (0:ℝ) 1, deriv line u + (1-ρ)*(V (weightsAt A z t) - r) ≤ residual A Γ χ z t u) :
    ∃ G ⊆ E, MeasurableSet G ∧
      q - (Measure.pi fun _ : Fin N => μ).real Eᶜ < (Measure.pi fun _ : Fin N => μ).real G ∧
      ∀ z ∈ G, ∀ᶠ t in atTop, StrictOn (score (weightsAt A z t)) f := by
  obtain ⟨hm, hi, hpos, hint, hle⟩ := drift_excess A μ h hN hn hlim tail htm hti htn htail
  refine permanent_of_observed_drift_on (Measure.pi fun _ : Fin N => μ) score S rule f hf hid V
    above (weightsAt A) (fun z t => weightsAt A z (t+1) - weightsAt A z t) (fun z t => by abel) B
    hρ hlt hr _ hm hi (ae_of_all _ hpos) (hint.trans_lt small) hE ?_
  intro z hz t ht
  obtain ⟨hc, hd, hs⟩ := slope z hz t ht
  exact ⟨hc, hd, fun u hu => by linarith [hs u hu, hle z t ht u hu]⟩

/-- Starts of N units whose every initial weight lies within C₀. -/
def box (N : ℕ) (C₀ : ℝ) : Set (Fin N → P → ℝ) := {z | ∀ i p, |z i p| ≤ C₀}

theorem measurableSet_box [Fintype P] (N : ℕ) (C₀ : ℝ) : MeasurableSet (box (P := P) N C₀) := by
  have e : box (P := P) N C₀ = ⋂ i, ⋂ p, {z | |z i p| ≤ C₀} := by
    ext z
    simp [box]
  rw [e]
  exact MeasurableSet.iInter fun i => MeasurableSet.iInter fun p => measurableSet_le
    (continuous_abs.measurable.comp ((measurable_pi_apply p).comp (measurable_pi_apply i)))
    measurable_const

/-- The union bound: if every coordinate of the initialization law leaves
[-C₀, C₀] with probability at most δ, a start of N units leaves the box
with probability at most N · |P| · δ. -/
theorem box_compl_le [Fintype P] (μ : Measure (P → ℝ)) [IsProbabilityMeasure μ] {N : ℕ}
    {C₀ δ : ℝ} (htail : ∀ p, μ.real {θ | C₀ < |θ p|} ≤ δ) :
    (Measure.pi fun _ : Fin N => μ).real (box N C₀)ᶜ ≤ N * Fintype.card P * δ := by
  have cover : (box (P := P) N C₀)ᶜ ⊆ ⋃ i, ⋃ p, {z : Fin N → P → ℝ | C₀ < |z i p|} := by
    intro z hz
    simp only [box, Set.mem_compl_iff, Set.mem_ofPred_eq, not_forall, not_le] at hz
    obtain ⟨i, p, hip⟩ := hz
    exact Set.mem_iUnion.2 ⟨i, Set.mem_iUnion.2 ⟨p, hip⟩⟩
  have marginal (i : Fin N) (p : P) :
      (Measure.pi fun _ : Fin N => μ).real {z | C₀ < |z i p|} = μ.real {θ | C₀ < |θ p|} := by
    have hs : MeasurableSet {θ : P → ℝ | C₀ < |θ p|} :=
      measurableSet_lt measurable_const (continuous_abs.measurable.comp (measurable_pi_apply p))
    exact congrArg ENNReal.toReal
      ((measurePreserving_eval (fun _ : Fin N => μ) i).measure_preimage hs.nullMeasurableSet)
  calc _ ≤ (Measure.pi fun _ : Fin N => μ).real (⋃ i, ⋃ p, {z : Fin N → P → ℝ | C₀ < |z i p|}) :=
        measureReal_mono cover
    _ ≤ ∑ i, ∑ p, (Measure.pi fun _ : Fin N => μ).real {z : Fin N → P → ℝ | C₀ < |z i p|} :=
        (measureReal_iUnion_fintype_le _).trans
          (Finset.sum_le_sum fun i _ => measureReal_iUnion_fintype_le _)
    _ ≤ ∑ _i : Fin N, ∑ _p : P, δ := by
        gcongr with i _ p _
        rw [marginal]
        exact htail p
    _ = N * Fintype.card P * δ := by
        simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
        ring

/-- Permanent recovery for the certificate F ∘ mean ψ from initial weights
that need not be bounded: if every coordinate of the initialization law
leaves [-C₀, C₀] with probability at most δ, recovery has probability above
q - N · |P| · δ, on starts within C₀. -/
theorem permanent_of_certificate {X Act R : Type*} [Fintype X] [Fintype Act] [Nonempty Act]
    [MeasurableSpace Act] [MeasurableSingletonClass Act] [Fintype R] [Fintype P] [Fintype K]
    [Fintype J] (A : Training P K) (μ : Measure (P → ℝ)) [IsProbabilityMeasure μ]
    (F : (J → ℝ) → ℝ) (ψ : (P → ℝ) → J → ℝ) (hF : Differentiable ℝ F) (hψ : Differentiable ℝ ψ)
    {η LG Lφ Cφ LΓ Lχ Mχ Cχ C₀ δ ρ r q rbar : ℝ} (hδ : ∀ p, μ.real {θ | C₀ < |θ p|} ≤ δ)
    (h : Regular A (certificateMap F ρ r) (certificateFeatures ψ (η * (max C₀ 120 / 10 + 12)))
      η LG Lφ Cφ LΓ Lχ Mχ Cχ) {N : ℕ} (hN : 0 < N) {B H n : ℕ} (hn : 0 < n)
    (score : (Fin N → P → ℝ) → X → Act → ℝ) (S : Finset X) (rule : R → Set (X → Act))
    (f : X → Act) (hf : ∀ j, f ∈ rule j)
    (hid : Set.InjOn (fun g : X → Act => fun x : S => g x) (⋂ j, rule j))
    (above : ∀ v, risk (tableLaw (score v)) S (fun x => f x) rule ≤ F (meanField ψ v))
    (hρ : 0 ≤ ρ) (hlt : ρ < 1) (hr : r < 1/2)
    (hlim : ∀ t ∈ Icc B H, ∀ u ∈ Set.Ioo (0:ℝ) 1, limitResidual A μ (certificateMap F ρ r)
      (certificateFeatures ψ (η * (max C₀ 120 / 10 + 12))) t u ≤ rbar)
    (tail : (Fin N → P → ℝ) → ℝ) (htm : Measurable tail)
    (hti : Integrable tail (Measure.pi fun _ : Fin N => μ)) (htn : ∀ z, 0 ≤ tail z)
    (htail : ∀ z t, H < t → ∀ u ∈ Set.Ioo (0:ℝ) 1, residual A (certificateMap F ρ r)
      (certificateFeatures ψ (η * (max C₀ 120 / 10 + 12))) z t u ≤ tail z)
    (small : max rbar 0 + widthError A.eps η LG Lφ Cφ LΓ Lχ Mχ Cχ (Fintype.card K)
        (Fintype.card (Fin 3 × J)) N B H n + ∫ z, tail z ∂(Measure.pi fun _ : Fin N => μ) <
      (1-q)*((1-ρ)*(1/2-r))) :
    ∃ G ⊆ box N C₀, MeasurableSet G ∧
      q - N * Fintype.card P * δ < (Measure.pi fun _ : Fin N => μ).real G ∧
      ∀ z ∈ G, ∀ᶠ t in atTop, StrictOn (score (weightsAt A z t)) f := by
  obtain ⟨G, hGE, hG, hq, hs⟩ := permanent_of_width_transfer_on A μ h hN hn score S rule f hf hid
    (fun v => F (meanField ψ v)) above hρ hlt hr hlim tail htm hti htn htail small
    (measurableSet_box N C₀)
    fun z hz t _ => certificate_slope A F ψ hF hψ h.eps_pos.le h.rate h.rate_le z hz ρ r t
  exact ⟨G, hGE, hG, by linarith [box_compl_le μ (N := N) hδ], hs⟩

section Examples

theorem certificateFeatures_zero (D : ℝ) :
    certificateFeatures (P := P) (fun _ (_ : J) => (0:ℝ)) D = fun _ _ _ _ => 0 := by
  funext u θ θ' q
  obtain ⟨k, j⟩ := q
  fin_cases k <;> simp [certificateFeatures]

/-- The identity training with the zero certificate is regular. -/
theorem identityCertificateRegular [Fintype J] (ρ r D : ℝ) :
    Regular identityTraining (certificateMap (fun _ : J → ℝ => (0:ℝ)) ρ r)
      (certificateFeatures (fun _ (_ : J) => (0:ℝ)) D) 1 1 0 0 0 0 0 0 where
  eps_pos := one_pos
  rate _ := ⟨zero_le_one, le_rfl⟩
  rate_le := by norm_num
  LG_nonneg := zero_le_one
  Lφ_nonneg := le_rfl
  LΓ_nonneg := le_rfl
  Lχ_nonneg := le_rfl
  grad := identityTraining_grad
  field := identityTraining_field
  field_bound _ _ := by simp [identityTraining]
  map _ _ _ _ := by simp [certificateMap]
  features _ _ _ _ _ := by simp [certificateFeatures_zero, l1]
  segment _ _ _ _ := by simp [certificateFeatures_zero, l1]
  feature_bound _ _ _ _ := by simp [certificateFeatures_zero]

example := permanent_of_width_transfer_on identityTraining (Measure.dirac 0) identityRegular
  (N := 2) two_pos (B := 1) (H := 3) (n := 4) (by norm_num)
  (fun _ (_ : Fin 1) (_ : Fin 1) => (0:ℝ)) ∅ (fun _ : Fin 1 => Set.univ) (fun _ => 0)
  (fun _ => Set.mem_univ _) (fun _ _ _ _ _ => Subsingleton.elim _ _) (fun _ => 0)
  (fun _ => by simp [risk]) (ρ := 0) (r := 0) (q := 0) (rbar := 0) le_rfl one_pos (by norm_num)
  (fun _ _ _ _ => by simp [limitResidual]) (fun _ => 0) measurable_const (integrable_const _)
  (fun _ => le_rfl) (fun _ _ _ _ _ => by simp [residual]) (by norm_num [widthError])
  (measurableSet_box 2 1) (fun _ _ _ _ => ⟨continuousOn_const, differentiableOn_const _, fun _ _ => by
    unfold segmentRisk
    simp [residual]⟩)
example := box_compl_le (P := Fin 1) (Measure.dirac 0) (N := 2) (C₀ := 1) (δ := 1)
  fun _ => measureReal_le_one
example := permanent_of_certificate identityTraining (Measure.dirac 0) (fun _ : Fin 1 → ℝ => (0:ℝ))
  (fun _ (_ : Fin 1) => (0:ℝ)) (differentiable_const _) (differentiable_const _) (C₀ := 1) (δ := 1)
  (fun _ => measureReal_le_one) (identityCertificateRegular 0 0 _) (N := 2) two_pos (B := 1)
  (H := 3) (n := 4) (by norm_num) (fun _ (_ : Fin 1) (_ : Fin 1) => (0:ℝ)) ∅
  (fun _ : Fin 1 => Set.univ) (fun _ => 0) (fun _ => Set.mem_univ _)
  (fun _ _ _ _ _ => Subsingleton.elim _ _) (fun _ => by simp [risk]) (q := 0) (rbar := 0) le_rfl
  one_pos (by norm_num) (fun _ _ _ _ => by simp [limitResidual, certificateMap]) (fun _ => 0)
  measurable_const (integrable_const _) (fun _ => le_rfl)
  (fun _ _ _ _ _ => by simp [residual, certificateMap]) (by norm_num [widthError])

end Examples

end Transformer.Grokking.FiniteRecovery.MeanFieldWidth
