module

public import Transformer.Modes.Section2_ParameterTransversality
public import Mathlib.Probability.Independence.Basic

/-!
# Gaussian sample data satisfy the original KDE parameter transversality premise

ArXiv:2412.09080v3, §2.1, `eq:Fn` and the expected
mode/upcrossing identity. Independent atomless real coordinates
almost surely avoid every fixed pair displacement. The proof uses
the actual pair pushforward of the original product law and conditions
on one coordinate; it requires no joint density of the KDE field.

For each positive original bandwidth, this excludes repeated samples
and squared pair distance 4/β simultaneously over the finite family.
The deterministic original sample-derivative calculation then gives
transversality of the whole field in its sample parameters at every
query zero, including singleton samples.

This establishes the regularity of the joint zero set. It does not
assume, or yet assert, that the query derivative is nonzero there:
that almost-sure conclusion requires the subsequent critical-value
argument for projection of this regular zero set to sample space.
-/

@[expose] public section

open scoped BigOperators NNReal ENNReal
open Real MeasureTheory ProbabilityTheory

namespace Transformer.Modes

variable {n : ℕ}

/-- Independent atomless real coordinates almost surely avoid each fixed pair displacement.
Source: arXiv:2412.09080v3, §2.1, the original independent continuous sample in `eq:Fn`. -/
theorem ae_pi_pair_sub_ne (μ : Idx n → Measure ℝ) [∀ i, IsProbabilityMeasure (μ i)]
    [∀ i, NullSingletonClass (μ i)] (i j : Idx n) (hij : i ≠ j) (c : ℝ) :
    ∀ᵐ X ∂Measure.pi μ, X i - X j ≠ c := by
  have hmeas : MeasurableSet {p : ℝ × ℝ | p.1 - p.2 ≠ c} :=
    (measurableSet_eq_fun (measurable_fst.sub measurable_snd) measurable_const).compl
  have hcond : ∀ᵐ y ∂μ j, ∀ᵐ x ∂μ i, x - y ≠ c := by
    apply ae_of_all
    intro y
    filter_upwards [(μ i).ae_ne (y + c)] with x hx
    intro he
    apply hx
    linarith
  have hprod : ∀ᵐ p ∂(μ i).prod (μ j), p.1 - p.2 ≠ c :=
    (Measure.ae_prod_iff_ae_ae hmeas).mpr ((Measure.ae_ae_comm hmeas).mpr hcond)
  have hind : iIndepFun (fun a (X : Idx n → ℝ) => X a) (Measure.pi μ) :=
    iIndepFun_pi (fun _ => measurable_id.aemeasurable)
  have hmap : Measure.map (fun X : Idx n → ℝ => (X i, X j)) (Measure.pi μ) = (μ i).prod (μ j) := by
    rw [(indepFun_iff_map_prod_eq_prod_map_map (measurable_pi_apply i).aemeasurable
      (measurable_pi_apply j).aemeasurable).mp (hind.indepFun hij),
      (measurePreserving_eval μ i).map_eq, (measurePreserving_eval μ j).map_eq]
  have hp : MeasurePreserving (fun X : Idx n → ℝ => (X i, X j)) (Measure.pi μ) ((μ i).prod (μ j)) :=
    ⟨(measurable_pi_apply i).prodMk (measurable_pi_apply j), hmap⟩
  exact hp.quasiMeasurePreserving.ae hprod

/-- The original critical squared pair distance has zero probability under any independent atomless sample.
Source: arXiv:2412.09080v3, §2.1, the critical-displacement factors of `eq:Fn`. -/
theorem ae_pi_pair_critical_distance_ne (μ : Idx n → Measure ℝ) [∀ i, IsProbabilityMeasure (μ i)]
    [∀ i, NullSingletonClass (μ i)] (i j : Idx n) (hij : i ≠ j) (β : ℝ) (hβ : 0 < β) :
    ∀ᵐ X ∂Measure.pi μ, β * (X i - X j) ^ 2 ≠ 4 := by
  let r := Real.sqrt (4 / β)
  have hr : β * r ^ 2 = 4 := by
    rw [Real.sq_sqrt (div_nonneg (by norm_num) hβ.le)]
    field_simp
  filter_upwards [ae_pi_pair_sub_ne μ i j hij r, ae_pi_pair_sub_ne μ i j hij (-r)] with X hp hn
  intro he
  have hs : (X i - X j) ^ 2 = r ^ 2 := mul_left_cancel₀ hβ.ne' (he.trans hr.symm)
  rcases sq_eq_sq_iff_eq_or_eq_neg.mp hs with heq | heq
  · exact hp heq
  · exact hn heq

/-- The actual Gaussian sample simultaneously avoids all repeated and critical-distance pairs.
Source: arXiv:2412.09080v3, §2.1, the independent standard Gaussians in `eq:Fn`. -/
theorem ae_gaussianSample_pair_gaps (n : ℕ) (β : ℝ) (hβ : 0 < β) :
    ∀ᵐ X ∂gaussianSample n, ∀ i j, i ≠ j → X i ≠ X j ∧ β * (X i - X j) ^ 2 ≠ 4 := by
  let : NullSingletonClass (gaussianReal 0 1) := nullSingletonClass_gaussianReal (by norm_num)
  apply ae_all_iff.mpr
  intro i
  apply ae_all_iff.mpr
  intro j
  by_cases hij : i = j
  · subst j
    exact ae_of_all _ fun _ h => (h rfl).elim
  · filter_upwards [ae_pi_pair_sub_ne (fun _ : Idx n => gaussianReal 0 1) i j hij 0,
      ae_pi_pair_critical_distance_ne (fun _ : Idx n => gaussianReal 0 1) i j hij β hβ] with X hne hgap
    intro h
    exact ⟨fun he => hne (by rw [he, sub_self]), hgap⟩

/-- The data on which the original sample transversality argument applies form an open set.
The inequalities exclude finitely many closed pair-distance loci.
Source: arXiv:2412.09080v3, §2.1, the original parameter space of `eq:Fn`. -/
theorem isOpen_fieldF_regular_data (n : ℕ) (β : ℝ) :
    IsOpen {X : Idx n → ℝ | ∀ i j, i ≠ j → X i ≠ X j ∧ β * (X i - X j) ^ 2 ≠ 4} := by
  simp only [Set.ofPred_forall]
  apply isOpen_iInter_of_finite
  intro i
  apply isOpen_iInter_of_finite
  intro j
  apply isOpen_iInter_of_finite
  intro hij
  exact (isClosed_eq (continuous_apply i) (continuous_apply j)).isOpen_compl.inter
    (isClosed_eq (continuous_const.mul (((continuous_apply i).sub (continuous_apply j)).pow 2)) continuous_const).isOpen_compl

/-- Each explicit pair exception is null for the original Gaussian sample.
Source: arXiv:2412.09080v3, §2.1, the regular-level exceptions for `eq:Fn`. -/
theorem gaussianSample_pair_exception_null (n : ℕ) (β : ℝ) (hβ : 0 < β)
    (i j : Idx n) (hij : i ≠ j) :
    gaussianSample n {X | X i = X j ∨ β * (X i - X j) ^ 2 = 4} = 0 := by
  have hnone : ∀ᵐ X ∂gaussianSample n, ¬ (X i = X j ∨ β * (X i - X j) ^ 2 = 4) := by
    filter_upwards [ae_gaussianSample_pair_gaps n β hβ] with X hX
    rintro (he | he)
    · exact (hX i j hij).1 he
    · exact (hX i j hij).2 he
  simpa only [not_not] using ae_iff.mp hnone

/-- Almost every actual Gaussian data set makes the original field transverse in sample coordinates at all query zeros.
Source: arXiv:2412.09080v3, §2.1, the regular-level premise for the expected mode/upcrossing identity. -/
theorem ae_gaussianSample_fieldF_transverse (hn : 0 < n) (β : ℝ) (hβ : 0 < β) :
    ∀ᵐ X ∂gaussianSample n, ∀ t : ℝ, fieldF β X t = 0 →
      ∃ i : Idx n, deriv (fun s => fieldF β (Function.update X i s) t) (X i) ≠ 0 := by
  filter_upwards [ae_gaussianSample_pair_gaps n β hβ] with X hX
  exact fun t ht => fieldF_sample_transverse hn β X hX t ht

/-- Jointly singular zeros of the original field have zero probability in the original sample space.
Source: arXiv:2412.09080v3, §2.1, `eq:Fn`, before the critical-value projection argument. -/
theorem gaussianSample_fieldF_joint_singular_null (hn : 0 < n) (β : ℝ) (hβ : 0 < β) :
    gaussianSample n {X | ∃ t : ℝ, fieldF β X t = 0 ∧
      ∀ i : Idx n, deriv (fun s => fieldF β (Function.update X i s) t) (X i) = 0} = 0 := by
  have hnone : ∀ᵐ X ∂gaussianSample n, ¬ ∃ t : ℝ, fieldF β X t = 0 ∧
      ∀ i : Idx n, deriv (fun s => fieldF β (Function.update X i s) t) (X i) = 0 := by
    filter_upwards [ae_gaussianSample_fieldF_transverse hn β hβ] with X hX
    rintro ⟨t, ht, hall⟩
    obtain ⟨i, hi⟩ := hX t ht
    exact hi (hall i)
  simpa only [not_not] using ae_iff.mp hnone

/-- Independent genuinely diffuse coordinates, distinct selected indices,
a positive bandwidth and a positive sample count realize the probabilistic premises together. -/
example : ∃ μ : Idx 3 → Measure ℝ,
    (∀ i, IsProbabilityMeasure (μ i)) ∧ (∀ i, NullSingletonClass (μ i)) ∧
    (0 : Idx 3) ≠ 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℕ) < 3 := by
  refine ⟨fun _ => gaussianReal 0 1, fun _ => inferInstance,
    fun _ => nullSingletonClass_gaussianReal (by norm_num), by norm_num, one_pos, by norm_num⟩

/-- The original regular-data open set is nonempty at positive bandwidth and two distinct samples.
Source: arXiv:2412.09080v3, §2.1, `eq:Fn`. -/
example : ∃ X : Idx 2 → ℝ, (0 : ℝ) < 1 ∧
    ∀ i j, i ≠ j → X i ≠ X j ∧ (1 : ℝ) * (X i - X j) ^ 2 ≠ 4 := by
  refine ⟨![0, 1], one_pos, ?_⟩
  intro i j hij
  fin_cases i <;> fin_cases j <;> norm_num at hij <;> norm_num

end Transformer.Modes
