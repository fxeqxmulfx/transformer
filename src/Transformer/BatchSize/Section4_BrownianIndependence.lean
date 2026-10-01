/-
# Independence of vector Brownian future increments and the joint past

arXiv:2506.12543v1, Section 4.3, equations (2)--(3).
The coefficients may depend on all state coordinates. Consequently each
driving increment must be independent of the joint past, not just of its
own coordinate's past.
-/

import Transformer.BatchSize.Section4_BrownianNoise

open MeasureTheory ProbabilityTheory
open scoped NNReal

noncomputable section

namespace Transformer.BatchSize

/-- Independent pairs on each coordinate probability space yield
independent full left/right vectors on the product space. This supplies
the joint-past independence needed by the SDEs in Section 4.3 (2)--(3). -/
theorem independent_pairs_product {ι Ω A B : Type*} [Fintype ι]
    [MeasurableSpace Ω] [MeasurableSpace A] [MeasurableSpace B]
    (μ : ι → Measure Ω) [∀ i, IsProbabilityMeasure (μ i)]
    (f : ι → Ω → A) (g : ι → Ω → B)
    (hf : ∀ i, Measurable (f i)) (hg : ∀ i, Measurable (g i))
    (hind : ∀ i, IndepFun (f i) (g i) (μ i)) :
    IndepFun (fun ω i => f i (ω i)) (fun ω i => g i (ω i)) (Measure.pi μ) := by
  let F : (ι → Ω) → (ι → A) := fun ω i => f i (ω i)
  let G : (ι → Ω) → (ι → B) := fun ω i => g i (ω i)
  have hF : Measurable F := Measurable.of_eval fun i => (hf i).comp (measurable_pi_apply i)
  have hG : Measurable G := Measurable.of_eval fun i => (hg i).comp (measurable_pi_apply i)
  have hH : Measurable (fun (ω : ι → Ω) i => (f i (ω i), g i (ω i))) := by
    apply Measurable.of_eval
    intro i
    exact ((hf i).comp (measurable_pi_apply i)).prodMk
      ((hg i).comp (measurable_pi_apply i))
  have hmap (i : ι) : (μ i).map (fun ω => (f i ω, g i ω)) =
      ((μ i).map (f i)).prod ((μ i).map (g i)) :=
    (hind i).map_prod_eq_prod_map_map (hf i).aemeasurable (hg i).aemeasurable
  apply (indepFun_iff_map_prod_eq_prod_map_map hF.aemeasurable hG.aemeasurable).mpr
  change (Measure.pi μ).map (fun ω => (F ω, G ω)) = _
  rw [show (fun ω => (F ω, G ω)) =
    (MeasurableEquiv.arrowProdEquivProdArrow A B ι) ∘
      (fun ω i => (f i (ω i), g i (ω i))) from rfl,
    ← Measure.map_map (MeasurableEquiv.arrowProdEquivProdArrow A B ι).measurable
      hH,
    Measure.pi_map_pi (fun i => ((hf i).prodMk (hg i)).aemeasurable)]
  simp_rw [hmap]
  rw [(measurePreserving_arrowProdEquivProdArrow A B ι
    (fun i => (μ i).map (f i)) (fun i => (μ i).map (g i))).map_eq]
  rw [Measure.pi_map_pi (fun i => (hf i).aemeasurable),
    Measure.pi_map_pi (fun i => (hg i).aemeasurable)]

/-- Joint nonvacuity of the product-independence hypotheses,
Section 4.3: two independent coordinates on unit probability spaces. -/
example : (∀ i : Fin 2, Measurable (fun _ : Unit => (i : ℝ))) ∧
    (∀ _ : Fin 2, Measurable (fun _ : Unit => (0 : ℝ))) ∧
    (∀ i : Fin 2, IndepFun (fun _ : Unit => (i : ℝ))
      (fun _ : Unit => (0 : ℝ)) (Measure.dirac ())) := by
  exact ⟨fun _ => measurable_const, fun _ => measurable_const,
    fun i => indepFun_const_left (i : ℝ) (fun _ : Unit => (0 : ℝ))⟩

/-- Entire future increments of the vector driver are independent of
its entire joint past, Section 4.3 (2)--(3). This remains valid when a
diffusion coefficient depends on multiple state coordinates. -/
theorem vectorBrownian_future_past_independent (d : ℕ) (s : ℝ≥0) :
    IndepFun
      (fun (ω : BrownianSample d) k t => coordinateBrownian k (s + t) ω - coordinateBrownian k s ω)
      (fun (ω : BrownianSample d) k (t : Set.Iic s) => coordinateBrownian k t ω)
      (brownianNoiseLaw d) := by
  exact independent_pairs_product (fun _ : Fin d => gaussianLimit)
    (fun _ ω t => brownian (s + t) ω - brownian s ω)
    (fun _ ω (t : Set.Iic s) => brownian t ω)
    (fun _ => Measurable.of_eval fun t => (measurable_brownian _).sub (measurable_brownian _))
    (fun _ => Measurable.of_eval fun t => measurable_brownian _)
    (fun _ => isBrownianReal_brownian.toIsPreBrownianReal.indepFun_shift s)

/-- A single future increment is independent of all vector coordinates
observed up to s, Section 4.3 (2)--(3), including random adapted coefficients. -/
theorem coordinateBrownian_increment_past_independent {d : ℕ} (k : Fin d)
    (s t : ℝ≥0) (hst : s ≤ t) :
    IndepFun (fun ω : BrownianSample d => coordinateBrownian k t ω - coordinateBrownian k s ω)
      (fun (ω : BrownianSample d) j (u : Set.Iic s) => coordinateBrownian j u ω)
      (brownianNoiseLaw d) := by
  have hm : Measurable (fun p : Fin d → ℝ≥0 → ℝ => p k (t - s)) := by
    simpa only [Function.comp_def] using
      (measurable_pi_apply (t - s)).comp (measurable_pi_apply k)
  have hi := (vectorBrownian_future_past_independent d s).comp
    (φ := fun p : Fin d → ℝ≥0 → ℝ => p k (t - s))
    (ψ := id) hm measurable_id
  simpa only [Function.comp_def, id_eq, add_tsub_cancel_of_le hst] using hi

/-- Joint nonvacuity of the increment's time-order hypothesis,
Section 4.3 (2)--(3). -/
example : (1 : ℝ≥0) ≤ 2 := by norm_num

end Transformer.BatchSize
