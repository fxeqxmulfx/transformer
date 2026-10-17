import BrownianMotion.Gaussian.BrownianMotion
import Transformer.Wiener.Basic
import Transformer.Wiener.Filtration

/-!
# A cylindrical Wiener process on `L²(δ_θ₀)` with an independent initial datum

*Homogenized Transformers*, arXiv:2604.01978v1, App. A.2 `sec: cylindrical.wiener` and §2.4
`note:probabilistic_setting` work on a complete probability space whose filtration is
right-continuous, carrying a cylindrical Wiener process and initial tokens independent of it. We
construct one for `L²(δ_θ₀)`, the noise of a single feature `θ₀`, with an initial datum of any law
`μ₀`. The sample space is the completion of `E × (ℝ≥0 → ℝ)` under `μ₀ ⊗ gaussianLimit`; the
Brownian motion is `brownian`, the continuous modification of the coordinate process under the
Gaussian projective limit `gaussianLimit`, from the Brownian motion project of R. Degenne,
E. Marion, P. Pfaffelhuber et al. (github.com/RemyDegenne/brownian-motion, Apache 2.0, ported in
`third_party/Stochastic` as `BrownianMotion.Gaussian.BrownianMotion`); the noise is
`W(t)[h] = (∫ h dδ_θ₀) B t`; and the filtration is `wienerFiltrationWith`, the usual augmentation
of the filtration of the datum and the noise. Nothing is assumed beyond `μ₀` being a probability
measure.
-/

open MeasureTheory ProbabilityTheory Filter
open scoped NNReal InnerProductSpace

namespace Transformer.Wiener

section Completion

variable {α : Type*} {mα : MeasurableSpace α} {μ : Measure α}

/-- The identity from the completion of `α` under `μ` to `α` is measurable. -/
theorem measurable_completion : @Measurable (NullMeasurableSpace α μ) α _ _ fun ω => ω :=
  (measurable_id (α := α)).nullMeasurable.measurable'

/-- The identity from the completion of `α` under `μ` to `α` has law `μ`. -/
theorem hasLaw_completion :
    HasLaw (𝓧 := α) (fun ω : NullMeasurableSpace α μ => ω) μ μ.completion where
  aemeasurable := measurable_completion.aemeasurable
  map_eq := Measure.ext fun s hs => by
    rw [Measure.map_apply measurable_completion hs]
    rfl

/-- Completing a measure does not change which σ-algebras are independent. -/
theorem indep_completion_iff {m₁ m₂ : MeasurableSpace α} :
    Indep (Ω := @NullMeasurableSpace α mα μ) m₁ m₂ μ.completion ↔ Indep m₁ m₂ μ :=
  (Indep_iff (Ω := @NullMeasurableSpace α mα μ) m₁ m₂ μ.completion).trans (Indep_iff m₁ m₂ μ).symm

end Completion

section Dirac

variable {Θ Ω : Type*} [MeasurableSpace Θ] {m : MeasurableSpace Ω} {P : Measure Ω}
  {ℱ : Filtration ℝ≥0 m} {B : ℝ≥0 → Ω → ℝ} (θ₀ : Θ)

/-- In `L²(δ_θ₀)`, `‖h‖ = |∫ h dδ_θ₀|`: an element is its value at `θ₀`. -/
theorem norm_eq_abs_integral_dirac (h : Lp ℝ 2 (Measure.dirac θ₀)) :
    ‖h‖ = |∫ θ, h θ ∂Measure.dirac θ₀| := by
  have hsq : ‖h‖ ^ 2 = (∫ θ, h θ ∂Measure.dirac θ₀) ^ 2 := by
    rw [← real_inner_self_eq_norm_sq, L2.inner_def,
      integral_dirac' _ _ ((Lp.stronglyMeasurable h).inner (Lp.stronglyMeasurable h)),
      integral_dirac' _ _ (Lp.stronglyMeasurable h), real_inner_self_eq_norm_sq, Real.norm_eq_abs,
      sq_abs]
  rwa [← sq_eq_sq₀ (norm_nonneg _) (abs_nonneg _), sq_abs]

/-- `h ↦ ∫ h dδ_θ₀` is linear on `L²(δ_θ₀)`. -/
theorem integral_dirac_smul_add (a : ℝ) (h g : Lp ℝ 2 (Measure.dirac θ₀)) :
    ∫ θ, (a • h + g) θ ∂Measure.dirac θ₀ =
      a * (∫ θ, h θ ∂Measure.dirac θ₀) + ∫ θ, g θ ∂Measure.dirac θ₀ := by
  have hi (k : Lp ℝ 2 (Measure.dirac θ₀)) : Integrable k (Measure.dirac θ₀) :=
    (Lp.memLp k).integrable one_le_two
  rw [integral_congr_ae ((Lp.coeFn_add _ _).trans ((Lp.coeFn_smul a h).add EventuallyEq.rfl)),
    integral_add' ((hi h).smul a) (hi g)]
  simp only [Pi.smul_apply, smul_eq_mul, integral_const_mul]

/-- App. A.2: a Brownian motion `B` of variance `1` drives the cylindrical Wiener process
`W(t)[h] = (∫ h dδ_θ₀) B t` on `L²(δ_θ₀)`. -/
theorem isCylindricalWiener_dirac (hB : IsBrownian ℱ P 1 B) :
    IsCylindricalWiener (Measure.dirac θ₀) ℱ P
      fun t h ω => (∫ θ, h θ ∂Measure.dirac θ₀) * B t ω where
  linear t a h g := ae_of_all _ fun ω => by
    simp only [integral_dirac_smul_add, Pi.add_apply, Pi.smul_apply, smul_eq_mul]
    ring
  brownian h := by
    have hn : ‖h‖₊ = ‖∫ θ, h θ ∂Measure.dirac θ₀‖₊ :=
      NNReal.eq (by simp [norm_eq_abs_integral_dirac])
    rw [hn, ← mul_one (_ ^ 2)]
    exact hB.smul _

end Dirac

variable (Θ : Type) [MeasurableSpace Θ] (θ₀ : Θ) (E : Type) [MeasurableSpace E] (μ₀ : Measure E)
  [IsProbabilityMeasure μ₀]

/-- App. A.2: `brownian`, read on the second factor of the completion of `E × (ℝ≥0 → ℝ)` under
`μ₀ ⊗ gaussianLimit`, is a Brownian motion. -/
theorem isBrownianReal_completion :
    IsBrownianReal (fun t (ω : NullMeasurableSpace (E × (ℝ≥0 → ℝ)) (μ₀.prod gaussianLimit)) =>
      brownian t (ω : E × (ℝ≥0 → ℝ)).2) (μ₀.prod gaussianLimit).completion where
  hasLaw I := (isBrownianReal_brownian.hasLaw I).comp
    ((⟨measurable_snd.aemeasurable, by rw [Measure.map_snd_prod, measure_univ, one_smul]⟩ :
      HasLaw Prod.snd gaussianLimit (μ₀.prod gaussianLimit)).comp hasLaw_completion)
  cont := ae_of_all _ fun _ => continuous_brownian _

/-- App. A.2, §2.4: some complete probability space, with a filtration `ℱ` satisfying the usual
conditions, carries a cylindrical Wiener process `W` on `L²(δ_θ₀)` and an `ℱ 0`-measurable initial
datum `x₀` of law `μ₀` independent of the noise; `ℱ` is the filtration `wienerFiltrationWith` of
`x₀` and `W`. -/
theorem exists_isCylindricalWiener_dirac :
    ∃ (Ω : Type) (m : MeasurableSpace Ω) (P : Measure Ω) (ℱ : Filtration ℝ≥0 m)
      (W : ℝ≥0 → Lp ℝ 2 (Measure.dirac θ₀) → Ω → ℝ) (x₀ : Ω → E),
      IsProbabilityMeasure P ∧ UsualConditions ℱ P ∧
      IsCylindricalWiener (Measure.dirac θ₀) ℱ P W ∧ Measurable[ℱ 0] x₀ ∧ HasLaw x₀ μ₀ P ∧
      IndepOfWiener x₀ W P ∧ ∃ (_ : P.IsComplete) (hx₀ : Measurable x₀)
        (hW : ∀ t h, Measurable (W t h)), ℱ = wienerFiltrationWith P x₀ hx₀ W hW := by
  let P₀ := μ₀.prod gaussianLimit
  let B : ℝ≥0 → NullMeasurableSpace (E × (ℝ≥0 → ℝ)) P₀ → ℝ := fun t ω => brownian t ω.2
  let x₀ : NullMeasurableSpace (E × (ℝ≥0 → ℝ)) P₀ → E := fun ω => ω.1
  have hπ := hasLaw_completion (μ := P₀)
  have hBR : IsBrownianReal B P₀.completion := isBrownianReal_completion E μ₀
  have hBm t : Measurable (B t) :=
    (measurable_brownian t).comp (measurable_snd.comp measurable_completion)
  have hF := hBR.toIsPreBrownianReal.isFilteredPreBrownian hBm
  let W : ℝ≥0 → Lp ℝ 2 (Measure.dirac θ₀) → NullMeasurableSpace (E × (ℝ≥0 → ℝ)) P₀ → ℝ :=
    fun t h ω => (∫ θ, h θ ∂Measure.dirac θ₀) * B t ω
  have hW : IsCylindricalWiener (Measure.dirac θ₀) _ P₀.completion W :=
    isCylindricalWiener_dirac θ₀ (isBrownian_one_iff.2 ⟨hBR, hF.stronglyAdapted, hF.indep⟩)
  have hx₀ : Measurable x₀ := measurable_fst.comp measurable_completion
  have hind : IndepOfWiener x₀ W P₀.completion :=
    indep_of_indep_of_le_right (indep_completion_iff.2 ((IndepFun_iff_Indep _ _ _).1
      (indepFun_prod (μ := μ₀) (ν := gaussianLimit) measurable_id measurable_id)))
      (iSup₂_le fun t h => ((measurable_const.mul (measurable_brownian t)).comp
        (comap_measurable _)).comap_le)
  have h0 : MeasurableSpace.comap x₀ inferInstance ≤
      wienerFiltrationWith P₀.completion x₀ hx₀ W hW.measurable 0 :=
    (le_sup_left (b := noiseFiltration W hW.measurable 0)).trans (le_augment (P := P₀.completion)
      (Filtration.const ℝ≥0 _ hx₀.comap_le ⊔ noiseFiltration W hW.measurable) 0)
  refine ⟨_, _, P₀.completion, wienerFiltrationWith P₀.completion x₀ hx₀ W hW.measurable, W, x₀,
    ⟨measure_univ (μ := P₀)⟩, usualConditions_augment _,
    isCylindricalWiener_wienerFiltrationWith hW hx₀ hind,
    Measurable.of_comap_le h0,
    (⟨measurable_fst.aemeasurable, by rw [Measure.map_fst_prod, measure_univ, one_smul]⟩ :
      HasLaw Prod.fst μ₀ P₀).comp hπ, hind, inferInstance, hx₀, hW.measurable, rfl⟩

/-- The hypothesis of `isCylindricalWiener_dirac` holds for `brownian` under `gaussianLimit` with
its natural filtration. -/
example :=
  have hF := isBrownianReal_brownian.toIsPreBrownianReal.isFilteredPreBrownian measurable_brownian
  isCylindricalWiener_dirac ()
    (isBrownian_one_iff.2 ⟨isBrownianReal_brownian, hF.stronglyAdapted, hF.indep⟩)

/-- The hypotheses of `isBrownianReal_completion` and `exists_isCylindricalWiener_dirac` hold for
one feature and a standard Gaussian initial datum. -/
example := And.intro (isBrownianReal_completion ℝ (gaussianReal 0 1))
  (exists_isCylindricalWiener_dirac Unit () ℝ (gaussianReal 0 1))

end Transformer.Wiener
