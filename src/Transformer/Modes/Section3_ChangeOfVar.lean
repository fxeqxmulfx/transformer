/-
# The number of modes of a Gaussian KDE — the change of variables `eq:qt`

`eq:qt` of arXiv:2412.09080v3 relates the density `p_t` of
`n^{-1/2} Σ (Gᵢ(t), Gᵢ'(t))` to the density `q_t` of `n^{-1/2} Σ Yᵢ(t)`.  The
two sums differ by the affine bijection `z ↦ Σ_t^{-1/2}(z - μ_t)`
(`scaledSum_singleY`), whose Jacobian is `(det Σ_t)^{-1/2}`
(`map_whitenAffine_volume`).

Source: arXiv:2412.09080v3, `eq:qt`.
-/

import Transformer.Modes.Section3_Standardized

open Real MeasureTheory ProbabilityTheory
open scoped ENNReal

namespace Transformer
namespace Modes

/-- The whitening `L` of `Section3_Hermite`, as the linear map of the matrix
`[[1/√a, 0], [-b/√(aD), a/√(aD)]]`. -/
noncomputable def whitenLin (a b d : ℝ) : (ℝ × ℝ) →ₗ[ℝ] ℝ × ℝ :=
  Matrix.toLin (Module.Basis.finTwoProd ℝ) (Module.Basis.finTwoProd ℝ)
    !![1 / √a, 0; -b / √(a * (a * d - b ^ 2)), a / √(a * (a * d - b ^ 2))]

theorem whitenLin_apply (a b d : ℝ) (z : ℝ × ℝ) : whitenLin a b d z = whiten a b d z := by
  rw [whitenLin, Matrix.toLin_finTwoProd_apply, whiten]
  ext <;> simp only <;> ring

theorem det_whitenLin {a b d : ℝ} (ha : 0 < a) (hD : 0 < a * d - b ^ 2) :
    LinearMap.det (whitenLin a b d) = (√(a * d - b ^ 2))⁻¹ := by
  rw [whitenLin, LinearMap.det_toLin, Matrix.det_fin_two_of, Real.sqrt_mul ha.le]
  have h1 : 0 < √a := Real.sqrt_pos.2 ha
  have h2 : 0 < √(a * d - b ^ 2) := Real.sqrt_pos.2 hD
  have h3 : √a ^ 2 = a := Real.sq_sqrt ha.le
  field_simp
  rw [h3]; ring

/-- **The Jacobian of `z ↦ Σ^{-1/2}(z - μ)`**: it scales Lebesgue measure by
`(det Σ)^{-1/2}`, so its push-forward of Lebesgue measure is `(det Σ)^{1/2}`
times Lebesgue measure. -/
theorem map_whitenAffine_volume {a b d : ℝ} (ha : 0 < a) (hD : 0 < a * d - b ^ 2)
    (μ : ℝ × ℝ) :
    Measure.map (fun z : ℝ × ℝ => whiten a b d (z.1 - μ.1, z.2 - μ.2)) volume =
      ENNReal.ofReal √(a * d - b ^ 2) • volume := by
  have hdet := det_whitenLin ha hD
  have hne : LinearMap.det (whitenLin a b d) ≠ 0 := by
    rw [hdet]; exact inv_ne_zero (Real.sqrt_pos.2 hD).ne'
  have hL : Measurable (whitenLin a b d) :=
    (LinearMap.continuous_of_finiteDimensional _).measurable
  have hcomp : (fun z : ℝ × ℝ => whiten a b d (z.1 - μ.1, z.2 - μ.2)) =
      whitenLin a b d ∘ fun z => z - μ := by
    funext z; rw [Function.comp_apply, whitenLin_apply]; rfl
  rw [hcomp, ← Measure.map_map hL (measurable_sub_const μ), map_sub_right_eq_self,
    Measure.map_linearMap_addHaar_eq_smul_addHaar _ hne, hdet, inv_inv,
    abs_of_nonneg (Real.sqrt_nonneg _)]

/-- Pushing a density forward along a measurable equivalence. -/
theorem map_withDensity_comp {α β : Type*} [MeasurableSpace α] [MeasurableSpace β]
    (e : α ≃ᵐ β) (μ : Measure α) (g : β → ℝ≥0∞) :
    (μ.withDensity (g ∘ e)).map e = (μ.map e).withDensity g := by
  ext s hs
  rw [e.map_apply, withDensity_apply _ (e.measurable hs), withDensity_apply _ hs, e.restrict_map,
    lintegral_map_equiv]
  rfl

/-- `z ↦ Σ^{-1/2}(z - μ)`, for `Σ = [[a, b], [b, d]]` positive definite, as a
measurable equivalence of `ℝ²`. -/
noncomputable def whitenAffine {a b d : ℝ} (ha : 0 < a) (hD : 0 < a * d - b ^ 2) (μ : ℝ × ℝ) :
    (ℝ × ℝ) ≃ᵐ ℝ × ℝ where
  toFun z := whiten a b d (z.1 - μ.1, z.2 - μ.2)
  invFun w := (μ.1 + √a * w.1, μ.2 + (√(a * (a * d - b ^ 2)) * w.2 + b * √a * w.1) / a)
  left_inv z := by
    have h1 : 0 < √a := Real.sqrt_pos.2 ha
    have h2 : 0 < √(a * (a * d - b ^ 2)) := Real.sqrt_pos.2 (by positivity)
    ext <;> simp only [whiten] <;> field_simp <;> ring
  right_inv w := by
    have h1 : 0 < √a := Real.sqrt_pos.2 ha
    have h2 : 0 < √(a * (a * d - b ^ 2)) := Real.sqrt_pos.2 (by positivity)
    ext <;> simp only [whiten] <;> field_simp <;> ring
  measurable_toFun := by
    change Measurable fun z : ℝ × ℝ => whiten a b d (z.1 - μ.1, z.2 - μ.2)
    unfold whiten; fun_prop
  measurable_invFun := by
    change Measurable fun w : ℝ × ℝ =>
      (μ.1 + √a * w.1, μ.2 + (√(a * (a * d - b ^ 2)) * w.2 + b * √a * w.1) / a)
    fun_prop

/-- **`n^{-1/2} Σ Yᵢ(t) = Σ_t^{-1/2}(n^{-1/2} Σ (Gᵢ, Gᵢ') - μ_t)`**, pointwise. -/
theorem scaledSum_singleY {β : ℝ} (hβ : 0 < β) (t : ℝ) {n : ℕ} (hn : 1 ≤ n) (X : Fin n → ℝ) :
    scaledSum n (fun i => singleY β t (X i)) =
      whitenAffine (sigmaFst_pos hβ t) (sigmaDet_pos hβ t) (muFst n β t, muSnd n β t)
        (sumGG' n β t X) := by
  set L := whitenLin (sigmaFst β t) (sigmaCov β t) (sigmaSnd β t)
  have hsq : √(n : ℝ) ^ 2 = n := Real.sq_sqrt (Nat.cast_nonneg _)
  have hpos : 0 < √(n : ℝ) := Real.sqrt_pos.2 (by exact_mod_cast hn)
  change (√(n : ℝ))⁻¹ • ∑ i, singleY β t (X i) = whiten _ _ _ _
  simp only [singleY, ← whitenLin_apply]
  rw [← map_sum, ← map_smul]
  congr 1
  ext
  · simp only [Prod.smul_fst, Prod.fst_sum, smul_eq_mul, sumGG', muFst,
      Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    field_simp
    rw [hsq]
  · simp only [Prod.smul_snd, Prod.snd_sum, smul_eq_mul, sumGG', muSnd,
      Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    field_simp
    rw [hsq]

/-- **Equation (eq:qt)**, as the change of variables it is:
`p_t(x, y) = (det Σ_t)^{-1/2} q_t(Σ_t^{-1/2}[(x, y) - μ_t])`, in the form
"`q` is a density of `n^{-1/2} Σ Yᵢ(t)` iff this `p` is a density of
`n^{-1/2} Σ (Gᵢ, Gᵢ')`".  arXiv:2412.09080v3, `eq:qt`.  The source presupposes that `q_t` exists, which
for a single summand it does not (`Y(t)` lives on a curve); stated as an
equivalence, the identity carries `eq:qt` in both directions without that. -/
theorem isDensityOf_scaledSum_iff {β : ℝ} (hβ : 0 < β) (t : ℝ) {n : ℕ} (hn : 1 ≤ n)
    (q : ℝ × ℝ → ℝ) :
    IsDensityOf (Measure.pi fun _ : Fin n => lawY β t) (scaledSum n) q ↔
      IsDensityOf (Measure.pi fun _ : Fin n => gaussianReal 0 1) (sumGG' n β t)
        (fun z => sigmaDet β t ^ (-(1 : ℝ) / 2) *
          q (whiten (sigmaFst β t) (sigmaCov β t) (sigmaSnd β t)
            (z.1 - muFst n β t, z.2 - muSnd n β t))) := by
  have ha := sigmaFst_pos hβ t
  have hD := sigmaDet_pos hβ t
  set e := whitenAffine ha hD (muFst n β t, muSnd n β t)
  set γn := Measure.pi fun _ : Fin n => gaussianReal 0 1
  set c := sigmaDet β t ^ (-(1 : ℝ) / 2)
  have hc : 0 < c := Real.rpow_pos_of_pos hD _
  have hS : Measurable (sumGG' n β t) := by unfold sumGG' bigG bigG'; fun_prop
  have hsc : Measurable (scaledSum n) := by unfold scaledSum; fun_prop
  -- The two laws differ by `e`.
  have hpush : (Measure.pi fun _ : Fin n => lawY β t).map (scaledSum n) = (γn.map (sumGG' n β t)).map e := by
    change (Measure.pi fun _ : Fin n => (gaussianReal 0 1).map (singleY β t)).map _ = _
    rw [← Measure.pi_map_pi (hμ := fun _ => (inferInstance : SigmaFinite (lawY β t)))
      (fun _ => (measurable_singleY β t).aemeasurable),
      Measure.map_map hsc (Measurable.of_eval fun i => (measurable_singleY β t).comp
        (measurable_pi_apply i) : Measurable fun (X : Fin n → ℝ) i => singleY β t (X i)),
      Measure.map_map e.measurable hS]
    congr 1
    funext X
    exact scaledSum_singleY hβ t hn X
  -- The density `p` is carried by `e` to `q`.
  have hwd : (volume.withDensity fun z => ENNReal.ofReal (c * q (e z))).map e =
      volume.withDensity fun w => ENNReal.ofReal (q w) := by
    have hsplit : (fun z => ENNReal.ofReal (c * q (e z))) =
        ENNReal.ofReal c • ((fun w => ENNReal.ofReal (q w)) ∘ e) := by
      funext z; rw [ENNReal.ofReal_mul hc.le]; rfl
    rw [hsplit, withDensity_smul' _ _ ENNReal.ofReal_ne_top, Measure.map_smul,
      map_withDensity_comp, show ((volume : Measure (ℝ × ℝ)).map e) = _ from
        map_whitenAffine_volume ha hD _, withDensity_smul_measure, smul_smul,
      ← ENNReal.ofReal_mul hc.le]
    have h1 : c * √(sigmaFst β t * sigmaSnd β t - sigmaCov β t ^ 2) = 1 := by
      rw [show sigmaFst β t * sigmaSnd β t - sigmaCov β t ^ 2 = sigmaDet β t from rfl,
        Real.sqrt_eq_rpow, ← Real.rpow_add hD]
      norm_num
    rw [h1, ENNReal.ofReal_one, one_smul]
    exact e.measurable.aemeasurable
  change _ ↔ IsDensityOf γn (sumGG' n β t) fun z => c * q (e z)
  constructor
  · rintro ⟨hq0, hq⟩
    refine ⟨fun z => mul_nonneg hc.le (hq0 _), ?_⟩
    rw [hpush, ← hwd] at hq
    simpa [MeasurableEquiv.map_symm_map] using congrArg (Measure.map e.symm) hq
  · rintro ⟨hp0, hp⟩
    refine ⟨fun w => ?_, by rw [hpush, hp, hwd]⟩
    have := hp0 (e.symm w)
    rw [e.apply_symm_apply] at this
    exact (mul_nonneg_iff_of_pos_left hc).1 this

/-- The hypotheses of `isDensityOf_scaledSum_iff` and `scaledSum_singleY` are
satisfiable, and so are those of `det_whitenLin`, `map_whitenAffine_volume` and
`whitenAffine`, at `Σ = I₂`. -/
example : (0 : ℝ) < 1 ∧ 1 ≤ 1 ∧ (0 : ℝ) < 1 * 1 - 0 ^ 2 := ⟨one_pos, le_rfl, by norm_num⟩

end Modes
end Transformer
