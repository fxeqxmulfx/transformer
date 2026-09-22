/-
# The number of modes of a Gaussian KDE — `Y(t)` is standardized

The claim after `eq:Yi` of arXiv:2412.09080v3 that `Y(t)` has mean `0` and
covariance `I₂` "by construction".  Given that `Σ_t` is positive definite
(`Section3_SigmaPos`), it is linear algebra on the five moments of
`(G(t), G'(t))`, each finite since both are bounded.

Source: arXiv:2412.09080v3, `eq:Yi` and the sentence after it.
-/

import Transformer.Modes.Section3_SigmaPos

open Real MeasureTheory ProbabilityTheory

namespace Transformer
namespace Modes

/-- `Σ_t^{-1/2}(G - 𝔼G, G' - 𝔼G')` is standardized, for `Σ_t` given by its
entries `a, b, d`. -/
theorem isStandardized_map_whiten {β : ℝ} (hβ : 0 < β) (t : ℝ) {a b d : ℝ}
    (ha : a = sigmaFst β t) (hb : b = sigmaCov β t) (hd : d = sigmaSnd β t) :
    IsStandardized ((gaussianReal 0 1).map
      fun x => whiten a b d (bigG β t x - meanG β t, bigG' β t x - meanG' β t)) := by
  have ha0 : 0 < a := ha ▸ sigmaFst_pos hβ t
  have hD : 0 < a * d - b ^ 2 := by rw [ha, hb, hd]; exact sigmaDet_pos hβ t
  have hs1 : √a ^ 2 = a := Real.sq_sqrt ha0.le
  have hs2 : √(a * (a * d - b ^ 2)) ^ 2 = a * (a * d - b ^ 2) := Real.sq_sqrt (by positivity)
  set m := meanG β t
  set m' := meanG' β t
  have hsG : sqMeanG β t = a + m ^ 2 := by rw [ha, sigmaFst]; ring
  have hsGH : mulMeanGG' β t = b + m * m' := by rw [hb, sigmaCov]; ring
  have hsHH : sqMeanG' β t = d + m' ^ 2 := by rw [hd, sigmaSnd]; ring
  set L : ℝ × ℝ → ℝ × ℝ := fun p => whiten a b d (p.1 - m, p.2 - m')
  have cL : Continuous L := by simp only [L, whiten]; fun_prop
  have hmeas : Measurable fun x => L (bigG β t x, bigG' β t x) :=
    cL.measurable.comp ((continuous_bigG β t).prodMk (continuous_bigG' β t)).measurable
  have hmap : ∀ φ : ℝ × ℝ → ℝ, Continuous φ → ∫ z, φ z ∂(gaussianReal 0 1).map
      (fun x => L (bigG β t x, bigG' β t x)) =
      ∫ x, φ (L (bigG β t x, bigG' β t x)) ∂gaussianReal 0 1 := fun φ hφ =>
    integral_map hmeas.aemeasurable hφ.aestronglyMeasurable
  set k := b * m - a * m'
  set s := √(a * (a * d - b ^ 2))
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · -- Bounded, as `L` is continuous and `(G, G')` takes values in a box.
    obtain ⟨C, hC⟩ := (isCompact_Icc (a := ((-(1 + 2 / β), -2) : ℝ × ℝ)) (b := (1 + 2 / β, 2))
      ).exists_bound_of_continuousOn cL.continuousOn
    rw [memLp_map_measure_iff aestronglyMeasurable_id hmeas.aemeasurable]
    refine MemLp.of_bound hmeas.aestronglyMeasurable C (ae_of_all _ fun x => hC _ ?_)
    have h1 := abs_le.1 (abs_bigG_le hβ t x)
    have h2 := abs_le.1 (abs_bigG'_le hβ t x)
    exact ⟨⟨h1.1, h2.1⟩, ⟨h1.2, h2.2⟩⟩
  · rw [hmap _ continuous_fst, integral_eq_of_quad hβ t (-m / √a) (1 / √a) 0 0 0 0
      fun x => by simp only [L, whiten]; ring]
    ring
  · rw [hmap _ continuous_snd, integral_eq_of_quad hβ t (k / s) (-b / s) (a / s) 0 0 0
      fun x => by simp only [L, whiten, k, s]; ring]
    simp only [k]; ring
  · rw [hmap (fun z => z.1 ^ 2) (by fun_prop), integral_eq_of_quad hβ t (m ^ 2 / √a ^ 2)
      (-2 * m / √a ^ 2) 0 (1 / √a ^ 2) 0 0 fun x => by simp only [L, whiten]; ring, hs1, hsG]
    field_simp; ring
  · rw [hmap (fun z => z.2 ^ 2) (by fun_prop), integral_eq_of_quad hβ t (k ^ 2 / s ^ 2)
      (-2 * b * k / s ^ 2) (2 * a * k / s ^ 2) (b ^ 2 / s ^ 2) (-2 * a * b / s ^ 2) (a ^ 2 / s ^ 2)
      fun x => by simp only [L, whiten, k, s]; ring, hs2, hsG, hsGH, hsHH]
    field_simp; simp only [k]; ring
  · rw [hmap (fun z => z.1 * z.2) (by fun_prop), integral_eq_of_quad hβ t
      (-m * k / (√a * s)) ((k + m * b) / (√a * s)) (-m * a / (√a * s)) (-b / (√a * s))
      (a / (√a * s)) 0 fun x => by simp only [L, whiten, k, s]; ring, hsG, hsGH]
    simp only [k]; ring

/-- **`Y(t)` is standardized**: mean `0` and covariance `I₂`, "by
construction".  arXiv:2412.09080v3, after `eq:Yi`.  The source says it of
`q_t`, the law of `n^{-1/2} Σ Yᵢ(t)`; for independent copies that is the same
statement about one `Y(t)`.  It needs `Σ_t` to be positive definite, which
holds for `β > 0` (`sigmaFst_pos`, `sigmaDet_pos`). -/
theorem isStandardized_lawY {β : ℝ} (hβ : 0 < β) (t : ℝ) : IsStandardized (lawY β t) :=
  isStandardized_map_whiten hβ t rfl rfl rfl

/-- The hypotheses of `isStandardized_map_whiten` and `isStandardized_lawY` are
satisfiable. -/
example : (0 : ℝ) < 1 ∧ sigmaFst 1 0 = sigmaFst 1 0 := ⟨one_pos, rfl⟩

end Modes
end Transformer
