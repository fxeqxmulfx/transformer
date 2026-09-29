/-
# Attention's forward pass and Frank-Wolfe — the transition law of the process

`IsSAProcess` states the transition identity for a fixed target set `A`.  The estimates of
`lem: first.phase` and `lem: metastab.1` need it for a target set that depends on the current
configuration (`x_i^{t+1}` lands outside the convex combinations of its own group), so here the
identity is extended to every measurable set of pairs `(x^t, x_i^{t+1})`, by a monotone class
argument on rectangles.
-/

import Transformer.FrankWolfe.Section5_Process

open scoped BigOperators ENNReal
open Real MeasureTheory

namespace Transformer
namespace FrankWolfe

variable {d n : ℕ}

/-- `S` is a measurable event of the trajectory up to time `t`: whether a path lies in `S` depends
only on its first `t + 1` configurations. -/
def PastDet (t : ℕ) (S : Set (ℕ → Idx n → EucSpace d)) : Prop :=
  MeasurableSet S ∧ ∀ x y : ℕ → Idx n → EucSpace d, (∀ s ≤ t, x s = y s) → (x ∈ S ↔ y ∈ S)

theorem PastDet.mono {t t' : ℕ} {S : Set (ℕ → Idx n → EucSpace d)} (hS : PastDet t S)
    (h : t ≤ t') : PastDet t' S :=
  ⟨hS.1, fun x y hxy => hS.2 x y fun s hs => hxy s (hs.trans h)⟩

theorem PastDet.inter {t : ℕ} {S T : Set (ℕ → Idx n → EucSpace d)} (hS : PastDet t S)
    (hT : PastDet t T) : PastDet t (S ∩ T) :=
  ⟨hS.1.inter hT.1, fun x y hxy => and_congr (hS.2 x y hxy) (hT.2 x y hxy)⟩

theorem pastDet_univ (t : ℕ) : PastDet t (Set.univ : Set (ℕ → Idx n → EucSpace d)) :=
  ⟨MeasurableSet.univ, fun _ _ _ => Iff.rfl⟩

/-- An event about the configuration at a time `s ≤ t` is an event of the past. -/
theorem pastDet_coord {s t : ℕ} (hst : s ≤ t) {B : Set (Idx n → EucSpace d)}
    (hB : MeasurableSet B) : PastDet t {x : ℕ → Idx n → EucSpace d | x s ∈ B} :=
  ⟨measurable_pi_apply s hB, fun x y hxy => by
    simp only [Set.mem_ofPred_eq, hxy s hst]⟩

theorem continuous_attWeight (β : ℝ) (i j : Idx n) :
    Continuous fun X : Idx n → EucSpace d => attWeight β X i j := by
  unfold attWeight
  refine Continuous.div (by fun_prop) (by fun_prop) fun X => ?_
  exact (Finset.sum_pos (fun _ _ => Real.exp_pos _) ⟨i, Finset.mem_univ i⟩).ne'

/-- **The transition identity for target sets that depend on the configuration.**  For every
measurable set `𝒜` of pairs (configuration at time `t`, position of particle `i` at time `t + 1`)
and every event `S` of the past,

  `ℙ[S ∩ {(x^t, x_i^{t+1}) ∈ 𝒜}] = ∫_S Σ_j attWeight β (x^t) i j · 𝟙_𝒜(x^t, (1-γ)x_i^t + γ x_j^t) dℙ`.

`IsSAProcess` is this for `𝒜 = B × A`, and the rectangles generate. -/
theorem IsSAProcess.measure_inter_prod {β γ : ℝ} {X₀ : Idx n → EucSpace d}
    {P : Measure (ℕ → Idx n → EucSpace d)} (hP : IsSAProcess β γ X₀ P) (t : ℕ) (i : Idx n)
    {S : Set (ℕ → Idx n → EucSpace d)} (hS : PastDet t S)
    {𝒜 : Set ((Idx n → EucSpace d) × EucSpace d)} (h𝒜 : MeasurableSet 𝒜) :
    P (S ∩ {x | (x t, x (t + 1) i) ∈ 𝒜}) =
      ∫⁻ x in S, ∑ j, ENNReal.ofReal (attWeight β (x t) i j) *
        𝒜.indicator (fun _ => (1 : ℝ≥0∞)) (x t, (1 - γ) • x t i + γ • x t j) ∂P := by
  obtain ⟨hprob, -, hstep⟩ := hP
  have hf : Measurable fun x : ℕ → Idx n → EucSpace d => (x t, x (t + 1) i) :=
    (measurable_pi_apply t).prodMk ((measurable_pi_apply i).comp (measurable_pi_apply (t + 1)))
  have hcomb : ∀ j, Measurable fun x : ℕ → Idx n → EucSpace d =>
      (1 - γ) • x t i + γ • x t j := fun j => by fun_prop
  have hg : ∀ j, Measurable fun x : ℕ → Idx n → EucSpace d =>
      (x t, (1 - γ) • x t i + γ • x t j) := fun j => (measurable_pi_apply t).prodMk (hcomb j)
  have hw : ∀ j, Measurable fun x : ℕ → Idx n → EucSpace d =>
      ENNReal.ofReal (attWeight β (x t) i j) := fun j =>
    ENNReal.measurable_ofReal.comp
      ((continuous_attWeight β i j).measurable.comp (measurable_pi_apply t))
  set μ := P.restrict S with hμ
  have hrect : ∀ (B : Set (Idx n → EucSpace d)) (A : Set (EucSpace d)), MeasurableSet B →
      MeasurableSet A → μ.map (fun x => (x t, x (t + 1) i)) (B ×ˢ A) =
        (∑ j, (μ.withDensity fun x => ENNReal.ofReal (attWeight β (x t) i j)).map
          (fun x => (x t, (1 - γ) • x t i + γ • x t j))) (B ×ˢ A) := by
    intro B A hB hA
    have hBm : MeasurableSet {x : ℕ → Idx n → EucSpace d | x t ∈ B} := measurable_pi_apply t hB
    have hAm : MeasurableSet {x : ℕ → Idx n → EucSpace d | x (t + 1) i ∈ A} :=
      hA.preimage ((measurable_pi_apply i).comp (measurable_pi_apply (t + 1)))
    have hSB : PastDet t (S ∩ {x | x t ∈ B}) := hS.inter (pastDet_coord le_rfl hB)
    -- the left side is the event `S ∩ {x^t ∈ B} ∩ {x_i^{t+1} ∈ A}`
    have hL : μ.map (fun x => (x t, x (t + 1) i)) (B ×ˢ A) =
        P ((S ∩ {x | x t ∈ B}) ∩ {x | x (t + 1) i ∈ A}) := by
      rw [Measure.map_apply hf (hB.prod hA), hμ, Measure.restrict_apply (hf (hB.prod hA))]
      congr 1
      ext x
      simp only [Set.mem_inter_iff, Set.mem_preimage, Set.mem_prod, Set.mem_ofPred_eq]
      tauto
    -- the right side, as an integral over `S ∩ {x^t ∈ B}`
    have hR : (∑ j, (μ.withDensity fun x => ENNReal.ofReal (attWeight β (x t) i j)).map
          (fun x => (x t, (1 - γ) • x t i + γ • x t j))) (B ×ˢ A) =
        ∫⁻ x in S ∩ {x | x t ∈ B}, ∑ j, ENNReal.ofReal (attWeight β (x t) i j) *
          A.indicator (fun _ => (1 : ℝ≥0∞)) ((1 - γ) • x t i + γ • x t j) ∂P := by
      have hVm : ∀ j, MeasurableSet {x : ℕ → Idx n → EucSpace d |
          (1 - γ) • x t i + γ • x t j ∈ A} := fun j => (hcomb j) hA
      have hmj : ∀ j, Measurable fun x : ℕ → Idx n → EucSpace d =>
          ENNReal.ofReal (attWeight β (x t) i j) *
            A.indicator (fun _ => (1 : ℝ≥0∞)) ((1 - γ) • x t i + γ • x t j) := fun j =>
        (hw j).mul ((measurable_const.indicator hA).comp (hcomb j))
      rw [Measure.finsetSum_apply, lintegral_finsetSum _ fun j _ => hmj j]
      refine Finset.sum_congr rfl fun j _ => ?_
      rw [Measure.map_apply (hg j) (hB.prod hA), withDensity_apply _ ((hg j) (hB.prod hA)), hμ,
        Measure.restrict_restrict ((hg j) (hB.prod hA))]
      have hset : (fun x : ℕ → Idx n → EucSpace d => (x t, (1 - γ) • x t i + γ • x t j)) ⁻¹'
          (B ×ˢ A) ∩ S = {x | (1 - γ) • x t i + γ • x t j ∈ A} ∩ (S ∩ {x | x t ∈ B}) := by
        ext x
        simp only [Set.mem_inter_iff, Set.mem_preimage, Set.mem_prod, Set.mem_ofPred_eq]
        tauto
      rw [hset, ← Measure.restrict_restrict (hVm j), ← lintegral_indicator (hVm j)]
      refine lintegral_congr fun x => ?_
      by_cases hx : (1 - γ) • x t i + γ • x t j ∈ A <;> simp [Set.indicator, hx]
    rw [hL, hR]
    -- the hypothesis on `IsSAProcess`, for `S ∩ {x^t ∈ B}` and `A`
    have h1 := hstep t i A _ hA hSB.1 hSB.2
    have hnn : ∀ (x : ℕ → Idx n → EucSpace d) (j : Idx n),
        0 ≤ attWeight β (x t) i j * A.indicator (fun _ => (1 : ℝ)) ((1 - γ) • x t i + γ • x t j) :=
      fun x j => mul_nonneg (attWeight_pos β (x t) i j).le (Set.indicator_nonneg (fun _ _ => zero_le_one) _)
    have hmeas : Measurable fun x : ℕ → Idx n → EucSpace d => ∑ j,
        attWeight β (x t) i j * A.indicator (fun _ => (1 : ℝ)) ((1 - γ) • x t i + γ • x t j) :=
      Finset.measurable_sum _ fun j _ =>
        ((continuous_attWeight β i j).measurable.comp (measurable_pi_apply t)).mul
          ((measurable_const.indicator hA).comp (hcomb j))
    have hint : Integrable (fun x : ℕ → Idx n → EucSpace d => ∑ j,
        attWeight β (x t) i j * A.indicator (fun _ => (1 : ℝ)) ((1 - γ) • x t i + γ • x t j))
        (P.restrict (S ∩ {x | x t ∈ B})) := by
      refine Integrable.of_bound hmeas.aestronglyMeasurable 1 (Filter.Eventually.of_forall fun x => ?_)
      rw [Real.norm_eq_abs, abs_of_nonneg (Finset.sum_nonneg fun j _ => hnn x j)]
      calc ∑ j, attWeight β (x t) i j *
            A.indicator (fun _ => (1 : ℝ)) ((1 - γ) • x t i + γ • x t j)
          ≤ ∑ j, attWeight β (x t) i j * 1 :=
            Finset.sum_le_sum fun j _ => mul_le_mul_of_nonneg_left
              (Set.indicator_le_self' (fun _ _ => zero_le_one) _) (attWeight_pos β (x t) i j).le
        _ = 1 := by simp [sum_attWeight]
    rw [← ENNReal.ofReal_toReal (measure_ne_top P _), h1,
      ofReal_integral_eq_lintegral_ofReal hint (Filter.Eventually.of_forall fun x =>
        Finset.sum_nonneg fun j _ => hnn x j)]
    refine lintegral_congr fun x => ?_
    rw [ENNReal.ofReal_sum_of_nonneg fun j _ => hnn x j]
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [ENNReal.ofReal_mul (attWeight_pos β (x t) i j).le]
    congr 1
    by_cases hx : (1 - γ) • x t i + γ • x t j ∈ A <;> simp [Set.indicator, hx]
  have key : μ.map (fun x => (x t, x (t + 1) i)) =
      ∑ j, (μ.withDensity fun x => ENNReal.ofReal (attWeight β (x t) i j)).map
        (fun x => (x t, (1 - γ) • x t i + γ • x t j)) := by
    have : IsFiniteMeasure (μ.map fun x => (x t, x (t + 1) i)) :=
      Measure.isFiniteMeasure_map _ _
    refine ext_of_generate_finite _ generateFrom_prod.symm isPiSystem_prod ?_ ?_
    · rintro _ ⟨B, hB, A, hA, rfl⟩
      exact hrect B A hB hA
    · simpa using hrect Set.univ Set.univ MeasurableSet.univ MeasurableSet.univ
  have hmj : ∀ j, Measurable fun x : ℕ → Idx n → EucSpace d =>
      ENNReal.ofReal (attWeight β (x t) i j) *
        𝒜.indicator (fun _ => (1 : ℝ≥0∞)) (x t, (1 - γ) • x t i + γ • x t j) := fun j =>
    (hw j).mul ((measurable_const.indicator h𝒜).comp (hg j))
  have hL : μ.map (fun x => (x t, x (t + 1) i)) 𝒜 = P (S ∩ {x | (x t, x (t + 1) i) ∈ 𝒜}) := by
    rw [Measure.map_apply hf h𝒜, hμ, Measure.restrict_apply (hf h𝒜), Set.inter_comm]
    rfl
  have hR : (∑ j, (μ.withDensity fun x => ENNReal.ofReal (attWeight β (x t) i j)).map
        (fun x => (x t, (1 - γ) • x t i + γ • x t j))) 𝒜 =
      ∫⁻ x in S, ∑ j, ENNReal.ofReal (attWeight β (x t) i j) *
        𝒜.indicator (fun _ => (1 : ℝ≥0∞)) (x t, (1 - γ) • x t i + γ • x t j) ∂P := by
    rw [Measure.finsetSum_apply, lintegral_finsetSum _ fun j _ => hmj j]
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [Measure.map_apply (hg j) h𝒜, withDensity_apply _ ((hg j) h𝒜), hμ,
      ← lintegral_indicator ((hg j) h𝒜)]
    refine lintegral_congr fun x => ?_
    by_cases hx : (x t, (1 - γ) • x t i + γ • x t j) ∈ 𝒜 <;> simp [Set.indicator, hx]
  rw [← hL, key, hR]

end FrankWolfe
end Transformer
