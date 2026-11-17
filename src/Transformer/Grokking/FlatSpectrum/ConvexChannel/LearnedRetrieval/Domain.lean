module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRetrieval.Basic
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Domain

/-!
# Convex functional capacity with complete learned retrieval

Source: `review/flat_spectrum.md`, CM111. A previously learned base is retained
beside every admissible new retrieval head. The convexity concerns the whole
sequence-score function, not independent embedding or relative-score arrays.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRetrieval

open scoped BigOperators

variable {O E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℂ E]

/-- All bounded learned retrieval heads, plus the retained complete base,
CM111. This is not a fixed finite feature dictionary. -/
def generators (B : ℝ) (base : List O → O → ℝ) : Set (List O → O → ℝ) :=
  {base} ∪ {r | ∃ p : Parameters O E, Unit p ∧ r = B • score p}

/-- Joint function domain, CM111. The new learned head count is not fixed. -/
def domain (B : ℝ) (base : List O → O → ℝ) : Set (List O → O → ℝ) :=
  convexHull ℝ (generators (E := E) B base)

/-- The functional domain is convex, CM111; factor training is separate. -/
theorem domain_convex (B : ℝ) (base : List O → O → ℝ) :
    Convex ℝ (domain (E := E) B base) := convex_convexHull ℝ _

/-- Every retained base stays present, CM111. -/
theorem base_mem (B : ℝ) (base : List O → O → ℝ) : base ∈ domain (E := E) B base :=
  subset_convexHull ℝ _ (Or.inl rfl)

/-- Any complete learned admissible head is an eligible generator, CM111. -/
theorem head_mem (B : ℝ) (base : List O → O → ℝ) (p : Parameters O E) (hp : Unit p) :
    B • score p ∈ domain (E := E) B base :=
  subset_convexHull ℝ _ (Or.inr ⟨p, hp, rfl⟩)

/-- Every finite actual mixture is feasible, CM111, including zero weights
and duplicate generators. -/
theorem finite_feasible {J : Type*} [Fintype J] (B : ℝ) (base : List O → O → ℝ)
    (w : J → ℝ) (r : J → List O → O → ℝ) (hw : ∀ j, 0 ≤ w j)
    (hs : ∑ j, w j = 1) (hr : ∀ j, r j ∈ generators (E := E) B base) :
    (∑ j, w j • r j) ∈ domain (E := E) B base :=
  mem_convexHull_of_exists_fintype w r hw hs hr rfl

/-- Every generator has the same capacity as a bounded retained base,
CM111. Learned relative matching changes the prediction, not this bound. -/
theorem generator_bound (B : ℝ) (hB : 0 ≤ B) (base : List O → O → ℝ)
    (hb : ∀ xs c, |base xs c| ≤ B) (r : List O → O → ℝ)
    (hr : r ∈ generators (E := E) B base) (xs : List O) (c : O) : |r xs c| ≤ B := by
  rcases hr with rfl | ⟨p, hp, rfl⟩
  · exact hb xs c
  · simpa only [Pi.smul_apply, smul_eq_mul, abs_mul, abs_of_nonneg hB, mul_one] using
      mul_le_mul_of_nonneg_left (score_bound p hp xs c) hB

/-- Every joint-domain score obeys the complete capacity bound, CM111. -/
theorem domain_bound (B : ℝ) (hB : 0 ≤ B) (base : List O → O → ℝ)
    (hb : ∀ xs c, |base xs c| ≤ B) (r : List O → O → ℝ)
    (hr : r ∈ domain (E := E) B base) (xs : List O) (c : O) : |r xs c| ≤ B := by
  obtain ⟨J, hJ, w, z, hw, hs, hz, rfl⟩ := mem_convexHull_iff_exists_fintype.mp hr
  simp only [Finset.sum_apply, Pi.smul_apply, smul_eq_mul]
  calc
    |∑ j, w j * z j xs c| ≤ ∑ j, |w j * z j xs c| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ j, w j * B := by
      apply Finset.sum_le_sum
      intro j hj
      rw [abs_mul, abs_of_nonneg (hw j)]
      exact mul_le_mul_of_nonneg_left (generator_bound B hB base hb (z j) (hz j) xs c) (hw j)
    _ = B := by rw [← Finset.sum_mul, hs, one_mul]

/-- Adjoining a newly learned head preserves the same functional domain,
CM111. No efficient globally optimal head search is asserted. -/
theorem step_feasible (B eta : ℝ) (base r : List O → O → ℝ) (p : Parameters O E)
    (hr : r ∈ domain (E := E) B base) (hp : Unit p) (he : 0 ≤ eta) (he1 : eta ≤ 1) :
    (1 - eta) • r + eta • (B • score p) ∈ domain (E := E) B base :=
  domain_convex B base hr (head_mem B base p hp) (by linarith) he (by ring)

/-- Ordinary complete-class CE is convex in the joint function, CM111. -/
theorem loss_convex [Fintype O] (B : ℝ) (base : List O → O → ℝ) (xs : List O) (c : O) :
    ConvexOn ℝ (domain (E := E) B base)
      (fun r => Landscape.Neural.crossEntropy (r xs) c) := by
  have h := (ConvexRecall.convex_crossEntropy c).comp_linearMap (LearnedRecurrent.evaluation xs)
  have hu : ConvexOn ℝ Set.univ
      (fun r : List O → O → ℝ => Landscape.Neural.crossEntropy (r xs) c) := by
    simpa only [Set.preimage_univ, Function.comp_def, LearnedRecurrent.evaluation,
      LinearMap.coe_mk, AddHom.coe_mk] using h
  exact hu.subset (Set.subset_univ _) (domain_convex B base)

omit [NormedAddCommGroup E] [InnerProductSpace ℂ E] in
/-- A retained bounded base can oppose a new head by at most its complete
allocated capacity, CM111. This bounds the actual mixed output margin. -/
theorem retained_margin (B eta m bt bc zt zc : ℝ) (ht : |bt| ≤ B) (hc : |bc| ≤ B)
    (he : 0 ≤ eta) (he1 : eta ≤ 1) (hm : m ≤ zt - zc) :
    eta * m - 2 * (1 - eta) * B ≤
      ((1 - eta) * bt + eta * zt) - ((1 - eta) * bc + eta * zc) := by
  have hb : -2 * B ≤ bt - bc := by
    have ht' := abs_le.mp ht
    have hc' := abs_le.mp hc
    linarith
  have hbase := mul_le_mul_of_nonneg_left hb (show 0 ≤ 1 - eta by linarith)
  have hhead := mul_le_mul_of_nonneg_left hm he
  nlinarith

example : (3 / 4 : ℝ) * 2 - 2 * (1 - 3 / 4) * 1 ≤
    ((1 - 3 / 4) * (-1) + (3 / 4) * 1) - ((1 - 3 / 4) * 1 + (3 / 4) * (-1)) :=
  retained_margin 1 (3 / 4) 2 (-1) 1 1 (-1)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)

example : (1 : ℝ) • score (zeroParameters : Parameters Bool ℂ) ∈ domain (E := ℂ) 1 0 :=
  head_mem _ _ _ zero_unit

example : (∑ _ : Fin 1, (1 : ℝ) • (0 : List Bool → Bool → ℝ)) ∈ domain (E := ℂ) 1 0 :=
  finite_feasible _ _ (fun _ => 1) (fun _ => 0) (by norm_num) (by simp)
    (fun _ => Or.inl rfl)

example : |(0 : List Bool → Bool → ℝ) [false, true] true| ≤ 1 :=
  generator_bound (E := ℂ) _ (by norm_num) 0 (by simp) 0 (Or.inl rfl) _ _

example : |(0 : List Bool → Bool → ℝ) [false, true] true| ≤ 1 :=
  domain_bound (E := ℂ) _ (by norm_num) 0 (by simp) 0 (base_mem _ _) _ _

example : (1 - (1 / 2 : ℝ)) • (0 : List Bool → Bool → ℝ) +
    (1 / 2 : ℝ) • ((1 : ℝ) • score (zeroParameters : Parameters Bool ℂ)) ∈
    domain (E := ℂ) 1 0 :=
  step_feasible _ _ _ _ _ (base_mem _ _) zero_unit (by norm_num) (by norm_num)

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRetrieval
