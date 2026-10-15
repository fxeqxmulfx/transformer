/-
# Original Gaussian row pairs and nonzero bilinear queries

ArXiv:2604.01978v1, §2.3.3, assumption (G),
`eq: tformers.at.initialization`, and §7.3, `lemma:Laplace_method`.
The actual original rows from different Gaussian entry blocks have
their prescribed independent product law. The construction retains
the source's original matrix order `A = WW′ᵀ`.

Conditionally on a nonzero first row, its dot product with the second
row has a nondegenerate scalar Gaussian law, so it is almost surely
nonzero. The first row itself is almost surely nonzero in positive
dimension. Fubini then proves the actual two-row dot product nonzero
without assigning a Gaussian law to this bilinear random variable.

All hypotheses of independence are proved on the original entry
ensemble by selecting disjoint finite blocks. No assertion that a
product matrix is invertible or has a Gaussian law enters the proof.
-/

import Transformer.Homogenized.Section7_GaussianRowMoments

open Real MeasureTheory ProbabilityTheory
open scoped BigOperators NNReal

namespace Transformer.Homogenized

/-- A genuine nondegenerate scalar Gaussian does not hit zero.
Source: arXiv:2604.01978v1, §7.3, the Gaussian query normalization. -/
theorem ae_gaussianReal_ne_zero {μ : ℝ} {v : ℝ≥0} (hv : v ≠ 0) :
    ∀ᵐ z ∂gaussianReal μ v, z ≠ 0 := by
  let : NullSingletonClass (gaussianReal μ v) := nullSingletonClass_gaussianReal hv
  rw [ae_iff]
  simpa only [not_not, Set.ofPred_eq_eq_singleton] using measure_singleton (μ := gaussianReal μ v) (0 : ℝ)

/-- The original standard row is almost surely nonzero in positive
dimension. Source: arXiv:2604.01978v1, initialization (G). -/
theorem ae_gaussianRow_ne_zero {d : ℕ} (hd : 0 < d) :
    ∀ᵐ z ∂gaussianRowLaw d, z ≠ 0 := by
  let i : Fin d := ⟨0, hd⟩
  have hcoord : ∀ᵐ z ∂gaussianRowLaw d, z i ≠ 0 := by
    have h := ae_gaussianReal_ne_zero (μ := 0) (v := 1) one_ne_zero
    rw [← gaussianRowLaw_map_coordinate i] at h
    exact (ae_map_iff (measurable_pi_apply i).aemeasurable
      (measurableSet_singleton (0 : ℝ)).compl).mp h
  filter_upwards [hcoord] with z hz
  intro he
  exact hz (congrFun he i)

/-- The actual conditional variance is positive for a nonzero row.
Source: arXiv:2604.01978v1, §7.3, `lemma:Laplace_method`. -/
theorem sum_gaussianRow_coeff_sq_pos {d : ℕ} (g : Fin d → ℝ) (hg : g ≠ 0) :
    0 < ∑ i : Fin d, g i ^ 2 := by
  have hn : ∃ i, g i ≠ 0 := by
    by_contra! h
    apply hg
    funext i
    exact h i
  obtain ⟨i, hi⟩ := hn
  exact (sq_pos_of_ne_zero hi).trans_le
    (Finset.single_le_sum (fun j hj => sq_nonneg (g j)) (Finset.mem_univ i))

/-- With the first original row fixed and nonzero, its dot product
with the second row is almost surely nonzero. Source:
arXiv:2604.01978v1, §7.3, `lemma:Laplace_method`. -/
theorem ae_gaussianRow_dot_ne_zero {d : ℕ} (g : Fin d → ℝ) (hg : g ≠ 0) :
    ∀ᵐ z ∂gaussianRowLaw d, dotProduct g z ≠ 0 := by
  have hv : (∑ i : Fin d, g i ^ 2).toNNReal ≠ 0 :=
    (Real.toNNReal_pos.mpr (sum_gaussianRow_coeff_sq_pos g hg)).ne'
  exact (gaussianRow_linear_hasLaw g).ae_iff (by fun_prop) |>.mpr (ae_gaussianReal_ne_zero hv)

/-- The original bilinear query coordinate is almost surely nonzero.
Source: arXiv:2604.01978v1, §7.3, the normalized query in
`lemma:Laplace_method`, under the actual product initialization. -/
theorem ae_gaussianRow_pair_dot_ne_zero {d : ℕ} (hd : 0 < d) :
    ∀ᵐ p ∂(gaussianRowLaw d).prod (gaussianRowLaw d), dotProduct p.1 p.2 ≠ 0 := by
  apply Measure.ae_prod_iff_ae_ae
    ((measurableSet_eq_fun
      (by dsimp only [dotProduct]; fun_prop : Measurable
        (fun p : (Fin d → ℝ) × (Fin d → ℝ) => dotProduct p.1 p.2)) measurable_const).compl) |>.mpr
  filter_upwards [ae_gaussianRow_ne_zero hd] with g hg
  exact ae_gaussianRow_dot_ne_zero g hg

/-- An actual row of an original Gaussian entry block.
Source: arXiv:2604.01978v1, `eq: tformers.at.initialization`. -/
def originalGaussianRow {d : ℕ} (b : Fin 3) (i : Fin d)
    (z : GaussIdx d → ℝ) : Fin d → ℝ := fun q => z (b, i, q)

/-- An original row is measurable on the original entry ensemble.
Source: arXiv:2604.01978v1, initialization (G). -/
theorem measurable_originalGaussianRow {d : ℕ} (b : Fin 3) (i : Fin d) :
    Measurable (originalGaussianRow b i) :=
  Measurable.of_eval (fun q => measurable_pi_apply (b, i, q))

/-- The actual row pushforward is its prescribed standard product
law. Source: arXiv:2604.01978v1, initialization (G) at unit variance. -/
theorem originalGaussianRow_map {d : ℕ} (b : Fin 3) (i : Fin d) :
    (gaussEnsemble d 1 1).map (originalGaussianRow b i) = gaussianRowLaw d := by
  have hind : iIndepFun (fun p : GaussIdx d => fun z : GaussIdx d → ℝ => z p)
      (gaussEnsemble d 1 1) :=
    iIndepFun_pi (X := fun _ : GaussIdx d => id) (fun _ => aemeasurable_id)
  have hinj : Function.Injective (fun q : Fin d => ((b, i, q) : GaussIdx d)) := by
    intro q r h
    exact congrArg (fun p : GaussIdx d => p.2.2) h
  have hrow := (hind.precomp hinj).map_fun_eq_pi_map
    (fun q => (measurable_pi_apply (b, i, q)).aemeasurable)
  change (gaussEnsemble d 1 1).map (originalGaussianRow b i) = _ at hrow
  rw [hrow]
  have hm (q : Fin d) : (gaussEnsemble d 1 1).map (fun z => z (b, i, q)) = gaussianReal 0 1 := by
    simpa only [gaussBlock_apply, entryVar, one_pow, ite_self] using
      map_gaussBlock (d := d) 1 1 b i q
  simp only [hm]
  rfl

/-- Original rows from distinct original entry blocks are independent.
Source: arXiv:2604.01978v1, initialization (G). -/
theorem indepFun_originalGaussianRows {d : ℕ} (b c : Fin 3) (hbc : b ≠ c) (i j : Fin d) :
    IndepFun (originalGaussianRow b i) (originalGaussianRow c j) (gaussEnsemble d 1 1) := by
  classical
  let S := Finset.univ.filter (fun p : GaussIdx d => p.1 = b ∧ p.2.1 = i)
  let T := Finset.univ.filter (fun p : GaussIdx d => p.1 = c ∧ p.2.1 = j)
  have hST : Disjoint S T := Finset.disjoint_filter.mpr
    (fun p hp hb hc => hbc (hb.1.symm.trans hc.1))
  have hind : iIndepFun (fun p : GaussIdx d => fun z : GaussIdx d → ℝ => z p)
      (gaussEnsemble d 1 1) :=
    iIndepFun_pi (X := fun _ : GaussIdx d => id) (fun _ => aemeasurable_id)
  let f : (S → ℝ) → Fin d → ℝ := fun z q => z ⟨(b, i, q), by simp [S]⟩
  let g : (T → ℝ) → Fin d → ℝ := fun z q => z ⟨(c, j, q), by simp [T]⟩
  have h := (hind.indepFun_finset S T hST (fun p => measurable_pi_apply p)).comp
    (by fun_prop : Measurable f) (by fun_prop : Measurable g)
  exact h

/-- The actual pair pushforward equals the original independent row
product. Source: arXiv:2604.01978v1, §7.3,
the actual two Gaussian factors of `lemma:Laplace_method`. -/
theorem originalGaussianRow_pair_map {d : ℕ} (b c : Fin 3) (hbc : b ≠ c) (i j : Fin d) :
    (gaussEnsemble d 1 1).map (fun z => (originalGaussianRow b i z, originalGaussianRow c j z)) =
      (gaussianRowLaw d).prod (gaussianRowLaw d) := by
  rw [(indepFun_originalGaussianRows b c hbc i j).map_prod_eq_prod_map_map
    (measurable_originalGaussianRow b i).aemeasurable (measurable_originalGaussianRow c j).aemeasurable,
    originalGaussianRow_map, originalGaussianRow_map]

/-- Nonzero scalar variance, positive dimension, a nonzero actual
fixed row and two distinct original Gaussian blocks hold jointly.
The row indices in the preceding theorems may both be zero. -/
example : (1 : ℝ≥0) ≠ 0 ∧ (0 : ℕ) < 2 ∧ (![1, 2] : Fin 2 → ℝ) ≠ 0 ∧
    (1 : Fin 3) ≠ 2 := by
  refine ⟨one_ne_zero, by decide, ?_, by decide⟩
  intro h
  have he := congrFun h 0
  norm_num at he

end Transformer.Homogenized
