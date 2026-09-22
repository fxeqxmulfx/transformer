/-
# Homogenized Transformers — `f` and `g` are well defined

The first half of `thm:clustering_random_init` of arXiv:2604.01978v1: the two
expectations `f(γ)`, `g(γ)` of `eq:f_g_def_selfcontained` depend on neither the
simplex configuration realizing the overlap `γ` nor the indices.

Two simplex configurations with the same `γ` have the same Gram matrix, so
they differ by an orthogonal `O`; rotating the tokens is conjugating `A` by
`O`, under which the Gaussian head law is invariant (`HeadLawRotation.lean`).
A permutation of the tokens is again a simplex configuration with the same
`γ`, which moves the indices.

Source: arXiv:2604.01978v1, `eq:f_g_def_selfcontained` and the proof of
`lem:drift_on_simplex_selfcontained`.
-/

import Transformer.Homogenized.Simplex
import Transformer.Homogenized.HeadLawRotation

open scoped BigOperators NNReal
open Real MeasureTheory ProbabilityTheory

namespace Transformer
namespace Homogenized

/-- The law of a Gaussian head is a probability measure. -/
theorem isProbabilityMeasure_of_isGaussianHeadLaw {d : ℕ} {σV σA : ℝ≥0}
    {ρ : Measure (HeadParam d)} (hρ : IsGaussianHeadLaw d σV σA ρ) : IsProbabilityMeasure ρ := by
  obtain ⟨Ω, _, P, Vr, Wr, Wr', _, hV, hW, hW', -, -, -, -, rfl⟩ := hρ
  have hm : Measurable fun ω => Wr ω * (Wr' ω).transpose :=
    Measurable.of_eval fun i => Measurable.of_eval fun j => by
      simp only [Matrix.mul_apply, Matrix.transpose_apply]
      exact Finset.measurable_sum _ fun k _ =>
        ((hW.eval).eval (a := k)).mul ((hW'.eval).eval (a := k))
  exact (Measure.isProbabilityMeasure_map_iff (hV.prodMk hm).aemeasurable).2 inferInstance

/-- Rotating the tokens by `O` is conjugating the query-key matrix by `O`. -/
theorem attnProb_conjMat {d n : ℕ} (β : ℝ) (O : EucSpace d ≃ₗᵢ[ℝ] EucSpace d)
    (A : Matrix (Fin d) (Fin d) ℝ) (X : Idx n → EucSpace d) (i k : Idx n) :
    attnProb β (conjMat O A) X i k = attnProb β A (fun l => O (X l)) i k := by
  have h : ∀ x y, attnWeight β ((0, conjMat O A) : HeadParam d) x y =
      attnWeight β ((0, A) : HeadParam d) (O x) (O y) := fun x y => by
    rw [attnWeight, attnWeight, qkMap, qkMap, toEuclideanLin_conjMat, ← O.inner_map_map,
      O.apply_symm_apply]
  simp only [attnProb, h]

/-- Any functional of the attention probabilities has the same expectation on
two configurations with the same Gram matrix. -/
theorem integral_attnProb_of_gram {d n : ℕ} (β : ℝ) {σV σA : ℝ≥0}
    {ρ : Measure (HeadParam d)} (hρ : IsGaussianHeadLaw d σV σA ρ) {X Y : Idx n → EucSpace d}
    (hgram : ∀ k l : Idx n, inner (𝕜 := ℝ) (X k) (X l) = inner (𝕜 := ℝ) (Y k) (Y l))
    (Φ : (Idx n → Idx n → ℝ) → ℝ) :
    ∫ θ, Φ (attnProb β θ.2 X) ∂ρ = ∫ θ, Φ (attnProb β θ.2 Y) ∂ρ := by
  obtain ⟨O, hO⟩ := exists_linearIsometryEquiv_of_inner_eq hgram
  obtain rfl : Y = fun k => O (X k) := funext fun k => (hO k).symm
  refine ((measurePreserving_conjHead hρ O).integral_comp' _).symm.trans
    (integral_congr_ae (ae_of_all _ fun θ => ?_))
  dsimp only
  rw [conjHead_apply]
  congr 1
  funext i k
  exact attnProb_conjMat β O θ.2 X i k

/-- Permuting the tokens permutes the attention probabilities. -/
theorem attnProb_comp_perm {d n : ℕ} (β : ℝ) (A : Matrix (Fin d) (Fin d) ℝ)
    (X : Idx n → EucSpace d) (σ : Equiv.Perm (Idx n)) (i k : Idx n) :
    attnProb β A (X ∘ σ) i k = attnProb β A X (σ i) (σ k) := by
  simp only [attnProb, Function.comp_apply]
  rw [Equiv.sum_comp σ fun l => attnWeight β ((0, A) : HeadParam d) (X (σ i)) (X l)]

/-- The Gram matrix of a simplex configuration is `1` on the diagonal and `γ`
off it. -/
theorem IsSimplexConfig.inner_eq {d n : ℕ} {γ : ℝ} {x : Idx n → EucSpace d}
    (hx : IsSimplexConfig γ x) (k l : Idx n) :
    inner (𝕜 := ℝ) (x k) (x l) = if k = l then 1 else γ := by
  split_ifs with h
  · rw [h, real_inner_self_eq_norm_sq, hx.1 l, one_pow]
  · exact hx.2 k l h

theorem IsSimplexConfig.comp_perm {d n : ℕ} {γ : ℝ} {x : Idx n → EucSpace d}
    (hx : IsSimplexConfig γ x) (σ : Equiv.Perm (Idx n)) : IsSimplexConfig γ (x ∘ σ) :=
  ⟨fun i => hx.1 (σ i), fun i j h => hx.2 (σ i) (σ j) (σ.injective.ne h)⟩

/-- Two distinct indices can be moved to any two distinct indices. -/
theorem exists_perm_apply_apply {n : ℕ} {i j i' j' : Idx n} (hij : i ≠ j) (hij' : i' ≠ j') :
    ∃ σ : Equiv.Perm (Idx n), σ i = i' ∧ σ j = j' := by
  set a := Equiv.swap i i' j
  have ha : i' ≠ a := by
    rw [← Equiv.swap_apply_left i i']
    exact (Equiv.swap i i').injective.ne hij
  refine ⟨(Equiv.swap i i').trans (Equiv.swap a j'), ?_, ?_⟩
  · simp only [Equiv.trans_apply, Equiv.swap_apply_left]
    exact Equiv.swap_apply_of_ne_of_ne ha hij'
  · simp only [Equiv.trans_apply, Equiv.swap_apply_left, a]

/-- The expectation of a function with values in `[0, 1]` lies in `[0, 1]`. -/
theorem integral_mem_Icc_of_mem_Icc {α : Type*} [MeasurableSpace α] {μ : Measure α}
    [IsProbabilityMeasure μ] {h : α → ℝ} (hh : ∀ a, h a ∈ Set.Icc (0 : ℝ) 1) :
    ∫ a, h a ∂μ ∈ Set.Icc (0 : ℝ) 1 := by
  refine ⟨integral_nonneg fun a => (hh a).1, (le_abs_self _).trans ?_⟩
  have := norm_integral_le_of_norm_le_const (μ := μ) (C := 1) (ae_of_all _ fun a => by
    rw [Real.norm_eq_abs, abs_of_nonneg (hh a).1]; exact (hh a).2)
  rwa [probReal_univ, mul_one, Real.norm_eq_abs] at this

/-- **Theorem (thm:clustering_random_init), first half.**  The two expectations
of `eq:f_g_def_selfcontained`,

  `f(γ) = 𝔼 Σ_k (π^A_{i→k}(γ))²`,  `g(γ) = 𝔼 Σ_k π^A_{i→k}(γ) π^A_{j→k}(γ)`,

are well defined: they depend on neither the simplex configuration realizing
the overlap `γ` nor on the choice of indices, and they take values in `[0,1]`.

The expectation is over `A ∼ ρ*`, which is read off the head `θ = (V, A)`.

**What the source says and what is changed.**  The source fixes `d, n ≥ 2`.
The proof uses neither bound, and both are dropped: for `n < 2` the statement
about `g` is empty, and nothing depends on `d`.

Source: arXiv:2604.01978v1, `eq:f_g_def_selfcontained`, with the argument of
the proof of `lem:drift_on_simplex_selfcontained`. -/
theorem simplex_overlap_wellDefined {d n : ℕ} (β : ℝ)
    (σV σA : ℝ≥0) (ρ : Measure (HeadParam d)) (hρ : IsGaussianHeadLaw d σV σA ρ) :
    ∃ f g : ℝ → ℝ,
      (∀ γ : ℝ, f γ ∈ Set.Icc (0 : ℝ) 1) ∧ (∀ γ : ℝ, g γ ∈ Set.Icc (0 : ℝ) 1) ∧
      (∀ (γ : ℝ) (x : Idx n → EucSpace d), IsSimplexConfig γ x → ∀ i : Idx n,
        (∫ θ, ∑ k : Idx n, attnProb β θ.2 x i k ^ 2 ∂ρ) = f γ) ∧
      (∀ (γ : ℝ) (x : Idx n → EucSpace d), IsSimplexConfig γ x → ∀ i j : Idx n, i ≠ j →
        (∫ θ, ∑ k : Idx n, attnProb β θ.2 x i k * attnProb β θ.2 x j k ∂ρ) = g γ) := by
  have := isProbabilityMeasure_of_isGaussianHeadLaw hρ
  set F : (Idx n → EucSpace d) → Idx n → ℝ :=
    fun x i => ∫ θ, ∑ k : Idx n, attnProb β θ.2 x i k ^ 2 ∂ρ
  set G : (Idx n → EucSpace d) → Idx n → Idx n → ℝ :=
    fun x i j => ∫ θ, ∑ k : Idx n, attnProb β θ.2 x i k * attnProb β θ.2 x j k ∂ρ
  have hgram : ∀ {γ : ℝ} {x y : Idx n → EucSpace d}, IsSimplexConfig γ x →
      IsSimplexConfig γ y → ∀ k l : Idx n,
        inner (𝕜 := ℝ) (x k) (x l) = inner (𝕜 := ℝ) (y k) (y l) := fun hx hy k l => by
    rw [hx.inner_eq, hy.inner_eq]
  -- Permuting `y` by `σ` and comparing with `x` moves the indices.
  have hF : ∀ {γ : ℝ} {x y : Idx n → EucSpace d}, IsSimplexConfig γ x →
      IsSimplexConfig γ y → ∀ (σ : Equiv.Perm (Idx n)) (i : Idx n), F x i = F y (σ i) :=
    fun {_ _ y} hx hy σ i => by
      refine (integral_attnProb_of_gram β hρ (hgram hx (hy.comp_perm σ))
        fun P => ∑ k, P i k ^ 2).trans (integral_congr_ae (ae_of_all _ fun θ => ?_))
      simp only [attnProb_comp_perm]
      exact Equiv.sum_comp σ fun k => attnProb β θ.2 y (σ i) k ^ 2
  have hG : ∀ {γ : ℝ} {x y : Idx n → EucSpace d}, IsSimplexConfig γ x →
      IsSimplexConfig γ y → ∀ (σ : Equiv.Perm (Idx n)) (i j : Idx n),
        G x i j = G y (σ i) (σ j) :=
    fun {_ _ y} hx hy σ i j => by
      refine (integral_attnProb_of_gram β hρ (hgram hx (hy.comp_perm σ))
        fun P => ∑ k, P i k * P j k).trans (integral_congr_ae (ae_of_all _ fun θ => ?_))
      simp only [attnProb_comp_perm]
      exact Equiv.sum_comp σ fun k => attnProb β θ.2 y (σ i) k * attnProb β θ.2 y (σ j) k
  classical
  refine ⟨fun γ => if h : ∃ p : (Idx n → EucSpace d) × Idx n, IsSimplexConfig γ p.1
      then F h.choose.1 h.choose.2 else 0,
    fun γ => if h : ∃ p : (Idx n → EucSpace d) × Idx n × Idx n,
        IsSimplexConfig γ p.1 ∧ p.2.1 ≠ p.2.2
      then G h.choose.1 h.choose.2.1 h.choose.2.2 else 0, ?_, ?_, ?_, ?_⟩
  · intro γ
    dsimp only
    split_ifs with h
    · refine integral_mem_Icc_of_mem_Icc fun θ => ⟨Finset.sum_nonneg fun k _ =>
        sq_nonneg _, (Finset.sum_le_sum fun k _ => ?_).trans
          (sum_attnProb_le_one β θ.2 h.choose.1 h.choose.2)⟩
      rw [sq]
      exact mul_le_of_le_one_left (attnProb_nonneg _ _ _ _ _) (attnProb_le_one _ _ _ _ _)
    · exact ⟨le_rfl, zero_le_one⟩
  · intro γ
    dsimp only
    split_ifs with h
    · refine integral_mem_Icc_of_mem_Icc fun θ => ⟨Finset.sum_nonneg fun k _ =>
        mul_nonneg (attnProb_nonneg _ _ _ _ _) (attnProb_nonneg _ _ _ _ _),
        (Finset.sum_le_sum fun k _ => ?_).trans
          (sum_attnProb_le_one β θ.2 h.choose.1 h.choose.2.1)⟩
      exact mul_le_of_le_one_right (attnProb_nonneg _ _ _ _ _) (attnProb_le_one _ _ _ _ _)
    · exact ⟨le_rfl, zero_le_one⟩
  · intro γ x hx i
    have h : ∃ p : (Idx n → EucSpace d) × Idx n, IsSimplexConfig γ p.1 := ⟨(x, i), hx⟩
    dsimp only
    rw [dite_eq_left_of_eq_true (eq_true h)]
    exact hF hx h.choose_spec (Equiv.swap i h.choose.2) i |>.trans
      (by rw [Equiv.swap_apply_left])
  · intro γ x hx i j hij
    have h : ∃ p : (Idx n → EucSpace d) × Idx n × Idx n,
        IsSimplexConfig γ p.1 ∧ p.2.1 ≠ p.2.2 := ⟨(x, i, j), hx, hij⟩
    dsimp only
    rw [dite_eq_left_of_eq_true (eq_true h)]
    obtain ⟨σ, hσi, hσj⟩ := exists_perm_apply_apply hij h.choose_spec.2
    exact (hG hx h.choose_spec.1 σ i j).trans (by rw [hσi, hσj])

/-- The hypotheses of `simplex_overlap_wellDefined` are satisfiable. -/
example : IsGaussianHeadLaw 2 0 0 (Measure.dirac (0 : HeadParam 2)) :=
  isGaussianHeadLaw_dirac_zero 2

/-- The hypotheses of `integral_mem_Icc_of_mem_Icc` are satisfiable. -/
example : ∀ a : Unit, (fun _ : Unit => (0 : ℝ)) a ∈ Set.Icc (0 : ℝ) 1 ∧
    IsProbabilityMeasure (Measure.dirac ()) :=
  fun _ => ⟨⟨le_rfl, zero_le_one⟩, inferInstance⟩

/-- The hypotheses of `integral_attnProb_of_gram` are satisfiable, at `Y = X`. -/
example (X : Idx 2 → EucSpace 2) : IsGaussianHeadLaw 2 0 0 (Measure.dirac (0 : HeadParam 2)) ∧
    ∀ k l : Idx 2, inner (𝕜 := ℝ) (X k) (X l) = inner (𝕜 := ℝ) (X k) (X l) :=
  ⟨isGaussianHeadLaw_dirac_zero 2, fun _ _ => rfl⟩

/-- The hypotheses of `exists_perm_apply_apply` are satisfiable. -/
example : (0 : Idx 2) ≠ 1 := by decide

end Homogenized
end Transformer
