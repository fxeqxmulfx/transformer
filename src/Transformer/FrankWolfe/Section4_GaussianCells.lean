/-
# Gaussian vertices: all vertices own their cells with high probability

The qualitative probability limit of `prop: d.to.infty` in §4 of
arXiv:2508.09628v1. First choose a fixed radius making the conditional error
small. Then let the dimension grow so that the mass of that ball is small.
A union bound over the fixed number of ordered vertex pairs gives the
simultaneous conclusion.

The auxiliary result uses a uniform operator-norm bound. The source's
symmetric quadratic-form bounds supply it in `Section4_VertexGenericity`.
No symmetry assumption is needed once that operator-norm bound is given.
-/

import Transformer.FrankWolfe.Section4_GaussianBalls
import Mathlib.Topology.Algebra.InfiniteSum.ENNReal

open Real MeasureTheory ProbabilityTheory Filter
open scoped BigOperators ENNReal

namespace Transformer.FrankWolfe

/-- One independent pair fails the cell comparison with probability tending
to zero; arXiv:2508.09628v1, §4, `prop: d.to.infty`. -/
theorem tendsto_gaussian_pair_failure (B : (d : ℕ) → ParamMatrix d)
    {l M : ℝ} (hl : 0 < l) (hB : ∀ d, ‖B d‖ ≤ M)
    (hlower : ∀ d (x : EucSpace d), l * ‖x‖ ^ 2 ≤ inner (𝕜 := ℝ) (B d x) x) :
    Tendsto (fun d : ℕ =>
      ((stdGaussian (EucSpace d)).prod (stdGaussian (EucSpace d)))
        {p | inner (𝕜 := ℝ) (B d p.1) p.1 ≤ inner (𝕜 := ℝ) (B d p.1) p.2})
      atTop (nhds 0) := by
  apply tendsto_order.mpr
  constructor
  · intro ε hε
    exact (not_lt_of_ge (bot_le : (0 : ℝ≥0∞) ≤ ε) hε).elim
  · intro ε hε
    have hhalf : 0 < ε / 2 := ENNReal.half_pos hε.ne'
    obtain ⟨R, hR, herror⟩ := exists_gaussian_cell_radius hl M hhalf
    filter_upwards [eventually_stdGaussian_ball_mass_lt R hhalf] with d hd
    refine (gaussian_cell_pair_le (B d) hl hR (hB d) (hlower d)).trans_lt ?_
    simpa using ENNReal.add_lt_add hd herror

/-- The pair-limit hypotheses hold for the identity, `l = M = 1`;
arXiv:2508.09628v1, §4, `prop: d.to.infty`. -/
example : (0 : ℝ) < 1 ∧
    (∀ d : ℕ, ‖ContinuousLinearMap.id ℝ (EucSpace d)‖ ≤ 1) ∧
    (∀ d : ℕ, ∀ x : EucSpace d, (1 : ℝ) * ‖x‖ ^ 2 ≤ inner (𝕜 := ℝ) x x) := by
  refine ⟨one_pos, fun _ => ContinuousLinearMap.norm_id_le, ?_⟩
  intro d x
  rw [real_inner_self_eq_norm_sq, one_mul]

/-- A finite union bound for all ordered Gaussian vertex comparisons;
arXiv:2508.09628v1, §4, `prop: d.to.infty`. The number of vertices stays
fixed when the dimension subsequently tends to infinity. -/
theorem gaussian_tuple_failure_le (d κ : ℕ) (B : ParamMatrix d) :
    (Measure.pi fun _ : Fin κ => stdGaussian (EucSpace d))
        {v | ∃ i j : Fin κ, j ≠ i ∧ inner (𝕜 := ℝ) (B (v i)) (v i) ≤
          inner (𝕜 := ℝ) (B (v i)) (v j)} ≤
      (κ : ℝ≥0∞) ^ 2 *
        ((stdGaussian (EucSpace d)).prod (stdGaussian (EucSpace d)))
          {p | inner (𝕜 := ℝ) (B p.1) p.1 ≤ inner (𝕜 := ℝ) (B p.1) p.2} := by
  classical
  let μ := Measure.pi fun _ : Fin κ => stdGaussian (EucSpace d)
  let E : Fin κ × Fin κ → Set (Fin κ → EucSpace d) := fun ij =>
    {v | ij.2 ≠ ij.1 ∧ inner (𝕜 := ℝ) (B (v ij.1)) (v ij.1) ≤
      inner (𝕜 := ℝ) (B (v ij.1)) (v ij.2)}
  let p := ((stdGaussian (EucSpace d)).prod (stdGaussian (EucSpace d)))
    {z | inner (𝕜 := ℝ) (B z.1) z.1 ≤ inner (𝕜 := ℝ) (B z.1) z.2}
  have hset : {v | ∃ i j : Fin κ, j ≠ i ∧ inner (𝕜 := ℝ) (B (v i)) (v i) ≤
      inner (𝕜 := ℝ) (B (v i)) (v j)} = ⋃ ij, E ij := by
    ext v
    simp only [E, Set.mem_ofPred_eq, Set.mem_iUnion, Prod.exists]
  have hE : ∀ ij, μ (E ij) ≤ p := by
    intro ij
    by_cases hij : ij.1 = ij.2
    · have hempty : E ij = ∅ := by simp [E, hij]
      rw [hempty, measure_empty]
      exact bot_le
    · have hs : MeasurableSet {z : EucSpace d × EucSpace d |
        inner (𝕜 := ℝ) (B z.1) z.1 ≤ inner (𝕜 := ℝ) (B z.1) z.2} :=
        measurableSet_le (by fun_prop) (by fun_prop)
      calc μ (E ij) ≤ μ ((fun v => (v ij.1, v ij.2)) ⁻¹'
          {z | inner (𝕜 := ℝ) (B z.1) z.1 ≤ inner (𝕜 := ℝ) (B z.1) z.2}) :=
          measure_mono fun _ hv => hv.2
        _ = p := by
          rw [← Measure.map_apply ((measurable_pi_apply ij.1).prodMk
            (measurable_pi_apply ij.2)) hs, map_pair_stdGaussian hij]
  calc
    μ _ ≤ ∑' ij, μ (E ij) := by rw [hset]; exact measure_iUnion_le E
    _ ≤ ∑' _ij : Fin κ × Fin κ, p := ENNReal.tsum_le_tsum hE
    _ = (κ : ℝ≥0∞) ^ 2 * p := by simp [pow_two]

/-- All Gaussian vertices own their cells with probability tending to one
under a positive quadratic lower bound and a uniform operator-norm bound;
arXiv:2508.09628v1, §4, `prop: d.to.infty`.
The source's positive definite matrices supply these bounds. -/
theorem gaussian_vertices_own_cell_of_norm_bound (κ : ℕ)
    (B : (d : ℕ) → ParamMatrix d) {l M : ℝ}
    (hl : 0 < l) (hB : ∀ d, ‖B d‖ ≤ M)
    (hlower : ∀ d (x : EucSpace d), l * ‖x‖ ^ 2 ≤ inner (𝕜 := ℝ) (B d x) x) :
    Tendsto (fun d : ℕ => (Measure.pi fun _ : Fin κ => stdGaussian (EucSpace d))
      {v | ∀ i j : Fin κ, j ≠ i → inner (𝕜 := ℝ) (B d (v i)) (v j) <
        inner (𝕜 := ℝ) (B d (v i)) (v i)}) atTop (nhds 1) := by
  let bad := fun d : ℕ => {v : Fin κ → EucSpace d |
    ∃ i j : Fin κ, j ≠ i ∧ inner (𝕜 := ℝ) (B d (v i)) (v i) ≤
      inner (𝕜 := ℝ) (B d (v i)) (v j)}
  have hbad : ∀ d, MeasurableSet (bad d) := by
    intro d
    simp only [bad, Set.ofPred_exists]
    apply MeasurableSet.iUnion
    intro i
    apply MeasurableSet.iUnion
    intro j
    by_cases hij : j = i
    · simp [hij]
    · simpa [hij] using
        (measurableSet_le (by fun_prop) (by fun_prop) :
          MeasurableSet {v : Fin κ → EucSpace d | inner (𝕜 := ℝ) (B d (v i)) (v i) ≤
            inner (𝕜 := ℝ) (B d (v i)) (v j)})
  have hlim : Tendsto (fun d => (Measure.pi fun _ : Fin κ =>
      stdGaussian (EucSpace d)) (bad d)) atTop (nhds 0) := by
    apply tendsto_of_tendsto_of_tendsto_of_le_of_le tendsto_const_nhds
      (h := fun d => (κ : ℝ≥0∞) ^ 2 *
        ((stdGaussian (EucSpace d)).prod (stdGaussian (EucSpace d)))
          {p | inner (𝕜 := ℝ) (B d p.1) p.1 ≤ inner (𝕜 := ℝ) (B d p.1) p.2})
    · simpa using ENNReal.Tendsto.const_mul (tendsto_gaussian_pair_failure B hl hB hlower)
        (a := (κ : ℝ≥0∞) ^ 2) (Or.inr (by finiteness))
    · exact fun _ => bot_le
    · exact fun d => gaussian_tuple_failure_le d κ (B d)
  have hcompl : ∀ d, (bad d)ᶜ = {v : Fin κ → EucSpace d | ∀ i j, j ≠ i →
      inner (𝕜 := ℝ) (B d (v i)) (v j) < inner (𝕜 := ℝ) (B d (v i)) (v i)} := by
    intro d
    ext v
    simp [bad]
  have hsub := ENNReal.Tendsto.sub (tendsto_const_nhds (x := (1 : ℝ≥0∞))) hlim
    (Or.inl (by norm_num))
  simpa only [← hcompl, prob_compl_eq_one_sub (hbad _), tsub_zero] using hsub

/-- The simultaneous-limit hypotheses are also met by the identity in
every dimension; arXiv:2508.09628v1, §4, `prop: d.to.infty`. -/
example : (0 : ℝ) < 1 ∧
    (∀ d : ℕ, ‖ContinuousLinearMap.id ℝ (EucSpace d)‖ ≤ 1) ∧
    (∀ d : ℕ, ∀ x : EucSpace d, (1 : ℝ) * ‖x‖ ^ 2 ≤ inner (𝕜 := ℝ) x x) := by
  refine ⟨one_pos, fun _ => ContinuousLinearMap.norm_id_le, ?_⟩
  intro d x
  rw [real_inner_self_eq_norm_sq, one_mul]

/-- In particular, standard Gaussian vertices strictly maximize their own
Euclidean inner product against the other vertices with probability tending
to one; arXiv:2508.09628v1, §4, `prop: d.to.infty` at `B = I`.
This conclusion includes every fixed finite number of vertices. -/
theorem gaussian_vertices_own_cell_identity (κ : ℕ) :
    Tendsto (fun d : ℕ => (Measure.pi fun _ : Fin κ => stdGaussian (EucSpace d))
      {v | ∀ i j : Fin κ, j ≠ i → inner (𝕜 := ℝ) (v i) (v j) <
        inner (𝕜 := ℝ) (v i) (v i)}) atTop (nhds 1) := by
  apply gaussian_vertices_own_cell_of_norm_bound κ
    (fun d => ContinuousLinearMap.id ℝ (EucSpace d)) one_pos
    (fun _ => ContinuousLinearMap.norm_id_le)
  intro d x
  rw [ContinuousLinearMap.id_apply, real_inner_self_eq_norm_sq, one_mul]

end Transformer.FrankWolfe
