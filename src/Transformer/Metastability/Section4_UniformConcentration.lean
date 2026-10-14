/-
# The original uniform independent-sample concentration theorem

ArXiv:2410.06833v1, §4, `prop: concentration unif`, `eq: upto-t`,
and arXiv:2312.10794v5, Appendix D, Step 2, almost orthogonality.
For the original independent uniform sample, the actual successive
normalized-residual frame succeeds with probability at least
`1 - n d^(-1/64)` above an explicit dimension threshold depending
only on `n`. This is stronger than the original lower bound.

Each actual successful prefix gives a genuine orthonormal frame.
The proved original projection tail bounds its next failure section;
Fubini adds the actual prefix failure probability. Induction gives
the original independent-sample union bound. On the success event,
normalization produces genuine sphere points within the exact
original distance radius `sqrt(4 log(d)/d)`. Event inclusion then
proves the original statement with every original quantifier,
hypothesis, probability constant and distance constant unchanged.
-/

import Transformer.Metastability.Section4_IndependentPrefix

open Real MeasureTheory Set

namespace Transformer.Metastability

/-- The original next-step failure probability is bounded by actual
prefix failure plus the proved original uniform projection rate.
Source: arXiv:2410.06833v1, §4, `prop: concentration unif`;
arXiv:2312.10794v5, Appendix D, Step 2. Only successful prefixes
use orthonormality; failed prefixes use the probability bound one. -/
theorem iidSphere_projectionSuccess_compl_succ_le {d m n : ℕ}
    (hd : uniformProjectionThreshold n ≤ d) (hmn : m ≤ n)
    (ν : Measure (SSphere d)) (hν : IsUniformOn d ν) :
    (iidSphere d (m + 1) ν).real (projectionSuccess d (uniformProjectionRadius d) (m + 1))ᶜ ≤
      (iidSphere d m ν).real (projectionSuccess d (uniformProjectionRadius d) m)ᶜ +
        (d : ℝ) ^ (-(1 : ℝ) / 64) := by
  let : IsProbabilityMeasure ν := hν.1
  have hdone := ((uniformProjectionThreshold_spec n).2 d hd).1
  apply iidSphere_real_event_le_prefix_compl_add d m ν _
    (measurableSet_projectionSuccess _ _ _) _ (measurableSet_projectionSuccess _ _ _).compl _
    (Real.rpow_nonneg (by positivity) _)
  intro X hX
  have hsub : {x : SSphere d | Fin.snoc X x ∈ (projectionSuccess d (uniformProjectionRadius d) (m + 1))ᶜ} ⊆
      {x : SSphere d | uniformProjectionRadius d ≤
        ‖frameProjection (sequentialFrame d m X) (x : EucSpace d)‖} := by
    intro x hx
    change ¬ Fin.snoc X x ∈ projectionSuccess d (uniformProjectionRadius d) (m + 1) at hx
    change uniformProjectionRadius d ≤ ‖frameProjection (sequentialFrame d m X) (x : EucSpace d)‖
    apply le_of_not_gt
    intro hp
    exact hx ((mem_projectionSuccess_snoc_iff d m _ X x).mpr ⟨hX, hp⟩)
  apply (measureReal_mono hsub (measure_ne_top ν _)).trans
  exact uniform_frameProjection_tail_original_scale hd hmn ν hν _
    (orthonormal_sequentialFrame_of_success d m _ (uniformProjectionRadius_lt_one hdone) X hX)

/-- The original independent sample has total actual projection-failure
probability at most its length times the original dimension rate.
Source: arXiv:2410.06833v1, §4, the union bound in
`prop: concentration unif`; arXiv:2312.10794v5, Appendix D, Step 2. -/
theorem iidSphere_projectionSuccess_compl_le {d m n : ℕ}
    (hd : uniformProjectionThreshold n ≤ d) (hmn : m ≤ n)
    (ν : Measure (SSphere d)) (hν : IsUniformOn d ν) :
    (iidSphere d m ν).real (projectionSuccess d (uniformProjectionRadius d) m)ᶜ ≤
      (m : ℝ) * (d : ℝ) ^ (-(1 : ℝ) / 64) := by
  revert hmn
  induction m with
  | zero => intro hmn; simp [projectionSuccess]
  | succ m ih =>
    intro hmn
    have hprev := ih (by omega : m ≤ n)
    calc
      _ ≤ (iidSphere d m ν).real (projectionSuccess d (uniformProjectionRadius d) m)ᶜ +
          (d : ℝ) ^ (-(1 : ℝ) / 64) :=
        iidSphere_projectionSuccess_compl_succ_le hd (by omega) ν hν
      _ ≤ (m : ℝ) * (d : ℝ) ^ (-(1 : ℝ) / 64) + (d : ℝ) ^ (-(1 : ℝ) / 64) :=
        add_le_add hprev le_rfl
      _ = _ := by rw [Nat.cast_add, Nat.cast_one]; ring

/-- The actual independent sample succeeds at every original projection
with probability at least one minus its length times the original rate.
Source: arXiv:2410.06833v1, §4, `prop: concentration unif`;
arXiv:2312.10794v5, Appendix D, Step 2, the actual union bound for the
original inductive approximation. This is the measurable success event,
before transferring the result to the original existential event. -/
theorem iidSphere_projectionSuccess_ge_union_bound {d n : ℕ}
    (hd : uniformProjectionThreshold n ≤ d) (ν : Measure (SSphere d)) (hν : IsUniformOn d ν) :
    1 - (n : ℝ) * (d : ℝ) ^ (-(1 : ℝ) / 64) ≤
      (iidSphere d n ν).real (projectionSuccess d (uniformProjectionRadius d) n) := by
  let : IsProbabilityMeasure ν := hν.1
  let : IsProbabilityMeasure (iidSphere d n ν) := isProbabilityMeasure_iidSphere d n ν
  have hfail := iidSphere_projectionSuccess_compl_le hd le_rfl ν hν
  rw [probReal_compl_eq_one_sub (measurableSet_projectionSuccess _ _ _)] at hfail
  linarith

/-- The original independent uniform sample is genuinely near-orthogonal
with probability at least one minus its length times the original rate.
Source: arXiv:2410.06833v1, §4, `prop: concentration unif`, `eq: upto-t`.
The proved normalized residual supplies the original distance factor. -/
theorem iidSphere_nearOrthogonal_ge_union_bound {d n : ℕ}
    (hd : uniformProjectionThreshold n ≤ d) (ν : Measure (SSphere d)) (hν : IsUniformOn d ν) :
    1 - (n : ℝ) * (d : ℝ) ^ (-(1 : ℝ) / 64) ≤ (iidSphere d n ν).real (nearOrthogonal d n) := by
  let : IsProbabilityMeasure ν := hν.1
  exact (iidSphere_projectionSuccess_ge_union_bound hd ν hν).trans
    (measure_nearOrthogonal_ge_projectionSuccess hd ν)

/-- **Proposition (prop: concentration unif).**

For `n ≥ 2` there is `d⋆(n) > n` such that for every `d ≥ d⋆(n)`, an
i.i.d. uniform sample on `𝕊^{d-1}` lies in `nearOrthogonal` with
probability at least `1 - 2 n² d^{-1/64}`.

The original actual normalized-residual construction is measurable;
the proved uniform coordinate moments give the original projection
estimate, and Fubini proves the independent-sample induction. Its
stronger intermediate probability bound implies the stated one.
Every original hypothesis, quantifier and constant is retained.

Source: arXiv:2410.06833v1, §4, `prop: concentration unif`, `eq: upto-t`;
arXiv:2312.10794v5, Appendix D, Step 2. -/
theorem concentration_unif (n : ℕ) (hn : 2 ≤ n) :
    ∀ σ : ∀ d : ℕ, Measure (SSphere d), IsUniformFamily σ →
      ∃ d_star : ℕ, n < d_star ∧ ∀ d : ℕ, d_star ≤ d →
        1 - 2 * (n : ℝ)^2 * (d : ℝ) ^ (-(1 : ℝ) / 64)
          ≤ (iidSphere d n (σ d)).real (nearOrthogonal d n) := by
  intro σ hσ
  refine ⟨uniformProjectionThreshold n, (uniformProjectionThreshold_spec n).1, ?_⟩
  intro d hd
  have hdone := ((uniformProjectionThreshold_spec n).2 d hd).1
  have hbound := iidSphere_nearOrthogonal_ge_union_bound hd (σ d) (hσ d (by omega))
  have hnreal : (2 : ℝ) ≤ n := by exact_mod_cast hn
  have hcoeff : (n : ℝ) ≤ 2 * (n : ℝ) ^ 2 := by nlinarith
  have hscaled := mul_le_mul_of_nonneg_right hcoeff
    (Real.rpow_nonneg (by positivity : (0 : ℝ) ≤ d) (-(1 : ℝ) / 64))
  linarith

/-- The original sample-size and original uniform-family hypotheses
are jointly satisfiable, using the actual uniform laws in every
nonempty dimension and the original allowed sample size two. -/
example : (2 : ℕ) ≤ 2 ∧ IsUniformFamily (fun d => MeanField.uniformLaw d) := by
  refine ⟨le_rfl, fun d hd => ?_⟩
  exact isUniformOn_uniformLaw hd

/-- The actual dimension threshold, allowed original prefix length,
and original uniform law jointly satisfy every probability-step
hypothesis, with a concrete nonempty original sample space. -/
example : ∃ d : ℕ, uniformProjectionThreshold 2 ≤ d ∧ (1 : ℕ) ≤ 2 ∧
    IsUniformOn d (MeanField.uniformLaw d) ∧ Nonempty (Fin 2 → SSphere d) := by
  let d := uniformProjectionThreshold 2
  have hdone := ((uniformProjectionThreshold_spec 2).2 d le_rfl).1
  let x : SSphere d := ⟨EuclideanSpace.single ⟨0, by omega⟩ 1,
    mem_sphere_zero_iff_norm.mpr (by simp)⟩
  exact ⟨d, le_rfl, by decide, isUniformOn_uniformLaw (by omega), ⟨fun _ => x⟩⟩

end Transformer.Metastability
