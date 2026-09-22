/-
# Attention's forward pass and Frank-Wolfe — softmax attention collapses

Formalization of Alcalde, Geshkovski, Ruiz-Balet — arXiv:2508.09628v1,
*Attention's forward pass and Frank-Wolfe*, §5, `prop: origin`.

A trajectory of `eq: softmax.ODE` never leaves the convex hull of its initial
configuration (`softmaxFlow_mem_configHull`), hence stays in a ball, where
every score is at least one `c > 0`.  Each step then shrinks the diameter by
`1 - γc` and moves a particle by at most the diameter, so every particle is a
Cauchy sequence and all of them share the limit (`exists_tendsto_softmaxFlow`).
The hardmax dynamics, whose particles stop at the vertices of the hull, is
therefore an approximation of `(SA_β)` on finite time horizons only.
-/

import Transformer.FrankWolfe.Section5_Contraction

open scoped BigOperators
open Real Filter Topology

namespace Transformer
namespace FrankWolfe

variable {d n : ℕ}

/-- For `0 ≤ γ ≤ 1` a trajectory of `eq: softmax.ODE` stays in the convex hull of
its initial configuration. -/
theorem softmaxFlow_mem_configHull (β : ℝ) {γ : ℝ} (hγ0 : 0 ≤ γ) (hγ1 : γ ≤ 1)
    {x : ℕ → Idx n → EucSpace d} (hx : ∀ t i, x (t + 1) i = softmaxStep β γ (x t) i) :
    ∀ t i, x t i ∈ configHull (x 0) := by
  intro t
  induction t with
  | zero => exact fun i => subset_convexHull ℝ _ (Set.mem_range_self i)
  | succ t ih =>
    intro i
    rw [hx t i]
    exact convexHull_min (Set.range_subset_iff.mpr ih) (convex_convexHull ℝ _)
      (softmaxStep_mem_configHull β hγ0 hγ1 (x t) i)

/-- The hypotheses of `softmaxFlow_mem_configHull` are satisfiable: one particle
resting at the origin, `γ = 1`. -/
example : (0 : ℝ) ≤ 1 ∧ (1 : ℝ) ≤ 1 ∧
    ∀ (t : ℕ) (i : Idx 1), (fun (_ : ℕ) (_ : Idx 1) => (0 : EucSpace 1)) (t + 1) i =
      softmaxStep 1 1 ((fun (_ : ℕ) (_ : Idx 1) => (0 : EucSpace 1)) t) i :=
  ⟨zero_le_one, le_rfl, fun _ _ => by simp [softmaxStep]⟩

/-- **The particles of `eq: softmax.ODE` converge to one point of the hull**, for
every `β`, every `γ ∈ (0, 1]` and at least one particle. -/
theorem exists_tendsto_softmaxFlow (β : ℝ) {γ : ℝ} (hγ0 : 0 < γ) (hγ1 : γ ≤ 1) (hn : 0 < n)
    {x : ℕ → Idx n → EucSpace d} (hx : ∀ t i, x (t + 1) i = softmaxStep β γ (x t) i) :
    ∃ xstar ∈ configHull (x 0), ∀ i, Tendsto (fun t => x t i) atTop (𝓝 xstar) := by
  have hhull := softmaxFlow_mem_configHull β hγ0.le hγ1 hx
  obtain ⟨R, hR⟩ : ∃ R, ∀ j, ‖x 0 j‖ ≤ R :=
    ⟨∑ j, ‖x 0 j‖, fun j => Finset.single_le_sum (f := fun j => ‖x 0 j‖)
      (fun k _ => norm_nonneg _) (Finset.mem_univ j)⟩
  have hball : ∀ t j, ‖x t j‖ ≤ R := fun t j =>
    mem_closedBall_zero_iff.mp (convexHull_min (Set.range_subset_iff.mpr fun k =>
      mem_closedBall_zero_iff.mpr (hR k)) (convex_closedBall 0 R) (hhull t j))
  obtain ⟨c, hc0, hc⟩ : ∃ c, 0 < c ∧ ∀ t i j, c ≤ attWeight β (x t) i j :=
    ⟨_, div_pos (exp_pos _) (mul_pos (Nat.cast_pos.mpr hn) (exp_pos _)),
      fun t i j => attWeight_ge β (x t) (hball t) i j⟩
  have i₀ : Idx n := ⟨0, hn⟩
  have hc1 : c ≤ 1 := (hc 0 i₀ i₀).trans (attWeight_le_one β (x 0) i₀ i₀)
  have hq0 : 0 ≤ 1 - γ * c := by nlinarith [mul_le_mul hγ1 hc1 hc0.le zero_le_one]
  have hq1 : 1 - γ * c < 1 := by linarith [mul_pos hγ0 hc0]
  have hdiam : ∀ t j j', ‖x t j - x t j'‖ ≤ (1 - γ * c) ^ t * (2 * R) := by
    intro t
    induction t with
    | zero =>
      intro j j'
      rw [pow_zero, one_mul]
      linarith [norm_sub_le (x 0 j) (x 0 j'), hR j, hR j']
    | succ t ih =>
      intro j j'
      rw [hx t j, hx t j', pow_succ', mul_assoc]
      exact norm_sub_softmaxStep_le β hγ0.le hγ1 hc0.le (x t) (hc t) ih j j'
  have hinc : ∀ i t, dist (x t i) (x (t + 1) i) ≤ 2 * R * (1 - γ * c) ^ t := fun i t => by
    rw [dist_comm, dist_eq_norm, hx t i, mul_comm]
    exact norm_softmaxStep_sub_le β hγ0.le hγ1 (x t) (hdiam t) i
  have hgeom : Tendsto (fun t : ℕ => (1 - γ * c) ^ t * (2 * R)) atTop (𝓝 0) := by
    simpa only [zero_mul] using (tendsto_pow_atTop_nhds_zero_of_lt_one hq0 hq1).mul_const (2 * R)
  obtain ⟨xstar, hlim⟩ := cauchySeq_tendsto_of_complete
    (cauchySeq_of_le_geometric _ _ hq1 (hinc i₀))
  refine ⟨xstar, ((Set.finite_range (x 0)).isCompact_convexHull (𝕜 := ℝ)).isClosed.mem_of_tendsto
    hlim (Eventually.of_forall fun t => hhull t i₀), fun i => hlim.congr_dist ?_⟩
  exact squeeze_zero (fun _ => dist_nonneg) (fun t => (dist_eq_norm _ _).trans_le (hdiam t i₀ i))
    hgeom

/-- The hypotheses of `exists_tendsto_softmaxFlow` are satisfiable: one particle
resting at the origin, `γ = 1`. -/
example : (0 : ℝ) < 1 ∧ (1 : ℝ) ≤ 1 ∧ 0 < 1 ∧
    ∀ (t : ℕ) (i : Idx 1), (fun (_ : ℕ) (_ : Idx 1) => (0 : EucSpace 1)) (t + 1) i =
      softmaxStep 1 1 ((fun (_ : ℕ) (_ : Idx 1) => (0 : EucSpace 1)) t) i :=
  ⟨one_pos, le_rfl, one_pos, fun _ _ => by simp [softmaxStep]⟩

/-- **Proposition (prop: origin).**
Suppose `β > 0`.  There is `γ_* ∈ (0,1)` such that for every `γ ∈ (0, γ_*)` and
every initial configuration `(x_i^0)_{i ∈ ⟦1,n⟧}` there is `x^* ∈ conv{x_i^0}`
with `x_i^t → x^*` for every particle `i` following `eq: softmax.ODE`, that is
`x_i^{t+1} = softmaxStep β γ (x^t) i`, the model `(SA_β)` with `V^t = γ I_d` and
`B^t ≡ I_d`.

Differences from the source, recorded here so that they can be audited:
- `0 < n` is added.  With no particles the hull is empty and there is no `x^*`
  (`not_softmax_collapse_zero`); the source's `⟦1,n⟧` presupposes a particle.
- `β > 0` is dropped and `γ_* = 1/2` is one admissible choice: the proof
  (`exists_tendsto_softmaxFlow`) works for every real `β` and every
  `γ ∈ (0, 1]`, so the statement proved is stronger than the source's.
- The source defers the proof, "mutatis mutandis", to Geshkovski, Rigollet,
  Ruiz-Balet, arXiv:2411.04551, Proposition 2.1, a continuous-time result on
  the sphere.  The proof here is the discrete contraction instead: on the hull
  every score is at least one `c > 0`, so a step shrinks the diameter by
  `1 - γc`.

Source: arXiv:2508.09628v1, §5, `prop: origin`. -/
theorem softmax_collapse (β : ℝ) (hn : 0 < n) :
    ∃ γstar ∈ Set.Ioo (0 : ℝ) 1, ∀ γ ∈ Set.Ioo (0 : ℝ) γstar,
      ∀ (X₀ : Idx n → EucSpace d) (x : ℕ → Idx n → EucSpace d),
        x 0 = X₀ → (∀ (t : ℕ) (i : Idx n), x (t + 1) i = softmaxStep β γ (x t) i) →
        ∃ xstar ∈ configHull X₀, ∀ i : Idx n,
          Filter.Tendsto (fun t => x t i) Filter.atTop (nhds xstar) := by
  refine ⟨1 / 2, ⟨by norm_num, by norm_num⟩, fun γ hγ X₀ x hx0 hx => ?_⟩
  subst hx0
  exact exists_tendsto_softmaxFlow β hγ.1 (by linarith [hγ.2]) hn hx

/-- The hypothesis of `softmax_collapse` is satisfiable: one particle. -/
example : 0 < 1 := one_pos

/-- **`prop: origin` fails without particles**: the hull of the empty
configuration is empty, so no `x^*` exists, whatever `β` and `γ_*`.  This is
why `softmax_collapse` assumes `0 < n`.

Source: arXiv:2508.09628v1, §5, `prop: origin`, at `n = 0`. -/
theorem not_softmax_collapse_zero (β : ℝ) :
    ¬ ∃ γstar ∈ Set.Ioo (0 : ℝ) 1, ∀ γ ∈ Set.Ioo (0 : ℝ) γstar,
      ∀ (X₀ : Idx 0 → EucSpace d) (x : ℕ → Idx 0 → EucSpace d),
        x 0 = X₀ → (∀ (t : ℕ) (i : Idx 0), x (t + 1) i = softmaxStep β γ (x t) i) →
        ∃ xstar ∈ configHull X₀, ∀ i : Idx 0,
          Filter.Tendsto (fun t => x t i) Filter.atTop (nhds xstar) := by
  rintro ⟨γstar, hγstar, h⟩
  obtain ⟨xstar, hmem, -⟩ := h (γstar / 2) ⟨by linarith [hγstar.1], by linarith [hγstar.1]⟩
    (fun i => i.elim0) (fun _ i => i.elim0) rfl (fun _ i => i.elim0)
  simp [configHull] at hmem

end FrankWolfe
end Transformer
