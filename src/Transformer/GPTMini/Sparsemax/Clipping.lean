import Transformer.GPTMini.Sparsemax.SelfRoute

/-!
# Counterexample to unconditional self-route erasure with epsilon clipping

The plan's H4 describes self-only sparse attention as erased by XSA.
arXiv:2603.09078v1, §2, equation `xsa` uses division by the squared norm;
Algorithm 1 and the lab at `73f8a0b` normalize with an epsilon floor.
Below that floor a nonzero residual remains, even at a strict unit score
gap. This corrects the implementation-level claim, not the paper's exact
orthogonal-projection formula for nonzero vectors.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.ConvexRecall Transformer.GPTMini.Convex
open scoped BigOperators

/-- Below the epsilon floor a pure self value is only partially erased.
Source: arXiv:2603.09078v1, §2, Algorithm 1, and lab `Attention.forward`
at `73f8a0b`; the coefficient follows the implemented clipped normalizer. -/
theorem xsaProjection_self_below_epsilon (cfg : Config) {T : ℕ} (eps : ℝ)
    (values : Fin T → EucSpace cfg.head_dim) (i : Fin T)
    (heps : 0 < eps) (hv : ‖values i‖ ≤ eps) :
    xsaProjection cfg eps values values i =
      (1 - ‖values i‖ ^ 2 / eps ^ 2) • values i := by
  dsimp only [xsaProjection, normL2]
  rw [max_eq_right hv, real_inner_smul_right, real_inner_self_eq_norm_sq, smul_smul]
  have hscalar : (1 / eps * ‖values i‖ ^ 2) * (1 / eps) = ‖values i‖ ^ 2 / eps ^ 2 := by
    field_simp
  rw [hscalar, sub_smul, one_smul]

/-- A nonzero half-unit value is below an epsilon of one.
Source context: Algorithm 1's clipped value normalization. -/
example (cfg : Config) : (0 : ℝ) < 1 ∧
    ‖((1 / 2 : ℝ) • EuclideanSpace.single ⟨0, cfg.head_dim_pos⟩ 1 :
      EucSpace cfg.head_dim)‖ ≤ 1 := by
  norm_num [norm_smul, PiLp.norm_single]

/-- A nonzero self value strictly below epsilon gives a nonzero XSA
output. Source: Algorithm 1 of arXiv:2603.09078v1, §2, with the lab's
epsilon floor; this refutes erasure without a norm hypothesis. -/
theorem xsaProjection_self_nonzero_below_epsilon (cfg : Config) {T : ℕ}
    (eps : ℝ) (values : Fin T → EucSpace cfg.head_dim) (i : Fin T)
    (heps : 0 < eps) (hn : 0 < ‖values i‖) (hv : ‖values i‖ < eps) :
    xsaProjection cfg eps values values i ≠ 0 := by
  rw [xsaProjection_self_below_epsilon cfg eps values i heps hv.le]
  have hs : ‖values i‖ ^ 2 < eps ^ 2 := by nlinarith
  have hc : 0 < 1 - ‖values i‖ ^ 2 / eps ^ 2 :=
    sub_pos.mpr ((div_lt_one (pow_pos heps 2)).mpr hs)
  exact smul_ne_zero (ne_of_gt hc) (norm_ne_zero_iff.mp (ne_of_gt hn))

/-- Strictly positive clipped values satisfy the non-erasure premises.
Source context: Algorithm 1 and the lab's value-normalization epsilon. -/
example (cfg : Config) : (0 : ℝ) < 1 ∧
    0 < ‖((1 / 2 : ℝ) • EuclideanSpace.single ⟨0, cfg.head_dim_pos⟩ 1 :
      EucSpace cfg.head_dim)‖ ∧
    ‖((1 / 2 : ℝ) • EuclideanSpace.single ⟨0, cfg.head_dim_pos⟩ 1 :
      EucSpace cfg.head_dim)‖ < 1 := by
  norm_num [norm_smul, PiLp.norm_single]

/-- The size of the clipped residual is exactly the retained coefficient
times the self-value norm. Source: arXiv:2603.09078v1, §2, Algorithm 1,
with the implemented epsilon floor at `73f8a0b`. -/
theorem xsaProjection_clipped_self_norm (cfg : Config) {T : ℕ} (eps : ℝ)
    (values : Fin T → EucSpace cfg.head_dim) (i : Fin T)
    (heps : 0 < eps) (hv : ‖values i‖ ≤ eps) :
    ‖xsaProjection cfg eps values values i‖ =
      (1 - ‖values i‖ ^ 2 / eps ^ 2) * ‖values i‖ := by
  rw [xsaProjection_self_below_epsilon cfg eps values i heps hv, norm_smul, Real.norm_eq_abs]
  have hs : ‖values i‖ ^ 2 ≤ eps ^ 2 := by nlinarith [norm_nonneg (values i)]
  have hc : 0 ≤ 1 - ‖values i‖ ^ 2 / eps ^ 2 :=
    sub_nonneg.mpr ((div_le_one (pow_pos heps 2)).mpr hs)
  rw [abs_of_nonneg hc]

/-- The residual-norm formula has a nonzero clipped input.
Source context: Algorithm 1's value-normalization epsilon. -/
example (cfg : Config) : (0 : ℝ) < 1 ∧
    ‖((1 / 2 : ℝ) • EuclideanSpace.single ⟨0, cfg.head_dim_pos⟩ 1 :
      EucSpace cfg.head_dim)‖ ≤ 1 := by
  norm_num [norm_smul, PiLp.norm_single]

/-- The actual sparsemax head has this residual on a clipped self route.
Source: sparsemax Proposition 1, arXiv:1602.02068v2, §2.2, and XSA
Algorithm 1, arXiv:2603.09078v1, §2. The score gap still selects self exactly. -/
theorem sparseXSAOutput_clipped_self (cfg : Config) {T : ℕ} (eps : ℝ)
    (scores : Fin T → ℝ) (values : Fin T → EucSpace cfg.head_dim) (i : Fin T)
    (heps : 0 < eps) (hv : ‖values i‖ ≤ eps)
    (hgap : ∀ j, j ≤ i → j ≠ i → scores j + 1 ≤ scores i) :
    sparseXSAOutput cfg eps scores values i =
      (1 - ‖values i‖ ^ 2 / eps ^ 2) • values i := by
  have hw := sparseWeights_eq_basis_of_gap scores i i le_rfl hgap
  have hout : (∑ j, sparseWeights scores i j • values j) = values i := by
    rw [hw]
    simp [basis]
  dsimp only [sparseXSAOutput, xsaProjection]
  rw [hout]
  exact xsaProjection_self_below_epsilon cfg eps values i heps hv

/-- The clipped-head premises include a competing, visible position.
Source context: sparsemax Proposition 1 and XSA Algorithm 1. -/
example (cfg : Config) : (0 : ℝ) < 1 ∧
    ‖((1 / 2 : ℝ) • EuclideanSpace.single ⟨0, cfg.head_dim_pos⟩ 1 :
      EucSpace cfg.head_dim)‖ ≤ 1 ∧
    ∀ j : Fin 2, j ≤ 1 → j ≠ 1 →
      (if j = 1 then (3 : ℝ) else 0) + 1 ≤ 3 := by
  refine ⟨by norm_num, by norm_num [norm_smul, PiLp.norm_single], ?_⟩
  intro j _ hj
  simp [hj]

/-- Concrete counterexample to H4 without the epsilon-norm condition:
a strict self winner, with nonzero value of norm one half and epsilon one,
has output of norm three eighths. Source: the implemented Algorithm 1
variant at `73f8a0b`; this is a real-arithmetic implementation counterexample. -/
theorem clipped_self_route_counterexample (cfg : Config) :
    ∃ scores : Fin 2 → ℝ, ∃ values : Fin 2 → EucSpace cfg.head_dim,
      (∀ j, j ≤ (1 : Fin 2) → j ≠ 1 → scores j + 1 < scores 1) ∧
      ‖values 1‖ = 1 / 2 ∧
      ‖sparseXSAOutput cfg 1 scores values 1‖ = 3 / 8 := by
  let unit : EucSpace cfg.head_dim := EuclideanSpace.single ⟨0, cfg.head_dim_pos⟩ 1
  let values : Fin 2 → EucSpace cfg.head_dim := fun _ => (1 / 2 : ℝ) • unit
  let scores : Fin 2 → ℝ := fun j => if j = 1 then 3 else 0
  have hnorm : ‖values 1‖ = 1 / 2 := by
    norm_num [values, unit, norm_smul, PiLp.norm_single]
  have hgap : ∀ j, j ≤ (1 : Fin 2) → j ≠ 1 → scores j + 1 < scores 1 := by
    intro j _ hj
    simp [scores, hj]
  refine ⟨scores, values, hgap, hnorm, ?_⟩
  rw [sparseXSAOutput_clipped_self cfg 1 scores values 1
    (by norm_num) (by rw [hnorm]; norm_num) (fun j hj hn => (hgap j hj hn).le)]
  rw [norm_smul, hnorm]
  norm_num

/-- The unconditional implementation-level H4 claim is false: a strict
self route need not be erased. Source: `EXPERIMENT_PLAN.md`, H4, compared
with Algorithm 1 of arXiv:2603.09078v1, §2, and lab commit `73f8a0b`.
The concrete norm counterexample above supplies the competing scores. -/
theorem clipped_self_route_not_erased (cfg : Config) :
    ∃ scores : Fin 2 → ℝ, ∃ values : Fin 2 → EucSpace cfg.head_dim,
      (∀ j, j ≤ (1 : Fin 2) → j ≠ 1 → scores j + 1 < scores 1) ∧
      sparseXSAOutput cfg 1 scores values 1 ≠ 0 := by
  obtain ⟨scores, values, hgap, _, hout⟩ := clipped_self_route_counterexample cfg
  refine ⟨scores, values, hgap, ?_⟩
  intro hz
  rw [hz, norm_zero] at hout
  norm_num at hout

end Transformer.GPTMini.Sparsemax
