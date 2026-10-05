import Transformer.GPTMini.Sparsemax.Basic
import Transformer.GPTMini.CausalMHA

/-!
# A saturated self route through the implemented XSA projection

arXiv:2603.09078v1, §2, equation `xsa` and Algorithm 1, combined with
arXiv:1602.02068v2, §2.2, Proposition 1. The lab at commit `73f8a0b`
normalizes values with `max(norm, eps)`. Exact self-route erasure therefore
requires the self-value norm to exceed epsilon. A strict score gap and
strict norm bound make the whole head locally zero, including value paths.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.ConvexRecall Transformer.GPTMini.Convex Filter
open scoped BigOperators Topology

/-- Apply the existing epsilon-clipped XSA to an actual causal sparsemax
row and its value stream. Source: lab `Attention.forward` at `73f8a0b`;
arXiv:2603.09078v1, §2, equation `xsa`, with sparsemax from §2.2 of
arXiv:1602.02068v2 instead of the paper's softmax. -/
def sparseXSAOutput (cfg : Config) {T : ℕ} (eps : ℝ) (scores : Fin T → ℝ)
    (values : Fin T → EucSpace cfg.head_dim) (i : Fin T) : EucSpace cfg.head_dim :=
  xsaProjection cfg eps values
    (fun r => ∑ j, sparseWeights scores r j • values j) i

/-- Above epsilon, implemented XSA erases a pure self value exactly.
Source: arXiv:2603.09078v1, §2, equation `xsa` and Algorithm 1.
The added norm hypothesis accounts for the lab's epsilon clipping. -/
theorem xsaProjection_self_above_epsilon (cfg : Config) {T : ℕ} (eps : ℝ)
    (values : Fin T → EucSpace cfg.head_dim) (i : Fin T)
    (heps : 0 ≤ eps) (hv : eps < ‖values i‖) :
    xsaProjection cfg eps values values i = 0 := by
  have hn : ‖values i‖ ≠ 0 := ne_of_gt (lt_of_le_of_lt heps hv)
  dsimp only [xsaProjection, normL2]
  rw [max_eq_left hv.le, real_inner_smul_right, real_inner_self_eq_norm_sq, smul_smul]
  have hscalar : (1 / ‖values i‖ * ‖values i‖ ^ 2) * (1 / ‖values i‖) = 1 := by
    field_simp
  rw [hscalar, one_smul, sub_self]

/-- The added epsilon premise is realized by a unit self value.
Source context: lab `Attention.forward` at `73f8a0b`, epsilon-clipped XSA. -/
example (cfg : Config) : (0 : ℝ) ≤ 1 / 2 ∧
    1 / 2 < ‖(EuclideanSpace.single ⟨0, cfg.head_dim_pos⟩ 1 : EucSpace cfg.head_dim)‖ := by
  norm_num [PiLp.norm_single]

/-- A self winner separated by a unit gap has zero XSA output when its
value is above epsilon. Source: arXiv:1602.02068v2, §2.2, Proposition 1,
and arXiv:2603.09078v1, §2, equation `xsa`; clipping premise is explicit. -/
theorem sparseXSAOutput_eq_zero_of_gap (cfg : Config) {T : ℕ} (eps : ℝ)
    (scores : Fin T → ℝ) (values : Fin T → EucSpace cfg.head_dim) (i : Fin T)
    (heps : 0 ≤ eps) (hv : eps < ‖values i‖)
    (hgap : ∀ j, j ≤ i → j ≠ i → scores j + 1 ≤ scores i) :
    sparseXSAOutput cfg eps scores values i = 0 := by
  have hw := sparseWeights_eq_basis_of_gap scores i i le_rfl hgap
  have hout : (∑ j, sparseWeights scores i j • values j) = values i := by
    rw [hw]
    simp [basis]
  dsimp only [sparseXSAOutput, xsaProjection]
  rw [hout]
  exact xsaProjection_self_above_epsilon cfg eps values i heps hv

/-- Nontrivial two-slot self winners satisfy all zero-output premises.
Source context: sparsemax Proposition 1 and equation `xsa`. -/
example (cfg : Config) : (0 : ℝ) ≤ 1 / 2 ∧
    1 / 2 < ‖(EuclideanSpace.single ⟨0, cfg.head_dim_pos⟩ 1 : EucSpace cfg.head_dim)‖ ∧
    ∀ j : Fin 2, j ≤ 1 → j ≠ 1 →
      (if j = 1 then (3 : ℝ) else 0) + 1 ≤ 3 := by
  refine ⟨by norm_num, by norm_num [PiLp.norm_single], ?_⟩
  intro j _ hj
  simp [hj]

/-- Strict score and epsilon gaps give a neighborhood on which the head
is zero while both scores and all values vary. Source: arXiv:1602.02068v2,
§2.2, Proposition 1, and arXiv:2603.09078v1, §2, equation `xsa`.
This corrects the plan's unconditional H4 wording for epsilon clipping. -/
theorem sparseXSAOutput_eventually_zero (cfg : Config) {T : ℕ} (eps : ℝ)
    (scores : Fin T → ℝ) (values : Fin T → EucSpace cfg.head_dim) (i : Fin T)
    (heps : 0 ≤ eps) (hv : eps < ‖values i‖)
    (hgap : ∀ j, j ≤ i → j ≠ i → scores j + 1 < scores i) :
    ∀ᶠ state : (Fin T → ℝ) × (Fin T → EucSpace cfg.head_dim) in 𝓝 (scores, values),
      sparseXSAOutput cfg eps state.1 state.2 i = 0 := by
  have hweights := sparseWeights_eventually_eq_basis scores i i le_rfl hgap
  have hw : ∀ᶠ state : (Fin T → ℝ) × (Fin T → EucSpace cfg.head_dim) in 𝓝 (scores, values),
      sparseWeights state.1 i = basis i :=
    (show ContinuousAt
      (fun state : (Fin T → ℝ) × (Fin T → EucSpace cfg.head_dim) => state.1)
      (scores, values) from continuous_fst.continuousAt).tendsto.eventually hweights
  have hnorm : ContinuousAt
      (fun state : (Fin T → ℝ) × (Fin T → EucSpace cfg.head_dim) => ‖state.2 i‖)
      (scores, values) := ((continuous_apply i).comp continuous_snd).continuousAt.norm
  have hn := continuousAt_const.eventually_lt hnorm hv
  filter_upwards [hw, hn] with state hs hsval
  have hout : (∑ j, sparseWeights state.1 i j • state.2 j) = state.2 i := by
    rw [hs]
    simp [basis]
  dsimp only [sparseXSAOutput, xsaProjection]
  rw [hout]
  exact xsaProjection_self_above_epsilon cfg eps state.2 i heps hsval

/-- Strict score and norm gaps hold on an actual two-slot self route.
Source context: sparsemax Proposition 1 and equation `xsa`. -/
example (cfg : Config) : (0 : ℝ) ≤ 1 / 2 ∧
    1 / 2 < ‖(EuclideanSpace.single ⟨0, cfg.head_dim_pos⟩ 1 : EucSpace cfg.head_dim)‖ ∧
    ∀ j : Fin 2, j ≤ 1 → j ≠ 1 →
      (if j = 1 then (3 : ℝ) else 0) + 1 < 3 := by
  refine ⟨by norm_num, by norm_num [PiLp.norm_single], ?_⟩
  intro j _ hj
  simp [hj]

/-- Every score and value direction has zero derivative on that
neighborhood. Source: sparsemax Proposition 1 and equation `xsa`;
this does not eliminate other rows, residual paths or optimizer state. -/
theorem sparseXSAOutput_hasFDerivAt_zero (cfg : Config) {T : ℕ} (eps : ℝ)
    (scores : Fin T → ℝ) (values : Fin T → EucSpace cfg.head_dim) (i : Fin T)
    (heps : 0 ≤ eps) (hv : eps < ‖values i‖)
    (hgap : ∀ j, j ≤ i → j ≠ i → scores j + 1 < scores i) :
    HasFDerivAt (𝕜 := ℝ)
      (fun state : (Fin T → ℝ) × (Fin T → EucSpace cfg.head_dim) =>
        sparseXSAOutput cfg eps state.1 state.2 i) 0 (scores, values) := by
  exact (hasFDerivAt_const (𝕜 := ℝ) (0 : EucSpace cfg.head_dim) (scores, values)).congr_of_eventuallyEq
    (sparseXSAOutput_eventually_zero cfg eps scores values i heps hv hgap)

/-- The zero-derivative premises have a nonzero value and competing slot.
Source context: sparsemax Proposition 1 and equation `xsa`. -/
example (cfg : Config) : (0 : ℝ) ≤ 1 / 2 ∧
    1 / 2 < ‖(EuclideanSpace.single ⟨0, cfg.head_dim_pos⟩ 1 : EucSpace cfg.head_dim)‖ ∧
    ∀ j : Fin 2, j ≤ 1 → j ≠ 1 →
      (if j = 1 then (4 : ℝ) else 0) + 1 < 4 := by
  refine ⟨by norm_num, by norm_num [PiLp.norm_single], ?_⟩
  intro j _ hj
  simp [hj]

/-- Continuous score/value parameterizations inherit the zero derivative,
including query/key/value parameter paths. Source: sparsemax Proposition 1
and equation `xsa`, with the strict epsilon correction above. Continuity
is explicit; no assumption about another row or the optimizer is made. -/
theorem sparseXSAOutput_parameter_hasFDerivAt_zero (cfg : Config) {T : ℕ}
    {P : Type*} [NormedAddCommGroup P] [NormedSpace ℝ P] (eps : ℝ)
    (scores : P → Fin T → ℝ) (values : P → Fin T → EucSpace cfg.head_dim)
    (p : P) (i : Fin T) (hs : ContinuousAt scores p) (hv : ContinuousAt values p)
    (heps : 0 ≤ eps) (hn : eps < ‖values p i‖)
    (hgap : ∀ j, j ≤ i → j ≠ i → scores p j + 1 < scores p i) :
    HasFDerivAt (𝕜 := ℝ)
      (fun parameter => sparseXSAOutput cfg eps (scores parameter) (values parameter) i) 0 p := by
  have hzero := sparseXSAOutput_eventually_zero cfg eps (scores p) (values p) i heps hn hgap
  have he := (hs.prodMk hv).tendsto.eventually hzero
  exact (hasFDerivAt_const (𝕜 := ℝ) (0 : EucSpace cfg.head_dim) p).congr_of_eventuallyEq he

/-- A varying parameter satisfies the continuity, strict-gap and value
premises. Source context: the corrected self-route theorem for equation
`xsa`; the score path moves both competing scores with the parameter. -/
example (cfg : Config) :
    ContinuousAt (fun p : ℝ => fun j : Fin 2 => if j = 1 then p + 3 else p) 0 ∧
    ContinuousAt (fun _p : ℝ => fun _j : Fin 2 =>
      (EuclideanSpace.single ⟨0, cfg.head_dim_pos⟩ 1 : EucSpace cfg.head_dim)) 0 ∧
    (0 : ℝ) ≤ 1 / 2 ∧
    1 / 2 < ‖(EuclideanSpace.single ⟨0, cfg.head_dim_pos⟩ 1 : EucSpace cfg.head_dim)‖ ∧
    ∀ j : Fin 2, j ≤ 1 → j ≠ 1 →
      (if j = 1 then (3 : ℝ) else 0) + 1 < 3 := by
  refine ⟨?_, continuousAt_const, by norm_num, by norm_num [PiLp.norm_single], ?_⟩
  · apply continuousAt_pi.mpr
    intro j
    by_cases hj : j = 1
    · simp only [hj, ite_true]
      exact continuousAt_id.add continuousAt_const
    · simp only [hj, ite_false]
      exact continuousAt_id
  · intro j _ hj
    simp [hj]

end Transformer.GPTMini.Sparsemax
