/-
# The emergence of clusters in self-attention dynamics — `t:boolean`, the reduction

§7 of arXiv:2305.05465v6 opens: "in the statement, `V` is a positive scalar,
but by reparametrizing time we may assume that `V = 1` … to ease notations we
focus on `QK = 1`, but the proof adapts straightforwardly to the setting
`QK > 0`", and it relabels the tokens so that `x_1(0) < … < x_n(0)`.

This file carries that reduction out.  In `d = 1` every map is a scalar:
`V u = v u` and `⟨Q u, K w⟩ = c u w` with `v, c > 0`.  If `x` solves
`eq:trans_dyn` then `y_i(s) = √c x_{σ(i)}(s / v)` solves `e:Idnonresca` for
every relabelling `σ` (`idNonrescaledDynamics_rescale`), and its
self-attention matrix at time `v t` is that of `x` at time `t`, with rows and
columns relabelled (`attentionMatrix_rescale`).  Instead of adapting the proof
to `QK > 0`, the factor `c` is absorbed into the tokens.

Source: arXiv:2305.05465v6, §7, the paragraph before `e:Idnonresca`.
-/

import Transformer.Clusters.Section7_Symmetric
import Transformer.Clusters.Section7_UnboundedParticles

open scoped BigOperators
open Real Filter Topology

namespace Transformer
namespace Clusters

variable {n : ℕ}

/-- The softmax commutes with relabelling the scores. -/
theorem softmaxWeight_comp_perm (u : Idx n → ℝ) (σ : Equiv.Perm (Idx n)) (j : Idx n) :
    Perspective.softmaxWeight (u ∘ σ) j = Perspective.softmaxWeight u (σ j) := by
  simp only [Perspective.softmaxWeight, Function.comp_apply]
  rw [Equiv.sum_comp σ (fun k => Real.exp (u k))]

/-- A vector of `ℝ^1` is its coordinate times the unit vector. -/
theorem eq_smul_unit1 (u : EucSpace 1) : u = u 0 • unit1 := by
  ext k
  fin_cases k
  simp [unit1]

/-- In `d = 1` a linear map is multiplication by its value at the unit vector. -/
theorem coord_apply_one (L : ParamMatrix 1) (u : EucSpace 1) : L u 0 = (L unit1) 0 * u 0 := by
  conv_lhs => rw [eq_smul_unit1 u, map_smul]
  simp [mul_comm]

/-- In `d = 1` the attention score is `c u w`, with `c = ⟨Q e, K e⟩`. -/
theorem inner_apply_one (Q K : ParamMatrix 1) (u w : EucSpace 1) :
    inner (𝕜 := ℝ) (Q u) (K w) = inner (𝕜 := ℝ) (Q unit1) (K unit1) * (u 0 * w 0) := by
  conv_lhs => rw [eq_smul_unit1 u, eq_smul_unit1 w, map_smul, map_smul]
  rw [real_inner_smul_left, real_inner_smul_right]
  ring

/-- In `d = 1` the self-attention matrix is the softmax of `c x_i x_l`. -/
theorem attentionMatrix_eq_one_dim (Q K : ParamMatrix 1) (Y : Idx n → EucSpace 1) (i j : Idx n) :
    attentionMatrix Q K Y i j = Perspective.softmaxWeight
      (fun l => inner (𝕜 := ℝ) (Q unit1) (K unit1) * (Y i 0 * Y l 0)) j := by
  simp only [attentionMatrix]
  congr 1
  funext l
  exact inner_apply_one Q K (Y i) (Y l)

/-- **`eq:trans_dyn` in `d = 1`, one coordinate at a time.** -/
theorem hasDerivAt_coord_of_transformer {Q K V : ParamMatrix 1} {X : ℝ → Idx n → EucSpace 1}
    (hX : TransformerDynamics Q K V X) (t : ℝ) (i : Idx n) :
    HasDerivAt (fun s => X s i 0)
      (∑ j, Perspective.softmaxWeight
        (fun l => inner (𝕜 := ℝ) (Q unit1) (K unit1) * (X t i 0 * X t l 0)) j *
          ((V unit1) 0 * X t j 0)) t := by
  have h := (EuclideanSpace.proj (𝕜 := ℝ) (0 : Fin 1)).hasFDerivAt.comp_hasDerivAt t (hX t i)
  have e : (EuclideanSpace.proj (𝕜 := ℝ) (0 : Fin 1))
      (∑ j, attentionMatrix Q K (X t) i j • V (X t j)) =
      ∑ j, Perspective.softmaxWeight
        (fun l => inner (𝕜 := ℝ) (Q unit1) (K unit1) * (X t i 0 * X t l 0)) j *
          ((V unit1) 0 * X t j 0) := by
    simp only [map_sum, map_smul, smul_eq_mul]
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [attentionMatrix_eq_one_dim, ← coord_apply_one]
    rfl
  rw [e] at h
  exact h

/-- **The reduction to `e:Idnonresca`.**  If `x` solves `eq:trans_dyn` in
`d = 1` with `V = v > 0` and `QK = c > 0`, then `y_i(s) = √c x_{σ(i)}(s / v)`
solves `e:Idnonresca`, for any relabelling `σ`.

Source: arXiv:2305.05465v6, §7, "by reparametrizing time we may assume that
`V = 1`". -/
theorem idNonrescaledDynamics_rescale {Q K V : ParamMatrix 1} {X : ℝ → Idx n → EucSpace 1}
    (hX : TransformerDynamics Q K V X) (hv : 0 < (V unit1) 0)
    (hc : 0 < inner (𝕜 := ℝ) (Q unit1) (K unit1)) (σ : Equiv.Perm (Idx n)) :
    IdNonrescaledDynamics (fun s i =>
      (√(inner (𝕜 := ℝ) (Q unit1) (K unit1)) * X (s / (V unit1) 0) (σ i) 0) • unit1) := by
  intro s i
  have hτ : HasDerivAt (fun r => r / (V unit1) 0) (1 / (V unit1) 0) s :=
    (hasDerivAt_id s).div_const _
  have h := (((hasDerivAt_coord_of_transformer hX (s / (V unit1) 0) (σ i)).comp s hτ).const_mul
    √(inner (𝕜 := ℝ) (Q unit1) (K unit1))).smul_const unit1
  set c := inner (𝕜 := ℝ) (Q unit1) (K unit1)
  set v := (V unit1) 0
  refine h.congr_deriv ?_
  simp only [attentionMatrix_one_eq, coord_smul_unit1, smul_smul]
  rw [← Finset.sum_smul]
  congr 1
  have hcc : √c * √c = c := Real.mul_self_sqrt hc.le
  have hw : ∀ j, Perspective.softmaxWeight
      (fun l => √c * X (s / v) (σ i) 0 * (√c * X (s / v) (σ l) 0)) j =
      Perspective.softmaxWeight (fun l => c * (X (s / v) (σ i) 0 * X (s / v) l 0)) (σ j) := by
    intro j
    rw [← softmaxWeight_comp_perm]
    congr 1
    funext l
    simp only [Function.comp_apply]
    linear_combination (X (s / v) (σ i) 0 * X (s / v) (σ l) 0) * hcc
  simp only [hw]
  rw [Equiv.sum_comp σ (fun j => Perspective.softmaxWeight
    (fun l => c * (X (s / v) (σ i) 0 * X (s / v) l 0)) j * (√c * X (s / v) j 0))]
  rw [Finset.sum_mul, Finset.mul_sum]
  refine Finset.sum_congr rfl fun j _ => ?_
  field_simp

/-- **The relabelled, rescaled attention matrix.**  The self-attention matrix
of `y` at time `v t` is that of `x` at time `t`, rows and columns relabelled
by `σ`. -/
theorem attentionMatrix_rescale (Q K : ParamMatrix 1) (X : ℝ → Idx n → EucSpace 1) {v : ℝ}
    (hv : 0 < v) (hc : 0 < inner (𝕜 := ℝ) (Q unit1) (K unit1)) (σ : Equiv.Perm (Idx n))
    (t : ℝ) (i j : Idx n) :
    attentionMatrix (1 : ParamMatrix 1) 1 (fun k =>
        (√(inner (𝕜 := ℝ) (Q unit1) (K unit1)) * X (v * t / v) (σ k) 0) • unit1) i j =
      attentionMatrix Q K (X t) (σ i) (σ j) := by
  set c := inner (𝕜 := ℝ) (Q unit1) (K unit1)
  have hcc : √c * √c = c := Real.mul_self_sqrt hc.le
  rw [attentionMatrix_one_eq, attentionMatrix_eq_one_dim, ← softmaxWeight_comp_perm,
    mul_div_cancel_left₀ t hv.ne']
  congr 1
  funext l
  simp only [coord_smul_unit1, Function.comp_apply]
  linear_combination (X t (σ i) 0 * X t (σ l) 0) * hcc

/-- The hypotheses of the reduction are satisfiable: at `Q = K = V = I_1`, with
the tokens pinned at the origin, `v = c = 1`. -/
example :
    TransformerDynamics (n := 1) (1 : ParamMatrix 1) 1 1 (fun _ _ => 0) ∧
      0 < ((1 : ParamMatrix 1) unit1) 0 ∧
      0 < inner (𝕜 := ℝ) ((1 : ParamMatrix 1) unit1) ((1 : ParamMatrix 1) unit1) := by
  refine ⟨fun t i => by simpa using hasDerivAt_const t (0 : EucSpace 1), ?_, ?_⟩ <;>
    simp [unit1]

end Clusters
end Transformer
