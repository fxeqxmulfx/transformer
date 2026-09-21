/-
# The emergence of clusters in self-attention dynamics — how fast a
  coordinate may grow

§9 of arXiv:2305.05465v6, continued: the growth bound `l:nottoofassst` on the
eigencoordinates of the unrescaled tokens, and its replacement `e:notdi` when
`V` is not diagonalizable (`r:notdiagonalizable`).

**What the source says and what is carried here.**

* Both statements are about `x_i(t) = e^{tV}z_i(t)`, the tokens of
  `eq:trans_dyn`; that is how they are stated here, and
  `transformerDynamics_iff_rescaled` is what reads them back on `z_i`.

* `l:nottoofassst` is stated for any `k ∈ [d]`, so its `φ*_k` and `λ_k` may be
  complex and `|λ_k|` is a modulus.  `IsEigenFunctional` is stated over
  `RCLike`, so the complex case is `𝕜 = ℂ`: an `ℝ`-linear `f : ℝ^d → ℂ` with
  `f(Vz) = λ f(z)` is exactly a dual eigenvector of the complexification
  restricted to real vectors, and `|λ_k|` is `‖lam‖`.

* `r:notdiagonalizable` recalls a standard normal form — a Jordan basis whose
  superdiagonal entries are an arbitrary `ε ≠ 0` (Horn–Johnson, Cor. 3.1.21) —
  and derives from it the bound `‖π_F(Vx)‖ ≤ (|λ_k| + δ)‖π_F(x)‖`, which is
  what the modified proof runs on.  That bound is `IsProjContraction`, carried
  as the hypothesis of `e:notdi`; the normal form itself is quoted linear
  algebra, not a claim of the paper, and is not restated.  Only linearity of
  `proj` is used, not that it is an orthogonal projection.

Source: arXiv:2305.05465v6, `l:nottoofassst`, `r:notdiagonalizable`,
`e:notdi`.
-/

import Transformer.Clusters.Section9_Limits
import Transformer.Clusters.Section9_Fj

open scoped BigOperators
open Real Filter Topology

namespace Transformer
namespace Clusters

variable {d n : ℕ}

/-! ### `l:nottoofassst` -/

/-- At `V = 0` every functional is an eigenfunctional for `λ = 0`; together
with the stationary configuration of `transformerDynamics_const` this
witnesses the hypotheses below. -/
theorem isEigenFunctional_zero {𝕜 : Type*} [RCLike 𝕜] (f : EucSpace d →L[ℝ] 𝕜) :
    IsEigenFunctional (0 : ParamMatrix d) f 0 := by
  intro z
  simp

/-- **Lemma (l:nottoofassst).**  For every `k ∈ [d]` and `i ∈ [n]` there is a
constant `C > 0` with

  `|φ*_k(e^{tV}z_i(t))| ≤ C e^{|λ_k| t}`  for all `t ≥ 0`.

As in the source: `d/dt |φ*_k(x_i)|² ≤ 2|λ_k| |φ*_k(x_i)|²` at an index
where `|φ*_k(x_i)|` is largest, by `eq:trans_dyn` and the row sum of `eq:P`,
and Grönwall for the maximum, `le_sup'_mul_exp_of_hasDerivAt`.

Source: arXiv:2305.05465v6, `l:nottoofassst`. -/
theorem norm_eigenFunctional_le (Q K V : ParamMatrix d) {𝕜 : Type*} [RCLike 𝕜]
    (f : EucSpace d →L[ℝ] 𝕜) (lam : 𝕜) (hf : IsEigenFunctional V f lam)
    (X : ℝ → Idx n → EucSpace d) (hX : TransformerDynamics Q K V X) (i : Idx n) :
    ∃ C : ℝ, 0 < C ∧ ∀ t : ℝ, 0 ≤ t → ‖f (X t i)‖ ≤ C * Real.exp (‖lam‖ * t) := by
  have : Nonempty (Idx n) := ⟨i⟩
  set D := fun j t => ∑ k : Idx n, attentionMatrix Q K (X t) j k • (lam * f (X t k))
  have hw : ∀ j t, HasDerivAt (fun s => f (X s j)) (D j t) t := by
    intro j t
    have := (f.hasFDerivAt (x := X t j)).comp_hasDerivAt t (hX t j)
    simp only [map_sum, map_smul, show ∀ z, f (V z) = lam * f z from hf] at this
    exact this
  set S := Finset.univ.sup' Finset.univ_nonempty fun j => ‖f (X 0 j)‖ ^ 2
  have hS : 0 ≤ S := (sq_nonneg _).trans (Finset.le_sup' (fun j => ‖f (X 0 j)‖ ^ 2)
    (Finset.mem_univ i))
  refine ⟨Real.sqrt S + 1, by positivity, fun t ht => ?_⟩
  have h := le_sup'_mul_exp_of_hasDerivAt (fun j t => ‖f (X t j)‖ ^ 2) _ (2 * ‖lam‖)
    (fun j t => hasDerivAt_norm_sq_rclike (hw j t)) (fun j s hj => ?_) i ht
  · rw [mul_assoc] at h
    exact le_sqrt_add_one_mul_exp (norm_nonneg _) hS h
  · have hk : ∀ k, ‖lam * f (X s k)‖ ≤ ‖lam‖ * ‖f (X s j)‖ := fun k => by
      rw [norm_mul]
      exact mul_le_mul_of_nonneg_left
        ((pow_le_pow_iff_left₀ (norm_nonneg _) (norm_nonneg _) two_ne_zero).1 (hj k))
        (norm_nonneg _)
    have hD := norm_sum_attentionMatrix_smul_le Q K (X s) j _ hk
    have hre := RCLike.re_le_norm ((starRingEnd 𝕜) (f (X s j)) * D j s)
    rw [norm_mul, RCLike.norm_conj] at hre
    have := mul_le_mul_of_nonneg_left hD (norm_nonneg (f (X s j)))
    nlinarith

/-- The hypotheses of `norm_eigenFunctional_le` are satisfiable: at `V = 0`
every functional is an eigenfunctional for `λ = 0`, and no token moves. -/
example (Q K : ParamMatrix d) (f : EucSpace d →L[ℝ] ℂ) (X : Idx n → EucSpace d) :
    IsEigenFunctional (0 : ParamMatrix d) f 0 ∧
      TransformerDynamics Q K 0 (fun _ => X) :=
  ⟨isEigenFunctional_zero f, transformerDynamics_const Q K X⟩

/-! ### `r:notdiagonalizable` -/

/-- **The key observation of `r:notdiagonalizable`.**  On a Jordan subspace
`F`, with the superdiagonal of the block taken small enough,

  `‖π_F(Vx)‖ ≤ (|λ_k| + δ) ‖π_F(x)‖`.

Source: arXiv:2305.05465v6, `r:notdiagonalizable`. -/
def IsProjContraction (V proj : ParamMatrix d) (μ : ℝ) : Prop :=
  ∀ x : EucSpace d, ‖proj (V x)‖ ≤ μ * ‖proj x‖

/-- At `V = 0` the bound holds for any `proj` and any `μ ≥ 0`. -/
theorem isProjContraction_zero (proj : ParamMatrix d) {μ : ℝ} (hμ : 0 ≤ μ) :
    IsProjContraction (0 : ParamMatrix d) proj μ := by
  intro x
  simpa using mul_nonneg hμ (norm_nonneg (proj x))

/-- **Equation (e:notdi).**  The replacement for `l:nottoofassst` when `V` is
not diagonalizable: on each Jordan subspace,

  `∃ C > 0, ∀ t ≥ 0, ∀ i ∈ [n],  ‖π_F(e^{tV}z_i(t))‖ ≤ C e^{(|λ_k| + δ)t}`.

As in the source, the argument of `l:nottoofassst` with
`d/dt ‖π_F(x_i(t))‖²` in place of `d/dt |φ*_k(x_i(t))|²`.

Source: arXiv:2305.05465v6, `e:notdi`. -/
theorem norm_proj_le (Q K V proj : ParamMatrix d) (μ : ℝ) (hμ : 0 ≤ μ)
    (hπ : IsProjContraction V proj μ) (X : ℝ → Idx n → EucSpace d)
    (hX : TransformerDynamics Q K V X) :
    ∃ C : ℝ, 0 < C ∧ ∀ t : ℝ, 0 ≤ t → ∀ i : Idx n,
      ‖proj (X t i)‖ ≤ C * Real.exp (μ * t) := by
  rcases isEmpty_or_nonempty (Idx n) with hn | hn
  · exact ⟨1, one_pos, fun _ _ i => isEmptyElim i⟩
  set D := fun j t => ∑ k : Idx n, attentionMatrix Q K (X t) j k • proj (V (X t k))
  have hw : ∀ j t, HasDerivAt (fun s => proj (X s j)) (D j t) t := by
    intro j t
    have := (proj.hasFDerivAt (x := X t j)).comp_hasDerivAt t (hX t j)
    simp only [map_sum, map_smul] at this
    exact this
  set S := Finset.univ.sup' Finset.univ_nonempty fun j => ‖proj (X 0 j)‖ ^ 2
  have hS : 0 ≤ S := (sq_nonneg _).trans (Finset.le_sup' (fun j => ‖proj (X 0 j)‖ ^ 2)
    (Finset.mem_univ (Classical.arbitrary _)))
  refine ⟨Real.sqrt S + 1, by positivity, fun t ht i => ?_⟩
  have h := le_sup'_mul_exp_of_hasDerivAt (fun j t => ‖proj (X t j)‖ ^ 2) _ (2 * μ)
    (fun j t => (hw j t).norm_sq) (fun j s hj => ?_) i ht
  · rw [mul_assoc] at h
    exact le_sqrt_add_one_mul_exp (norm_nonneg _) hS h
  · have hk : ∀ k, ‖proj (V (X s k))‖ ≤ μ * ‖proj (X s j)‖ := fun k =>
      (hπ _).trans (mul_le_mul_of_nonneg_left
        ((pow_le_pow_iff_left₀ (norm_nonneg _) (norm_nonneg _) two_ne_zero).1 (hj k)) hμ)
    have hD := norm_sum_attentionMatrix_smul_le Q K (X s) j _ hk
    have hin := real_inner_le_norm (proj (X s j)) (D j s)
    have := mul_le_mul_of_nonneg_left hD (norm_nonneg (proj (X s j)))
    nlinarith

/-- The hypotheses of `norm_proj_le` are satisfiable at `V = 0`. -/
example (Q K proj : ParamMatrix d) (X : Idx n → EucSpace d) :
    (0 : ℝ) ≤ 0 ∧ IsProjContraction (0 : ParamMatrix d) proj 0 ∧
      TransformerDynamics Q K 0 (fun _ => X) :=
  ⟨le_rfl, isProjContraction_zero proj le_rfl, transformerDynamics_const Q K X⟩

end Clusters
end Transformer
