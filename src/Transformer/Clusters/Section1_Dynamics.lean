/-
# The emergence of clusters in self-attention dynamics — the dynamics

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2305.05465v6.

This file carries the objects of §1: the self-attention matrix `eq:P`, the
transformer dynamics `eq:trans_dyn` it drives, the forward Euler iteration
`e:discreteequation` of `r:discretetime`, and the positivity hypothesis
`QᵀK ≻ 0` that every clustering theorem of the paper is stated under.

**What the source says and what is carried here.**

* Unlike the sphere dynamics of `Transformer.Perspective`, the tokens here
  live in `ℝ^d` and there is no projection and no inverse temperature: `β` is
  absorbed into `Q`.

* `P_ij(t)` is `attentionMatrix Q K (X t) i j`, a function of the
  configuration rather than of a curve and a time, so that it can be evaluated
  along the continuous and the discrete dynamics alike.  It is
  `Perspective.softmaxWeight` of the score row, which is where its two
  defining properties — non-negativity and row sum `1` — come from.

* The source notes that `Q` and `K` "need not be square".  They are carried as
  square `d × d` maps, and nothing is lost: they enter only through the
  bilinear form `(u,v) ↦ ⟨Qu, Kv⟩`, and every form a rectangular pair realizes
  is realized by a square pair.  A rectangular `Q : ℝ^d → ℝ^m` with `m < d`
  is in any case excluded by `QᵀK ≻ 0`, which forces `QᵀK` to have full rank.

* `QᵀK ≻ 0` is `IsPosDefQK`: the form `⟨Q·, K·⟩` is symmetric and positive
  definite.  Symmetry is part of it — the source takes the square root
  `A := (QᵀK)^{1/2}` in `eq: A`, which presupposes it.

Source: arXiv:2305.05465v6, `eq:trans_dyn`, `eq:P`, `r:discretetime`,
`e:discreteequation`, `eq: A`.
-/

import Transformer.Perspective.Softmax

open scoped BigOperators
open Real

namespace Transformer
namespace Clusters

variable {d n : ℕ}

/-! ### The self-attention matrix -/

/-- **Equation (eq:P).**  The self-attention matrix of a configuration:

  `P_ij = e^{⟨Q x_i, K x_j⟩} / Σ_ℓ e^{⟨Q x_i, K x_ℓ⟩}`.

Source: arXiv:2305.05465v6, `eq:P`. -/
noncomputable def attentionMatrix (Q K : ParamMatrix d) (X : Idx n → EucSpace d)
    (i j : Idx n) : ℝ :=
  Perspective.softmaxWeight (fun l : Idx n => inner (𝕜 := ℝ) (Q (X i)) (K (X l))) j

/-- Every entry of the self-attention matrix is positive. -/
theorem attentionMatrix_pos (Q K : ParamMatrix d) (X : Idx n → EucSpace d)
    (i j : Idx n) : 0 < attentionMatrix Q K X i j := by
  have hn : 0 < n := Fin.pos_iff_nonempty.mpr ⟨j⟩
  exact div_pos (Real.exp_pos _)
    (Perspective.softmaxPartition_pos hn
      (fun l : Idx n => inner (𝕜 := ℝ) (Q (X i)) (K (X l))))

/-- Every entry of the self-attention matrix is non-negative. -/
theorem attentionMatrix_nonneg (Q K : ParamMatrix d) (X : Idx n → EucSpace d)
    (i j : Idx n) : 0 ≤ attentionMatrix Q K X i j :=
  Perspective.softmaxWeight_nonneg _ j

/-- **The self-attention matrix is stochastic**: each of its rows sums to `1`.
This is what makes `Σ_j P_ij V x_j` an average of the values `V x_j`, and it
is used in every proof of the paper. -/
theorem sum_attentionMatrix (hn : 0 < n) (Q K : ParamMatrix d)
    (X : Idx n → EucSpace d) (i : Idx n) :
    ∑ j : Idx n, attentionMatrix Q K X i j = 1 :=
  Perspective.sum_softmaxWeight hn _

/-- The hypothesis of `sum_attentionMatrix` is satisfiable. -/
example : 0 < 1 := one_pos

/-! ### The dynamics -/

/-- **Equation (eq:trans_dyn).**  The transformer dynamics: `n` tokens of
`ℝ^d` moving by

  `ẋ_i(t) = Σ_j P_ij(t) V x_j(t)`.

Source: arXiv:2305.05465v6, `eq:trans_dyn`. -/
def TransformerDynamics (Q K V : ParamMatrix d) (X : ℝ → Idx n → EucSpace d) : Prop :=
  ∀ (t : ℝ) (i : Idx n),
    HasDerivAt (fun s => X s i)
      (∑ j : Idx n, attentionMatrix Q K (X t) i j • V (X t j)) t

/-- **Equation (e:discreteequation).**  The forward Euler iteration with time
step `Δt`, the discrete-time transformer of `r:discretetime`:

  `x_i((k+1)Δt) = x_i(kΔt) + Δt Σ_j P_ij(kΔt) V x_j(kΔt)`.

Source: arXiv:2305.05465v6, `r:discretetime`, `e:discreteequation`. -/
def DiscreteTransformer (Δt : ℝ) (Q K V : ParamMatrix d)
    (X : ℕ → Idx n → EucSpace d) : Prop :=
  ∀ (k : ℕ) (i : Idx n),
    X (k + 1) i = X k i + Δt • ∑ j : Idx n, attentionMatrix Q K (X k) i j • V (X k j)

/-- **A single token does not move.**  With `n = 1` the only attention weight
is `1` and the dynamics reads `ẋ = V x`; at `V = 0` the constant curve solves
`eq:trans_dyn`.  This is the one solution available in closed form, and it is
what witnesses the hypotheses of the theorems below. -/
theorem transformerDynamics_const (Q K : ParamMatrix d) (X : Idx n → EucSpace d) :
    TransformerDynamics Q K 0 (fun _ => X) := by
  intro t i
  simpa using (hasDerivAt_const t (X i))

/-- The discrete dynamics agrees: with `V = 0` no token moves. -/
theorem discreteTransformer_const (Δt : ℝ) (Q K : ParamMatrix d)
    (X : Idx n → EucSpace d) : DiscreteTransformer Δt Q K 0 (fun _ => X) := by
  intro k i
  simp

/-! ### `QᵀK ≻ 0` -/

/-- **The hypothesis `QᵀK ≻ 0`.**  The bilinear form `(u,v) ↦ ⟨Qu, Kv⟩` is
symmetric and positive definite.

Source: arXiv:2305.05465v6, `t:Idcase11int`, `d:goodmulti` (i). -/
def IsPosDefQK (Q K : ParamMatrix d) : Prop :=
  (∀ u v : EucSpace d, inner (𝕜 := ℝ) (Q u) (K v) = inner (𝕜 := ℝ) (Q v) (K u)) ∧
    ∀ u : EucSpace d, u ≠ 0 → 0 < inner (𝕜 := ℝ) (Q u) (K u)

/-- **`V ≻ 0`.**  Positive definiteness of a single map, the hypothesis
`t:boolean` puts on the value matrix; in `d = 1` it is the scalar condition
`V > 0`.

Source: arXiv:2305.05465v6, `t:boolean`. -/
def IsPosDefOp (V : ParamMatrix d) : Prop :=
  ∀ u : EucSpace d, u ≠ 0 → 0 < inner (𝕜 := ℝ) (V u) u

/-- The identity is positive definite, so `V ≻ 0` is satisfiable. -/
theorem isPosDefOp_id (d : ℕ) :
    IsPosDefOp (ContinuousLinearMap.id ℝ (EucSpace d)) :=
  fun _ hu => real_inner_self_pos.mpr hu

/-- **Equation (eq: A).**  `A = (QᵀK)^{1/2}`, carried by what characterizes it:
a symmetric positive definite map whose square is `QᵀK`, written as the
identity `⟨Qu, Kv⟩ = ⟨Au, Av⟩` that the proofs use it through.

Source: arXiv:2305.05465v6, `eq: A`. -/
def IsAttentionRoot (Q K A : ParamMatrix d) : Prop :=
  (∀ u v : EucSpace d, inner (𝕜 := ℝ) (A u) v = inner (𝕜 := ℝ) u (A v)) ∧
    (∀ u : EucSpace d, u ≠ 0 → 0 < inner (𝕜 := ℝ) (A u) u) ∧
      ∀ u v : EucSpace d, inner (𝕜 := ℝ) (Q u) (K v) = inner (𝕜 := ℝ) (A u) (A v)

/-- **A square root witnesses `QᵀK ≻ 0`.**  The form `⟨Au, Av⟩` is symmetric
outright, and it is positive definite because a positive definite `A` is
injective. -/
theorem isPosDefQK_of_isAttentionRoot {Q K A : ParamMatrix d}
    (h : IsAttentionRoot Q K A) : IsPosDefQK Q K := by
  obtain ⟨-, hpos, hform⟩ := h
  refine ⟨fun u v => ?_, fun u hu => ?_⟩
  · rw [hform, hform, real_inner_comm]
  · rw [hform, real_inner_self_eq_norm_sq]
    have hAu : A u ≠ 0 := by
      intro hzero
      have := hpos u hu
      rw [hzero, inner_zero_left] at this
      exact lt_irrefl 0 this
    positivity

/-- The identity is its own attention root, so both hypotheses are
satisfiable: `Q = K = A = I_d` is the case `Q = K = I_d` of the source's
figures. -/
theorem isAttentionRoot_id (d : ℕ) :
    IsAttentionRoot (d := d) (ContinuousLinearMap.id ℝ (EucSpace d))
      (ContinuousLinearMap.id ℝ (EucSpace d)) (ContinuousLinearMap.id ℝ (EucSpace d)) :=
  ⟨fun _ _ => rfl, fun _ hu => real_inner_self_pos.mpr hu, fun _ _ => rfl⟩

/-- Hence `QᵀK ≻ 0` is satisfiable. -/
example (d : ℕ) : IsPosDefQK (d := d) (ContinuousLinearMap.id ℝ (EucSpace d))
    (ContinuousLinearMap.id ℝ (EucSpace d)) :=
  isPosDefQK_of_isAttentionRoot (isAttentionRoot_id d)

end Clusters
end Transformer
