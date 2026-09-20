/-
# Attention's forward pass and Frank-Wolfe — the models

Formalization of Alcalde, Geshkovski, Ruiz-Balet — arXiv:2508.09628v1,
*Attention's forward pass and Frank-Wolfe*, §1.

The paper studies an encoder-only, single-head, MLP-free Transformer in
discrete time, with the tokens living in `ℝ^d` rather than on the sphere.  The
layer map is `eq:sa`, its renormalized form `eq: rescaled.Tformers`, and the
object of study is the singular limit `β → +∞`, `eq: hardmax.dynamics.V`,
which for `V^t = h^t I_d` becomes the Frank-Wolfe iteration `(SA_∞)`.

This file carries the vocabulary: the four dynamics, the leader set `𝒞_i^t`,
the convex hull `𝒦^t` of a configuration, the quadratic objective `J`, and the
cells `𝒞_i(v)` of `eq: cells`.  The theorems are in the `Section2_*` …
`Section5_*` files.
-/

import Transformer.Basic
import Mathlib.Analysis.Convex.Combination
import Mathlib.Analysis.Convex.Extreme
import Mathlib.Analysis.InnerProductSpace.Positive

open scoped BigOperators
open Real

namespace Transformer
namespace FrankWolfe

variable {d n κ : ℕ}

/-! ### The finite-`β` models -/

/-- **Equation (eq:sa).**  One layer of the self-attention model,

  `x_i^{t+1} = x_i^t + V Σ_j (e^{β ⟨B x_i^t, x_j^t⟩} / Σ_k e^{β ⟨B x_i^t, x_k^t⟩}) x_j^t`,

with key-query matrix `B` and value matrix `V`.

Source: arXiv:2508.09628v1, §1.1, `eq:sa`. -/
noncomputable def selfAttentionStep (β : ℝ) (V B : ParamMatrix d)
    (X : Idx n → EucSpace d) (i : Idx n) : EucSpace d :=
  X i + V ((∑ k : Idx n, Real.exp (β * inner (𝕜 := ℝ) (B (X i)) (X k)))⁻¹ •
    ∑ j : Idx n, Real.exp (β * inner (𝕜 := ℝ) (B (X i)) (X j)) • X j)

/-- **Equation (SA_β), `eq: softmax.ODE`.**  The renormalized layer at
`V^t = γ I_d` and `B^t ≡ I_d`:

  `x_i^{t+1} = (1 - γ) x_i^t + γ Σ_j (e^{β ⟨x_i^t, x_j^t⟩} / Σ_k e^{β ⟨x_i^t, x_k^t⟩}) x_j^t`.

Source: arXiv:2508.09628v1, §5, `eq: softmax.ODE`. -/
noncomputable def softmaxStep (β γ : ℝ) (X : Idx n → EucSpace d) (i : Idx n) : EucSpace d :=
  (1 - γ) • X i + γ • ((∑ k : Idx n, Real.exp (β * inner (𝕜 := ℝ) (X i) (X k)))⁻¹ •
    ∑ j : Idx n, Real.exp (β * inner (𝕜 := ℝ) (X i) (X j)) • X j)

/-- `P` is the preconditioner of the renormalized layer `eq: rescaled.Tformers`
built from the value matrix `V`, namely `P = (I_d + V)^{-1} V`.

Written as `(I_d + V) P = V`, which says the same thing whenever `I_d + V` is
invertible — the paper's standing assumption — and needs no inverse.

Source: arXiv:2508.09628v1, §1.1, `eq: rescaled.Tformers`, and §2.2. -/
def IsPreconditioner (V P : ParamMatrix d) : Prop :=
  ∀ x : EucSpace d, P x + V (P x) = V x

/-- **Equation (eq: rescaled.Tformers).**  The renormalized layer

  `x_i^{t+1} = x_i^t + P (Σ_j (e^{β ⟨B x_i^t, x_j^t⟩} / Σ_k e^{β ⟨B x_i^t, x_k^t⟩}) x_j^t - x_i^t)`,

with `P = (I_d + V)^{-1} V` carried as a parameter, pinned by
`IsPreconditioner`.

Source: arXiv:2508.09628v1, §1.1, `eq: rescaled.Tformers`. -/
noncomputable def renormalizedStep (β : ℝ) (P B : ParamMatrix d)
    (X : Idx n → EucSpace d) (i : Idx n) : EucSpace d :=
  X i + P ((∑ k : Idx n, Real.exp (β * inner (𝕜 := ℝ) (B (X i)) (X k)))⁻¹ •
    (∑ j : Idx n, Real.exp (β * inner (𝕜 := ℝ) (B (X i)) (X j)) • X j) - X i)

/-! ### The hardmax limit -/

/-- The leader set `𝒞_i^t` of `eq: hardmax.dynamics.V`: those particles at
which `y ↦ ⟨B x_i^t, y⟩` is maximal over the configuration.

Source: arXiv:2508.09628v1, §1.2, the display under `eq: hardmax.dynamics.V`. -/
noncomputable def leaderIdx (B : ParamMatrix d) (X : Idx n → EucSpace d) (i : Idx n) :
    Finset (Idx n) :=
  open Classical in
  Finset.univ.filter fun j => ∀ k : Idx n,
    inner (𝕜 := ℝ) (B (X i)) (X k) ≤ inner (𝕜 := ℝ) (B (X i)) (X j)

/-- **Equation (eq: hardmax.dynamics.V).**  The `β → +∞` limit of
`eq: rescaled.Tformers`:

  `x_i^{t+1} = x_i^t + P (#𝒞_i^t)^{-1} Σ_{y ∈ 𝒞_i^t} y - x_i^t)`,

with `P = (I_d + V)^{-1} V`.

Source: arXiv:2508.09628v1, §1.2, `eq: hardmax.dynamics.V`. -/
noncomputable def hardmaxAverageStep (P B : ParamMatrix d)
    (X : Idx n → EucSpace d) (i : Idx n) : EucSpace d :=
  X i + P (((leaderIdx B X i).card : ℝ)⁻¹ • (∑ j ∈ leaderIdx B X i, X j) - X i)

/-- The convex hull `𝒦^t = conv{x_i^t}` of a configuration. -/
def configHull (X : Idx n → EucSpace d) : Set (EucSpace d) :=
  convexHull ℝ (Set.range X)

/-- `y` maximizes the linear functional `⟨a, ·⟩` over `K`. -/
def IsMaximizerOn (a : EucSpace d) (K : Set (EucSpace d)) (y : EucSpace d) : Prop :=
  y ∈ K ∧ ∀ z ∈ K, inner (𝕜 := ℝ) a z ≤ inner (𝕜 := ℝ) a y

/-- **Equation (SA_∞), `HSA`.**  One step of the hardmax dynamics at
`V^t = h^t I_d`, `γ^t = h^t / (1 + h^t)`:

  `x_i^{t+1} = x_i^t + γ^t (argmax_{y ∈ 𝒦^t} ⟨B^t x_i^t, y⟩ - x_i^t)`.

The `argmax` is carried as an existential rather than as a function: it is a
singleton only for almost every configuration (`lem:singleLeader`), so a step
is a *relation* between the configuration and its successor, and any theorem
proved about it holds for every selection of maximizers.

Source: arXiv:2508.09628v1, §2.3, `(SA_∞)`. -/
def IsHardmaxStep (B : ParamMatrix d) (γ : ℝ) (X Y : Idx n → EucSpace d) : Prop :=
  ∀ i : Idx n, ∃ y : EucSpace d, IsMaximizerOn (B (X i)) (configHull X) y ∧
    Y i = X i + γ • (y - X i)

/-- A trajectory of `(SA_∞)` with time-dependent key-query matrices and
step-sizes. -/
def HardmaxFlow (B : ℕ → ParamMatrix d) (γ : ℕ → ℝ) (x : ℕ → Idx n → EucSpace d) : Prop :=
  ∀ t : ℕ, IsHardmaxStep (B t) (γ t) (x t) (x (t + 1))

/-! ### The quadratic objective and the cells -/

/-- The quadratic objective `J(x) = ½ ⟨B x, x⟩` of §2.2, whose Frank-Wolfe
iteration `(SA_∞)` is.

Source: arXiv:2508.09628v1, §2.2 and §3. -/
noncomputable def quadForm (B : ParamMatrix d) (x : EucSpace d) : ℝ :=
  (1 / 2 : ℝ) * inner (𝕜 := ℝ) (B x) x

/-- **Equation (eq: cells).**  The cell of the vertex `v i` of a polytope `K`:

  `𝒞_i(v) = {x ∈ K : ⟨B x, v_i⟩ = max_{y ∈ K} ⟨B x, y⟩}`,

written with the maximality spelled out as an inequality.

Source: arXiv:2508.09628v1, §4, `eq: cells`. -/
def cell (B : ParamMatrix d) (K : Set (EucSpace d)) (v : Idx κ → EucSpace d) (i : Idx κ) :
    Set (EucSpace d) :=
  {x ∈ K | ∀ y ∈ K, inner (𝕜 := ℝ) (B x) y ≤ inner (𝕜 := ℝ) (B x) (v i)}

/-- `B` is positive definite, `B ≻ 0`: symmetric, with `⟨B x, x⟩ > 0` away from
the origin.  `ContinuousLinearMap.IsPositive` is the paper's `B ≽ 0`. -/
def IsPosDef (B : ParamMatrix d) : Prop :=
  B.IsSymmetric ∧ ∀ x : EucSpace d, x ≠ 0 → 0 < inner (𝕜 := ℝ) (B x) x

/-- `v` enumerates the vertices of the polytope `K` without repetition. -/
def IsVertexList (K : Set (EucSpace d)) (v : Idx κ → EucSpace d) : Prop :=
  Function.Injective v ∧ Set.range v = Set.extremePoints ℝ K

end FrankWolfe
end Transformer
