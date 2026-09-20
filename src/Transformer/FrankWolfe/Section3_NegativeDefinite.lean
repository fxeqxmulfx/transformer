/-
# Attention's forward pass and Frank-Wolfe — negative-definite key-query

Formalization of Alcalde, Geshkovski, Ruiz-Balet — arXiv:2508.09628v1,
*Attention's forward pass and Frank-Wolfe*, §3.

Reparametrizing `B^t = -B_*^t` turns `(SA_∞)` into the Frank-Wolfe iteration
`eq: hardmax.dynamics.V_fw` for the convex quadratic `J^t(x) = ½ ⟨B_*^t x, x⟩`,
and the standard Frank-Wolfe rate gives `thm: fw.cluster`: the objective falls
like `1/t`, so the particles cluster at the origin.
-/

import Transformer.FrankWolfe.Section1_Models

open scoped BigOperators
open Real

namespace Transformer
namespace FrankWolfe

variable {d n : ℕ}

/-- `y` minimizes the linear functional `⟨a, ·⟩` over `K`. -/
def IsMinimizerOn (a : EucSpace d) (K : Set (EucSpace d)) (y : EucSpace d) : Prop :=
  y ∈ K ∧ ∀ z ∈ K, inner (𝕜 := ℝ) a y ≤ inner (𝕜 := ℝ) a z

/-- **Equation (eq: hardmax.dynamics.V_fw).**  One Frank-Wolfe step for
`J(x) = ½ ⟨B_* x, x⟩` over `𝒦^t`:

  `x_i^{t+1} = x_i^t + γ^t (argmin_{y ∈ 𝒦^t} ⟨B_* x_i^t, y⟩ - x_i^t)`.

Source: arXiv:2508.09628v1, §3, `eq: hardmax.dynamics.V_fw`. -/
def IsFrankWolfeStep (Bs : ParamMatrix d) (γ : ℝ) (X Y : Idx n → EucSpace d) : Prop :=
  ∀ i : Idx n, ∃ y : EucSpace d, IsMinimizerOn (Bs (X i)) (configHull X) y ∧
    Y i = X i + γ • (y - X i)

/-- The reparametrization `B^t = -B_*^t` of §3: the Frank-Wolfe step for
`B_*` is the hardmax step for `-B_*`.

Source: arXiv:2508.09628v1, §3, the display above `eq: hardmax.dynamics.V_fw`. -/
theorem isFrankWolfeStep_iff_isHardmaxStep (Bs : ParamMatrix d) (γ : ℝ)
    (X Y : Idx n → EucSpace d) :
    IsFrankWolfeStep Bs γ X Y ↔ IsHardmaxStep (-Bs) γ X Y := by
  constructor <;> intro h i <;> obtain ⟨y, ⟨hyK, hy⟩, hYi⟩ := h i <;>
    refine ⟨y, ⟨hyK, fun z hz => ?_⟩, hYi⟩ <;>
    simpa using hy z hz

/-- The hypotheses are satisfiable: one particle resting at the origin. -/
example : IsFrankWolfeStep (ContinuousLinearMap.id ℝ (EucSpace 1)) (1 / 2)
    (fun _ : Idx 1 => (0 : EucSpace 1)) (fun _ : Idx 1 => (0 : EucSpace 1)) := by
  refine fun i => ⟨0, ⟨subset_convexHull ℝ _ ⟨i, rfl⟩, ?_⟩, by simp⟩
  intro z _
  simp

/-- **Theorem (thm: fw.cluster) — Frank-Wolfe convergence to a cluster.**

Suppose `B_*^t - B_*^{t+1} ≽ 0` and `B_*^t ≽ 0` for all `t ≥ 0`.  Fix
`γ^t = 2/(t+2)` and suppose `0 ∈ 𝒦^0`.  Then particles evolving according to
`eq: hardmax.dynamics.V_fw` satisfy

  `J^t(x_i^{t+1}) ≤ (2/(t+1)) · λ_max(B_*^0) · 𝖽(𝒦^0)^2`.

**What the source says and what is changed here.**  `λ_max(B_*^0)` is carried
as any `lam` with `⟨B_*^0 y, y⟩ ≤ lam ‖y‖²` for all `y`.  The largest
eigenvalue is one such `lam` — it is the least one — so quantifying over all of
them neither weakens nor strengthens the claim: the paper's instance is the
instance `lam = λ_max(B_*^0)`, and every other `lam` satisfying the hypothesis
is larger, hence gives a weaker conclusion that follows from it.

The step size is the theorem's own, `γ^t = 2/(t+2)`; the paragraph after the
theorem says instead "we fix `γ^t = 2/(t+1)`", but the statement and the proof
of `sec: bach.proof` both run on `2/(t+2)` — the proof's recursion
`a^{t+1} = (t+2)J^{t+1} ≤ (t+2)(1 - 2/(t+2))a^t/(t+1) + 4C` uses it — so the
paragraph is the odd one out.

Not proved here.

Source: arXiv:2508.09628v1, §3, `thm: fw.cluster`. -/
theorem fw_cluster (Bs : ℕ → ParamMatrix d) (γ : ℕ → ℝ)
    (x : ℕ → Idx n → EucSpace d) (lam : ℝ)
    (hpos : ∀ t : ℕ, ContinuousLinearMap.IsPositive (Bs t))
    (hmono : ∀ t : ℕ, ContinuousLinearMap.IsPositive (Bs t - Bs (t + 1)))
    (hγ : ∀ t : ℕ, γ t = 2 / ((t : ℝ) + 2))
    (h0 : (0 : EucSpace d) ∈ configHull (x 0))
    (hlam : ∀ y : EucSpace d, inner (𝕜 := ℝ) (Bs 0 y) y ≤ lam * ‖y‖ ^ 2)
    (hflow : ∀ t : ℕ, IsFrankWolfeStep (Bs t) (γ t) (x t) (x (t + 1))) :
    ∀ (t : ℕ) (i : Idx n),
      quadForm (Bs t) (x (t + 1) i)
        ≤ 2 / ((t : ℝ) + 1) * lam * Metric.diam (configHull (x 0)) ^ 2 := by
  sorry

/-- The hypotheses of `fw_cluster` are satisfiable: one particle resting at
the origin, `B_*^t ≡ 0`, `lam = 0`. -/
example : (∀ t : ℕ, ContinuousLinearMap.IsPositive ((fun _ : ℕ => (0 : ParamMatrix 1)) t)) ∧
    (∀ t : ℕ, ContinuousLinearMap.IsPositive
      ((fun _ : ℕ => (0 : ParamMatrix 1)) t - (fun _ : ℕ => (0 : ParamMatrix 1)) (t + 1))) ∧
    (∀ t : ℕ, (fun s : ℕ => 2 / ((s : ℝ) + 2)) t = 2 / ((t : ℝ) + 2)) ∧
    (0 : EucSpace 1) ∈ configHull ((fun _ : ℕ => fun _ : Idx 1 => (0 : EucSpace 1)) 0) ∧
    (∀ y : EucSpace 1, inner (𝕜 := ℝ) ((0 : ParamMatrix 1) y) y ≤ (0 : ℝ) * ‖y‖ ^ 2) ∧
    (∀ t : ℕ, IsFrankWolfeStep ((fun _ : ℕ => (0 : ParamMatrix 1)) t)
      ((fun s : ℕ => 2 / ((s : ℝ) + 2)) t)
      ((fun _ : ℕ => fun _ : Idx 1 => (0 : EucSpace 1)) t)
      ((fun _ : ℕ => fun _ : Idx 1 => (0 : EucSpace 1)) (t + 1))) := by
  refine ⟨fun t => ContinuousLinearMap.isPositive_zero, fun t => by
    simp, fun t => rfl,
    subset_convexHull ℝ _ ⟨0, rfl⟩, fun y => by simp, fun t i => ⟨0, ⟨?_, ?_⟩, by simp⟩⟩
  · exact subset_convexHull ℝ _ ⟨i, rfl⟩
  · intro z _
    simp

end FrankWolfe
end Transformer
