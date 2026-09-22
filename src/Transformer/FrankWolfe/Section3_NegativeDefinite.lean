/-
# Attention's forward pass and Frank-Wolfe — negative-definite key-query

Formalization of Alcalde, Geshkovski, Ruiz-Balet — arXiv:2508.09628v1,
*Attention's forward pass and Frank-Wolfe*, §3.

Reparametrizing `B^t = -B_*^t` turns `(SA_∞)` into the Frank-Wolfe iteration
`eq: hardmax.dynamics.V_fw` for the convex quadratic `J^t(x) = ½ ⟨B_*^t x, x⟩`,
for which the source claims the standard Frank-Wolfe rate, `thm: fw.cluster`.
That claim is false as stated; it is refuted in `Section3_FWClusterFalse.lean`.
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

end FrankWolfe
end Transformer
