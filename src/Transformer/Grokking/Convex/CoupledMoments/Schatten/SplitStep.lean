module

public import Mathlib.Algebra.Module.LinearMap.Defs
public import Mathlib.Basic.Real.Basic

/-!
# A linear isometry takes splitting steps to splitting steps

Source: `review/convex/moments.md`, CM121. One step of three-operator splitting takes two nearest
points and one gradient: `x = P_A z`, `y = P_B (2 x - z - γ g x)`, `z' = z + y - x`
(`SplitStep`). A linear map that keeps the distance, maps both sets onto those of a second
problem and carries the first gradient to the second takes a step to a step (`SplitStep.map`).
Nearest points are a relation here, so this needs neither their existence nor their uniqueness.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.CoupledMoments.Schatten

variable {E E' : Type*}

/-- `x` is a point of `K` nearest to `z` for the cost `d`. -/
def IsNearest (d : E → E → ℝ) (K : Set E) (z x : E) : Prop :=
  x ∈ K ∧ ∀ w ∈ K, d z x ≤ d z w

/-- A map that keeps the cost sends a nearest point of `K` to a nearest point of its image. -/
theorem IsNearest.map {d : E → E → ℝ} {d' : E' → E' → ℝ} {K : Set E} {z x : E} (T : E → E')
    (hT : ∀ a b, d' (T a) (T b) = d a b) (h : IsNearest d K z x) :
    IsNearest d' (T '' K) (T z) (T x) := by
  refine ⟨⟨x, h.1, rfl⟩, ?_⟩
  rintro _ ⟨w, hw, rfl⟩
  rw [hT, hT]
  exact h.2 w hw

example : IsNearest (fun a b : ℝ => (a - b) ^ 2) (id '' {0}) (id 1) (id 0) :=
  IsNearest.map (d := fun a b : ℝ => (a - b) ^ 2) id (fun _ _ => rfl)
    ⟨rfl, fun w hw => by rw [Set.mem_singleton_iff.1 hw]⟩

variable [AddCommGroup E] [Module ℝ E] [AddCommGroup E'] [Module ℝ E']

/-- One step of the three-operator splitting with nearest points as projections:
`x = P_A z`, `y = P_B (2 x - z - γ g x)`, `z' = z + y - x`. The gradient `g` reads `x`. -/
def SplitStep (d : E → E → ℝ) (A B : Set E) (γ : ℝ) (g : E → E) (z x z' : E) : Prop :=
  IsNearest d A z x ∧ ∃ y, IsNearest d B ((2 : ℝ) • x - z - γ • g x) y ∧ z' = z + y - x

/-- **A linear map that keeps the distance, both sets and the gradient takes a step to a
step.** -/
theorem SplitStep.map {d : E → E → ℝ} {d' : E' → E' → ℝ} {A B : Set E} {A' B' : Set E'} {γ : ℝ}
    {g : E → E} {g' : E' → E'} {z x z' : E} (T : E →ₗ[ℝ] E')
    (hT : ∀ a b, d' (T a) (T b) = d a b) (hA : T '' A = A') (hB : T '' B = B')
    (hg : ∀ w, g' (T w) = T (g w)) (h : SplitStep d A B γ g z x z') :
    SplitStep d' A' B' γ g' (T z) (T x) (T z') := by
  subst hA hB
  obtain ⟨hx, y, hy, rfl⟩ := h
  refine ⟨hx.map T hT, T y, ?_, by simp only [map_sub, map_add]⟩
  simpa only [map_sub, map_smul, hg] using hy.map T hT

example : SplitStep (fun a b : ℝ => (a - b) ^ 2) {0} {0} 0 id 0 0 0 := by
  have h : SplitStep (fun a b : ℝ => (a - b) ^ 2) {0} {0} 0 id 0 0 0 :=
    ⟨⟨rfl, fun w hw => by rw [Set.mem_singleton_iff.1 hw]⟩, 0,
      ⟨by simp, fun w hw => by simp [Set.mem_singleton_iff.1 hw]⟩, by simp⟩
  simpa using SplitStep.map (A' := {0}) (B' := {0}) (g' := id) (LinearMap.id : ℝ →ₗ[ℝ] ℝ)
    (fun _ _ => rfl) (by simp) (by simp) (fun _ => rfl) h

end Transformer.Grokking.Convex.CoupledMoments.Schatten
