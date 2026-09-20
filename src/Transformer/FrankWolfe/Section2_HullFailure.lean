/-
# Attention's forward pass and Frank-Wolfe — the hull need not shrink

Formalization of Alcalde, Geshkovski, Ruiz-Balet — arXiv:2508.09628v1,
*Attention's forward pass and Frank-Wolfe*, §2, the remark following
`lem:convHullDecreases`.

`configHull_subset_of_isHardmaxStep` holds because the step is a convex
combination of `x_i^t` and a point of `𝒦^t`.  With a general positive
diagonal value matrix `V^t` each *coordinate* of `x_i^{t+1}` is still a convex
combination of coordinates of points of `𝒦^t`, and that is not enough: the
source exhibits a configuration leaving its own hull, and it is proved here.
-/

import Transformer.FrankWolfe.Section2_Derivations

open scoped BigOperators

namespace Transformer
namespace FrankWolfe

/-- The diagonal map `diag(a, b)` on `ℝ²`. -/
noncomputable def diagTwo (a b : ℝ) : ParamMatrix 2 :=
  LinearMap.toContinuousLinearMap
    { toFun := fun x => EuclideanSpace.single 0 (a * x 0) + EuclideanSpace.single 1 (b * x 1)
      map_add' := by
        intro x y
        ext i
        fin_cases i <;> simp <;> ring
      map_smul' := by
        intro c x
        ext i
        fin_cases i <;> simp <;> ring }

@[simp]
theorem diagTwo_apply (a b : ℝ) (x : EucSpace 2) (i : Fin 2) :
    (diagTwo a b) x i = if i = 0 then a * x 0 else b * x 1 := by
  fin_cases i <;> simp [diagTwo]

/-- The unit triangle `(0, e₁, e₂)` of the source's remark. -/
noncomputable def triangle : Idx 3 → EucSpace 2 :=
  ![0, EuclideanSpace.single 0 1, EuclideanSpace.single 1 1]

theorem triangle_subset : configHull triangle ⊆ {z : EucSpace 2 | z 0 + z 1 ≤ 1} := by
  refine convexHull_min ?_ ?_
  · rintro _ ⟨j, rfl⟩
    fin_cases j <;> simp [triangle]
  · have hlin : IsLinearMap ℝ (fun z : EucSpace 2 => z 0 + z 1) :=
      ⟨fun x y => by simp; ring, fun c x => by simp; ring⟩
    exact convex_halfSpace_le hlin 1

theorem triangle_nonneg : configHull triangle ⊆ {z : EucSpace 2 | 0 ≤ z 0} := by
  refine convexHull_min ?_ ?_
  · rintro _ ⟨j, rfl⟩
    fin_cases j <;> simp [triangle]
  · have hlin : IsLinearMap ℝ (fun z : EucSpace 2 => -z 0) :=
      ⟨fun x y => by simp; ring, fun c x => by simp⟩
    simpa [neg_nonpos] using convex_halfSpace_le hlin 0

theorem inner_single_one (i : Fin 2) (z : EucSpace 2) :
    inner (𝕜 := ℝ) (EuclideanSpace.single i (1 : ℝ)) z = z i := by
  simp [PiLp.inner_apply, PiLp.single_apply]

/-- **The remark after `lem:convHullDecreases`, proved.**

Take `V = diag(3/2, 7/3)`, so that `P = (I₂ + V)⁻¹V = diag(0.6, 0.7)`, and the
unit triangle `(x_1, x_2, x_3) = (0, e₁, e₂)` in `ℝ²`.  With a key-query matrix
for which `argmax_{y ∈ 𝒦} ⟨B x_2, y⟩ = x_3` — the functional `y ↦ ⟨e₂, y⟩`
attains its maximum over the triangle at `e₂` and nowhere else — the
preconditioned hardmax step sends `x_2` to

  `x_2 + P(x_3 - x_2) = (0.4, 0.7) ∉ 𝒦`.

So `configHull_subset_of_isHardmaxStep` genuinely needs the step to be a
convex combination, and fails for the renormalized layer
`eq: rescaled.Tformers` with a general positive diagonal `V`.

**What the source says and what is changed here.**  The source names `P`
directly; here `V` is exhibited too, together with `IsPreconditioner V P`, so
that `P` is a preconditioner of an admissible value matrix and not merely a
diagonal matrix with entries in `(0,1)`.  The key-query matrix `B` is replaced
by the vector `a = B x_2` it enters through, and the maximizer is shown to be
unique, so no choice of `argmax` avoids the conclusion.

Source: arXiv:2508.09628v1, §2, the remark following `lem:convHullDecreases`. -/
theorem not_configHull_subset_of_preconditioner :
    ∃ (V P : ParamMatrix 2) (a y : EucSpace 2),
      IsPreconditioner V P ∧
      IsMaximizerOn a (configHull triangle) y ∧
      (∀ z, IsMaximizerOn a (configHull triangle) z → z = y) ∧
      triangle 1 + P (y - triangle 1) ∉ configHull triangle := by
  refine ⟨diagTwo (3 / 2) (7 / 3), diagTwo 0.6 0.7, EuclideanSpace.single 1 1,
    EuclideanSpace.single 1 1, ?_, ⟨?_, ?_⟩, ?_, ?_⟩
  · intro x
    ext i
    fin_cases i <;> simp <;> ring
  · exact subset_convexHull ℝ _ ⟨2, by simp [triangle]⟩
  · intro z hz
    rw [inner_single_one, inner_single_one]
    have h1 := triangle_subset hz
    have h2 := triangle_nonneg hz
    simp only [Set.mem_ofPred_eq] at h1 h2
    simp only [PiLp.single_apply]
    norm_num
    linarith
  · intro z hz
    have hy : inner (𝕜 := ℝ) (EuclideanSpace.single (1 : Fin 2) (1 : ℝ))
        (EuclideanSpace.single (1 : Fin 2) (1 : ℝ)) ≤
        inner (𝕜 := ℝ) (EuclideanSpace.single (1 : Fin 2) (1 : ℝ)) z :=
      hz.2 _ (subset_convexHull ℝ _ ⟨2, by simp [triangle]⟩)
    rw [inner_single_one, inner_single_one] at hy
    have h1 := triangle_subset hz.1
    have h2 := triangle_nonneg hz.1
    simp only [Set.mem_ofPred_eq] at h1 h2
    simp only [PiLp.single_apply] at hy
    ext i
    fin_cases i <;> simp only [PiLp.single_apply] <;> norm_num at hy ⊢ <;> linarith
  · intro hmem
    have h1 := triangle_subset hmem
    simp only [Set.mem_ofPred_eq, PiLp.add_apply, PiLp.sub_apply, diagTwo_apply,
      triangle, PiLp.single_apply] at h1
    norm_num at h1

end FrankWolfe
end Transformer
