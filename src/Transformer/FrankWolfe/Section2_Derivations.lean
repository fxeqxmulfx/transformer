/-
# Attention's forward pass and Frank-Wolfe — derivations

Formalization of Alcalde, Geshkovski, Ruiz-Balet — arXiv:2508.09628v1,
*Attention's forward pass and Frank-Wolfe*, §2.

Two facts about the hardmax dynamics before any spectral assumption on the
key-query matrix:

* `lem:singleLeader` — for almost every initial configuration the leader set
  `𝒞_i^t` is a singleton at every step, so `eq: hardmax.dynamics.V` really is
  the `argmax` iteration `eq: hardmax.dynamics.V_ae`;
* `lem:convHullDecreases` — the convex hull of the configuration shrinks.
-/

import Transformer.FrankWolfe.Section1_Models
import Mathlib.MeasureTheory.Constructions.Pi
import Mathlib.MeasureTheory.Measure.Haar.InnerProductSpace

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace FrankWolfe

variable {d n : ℕ}

/-- A trajectory of `eq: hardmax.dynamics.V`.  Unlike `(SA_∞)` this is a
genuine function of the configuration — the average over the leader set is
defined whether or not that set is a singleton — so the trajectory is
determined by its initial datum. -/
def AverageFlow (P B : ℕ → ParamMatrix d) (x : ℕ → Idx n → EucSpace d) : Prop :=
  ∀ (t : ℕ) (i : Idx n), x (t + 1) i = hardmaxAverageStep (P t) (B t) (x t) i

/-- **Lemma (lem:singleLeader).**  Suppose `B^t` is invertible for all `t ≥ 0`.
Then for almost every initial configuration `(x_i^0) ∈ (ℝ^d)^n`,

  `#𝒞_i^t = 1`

for all `t ≥ 0` and `i ∈ [1, n]`.

The paper's proof: the leaders coincide only on the finite union of the
hyperplanes `H_ij^t = {x : ⟨B^t x, v_i^t - v_j^t⟩ = 0}`, which is Lebesgue-null
since `B^t` is invertible, and the step map is piecewise affine, so it does not
push a set of positive measure into a null set.  That last step is not right:
at `V^t = I_d`, so `P = I_d/2`, and `B^t = -I_1`, two tokens `x_1 > 0 > x_2`
lead each other and both land on `(x_1 + x_2)/2`, so the step maps an open set
onto the diagonal.  The statement survives this because `𝒞_i^t` is a set of
points (`leaderSet`), but the proof does not.

`P^t` is pinned by `IsPreconditioner` to the source's `(I_d + V^t)^{-1} V^t`.

Not proved here.

Source: arXiv:2508.09628v1, §2.1, `lem:singleLeader`. -/
theorem singleLeader (V P B : ℕ → ParamMatrix d) (hP : ∀ t : ℕ, IsPreconditioner (V t) (P t))
    (hB : ∀ t : ℕ, Function.Bijective (B t)) :
    ∀ᵐ X₀ : Idx n → EucSpace d,
      ∀ x : ℕ → Idx n → EucSpace d, x 0 = X₀ → AverageFlow P B x →
        ∀ (t : ℕ) (i : Idx n), (leaderSet (B t) (x t) i).card = 1 := by
  sorry

/-- The hypotheses of `singleLeader` are satisfiable: `V = 0`, `P = 0`, and the
identity is invertible. -/
example (d : ℕ) : IsPreconditioner (0 : ParamMatrix d) 0 ∧
    Function.Bijective (ContinuousLinearMap.id ℝ (EucSpace d)) :=
  ⟨fun x => by simp, Function.bijective_id⟩

/-- **Lemma (lem:convHullDecreases).**  If `γ^t ∈ (0, 1)` then one step of
`(SA_∞)` shrinks the convex hull of the configuration:

  `𝒦^{t+1} ⊆ 𝒦^t`.

Source: arXiv:2508.09628v1, §2.3, `lem:convHullDecreases`. -/
theorem configHull_subset_of_isHardmaxStep {B : ParamMatrix d} {γ : ℝ}
    {X Y : Idx n → EucSpace d} (hγ : γ ∈ Set.Ioo (0 : ℝ) 1)
    (hstep : IsHardmaxStep B γ X Y) :
    configHull Y ⊆ configHull X := by
  refine convexHull_min ?_ (convex_convexHull ℝ _)
  rintro _ ⟨i, rfl⟩
  obtain ⟨y, ⟨hyK, -⟩, hYi⟩ := hstep i
  have hXi : X i ∈ configHull X := subset_convexHull ℝ _ ⟨i, rfl⟩
  have hcomb : X i + γ • (y - X i) = (1 - γ) • X i + γ • y := by module
  rw [hYi, hcomb]
  exact (convex_convexHull ℝ _) hXi hyK (by linarith [hγ.2]) hγ.1.le (by ring)

/-- The hypotheses of `configHull_subset_of_isHardmaxStep` are satisfiable: one
particle at the origin, which stays there, with `γ = 1/2`. -/
example : (1 / 2 : ℝ) ∈ Set.Ioo (0 : ℝ) 1 ∧
    IsHardmaxStep (ContinuousLinearMap.id ℝ (EucSpace 1)) (1 / 2)
      (fun _ : Idx 1 => (0 : EucSpace 1)) (fun _ : Idx 1 => (0 : EucSpace 1)) := by
  refine ⟨by norm_num, fun i => ⟨0, ⟨?_, ?_⟩, by simp⟩⟩
  · exact subset_convexHull ℝ _ ⟨i, rfl⟩
  · intro z _
    simp

/-- **Lemma (lem:convHullDecreases), the trajectory form.**  Along `(SA_∞)`
with `γ^t ∈ (0, 1)` the map `t ↦ 𝒦^t` is decreasing.

Source: arXiv:2508.09628v1, §2.3, `lem:convHullDecreases`. -/
theorem configHull_antitone {B : ℕ → ParamMatrix d} {γ : ℕ → ℝ}
    {x : ℕ → Idx n → EucSpace d} (hγ : ∀ t, γ t ∈ Set.Ioo (0 : ℝ) 1)
    (hflow : HardmaxFlow B γ x) {s t : ℕ} (hst : s ≤ t) :
    configHull (x t) ⊆ configHull (x s) := by
  induction t with
  | zero => simpa using (Nat.le_zero.mp hst) ▸ subset_rfl
  | succ t ih =>
    rcases Nat.lt_or_ge s (t + 1) with h | h
    · exact (configHull_subset_of_isHardmaxStep (hγ t) (hflow t)).trans
        (ih (Nat.lt_succ_iff.mp h))
    · have : s = t + 1 := le_antisymm hst h
      exact this ▸ subset_rfl

/-- The hypotheses of `configHull_antitone` are satisfiable: one particle
resting at the origin, `γ^t ≡ 1/2`. -/
example : (∀ t : ℕ, (fun _ : ℕ => (1 / 2 : ℝ)) t ∈ Set.Ioo (0 : ℝ) 1) ∧
    HardmaxFlow (fun _ : ℕ => ContinuousLinearMap.id ℝ (EucSpace 1))
      (fun _ : ℕ => (1 / 2 : ℝ)) (fun _ : ℕ => fun _ : Idx 1 => (0 : EucSpace 1)) := by
  refine ⟨fun t => by norm_num, fun t i => ⟨0, ⟨?_, ?_⟩, by simp⟩⟩
  · exact subset_convexHull ℝ _ ⟨i, rfl⟩
  · intro z _
    simp

end FrankWolfe
end Transformer
