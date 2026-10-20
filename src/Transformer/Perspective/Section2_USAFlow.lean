module

public import Transformer.Perspective.Section2_USAField
public import Transformer.Perspective.Section2_SphereFlow

/-!
# Global existence and uniqueness of the original unnormalized attention ODE

Sources: arXiv:2312.10794v5, §2, `USA`, and arXiv:2512.01868v4, §4,
`thm:clustering_finite`. The smooth ambient field has linear growth.
Its norm defect solves a scalar homogeneous ODE, so unit initial vectors
stay unit for every real time. On that invariant state space the two
auxiliary damping factors are exactly one and the ODE is the original USA.
No global solution, sphere constraint along the path, or convergence
claim is included in the definition of the extension.
-/

@[expose] public section

open scoped BigOperators
open Real

namespace Transformer.Perspective

variable {d n : ℕ}

/-- The norm defect vanishes along the actual ambient ODE, by uniqueness
for a homogeneous scalar equation. Source: arXiv:2312.10794v5, §2,
the tangent projection in `USA`. -/
theorem norm_eq_one_of_usaGlobalField (β : ℝ) (Y : ℝ → Idx n → EucSpace d)
    (hY : ∀ t : ℝ, HasDerivAt Y (usaGlobalField β (Y t)) t)
    (h0 : ∀ i : Idx n, ‖Y 0 i‖ = 1) (t : ℝ) (i : Idx n) : ‖Y t i‖ = 1 := by
  have hc : Continuous Y := continuous_iff_continuousAt.mpr fun s => (hY s).continuousAt
  let v (s : ℝ) := usaExtensionAverage β (Y s) i
  let a (s : ℝ) := -2 * saExtensionScale (Y s) * inner (𝕜 := ℝ) (Y s i) (v s)
  have hv : Continuous v := (contDiff_usaExtensionAverage β i).continuous.comp hc
  have ha : Continuous a :=
    (continuous_const.mul (contDiff_saExtensionScale.continuous.comp hc)).mul
      (((continuous_apply i).comp hc).inner hv)
  have hdefect (s : ℝ) : HasDerivAt (fun u => ‖Y u i‖ ^ 2 - 1)
      (a s * (‖Y s i‖ ^ 2 - 1)) s := by
    convert ((hasDerivAt_pi.mp (hY s)) i).norm_sq.sub_const 1 using 1
    simp only [usaGlobalField, Pi.smul_apply, proj, real_inner_smul_right,
      inner_sub_right, real_inner_self_eq_norm_sq]
    dsimp [a, v]
    ring
  have hz := scalar_eq_zero_of_linearODE a (fun s => ‖Y s i‖ ^ 2 - 1) ha hdefect
    (by rw [h0 i]; norm_num) t
  nlinarith [norm_nonneg (Y t i)]

/-- Unit consensus gives a complete witness for the norm-invariance
hypotheses. Source: arXiv:2312.10794v5, §2, the one-token equilibrium. -/
example (β : ℝ) :
    (∀ t : ℝ, HasDerivAt (fun _ => fun _ : Idx 1 => (basePoint 0 : EucSpace 1))
      (usaGlobalField β (fun _ : Idx 1 => (basePoint 0 : EucSpace 1))) t) ∧
      (∀ i : Idx 1, ‖(fun _ : Idx 1 => (basePoint 0 : EucSpace 1)) i‖ = 1) := by
  have hu : ∀ i : Idx 1, ‖(fun _ : Idx 1 => (basePoint 0 : EucSpace 1)) i‖ = 1 :=
    fun i => by simp [basePoint, PiLp.norm_single]
  have hf : usaGlobalField β (fun _ : Idx 1 => (basePoint 0 : EucSpace 1)) = 0 := by
    funext i
    rw [usaGlobalField_apply β _ hu i]
    simp only [Fin.sum_univ_one, Nat.cast_one, inv_one, one_smul]
    exact proj_smul_self (hu 0) _
  exact ⟨fun t => by rw [hf]; exact hasDerivAt_const t _, hu⟩

/-- Every spherical initial tuple has an actual global `USA` trajectory,
for every real temperature and both positive and negative times.
The proof constructs it from the ambient field and proves invariance.
Sources: arXiv:2312.10794v5, §2; arXiv:2512.01868v4, §4. -/
theorem exists_USA (β : ℝ) (X₀ : SphereTuple d n) :
    ∃ X : ℝ → SphereTuple d n, X 0 = X₀ ∧ USA d n β X := by
  have hF : LocallyLipschitz (usaGlobalField (d := d) (n := n) β) :=
    ((contDiff_usaGlobalField β).of_le (show (1 : WithTop ℕ∞) ≤ (⊤ : ℕ∞) by simp)).locallyLipschitz
  obtain ⟨Y, hY0, hY⟩ := Landscape.GlobalFlow.exists_global hF
    (by positivity) (norm_usaGlobalField_le β) (fun i => (X₀ i : EucSpace d))
  have hu (t : ℝ) (i : Idx n) : ‖Y t i‖ = 1 :=
    norm_eq_one_of_usaGlobalField β Y hY
      (fun j => by rw [hY0]; exact mem_sphere_zero_iff_norm.mp (X₀ j).2) t i
  let X (t : ℝ) (i : Idx n) : SSphere d := ⟨Y t i, mem_sphere_zero_iff_norm.mpr (hu t i)⟩
  refine ⟨X, ?_, ?_⟩
  · funext i
    exact Subtype.ext (congrFun hY0 i)
  · intro t i
    have h := (hasDerivAt_pi.mp (hY t)) i
    rw [usaGlobalField_apply β _ (hu t) i] at h
    exact h

/-- Existence is realized for two distinct tokens, not just for a
stationary consensus datum. Only their initial norms are supplied;
the construction proves the sphere constraint along the entire flow.
Sources: arXiv:2312.10794v5, §2; arXiv:2512.01868v4, §4. -/
example : ∃ X : ℝ → SphereTuple 2 2,
    X 0 0 = basePoint 1 ∧
      (X 0 1 : EucSpace 2) = EuclideanSpace.single 1 1 ∧ USA 2 2 0 X := by
  let e : SSphere 2 := ⟨EuclideanSpace.single 1 1, by
    rw [mem_sphere_zero_iff_norm]
    simp [PiLp.norm_single]⟩
  let X₀ : SphereTuple 2 2 := ![basePoint 1, e]
  obtain ⟨X, h0, hX⟩ := exists_USA 0 X₀
  refine ⟨X, ?_, ?_, hX⟩
  · rw [h0]
    rfl
  · rw [h0]
    rfl

/-- Every source `USA` solution also follows the smooth ambient extension.
Sources: arXiv:2312.10794v5, §2; arXiv:2512.01868v4, §4. -/
theorem hasDerivAt_usaGlobalField_of_USA (β : ℝ) (X : ℝ → SphereTuple d n)
    (hX : USA d n β X) (t : ℝ) :
    HasDerivAt (fun s i => (X s i : EucSpace d))
      (usaGlobalField β (fun i => (X t i : EucSpace d))) t := by
  apply hasDerivAt_pi.mpr
  intro i
  rw [usaGlobalField_apply β _ (fun j => mem_sphere_zero_iff_norm.mp (X t j).2) i]
  exact hX t i

/-- The source equation has an actual solution, so the preceding
extension hypothesis is realized, including the derivative and datum. -/
example (β : ℝ) (X₀ : SphereTuple d n) :
    ∃ X : ℝ → SphereTuple d n, X 0 = X₀ ∧ USA d n β X := exists_USA β X₀

/-- Uniqueness holds for the original `USA` equation, by smooth ambient
ODE uniqueness. Source: arXiv:2312.10794v5, §2, the Cauchy problem. -/
theorem USA_unique (β : ℝ) (X Y : ℝ → SphereTuple d n)
    (hX : USA d n β X) (hY : USA d n β Y) (h0 : X 0 = Y 0) : X = Y := by
  have hF : LocallyLipschitz (usaGlobalField (d := d) (n := n) β) :=
    ((contDiff_usaGlobalField β).of_le (show (1 : WithTop ℕ∞) ≤ (⊤ : ℕ∞) by simp)).locallyLipschitz
  have heq := Landscape.GlobalFlow.eq_of_hasDerivAt hF
    (hasDerivAt_usaGlobalField_of_USA β X hX) (hasDerivAt_usaGlobalField_of_USA β Y hY)
    (by rw [h0])
  funext t i
  exact Subtype.ext (congrFun (congrFun heq t) i)

/-- The hypotheses of uniqueness hold for the same actual trajectory.
Source: arXiv:2312.10794v5, §2, the original initial-value problem. -/
example (β : ℝ) (X₀ : SphereTuple d n) :
    ∃ X : ℝ → SphereTuple d n, USA d n β X ∧ USA d n β X ∧ X 0 = X 0 := by
  obtain ⟨X, _, hX⟩ := exists_USA β X₀
  exact ⟨X, hX, hX, rfl⟩

/-- Global existence and uniqueness for every actual spherical datum.
The initial-value problem itself has no probabilistic or clustering assumption.
Sources: arXiv:2312.10794v5, §2; arXiv:2512.01868v4, §4. -/
theorem exists_unique_USA (β : ℝ) (X₀ : SphereTuple d n) :
    ∃! X : ℝ → SphereTuple d n, X 0 = X₀ ∧ USA d n β X := by
  obtain ⟨X, h0, hX⟩ := exists_USA β X₀
  refine ⟨X, ⟨h0, hX⟩, ?_⟩
  intro Y hY
  exact USA_unique β Y X hY.2 hX (hY.1.trans h0.symm)

end Transformer.Perspective
