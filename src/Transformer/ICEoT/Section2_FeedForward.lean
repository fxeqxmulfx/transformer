/-
# IC-EoT: the feed-forward ICNN background

arXiv:2603.22095v2, §2.2.1, Eq. (1). Passthrough weights are unrestricted
for one-step convexity. Layer widths may vary; the initial state is zero.
No non-negativity assumption on the activation values is needed here.
-/

import Transformer.ICEoT.Section3_Closure

noncomputable section

namespace Transformer.ICEoT

/-- Convex monotone activations of §2.2.1–2.2.2, without the additional
non-negative-range condition required by the multiplicative architectures. -/
def ConvexMonotone (φ : ℝ → ℝ) : Prop := ConvexOn ℝ Set.univ φ ∧ Monotone φ

/-- Any fixed affine map preserves convex combinations, regardless of its
weight signs; §2.2.1, Eq. (1), passthrough branch. -/
theorem affine_combination {I O : Type*} [Fintype I]
    (W : O → I → ℝ) (bias : O → ℝ) (x y : I → ℝ) (a b : ℝ) (hab : a + b = 1) :
    affine W bias (a • x + b • y) = a • affine W bias x + b • affine W bias y := by
  funext o
  simp only [affine, Pi.add_apply, Pi.smul_apply, smul_eq_mul]
  have hc : a * bias o + b * bias o = bias o := by rw [← add_mul, hab, one_mul]
  have ht : ∀ i, W o i * (a * x i + b * y i) =
      a * (W o i * x i) + b * (W o i * y i) := by intro i; ring
  simp_rw [ht]
  rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum]
  linarith

example : (1 / 3 : ℝ) + 2 / 3 = 1 := by norm_num

/-- The unrestricted passthrough map is convex; §2.2.1, Eq. (1). -/
theorem unrestrictedAffine_convex {I O : Type*} [Fintype I]
    (W : O → I → ℝ) (bias : O → ℝ) : ComponentwiseConvex (affine W bias) := by
  refine ⟨convex_univ, ?_⟩
  intro x hx y hy a b ha hb hab
  exact le_of_eq (affine_combination W bias x y a b hab)

/-- Scalar activation composition without a non-negative-range assumption;
§2.2.1, Eq. (1), and §2.2.2, Eqs. (2), (3). -/
theorem convexMonotone_activation {E O : Type*} [AddCommGroup E] [Module ℝ E]
    (φ : ℝ → ℝ) (f : E → O → ℝ) (hφ : ConvexMonotone φ)
    (hf : ComponentwiseConvex f) : ComponentwiseConvex (fun x o => φ (f x o)) := by
  refine ⟨convex_univ, ?_⟩
  intro x hx y hy a b ha hb hab o
  exact (hφ.2 (hf.2 hx hy ha hb hab o)).trans
    (hφ.1.2 (Set.mem_univ _) (Set.mem_univ _) ha hb hab)

example : ConvexMonotone relu ∧ ComponentwiseConvex (id : (Fin 1 → ℝ) → Fin 1 → ℝ) :=
  ⟨⟨relu_conditions.1, relu_conditions.2.1⟩, convexOn_id convex_univ⟩

/-- A possibly rectangular ICNN layer, Eq. (1), §2.2.1. -/
structure FCLayer (din hin hout : ℕ) where
  hidden : Fin hout → Fin hin → ℝ
  passthrough : Fin hout → Fin din → ℝ
  bias : Fin hout → ℝ
  activation : ℝ → ℝ

/-- A hidden-positive layer with negative passthrough weights, witnessing
the unrestricted branch of §2.2.1, Eq. (1). -/
def witnessFCLayer : FCLayer 1 1 1 where
  hidden := fun _ _ => 1
  passthrough := fun _ _ => -1
  bias := fun _ => 0
  activation := relu

/-- A layer meeting the stricter recursive-prediction condition; §2.2.1. -/
def monotoneFCLayer : FCLayer 1 1 1 :=
  { witnessFCLayer with passthrough := fun _ _ => 1 }

/-- The actual Eq. (1) recurrence with arbitrary layer widths; §2.2.1. -/
def fnnRun {din : ℕ} (width : ℕ → ℕ)
    (p : (k : ℕ) → FCLayer din (width k) (width (k + 1))) :
    (k : ℕ) → (Fin din → ℝ) → Fin (width k) → ℝ
  | 0, _ => 0
  | k + 1, x =>
    fun r => (p k).activation
      (affine (p k).hidden (fun _ => 0) (fnnRun width p k x) r +
        affine (p k).passthrough (p k).bias x r)

/-- The background ICNN convexity claim, with unrestricted passthrough
weights and only non-negative hidden weights; §2.2.1, Eq. (1).
The source also specifies `W_0^(z)=0`; since `z_0=0`, that redundant
initial weight restriction is unnecessary for this stronger result. -/
theorem fnnRun_convex {din : ℕ} (width : ℕ → ℕ)
    (p : (k : ℕ) → FCLayer din (width k) (width (k + 1))) (L : ℕ)
    (hp : ∀ k < L, Nonnegative (p k).hidden ∧ ConvexMonotone (p k).activation) :
    ComponentwiseConvex (fnnRun width p L) := by
  induction L with
  | zero => exact convexOn_const _ convex_univ
  | succ L ih =>
    have hs := (affine_convex (p L).hidden (fun _ => 0) _ (hp L (by omega)).1
      (ih (fun k hk => hp k (by omega)))).add
        (unrestrictedAffine_convex (p L).passthrough (p L).bias)
    exact convexMonotone_activation _ _ (hp L (by omega)).2 hs

example : ∀ k : ℕ, k < 3 →
    Nonnegative ((fun _ : ℕ => witnessFCLayer) k).hidden ∧
      ConvexMonotone ((fun _ : ℕ => witnessFCLayer) k).activation :=
  fun _ _ => ⟨fun _ _ => zero_le_one, relu_conditions.1, relu_conditions.2.1⟩

/-- Requiring non-negative passthrough weights also gives monotonicity,
the recursive-prediction condition discussed in §2.2.1. -/
theorem fnnRun_monotone {din : ℕ} (width : ℕ → ℕ)
    (p : (k : ℕ) → FCLayer din (width k) (width (k + 1))) (L : ℕ)
    (hp : ∀ k < L, Nonnegative (p k).hidden ∧ Nonnegative (p k).passthrough ∧
      Monotone (p k).activation) : Monotone (fnnRun width p L) := by
  induction L with
  | zero => exact monotone_const
  | succ L ih =>
    intro x y h r
    apply (hp L (by omega)).2.2
    exact add_le_add
      (affine_monotone (p L).hidden (fun _ => 0) _ (hp L (by omega)).1
        (ih (fun k hk => hp k (by omega))) h r)
      (affine_monotone (p L).passthrough (p L).bias id (hp L (by omega)).2.1 monotone_id h r)

example : ∀ k : ℕ, k < 3 →
    Nonnegative ((fun _ : ℕ => monotoneFCLayer) k).hidden ∧
    Nonnegative ((fun _ : ℕ => monotoneFCLayer) k).passthrough ∧
    Monotone ((fun _ : ℕ => monotoneFCLayer) k).activation :=
  fun _ _ => ⟨fun _ _ => zero_le_one, fun _ _ => zero_le_one, relu_conditions.2.1⟩

end Transformer.ICEoT
