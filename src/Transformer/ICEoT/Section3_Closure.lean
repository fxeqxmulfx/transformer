/-
# IC-EoT: componentwise convexity and closure

Xu, Zhi, Jiang, arXiv:2603.22095v2, §3.3, Definition 1 and Lemma 1.
Functions into finite real arrays use the pointwise order. Thus vector-valued
`ConvexOn` is exactly convexity of every scalar component, not convexity of
the graph. Biases and positional encodings are fixed, with unrestricted signs.
-/

import Mathlib.Analysis.Convex.Function
import Mathlib.Tactic

noncomputable section

namespace Transformer.ICEoT

variable {E : Type*} [AddCommGroup E] [Module ℝ E]
variable {I O : Type*}

/-- Componentwise convexity on the whole input space; §3.3, Definition 1. -/
def ComponentwiseConvex (f : E → O → ℝ) : Prop := ConvexOn ℝ Set.univ f

/-- The non-negative, convex, non-decreasing scalar activations of
§3.3, Assumption 1. These are assumptions on an actual function. -/
def ActivationConditions (φ : ℝ → ℝ) : Prop :=
  ConvexOn ℝ Set.univ φ ∧ Monotone φ ∧ ∀ x, 0 ≤ φ x

/-- Non-negative weights, with output index first; §3.3, Assumption 1. -/
def Nonnegative (W : O → I → ℝ) : Prop := ∀ o i, 0 ≤ W o i

/-- The affine map in §3.3, Lemma 1, written as coordinate sums. -/
def affine [Fintype I] (W : O → I → ℝ) (b : O → ℝ) (x : I → ℝ) : O → ℝ :=
  fun o => (∑ i, W o i * x i) + b o

/-- Vector-valued convexity is equivalent to the scalar definition in
§3.3, Definition 1. -/
theorem componentwiseConvex_iff (f : E → O → ℝ) :
    ComponentwiseConvex f ↔ ∀ o, ConvexOn ℝ Set.univ (fun x => f x o) := by
  constructor
  · intro h o
    exact ⟨convex_univ, fun _ hx _ hy _ _ ha hb hab => h.2 hx hy ha hb hab o⟩
  · intro h
    exact ⟨convex_univ, fun _ hx _ hy _ _ ha hb hab o => (h o).2 hx hy ha hb hab⟩

/-- Coordinatewise addition preserves both properties; §3.3, Lemma 1
and the residual additions in Eqs. (23), (25). -/
theorem add_properties [Preorder E] (f g : E → O → ℝ)
    (hf : ComponentwiseConvex f ∧ Monotone f)
    (hg : ComponentwiseConvex g ∧ Monotone g) :
    ComponentwiseConvex (fun x => f x + g x) ∧ Monotone (fun x => f x + g x) :=
  ⟨hf.1.add hg.1, fun _ _ h => add_le_add (hf.2 h) (hg.2 h)⟩

example : ComponentwiseConvex (fun x : ℝ => fun _ : Fin 1 => x) ∧
    Monotone (fun x : ℝ => fun _ : Fin 1 => x) := by
  exact ⟨⟨convex_univ, fun _ _ _ _ _ _ _ _ _ _ => le_rfl⟩, fun _ _ h _ => h⟩

/-- Non-negative affine closure, convexity part of §3.3, Lemma 1. -/
theorem affine_convex [Fintype I] (W : O → I → ℝ) (b : O → ℝ)
    (f : E → I → ℝ) (hW : Nonnegative W) (hf : ComponentwiseConvex f) :
    ComponentwiseConvex (fun x => affine W b (f x)) := by
  refine ⟨convex_univ, ?_⟩
  intro x hx y hy a c ha hc hac o
  have hsum := Finset.sum_le_sum (fun i (_ : i ∈ Finset.univ) =>
    mul_le_mul_of_nonneg_left (hf.2 hx hy ha hc hac i) (hW o i))
  simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul] at hsum ⊢
  have hterm : ∀ i, W o i * (a * f x i + c * f y i) =
      a * (W o i * f x i) + c * (W o i * f y i) := by intro i; ring
  simp_rw [hterm] at hsum
  rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum] at hsum
  simp only [affine]
  have hbias : a * b o + c * b o = b o := by rw [← add_mul, hac, one_mul]
  nlinarith only [hsum, hbias]

example : Nonnegative (fun _ _ : Fin 1 => (1 : ℝ)) ∧
    ComponentwiseConvex (fun x : ℝ => fun _ : Fin 1 => x) := by
  exact ⟨fun _ _ => zero_le_one,
    ⟨convex_univ, fun _ _ _ _ _ _ _ _ _ _ => le_rfl⟩⟩

omit [AddCommGroup E] [Module ℝ E] in
/-- Non-negative affine closure, monotonicity part of §3.3, Lemma 1. -/
theorem affine_monotone [Fintype I] [Preorder E]
    (W : O → I → ℝ) (b : O → ℝ) (f : E → I → ℝ)
    (hW : Nonnegative W) (hf : Monotone f) :
    Monotone (fun x => affine W b (f x)) := by
  intro x y h o
  exact add_le_add (Finset.sum_le_sum fun i (_ : i ∈ Finset.univ) =>
    mul_le_mul_of_nonneg_left (hf h i) (hW o i)) le_rfl

example : Nonnegative (fun _ _ : Fin 1 => (1 : ℝ)) ∧
    Monotone (fun x : ℝ => fun _ : Fin 1 => x) :=
  ⟨fun _ _ => zero_le_one, fun _ _ h _ => h⟩

/-- Monotone convex composition, the second assertion of §3.3, Lemma 1.
The outer map can couple coordinates; it need not act diagonally. -/
theorem composition_convex {J : Type*} (f : E → I → ℝ)
    (g : (I → ℝ) → J → ℝ) (hf : ComponentwiseConvex f)
    (hg : ComponentwiseConvex g) (hm : Monotone g) :
    ComponentwiseConvex (fun x => g (f x)) := by
  refine ⟨convex_univ, ?_⟩
  intro x hx y hy a b ha hb hab
  exact (hm (hf.2 hx hy ha hb hab)).trans
    (hg.2 (Set.mem_univ _) (Set.mem_univ _) ha hb hab)

example : ComponentwiseConvex (fun x : ℝ => fun _ : Fin 1 => x) ∧
    ComponentwiseConvex (id : (Fin 1 → ℝ) → Fin 1 → ℝ) ∧
    Monotone (id : (Fin 1 → ℝ) → Fin 1 → ℝ) :=
  ⟨⟨convex_univ, fun _ _ _ _ _ _ _ _ _ _ => le_rfl⟩,
    convexOn_id convex_univ, monotone_id⟩

/-- The full composition conclusion of §3.3, Lemma 1: convexity and
monotonicity hold even when the outer map couples coordinates. -/
theorem composition_properties [Preorder E] {J : Type*} (f : E → I → ℝ)
    (g : (I → ℝ) → J → ℝ) (hf : ComponentwiseConvex f ∧ Monotone f)
    (hg : ComponentwiseConvex g ∧ Monotone g) :
    ComponentwiseConvex (fun x => g (f x)) ∧ Monotone (fun x => g (f x)) :=
  ⟨composition_convex f g hf.1 hg.1 hg.2, hg.2.comp hf.2⟩

example : (ComponentwiseConvex (id : (Fin 1 → ℝ) → Fin 1 → ℝ) ∧
    Monotone (id : (Fin 1 → ℝ) → Fin 1 → ℝ)) ∧
    (ComponentwiseConvex (id : (Fin 1 → ℝ) → Fin 1 → ℝ) ∧
    Monotone (id : (Fin 1 → ℝ) → Fin 1 → ℝ)) :=
  ⟨⟨convexOn_id convex_univ, monotone_id⟩, ⟨convexOn_id convex_univ, monotone_id⟩⟩

/-- Componentwise activation closure; §3.3, Lemma 1. -/
theorem activation_properties [Preorder E] (φ : ℝ → ℝ) (f : E → O → ℝ)
    (hφ : ActivationConditions φ) (hf : ComponentwiseConvex f ∧ Monotone f) :
    ComponentwiseConvex (fun x o => φ (f x o)) ∧
      Monotone (fun x o => φ (f x o)) := by
  constructor
  · refine ⟨convex_univ, ?_⟩
    intro x hx y hy a b ha hb hab o
    exact (hφ.2.1 (hf.1.2 hx hy ha hb hab o)).trans
      (hφ.1.2 (Set.mem_univ _) (Set.mem_univ _) ha hb hab)
  · intro x y h o
    exact hφ.2.1 (hf.2 h o)

/-- ReLU satisfies all activation hypotheses, including at its kink;
§3.2, Eq. (17), and §3.3, Assumption 1. -/
def relu (x : ℝ) : ℝ := max 0 x

/-- The implementation's activation meets Assumption 1; §3.3. -/
theorem relu_conditions : ActivationConditions relu := by
  refine ⟨(convexOn_const 0 convex_univ).sup (convexOn_id convex_univ), ?_, ?_⟩
  · intro x y h
    exact max_le_max le_rfl h
  · exact fun x => le_max_left 0 x

example : ActivationConditions relu ∧
    (ComponentwiseConvex (fun x : ℝ => fun _ : Fin 1 => x) ∧
      Monotone (fun x : ℝ => fun _ : Fin 1 => x)) :=
  ⟨relu_conditions, ⟨⟨convex_univ, fun _ _ _ _ _ _ _ _ _ _ => le_rfl⟩,
    fun _ _ h _ => h⟩⟩

end Transformer.ICEoT
