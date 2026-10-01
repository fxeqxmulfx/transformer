/-
# IC-EoT: convexity of the multi-head additive attention layer

arXiv:2603.22095v2, §3.3, Proposition 1. These proofs use the actual
query, key, shared latent, diagonal gate/value, averaging and projection
equations, rather than assuming the attention layer's desired properties.
-/

import Transformer.ICEoT.Section3_AttentionModel

noncomputable section

namespace Transformer.ICEoT

variable {E : Type*} [AddCommGroup E] [Module ℝ E] [Preorder E]
variable {n m h heads : ℕ}

/-- The latent channel is convex, monotone and non-negative, as used in
§3.3, Proposition 1, Eqs. (15)–(17). -/
theorem pairLatent_properties (p : Head m h) (F : E → Sequence n m)
    (hp : HeadConditions p) (hF : ComponentwiseConvex F ∧ Monotone F)
    (i j : Fin n) :
    (ComponentwiseConvex (fun x r => pairLatent p (F x) i j r) ∧
      Monotone (fun x r => pairLatent p (F x) i j r)) ∧
      ∀ x r, 0 ≤ pairLatent p (F x) i j r := by
  have hi : ComponentwiseConvex (fun x r => F x (i, r)) ∧
      Monotone (fun x r => F x (i, r)) :=
    ⟨⟨convex_univ, fun _ hx _ hy _ _ ha hb hab r => hF.1.2 hx hy ha hb hab (i, r)⟩,
      fun _ _ h r => hF.2 h (i, r)⟩
  have hj : ComponentwiseConvex (fun x r => F x (j, r)) ∧
      Monotone (fun x r => F x (j, r)) :=
    ⟨⟨convex_univ, fun _ hx _ hy _ _ ha hb hab r => hF.1.2 hx hy ha hb hab (j, r)⟩,
      fun _ _ h r => hF.2 h (j, r)⟩
  have hq := And.intro (affine_convex p.query (fun _ => 0) _ hp.1 hi.1)
    (affine_monotone p.query (fun _ => 0) _ hp.1 hi.2)
  have hk := And.intro (affine_convex p.key p.latentBias _ hp.2.1 hj.1)
    (affine_monotone p.key p.latentBias _ hp.2.1 hj.2)
  have hs := add_properties _ _ hq hk
  have hz := activation_properties p.latentActivation _ hp.2.2.2.2.1 hs
  constructor
  · simpa [pairLatent, affine, add_assoc] using hz
  · intro x r
    exact hp.2.2.2.2.1.2.2 _

example : HeadConditions unitHead ∧
    (ComponentwiseConvex (id : Sequence 1 1 → Sequence 1 1) ∧
      Monotone (id : Sequence 1 1 → Sequence 1 1)) :=
  ⟨unitHead_conditions, convexOn_id convex_univ, monotone_id⟩

/-- Each source contribution satisfies the scalar product lemma;
§3.3, Proposition 1, Eqs. (18)–(20). -/
theorem pairContribution_properties (p : Head m h) (F : E → Sequence n m)
    (hp : HeadConditions p) (hF : ComponentwiseConvex F ∧ Monotone F)
    (i j : Fin n) :
    ComponentwiseConvex (fun x r => pairContribution p (F x) i j r) ∧
      Monotone (fun x r => pairContribution p (F x) i j r) := by
  have hz := pairLatent_properties p F hp hF i j
  have hr : ∀ r, ConvexOn ℝ Set.univ (fun x => pairContribution p (F x) i j r) ∧
      Monotone (fun x => pairContribution p (F x) i j r) := by
    intro r
    apply gateValue_composition p.gateActivation (p.gateScale r)
      (p.valueScale r) (p.gateBias r) _ hp.2.2.2.2.2 (hp.2.2.1 r) (hp.2.2.2.1 r)
    · exact ⟨(componentwiseConvex_iff _).mp hz.1.1 r, fun _ _ h => hz.1.2 h r⟩
    · exact fun x => hz.2 x r
  exact ⟨(componentwiseConvex_iff _).mpr (fun r => (hr r).1),
    fun _ _ h r => (hr r).2 h⟩

example : HeadConditions unitHead ∧
    (ComponentwiseConvex (id : Sequence 1 1 → Sequence 1 1) ∧
      Monotone (id : Sequence 1 1 → Sequence 1 1)) :=
  ⟨unitHead_conditions, convexOn_id convex_univ, monotone_id⟩

/-- Source averaging preserves convexity and monotonicity;
§3.3, Proposition 1, Eq. (21). -/
theorem headContext_properties (p : Head m h) (F : E → Sequence (n + 1) m)
    (hp : HeadConditions p) (hF : ComponentwiseConvex F ∧ Monotone F) :
    ComponentwiseConvex (fun x => headContext p (F x)) ∧
      Monotone (fun x => headContext p (F x)) := by
  have hc := fun i j => pairContribution_properties p F hp hF i j
  have hn : 0 ≤ (n + 1 : ℝ) := by positivity
  constructor
  · refine ⟨convex_univ, ?_⟩
    intro x hx y hy a b ha hb hab ir
    have hs := Finset.sum_le_sum fun j (_ : j ∈ Finset.univ) =>
      (hc ir.1 j).1.2 hx hy ha hb hab ir.2
    simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul] at hs ⊢
    simp only [headContext]
    rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum] at hs
    simpa [add_div, mul_div_assoc] using div_le_div_of_nonneg_right hs hn
  · intro x y h ir
    exact div_le_div_of_nonneg_right (Finset.sum_le_sum fun j _ =>
      (hc ir.1 j).2 h ir.2) hn

example : HeadConditions unitHead ∧
    (ComponentwiseConvex (id : Sequence 1 1 → Sequence 1 1) ∧
      Monotone (id : Sequence 1 1 → Sequence 1 1)) :=
  ⟨unitHead_conditions, convexOn_id convex_univ, monotone_id⟩

/-- Proposition 1 in full: Eqs. (15)–(22) preserve both componentwise
properties under Assumption 1; arXiv:2603.22095v2, §3.3. -/
theorem attention_properties (p : MultiHead m h heads)
    (F : E → Sequence (n + 1) m) (hp : MultiHeadConditions p)
    (hF : ComponentwiseConvex F ∧ Monotone F) :
    ComponentwiseConvex (fun x => attention p (F x)) ∧
      Monotone (fun x => attention p (F x)) := by
  have hh := fun a => headContext_properties (p.head a) F (hp.1 a) hF
  have hi : ∀ i : Fin (n + 1),
      ComponentwiseConvex (fun x (ar : Fin heads × Fin h) =>
        headContext (p.head ar.1) (F x) (i, ar.2)) ∧
      Monotone (fun x (ar : Fin heads × Fin h) =>
        headContext (p.head ar.1) (F x) (i, ar.2)) := by
    intro i
    exact ⟨⟨convex_univ, fun _ hx _ hy _ _ ha hb hab ar =>
      (hh ar.1).1.2 hx hy ha hb hab (i, ar.2)⟩,
      fun _ _ h ar => (hh ar.1).2 h (i, ar.2)⟩
  constructor
  · exact ⟨convex_univ, fun _ hx _ hy _ _ ha hb hab ir =>
      (affine_convex p.projection p.bias _ hp.2 (hi ir.1).1).2 hx hy ha hb hab ir.2⟩
  · intro x y h ir
    exact affine_monotone p.projection p.bias _ hp.2 (hi ir.1).2 h ir.2

example : MultiHeadConditions unitAttention ∧
    (ComponentwiseConvex (id : Sequence 1 1 → Sequence 1 1) ∧
      Monotone (id : Sequence 1 1 → Sequence 1 1)) :=
  ⟨unitAttention_conditions, convexOn_id convex_univ, monotone_id⟩

end Transformer.ICEoT
