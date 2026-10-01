/-
# IC-EoT: residual-block and complete-network convexity

arXiv:2603.22095v2, §3.3, Proposition 2, Theorem 1 and Corollary 1.
Theorems refer to the actual equations in `Section3_EncoderModel`.
-/

import Transformer.ICEoT.Section3_EncoderModel

noncomputable section

namespace Transformer.ICEoT

variable {E : Type*} [AddCommGroup E] [Module ℝ E] [Preorder E]
variable {n m h heads ff out : ℕ} {I : Type*} [Fintype I]

/-- The feed-forward sublayer preserves both properties; §3.3,
Proposition 2, Eq. (24). -/
theorem feedForward_properties (p : Block m h heads ff)
    (F : E → Sequence (n + 1) m) (hp : BlockConditions p)
    (hF : ComponentwiseConvex F ∧ Monotone F) :
    ComponentwiseConvex (fun x => feedForward p (F x)) ∧
      Monotone (fun x => feedForward p (F x)) := by
  have hi : ∀ i : Fin (n + 1), ComponentwiseConvex (fun x r => F x (i, r)) ∧
      Monotone (fun x r => F x (i, r)) := fun i =>
    ⟨⟨convex_univ, fun _ hx _ hy _ _ ha hb hab r => hF.1.2 hx hy ha hb hab (i, r)⟩,
      fun _ _ h r => hF.2 h (i, r)⟩
  have hf := fun i => activation_properties p.activation _ hp.2.2.2
    ⟨affine_convex p.first p.firstBias _ hp.2.1 (hi i).1,
      affine_monotone p.first p.firstBias _ hp.2.1 (hi i).2⟩
  constructor
  · exact ⟨convex_univ, fun _ hx _ hy _ _ ha hb hab ir =>
      (affine_convex p.second p.secondBias _ hp.2.2.1 (hf ir.1).1).2 hx hy ha hb hab ir.2⟩
  · intro x y h ir
    exact affine_monotone p.second p.secondBias _ hp.2.2.1 (hf ir.1).2 h ir.2

example : BlockConditions unitBlock ∧
    (ComponentwiseConvex (id : Sequence 1 1 → Sequence 1 1) ∧
      Monotone (id : Sequence 1 1 → Sequence 1 1)) :=
  ⟨unitBlock_conditions, convexOn_id convex_univ, monotone_id⟩

/-- Proposition 2, including both residual additions, under Assumption 1;
§3.3, Eqs. (23)–(25). -/
theorem encoderBlock_properties (p : Block m h heads ff)
    (F : E → Sequence (n + 1) m) (hp : BlockConditions p)
    (hF : ComponentwiseConvex F ∧ Monotone F) :
    ComponentwiseConvex (fun x => encoderBlock p (F x)) ∧
      Monotone (fun x => encoderBlock p (F x)) := by
  have hr := add_properties F _ hF (attention_properties p.att F hp.1 hF)
  exact add_properties _ _ hr (feedForward_properties p _ hp hr)

example : BlockConditions unitBlock ∧
    (ComponentwiseConvex (id : Sequence 1 1 → Sequence 1 1) ∧
      Monotone (id : Sequence 1 1 → Sequence 1 1)) :=
  ⟨unitBlock_conditions, convexOn_id convex_univ, monotone_id⟩

/-- Induction over every encoder block; §3.3, Theorem 1. -/
theorem encoderStack_properties (bs : List (Block m h heads ff))
    (F : E → Sequence (n + 1) m) (hbs : ∀ b ∈ bs, BlockConditions b)
    (hF : ComponentwiseConvex F ∧ Monotone F) :
    ComponentwiseConvex (fun x => encoderStack bs (F x)) ∧
      Monotone (fun x => encoderStack bs (F x)) := by
  induction bs generalizing F with
  | nil => exact hF
  | cons b bs ih =>
    exact ih _ (fun c hc => hbs c (List.mem_cons_of_mem _ hc))
      (encoderBlock_properties b F (hbs b (List.mem_cons_self)) hF)

example : (∀ b ∈ [unitBlock, unitBlock], BlockConditions b) ∧
    (ComponentwiseConvex (id : Sequence 1 1 → Sequence 1 1) ∧
      Monotone (id : Sequence 1 1 → Sequence 1 1)) := by
  refine ⟨?_, convexOn_id convex_univ, monotone_id⟩
  intro b hb
  simp only [List.mem_cons, List.not_mem_nil, or_false, or_self] at hb
  rw [hb]
  exact unitBlock_conditions

/-- The input embedding with arbitrary fixed biases and positional encoding;
§3.3, Theorem 1, Eq. (14). -/
theorem embed_properties (p : Encoder n I m h heads ff out)
    (hp : Nonnegative p.embedding) :
    ComponentwiseConvex (embed p) ∧ Monotone (embed p) := by
  have hi : ∀ i : Fin (n + 1),
      ComponentwiseConvex (fun X : (Fin (n + 1) × I) → ℝ => fun r => X (i, r)) ∧
      Monotone (fun X : (Fin (n + 1) × I) → ℝ => fun r => X (i, r)) := fun _ =>
    ⟨⟨convex_univ, fun _ _ _ _ _ _ _ _ _ _ => le_rfl⟩, fun _ _ h r => h (_, r)⟩
  constructor
  · refine ⟨convex_univ, ?_⟩
    intro x hx y hy a b ha hb hab ir
    have hc := (affine_convex p.embedding p.embeddingBias _ hp (hi ir.1).1).2
      hx hy ha hb hab ir.2
    change affine _ _ _ _ + _ ≤ a * (affine _ _ _ _ + _) + b * (affine _ _ _ _ + _)
    have hpos : a * p.pos ir + b * p.pos ir = p.pos ir := by rw [← add_mul, hab, one_mul]
    simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul] at hc ⊢
    nlinarith only [hc, hpos]
  · intro x y h ir
    exact add_le_add (affine_monotone p.embedding p.embeddingBias _ hp (hi ir.1).2 h ir.2)
      le_rfl

example : Nonnegative (unitEncoder 2).embedding := fun _ _ => zero_le_one

/-- Theorem 1: the complete IC-EoT, for any number of blocks, is
componentwise convex and non-decreasing in its expanded input.
Source: arXiv:2603.22095v2, §3.3, Assumption 1, Eqs. (14)–(27). -/
theorem predict_properties (p : Encoder n I m h heads ff out)
    (hp : EncoderConditions p) : ComponentwiseConvex (predict p) ∧ Monotone (predict p) := by
  have hs := encoderStack_properties p.blocks (embed p) hp.2.1 (embed_properties p hp.1)
  have hl : ComponentwiseConvex
      (fun X r => encoderStack p.blocks (embed p X) (Fin.last n, r)) ∧
      Monotone (fun X r => encoderStack p.blocks (embed p X) (Fin.last n, r)) :=
    ⟨⟨convex_univ, fun _ hx _ hy _ _ ha hb hab r => hs.1.2 hx hy ha hb hab (Fin.last n, r)⟩,
      fun _ _ h r => hs.2 h (Fin.last n, r)⟩
  exact ⟨affine_convex p.readout p.readoutBias _ hp.2.2 hl.1,
    affine_monotone p.readout p.readoutBias _ hp.2.2 hl.2⟩

example : EncoderConditions (unitEncoder 2) := unitEncoder_conditions 2

/-- The expansion is affine even though its negative branch is decreasing;
§3.3, Corollary 1, Eq. (13). -/
theorem expand_combination {d k : ℕ} (X Y : Sequence k d) (a b : ℝ) :
    expand (a • X + b • Y) = a • expand X + b • expand Y := by
  funext ir
  rcases ir with ⟨i, s, r⟩
  cases s <;> simp [expand, add_comm]

/-- Corollary 1: convexity in the original history, with no monotonicity
claim for that history; arXiv:2603.22095v2, §3.3. -/
theorem predictOriginal_convex {d : ℕ} (p : Encoder n (Bool × Fin d) m h heads ff out)
    (hp : EncoderConditions p) : ComponentwiseConvex (predictOriginal p) := by
  refine ⟨convex_univ, ?_⟩
  intro X hx Y hy a b ha hb hab
  unfold predictOriginal
  rw [expand_combination]
  exact (predict_properties p hp).1.2 (Set.mem_univ _) (Set.mem_univ _) ha hb hab

example : EncoderConditions (unitEncoder 3) := unitEncoder_conditions 3

/-- Positive output scales in de-standardization preserve both properties;
§3.2, paragraph after Eq. (27). -/
theorem destandardize_properties {O : Type*} (f : E → O → ℝ)
    (scale mean : O → ℝ) (hs : ∀ o, 0 < scale o)
    (hf : ComponentwiseConvex f ∧ Monotone f) :
    ComponentwiseConvex (fun x o => scale o * f x o + mean o) ∧
      Monotone (fun x o => scale o * f x o + mean o) := by
  constructor
  · refine ⟨convex_univ, ?_⟩
    intro x hx y hy a b ha hb hab o
    have h := mul_le_mul_of_nonneg_left (hf.1.2 hx hy ha hb hab o) (le_of_lt (hs o))
    have hc : a * mean o + b * mean o = mean o := by rw [← add_mul, hab, one_mul]
    simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul] at h ⊢
    nlinarith only [h, hc]
  · intro x y h o
    exact add_le_add (mul_le_mul_of_nonneg_left (hf.2 h o) (le_of_lt (hs o))) le_rfl

example : (∀ _ : Fin 1, (0 : ℝ) < 1) ∧
    (ComponentwiseConvex (id : (Fin 1 → ℝ) → Fin 1 → ℝ) ∧
      Monotone (id : (Fin 1 → ℝ) → Fin 1 → ℝ)) :=
  ⟨fun _ => zero_lt_one, convexOn_id convex_univ, monotone_id⟩

end Transformer.ICEoT
