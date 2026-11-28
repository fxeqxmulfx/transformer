module

public import Transformer.Grokking.FiniteRecovery.RectangleLoss

/-! An observable loss threshold for binary factor recovery. The data
term reads the two staircase diagonals. The two patch penalties use only
the model's logits. Their joint threshold also enforces a strict margin. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.RectangleMargin

open Finset
open scoped BigOperators

/-- Squared shortfall below unit absolute gap at an unlabeled position. -/
def shortfall (g : ℝ) := (max 0 (1 - |g|))^2

/-- Binary cross-entropy in terms of the raw logit difference z₀−z₁. -/
noncomputable def binaryCE (g : ℝ) (y : ZMod 2) :=
  Real.log (1 + Real.exp (if y = 0 then -g else g))

theorem binaryCE_nonneg (g : ℝ) (y : ZMod 2) : 0 ≤ binaryCE g y :=
  Real.log_nonneg (by linarith [Real.exp_pos (if y = 0 then -g else g)])

/-- Cross-entropy below log 2 implies a strictly correct binary label. -/
theorem strict_of_binaryCE (g : ℝ) (y : ZMod 2) (h : binaryCE g y < Real.log 2) :
    g ≠ 0 ∧ negativeBit g = y := by
  have hz : (if y = 0 then -g else g) < 0 := by
    by_contra! hn
    have hex := Real.one_le_exp hn
    exact h.not_ge (Real.log_le_log (by norm_num) (by linarith only [hex]))
  rcases (by decide : ∀ a : ZMod 2, a = 0 ∨ a = 1) y with rfl | rfl
  · have pos : 0 < g := by simpa using hz
    simp [negativeBit,pos.not_gt,pos.ne']
  · have neg : g < 0 := by simpa using hz
    simp [negativeBit,neg,neg.ne]

variable {I J : Type*}

def patchShortfall (p : Finset I × Finset J) (g : I → J → ℝ) :=
  ∑ i ∈ p.1, ∑ j ∈ p.2, shortfall (g i j)

private theorem patch_nonneg (p : Finset I × Finset J) (g : I → J → ℝ)
    (u : Finset I × Finset J → I → ℝ) (v : Finset I × Finset J → J → ℝ) :
    0 ≤ rectangleError p g (u p) (v p) ∧ 0 ≤ patchShortfall p g := by
  constructor <;> exact sum_nonneg fun _ _ => sum_nonneg fun _ _ => sq_nonneg _

/-- If every four-corner rectangle belongs to at least count patches,
a joint squared loss below count/4 enforces a half-unit absolute gap
on any finite rectangular sampling design with at least two indices per axis. -/
theorem margin_of_joint_loss [DecidableEq I] [DecidableEq J] [Nontrivial I] [Nontrivial J]
    (samples : Finset (Finset I × Finset J)) (g : I → J → ℝ)
    (u : Finset I × Finset J → I → ℝ) (v : Finset I × Finset J → J → ℝ) {count : ℕ}
    (cover : ∀ i k j l, i ≠ k → j ≠ l → count ≤ (coveringRectangles samples i k j l).card)
    (small : ∑ p ∈ samples, (rectangleError p g (u p) (v p) + patchShortfall p g) < count/4)
    (i : I) (j : J) : 1/2 ≤ |g i j| := by
  by_contra! bad
  obtain ⟨k,hk⟩ := exists_ne i
  obtain ⟨l,hl⟩ := exists_ne j
  let C := coveringRectangles samples i k j l
  have large (p : Finset I × Finset J) (hp : p ∈ C) : 1/4 ≤ rectangleError p g (u p) (v p) + patchShortfall p g := by
    obtain ⟨rows,cols⟩ := (mem_filter.mp hp).2
    have hi : i ∈ p.1 := rows (by simp)
    have hj : j ∈ p.2 := cols (by simp)
    have point : 1/4 ≤ shortfall (g i j) := by
      rw [shortfall, max_eq_right (by linarith : 0 ≤ 1 - |g i j|)]
      nlinarith [sq_nonneg (1/2 - |g i j|)]
    have colSum := single_le_sum (f := fun b => shortfall (g i b)) (fun _ _ => sq_nonneg _) hj
    have rowSum := single_le_sum (f := fun a => ∑ b ∈ p.2, shortfall (g a b))
      (fun _ _ => sum_nonneg fun _ _ => sq_nonneg _) hi
    exact (point.trans (colSum.trans rowSum)).trans (le_add_of_nonneg_left (patch_nonneg p g u v).1)
  apply small.not_ge
  calc
    (count:ℝ)/4 ≤ (C.card:ℝ)/4 :=
      div_le_div_of_nonneg_right (by exact_mod_cast cover i k j l hk.symm hl.symm) (by norm_num)
    _ = ∑ _p ∈ C, (1/4 : ℝ) := by simp; ring
    _ ≤ ∑ p ∈ C, _ := sum_le_sum large
    _ ≤ _ := sum_le_sum_of_subset_of_nonneg (filter_subset _ _) fun p _ _ =>
      add_nonneg (patch_nonneg p g u v).1 (patch_nonneg p g u v).2

/-- The class of tables with unknown additive binary row and column labels. -/
def binaryFactors (I J : Type*) : Set (I × J → ZMod 2) :=
  {f | ∃ (a : I → ZMod 2) (b : J → ZMod 2), ∀ i j, f (i,j) = a i + b j}

/-- Any identifying mask and any rectangular sampling design with positive
four-corner coverage work for the whole binary factor class. The hypotheses
read only observed labels, unlabeled logits and chosen local factors. -/
theorem strict_on_identifying_mask [DecidableEq I] [DecidableEq J] [Nontrivial I] [Nontrivial J]
    (samples : Finset (Finset I × Finset J)) (g : I → J → ℝ)
    (u : Finset I × Finset J → I → ℝ) (v : Finset I × Finset J → J → ℝ) {count : ℕ}
    (cover : ∀ i k j l, i ≠ k → j ≠ l → count ≤ (coveringRectangles samples i k j l).card)
    (small : ∑ p ∈ samples, (rectangleError p g (u p) (v p) + patchShortfall p g) < count/4)
    (S : Set (I × J)) (identifies : Set.InjOn (fun f : I × J → ZMod 2 => fun x : S => f x) (binaryFactors I J))
    (f : I × J → ZMod 2) (hf : f ∈ binaryFactors I J)
    (obs : ∀ x : S, binaryCE (g x.val.1 x.val.2) (f x) < Real.log 2) :
    ∀ i j, g i j ≠ 0 ∧ negativeBit (g i j) = f (i,j) := by
  have margin := margin_of_joint_loss samples g u v cover small
  have alone : ∑ p ∈ samples, rectangleError p g (u p) (v p) < count * (1/2 : ℝ)^2 :=
    (sum_le_sum fun p _ => le_add_of_nonneg_right (patch_nonneg p g u v).2).trans_lt
      (small.trans_eq (by ring))
  let bits := fun i j => negativeBit (g i j)
  have rect := rectangle_sign_law samples g u v (m := 1/2) (by norm_num) margin cover alone
  let i₀ : I := Classical.arbitrary _
  let j₀ : J := Classical.arbitrary _
  have member : (fun x : I × J => bits x.1 x.2) ∈ binaryFactors I J :=
    ⟨fun i => bits i j₀, fun j => bits i₀ j - bits i₀ j₀, bits_factor_of_rectangle_law bits i₀ j₀ rect⟩
  have eq := identifies member hf (funext fun x => (strict_of_binaryCE _ _ (obs x)).2)
  exact fun i j => ⟨abs_pos.mp ((by norm_num : (0:ℝ) < 1/2).trans_le (margin i j)),
    congrFun eq (i,j)⟩

abbrev Patch := Finset (Fin 16) × Finset (Fin 16)

def patches : Finset Patch := univ.powersetCard 5 ×ˢ univ.powersetCard 5

/-- Expected sum of the two half-mean-square penalties on a uniform five-by-five patch. -/
noncomputable def penalty (g : Fin 16 → Fin 16 → ℝ) (u v : Patch → Fin 16 → ℝ) :=
  (∑ p ∈ patches, (rectangleError p g (u p) (v p) + patchShortfall p g)) / (50 * 4368^2)

private theorem margin_of_small_penalty (g : Fin 16 → Fin 16 → ℝ) (u v : Patch → Fin 16 → ℝ)
    (small : penalty g u v < 1/28800) : ∀ i j, 1/2 ≤ |g i j| := by
  apply margin_of_joint_loss patches g u v (count := 132496)
  · intro i k j l hik hjl
    dsimp only [patches]
    rw [card_covering_uniform _ _ (by decide) (by decide) _ _ _ _ hik hjl]
    decide
  · unfold penalty at small
    norm_num at small ⊢
    linarith only [small]

/-- Only the 31 supplied signs enter the supervised part of the loss. -/
noncomputable def observedCE (g : Fin 16 → Fin 16 → ℝ) (d : Fin 16 → ZMod 2) (e : Fin 15 → ZMod 2) :=
  ((∑ i, binaryCE (g i i) (d i)) + ∑ i : Fin 15, binaryCE (g i.castSucc i.succ) (e i))/31

/-- One scalar objective, independently of optimizer or initialization,
certifies the complete table for every unknown additive binary factor pair. -/
theorem strict_of_small_objective (g : Fin 16 → Fin 16 → ℝ) (u v : Patch → Fin 16 → ℝ)
    (a b : Fin 16 → ZMod 2)
    (h : observedCE g (fun i => a i+b i) (fun j => a j.castSucc+b j.succ) + penalty g u v < 1/28800) :
    ∀ i j, g i j ≠ 0 ∧ negativeBit (g i j) = a i + b j := by
  let D := fun i : Fin 16 => binaryCE (g i i) (a i + b i)
  let E := fun j : Fin 15 => binaryCE (g j.castSucc j.succ) (a j.castSucc + b j.succ)
  have hd : 0 ≤ ∑ i, D i := sum_nonneg fun _ _ => binaryCE_nonneg _ _
  have he : 0 ≤ ∑ j, E j := sum_nonneg fun _ _ => binaryCE_nonneg _ _
  change ((∑ i, D i) + ∑ j, E j)/31 + penalty g u v < _ at h
  have pn : 0 ≤ penalty g u v := div_nonneg (sum_nonneg fun p _ =>
    add_nonneg (patch_nonneg p g u v).1 (patch_nonneg p g u v).2) (by norm_num)
  have loglb := Real.one_sub_inv_le_log_of_pos (by norm_num : 0 < (2:ℝ))
  norm_num at loglb
  have cap : (∑ i, D i) + ∑ j, E j < Real.log 2 := by linarith only [h,pn,loglb]
  have small : penalty g u v < 1/28800 := by linarith only [h,hd,he]
  apply five_of_sixteen_gap_recovery g u v a b (m := 1/2) (by norm_num) (margin_of_small_penalty _ _ _ small)
  · intro i
    exact (strict_of_binaryCE _ _ (((single_le_sum (f := D) (fun _ _ => binaryCE_nonneg _ _) (mem_univ i)).trans
      (le_add_of_nonneg_right he)).trans_lt cap)).2
  · intro j
    exact (strict_of_binaryCE _ _ (((single_le_sum (f := E) (fun _ _ => binaryCE_nonneg _ _) (mem_univ j)).trans
      (le_add_of_nonneg_left hd)).trans_lt cap)).2
  · change (∑ p ∈ patches, rectangleError p g (u p) (v p)) / (50 * 4368 ^ 2) < _
    have mono : (∑ p ∈ patches, rectangleError p g (u p) (v p)) / (50 * 4368 ^ 2) ≤ penalty g u v :=
      div_le_div_of_nonneg_right (sum_le_sum fun p _ => le_add_of_nonneg_right (patch_nonneg p g u v).2) (by norm_num)
    exact (mono.trans_lt small).trans_eq (by norm_num)

/-- Vanishing observable loss along any sequence of full-model logit
tables suffices for eventual exact generalization. -/
theorem eventually_strict_of_loss_zero (g : ℕ → Fin 16 → Fin 16 → ℝ) (u v : ℕ → Patch → Fin 16 → ℝ)
    (a b : Fin 16 → ZMod 2) (h : Filter.Tendsto (fun t =>
      observedCE (g t) (fun i => a i+b i) (fun j => a j.castSucc+b j.succ) + penalty (g t) (u t) (v t))
      Filter.atTop (nhds 0)) :
    ∀ᶠ t in Filter.atTop, ∀ i j, g t i j ≠ 0 ∧ negativeBit (g t i j) = a i+b j := by
  filter_upwards [h.eventually_lt_const (by norm_num : (0:ℝ) < 1/28800)] with t ht
  exact strict_of_small_objective _ _ _ _ _ ht

example : negativeBit (2:ℝ) = 0 ∧ shortfall 2 = 0 := by norm_num [negativeBit, shortfall]

end Transformer.Grokking.FiniteRecovery.RectangleMargin
