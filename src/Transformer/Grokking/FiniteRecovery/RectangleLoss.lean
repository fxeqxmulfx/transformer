module

public import Transformer.Grokking.FiniteRecovery.CheckpointCertificate
public import Mathlib.Data.Finset.Powerset

/-! A certificate for training with the squared residual of a rank-one
approximation on unlabeled rectangular minibatches. Each bad sign
rectangle contributes to every sampled patch that contains its corners. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery

open Finset
open scoped BigOperators

variable {I J : Type*}

/-- Sum of squared logit-gap residuals on one rectangular patch. The
trainer uses the leading singular component for the local row/column factors. -/
def rectangleError (p : Finset I × Finset J) (score : I → J → ℝ) (u : I → ℝ) (v : J → ℝ) : ℝ :=
  ∑ i ∈ p.1, ∑ j ∈ p.2, (score i j - u i * v j)^2

private theorem cell_le_rectangle (p : Finset I × Finset J) (score : I → J → ℝ)
    (u : I → ℝ) (v : J → ℝ) {i : I} (hi : i ∈ p.1) {j : J} (hj : j ∈ p.2) :
    (score i j - u i * v j)^2 ≤ rectangleError p score u v := by
  exact (single_le_sum (f := fun b => (score i b - u i * v b)^2) (fun _ _ => sq_nonneg _) hj).trans
    (single_le_sum (f := fun a => ∑ b ∈ p.2, (score a b - u a * v b)^2)
      (fun _ _ => sum_nonneg fun _ _ => sq_nonneg _) hi)

variable [DecidableEq I] [DecidableEq J]

/-- Patches which contain all four entries of one rectangle. -/
def coveringRectangles (patches : Finset (Finset I × Finset J)) (i k : I) (j l : J) :=
  patches.filter fun p => {i,k} ⊆ p.1 ∧ {j,l} ⊆ p.2

/-- Below the loss threshold every sign rectangle is additive, even when
different patches use different fitted factors. The margin refers only to
the absolute model gap. Its computation needs no hidden target. -/
theorem rectangle_sign_law (patches : Finset (Finset I × Finset J)) (score : I → J → ℝ)
    (u : Finset I × Finset J → I → ℝ) (v : Finset I × Finset J → J → ℝ)
    {m : ℝ} (hm : 0 < m) (margin : ∀ i j, m ≤ |score i j|) {count : ℕ}
    (cover : ∀ i k j l, i ≠ k → j ≠ l → count ≤ (coveringRectangles patches i k j l).card)
    (small : ∑ p ∈ patches, rectangleError p score (u p) (v p) < count * m^2)
    (i k : I) (j l : J) :
    negativeBit (score i j) + negativeBit (score k l) =
      negativeBit (score i l) + negativeBit (score k j) := by
  by_cases hik : i = k
  · subst k; exact add_comm _ _
  by_cases hjl : j = l
  · subst l; rfl
  by_contra bad
  let C := coveringRectangles patches i k j l
  have loss (p) (hp : p ∈ C) : m^2 ≤ rectangleError p score (u p) (v p) := by
    by_contra! h
    obtain ⟨rows,cols⟩ := (mem_filter.mp hp).2
    have cell (a : I) (ha : a ∈ p.1) (b : J) (hb : b ∈ p.2) :
        negativeBit (score a b) = negativeBit (u p a) + negativeBit (v p b) := by
      have hs := (cell_le_rectangle p score (u p) (v p) ha hb).trans_lt h
      have close : |score a b - u p a * v p b| < m := by
        nlinarith [sq_abs (score a b - u p a * v p b), abs_nonneg (score a b - u p a * v p b)]
      rw [abs_sub_comm] at close
      obtain ⟨_,hn,eq⟩ := strict_sign_of_residual (close.trans_le (margin a b))
      exact eq.symm.trans (negativeBit_mul (left_ne_zero_of_mul hn) (right_ne_zero_of_mul hn))
    apply bad
    rw [cell i (rows (by simp)) j (cols (by simp)), cell k (rows (by simp)) l (cols (by simp)),
      cell i (rows (by simp)) l (cols (by simp)), cell k (rows (by simp)) j (cols (by simp))]
    abel
  apply small.not_ge
  calc
    (count : ℝ) * m^2 ≤ (C.card : ℝ) * m^2 :=
      mul_le_mul_of_nonneg_right (by exact_mod_cast cover i k j l hik hjl) (sq_nonneg _)
    _ = ∑ _p ∈ C, m^2 := by simp
    _ ≤ ∑ p ∈ C, rectangleError p score (u p) (v p) := sum_le_sum loss
    _ ≤ _ := sum_le_sum_of_subset_of_nonneg (filter_subset _ _) fun _ _ _ =>
      sum_nonneg fun _ _ => sum_nonneg fun _ _ => sq_nonneg _

omit [DecidableEq I] [DecidableEq J] in
/-- A four-corner law on any rectangular shape factors all sign bits.
Its restriction to an identifying observation mask therefore determines
all hidden signs within the additive class. -/
theorem bits_factor_of_rectangle_law (bits : I → J → ZMod 2) (i₀ : I) (j₀ : J)
    (rect : ∀ i k j l, bits i j + bits k l = bits i l + bits k j) :
    ∀ i j, bits i j = bits i j₀ + (bits i₀ j - bits i₀ j₀) := by
  intro i j
  rw [← add_sub_assoc]
  exact eq_sub_iff_add_eq.mpr (rect i i₀ j j₀)

/-- Matching the two observed staircase diagonals fixes every additive
bit once the rectangle law holds. -/
theorem staircase_of_rectangle_law {s : ℕ} (bits : Fin (s+1) → Fin (s+1) → ZMod 2)
    (row col : Fin (s+1) → ZMod 2)
    (rect : ∀ i k j l, bits i j + bits k l = bits i l + bits k j)
    (hd : ∀ i, bits i i = row i + col i)
    (he : ∀ j : Fin s, bits j.castSucc j.succ = row j.castSucc + col j.succ) :
    ∀ i j, bits i j = row i + col j := by
  let a i := bits i 0
  let b j := bits 0 j - bits 0 0
  have h (i j) : bits i j = a i + b j := bits_factor_of_rectangle_law bits 0 0 rect i j
  have d : (fun i => a i + b i) = fun i => row i + col i := funext fun i => (h i i).symm.trans (hd i)
  have e : (fun j : Fin s => a j.castSucc + b j.succ) = fun j => row j.castSucc + col j.succ :=
    funext fun j => (h _ _).symm.trans (he j)
  intro i j
  rw [h, ← staircaseDecode_additive a b, d, e, staircaseDecode_additive]

/-- Exact coverage for independent, uniformly chosen row and column
subsets. The rows and columns may have different cardinalities. -/
theorem card_covering_uniform [Fintype I] [Fintype J] (r c : ℕ) (hr : 2 ≤ r) (hc : 2 ≤ c)
    (i k : I) (j l : J) (hik : i ≠ k) (hjl : j ≠ l) :
    (coveringRectangles (univ.powersetCard r ×ˢ univ.powersetCard c) i k j l).card =
      (Fintype.card I - 2).choose (r - 2) * (Fintype.card J - 2).choose (c - 2) := by
  have h : coveringRectangles (univ.powersetCard r ×ˢ univ.powersetCard c) i k j l =
      ((univ.powersetCard r).filter ({i,k} ⊆ ·)) ×ˢ ((univ.powersetCard c).filter ({j,l} ⊆ ·)) := by
    ext ⟨a,b⟩
    simp [coveringRectangles, and_assoc, and_left_comm]
  rw [h, card_product, card_filter_powersetCard_subset _ _ _ (subset_univ _) (by simpa [hik] using hr),
    card_filter_powersetCard_subset _ _ _ (subset_univ _) (by simpa [hjl] using hc)]
  simp [hik,hjl]

/-- The eight-bit experiment samples five rows and five columns out of
sixteen. Its expected half-mean-square loss below m²/7200, minimum
absolute gap at least m, and the 31 observed signs certify every answer. -/
theorem five_of_sixteen_gap_recovery (score : Fin 16 → Fin 16 → ℝ)
    (u v : Finset (Fin 16) × Finset (Fin 16) → Fin 16 → ℝ) (row col : Fin 16 → ZMod 2)
    {m : ℝ} (hm : 0 < m) (margin : ∀ i j, m ≤ |score i j|)
    (hd : ∀ i, negativeBit (score i i) = row i + col i)
    (he : ∀ i : Fin 15, negativeBit (score i.castSucc i.succ) = row i.castSucc + col i.succ)
    (small : (∑ p ∈ (univ.powersetCard 5 ×ˢ univ.powersetCard 5),
      rectangleError p score (u p) (v p)) / (50 * 4368^2) < m^2 / 7200) :
    ∀ i j, score i j ≠ 0 ∧ negativeBit (score i j) = row i + col j := by
  have cover (i k j l : Fin 16) (hik : i ≠ k) (hjl : j ≠ l) :
      132496 ≤ (coveringRectangles (univ.powersetCard 5 ×ˢ univ.powersetCard 5) i k j l).card := by
    rw [card_covering_uniform _ _ (by decide) (by decide) _ _ _ _ hik hjl]
    decide
  have law := rectangle_sign_law _ score u v hm margin cover (by
    norm_num at small ⊢
    linarith only [small])
  exact fun i j => ⟨abs_pos.mp (hm.trans_le (margin i j)),
    staircase_of_rectangle_law _ row col law hd he i j⟩

/-- Along any sequence of model checkpoints, eventual observed fit, a
persistent absolute margin and expected rectangle loss tending to zero
imply eventual strict recovery on every hidden input simultaneously. -/
theorem eventually_five_of_sixteen {m : ℝ} (hm : 0 < m) (score : ℕ → Fin 16 → Fin 16 → ℝ)
    (u v : ℕ → Finset (Fin 16) × Finset (Fin 16) → Fin 16 → ℝ) (row col : Fin 16 → ZMod 2)
    (fit : ∀ᶠ t in Filter.atTop, (∀ i j, m ≤ |score t i j|) ∧
      (∀ i, negativeBit (score t i i) = row i + col i) ∧
      (∀ i : Fin 15, negativeBit (score t i.castSucc i.succ) = row i.castSucc + col i.succ))
    (h : Filter.Tendsto (fun t => (∑ p ∈ (univ.powersetCard 5 ×ˢ univ.powersetCard 5),
      rectangleError p (score t) (u t p) (v t p)) / (50 * 4368^2)) Filter.atTop (nhds 0)) :
    ∀ᶠ t in Filter.atTop, ∀ i j, score t i j ≠ 0 ∧ negativeBit (score t i j) = row i + col j := by
  have pos : 0 < m^2 / 7200 := by positivity
  filter_upwards [fit, h.eventually_lt_const pos] with t ht ht'
  exact five_of_sixteen_gap_recovery _ _ _ _ _ hm ht.1 ht.2.1 ht.2.2 ht'

example (i k j l : Fin 16) (hik : i ≠ k) (hjl : j ≠ l) :
    144 * (coveringRectangles (univ.powersetCard 5 ×ˢ univ.powersetCard 5) i k j l).card =
      ((univ : Finset (Fin 16)).powersetCard 5).card ^ 2 := by
  rw [card_covering_uniform _ _ (by decide) (by decide) _ _ _ _ hik hjl, card_powersetCard]
  decide

/-- Strict signs of the raw binary gap give strict realization by the
full Argand at its prediction position, at arbitrary width and depth. -/
theorem Argand.realizes_of_binary_logit_gaps {T H d L : ℕ} (model : Model (ZMod 2) H d L)
    (r : ℕ) (f : Word (ZMod 2) (T+1) → ZMod 2)
    (gaps : ∀ w, let z := model.logits r w (Fin.last T)
      z 0 - z 1 ≠ 0 ∧ negativeBit (z 0 - z 1) = f w) : model.realizes r f := by
  intro w b hb
  let z := model.logits r w (Fin.last T)
  obtain ⟨hne,hs⟩ := gaps w
  have either : ∀ a : ZMod 2, a = 0 ∨ a = 1 := by decide
  by_cases h : z 0 - z 1 < 0
  · have hf : f w = 1 := by simpa [negativeBit, z, h] using hs.symm
    rw [hf] at hb ⊢
    rw [(either b).resolve_right hb]
    exact sub_neg.mp h
  · have hf : f w = 0 := by simpa [negativeBit, z, h] using hs.symm
    rw [hf] at hb ⊢
    rw [(either b).resolve_left hb]
    exact sub_pos.mp (lt_of_le_of_ne (le_of_not_gt h) (Ne.symm hne))

end Transformer.Grokking.FiniteRecovery
