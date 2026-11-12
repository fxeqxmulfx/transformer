module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.JointHead

/-!
# Retain shallower features when adding nonlinear depth

Source: `review/convex_depth.md`, CD9. Final-depth-only path libraries need not
contain their shallower predecessors. Append the new complete-path features
without deleting the old coordinates. Zero extension preserves each old head's
logits, ordinary CE and absolute-sum budget. This proves nested capacity, not
successful optimization or correctness of any missing label.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth

open scoped BigOperators

variable {K A X C S : Type*} [Fintype K] [Fintype A]

/-- Append a fixed feature family while retaining every old coordinate (CD9). -/
def retainedFeatures (φ : K → X → ℝ) (ψ : A → X → ℝ) : K ⊕ A → X → ℝ :=
  Sum.elim φ ψ

/-- Embed a trained head by setting only the newly added coordinates to zero
(CD9). Other feasible extended heads may use those new features. -/
def retainHead (W : K → C → ℝ) : K ⊕ A → C → ℝ := Sum.elim W 0

/-- Retaining old coordinates reproduces their actual class logits on every
input, even when new features contain deeper nonlinear interactions (CD9). -/
theorem retained_logits (φ : K → X → ℝ) (ψ : A → X → ℝ) (W : K → C → ℝ) (x : X) :
    headLogits (retainedFeatures φ ψ) (retainHead W) x = headLogits φ W x := by
  funext c
  unfold headLogits
  rw [Fintype.sum_sum_type]
  change (∑ k, W k c * φ k x) + (∑ a, 0 * ψ a x) = ∑ k, W k c * φ k x
  simp

/-- Zero extension uses exactly the same per-class budget; adding depth does
not require increasing R to retain an existing solution (CD9). -/
theorem retained_budget_iff (W : K → C → ℝ) (R : ℝ) :
    retainHead (A := A) W ∈ headBudget R ↔ W ∈ headBudget R := by
  change (∀ c, ∑ i, |retainHead W i c| ≤ R) ↔ (∀ c, ∑ k, |W k c| ≤ R)
  simp [retainHead, Fintype.sum_sum_type]

/-- Ordinary CE is unchanged for the retained head, with every original
observation and its multiplicity still in the sum (CD9). -/
theorem retained_loss [Fintype C] [Fintype S] (φ : K → X → ℝ) (ψ : A → X → ℝ)
    (inputs : S → X) (labels : S → C) (W : K → C → ℝ) :
    headLoss (retainedFeatures φ ψ) inputs labels (retainHead W) =
      headLoss φ inputs labels W := by
  unfold headLoss
  apply Finset.sum_congr rfl
  intro s hs
  rw [retained_logits]

/-- Every budget-feasible CE value in the old family remains achievable in the
larger convex family. This is a capacity inclusion, not a training guarantee (CD9). -/
theorem retained_loss_image [Fintype C] [Fintype S] (φ : K → X → ℝ) (ψ : A → X → ℝ)
    (inputs : S → X) (labels : S → C) (R : ℝ) :
    headLoss φ inputs labels '' headBudget R ⊆
      headLoss (retainedFeatures φ ψ) inputs labels '' headBudget R := by
  rintro y ⟨W, hW, rfl⟩
  exact ⟨retainHead W, (retained_budget_iff W R).2 hW,
    retained_loss φ ψ inputs labels W⟩

omit [Fintype K] [Fintype A] in
/-- Appending bounded terminal features preserves their common bound, so CD8's
logit bound remains R*M instead of growing with the number of coordinates (CD9). -/
theorem retained_features_bound (φ : K → X → ℝ) (ψ : A → X → ℝ) (x : X) (M : ℝ)
    (hφ : ∀ k, |φ k x| ≤ M) (hψ : ∀ a, |ψ a x| ≤ M) :
    ∀ i, |retainedFeatures φ ψ i x| ≤ M := by
  intro i
  cases i with
  | inl k => exact hφ k
  | inr a => exact hψ a

/-- A concrete nonzero head, with a nonlinear appended feature, witnesses that
retention preserves a real feasible point without replacing the loss (CD9). -/
example : retainHead (A := Unit) (fun (_ : Fin 2) (c : Bool) => if c then (1 / 2 : ℝ) else -1 / 2) ∈
    headBudget 1 := by
  apply (retained_budget_iff _ _).2
  intro c
  cases c <;> norm_num [Fin.sum_univ_two]

example := retained_features_bound (fun (_ : Unit) (x : ℝ) => x)
  (fun (_ : Unit) (x : ℝ) => x ^ 2) 2 4
  (by intro k; norm_num) (by intro a; norm_num)

variable {J H E : Type} [Fintype J] [Fintype H]

/-- Terminal coordinates from every complete path of length at most d (CD9).
The recursion adds depth d+1 while keeping all shorter feature coordinates. -/
def PrefixFeatures (J H : Type) : ℕ → Type
  | 0 => Path J 0 × H
  | d + 1 => PrefixFeatures J H d ⊕ (Path J (d + 1) × H)

instance prefixFeaturesFintype (J H : Type) [Fintype J] [Fintype H] (d : ℕ) :
    Fintype (PrefixFeatures J H d) := by
  induction d with
  | zero => exact inferInstanceAs (Fintype (Path J 0 × H))
  | succ d ih =>
    letI := ih
    exact inferInstanceAs (Fintype (PrefixFeatures J H d ⊕ (Path J (d + 1) × H)))

/-- Every retained feature is still evaluated after its own complete nonlinear
path; appending depth does not move a block onto an averaged state (CD9). -/
def prefixFeatures (B : ℕ → J → E → E) (terminal : E → H → ℝ) :
    (d : ℕ) → PrefixFeatures J H d → E → ℝ
  | 0 => headFeatures B 0 terminal
  | d + 1 => retainedFeatures (prefixFeatures B terminal d) (headFeatures B (d + 1) terminal)

/-- Exact storage count for retained prefixes. Retention keeps a geometric sum
of path counts; it does not remove the lifting's exponential cost (CD9). -/
theorem prefix_feature_card (d : ℕ) :
    Fintype.card (PrefixFeatures J H d) =
      ∑ k ∈ Finset.range (d + 1), Fintype.card J ^ k * Fintype.card H := by
  induction d with
  | zero =>
    change Fintype.card (Path J 0 × H) = _
    rw [head_feature_card]
    simp
  | succ d ih =>
    change Fintype.card (PrefixFeatures J H d ⊕ (Path J (d + 1) × H)) = _
    rw [Fintype.card_sum, ih, head_feature_card,
      Finset.sum_range_succ (fun k => Fintype.card J ^ k * Fintype.card H) (d + 1)]

/-- Every shallower bounded head embeds at the next depth with all predictions
unchanged. Consequently an optimal deeper family cannot lose old capacity,
although a numerical solver can still fail to find a good point (CD9). -/
theorem prefix_head_extension (B : ℕ → J → E → E) (terminal : E → H → ℝ)
    {d : ℕ} {R : ℝ} {W : PrefixFeatures J H d → C → ℝ} (hW : W ∈ headBudget R) :
    ∃ V : PrefixFeatures J H (d + 1) → C → ℝ, V ∈ headBudget R ∧
      ∀ x, headLogits (prefixFeatures B terminal (d + 1)) V x =
        headLogits (prefixFeatures B terminal d) W x := by
  refine ⟨retainHead W, (retained_budget_iff W R).2 hW, ?_⟩
  intro x
  exact retained_logits (prefixFeatures B terminal d) (headFeatures B (d + 1) terminal) W x

example := prefix_head_extension (C := Bool)
  (fun _ (_ : Unit) (x : ℝ) => x + x ^ 2) (fun x (_ : Unit) => x)
  (d := 1) (R := 1) (W := fun _ c => if c then 1 / 2 else -1 / 2)
  (by intro c; cases c <;> norm_num [prefix_feature_card])

example : Fintype.card (PrefixFeatures Bool (Fin 3) 2) = 21 := by
  norm_num [prefix_feature_card, Finset.sum_range_succ]

end Transformer.Grokking.FlatSpectrum.ConvexDepth
