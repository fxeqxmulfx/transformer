import Transformer.GPTMini.Sparsemax.NearestPrototypeCodes

/-!
# Conditional unseen-query guarantees for sparse nearest memory

Derived architecture for arXiv:1602.02068v2, Eq. (1), followed by the
common value operation at `73f8a0b`. Actual decoded outputs evaluate the
global output table at the nearest registered observation. If prototype
outputs have error at most epsilon and the target is L-Lipschitz in the
data distance, unseen error is at most epsilon plus L times nearest distance.
A data cover of radius delta gives epsilon plus L times delta uniformly.

Regularity and coverage are explicit hypotheses with nonconstant examples.
They are not consequences of interpolation or claims about Shakespeare.
Two concrete target functions agree on every registered observation but
disagree at an unseen one, recording why an unconditional guarantee fails.
No model task loss or FFN is selected by these forward-map results.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

/-- Actual sparsemax and one common decoded value table evaluate the nearest output row.
Source: the exact memory chart after arXiv:1602.02068v2, Eq. (1). -/
theorem jointNearestForward_eq {Key : Type*} {R N D : ℕ} (cap floor : ℝ)
    (G : EmbeddingGram (N + 1)) (distance : Key → Key → ℝ)
    (prototypes : Fin (N + 1) → Key) (queries : Fin R → Key)
    (Z : Matrix (Fin (N + 1)) (Fin D) ℝ) (hf : 1 / 2 < floor)
    (hG : G ∈ memoryGramDomain N cap floor) :
    jointContextMemoryForward (nearestPrototypeCodes distance prototypes queries) (G, Z) =
      Matrix.of (fun r => Z (nearestPrototype distance prototypes (queries r))) := by
  rw [jointContextMemoryForward_eq cap floor (nearestPrototypeCodes distance prototypes queries)
    _ hf (nearestPrototypeCodes_mem _ _ _) hG]
  exact oneHotContextCodes_mul _ Z

/-- An unseen real query and nonconstant values inhabit every actual-forward premise. -/
example : jointContextMemoryForward (nearestPrototypeCodes (fun x y : ℝ => |x - y|)
    (fun j : Fin 2 => (2 * j.val : ℝ)) (fun _ : Fin 1 => (1 / 2 : ℝ)))
    (memoryIdentityGram 1, Matrix.of (fun j : Fin 2 => fun _ : Fin 1 => (2 * j.val : ℝ))) =
    Matrix.of (fun _ : Fin 1 => fun _ : Fin 1 =>
      (2 * (nearestPrototype (fun x y : ℝ => |x - y|)
        (fun j : Fin 2 => (2 * j.val : ℝ)) (1 / 2)).val : ℝ)) :=
  jointNearestForward_eq 1 (3 / 4) _ _ _ _ _ (by norm_num)
    (memoryIdentityGram_mem _ _ _ (by norm_num) (by norm_num))

/-- Prototype fit and target regularity bound the error at every nearest query.
Source: the derived nearest predictor before arXiv:1602.02068v2, Eq. (1).
Regularity concerns output targets, not latent attention labels. -/
theorem nearestPrototype_prediction_error {Key : Type*} {N D : ℕ}
    (distance : Key → Key → ℝ) (prototypes : Fin (N + 1) → Key)
    (Z : Matrix (Fin (N + 1)) (Fin D) ℝ) (target : Key → Fin D → ℝ) (epsilon L : ℝ)
    (hfit : ∀ j d, |Z j d - target (prototypes j) d| ≤ epsilon)
    (hreg : ∀ x y d, |target x d - target y d| ≤ L * distance x y)
    (query : Key) (d : Fin D) :
    |Z (nearestPrototype distance prototypes query) d - target query d| ≤
      epsilon + L * distance query (prototypes (nearestPrototype distance prototypes query)) := by
  let j := nearestPrototype distance prototypes query
  have hr := hreg query (prototypes j) d
  rw [abs_sub_comm] at hr
  exact (abs_sub_le (Z j d) (target (prototypes j) d) (target query d)).trans
    (add_le_add (hfit j d) hr)

/-- A nonconstant Lipschitz target and exact registered outputs inhabit both premises. -/
example : |(2 * (nearestPrototype (fun x y : ℝ => |x - y|)
    (fun j : Fin 2 => (2 * j.val : ℝ)) (1 / 2)).val : ℝ) - 1 / 2| ≤
    0 + 1 * |(1 / 2 : ℝ) - 2 * (nearestPrototype (fun x y : ℝ => |x - y|)
      (fun j : Fin 2 => (2 * j.val : ℝ)) (1 / 2)).val| := by
  apply nearestPrototype_prediction_error (fun x y : ℝ => |x - y|)
    (fun j : Fin 2 => (2 * j.val : ℝ))
    (Matrix.of (fun j : Fin 2 => fun _ : Fin 1 => (2 * j.val : ℝ)))
    (fun x : ℝ => fun _ : Fin 1 => x) 0 1 ?_ ?_ (1 / 2) 0
  · intro j d
    norm_num
  · intro x y d
    simp only [one_mul, le_refl]

/-- The error estimate holds for actual learned attention and its global original values.
Source: the exact decoder after arXiv:1602.02068v2, Eq. (1), and the nearest error bound. -/
theorem jointNearestForward_error {Key : Type*} {R N D : ℕ} (cap floor : ℝ)
    (G : EmbeddingGram (N + 1)) (distance : Key → Key → ℝ)
    (prototypes : Fin (N + 1) → Key) (queries : Fin R → Key)
    (Z : Matrix (Fin (N + 1)) (Fin D) ℝ) (target : Key → Fin D → ℝ) (epsilon L : ℝ)
    (hf : 1 / 2 < floor) (hG : G ∈ memoryGramDomain N cap floor)
    (hfit : ∀ j d, |Z j d - target (prototypes j) d| ≤ epsilon)
    (hreg : ∀ x y d, |target x d - target y d| ≤ L * distance x y)
    (r : Fin R) (d : Fin D) :
    |jointContextMemoryForward (nearestPrototypeCodes distance prototypes queries) (G, Z) r d -
      target (queries r) d| ≤
      epsilon + L * distance (queries r)
        (prototypes (nearestPrototype distance prototypes (queries r))) := by
  rw [jointNearestForward_eq cap floor G distance prototypes queries Z hf hG]
  exact nearestPrototype_prediction_error distance prototypes Z target epsilon L hfit hreg _ _

/-- A real learned-memory block, nonconstant target and unseen query satisfy all premises. -/
example : |jointContextMemoryForward (nearestPrototypeCodes (fun x y : ℝ => |x - y|)
    (fun j : Fin 2 => (2 * j.val : ℝ)) (fun _ : Fin 1 => (1 / 2 : ℝ)))
    (memoryIdentityGram 1, Matrix.of (fun j : Fin 2 => fun _ : Fin 1 => (2 * j.val : ℝ))) 0 0 -
    1 / 2| ≤ 0 + 1 * |(1 / 2 : ℝ) - 2 * (nearestPrototype (fun x y : ℝ => |x - y|)
      (fun j : Fin 2 => (2 * j.val : ℝ)) (1 / 2)).val| := by
  apply jointNearestForward_error 1 (3 / 4) _ _ _ _ _ (fun x : ℝ => fun _ : Fin 1 => x) 0 1
    (by norm_num) (memoryIdentityGram_mem _ _ _ (by norm_num) (by norm_num))
  · intro j d
    norm_num
  · intro x y d
    simp only [one_mul, le_refl]

/-- A covered query set has uniform error epsilon plus L times the data-cover radius.
Source: the derived actual nearest predictor after arXiv:1602.02068v2, Eq. (1).
Coverage and regularity are explicit assumptions, independent of learned routing targets. -/
theorem jointNearestForward_cover_error {Key : Type*} {R N D : ℕ} (cap floor : ℝ)
    (G : EmbeddingGram (N + 1)) (distance : Key → Key → ℝ)
    (prototypes : Fin (N + 1) → Key) (queries : Fin R → Key)
    (Z : Matrix (Fin (N + 1)) (Fin D) ℝ) (target : Key → Fin D → ℝ) (epsilon L radius : ℝ)
    (hf : 1 / 2 < floor) (hG : G ∈ memoryGramDomain N cap floor)
    (hfit : ∀ j d, |Z j d - target (prototypes j) d| ≤ epsilon)
    (hreg : ∀ x y d, |target x d - target y d| ≤ L * distance x y) (hL : 0 ≤ L)
    (hcover : ∀ r, ∃ j, distance (queries r) (prototypes j) ≤ radius)
    (r : Fin R) (d : Fin D) :
    |jointContextMemoryForward (nearestPrototypeCodes distance prototypes queries) (G, Z) r d -
      target (queries r) d| ≤ epsilon + L * radius := by
  obtain ⟨j, hj⟩ := hcover r
  have hn := (nearestPrototype_min distance prototypes (queries r) j).trans hj
  exact (jointNearestForward_error cap floor G distance prototypes queries Z target epsilon L
    hf hG hfit hreg r d).trans (add_le_add (le_refl epsilon) (mul_le_mul_of_nonneg_left hn hL))

/-- Imperfect registered values and an unseen query inhabit the uniform generalization bound. -/
example : |jointContextMemoryForward (nearestPrototypeCodes (fun x y : ℝ => |x - y|)
    (fun j : Fin 2 => (2 * j.val : ℝ)) (fun _ : Fin 1 => (1 / 2 : ℝ)))
    (memoryIdentityGram 1, Matrix.of (fun j : Fin 2 => fun _ : Fin 1 => (2 * j.val + 1 / 10 : ℝ)))
      0 0 - 1 / 2| ≤ 3 / 5 := by
  calc
    _ ≤ (1 / 10 : ℝ) + 1 * (1 / 2) := by
      apply jointNearestForward_cover_error 1 (3 / 4) _ _ _ _ _
        (fun x : ℝ => fun _ : Fin 1 => x) (1 / 10) 1 (1 / 2) (by norm_num)
        (memoryIdentityGram_mem _ _ _ (by norm_num) (by norm_num))
      · intro j d
        change |(2 * (j.val : ℝ) + 1 / 10) - 2 * (j.val : ℝ)| ≤ 1 / 10
        ring_nf
        norm_num
      · intro x y d
        simp only [one_mul, le_refl]
      · norm_num
      · intro r
        exact ⟨0, by norm_num⟩
    _ = _ := by norm_num

/-- One nonconstant target on a three-observation universe.
Source: a counterexample to unconditional generalization from finite fitting
in the derived arXiv:1602.02068v2, Eq. (1) architecture. -/
def unseenTargetBase (q : Fin 3) : ℝ := q.val

/-- Another nonconstant target, changed only at the unregistered observation.
Source: the same arXiv:1602.02068v2, Eq. (1) finite-data counterexample. -/
def unseenTargetAlternative (q : Fin 3) : ℝ := q.val + if q = 2 then 1 else 0

/-- Complete agreement on registered observations does not determine the unseen target.
Source: counterexample to that unconditional claim for arXiv:1602.02068v2, Eq. (1) memory. -/
theorem unseenTarget_registered_agreement :
    (∀ j : Fin 2, unseenTargetBase j.castSucc = unseenTargetAlternative j.castSucc) ∧
      unseenTargetBase 2 ≠ unseenTargetAlternative 2 := by
  constructor
  · intro j
    fin_cases j <;> norm_num [unseenTargetBase, unseenTargetAlternative]
  · norm_num [unseenTargetBase, unseenTargetAlternative]

/-- Every prediction has unseen error at least one half for one indistinguishable target.
Source: the explicit finite-data counterexample for arXiv:1602.02068v2, Eq. (1).
This applies even when both registered observations are fit exactly. -/
theorem unseenTarget_prediction_lowerBound (prediction : Fin 3 → ℝ) :
    (1 / 2 : ℝ) ≤ |prediction 2 - unseenTargetBase 2| ∨
      (1 / 2 : ℝ) ≤ |prediction 2 - unseenTargetAlternative 2| := by
  have hleft := le_abs_self (prediction 2 - 2)
  have hright := neg_le_abs (prediction 2 - 3)
  by_cases h : prediction 2 ≤ 5 / 2
  · right
    norm_num [unseenTargetAlternative]
    linarith
  · left
    norm_num [unseenTargetBase]
    linarith

end Transformer.GPTMini.Sparsemax
