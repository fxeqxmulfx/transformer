module

public import Transformer.Grokking.Convex.CoupledMoments.Schatten.Shift
public import Transformer.Grokking.FlatSpectrum.Training.Logits

/-!
# The CE gradient moves with the class shift

Source: `review/convex/moments.md`, CM121. The learner of `convex_stress_schatten` reads the
gradient of observed CE, softmax minus one-hot in each observed cell (`ceGrad`). It is a gradient
of the loss (`loss_first_order`). The class shift moves the softmax (`probability_shift`) and the
one-hot target (`targetLogits_shift`), so the gradient for the labels `y + u i + v j` at the
shifted table is the shifted gradient for `y` (`ceGrad_shift`). The same holds for the retained
gradient of a minibatch step, which refreshes the selected cells and keeps the others
(`refresh_shift`).
-/

@[expose] public section

namespace Transformer.Grokking.Convex.CoupledMoments.Schatten

open Finset ConvexRecall
open Landscape.Neural (crossEntropy)
open Geometry (probability)
open FlatSpectrum.Training (crossEntropy_first_order)

variable {m n C S : Type*} [AddCommGroup C]

/-- The one-hot target moves with the class. -/
lemma targetLogits_shift [DecidableEq C] (y a b c : C) :
    targetLogits (y + a + b) c = targetLogits y (c - a - b) := by
  simp only [targetLogits, sub_sub, sub_eq_iff_eq_add, add_assoc]

variable [Fintype C]

/-- The softmax of a shifted row is the shifted softmax. -/
theorem probability_shift (u : m → C) (v : n → C) (P : Table m n C) (i : m) (j : n) (c : C) :
    probability (shift u v P i j) c = probability (P i j) (c - u i - v j) := by
  simp only [probability, shift, sub_sub]
  rw [sum_shift (fun d => Real.exp (P i j d)) (u i + v j)]

variable [DecidableEq m] [DecidableEq n] [DecidableEq C]

/-- Softmax minus one-hot, summed over the observations `sel` of each cell. -/
noncomputable def ceGrad (cell : S → m × n) (labels : S → C) (sel : Finset S)
    (P : Table m n C) : Table m n C :=
  fun i j c => ∑ s ∈ sel with cell s = (i, j), (probability (P i j) c - targetLogits (labels s) c)

/-- **The CE gradient moves with the shift.** -/
theorem ceGrad_shift (cell : S → m × n) (labels : S → C) (sel : Finset S) (u : m → C)
    (v : n → C) (P : Table m n C) :
    ceGrad cell (fun s => labels s + u (cell s).1 + v (cell s).2) sel (shift u v P) =
      shift u v (ceGrad cell labels sel P) := by
  funext i j c
  refine sum_congr rfl fun s hs => ?_
  have hcell := (mem_filter.1 hs).2
  rw [probability_shift]
  simp only [hcell, targetLogits_shift]

/-- The retained gradient of a minibatch step: a cell that a selected observation covers gets the
gradient at `x`, every other cell keeps `M`. -/
noncomputable def refresh (cell : S → m × n) (labels : S → C) (sel : Finset S)
    (x M : Table m n C) : Table m n C :=
  fun i j c =>
    if (sel.filter fun s => cell s = (i, j)).Nonempty then ceGrad cell labels sel x i j c
    else M i j c

/-- **The retained gradient moves with the shift.** -/
theorem refresh_shift (cell : S → m × n) (labels : S → C) (sel : Finset S) (u : m → C)
    (v : n → C) (x M : Table m n C) :
    refresh cell (fun s => labels s + u (cell s).1 + v (cell s).2) sel (shift u v x)
        (shift u v M) = shift u v (refresh cell labels sel x M) := by
  funext i j c
  simp only [refresh, ceGrad_shift]
  rfl

omit [AddCommGroup C] in
/-- The inner product with the CE gradient of one observation. -/
lemma sum_grad_mul (z d : C → ℝ) (y : C) :
    ∑ c, (probability z c - targetLogits y c) * d c = ∑ c, probability z c * d c - d y := by
  simp only [sub_mul, sum_sub_distrib, targetLogits, ite_mul, one_mul, zero_mul, sum_ite_eq',
    mem_univ, ↓reduceIte]

omit [AddCommGroup C] in
/-- **`ceGrad` is a gradient of observed CE**: the first-order inequality of the convex loss. -/
theorem loss_first_order [Fintype m] [Fintype n] [Fintype S] (cell : S → m × n) (labels : S → C)
    (P Q : Table m n C) :
    loss cell labels P + ∑ i, ∑ j, ∑ c, ceGrad cell labels univ P i j c * (Q i j c - P i j c) ≤
      loss cell labels Q := by
  have hpair : ∑ i, ∑ j, ∑ c, ceGrad cell labels univ P i j c * (Q i j c - P i j c) =
      ∑ s, (∑ c, probability (P (cell s).1 (cell s).2) c *
          (Q (cell s).1 (cell s).2 c - P (cell s).1 (cell s).2 c) -
        (Q (cell s).1 (cell s).2 (labels s) - P (cell s).1 (cell s).2 (labels s))) := by
    rw [← Fintype.sum_prod_type' (f := fun i j => ∑ c,
      ceGrad cell labels univ P i j c * (Q i j c - P i j c))]
    rw [← sum_fiberwise (s := univ) (g := cell)]
    refine sum_congr rfl fun ij _ => ?_
    simp only [ceGrad, sum_mul]
    rw [sum_comm]
    refine sum_congr rfl fun s hs => ?_
    rw [(mem_filter.1 hs).2, ← sum_grad_mul]
  rw [hpair, loss, loss, ← sum_add_distrib]
  exact sum_le_sum fun s _ => crossEntropy_first_order _ _ _

end Transformer.Grokking.Convex.CoupledMoments.Schatten
