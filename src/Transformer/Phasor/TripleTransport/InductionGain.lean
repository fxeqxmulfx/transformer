module

public import Transformer.Phasor.TripleTransport.DenseInduction
public import Transformer.Imbalance.Section3_Derivatives

/-!
# Learning the gain of a dense induction read

Source: the dense induction experiment and arXiv:2402.19449v2, §3.2,
equation (3). The gate receives a gradient through the usual
tied classifier and cross-entropy over supervised positions.
-/

@[expose] public section

namespace Transformer.Phasor.TripleTransport.DenseInduction
open scoped BigOperators

/-- The classifier is tied to the input embedding. At a fixed prefix,
all non-structural scores are z and the retrieved vector contributes v.
With strictly positive softmax mass on at least one inferior response,
ordinary cross-entropy has a strictly negative gain derivative if all
classes score no higher than the target along that direction. -/
theorem gain_descent {n : ℕ} (z v : Fin n → ℝ) (y : Fin n) (g : ℝ)
    (top : ∀ j, v j ≤ v y) (rival : ∃ j, v j < v y) :
    deriv (fun s => Imbalance.crossEntropy (fun j => z j + s * v j) y) g < 0 := by
  let q := fun j => z j + g * v j
  let p := Perspective.softmaxWeight q
  have hn : 0 < n := by have := y.isLt; omega
  have sum_one : ∑ j, p j = 1 := Perspective.sum_softmaxWeight hn q
  have positive (j : Fin n) : 0 < p j := by
    exact div_pos (Real.exp_pos _) (Perspective.softmaxPartition_pos hn q)
  have bound : ∑ j, p j * v j < ∑ j, p j * v y := by
    apply Finset.sum_lt_sum (fun j _ => mul_le_mul_of_nonneg_left (top j) (positive j).le)
    obtain ⟨j,less⟩ := rival
    exact ⟨j,Finset.mem_univ _,mul_lt_mul_of_pos_left less (positive j)⟩
  rw [← Finset.sum_mul,sum_one,one_mul] at bound
  rw [(Imbalance.hasDerivAt_crossEntropy_line z v y g).deriv]
  exact sub_neg.mpr bound

/-- Dot-product scores of an arbitrary tied output matrix applied to an
inductive read. The lookup, gates and forward calculation need no target. -/
theorem tied_gain_descent {n d : ℕ} (E : Fin n → Fin d → ℝ) (context value : Fin d → ℝ)
    (y : Fin n) (g : ℝ) (top : ∀ j, product (E j) value ≤ product (E y) value)
    (rival : ∃ j, product (E j) value < product (E y) value) :
    deriv (fun s => Imbalance.crossEntropy
      (fun j => product (E j) (fun k => context k + s * value k)) y) g < 0 := by
  have affine (j : Fin n) (s : ℝ) :
      product (E j) (fun k => context k + s * value k) =
        product (E j) context + s * product (E j) value := by
    unfold product
    simp only [mul_add,Finset.sum_add_distrib,Finset.mul_sum]
    congr 1
    exact Finset.sum_congr rfl (fun _ _ => by ring)
  simp_rw [affine]
  exact gain_descent _ _ y g top rival

example (z : Fin 2 → ℝ) (g : ℝ) :
    deriv (fun s => Imbalance.crossEntropy (fun j => z j + s * (![1,0] : Fin 2 → ℝ) j) 0) g < 0 := by
  apply gain_descent
  · intro j; fin_cases j <;> norm_num
  · exact ⟨1,by norm_num⟩

example (g : ℝ) : deriv (fun s => Imbalance.crossEntropy
    (fun j : Fin 2 => product (axis j) (fun k => (0 : ℝ) + s * axis 0 k)) 0) g < 0 := by
  apply tied_gain_descent
  · intro j; fin_cases j <;> norm_num [product,axis,Fin.sum_univ_two]
  · exact ⟨1,by norm_num [product,axis,Fin.sum_univ_two]⟩

/-- A shared gain receives a strict descent direction from the mean loss
when every supervised row meets the same per-row alignment condition.
Only supervised positions are indexed in this average. -/
theorem mean_gain_descent {b n : ℕ} [NeZero b] (z v : Fin b → Fin n → ℝ)
    (target : Fin b → Fin n) (g : ℝ)
    (top : ∀ i j, v i j ≤ v i (target i))
    (rival : ∀ i, ∃ j, v i j < v i (target i)) :
    deriv (fun s => (∑ i, Imbalance.crossEntropy (fun j => z i j + s * v i j) (target i)) /
      (b : ℝ)) g < 0 := by
  let loss := fun (i : Fin b) (s : ℝ) =>
    Imbalance.crossEntropy (fun j => z i j + s * v i j) (target i)
  have each (i : Fin b) : HasDerivAt (loss i) (deriv (loss i) g) g :=
    (Imbalance.hasDerivAt_crossEntropy_line (z i) (v i) (target i) g).differentiableAt.hasDerivAt
  have total : HasDerivAt (fun s => ∑ i, loss i s) (∑ i, deriv (loss i) g) g :=
    HasDerivAt.fun_sum (fun i _ => each i)
  rw [(total.div_const (b : ℝ)).deriv]
  apply div_neg_of_neg_of_pos
  · exact Finset.sum_neg (fun i _ => gain_descent (z i) (v i) (target i) g (top i) (rival i))
      Finset.univ_nonempty
  · exact_mod_cast NeZero.pos b

example (z : Fin 3 → Fin 2 → ℝ) (g : ℝ) :
    deriv (fun s => (∑ i, Imbalance.crossEntropy
      (fun j => z i j + s * (![1,0] : Fin 2 → ℝ) j) 0) / 3) g < 0 := by
  apply mean_gain_descent z (fun _ => ![1,0]) (fun _ => 0) g
  · intro i j
    fin_cases j <;> norm_num
  · exact fun _ => ⟨1,by norm_num⟩

end Transformer.Phasor.TripleTransport.DenseInduction
