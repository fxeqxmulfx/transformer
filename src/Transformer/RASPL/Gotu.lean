/-
# What the transformer learns and the interpolator cannot

Zhou et al. — arXiv:2310.16028v1, "What Algorithms can Transformers Learn?",
§4 ("A Case Study: Boolean AND") and Appendix D.

The experiment: train on 20-bit strings whose last five bits are always `1`,
with the boolean AND of all twenty as the label.  The trained transformer
reaches 100% accuracy on test strings that *do* vary those five bits, so it
has learned the AND of all twenty; the minimum-degree interpolator of the same
training data provably has not, since it cannot depend on a coordinate the
training set never varied.

Both halves are here: `andProg_apply` is the one-line RASP-L program the paper
gives for the task —

    def output(x): kqv(x, full(x, 0), full(x, 0), equals, default=1)

— computing the AND of every bit seen so far, and `minDeg_ne_andAll` is the
separation.  Bits are `Bool` with `true` for the bit `1`; the `±1` reading the
characters of `RASPL.Fourier` use is the standard translation of Appendix E
("we state this with the boolean hypercube identified with `{±1}^n` ... but
these statements can be translated to `{0,1}^n`").
-/

import Transformer.RASPL.Defs
import Transformer.RASPL.MinDegree

namespace Transformer
namespace RASPL

open RASP

variable {n : ℕ}

/-- The boolean AND of all `n` input bits. -/
def andAll (n : ℕ) : Cube n → ℝ := fun x => if ∀ i, x i = true then 1 else 0

/-- The paper's one-line RASP-L program for the task: attend to every earlier
position carrying the bit `0`, and report the default `1` when there is
none. -/
noncomputable def andProg (x : Cube n) : Seq n ℝ :=
  kqv x (full n false) (full n (0 : ℝ)) (fun a b => decide (a = b)) 1

/-- **The program computes the running AND.** -/
theorem andProg_apply (x : Cube n) (i : Fin n) :
    andProg x i = if ∀ j : Fin n, (j : ℕ) ≤ (i : ℕ) → x j = true then 1 else 0 := by
  have hcard : (selected (select x (full n false) (fun a b => decide (a = b))) i).card = 0
      ↔ ∀ j : Fin n, (j : ℕ) ≤ (i : ℕ) → x j = true := by
    rw [Finset.card_eq_zero, Finset.eq_empty_iff_forall_notMem]
    constructor
    · intro h j hj
      have hnot := h j
      rw [mem_selected_select] at hnot
      by_contra hx
      have hxf : x j = false := by
        cases hxj : x j
        · rfl
        · exact absurd hxj hx
      exact hnot ⟨hj, by simp [full, hxf]⟩
    · intro h j hj
      rw [mem_selected_select] at hj
      have hx := h j hj.1
      have hxf : x j = false := of_decide_eq_true hj.2
      rw [hx] at hxf
      exact Bool.noConfusion hxf
  rw [andProg, kqv, aggrMean, aggregate]
  by_cases h : ∀ j : Fin n, (j : ℕ) ≤ (i : ℕ) → x j = true
  · rw [if_pos (hcard.2 h), if_pos h]
  · rw [if_neg (fun hc => h (hcard.1 hc)), if_neg h]
    simp [full]

/-- At the last position the running AND is the AND of the whole input, so
the program's final output is exactly the task's label. -/
theorem andProg_last (x : Cube (n + 1)) : andProg x (Fin.last n) = andAll (n + 1) x := by
  rw [andProg_apply, andAll]
  by_cases h : ∀ i, x i = true
  · rw [if_pos (fun j _ => h j), if_pos h]
  · rw [if_neg, if_neg h]
    intro hall
    exact h fun j => hall j (Nat.le_of_lt_succ j.isLt)

/-- **The minimum-degree interpolator cannot be the AND.**  It is blind to
any coordinate the training set holds constant, while the AND is not. -/
theorem minDeg_ne_andAll {T : Finset (Cube n)} {g : Cube n → ℝ} {i : Fin n}
    (hmin : MinDegInterpolator T (andAll n) g) (hconst : ConstantOn T i)
    (hmem : (fun _ => true) ∈ T) : g ≠ andAll n := by
  have hdep := minDeg_not_dependsOn hmin hconst
  have h1 : g (fun _ => true) = 1 := by
    rw [hmin.1 _ hmem, andAll, if_pos (fun _ => rfl)]
  intro hg
  have h2 : g (Function.update (fun _ => true) i false) = 1 := by
    rw [apply_update_of_not_dependsOn hdep, h1]
  rw [hg, andAll, if_neg] at h2
  · norm_num at h2
  · intro hall
    have hi := hall i
    rw [Function.update_self] at hi
    exact Bool.noConfusion hi

/-- The sets of size `0`. -/
lemma filter_card_zero : (Finset.univ.filter fun S : Finset (Fin n) => S.card = 0) = {∅} := by
  ext S
  simp [Finset.card_eq_zero]

/-- The constant function `1` is the character at `∅`. -/
lemma coeff_const_one (S : Finset (Fin n)) :
    coeff (fun _ : Cube n => (1 : ℝ)) S = if S = ∅ then 1 else 0 := by
  rw [show (fun _ : Cube n => (1 : ℝ)) = chi (∅ : Finset (Fin n)) from
    funext fun x => (chi_empty x).symm, coeff_chi]

lemma levelWeight_const_one_zero : levelWeight (fun _ : Cube n => (1 : ℝ)) 0 = 1 := by
  rw [levelWeight, filter_card_zero, Finset.sum_singleton, coeff_const_one, if_pos rfl]
  norm_num

lemma levelWeight_const_one_of_ne {k : ℕ} (hk : k ≠ 0) :
    levelWeight (fun _ : Cube n => (1 : ℝ)) k = 0 := by
  refine Finset.sum_eq_zero fun S hS => ?_
  simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hS
  rw [coeff_const_one, if_neg (fun h : S = ∅ => hk (by rw [← hS, h, Finset.card_empty]))]
  ring

/-- The hypotheses of `minDeg_ne_andAll` are satisfiable.  On one bit with
the single training point `1`, the constant function `1` is a minimum-degree
interpolator of the AND, and that training point fixes the only
coordinate. -/
example : MinDegInterpolator ({fun _ => true} : Finset (Cube 1)) (andAll 1) (fun _ => 1) ∧
    ConstantOn ({fun _ => true} : Finset (Cube 1)) 0 ∧
    (fun _ => true) ∈ ({fun _ => true} : Finset (Cube 1)) := by
  refine ⟨⟨fun x hx => ?_, fun h hint => ?_⟩, ⟨true, fun x hx => by
    rw [Finset.mem_singleton.1 hx]⟩, Finset.mem_singleton_self _⟩
  · rw [Finset.mem_singleton.1 hx, andAll, if_pos (fun _ => rfl)]
  · rintro ⟨d, hlt, heq⟩
    rcases Nat.eq_zero_or_pos d with hd | hd
    · subst hd
      have h1 : levelWeight h 1 = 0 := by
        rw [heq 1 Nat.one_pos, levelWeight_const_one_of_ne Nat.one_ne_zero]
      have hc0 : coeff h {0} = 0 := by
        rw [levelWeight, show (Finset.univ.filter fun S : Finset (Fin 1) => S.card = 1)
          = {{0}} from by decide, Finset.sum_singleton] at h1
        exact pow_eq_zero_iff two_ne_zero |>.1 h1
      have hconst : ∀ x : Cube 1, h x = coeff h ∅ := by
        intro x
        rw [← sum_coeff_mul_chi h x, show (Finset.univ : Finset (Finset (Fin 1)))
          = {∅, {0}} from by decide, Finset.sum_pair (by decide), chi_empty, hc0]
        ring
      have hone : coeff h ∅ = 1 := by
        rw [← hconst (fun _ => true), hint _ (Finset.mem_singleton_self _), andAll,
          if_pos (fun _ => rfl)]
      rw [levelWeight, filter_card_zero, Finset.sum_singleton, hone,
        levelWeight_const_one_zero] at hlt
      norm_num at hlt
    · rw [levelWeight_const_one_of_ne hd.ne'] at hlt
      have hnn : (0 : ℝ) ≤ levelWeight h d := Finset.sum_nonneg fun S _ => sq_nonneg _
      linarith

end RASPL
end Transformer
