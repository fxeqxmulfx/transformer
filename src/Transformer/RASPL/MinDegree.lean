/-
# The minimum-degree interpolator ignores the coordinates its train set fixes

Zhou et al. — arXiv:2310.16028v1, "What Algorithms can Transformers Learn?",
Lemma 4.1 (`lem:mindegree`) and its proof, which is the one-line consequence
of Lemma E.1: "if an interpolator depends on bits which are always constant
on the train set `S`, then its degree-profile could be reduced without
changing its output on `S`, and thus it cannot be a min-degree-interpolator."

This is the formal side of §4's experiment.  The trained transformer computes
the boolean AND of *all* its input bits, including the ones the training set
never varied; the minimum-degree interpolator provably cannot, which is what
`RASPL.Gotu` draws out of this file.
-/

import Transformer.RASPL.Degree

namespace Transformer
namespace RASPL

variable {n : ℕ}

/-- `g` agrees with `f` on the train set `T`. -/
def InterpolatesOn (T : Finset (Cube n)) (f g : Cube n → ℝ) : Prop := ∀ x ∈ T, g x = f x

/-- `g` is a minimum-degree interpolator of `f` on `T`: it interpolates, and
no interpolator has a strictly smaller degree profile.  This is the `argmin`
of `lem:mindegree`. -/
def MinDegInterpolator (T : Finset (Cube n)) (f g : Cube n → ℝ) : Prop :=
  InterpolatesOn T f g ∧ ∀ h : Cube n → ℝ, InterpolatesOn T f h → ¬ DegPLt h g

/-- The train set never varies coordinate `i`: its projection to `i` is a
singleton. -/
def ConstantOn (T : Finset (Cube n)) (i : Fin n) : Prop := ∃ b : Bool, ∀ x ∈ T, x i = b

/-- **Lemma 4.1.**  A minimum-degree interpolator does not depend on a
coordinate its train set holds constant: restricting that coordinate to the
constant value still interpolates, and strictly lowers the degree profile. -/
theorem minDeg_not_dependsOn {T : Finset (Cube n)} {f g : Cube n → ℝ} {i : Fin n}
    (hmin : MinDegInterpolator T f g) (hconst : ConstantOn T i) : ¬ DependsOn g i := by
  intro hdep
  obtain ⟨b, hb⟩ := hconst
  refine hmin.2 (restrict i b g) (fun x hx => ?_) (degP_restrict_lt hdep b)
  rw [restrict, ← hb x hx, Function.update_eq_self, hmin.1 x hx]

/-- Not depending on a coordinate means not seeing it at all. -/
lemma apply_update_of_not_dependsOn {g : Cube n → ℝ} {i : Fin n} (h : ¬ DependsOn g i)
    (x : Cube n) (b : Bool) : g (Function.update x i b) = g x := by
  have key : ∀ y : Cube n, g (Function.update y i false) = g (Function.update y i true) := by
    intro y
    by_contra hy
    exact h ⟨y, hy⟩
  have hx : g x = g (Function.update x i (x i)) := by rw [Function.update_eq_self]
  cases b with
  | false =>
      cases hxi : x i with
      | false => rw [hx, hxi]
      | true => rw [hx, hxi, ← key x]
  | true =>
      cases hxi : x i with
      | false => rw [hx, hxi, key x]
      | true => rw [hx, hxi]

/-- And so a function blind to every coordinate of `I` is determined by the
coordinates outside `I` — the conclusion as `lem:mindegree` states it. -/
theorem eq_of_forall_not_dependsOn {g : Cube n → ℝ} {I : Finset (Fin n)}
    (h : ∀ i ∈ I, ¬ DependsOn g i) {x y : Cube n} (hxy : ∀ j, j ∉ I → x j = y j) :
    g x = g y := by
  classical
  induction I using Finset.induction generalizing x with
  | empty =>
      have : x = y := funext fun j => hxy j (Finset.notMem_empty j)
      rw [this]
  | insert i I hi ih =>
      have hstep : g x = g (Function.update x i (y i)) :=
        (apply_update_of_not_dependsOn (h i (Finset.mem_insert_self i I)) x (y i)).symm
      refine hstep.trans (ih (fun j hj => h j (Finset.mem_insert_of_mem hj)) fun j hj => ?_)
      by_cases hji : j = i
      · subst hji
        exact Function.update_self _ _ _
      · rw [Function.update_of_ne hji]
        exact hxy j (fun hmem => (Finset.mem_insert.1 hmem).elim hji (fun h' => hj h'))

/-- **Lemma 4.1, as stated.**  For `I` the coordinates the train set holds
constant, the minimum-degree interpolator satisfies
`x_j = y_j ∀ j ∉ I → g(x) = g(y)`. -/
theorem minDeg_eq_of_constant {T : Finset (Cube n)} {f g : Cube n → ℝ} {I : Finset (Fin n)}
    (hmin : MinDegInterpolator T f g) (hconst : ∀ i ∈ I, ConstantOn T i) {x y : Cube n}
    (hxy : ∀ j, j ∉ I → x j = y j) : g x = g y :=
  eq_of_forall_not_dependsOn (fun i hi => minDeg_not_dependsOn hmin (hconst i hi)) hxy

/-- The weights of a function are never negative, so nothing has a smaller
degree profile than the zero function. -/
lemma not_degPLt_zero (h : Cube n → ℝ) : ¬ DegPLt h (fun _ => 0) := by
  intro ⟨d, hlt, _⟩
  have hzero : levelWeight (fun _ : Cube n => (0 : ℝ)) d = 0 := by
    refine Finset.sum_eq_zero fun S _ => ?_
    rw [show coeff (fun _ : Cube n => (0 : ℝ)) S = 0 by simp [coeff]]
    ring
  have : 0 ≤ levelWeight h d := Finset.sum_nonneg fun S _ => sq_nonneg _
  rw [hzero] at hlt
  linarith

/-- The hypotheses are satisfiable, and on a nonempty train set: a single
training point holds every coordinate constant, and the zero function is a
minimum-degree interpolator of itself on it. -/
example : MinDegInterpolator {(fun _ => true : Cube 1)} (fun _ => 0) (fun _ => 0) ∧
    ConstantOn {(fun _ => true : Cube 1)} 0 :=
  ⟨⟨fun _ _ => rfl, fun h _ => not_degPLt_zero h⟩, ⟨true, fun x hx => by rw [Finset.mem_singleton.1 hx]⟩⟩

end RASPL
end Transformer
