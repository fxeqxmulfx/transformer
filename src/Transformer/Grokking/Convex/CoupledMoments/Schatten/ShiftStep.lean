module

public import Transformer.Grokking.Convex.CoupledMoments.Schatten.ShiftGradient
public import Transformer.Grokking.Convex.CoupledMoments.Schatten.SplitStep

/-!
# Every step of the splitting learner commutes with the class shift

Source: `review/convex/moments.md`, CM121. The learner of `convex_stress_schatten` repeats the
splitting step of `SplitStep.lean`. The class shift satisfies `SplitStep.map` between the labels
`y` and the labels `y + u i + v j`: it keeps the squared distance (`sqDist_shift`) and maps the
probability rows, the observed one-hot rows and the moment ball onto their counterparts
(`shift_image_rows`, `shift_image_observed`, `shift_image_momentBall`). So the full-batch,
minibatch and forward-only steps commute with it (`fullStep_shift`, `miniStep_shift`,
`forwardStep_shift`), and so does the final mixture with uniform mass (`shift_blend`).
A step of one problem maps to a step of the other. That nearest points are unique, for closed
convex sets, is the Hilbert projection theorem, not restated here.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.CoupledMoments.Schatten

open Finset FlatSpectrum ConvexRecall
open scoped Matrix.Norms.L2Operator

variable {m n C S F : Type*}

/-- The squared Euclidean distance between two tables. -/
def sqDist [Fintype m] [Fintype n] [Fintype C] (P Q : Table m n C) : ℝ :=
  ∑ i, ∑ j, ∑ c, (P i j c - Q i j c) ^ 2

/-- A point of `K` is its own nearest point. -/
lemma isNearest_self [Fintype m] [Fintype n] [Fintype C] {K : Set (Table m n C)}
    {z : Table m n C} (hz : z ∈ K) : IsNearest sqDist K z z := by
  refine ⟨hz, fun w _ => ?_⟩
  have h0 : sqDist z z = 0 := by simp [sqDist]
  rw [h0]
  exact sum_nonneg fun i _ => sum_nonneg fun j _ => sum_nonneg fun c _ => sq_nonneg _

example := isNearest_self (K := {P | Rows P}) (z := pointMass fun (_ _ : Fin 1) => (0 : Fin 2))
  (rows_pointMass _)

/-- A common point of both sets is a fixed point of the step without a gradient term. -/
lemma splitStep_self [Fintype m] [Fintype n] [AddCommGroup C] [Fintype C]
    {A B : Set (Table m n C)} {g : Table m n C → Table m n C} {z : Table m n C} (hA : z ∈ A)
    (hB : z ∈ B) : SplitStep sqDist A B 0 g z z z := by
  refine ⟨isNearest_self hA, z, ?_, by abel⟩
  have h : (2 : ℝ) • z - z - (0 : ℝ) • g z = z := by
    rw [two_smul, zero_smul, sub_zero, add_sub_cancel_right]
  rw [h]
  exact isNearest_self hB

example {K : Type*} [AddCommGroup K] [Fintype K] [DecidableEq K] (a : K) :=
  splitStep_self (A := {P | Rows P}) (B := Set.univ) (g := id)
    (z := pointMass fun (_ _ : Fin 1) => a) (rows_pointMass _) trivial

variable [AddCommGroup C]

/-- The class shift as a linear map of tables. -/
def shiftLinear (u : m → C) (v : n → C) : Table m n C →ₗ[ℝ] Table m n C where
  toFun := shift u v
  map_add' _ _ := rfl
  map_smul' _ _ := rfl

lemma shift_shift_neg (u : m → C) (v : n → C) (Q : Table m n C) :
    shift u v (shift (-u) (-v) Q) = Q := by
  simpa using shift_neg_shift (-u) (-v) Q

/-- The shift maps `K` onto `K'` when membership corresponds. -/
lemma shift_image_eq {K K' : Set (Table m n C)} {u : m → C} {v : n → C}
    (h : ∀ P, shift u v P ∈ K' ↔ P ∈ K) : shift u v '' K = K' := by
  ext Q
  refine ⟨?_, fun hQ => ⟨shift (-u) (-v) Q, (h _).1 ?_, shift_shift_neg u v Q⟩⟩
  · rintro ⟨P, hP, rfl⟩
    exact (h P).2 hP
  · rwa [shift_shift_neg]

example {K : Type*} [AddCommGroup K] (u v : Fin 1 → K) :
    shift u v '' Set.univ = Set.univ :=
  shift_image_eq fun _ => by simp

variable [Fintype C]

theorem shift_image_rows (u : m → C) (v : n → C) :
    shift u v '' {P : Table m n C | Rows P} = {P | Rows P} :=
  shift_image_eq fun P => rows_shift_iff u v P

/-- The shift commutes with the common mixture of CM4. -/
theorem shift_blend (u : m → C) (v : n → C) (α : ℝ) (P : Table m n C) :
    shift u v (blend α P) = blend α (shift u v P) := rfl

/-- The set `A` of the forward-only mode: probability rows with one-hot observed labels. -/
def ObservedRows [DecidableEq C] (cell : S → m × n) (labels : S → C) : Set (Table m n C) :=
  {P | Rows P ∧ ∀ s, P (cell s).1 (cell s).2 = targetLogits (labels s)}

theorem shift_image_observed [DecidableEq C] (cell : S → m × n) (labels : S → C) (u : m → C)
    (v : n → C) :
    shift u v '' ObservedRows cell labels =
      ObservedRows cell fun s => labels s + u (cell s).1 + v (cell s).2 :=
  shift_image_eq fun P => and_congr (rows_shift_iff u v P) (shift_fit_iff cell labels u v P)

variable [Fintype m] [Fintype n]

theorem sqDist_shift (u : m → C) (v : n → C) (P Q : Table m n C) :
    sqDist (shift u v P) (shift u v Q) = sqDist P Q := by
  refine sum_congr rfl fun i _ => sum_congr rfl fun j _ => ?_
  simp only [shift, sub_sub]
  exact sum_shift (fun c => (P i j c - Q i j c) ^ 2) (u i + v j)

variable [DecidableEq n]

/-- The set `B`: operator and nuclear budgets of the character codes, with no row condition. -/
def MomentBall (χ : F → AddChar C ℂ) (R B : F → ℝ) : Set (Table m n C) :=
  {P | ∀ f, ‖moment (χ f) P‖ ≤ R f ∧ nuclearNorm (moment (χ f) P) ≤ B f}

/-- A table lies in the ball of its own norms. -/
lemma mem_momentBall_self (χ : F → AddChar C ℂ) (z : Table m n C) :
    z ∈ MomentBall χ (fun f => ‖moment (χ f) z‖) fun f => nuclearNorm (moment (χ f) z) :=
  fun _ => ⟨le_rfl, le_rfl⟩

variable [DecidableEq m]

theorem shift_image_momentBall (χ : F → AddChar C ℂ) (R B : F → ℝ) (u : m → C) (v : n → C) :
    shift u v '' (MomentBall χ R B : Set (Table m n C)) = MomentBall χ R B :=
  shift_image_eq fun P => by
    simp only [MomentBall, Set.mem_ofPred_eq, norm_moment_shift, nuclearNorm_moment_shift]

variable [DecidableEq C]

/-- **A full-batch step commutes with the shift.** -/
theorem fullStep_shift (cell : S → m × n) (labels : S → C) (sel : Finset S)
    (χ : F → AddChar C ℂ) (R B : F → ℝ) (u : m → C) (v : n → C) {γ : ℝ} {z x z' : Table m n C}
    (h : SplitStep sqDist {P | Rows P} (MomentBall χ R B) γ (ceGrad cell labels sel) z x z') :
    SplitStep sqDist {P | Rows P} (MomentBall χ R B) γ
      (ceGrad cell (fun s => labels s + u (cell s).1 + v (cell s).2) sel)
      (shift u v z) (shift u v x) (shift u v z') :=
  h.map (g := ceGrad cell labels sel) (shiftLinear u v) (sqDist_shift u v) (shift_image_rows u v)
    (shift_image_momentBall χ R B u v) (ceGrad_shift cell labels sel u v)

/-- **A minibatch step commutes with the shift**; its new retained gradient at `x` moves by
`refresh_shift`. -/
theorem miniStep_shift (cell : S → m × n) (labels : S → C) (sel : Finset S)
    (χ : F → AddChar C ℂ) (R B : F → ℝ) (u : m → C) (v : n → C) {γ : ℝ}
    {z x z' M : Table m n C}
    (h : SplitStep sqDist {P | Rows P} (MomentBall χ R B) γ
      (fun w => refresh cell labels sel w M) z x z') :
    SplitStep sqDist {P | Rows P} (MomentBall χ R B) γ
      (fun w => refresh cell (fun s => labels s + u (cell s).1 + v (cell s).2) sel w
        (shift u v M))
      (shift u v z) (shift u v x) (shift u v z') :=
  h.map (g := fun w => refresh cell labels sel w M) (shiftLinear u v) (sqDist_shift u v)
    (shift_image_rows u v) (shift_image_momentBall χ R B u v)
    fun w => refresh_shift cell labels sel u v w M

/-- **A forward-only step commutes with the shift.** -/
theorem forwardStep_shift (cell : S → m × n) (labels : S → C) (χ : F → AddChar C ℂ)
    (R B : F → ℝ) (u : m → C) (v : n → C) {γ : ℝ} {z x z' : Table m n C}
    (h : SplitStep sqDist (ObservedRows cell labels) (MomentBall χ R B) γ 0 z x z') :
    SplitStep sqDist (ObservedRows cell fun s => labels s + u (cell s).1 + v (cell s).2)
      (MomentBall χ R B) γ 0 (shift u v z) (shift u v x) (shift u v z') :=
  h.map (shiftLinear u v) (sqDist_shift u v) (shift_image_observed cell labels u v)
    (shift_image_momentBall χ R B u v) fun _ => (map_zero (shiftLinear u v)).symm

section Examples

variable {K : Type*} [AddCommGroup K] [Fintype K] [DecidableEq K]

example (χ : AddChar K ℂ) (a b : K) :=
  fullStep_shift (fun _ : Fin 1 => ((0 : Fin 1), (0 : Fin 1))) (fun _ => a) univ
    (fun _ : Unit => χ) _ _ (fun _ => b) (fun _ => b)
    (splitStep_self (z := pointMass fun _ _ => a) (rows_pointMass _) (mem_momentBall_self _ _))

example (χ : AddChar K ℂ) (a b : K) :=
  miniStep_shift (M := 0) (fun _ : Fin 1 => ((0 : Fin 1), (0 : Fin 1))) (fun _ => a) univ
    (fun _ : Unit => χ) _ _ (fun _ => b) (fun _ => b)
    (splitStep_self (z := pointMass fun _ _ => a) (rows_pointMass _) (mem_momentBall_self _ _))

example (χ : AddChar K ℂ) (a b : K) :=
  forwardStep_shift (fun _ : Fin 1 => ((0 : Fin 1), (0 : Fin 1))) (fun _ => a)
    (fun _ : Unit => χ) _ _ (fun _ => b) (fun _ => b)
    (splitStep_self (z := pointMass fun _ _ => a) ⟨rows_pointMass _, fun _ => rfl⟩
      (mem_momentBall_self _ _))

end Examples

end Transformer.Grokking.Convex.CoupledMoments.Schatten
