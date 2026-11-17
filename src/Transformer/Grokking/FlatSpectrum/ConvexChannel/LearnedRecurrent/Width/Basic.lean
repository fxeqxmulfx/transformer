module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Pair.Basic

/-!
# A polynomial obstruction shared by every plain two-state block

Source: `review/flat_spectrum.md`, CM110. The squared commutator of two full
complex two-by-two matrices is scalar. The resulting eight-word identity
survives every finite mixture, regardless of learned values or norm budgets.
It applies to shared plain transitions, not position-dependent rotary factors.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Width

variable {O : Type*}

/-- CM110: the ordinary product of both full complex transition matrices. -/
noncomputable def multiply (m n : Pair.Matrix) : Pair.Matrix :=
  ((m.1.1 * n.1.1 + m.1.2 * n.2.1, m.1.1 * n.1.2 + m.1.2 * n.2.2),
   (m.2.1 * n.1.1 + m.2.2 * n.2.1, m.2.1 * n.1.2 + m.2.2 * n.2.2))

/-- CM110: coordinate multiplication is actual sequential execution. -/
theorem multiply_apply (m n : Pair.Matrix) (v : Pair.State) :
    Pair.operator (multiply m n) v = Pair.operator m (Pair.operator n v) := by
  simp only [Pair.operator_apply, multiply]
  apply Prod.ext <;> dsimp only <;> ring

/-- CM110: both orders of the full learned matrices enter the commutator. -/
noncomputable def commutator (m n : Pair.Matrix) : Pair.Matrix :=
  multiply m n - multiply n m

/-- CM110: no diagonal restriction is assumed; the actual commutator is traceless. -/
theorem commutator_trace (m n : Pair.Matrix) :
    (commutator m n).1.1 + (commutator m n).2.2 = 0 := by
  dsimp [commutator, multiply]
  ring

/-- CM110: the full squared commutator is scalar for arbitrary complex entries. -/
theorem commutator_square (m n : Pair.Matrix) :
    let k := commutator m n
    let q := k.1.1 ^ 2 + k.1.2 * k.2.1
    multiply k k = ((q, 0), (0, q)) := by
  dsimp [commutator, multiply]
  apply Prod.ext <;> apply Prod.ext <;> dsimp only <;> ring

/-- CM110: the actual eight-word signed score difference, in input order. -/
def obstruction (f : List O → ℝ) (a b c : O) : ℝ :=
  f [a,b,a,b,c] + f [b,a,b,a,c] + f [c,a,b,b,a] + f [c,b,a,a,b] -
    f [a,b,b,a,c] - f [b,a,a,b,c] - f [c,a,b,a,b] - f [c,b,a,b,a]

/-- CM110: complete complex coordinates, with no diagonal or real restriction,
satisfy the squared-commutator identity at the actual executed states. -/
theorem state_identity (v : Pair.State) (t : O → Pair.Matrix) (a b c : O) :
    let f := Pair.parameters v t (fun o => (t o).1)
    runFrom f v [a,b,a,b,c] + runFrom f v [b,a,b,a,c] +
      runFrom f v [c,a,b,b,a] + runFrom f v [c,b,a,a,b] =
    runFrom f v [a,b,b,a,c] + runFrom f v [b,a,a,b,c] +
      runFrom f v [c,a,b,a,b] + runFrom f v [c,b,a,b,a] := by
  dsimp only
  apply Prod.ext <;>
    simp only [runFrom, Pair.parameters, Pair.operator_apply, Prod.fst_add,
      Prod.snd_add] <;> ring

/-- CM110: arbitrary full learned readouts retain the execution identity. -/
theorem parameters_obstruction_zero (v : Pair.State) (t : O → Pair.Matrix)
    (u : O → Pair.State) (a b c output : O) :
    obstruction (fun xs => score (Pair.parameters v t u) xs output) a b c = 0 := by
  have hs := state_identity v t a b c
  have hr := congrArg (fun z => (Pair.row (u output) z).re) hs
  simp only [map_add, Complex.add_re] at hr
  unfold obstruction score complex
  simp only [Pair.parameters] at hr ⊢
  dsimp only [runFrom] at hr ⊢
  linarith

/-- CM110: the restriction concerns all complex linear two-state recurrences,
not a selected coordinate parameterization. -/
theorem pair_obstruction_zero (f : Parameters O Pair.State) (a b c output : O) :
    obstruction (fun xs => score f xs output) a b c = 0 := by
  rw [← Pair.reconstruction f]
  exact parameters_obstruction_zero _ _ _ a b c output

/-- CM110: obstruction is linear in the complete function values. -/
theorem obstruction_add (f g : List O → ℝ) (a b c : O) :
    obstruction (fun xs => f xs + g xs) a b c =
      obstruction f a b c + obstruction g a b c := by
  unfold obstruction
  ring

/-- CM110: scalar mixture mass cannot escape the polynomial identity. -/
theorem obstruction_scale (w : ℝ) (f : List O → ℝ) (a b c : O) :
    obstruction (fun xs => w * f xs) a b c = w * obstruction f a b c := by
  unfold obstruction
  ring

/-- CM110: uniform score error controls all eight terms of the actual relation. -/
theorem obstruction_lipschitz (f g : List O → ℝ) (ε : ℝ)
    (h : ∀ xs, |f xs - g xs| ≤ ε) (a b c : O) :
    |obstruction f a b c - obstruction g a b c| ≤ 8 * ε := by
  have h1 := (abs_le.mp (h [a,b,a,b,c]))
  have h2 := (abs_le.mp (h [b,a,b,a,c]))
  have h3 := (abs_le.mp (h [c,a,b,b,a]))
  have h4 := (abs_le.mp (h [c,b,a,a,b]))
  have h5 := (abs_le.mp (h [a,b,b,a,c]))
  have h6 := (abs_le.mp (h [b,a,a,b,c]))
  have h7 := (abs_le.mp (h [c,a,b,a,b]))
  have h8 := (abs_le.mp (h [c,b,a,b,a]))
  rw [abs_le]
  unfold obstruction
  constructor <;> linarith

example (f : List O → ℝ) (a b c : O) :
    |obstruction f a b c - obstruction f a b c| ≤ 8 * (0 : ℝ) :=
  obstruction_lipschitz f f 0 (by intro xs; simp) a b c

/-- CM110: retain every summand, including zero weights and duplicate blocks. -/
theorem obstruction_sum {J : Type*} [Fintype J] (f : J → List O → ℝ) (a b c : O) :
    obstruction (fun xs => ∑ j, f j xs) a b c = ∑ j, obstruction (f j) a b c := by
  simp only [obstruction, Finset.sum_add_distrib, Finset.sum_sub_distrib]

/-- CM110: arbitrarily many independently learned two-state blocks still
satisfy the same identity. Even arbitrary signed weights cannot remove it. -/
theorem mixture_obstruction_zero {J : Type*} [Fintype J]
    (f : J → Parameters O Pair.State) (w : J → ℝ) (a b c output : O) :
    obstruction (fun xs => ∑ j, w j * score (f j) xs output) a b c = 0 := by
  rw [obstruction_sum]
  apply Finset.sum_eq_zero
  intro j _
  rw [obstruction_scale, pair_obstruction_zero, mul_zero]

/-- CM110: a held position-dependent parent is not assumed to satisfy the
identity. Adding plain blocks leaves precisely its scaled obstruction. -/
theorem held_parent_obstruction {J : Type*} [Fintype J]
    (parent : List O → ℝ) (mass : ℝ) (f : J → Parameters O Pair.State)
    (w : J → ℝ) (a b c output : O) :
    obstruction (fun xs => mass * parent xs + ∑ j, w j * score (f j) xs output) a b c =
      mass * obstruction parent a b c := by
  rw [obstruction_add, obstruction_scale, mixture_obstruction_zero, add_zero]

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Width
