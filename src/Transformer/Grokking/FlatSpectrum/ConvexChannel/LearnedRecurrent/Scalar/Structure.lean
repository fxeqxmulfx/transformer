module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Scalar.Basic

/-!
# Exact absent-object response and the scalar order limitation

Source: `review/convex/moments.md`, CM96. Every complex linear operator on the
scalar state is multiplication by a learned scalar. Skipping proposals for an
object absent from every input record changes no forward or loss. Parameters
remain present. Conversely all scalar-domain functions are permutation
invariant: learned depth alone does not make this subclass order-sensitive.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Scalar

open scoped BigOperators

variable {O : Type*}

/-- Every scalar complex linear map has the complete native scalar
representation, CM96. No direction or imaginary coordinate is omitted. -/
theorem operator_value (L : ℂ →L[ℂ] ℂ) (z : ℂ) : L z = L 1 * z := by
  calc
    L z = L (z • (1 : ℂ)) := by simp
    _ = z • L 1 := map_smul L z 1
    _ = L 1 * z := by simp [smul_eq_mul, mul_comm]

/-- Reconstruction includes every initial, transition and output
parameter of the scalar operator family, CM96. -/
theorem reconstruction (f : Parameters O ℂ) :
    parameters f.initial (fun o => f.transition o 1) (fun c => f.readout c 1) = f := by
  cases f with
  | mk a t u =>
    have ht : (fun o => multiply (t o 1)) = t := by
      funext o
      apply ContinuousLinearMap.ext
      intro z
      exact (multiply_apply _ _).trans (operator_value (t o) z).symm
    have hu : (fun c => multiply (u c 1)) = u := by
      funext c
      apply ContinuousLinearMap.ext
      intro z
      exact (multiply_apply _ _).trans (operator_value (u c) z).symm
    change Parameters.mk a (fun o => multiply (t o 1)) (fun c => multiply (u c 1)) =
      Parameters.mk a t u
    rw [ht, hu]

/-- Changing an absent object's scalar leaves the actual complex
execution unchanged, CM96. The coordinate remains stored and trainable. -/
theorem absent_complex [DecidableEq O] (a : ℂ) (t u : O → ℂ) (o : O) (z : ℂ)
    (xs : List O) (ho : o ∉ xs) (c : O) :
    complex (parameters a (Function.update t o z) u) xs c = complex (parameters a t u) xs c := by
  have hmap : xs.map (Function.update t o z) = xs.map t := by
    apply List.map_congr_left
    intro x hx
    rw [Function.update_apply]
    split_ifs with he
    · exact False.elim (ho (he ▸ hx))
    · rfl
  rw [complex_product, complex_product, hmap]

/-- The exact zero response also holds for the real original-class
scores used by CE, CM96. It is not a small-gradient cutoff. -/
theorem absent_score [DecidableEq O] (a : ℂ) (t u : O → ℂ) (o : O) (z : ℂ)
    (xs : List O) (ho : o ∉ xs) (c : O) :
    score (parameters a (Function.update t o z) u) xs c = score (parameters a t u) xs c :=
  congrArg Complex.re (absent_complex a t u o z xs ho c)

/-- Every complete scalar recurrence loses order under a permutation of
the input objects, CM96. Higher-dimensional operators do not share this proof. -/
theorem complex_permutation (f : Parameters O ℂ) (xs ys : List O) (h : xs.Perm ys) (c : O) :
    complex f xs c = complex f ys c := by
  conv_lhs => rw [← reconstruction f]
  conv_rhs => rw [← reconstruction f]
  rw [complex_product, complex_product, (h.map (fun o => f.transition o 1)).prod_eq]

/-- Real scalar class scores inherit the exact order limitation, CM96. -/
theorem score_permutation (f : Parameters O ℂ) (xs ys : List O) (h : xs.Perm ys) (c : O) :
    score f xs c = score f ys c := congrArg Complex.re (complex_permutation f xs ys h c)

/-- Finite mixtures cannot recover an order distinction erased by every
scalar component, CM96. This applies to the whole declared scalar domain. -/
theorem domain_permutation (B : ℝ) (r : List O → O → ℝ) (hr : r ∈ domain O ℂ B)
    (xs ys : List O) (h : xs.Perm ys) (c : O) : r xs c = r ys c := by
  obtain ⟨J, hJ, w, f, hw, hm, hf, rfl⟩ := finite_representation B r hr
  simp only [Finset.sum_apply, Pi.smul_apply, smul_eq_mul]
  apply Finset.sum_congr rfl
  intro j hj
  rw [score_permutation (f j) xs ys h c]

/-- Even a two-object order-dependent target cannot be strictly
classified by the scalar domain, CM96. No stress success is claimed for it. -/
theorem order_not_realizable (B : ℝ) (r : List Bool → Bool → ℝ) (hr : r ∈ domain Bool ℂ B) :
    ¬ (r [false, true] true < r [false, true] false ∧
      r [true, false] false < r [true, false] true) := by
  intro h
  have hp : ([false, true] : List Bool).Perm [true, false] := List.Perm.swap true false []
  have ht := domain_permutation B r hr _ _ hp true
  have hf := domain_permutation B r hr _ _ hp false
  linarith [h.1, h.2]

example : complex (parameters Complex.I (Function.update (fun _ : Bool => (1 : ℂ)) true Complex.I)
    (fun c => if c then Complex.I else -Complex.I)) [false] true =
    complex (parameters Complex.I (fun _ => 1) (fun c => if c then Complex.I else -Complex.I))
      [false] true := absent_complex _ _ _ true _ _ (by simp) _

example : score (parameters Complex.I (Function.update (fun _ : Bool => (1 : ℂ)) true Complex.I)
    (fun c => if c then Complex.I else -Complex.I)) [false] true =
    score (parameters Complex.I (fun _ => 1) (fun c => if c then Complex.I else -Complex.I))
      [false] true := absent_score _ _ _ true _ _ (by simp) _

example : complex (parameters Complex.I (fun b : Bool => if b then -1 else 1)
    (fun c => if c then Complex.I else -Complex.I)) [false, true] true =
    complex (parameters Complex.I (fun b : Bool => if b then -1 else 1)
      (fun c => if c then Complex.I else -Complex.I)) [true, false] true :=
  complex_permutation _ _ _ (List.Perm.swap true false []) _

example : score (parameters Complex.I (fun b : Bool => if b then -1 else 1)
    (fun c => if c then Complex.I else -Complex.I)) [false, true] true =
    score (parameters Complex.I (fun b : Bool => if b then -1 else 1)
      (fun c => if c then Complex.I else -Complex.I)) [true, false] true :=
  score_permutation _ _ _ (List.Perm.swap true false []) _

example : (0 : List Bool → Bool → ℝ) [false, true] true = 0 :=
  domain_permutation 1 0 (zero_mem 1) _ [true, false] (List.Perm.swap true false []) true

example : ¬ ((0 : List Bool → Bool → ℝ) [false, true] true < 0 ∧ (0 : ℝ) < 0) :=
  order_not_realizable 1 0 (zero_mem 1)

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Scalar
