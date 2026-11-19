module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Projection.Mixture

/-!
# Finite certificates for a complete basis of feasible query directions

Source: `review/convex/moments.md`, CM81. Kernel membership, free-coordinate
identity and coordinate reconstruction from all original equality rows are
finite arithmetic checks. Completeness is a theorem, not a certificate field.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Projection.KernelBasis

open scoped BigOperators

variable {D R J : Type*} [Fintype D] [Fintype R] [Fintype J]

/-- A full-coordinate linear combination of query directions, CM81. -/
def combine (basis : J → D → ℝ) (coefficients : J → ℝ) : D → ℝ :=
  fun d => ∑ j, coefficients j * basis j d

/-- The row that reads one original parameter coordinate, CM81. -/
def coordinate [DecidableEq D] (d e : D) : ℝ := if d = e then 1 else 0

/-- Finite checkable kernel-basis data, CM81. Every coordinate residual has
an explicit linear representation in the original equality rows. -/
def Certificate [DecidableEq D] [DecidableEq J] (rows : R → D → ℝ)
    (basis : J → D → ℝ) (free : J → D) (recovery : D → R → ℝ) : Prop :=
  (∀ j r, dot (rows r) (basis j) = 0) ∧
  (∀ j k, basis j (free k) = if j = k then 1 else 0) ∧
  ∀ d e, coordinate d e - combine (fun j => coordinate (free j)) (fun j => basis j d) e =
    combine rows (recovery d) e

/-- Symmetry of the finite real inner product, CM81. -/
theorem dot_swap (a b : D → ℝ) : dot a b = dot b a := by
  apply Finset.sum_congr rfl
  intro d hd
  ring

/-- A coordinate row reads that coordinate exactly, CM81. -/
theorem coordinate_dot [DecidableEq D] (d : D) (v : D → ℝ) :
    dot (coordinate d) v = v d := by
  simp [dot, coordinate]

/-- Exact pairing with a complete basis combination, CM81. -/
theorem dot_combine (basis : J → D → ℝ) (coefficients : J → ℝ) (v : D → ℝ) :
    dot (combine basis coefficients) v = ∑ j, coefficients j * dot (basis j) v := by
  simp only [dot, combine, Finset.sum_mul]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro j hj
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro d hd
  ring

/-- Exact pairing with a difference of full coordinate arrays, CM81. -/
theorem dot_sub (a b v : D → ℝ) : dot (a - b) v = dot a v - dot b v := by
  simp only [dot, Pi.sub_apply, sub_mul, Finset.sum_sub_distrib]

/-- Dotting a checked coordinate identity with any vector gives its exact
residual formula, CM81. No kernel premise is used in this algebraic step. -/
theorem certificate_residual [DecidableEq D] [DecidableEq J]
    {rows : R → D → ℝ} {basis : J → D → ℝ} {free : J → D} {recovery : D → R → ℝ}
    (h : Certificate rows basis free recovery) (v : D → ℝ) (d : D) :
    v d - ∑ j, basis j d * v (free j) = ∑ r, recovery d r * dot (rows r) v := by
  have he : coordinate d - combine (fun j => coordinate (free j)) (fun j => basis j d) =
      combine rows (recovery d) := funext (h.2.2 d)
  have hp := congrArg (fun a => dot a v) he
  rw [dot_sub, coordinate_dot, dot_combine, dot_combine] at hp
  simpa only [coordinate_dot] using hp

/-- All kernel vectors reconstruct from their free coordinates, CM81.
The certificate proves completeness over every vector, not just tested probes. -/
theorem kernel_reconstruct [DecidableEq D] [DecidableEq J]
    {rows : R → D → ℝ} {basis : J → D → ℝ} {free : J → D} {recovery : D → R → ℝ}
    (h : Certificate rows basis free recovery) {v : D → ℝ}
    (hv : ∀ r, dot (rows r) v = 0) : v = combine basis (fun j => v (free j)) := by
  funext d
  have he := certificate_residual h v d
  simp only [hv, mul_zero, Finset.sum_const_zero, sub_eq_zero] at he
  simpa only [combine, mul_comm] using he

/-- Every represented direction satisfies every equality row, CM81. -/
theorem combine_kernel [DecidableEq D] [DecidableEq J]
    {rows : R → D → ℝ} {basis : J → D → ℝ} {free : J → D} {recovery : D → R → ℝ}
    (h : Certificate rows basis free recovery) (coefficients : J → ℝ) (r : R) :
    dot (rows r) (combine basis coefficients) = 0 := by
  rw [dot_swap, dot_combine]
  apply Finset.sum_eq_zero
  intro j hj
  rw [dot_swap, h.1 j r, mul_zero]

/-- Free coordinates recover every supplied coefficient, CM81. This also
certifies independence of the proposed basis, including an empty basis. -/
theorem combine_free [DecidableEq D] [DecidableEq J]
    {rows : R → D → ℝ} {basis : J → D → ℝ} {free : J → D} {recovery : D → R → ℝ}
    (h : Certificate rows basis free recovery) (coefficients : J → ℝ) (j : J) :
    combine basis coefficients (free j) = coefficients j := by
  simp only [combine, h.2.1, mul_ite, mul_one, mul_zero]
  exact Fintype.sum_ite_eq' j coefficients

/-- The checked family covers exactly the full equality kernel, CM81. -/
theorem kernel_iff [DecidableEq D] [DecidableEq J]
    {rows : R → D → ℝ} {basis : J → D → ℝ} {free : J → D} {recovery : D → R → ℝ}
    (h : Certificate rows basis free recovery) (v : D → ℝ) :
    (∀ r, dot (rows r) v = 0) ↔ ∃ coefficients, v = combine basis coefficients := by
  constructor
  · intro hv
    exact ⟨fun j => v (free j), kernel_reconstruct h hv⟩
  · rintro ⟨coefficients, rfl⟩ r
    exact combine_kernel h coefficients r

/-- A complete represented direction has unique basis coordinates, CM81. -/
theorem coefficients_unique [DecidableEq D] [DecidableEq J]
    {rows : R → D → ℝ} {basis : J → D → ℝ} {free : J → D} {recovery : D → R → ℝ}
    (h : Certificate rows basis free recovery) {a b : J → ℝ}
    (he : combine basis a = combine basis b) : a = b := by
  funext j
  have hj := congrArg (fun v => v (free j)) he
  simpa only [combine_free h] using hj

/-- Equal pairings on the certified basis imply equal work on every kernel
direction, CM81. Approximate finite differences do not satisfy this premise
automatically; their directional error still needs its own bound. -/
theorem pairing_agreement [DecidableEq D] [DecidableEq J]
    {rows : R → D → ℝ} {basis : J → D → ℝ} {free : J → D} {recovery : D → R → ℝ}
    (h : Certificate rows basis free recovery) {v g q : D → ℝ}
    (hv : ∀ r, dot (rows r) v = 0) (he : ∀ j, dot (basis j) g = dot (basis j) q) :
    dot v g = dot v q := by
  rw [kernel_reconstruct h hv, dot_combine, dot_combine]
  apply Finset.sum_congr rfl
  intro j hj
  rw [he j]

/-- The original opposite row reconstructs the diagonal example's second
coordinate from its first one, CM81. No original equality row is dropped. -/
def diagonalRecovery (d r : Bool) : ℝ := if d = false ∧ r = false then 1 else 0

/-- A nonzero basis for the degenerate diagonal cone, CM81. -/
theorem diagonal_certificate :
    Certificate diagonalNormal (fun _ : Unit => fun _ => 1) (fun _ => true) diagonalRecovery := by
  refine ⟨?_, ?_, ?_⟩
  · intro j r
    cases r <;> norm_num [dot, diagonalNormal, Fintype.sum_bool]
  · intro j k
    cases j
    cases k
    norm_num
  · intro d e
    cases d <;> cases e <;>
      norm_num [coordinate, combine, diagonalNormal, diagonalRecovery, Fintype.sum_bool]

example := certificate_residual diagonal_certificate (fun _ => 3) false
example := kernel_reconstruct diagonal_certificate (v := fun _ => 3)
  (by intro r; cases r <;> norm_num [dot, diagonalNormal, Fintype.sum_bool])
example := combine_kernel diagonal_certificate (fun _ => 3) false
example := combine_free diagonal_certificate (fun _ => 3) ()
example := kernel_iff diagonal_certificate (fun _ => 3)
example := coefficients_unique diagonal_certificate (a := fun _ => 3) (b := fun _ => 3) rfl
example := pairing_agreement diagonal_certificate (v := fun _ => 3)
  (g := fun d => if d then 2 else 0) (q := fun _ => 1)
  (by intro r; cases r <;> norm_num [dot, diagonalNormal, Fintype.sum_bool])
  (by norm_num [dot, Fintype.sum_bool])

end Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Projection.KernelBasis
