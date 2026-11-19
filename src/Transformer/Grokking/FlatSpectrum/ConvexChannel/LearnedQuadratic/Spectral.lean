module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedQuadratic.Sequence
public import Transformer.Grokking.FlatSpectrum.FlatCapacity

/-!
# Flat-spectrum bounds for the actual complex quadratic sequence output

Source: `review/convex/moments.md`, CM85 and R3. A proof-only expansion through
all ordered address pairs certifies the actual recovered complex output's
width. Inference is still the learned representation/activation/readout path.
Flatness is a necessary rank/error bound, never a missing-label oracle or a
native convergence claim. The bound can be nonpositive and then gives no
positive error floor.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedQuadratic.Spectral

open scoped BigOperators Matrix ComplexOrder MatrixOrder Matrix.Norms.Elementwise
open LearnedHinge (Coordinate Prefix)

/-- Expand the quadratic form only for its actual real address coefficients,
CM85. Every matrix entry remains in the double sum. -/
theorem quadratic_real {D : Type*} [Fintype D] (M : Matrix D D ℂ) (r : D → ℝ) :
    quadratic M (fun d => (r d : ℂ)) = ∑ d, ∑ e, (M d e).re * r d * r e := by
  simp only [quadratic, Matrix.mulVec, dotProduct, Finset.mul_sum, Complex.re_sum,
    Pi.star_apply, Complex.star_def, Complex.conj_ofReal]
  apply Finset.sum_congr rfl
  intro d hd
  apply Finset.sum_congr rfl
  intro e he
  simp
  ring

variable {O X : Type*} [Fintype O] [DecidableEq O] {T : ℕ}

/-- The actual complex output operator before taking its final real logits,
CM85. Its columns are original immutable prefixes. -/
noncomputable def output (p : Parameters (Coordinate O T) O) (data : X → Prefix O T) :
    Matrix O X ℂ := fun c x =>
  sequenceBranch (p c true) (data x) - sequenceBranch (p c false) (data x)

/-- Full signed coefficients for a diagnostic expansion, CM85. These are
derived from the joint state and do not replace the learned inference path. -/
noncomputable def coefficients (p : Parameters (Coordinate O T) O) :
    Matrix O (Coordinate O T × Coordinate O T) ℂ := fun c de =>
  (1 + Complex.I) * ((p c true de.1 de.2).re - (p c false de.1 de.2).re : ℝ)

/-- All ordered raw-address pairs in the diagnostic expansion, CM85. -/
def addressPairs (data : X → Prefix O T) :
    Matrix (Coordinate O T × Coordinate O T) X ℂ := fun de x =>
  (LearnedHinge.Projection.rawRow (data x) de.1 *
    LearnedHinge.Projection.rawRow (data x) de.2 : ℝ)

/-- The full factorization is an identity for actual learned complex
inference, CM85. Neither component, sign, class nor coordinate is dropped. -/
theorem output_factorization {p : Parameters (Coordinate O T) O} (hp : p ∈ domain)
    (data : X → Prefix O T) : output p data = coefficients p * addressPairs data := by
  ext c x
  simp only [output, sequenceBranch_eq, branch_eq (hp c true), branch_eq (hp c false)]
  unfold inputRow
  rw [quadratic_real, quadratic_real]
  simp only [Matrix.mul_apply, Fintype.sum_prod_type, coefficients, addressPairs,
    Complex.ofReal_sum, Complex.ofReal_sub, Complex.ofReal_mul, Finset.mul_sum,
    ← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro d hd
  apply Finset.sum_congr rfl
  intro e he
  ring

variable [Fintype X] [DecidableEq X]

/-- R3 bounds the actual complex output's approximation error to a flat
target by its complete ordered-pair width, CM85. This is not a fit guarantee. -/
theorem flat_error_bound {K : Matrix O X ℂ} {s : ℝ} (hs : 0 < s) (hK : IsFlat K s)
    {p : Parameters (Coordinate O T) O} (hp : p ∈ domain) (data : X → Prefix O T) :
    s * ((min (Fintype.card O) (Fintype.card X) : ℕ) -
      Fintype.card (Coordinate O T × Coordinate O T)) ≤ frobeniusSq (K - output p data) := by
  rw [output_factorization hp]
  exact hK.mul_sub_card_le_frobeniusSq_sub hs _ _

/-- Exact representation of a nonzero flat target needs the stated width,
CM85. Flatness supplies no unseen-answer or optimizer theorem. -/
theorem flat_width_necessary {K : Matrix O X ℂ} {s : ℝ} (hs : s ≠ 0) (hK : IsFlat K s)
    {p : Parameters (Coordinate O T) O} (hp : p ∈ domain) (data : X → Prefix O T)
    (hexact : K = output p data) :
    min (Fintype.card O) (Fintype.card X) ≤
      Fintype.card (Coordinate O T × Coordinate O T) := by
  rw [output_factorization hp] at hexact
  exact hK.card_le_of_eq_mul hs hexact

example := output_factorization (O := Bool) (T := 1) (p := 0) zero_mem
  (fun x : Bool => (⟨[x], by simp⟩ : Prefix Bool 1))
example := flat_error_bound (O := Bool) (X := Bool) (T := 0)
  (by norm_num : (0 : ℝ) < 1) isFlat_one (p := 0) zero_mem
  (fun _ => ⟨[], by simp⟩)
example : min (Fintype.card Unit) (Fintype.card Unit) ≤
    Fintype.card (Coordinate Unit 0 × Coordinate Unit 0) := by
  obtain ⟨p, hp, hflat⟩ : ∃ p : Parameters (Coordinate Unit 0) Unit,
      p ∈ domain ∧ IsFlat (output p (fun _ : Unit => ⟨[], by simp⟩)) 2 := by
    let p : Parameters (Coordinate Unit 0) Unit := fun _ s => if s then 1 else 0
    have hp : p ∈ domain := by
      intro c s
      cases s
      · exact Matrix.PosSemidef.zero
      · exact Matrix.PosSemidef.one
    refine ⟨p, hp, ?_⟩
    left
    ext c d
    simp only [output_factorization hp]
    have hd : (default : Coordinate Unit 0) = none := Subsingleton.elim _ _
    norm_num [p, coefficients, addressPairs, LearnedHinge.Projection.rawRow, hd,
      Matrix.mul_apply, Matrix.conjTranspose_apply, Fintype.sum_prod_type, Fintype.sum_option]
    apply Complex.ext <;> norm_num
  exact flat_width_necessary (by norm_num : (2 : ℝ) ≠ 0) hflat hp
    (fun _ => ⟨[], by simp⟩) rfl

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedQuadratic.Spectral
