module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedBilinear.Basic
public import Transformer.Grokking.FlatSpectrum.FlatCapacity

/-!
# Flat-spectrum width bounds for the actual recovered bilinear interaction

Source: `review/convex/moments.md`, CM86 and R3. Complete complex basis probes
identify the joint matrix of the actual learned computation. Thus a physical
pair of encoders reproducing a flat matrix needs at least its full flat rank;
insufficient width has the usual squared Frobenius error floor. These bounds
do not provide missing labels or a learning algorithm. The basis probes below
are mathematical coefficient probes, not additional supervised raw sequences.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedBilinear.Spectral

open scoped BigOperators Matrix ComplexOrder Matrix.Norms.Elementwise
open FlatSpectrum

variable {Q D H : Type*} [Fintype Q] [Fintype D] [DecidableEq Q] [DecidableEq D]

/-- A complete complex basis probe of the actual recovered computation
returns its joint coefficient, including the fixed output phase, CM86. -/
theorem forward_entry (M : Matrix Q D ℂ) (q : Q) (d : D) :
    forward M (Pi.single q 1) (Pi.single d 1) = (1 + Complex.I) * M q d := by
  rw [forward_eq, Pi.star_single, star_one, Matrix.mulVec_single]
  simp

/-- Equality of all actual complex basis responses identifies the complete
joint matrix, CM86. Equality of raw real logits alone is not asserted to do so. -/
theorem probes_identify {M N : Matrix Q D ℂ}
    (h : ∀ q d, forward M (Pi.single q 1) (Pi.single d 1) =
      forward N (Pi.single q 1) (Pi.single d 1)) : M = N := by
  ext q d
  have he := h q d
  rw [forward_entry, forward_entry] at he
  have hn : (1 + Complex.I : ℂ) ≠ 0 := by
    intro hz
    have hr := congrArg Complex.re hz
    norm_num at hr
  exact mul_left_cancel₀ hn he

example : (1 : Matrix (Fin 2) (Fin 2) ℂ) = 1 := probes_identify (fun _ _ => rfl)

variable [Fintype H]

/-- R3 bounds approximation of a flat joint interaction by physical learned
encoders of any width, CM86. All complex coefficient errors are charged. -/
theorem flat_physical_error {M : Matrix Q D ℂ} {s : ℝ} (hs : 0 < s) (hM : IsFlat M s)
    (L : Matrix Q H ℂ) (R : Matrix D H ℂ) :
    s * ((min (Fintype.card Q) (Fintype.card D) : ℕ) - Fintype.card H) ≤
      frobeniusSq (M - L * R.conjTranspose) :=
  hM.mul_sub_card_le_frobeniusSq_sub hs L R.conjTranspose

/-- Every physical pair exactly reproducing all complex coefficient probes
of a flat learned joint matrix needs its full flat width, CM86. -/
theorem flat_physical_width {M : Matrix Q D ℂ} {s : ℝ} (hs : s ≠ 0) (hM : IsFlat M s)
    (L : Matrix Q H ℂ) (R : Matrix D H ℂ)
    (h : ∀ q d, forward M (Pi.single q 1) (Pi.single d 1) =
      (1 + Complex.I) * interaction L R (Pi.single q 1) (Pi.single d 1)) :
    min (Fintype.card Q) (Fintype.card D) ≤ Fintype.card H := by
  have he : M = L * R.conjTranspose := probes_identify (by
    intro q d
    rw [physical_coverage]
    exact h q d)
  exact hM.card_le_of_eq_mul hs he

example := flat_physical_error (Q := Fin 2) (D := Fin 2)
  (by norm_num : (0 : ℝ) < 1) isFlat_one
  (0 : Matrix (Fin 2) (Fin 1) ℂ) (0 : Matrix (Fin 2) (Fin 1) ℂ)

example : min (Fintype.card (Fin 2)) (Fintype.card (Fin 2)) ≤ Fintype.card (Fin 2) :=
  flat_physical_width (by norm_num : (1 : ℝ) ≠ 0) isFlat_one
    (1 : Matrix (Fin 2) (Fin 2) ℂ) (1 : Matrix (Fin 2) (Fin 2) ℂ) (by
      intro q d
      have h := physical_coverage (1 : Matrix (Fin 2) (Fin 2) ℂ)
        (1 : Matrix (Fin 2) (Fin 2) ℂ) (Pi.single q 1) (Pi.single d 1)
      simpa using h)

end Transformer.Grokking.Convex.ConvexChannel.LearnedBilinear.Spectral
