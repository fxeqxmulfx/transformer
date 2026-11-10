module

public import Transformer.Grokking.FlatSpectrum.Training.Lifted

/-!
# A certificate for the global minimum of the weight-decayed loss

Source: `review/flat_spectrum.md`, *Training reaches the optimal fit* (added), item WD2.
The loss `L` is convex with gradient `G`, in the first-order form
`L X + ⟨G X, Y − X⟩ ≤ L Y`. At a factorization `U Vᴴ` put `M = [[λ, G], [Gᴴ, λ]]` with
`G = G (U Vᴴ)`.

* If `M [U; V] = 0` (the point is stationary) and `M` is positive semidefinite, then `U Vᴴ` has
  the least weight-decayed loss among the factorizations of every width
  (`isDecayedMin_of_posSemidef`): for another `W' = [U'; V']`,
  `0 ≤ ⟨W', M W'⟩ = λ (‖U'‖² + ‖V'‖²) + 2 ⟨G, U' V'ᴴ⟩`, with equality at `W`, and convexity
  compares the two losses.
* Such a least factorization is Lemma 1 with a penalty: its product minimizes
  `L + λ ‖·‖_S1` over all matrices (`IsDecayedMin.add_nuclearNorm_le`) and, for `λ > 0`, its
  weight decay `½ (‖U‖² + ‖V‖²)` equals `‖U Vᴴ‖_S1` (`IsDecayedMin.weightDecay_eq_nuclearNorm`);
  the two together characterize it (`isDecayedMin_iff`).

The example is the linear loss `L X = −Re X` of a `1 × 1` matrix with `λ = 1`: the nonzero
factorization `1 · 1` is stationary, its lifted gradient `[[1, −1], [−1, 1]]` is positive
semidefinite, and it is a least one (`isDecayedMin_linear_one`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix
open scoped ComplexOrder

variable {m n k : Type*} [Fintype m] [Fintype n] [Fintype k] [DecidableEq m] [DecidableEq n]

/-- **A stationary point whose lifted gradient is positive semidefinite is a global minimum**
(`review/flat_spectrum.md`, item WD2): for a convex loss `L` with gradient `G`, if
`M [U; V] = 0` and `M ≥ 0` for `M = [[λ, G], [Gᴴ, λ]]` at `G = G (U Vᴴ)`, then `U Vᴴ` has the
least weight-decayed loss among the factorizations of every width. -/
theorem isDecayedMin_of_posSemidef {L : Matrix m n ℂ → ℝ} {G : Matrix m n ℂ → Matrix m n ℂ}
    (hconv : ∀ X Y, L X + pairing (G X) (Y - X) ≤ L Y) {lam : ℝ} {U : Matrix m k ℂ}
    {V : Matrix n k ℂ} (hstat : lift lam (G (U * Vᴴ)) * stack U V = 0)
    (hpsd : (lift lam (G (U * Vᴴ))).PosSemidef) : IsDecayedMin L lam U V := by
  intro r U' V'
  have h0 : pairing (stack U V) (lift lam (G (U * Vᴴ)) * stack U V) = 0 := by
    rw [hstat, pairing_zero_right]
  have h1 := pairing_nonneg_of_posSemidef hpsd (stack U' V')
  rw [pairing_stack_lift] at h0 h1
  have h2 := hconv (U * Vᴴ) (U' * V'ᴴ)
  rw [pairing_sub_right] at h2
  unfold decayedLoss
  linarith

/-- The linear loss `L X = −Re X` of a `1 × 1` matrix with weight decay `1`: the factorization
`1 · 1` is a least one. It is stationary, `M [1; 1] = 0` for `M = [[1, −1], [−1, 1]]`, and `M`
is positive semidefinite. -/
theorem isDecayedMin_linear_one :
    IsDecayedMin (fun X : Matrix (Fin 1) (Fin 1) ℂ => pairing (-1) X) 1
      (1 : Matrix (Fin 1) (Fin 1) ℂ) (1 : Matrix (Fin 1) (Fin 1) ℂ) := by
  refine isDecayedMin_of_posSemidef (G := fun _ => -1)
    (fun X Y => by rw [pairing_sub_right]; exact (add_sub_cancel _ _).le) ?_ ?_
  · rw [lift_mul_stack, ← stack_zero]
    congr 1 <;> simp
  · exact lift_posSemidef_of_frobeniusSq_le zero_le_one (by simp [frobeniusSq])

omit [DecidableEq m] in
/-- **The product of a least factorization minimizes `L + λ ‖·‖_S1`**
(`review/flat_spectrum.md`, item WD2, by Lemma 1). -/
theorem IsDecayedMin.add_nuclearNorm_le {L : Matrix m n ℂ → ℝ} {lam : ℝ} (hlam : 0 ≤ lam)
    {U : Matrix m k ℂ} {V : Matrix n k ℂ} (h : IsDecayedMin L lam U V) (Y : Matrix m n ℂ) :
    L (U * Vᴴ) + lam * nuclearNorm (U * Vᴴ) ≤ L Y + lam * nuclearNorm Y := by
  obtain ⟨r, U', V', hY, hc⟩ := (nuclearNorm_isLeast Y).1
  have h1 := h r U' V'
  have h2 := mul_le_mul_of_nonneg_left (nuclearNorm_le_of_eq_mul U V rfl) hlam
  unfold decayedLoss at h1
  rw [← hY, ← hc] at h1
  linarith

example := isDecayedMin_linear_one.add_nuclearNorm_le zero_le_one

omit [DecidableEq m] in
/-- **At a least factorization the weight decay is the nuclear norm of the product**
(`review/flat_spectrum.md`, item WD2): `½ (‖U‖² + ‖V‖²) = ‖U Vᴴ‖_S1` for `λ > 0`. -/
theorem IsDecayedMin.weightDecay_eq_nuclearNorm {L : Matrix m n ℂ → ℝ} {lam : ℝ}
    (hlam : 0 < lam) {U : Matrix m k ℂ} {V : Matrix n k ℂ} (h : IsDecayedMin L lam U V) :
    (frobeniusSq U + frobeniusSq V) / 2 = nuclearNorm (U * Vᴴ) := by
  obtain ⟨r, U', V', hX, hc⟩ := (nuclearNorm_isLeast (U * Vᴴ)).1
  have h1 := h r U' V'
  unfold decayedLoss at h1
  rw [← hX, ← hc] at h1
  exact le_antisymm (le_of_mul_le_mul_left (by linarith) hlam)
    (nuclearNorm_le_of_eq_mul U V rfl)

example := isDecayedMin_linear_one.weightDecay_eq_nuclearNorm one_pos

omit [DecidableEq m] in
/-- **Least factorizations are the penalized form of M** (`review/flat_spectrum.md`, item WD2):
for `λ > 0`, `U Vᴴ` has the least weight-decayed loss exactly when its product minimizes
`L + λ ‖·‖_S1` and its weight decay equals the nuclear norm of the product. -/
theorem isDecayedMin_iff {L : Matrix m n ℂ → ℝ} {lam : ℝ} (hlam : 0 < lam) {U : Matrix m k ℂ}
    {V : Matrix n k ℂ} :
    IsDecayedMin L lam U V ↔
      (∀ Y, L (U * Vᴴ) + lam * nuclearNorm (U * Vᴴ) ≤ L Y + lam * nuclearNorm Y) ∧
        (frobeniusSq U + frobeniusSq V) / 2 = nuclearNorm (U * Vᴴ) := by
  refine ⟨fun h => ⟨h.add_nuclearNorm_le hlam.le, h.weightDecay_eq_nuclearNorm hlam⟩, ?_⟩
  rintro ⟨hmin, hc⟩ r U' V'
  have h1 := hmin (U' * V'ᴴ)
  have h2 := mul_le_mul_of_nonneg_left (nuclearNorm_le_of_eq_mul U' V' rfl) hlam.le
  unfold decayedLoss
  rw [hc]
  linarith

example := (isDecayedMin_iff one_pos).mp isDecayedMin_linear_one

end Transformer.Grokking.FlatSpectrum.Training
