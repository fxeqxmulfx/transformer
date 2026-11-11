module

public import Transformer.Grokking.FlatSpectrum.Roots
public import Transformer.Grokking.FlatSpectrum.Training.Convergent
public import Transformer.Grokking.FlatSpectrum.Training.NuclearMin

/-!
# Stationary points are balanced

Source: `review/flat_spectrum.md`, *Training reaches the optimal fit* (added), item WD14,
statement 6; the item "Width" of WD11.

Let `W = [U; V]` be stationary, `M W = 0` for `M = [[λ, G], [Gᴴ, λ]]` with `λ ≠ 0`: the
limits of the runs of WD8 and WD9 are such points, at every width. Then `G V = −λ U` and
`Gᴴ U = −λ V`, and both give `Uᴴ G V`:

* `Uᴴ U = Vᴴ V` (`conjTranspose_mul_self_eq_of_stationary`), so `‖U‖ = ‖V‖`
  (`frobeniusSq_eq_of_stationary`);
* `(U Vᴴ)ᴴ (U Vᴴ) = (V Vᴴ)²` (`gram_mul_conjTranspose_of_stationary`), so the weight decay is
  the nuclear norm of the product, `½ (‖U‖² + ‖V‖²) = ‖U Vᴴ‖_S1`
  (`nuclearNorm_mul_conjTranspose_of_stationary`): the equality of Lemma 1 holds at every
  stationary point, not only at the decayed minima of WD2;
* the product has the rank of each factor (`rank_mul_conjTranspose_of_stationary`);
* for `λ > 0` a stationary point is a decayed minimum exactly when its product minimizes
  `L + λ ‖·‖_S1` (`isDecayedMin_iff_isNuclearMin_of_stationary`), and so is a limit of a run
  (`isDecayedMin_iff_isNuclearMin_of_tendsto`).

So what a limit of a narrow run is worth is read off its product: its weight-decayed loss is
`L(X) + λ ‖X‖_S1` at `X = U Vᴴ`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training.Scope

open Matrix Filter Topology
open scoped ComplexOrder

variable {m n k : Type*} [Fintype m] [Fintype n] [Fintype k] [DecidableEq m] [DecidableEq n]
  {L : Matrix m n ℂ → ℝ} {lam : ℝ} {G : Matrix m n ℂ} {U : Matrix m k ℂ} {V : Matrix n k ℂ}

omit [Fintype k] in
/-- **A stationary point is balanced** (`review/flat_spectrum.md`, item WD14): `M [U; V] = 0`
with `λ ≠ 0` gives `Uᴴ U = Vᴴ V`. -/
theorem conjTranspose_mul_self_eq_of_stationary (hlam : lam ≠ 0)
    (h : lift lam G * stack U V = 0) : Uᴴ * U = Vᴴ * V := by
  rw [lift_mul_stack] at h
  have h1 : G * V = -(lam • U) := eq_neg_of_add_eq_zero_left (by
    ext i j
    exact congrFun (congrFun h (Sum.inl i)) j)
  have h2 : Gᴴ * U = -(lam • V) := eq_neg_of_add_eq_zero_left (by
    ext i j
    exact congrFun (congrFun h (Sum.inr i)) j)
  have e1 : Uᴴ * (G * V) = -(lam • (Uᴴ * U)) := by
    rw [h1, Matrix.mul_neg, Matrix.mul_smul]
  have e2 : Uᴴ * (G * V) = -(lam • (Vᴴ * V)) := by
    have h3 := congrArg conjTranspose h2
    rw [conjTranspose_mul, conjTranspose_conjTranspose, conjTranspose_neg, conjTranspose_smul,
      star_trivial] at h3
    rw [← Matrix.mul_assoc, h3, Matrix.neg_mul, Matrix.smul_mul]
  exact smul_right_injective _ hlam (neg_injective (e1.symm.trans e2))

/-- At a stationary point the factors have the same norm. -/
theorem frobeniusSq_eq_of_stationary (hlam : lam ≠ 0) (h : lift lam G * stack U V = 0) :
    frobeniusSq U = frobeniusSq V := by
  rw [frobeniusSq_eq_re_trace, frobeniusSq_eq_re_trace,
    conjTranspose_mul_self_eq_of_stationary hlam h]

/-- At a stationary point `(U Vᴴ)ᴴ (U Vᴴ) = (V Vᴴ)ᴴ (V Vᴴ)`. -/
lemma gram_mul_conjTranspose_of_stationary (hlam : lam ≠ 0)
    (h : lift lam G * stack U V = 0) :
    (U * Vᴴ)ᴴ * (U * Vᴴ) = (V * Vᴴ)ᴴ * (V * Vᴴ) := by
  simp only [conjTranspose_mul, conjTranspose_conjTranspose, Matrix.mul_assoc]
  rw [← Matrix.mul_assoc Uᴴ, conjTranspose_mul_self_eq_of_stationary hlam h, Matrix.mul_assoc]

/-- **At a stationary point the weight decay is the nuclear norm of the product**
(`review/flat_spectrum.md`, item WD14; Lemma 1 with equality):
`‖U Vᴴ‖_S1 = ½ (‖U‖² + ‖V‖²)`. -/
theorem nuclearNorm_mul_conjTranspose_of_stationary (hlam : lam ≠ 0)
    (h : lift lam G * stack U V = 0) :
    nuclearNorm (U * Vᴴ) = (frobeniusSq U + frobeniusSq V) / 2 := by
  have hPX : (V * Vᴴ) * (V * Vᴴ) = (U * Vᴴ)ᴴ * (U * Vᴴ) := by
    rw [gram_mul_conjTranspose_of_stationary hlam h, conjTranspose_mul,
      conjTranspose_conjTranspose]
  have h1 := nuclearNorm_eq_trace_of_mul_self (posSemidef_self_mul_conjTranspose V) hPX
  have h2 : (V * Vᴴ).trace.re = frobeniusSq V := by
    rw [Matrix.trace_mul_comm, frobeniusSq_eq_re_trace]
  have h3 : nuclearNorm (U * Vᴴ) = frobeniusSq V := by
    rw [← h2, ← h1, Complex.ofReal_re]
  rw [h3, frobeniusSq_eq_of_stationary hlam h]
  ring

/-- **At a stationary point the product has the rank of each factor**
(`review/flat_spectrum.md`, item WD14). -/
theorem rank_mul_conjTranspose_of_stationary (hlam : lam ≠ 0)
    (h : lift lam G * stack U V = 0) :
    (U * Vᴴ).rank = U.rank ∧ (U * Vᴴ).rank = V.rank := by
  have hUV : U.rank = V.rank := by
    rw [← rank_conjTranspose_mul_self U, conjTranspose_mul_self_eq_of_stationary hlam h,
      rank_conjTranspose_mul_self]
  have hX : (U * Vᴴ).rank = V.rank := by
    rw [← rank_conjTranspose_mul_self (U * Vᴴ), gram_mul_conjTranspose_of_stationary hlam h,
      rank_conjTranspose_mul_self, rank_self_mul_conjTranspose]
  exact ⟨hX.trans hUV.symm, hX⟩

/-- **A stationary point is a decayed minimum exactly when its product is a trained product**
(`review/flat_spectrum.md`, item WD14), for `λ > 0`. -/
theorem isDecayedMin_iff_isNuclearMin_of_stationary (hlam : 0 < lam)
    (h : lift lam G * stack U V = 0) :
    IsDecayedMin L lam U V ↔ IsNuclearMin L lam (U * Vᴴ) := by
  refine ⟨fun hmin => hmin.isNuclearMin hlam.le, fun hX r U' V' => ?_⟩
  have h1 := hX (U' * V'ᴴ)
  have h2 := mul_le_mul_of_nonneg_left (nuclearNorm_le_of_eq_mul U' V' rfl) hlam.le
  rw [decayedLoss, decayedLoss, ← nuclearNorm_mul_conjTranspose_of_stationary hlam.ne' h]
  linarith

/-- **A limit of a run is a decayed minimum exactly when its product is a trained product**
(`review/flat_spectrum.md`, item WD14): for a continuous gradient field, `η > 0` and `λ > 0`,
at every width and from every start. -/
theorem isDecayedMin_iff_isNuclearMin_of_tendsto {G : Matrix m n ℂ → Matrix m n ℂ} {η : ℝ}
    {U₀ : Matrix m k ℂ} {V₀ : Matrix n k ℂ} (hG : Continuous G) (hη : 0 < η) (hlam : 0 < lam)
    (hlim : Tendsto (decayRun G η lam U₀ V₀) atTop (𝓝 (U, V))) :
    IsDecayedMin L lam U V ↔ IsNuclearMin L lam (U * Vᴴ) :=
  isDecayedMin_iff_isNuclearMin_of_stationary hlam
    (lift_mul_stack_eq_zero_of_tendsto_decayRun hG hη hlim)

/-! The witness: the stationary pair `U = V = 1` of the linear loss `−Re X` of a `1 × 1` matrix
with `λ = 1`, where `G = −1`; and the run from the zero factors for the gradient field
`X ↦ X − 1`, which stays at `0`. -/

private lemma stationary_one : lift (1 : ℝ) (-1 : Matrix (Fin 1) (Fin 1) ℂ) *
    stack (1 : Matrix (Fin 1) (Fin 1) ℂ) (1 : Matrix (Fin 1) (Fin 1) ℂ) = 0 := by
  rw [lift_mul_stack, ← stack_zero]
  congr 1 <;> simp

example := conjTranspose_mul_self_eq_of_stationary one_ne_zero stationary_one

example := frobeniusSq_eq_of_stationary one_ne_zero stationary_one

example := nuclearNorm_mul_conjTranspose_of_stationary one_ne_zero stationary_one

example := rank_mul_conjTranspose_of_stationary one_ne_zero stationary_one

example := isDecayedMin_iff_isNuclearMin_of_stationary
  (L := fun X : Matrix (Fin 1) (Fin 1) ℂ => pairing (-1) X) one_pos stationary_one

example : IsDecayedMin (fitLoss Set.univ (1 : Matrix (Fin 2) (Fin 2) ℂ)) 1
      (0 : Matrix (Fin 2) (Fin 2) ℂ) (0 : Matrix (Fin 2) (Fin 2) ℂ) ↔
    IsNuclearMin (fitLoss Set.univ (1 : Matrix (Fin 2) (Fin 2) ℂ)) 1
      ((0 : Matrix (Fin 2) (Fin 2) ℂ) * (0 : Matrix (Fin 2) (Fin 2) ℂ)ᴴ) :=
  isDecayedMin_iff_isNuclearMin_of_tendsto (G := fun X => X - 1) (η := 1)
    (continuous_id.sub continuous_const) one_pos one_pos
    (tendsto_const_nhds.congr fun t => by
      induction t with
      | zero => rfl
      | succ t ih =>
        rw [decayRun, Function.iterate_succ_apply', ← decayRun, ← ih]
        simp [decayStep])

end Transformer.Grokking.FlatSpectrum.Training.Scope
