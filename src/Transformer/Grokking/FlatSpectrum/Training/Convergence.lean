module

public import Transformer.Grokking.FlatSpectrum.Training.Certificate
public import Transformer.Grokking.FlatSpectrum.Training.Limit
public import Transformer.Grokking.FlatSpectrum.Training.LinearRun

/-!
# Gradient descent with weight decay converges only to a global minimum

Source: `review/flat_spectrum.md`, Lemma 1, "Training with weight decay then behaves like
minimum-nuclear-norm interpolation from Ω. This is the modeling step", and *Training reaches
the optimal fit* (added), item WD5.

The loss `L` is convex with a continuous gradient `G`. Run gradient descent with weight decay
`λ` and step `η > 0` on the factors of `U Vᴴ` (`decayRun`) from a start whose stacked factors
`[U₀; V₀]` have full row rank, with every step matrix `1 − η M_t` positive definite. If the run
converges, to `(U, V)`, then

* the limit is stationary and its lifted gradient is positive semidefinite
  (`lift_posSemidef_of_tendsto_decayRun`, by `posSemidef_of_tendsto`);
* `U Vᴴ` has the least weight-decayed loss among the factorizations of every width
  (`isDecayedMin_of_tendsto_decayRun`, by `isDecayedMin_of_posSemidef`);
* `U Vᴴ` minimizes `L + λ ‖·‖_S1` over all matrices, and for `λ > 0` its weight decay equals
  `‖U Vᴴ‖_S1` (`add_nuclearNorm_le_of_tendsto_decayRun`,
  `weightDecay_eq_nuclearNorm_of_tendsto_decayRun`).

So a run of the pure factorization cannot end at a saddle or at a local minimum that is not
global: the modeling step holds for every convergent run. Convergence itself, full row rank
(which needs width at least `|m| + |n|`) and the step condition are hypotheses. The example is
the linear loss on a `1 × 1` matrix, whose run converges to a nonzero limit.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix Filter Topology
open scoped ComplexOrder

variable {m n k : Type*} [Fintype m] [Fintype n] [Fintype k] [DecidableEq m] [DecidableEq n]
  {L : Matrix m n ℂ → ℝ} {G : Matrix m n ℂ → Matrix m n ℂ} {η lam : ℝ} {U₀ U : Matrix m k ℂ}
  {V₀ V : Matrix n k ℂ}

/-- **The limit of a run is stationary with a positive semidefinite lifted gradient**
(`review/flat_spectrum.md`, item WD5): for a continuous gradient field `G`, a step `η > 0`, a
start `[U₀; V₀]` of full row rank and positive definite step matrices, the limit `(U, V)` of a
convergent run has `M ≥ 0` and `M [U; V] = 0` for `M = [[λ, G], [Gᴴ, λ]]` at `G = G (U Vᴴ)`. -/
theorem lift_posSemidef_of_tendsto_decayRun (hG : Continuous G) (hη : 0 < η)
    (hfull : Function.Surjective (stack U₀ V₀).mulVec)
    (hstep : ∀ t, (1 - η • lift lam (G ((decayRun G η lam U₀ V₀ t).1 *
      (decayRun G η lam U₀ V₀ t).2ᴴ))).PosDef)
    (hlim : Tendsto (decayRun G η lam U₀ V₀) atTop (𝓝 (U, V))) :
    (lift lam (G (U * Vᴴ))).PosSemidef ∧ lift lam (G (U * Vᴴ)) * stack U V = 0 := by
  have hW : Tendsto (fun t => stack (decayRun G η lam U₀ V₀ t).1 (decayRun G η lam U₀ V₀ t).2)
      atTop (𝓝 (stack U V)) := (continuous_stack.tendsto (U, V)).comp hlim
  have hM : Tendsto (fun t => lift lam (G ((decayRun G η lam U₀ V₀ t).1 *
      (decayRun G η lam U₀ V₀ t).2ᴴ))) atTop (𝓝 (lift lam (G (U * Vᴴ)))) :=
    ((((continuous_lift lam).comp hG).comp continuous_mul_conjTranspose).tendsto (U, V)).comp hlim
  obtain ⟨B, hB⟩ := exists_frobeniusSq_le_of_tendsto hW
  refine ⟨posSemidef_of_tendsto (lift_isHermitian lam _) hM hη hstep
    (stack_decayRun_succ G η lam U₀ V₀) hfull hB, ?_⟩
  have hc : Continuous fun A : Matrix (m ⊕ n) (m ⊕ n) ℂ => 1 - η • A :=
    continuous_const.sub (continuous_id.const_smul η)
  have h1 : stack U V = (1 - η • lift lam (G (U * Vᴴ))) * stack U V :=
    eq_mul_of_tendsto_of_succ ((hc.tendsto _).comp hM) hW (stack_decayRun_succ G η lam U₀ V₀)
  rw [Matrix.sub_mul, Matrix.one_mul, Matrix.smul_mul, eq_comm, sub_eq_self, smul_eq_zero] at h1
  exact h1.resolve_left hη.ne'

/-- **Gradient descent with weight decay converges only to a global minimum**
(`review/flat_spectrum.md`, item WD5; Lemma 1's modeling step for the pure factorization): for
a convex loss `L` with continuous gradient `G`, a step `η > 0`, a start `[U₀; V₀]` of full row
rank and positive definite step matrices `1 − η M_t`, the limit `(U, V)` of a convergent run has
the least weight-decayed loss `L (U Vᴴ) + λ · ½ (‖U‖² + ‖V‖²)` among the factorizations of every
width. -/
theorem isDecayedMin_of_tendsto_decayRun (hconv : ∀ X Y, L X + pairing (G X) (Y - X) ≤ L Y)
    (hG : Continuous G) (hη : 0 < η) (hfull : Function.Surjective (stack U₀ V₀).mulVec)
    (hstep : ∀ t, (1 - η • lift lam (G ((decayRun G η lam U₀ V₀ t).1 *
      (decayRun G η lam U₀ V₀ t).2ᴴ))).PosDef)
    (hlim : Tendsto (decayRun G η lam U₀ V₀) atTop (𝓝 (U, V))) : IsDecayedMin L lam U V := by
  obtain ⟨hpsd, hstat⟩ := lift_posSemidef_of_tendsto_decayRun hG hη hfull hstep hlim
  exact isDecayedMin_of_posSemidef hconv hstat hpsd

/-- **The trained product minimizes `L + λ ‖·‖_S1`** (`review/flat_spectrum.md`, item WD5):
under the hypotheses of `isDecayedMin_of_tendsto_decayRun` with `λ ≥ 0`, the limit of the run
multiplies to a minimizer of the nuclear-norm-penalized loss over all matrices. -/
theorem add_nuclearNorm_le_of_tendsto_decayRun
    (hconv : ∀ X Y, L X + pairing (G X) (Y - X) ≤ L Y) (hG : Continuous G) (hη : 0 < η)
    (hlam : 0 ≤ lam) (hfull : Function.Surjective (stack U₀ V₀).mulVec)
    (hstep : ∀ t, (1 - η • lift lam (G ((decayRun G η lam U₀ V₀ t).1 *
      (decayRun G η lam U₀ V₀ t).2ᴴ))).PosDef)
    (hlim : Tendsto (decayRun G η lam U₀ V₀) atTop (𝓝 (U, V))) (Y : Matrix m n ℂ) :
    L (U * Vᴴ) + lam * nuclearNorm (U * Vᴴ) ≤ L Y + lam * nuclearNorm Y :=
  (isDecayedMin_of_tendsto_decayRun hconv hG hη hfull hstep hlim).add_nuclearNorm_le hlam Y

/-- **The trained weight decay is the nuclear norm of the trained product**
(`review/flat_spectrum.md`, item WD5): under the hypotheses of
`isDecayedMin_of_tendsto_decayRun` with `λ > 0`, `½ (‖U‖² + ‖V‖²) = ‖U Vᴴ‖_S1` at the limit. -/
theorem weightDecay_eq_nuclearNorm_of_tendsto_decayRun
    (hconv : ∀ X Y, L X + pairing (G X) (Y - X) ≤ L Y) (hG : Continuous G) (hη : 0 < η)
    (hlam : 0 < lam) (hfull : Function.Surjective (stack U₀ V₀).mulVec)
    (hstep : ∀ t, (1 - η • lift lam (G ((decayRun G η lam U₀ V₀ t).1 *
      (decayRun G η lam U₀ V₀ t).2ᴴ))).PosDef)
    (hlim : Tendsto (decayRun G η lam U₀ V₀) atTop (𝓝 (U, V))) :
    (frobeniusSq U + frobeniusSq V) / 2 = nuclearNorm (U * Vᴴ) :=
  (isDecayedMin_of_tendsto_decayRun hconv hG hη hfull hstep hlim).weightDecay_eq_nuclearNorm hlam

/-! The witness: the linear loss `L X = −Re X` of a `1 × 1` matrix, weight decay `1`, step `¼`,
from `U₀ = [1 0]`, `V₀ = [0 1]`. The run converges to `U = V = ½ (U₀ + V₀)`, which is not
zero. -/

private lemma witness_conv (X Y : Matrix (Fin 1) (Fin 1) ℂ) :
    pairing (-1) X + pairing (-1) (Y - X) ≤ pairing (-1) Y := by
  rw [pairing_sub_right]
  exact (add_sub_cancel _ _).le

private lemma witness_full : Function.Surjective (stack (fromCols (1 : Matrix (Fin 1) (Fin 1) ℂ)
    (0 : Matrix (Fin 1) (Fin 1) ℂ)) (fromCols (0 : Matrix (Fin 1) (Fin 1) ℂ)
      (1 : Matrix (Fin 1) (Fin 1) ℂ))).mulVec :=
  fun v => ⟨v, by rw [stack_fromCols_one, one_mulVec]⟩

private lemma witness_step :
    (1 - (1 / 4 : ℝ) • lift 1 (-1 : Matrix (Fin 1) (Fin 1) ℂ)).PosDef :=
  one_sub_smul_lift_posDef (g := 1) (by norm_num) zero_le_one (by simp [frobeniusSq])
    (by norm_num)

example := lift_posSemidef_of_tendsto_decayRun (G := fun _ => (-1 : Matrix (Fin 1) (Fin 1) ℂ))
  continuous_const (by norm_num) witness_full (fun _ => witness_step)
  (tendsto_decayRun_linear (by norm_num) (by norm_num) _ _)

example := isDecayedMin_of_tendsto_decayRun (L := fun X => pairing (-1) X)
  (G := fun _ => (-1 : Matrix (Fin 1) (Fin 1) ℂ)) witness_conv continuous_const (by norm_num)
  witness_full (fun _ => witness_step) (tendsto_decayRun_linear (by norm_num) (by norm_num) _ _)

example := add_nuclearNorm_le_of_tendsto_decayRun (L := fun X => pairing (-1) X)
  (G := fun _ => (-1 : Matrix (Fin 1) (Fin 1) ℂ)) witness_conv continuous_const (by norm_num)
  zero_le_one witness_full (fun _ => witness_step)
  (tendsto_decayRun_linear (by norm_num) (by norm_num) _ _)

example := weightDecay_eq_nuclearNorm_of_tendsto_decayRun (L := fun X => pairing (-1) X)
  (G := fun _ => (-1 : Matrix (Fin 1) (Fin 1) ℂ)) witness_conv continuous_const (by norm_num)
  one_pos witness_full (fun _ => witness_step)
  (tendsto_decayRun_linear (by norm_num) (by norm_num) _ _)

end Transformer.Grokking.FlatSpectrum.Training
