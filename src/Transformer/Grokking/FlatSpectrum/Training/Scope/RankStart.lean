module

public import Transformer.Grokking.FlatSpectrum.FlatCapacity
public import Transformer.Grokking.FlatSpectrum.Training.Margin
public import Transformer.Grokking.FlatSpectrum.Training.SquaredRun
public import Transformer.Grokking.FlatSpectrum.Training.WidthOne
public import Transformer.Grokking.FlatSpectrum.Training.Scope.StackRank

/-!
# The `2 × 2` identity: starts of rank at most one are exceptional at every width

Source: `review/flat_spectrum.md`, *Training reaches the optimal fit* (added), item WD14; the
item "Width" of WD11 and the counterexample of WD10.

For the squared loss on all cells of the `2 × 2` identity and a weight decay `0 ≤ λ < 1`:

* a trained product has rank two (`one_lt_rank_of_isNuclearMin_one`): a matrix `X` of rank at
  most one has `‖X‖_S1 = ‖X‖_S2`, and `Re tr X ≤ ‖X‖_S1`, so
  `½ ‖X − 1‖² + λ ‖X‖_S1 ≥ 1 − ½ (1 − λ)²`, above the value `2 λ − λ²` of `(1 − λ) · 1`;
* so a run from a start whose stack has rank at most one, of any width and with any step, does
  not converge to a decayed minimum (`not_isDecayedMin_one_of_rank_le_one`), by `StackRank`;
* under the explicit bounds of WD8 such a run converges, and its limit is stationary and not a
  decayed minimum (`exists_tendsto_decayRun_one_not_isDecayedMin`).

At width two, which is `min(m, n)`, almost every run with a small step ends at a decayed
minimum (WD10). The exceptional null set of that statement contains every start whose stack
has rank at most one.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training.Scope

open Matrix Filter Topology

variable {k : Type*} [Fintype k] {η lam κ ρ : ℝ}

/-- **A trained product of the `2 × 2` identity has rank two** (`review/flat_spectrum.md`, item
WD14): for the squared loss on all cells and `0 ≤ λ < 1`, a minimizer of `L + λ ‖·‖_S1` has a
rank above one. -/
theorem one_lt_rank_of_isNuclearMin_one (h0 : 0 ≤ lam) (h1 : lam < 1)
    {X : Matrix (Fin 2) (Fin 2) ℂ}
    (hX : IsNuclearMin (fitLoss Set.univ (1 : Matrix (Fin 2) (Fin 2) ℂ)) lam X) :
    1 < X.rank := by
  by_contra hr
  have hr1 : (X.rank : ℝ) ≤ 1 := by exact_mod_cast not_lt.1 hr
  have hup : fitLoss Set.univ 1 X + lam * nuclearNorm X ≤ 2 * lam - lam ^ 2 := by
    have h := hX ((√(1 - lam) • (1 : Matrix (Fin 2) (Fin 2) ℂ)) * (√(1 - lam) • 1)ᴴ)
    have h2 := mul_le_mul_of_nonneg_left (nuclearNorm_le_of_eq_mul
      (√(1 - lam) • (1 : Matrix (Fin 2) (Fin 2) ℂ))
      (√(1 - lam) • (1 : Matrix (Fin 2) (Fin 2) ℂ)) rfl) h0
    have h3 := decayedLoss_sqrt_smul_one h1.le
    rw [decayedLoss] at h3
    linarith
  have hf : nuclearNorm X ^ 2 = frobeniusSq X :=
    le_antisymm ((nuclearNorm_sq_le_rank_mul_frobeniusSq X).trans
      (mul_le_of_le_one_left (frobeniusSq_nonneg X) hr1)) (frobeniusSq_le_nuclearNorm_sq X)
  have htr : pairing X 1 ≤ nuclearNorm X := by
    have h := norm_trace_le_nuclearNorm (isContraction_one (n := Fin 2)) X
    rw [conjTranspose_one, Matrix.one_mul] at h
    have e : pairing X 1 = X.trace.re := by
      rw [pairing, Matrix.mul_one, trace_conjTranspose, Complex.star_def, Complex.conj_re]
    rw [e]
    exact (Complex.re_le_norm _).trans h
  have hmask : mask Set.univ (X - 1) = X - 1 := by
    ext i j
    simp [mask]
  have hL : fitLoss Set.univ 1 X = (frobeniusSq X - 2 * pairing X 1 + 2) / 2 := by
    rw [fitLoss, hmask, frobeniusSq_sub, frobeniusSq_one_fin_two]
  rw [hL, ← hf] at hup
  nlinarith [sq_nonneg (nuclearNorm X - (1 - lam)), sq_pos_of_pos (sub_pos.2 h1)]

/-- **A start of rank at most one does not reach a decayed minimum of the `2 × 2` identity**
(`review/flat_spectrum.md`, item WD14): for the squared loss on all cells and `0 ≤ λ < 1`, at
every width and with every step, a limit of a run from stacked factors of rank at most one is
not a decayed minimum. -/
theorem not_isDecayedMin_one_of_rank_le_one (h0 : 0 ≤ lam) (h1 : lam < 1)
    {U₀ V₀ U V : Matrix (Fin 2) k ℂ} (hrank : (stack U₀ V₀).rank ≤ 1)
    (hlim : Tendsto (decayRun (fun X => mask Set.univ (X - 1)) η lam U₀ V₀) atTop
      (𝓝 (U, V))) :
    ¬ IsDecayedMin (fitLoss Set.univ (1 : Matrix (Fin 2) (Fin 2) ℂ)) lam U V :=
  not_isDecayedMin_of_rank_lt h0
    (fun _ hX => hrank.trans_lt (one_lt_rank_of_isNuclearMin_one h0 h1 hX)) hlim

/-- **Such a run converges, to a stationary point that is not a decayed minimum**
(`review/flat_spectrum.md`, item WD14): for the squared loss on all cells of the `2 × 2`
identity, `0 < λ < 1` and the bounds of WD8 with `‖1‖ ≤ κ`, the run from a start in the ball
whose stack has rank at most one, of any width. -/
theorem exists_tendsto_decayRun_one_not_isDecayedMin (hη : 0 < η) (hlam : 0 < lam)
    (hlam1 : lam < 1) (hκ0 : 0 ≤ κ) (hκ : 2 ≤ κ ^ 2) (hρ : κ ^ 2 ≤ lam * ρ)
    (hstep : 2 * η * (lam ^ 2 + (ρ / 2 + κ) ^ 2) ≤ lam)
    (hdesc : η * (lam + (ρ / 2 + κ) + 2 * ρ) ≤ 1) {U₀ V₀ : Matrix (Fin 2) k ℂ}
    (hw : frobeniusSq U₀ + frobeniusSq V₀ ≤ ρ) (hrank : (stack U₀ V₀).rank ≤ 1) :
    ∃ U V : Matrix (Fin 2) k ℂ,
      Tendsto (decayRun (fun X => mask Set.univ (X - 1)) η lam U₀ V₀) atTop (𝓝 (U, V)) ∧
      lift lam (mask Set.univ (U * Vᴴ - 1)) * stack U V = 0 ∧
      ¬ IsDecayedMin (fitLoss Set.univ (1 : Matrix (Fin 2) (Fin 2) ℂ)) lam U V := by
  obtain ⟨U, V, hlim, hstat⟩ := exists_tendsto_decayRun_fitLoss (Ω := Set.univ) hη hlam hκ0
    (frobeniusSq_one_fin_two.trans_le hκ) hρ hstep hdesc hw
  exact ⟨U, V, hlim, hstat, not_isDecayedMin_one_of_rank_le_one hlam.le hlam1 hrank hlim⟩

/-! The witnesses: a trained product exists and has rank two; and the zero start of width two,
whose stack has rank `0`, with `λ = ½`, `κ = 2`, `ρ = 8` and `η = 1/200`. -/

example : ∃ X : Matrix (Fin 2) (Fin 2) ℂ, 1 < X.rank := by
  obtain ⟨U, V, h⟩ := exists_isDecayedMin_fitLoss (Set.univ : Set (Fin 2 × Fin 2)) 1
    (lam := 1 / 2) (by norm_num)
  exact ⟨_, one_lt_rank_of_isNuclearMin_one (by norm_num) (by norm_num)
    (h.isNuclearMin (by norm_num))⟩

example := exists_tendsto_decayRun_one_not_isDecayedMin (k := Fin 2) (η := 1 / 200)
  (lam := 1 / 2) (κ := 2) (ρ := 8) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num) (U₀ := 0) (V₀ := 0)
  (by simp [frobeniusSq]) (by rw [stack_zero, Matrix.rank_zero]; exact zero_le_one)

end Transformer.Grokking.FlatSpectrum.Training.Scope
