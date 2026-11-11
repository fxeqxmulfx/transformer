module

public import Transformer.Grokking.FlatSpectrum.Training.CrossEntropyRun
public import Transformer.Grokking.FlatSpectrum.Training.Scope.UnseenCellLimit

/-!
# One unseen cell of the kernel of signs: every small weight decay, and the run

Source: `review/flat_spectrum.md`, *Training with cross-entropy* (added), item XE15.
`UnseenCellLimit` treats sequences of trained products with a vanishing weight decay. It
treats every such sequence, so there is a threshold: for a `λ₀ > 0` and every weight decay
`0 < λ < λ₀`, every trained product of the three seen cells has `Re X(1, 1) < 0`
(`exists_pos_forall_re_unseen_neg`), and it decodes the unseen cell to the class of `−1`, not
to the class that the kernel shows there (`exists_pos_forall_unseen_wrong`).

The run of XE3 from stacked factors of full row rank converges to a decayed minimum, and the
product of a decayed minimum is a trained product. So for `0 < λ < λ₀`, under the two bounds
of XE3 on the step with the constants `a = 1`, `N = 3` and `|C| = 2`, the run ends at a
product with `Re X(1, 1) < 0` (`exists_pos_forall_tendsto_decayRun_unseen`). Such a run
exists: from `U₀ = [1 0]`, `V₀ = [0 1]` of width four, with a weight decay below `λ₀` and a
small step, the run converges, and its limit answers the unseen cell with the wrong class
(`exists_tendsto_decayRun_unseen_wrong`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training.Scope

open Matrix Filter Topology

universe u

/-- **For every small weight decay the unseen cell has a negative real part**
(`review/flat_spectrum.md`, item XE15): for a `λ₀ > 0` and every `0 < λ < λ₀`, every trained
product of the three seen cells of the kernel of signs has `Re X(1, 1) < 0`. -/
theorem exists_pos_forall_re_unseen_neg :
    ∃ lam₀ : ℝ, 0 < lam₀ ∧ ∀ lam : ℝ, 0 < lam → lam < lam₀ → ∀ X : Matrix (Fin 2) (Fin 2) ℂ,
      IsNuclearMin (ceLoss (codeProbe signCode threeCells) threeLabels) lam X →
        (X 1 1).re < 0 := by
  by_contra hno
  push Not at hno
  choose lams hpos hlt X hX hre using fun j : ℕ => hno (1 / ((j : ℝ) + 1)) (by positivity)
  have hlam : Tendsto lams atTop (𝓝 0) :=
    squeeze_zero (fun j => (hpos j).le) (fun j => (hlt j).le)
      tendsto_one_div_add_atTop_nhds_zero_nat
  obtain ⟨j, hj⟩ := (eventually_re_unseen_neg hpos hlam hX).exists
  exact absurd hj (not_lt.2 (hre j))

/-- **For every small weight decay the unseen cell is decoded to the wrong class**
(`review/flat_spectrum.md`, item XE15). -/
theorem exists_pos_forall_unseen_wrong :
    ∃ lam₀ : ℝ, 0 < lam₀ ∧ ∀ lam : ℝ, 0 < lam → lam < lam₀ → ∀ X : Matrix (Fin 2) (Fin 2) ℂ,
      IsNuclearMin (ceLoss (codeProbe signCode threeCells) threeLabels) lam X →
        ‖X 1 1 - signCode false‖ < ‖X 1 1 - signCode (signLabel (1, 1))‖ := by
  obtain ⟨lam₀, h₀, h⟩ := exists_pos_forall_re_unseen_neg
  refine ⟨lam₀, h₀, fun lam hlam hlt X hX => ?_⟩
  rw [show signLabel (1, 1) = true by decide]
  exact norm_sub_signCode_lt (h lam hlam hlt X hX)

/-- **The run ends at a product with a negative unseen cell** (`review/flat_spectrum.md`,
items XE3 and XE15): for `0 < λ < λ₀`, under the bounds of XE3 with `a = 1`, `N = 3` and
`|C| = 2`, the run from a start of full row rank converges, and the product of its limit has
`Re X(1, 1) < 0`. No convergence is assumed. -/
theorem exists_pos_forall_tendsto_decayRun_unseen :
    ∃ lam₀ : ℝ, 0 < lam₀ ∧ ∀ (k : Type u) [Fintype k] (η lam ρ : ℝ)
      (U₀ V₀ : Matrix (Fin 2) k ℂ), 0 < η → 0 < lam → lam < lam₀ →
      12 * Real.log 2 ≤ lam * ρ → 2 * η * (lam ^ 2 + 36) ≤ lam → η * (lam + 6 + 6 * ρ) ≤ 1 →
      frobeniusSq U₀ + frobeniusSq V₀ ≤ ρ → Function.Surjective (stack U₀ V₀).mulVec →
      ∃ U V : Matrix (Fin 2) k ℂ,
        Tendsto (decayRun (ceGrad (codeProbe signCode threeCells) threeLabels) η lam U₀ V₀)
          atTop (𝓝 (U, V)) ∧ ((U * Vᴴ) 1 1).re < 0 := by
  obtain ⟨lam₀, h₀, h⟩ := exists_pos_forall_re_unseen_neg
  refine ⟨lam₀, h₀, fun k _ η lam ρ U₀ V₀ hη hlam hlt hρ hstep hdesc hw hfull => ?_⟩
  obtain ⟨U, V, hlim, hmin⟩ := exists_tendsto_decayRun_ceLoss_isDecayedMin
    (A := codeProbe signCode threeCells) (y := threeLabels) (a := 1) (ρ := ρ) hη hlam
    zero_le_one (fun s c => by rw [frobeniusSq_codeProbe, norm_signCode])
    (by rw [Fintype.card_fin, Fintype.card_bool]; push_cast; linarith)
    (by rw [Fintype.card_fin]; push_cast; linarith)
    (by rw [Fintype.card_fin]; push_cast; linarith) hw hfull
  exact ⟨U, V, hlim, h lam hlam hlt _ (hmin.isNuclearMin hlam.le)⟩

/-- **A run from full row rank that answers the unseen cell with the wrong class**
(`review/flat_spectrum.md`, item XE15): for some weight decay and some step, the run on the
three seen cells of the kernel of signs from `U₀ = [1 0]`, `V₀ = [0 1]` converges, and the
codeword of the class of `−1` is nearer to the unseen cell of its limit than the codeword of
the class that the kernel shows there. -/
theorem exists_tendsto_decayRun_unseen_wrong :
    ∃ (η lam : ℝ) (U V : Matrix (Fin 2) (Fin 2 ⊕ Fin 2) ℂ), 0 < η ∧ 0 < lam ∧
      Tendsto (decayRun (ceGrad (codeProbe signCode threeCells) threeLabels) η lam
        (fromCols 1 0) (fromCols 0 1)) atTop (𝓝 (U, V)) ∧
        ‖(U * Vᴴ) 1 1 - signCode false‖ < ‖(U * Vᴴ) 1 1 - signCode (signLabel (1, 1))‖ := by
  obtain ⟨lam₀, h₀, h⟩ := exists_pos_forall_tendsto_decayRun_unseen.{0}
  have hlog := Real.log_pos one_lt_two
  have hlam : 0 < lam₀ / 2 := by positivity
  have hρ : (4 : ℝ) ≤ 4 + 12 * Real.log 2 / (lam₀ / 2) := le_add_of_nonneg_right (by positivity)
  have hη : 0 < min (lam₀ / 2 / (2 * ((lam₀ / 2) ^ 2 + 36)))
      (1 / (lam₀ / 2 + 6 + 6 * (4 + 12 * Real.log 2 / (lam₀ / 2)))) :=
    lt_min (by positivity) (by positivity)
  obtain ⟨U, V, hlim, hre⟩ := h (Fin 2 ⊕ Fin 2) _ (lam₀ / 2) _ (fromCols 1 0) (fromCols 0 1)
    hη hlam (by linarith)
    (by rw [mul_add, mul_div_cancel₀ _ hlam.ne']; linarith)
    (by
      have h1 := min_le_left (lam₀ / 2 / (2 * ((lam₀ / 2) ^ 2 + 36)))
        (1 / (lam₀ / 2 + 6 + 6 * (4 + 12 * Real.log 2 / (lam₀ / 2))))
      rw [le_div_iff₀ (by positivity)] at h1
      linarith)
    ((le_div_iff₀ (by positivity)).1 (min_le_right _ _))
    (by
      rw [← frobeniusSq_stack, stack_fromCols_one]
      refine le_trans (le_of_eq ?_) hρ
      norm_num [frobeniusSq, Matrix.one_apply])
    fun v => ⟨v, by rw [stack_fromCols_one, one_mulVec]⟩
  refine ⟨_, _, U, V, hη, hlam, hlim, ?_⟩
  rw [show signLabel (1, 1) = true by decide]
  exact norm_sub_signCode_lt hre

end Transformer.Grokking.FlatSpectrum.Training.Scope
