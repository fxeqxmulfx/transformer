module

public import Transformer.Grokking.FlatSpectrum.Training.Scope.SeparateCellsRun
public import Transformer.Grokking.FlatSpectrum.Training.Scope.SignRate

/-!
# Samples of the code `±1` in distinct rows and columns: a rate

Source: `review/flat_spectrum.md`, *Training with cross-entropy* (added), item XE20,
statement 4; the items "Not covered" of XE17 and XE19.

Let no two samples lie in one row or in one column, let `η > 0` be any step, and let
`U_i + y V_j ≠ 0` at the rows of every sample. The loss along the run without weight decay is
the sum of the losses of the sign runs of the samples (`ceLoss_decayRun_signCode`), and each
of them is at most `1 / (1 + η a (t − T))` from its step `T` on, with its `a > 0` (XE19,
statement 5). So with the latest of these steps and the least of these `a`, from some step
`T` on the loss is at most `|S| / (1 + c (t − T))` for a `c > 0`
(`exists_ceLoss_decayRun_signCode_le`): it falls at least like `1 / t`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training.Scope

open Matrix Filter Topology

variable {m n k S : Type*} [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n] [Fintype k]
  [Fintype S] {cell : S → m × n} {y : S → Bool} {η : ℝ}

local notation "RC" => decayRun (ceGrad (codeProbe signCode cell) y)

/-- **Without weight decay the loss falls at least like `1 / t`** (`review/flat_spectrum.md`,
item XE20, statement 4): for samples of the code `±1`, no two in one row or in one column,
every step `η > 0` and every start with `U_i + y V_j ≠ 0` at the rows of every sample, from
some step `T` on the cross-entropy is at most `|S| / (1 + c (t − T))`, for a `c > 0`. -/
theorem exists_ceLoss_decayRun_signCode_le [Nonempty S]
    (hrow : Function.Injective fun s => (cell s).1)
    (hcol : Function.Injective fun s => (cell s).2) (hη : 0 < η) {U₀ : Matrix m k ℂ}
    {V₀ : Matrix n k ℂ}
    (h : ∀ s, ∃ b, U₀ (cell s).1 b + classSign (y s) • V₀ (cell s).2 b ≠ 0) :
    ∃ T : ℕ, ∃ c : ℝ, 0 < c ∧ ∀ t : ℕ, T ≤ t →
      ceLoss (codeProbe signCode cell) y ((RC η 0 U₀ V₀ t).1 * (RC η 0 U₀ V₀ t).2ᴴ) ≤
        Fintype.card S / (1 + c * (t - T)) := by
  choose T haT hT using fun s => exists_ceLoss_signProduct_le hη
    ((rowOf_add_ne_zero_iff _ _ _ U₀ V₀).2 (h s))
  obtain ⟨a, ha⟩ : ∃ a : S → ℝ, ∀ s, a s = frobeniusSq
      ((decayRun (ceGrad signProbe fun _ => true) η 0 (rowOf (cell s).1 U₀)
        (classSign (y s) • rowOf (cell s).2 V₀) (T s)).1 +
      (decayRun (ceGrad signProbe fun _ => true) η 0 (rowOf (cell s).1 U₀)
        (classSign (y s) • rowOf (cell s).2 V₀) (T s)).2) := ⟨_, fun _ => rfl⟩
  simp only [← ha] at haT hT
  obtain ⟨s₀, -, hs₀⟩ := Finset.exists_min_image Finset.univ a Finset.univ_nonempty
  refine ⟨Finset.univ.sup T, η * a s₀, mul_pos hη (haT s₀), fun t ht => ?_⟩
  have h1 : (0 : ℝ) ≤ (t : ℝ) - (Finset.univ.sup T : ℕ) := sub_nonneg.2 (by exact_mod_cast ht)
  have hpos : 0 < 1 + η * a s₀ * ((t : ℝ) - (Finset.univ.sup T : ℕ)) := by
    have := mul_nonneg (mul_pos hη (haT s₀)).le h1
    linarith
  have hterm : ∀ s, ceLoss signProbe (fun _ => true)
      (signProduct η (rowOf (cell s).1 U₀) (classSign (y s) • rowOf (cell s).2 V₀) t) ≤
        1 / (1 + η * a s₀ * ((t : ℝ) - (Finset.univ.sup T : ℕ))) := fun s => by
    have hTs : T s ≤ Finset.univ.sup T := Finset.le_sup (Finset.mem_univ s)
    have h2 : (T s : ℝ) ≤ (Finset.univ.sup T : ℕ) := by exact_mod_cast hTs
    refine (hT s t (hTs.trans ht)).trans (one_div_le_one_div_of_le hpos ?_)
    have h3 : η * a s₀ * ((t : ℝ) - (Finset.univ.sup T : ℕ)) ≤ η * a s * ((t : ℝ) - T s) :=
      mul_le_mul (mul_le_mul_of_nonneg_left (hs₀ s (Finset.mem_univ s)) hη.le) (by linarith) h1
        (mul_pos hη (haT s)).le
    linarith
  rw [ceLoss_decayRun_signCode hrow hcol]
  calc ∑ s, ceLoss signProbe (fun _ => true)
        (signProduct η (rowOf (cell s).1 U₀) (classSign (y s) • rowOf (cell s).2 V₀) t)
      ≤ ∑ _s : S, 1 / (1 + η * a s₀ * ((t : ℝ) - (Finset.univ.sup T : ℕ))) :=
        Finset.sum_le_sum fun s _ => hterm s
    _ = Fintype.card S / (1 + η * a s₀ * ((t : ℝ) - (Finset.univ.sup T : ℕ))) := by
        rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
        ring

/-! The witness: the two samples at the diagonal cells of a `2 × 2` product with the labels
`(+, −)`, the start `U₀ = 1`, `V₀ = 0` and the step `1/10`. -/

example := exists_ceLoss_decayRun_signCode_le (y := ![true, false]) injective_diagCell_fst
  injective_diagCell_snd (η := 1 / 10) (by norm_num) (diagCell_one_add_zero_ne ![true, false])

end Transformer.Grokking.FlatSpectrum.Training.Scope
