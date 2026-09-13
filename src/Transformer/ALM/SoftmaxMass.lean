/-
# Where the mass on the tie comes from

Every tie bound in this development takes `hmass : 1 - ε ≤ w b + w c` as a
hypothesis — `softmax_output_close_level`, `softmax_head_resolves_average`,
`softmax_head_resolves_latest`, and through them every head statement down to
`fp_head_tie_resolves`.  Nothing discharged it.  For a single winner the
corresponding bound is a theorem about the data (`softmax_winner_ge`,
`softmax_winner_lengthfree`, and the sharp forms), used in five places; for a
tie the same fact was assumed each time, so the head bounds held of a
hypothesis about the softmax rather than of a hypothesis about the keys.

`softmax_tie_mass` is the sharp form: whatever the losing keys contribute
relative to the tied score, the two tied weights together carry all but that.
`softmax_tie_mass_ge` is it in the form the data supplies — a score gap `δ`
below the tied value gives `ε = (n-2)·e^{-βδ}/2` — and
`softmax_head_resolves_average_of_gap` is the head bound with that `ε` put in,
so on a genuine tie with a genuine gap nothing about the softmax is assumed at
all.

Source: `transformer_vm/attention/hull2d_cht.h`, lines 70-83; the softmax
side is the standard estimate of `Transformer.ALM.Softmax`.
-/

import Transformer.ALM.SoftmaxTie

open scoped BigOperators

namespace Transformer
namespace ALM

variable {n : ℕ}

/-- **The tied pair carries everything the losers do not.**  With both scores
equal to `σ`, the two weights sum to `2A/(2A+R)` for `A = exp(βσ)` and `R` the
losers' contribution, so a bound `R ≤ 2εA` is a bound `1 - ε` from below on
the pair.  This is `softmax_winner_sharp` with the winner doubled. -/
theorem softmax_tie_mass (β : ℝ) (s : Fin n → ℝ) (b c : Fin n)
    (hbc : b ≠ c) (σ : ℝ) (hb : s b = σ) (hc : s c = σ) (ε : ℝ)
    (htail : ∑ j ∈ (Finset.univ.erase b).erase c, Real.exp (β * s j)
        ≤ 2 * ε * Real.exp (β * σ)) :
    1 - ε ≤ Real.exp (β * s b) / ∑ k, Real.exp (β * s k)
        + Real.exp (β * s c) / ∑ k, Real.exp (β * s k) := by
  set A := Real.exp (β * σ) with hA
  have hApos : 0 < A := Real.exp_pos _
  set R := ∑ j ∈ (Finset.univ.erase b).erase c, Real.exp (β * s j) with hR
  have hRnn : 0 ≤ R := Finset.sum_nonneg fun j _ => (Real.exp_pos _).le
  have hD : ∑ k, Real.exp (β * s k) = 2 * A + R := by
    rw [← Finset.add_sum_erase _ (fun j => Real.exp (β * s j)) (Finset.mem_univ b),
      ← Finset.add_sum_erase _ (fun j => Real.exp (β * s j))
        (Finset.mem_erase.mpr ⟨hbc.symm, Finset.mem_univ c⟩), hb, hc, hA]
    ring
  have hDpos : 0 < 2 * A + R := by linarith
  have hεnn : 0 ≤ ε := by
    have h2 : 0 ≤ 2 * ε * A := le_trans hRnn htail
    nlinarith
  have hsum : A / (2 * A + R) + A / (2 * A + R) = (A + A) / (2 * A + R) := by ring
  rw [hb, hc, ← hA, hD, hsum, le_div_iff₀ hDpos]
  nlinarith

/-- The hypothesis is satisfiable: two keys and nothing else, so the losers
contribute nothing and the pair carries all the mass. -/
example : (1 : ℝ) - 0 ≤ Real.exp (1 * (fun _ : Fin 2 => (0 : ℝ)) 0)
      / ∑ k : Fin 2, Real.exp (1 * (fun _ : Fin 2 => (0 : ℝ)) k)
    + Real.exp (1 * (fun _ : Fin 2 => (0 : ℝ)) 1)
      / ∑ k : Fin 2, Real.exp (1 * (fun _ : Fin 2 => (0 : ℝ)) k) := by
  refine softmax_tie_mass 1 (fun _ : Fin 2 => (0 : ℝ)) 0 1 (by decide) 0 rfl rfl 0 ?_
  rw [show ((Finset.univ.erase (0 : Fin 2)).erase 1) = ∅ from by decide]
  simp

/-- **And a score gap supplies the bound.**  If every key outside the tie
scores at least `δ` below it, each of the `n - 2` losing terms is at most
`e^{-βδ}` times the tied one, so the pair carries `1 - (n-2)e^{-βδ}/2`.  This
is the two-key form of `softmax_winner_ge`: a statement about the keys, not
about the softmax. -/
theorem softmax_tie_mass_ge (β : ℝ) (hβ : 0 ≤ β) (s : Fin n → ℝ) (b c : Fin n)
    (hbc : b ≠ c) (σ : ℝ) (hb : s b = σ) (hc : s c = σ) (δ : ℝ)
    (hgap : ∀ j, j ≠ b → j ≠ c → s j + δ ≤ σ) :
    1 - ((n : ℝ) - 2) * Real.exp (-(β * δ)) / 2
      ≤ Real.exp (β * s b) / ∑ k, Real.exp (β * s k)
        + Real.exp (β * s c) / ∑ k, Real.exp (β * s k) := by
  have hcard : ((Finset.univ.erase b).erase c).card = n - 2 := by
    rw [Finset.card_erase_of_mem (Finset.mem_erase.mpr ⟨hbc.symm, Finset.mem_univ c⟩),
      Finset.card_erase_of_mem (Finset.mem_univ b), Finset.card_univ, Fintype.card_fin]
    omega
  have hn : 2 ≤ n := by
    have h2 : ({b, c} : Finset (Fin n)).card ≤ Fintype.card (Fin n) := Finset.card_le_univ _
    rwa [Finset.card_pair hbc, Fintype.card_fin] at h2
  refine softmax_tie_mass β s b c hbc σ hb hc _ ?_
  have hterm : ∀ j ∈ (Finset.univ.erase b).erase c,
      Real.exp (β * s j) ≤ Real.exp (-(β * δ)) * Real.exp (β * σ) := by
    intro j hj
    have hjc : j ≠ c := (Finset.mem_erase.mp hj).1
    have hjb : j ≠ b := (Finset.mem_erase.mp (Finset.mem_erase.mp hj).2).1
    rw [← Real.exp_add]
    refine Real.exp_le_exp.mpr ?_
    have := hgap j hjb hjc
    nlinarith
  calc ∑ j ∈ (Finset.univ.erase b).erase c, Real.exp (β * s j)
      ≤ (((Finset.univ.erase b).erase c).card : ℝ)
          * (Real.exp (-(β * δ)) * Real.exp (β * σ)) := by
        simpa using Finset.sum_le_card_nsmul _ _ _ hterm
    _ = 2 * (((n : ℝ) - 2) * Real.exp (-(β * δ)) / 2) * Real.exp (β * σ) := by
        rw [hcard, Nat.cast_sub hn]
        push_cast
        ring

/-- The hypotheses are satisfiable: two tied keys with no third key to lose,
where the gap condition is vacuous and the bound is `1`. -/
example : (1 : ℝ) - (((2 : ℕ) : ℝ) - 2) * Real.exp (-(1 * 1)) / 2
      ≤ Real.exp (1 * (fun _ : Fin 2 => (0 : ℝ)) 0)
        / ∑ k : Fin 2, Real.exp (1 * (fun _ : Fin 2 => (0 : ℝ)) k)
      + Real.exp (1 * (fun _ : Fin 2 => (0 : ℝ)) 1)
        / ∑ k : Fin 2, Real.exp (1 * (fun _ : Fin 2 => (0 : ℝ)) k) :=
  softmax_tie_mass_ge 1 zero_le_one (fun _ : Fin 2 => (0 : ℝ)) 0 1 (by decide) 0 rfl rfl 1
    (fun j hj0 hj1 => by fin_cases j <;> simp_all)

/-! ### And so the head bound assumes nothing about the softmax -/

/-- **The head's tie bound, priced by the keys.**  `softmax_head_resolves_average`
with its mass hypothesis discharged: two keys tie, every other key scores at
least `δ` lower, and the head's output is within `((n-2)e^{-βδ}/2)·C` of what
`resolve` writes out under `TieBreak::AVERAGE`.  Every remaining hypothesis is
about the keys, the payloads, or the log — none about the softmax itself. -/
theorem softmax_head_resolves_average_of_gap [Nonempty (Fin n)] (β : ℝ) (hβ : 0 ≤ β)
    (s : Fin n → ℝ) (V : Fin n → ℝ × ℝ) (b c : Fin n) (hbc : b ≠ c)
    (σ : ℝ) (hb : s b = σ) (hc : s c = σ) (δ : ℝ)
    (hgap : ∀ j, j ≠ b → j ≠ c → s j + δ ≤ σ)
    (M : ℕ → Meta) (p r : ℕ) (sp sr : ℤ) (hsp : 0 ≤ sp) (hsr : 0 ≤ sr)
    (hMp : M p = Meta.empty.add (V b) sp) (hMr : M r = Meta.empty.add (V c) sr)
    (C : ℝ) (hC : ∀ j, ‖V j - (((V b).1 + (V c).1) / 2, ((V b).2 + (V c).2) / 2)‖ ≤ C) :
    ‖(∑ j, (Real.exp (β * s j) / ∑ k, Real.exp (β * s k)) • V j)
        - (scanCombined M p r).resolveAverage‖
      ≤ (((n : ℝ) - 2) * Real.exp (-(β * δ)) / 2) * C :=
  softmax_head_resolves_average β s V b c hbc σ hb hc M p r sp sr hsp hsr hMp hMr _ C
    (softmax_tie_mass_ge β hβ s b c hbc σ hb hc δ hgap) hC

/-- The hypotheses are satisfiable, and the bound is then exact: two keys with
the same payload, no third key, so the gap condition is vacuous, the tie
carries all the mass and the head returns the resolved value itself. -/
example :
    ‖(∑ _j : Fin 2, (Real.exp (1 * (0 : ℝ)) / ∑ _k : Fin 2, Real.exp (1 * (0 : ℝ)))
          • ((0 : ℝ), (0 : ℝ)))
        - Meta.resolveAverage
            (scanCombined (fun _ : ℕ => Meta.empty.add ((0 : ℝ), (0 : ℝ)) 0) 0 0)‖
      ≤ ((((2 : ℕ) : ℝ) - 2) * Real.exp (-(1 * 1)) / 2) * 0 :=
  softmax_head_resolves_average_of_gap 1 zero_le_one (fun _ : Fin 2 => (0 : ℝ))
    (fun _ => ((0 : ℝ), (0 : ℝ))) 0 1 (by decide) 0 rfl rfl 1
    (fun j hj0 hj1 => by fin_cases j <;> simp_all)
    (fun _ : ℕ => Meta.empty.add ((0 : ℝ), (0 : ℝ)) 0) 0 0 0 0 le_rfl le_rfl rfl rfl 0
    (fun j => by norm_num)

end ALM
end Transformer
