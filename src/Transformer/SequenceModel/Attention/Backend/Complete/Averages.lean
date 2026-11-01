module

public import Transformer.SequenceModel.Attention.Backend.Complete.Basic

/-!
# Polynomial certificates at every activation and causal average

Source: `review/flat_spectrum.md`, *Complete captured execution*, L2.
These adapters retain the two-token causal mask and work with the captured
score tensor, its own value tensor and independent polynomial witnesses.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Backend.Complete

open GPTMini Stability
open scoped BigOperators

/-- L2: all SiLU coordinates, including the source width-eleven hidden layer. -/
theorem silu_vector_checked {d : ℕ} (hd : d ≤ 16) (x z : EucSpace d) (e : Fin d → ℝ)
    (hx : ∀ c, |x c| ≤ 1) (he : ∀ c, 0 ≤ e c)
    (hr : ∀ c, |e c - expPolynomial (-x c) 12| ≤ 1 / 10000000000)
    (hz : ∀ c, |z c - x c / (1 + e c)| ≤ 1 / 1000000) :
    ‖z - WithLp.toLp 2 (fun c => Landscape.Neural.silu (x c))‖ ≤ 1 / 10000 := by
  have hs : Real.sqrt (d : ℝ) ≤ 4 := Real.sqrt_le_iff.mpr
    ⟨by norm_num, by exact_mod_cast hd⟩
  have h := norm_le_coordinates (z - WithLp.toLp 2 (fun c => Landscape.Neural.silu (x c)))
    (2 / 1000000) (by norm_num)
    (fun c => silu_checked (x c) (z c) (e c) (hx c) (he c) (hr c) (hz c))
  nlinarith

/-- L2: each causal row is certified without exposing hidden SDPA weights. -/
theorem average_checked (scores : Fin 2 → Fin 2 → ℝ) (i : Fin 2)
    (e : Fin 2 → ℝ) (v : Fin 2 → EucSpace 4) (z : EucSpace 4)
    (hx : ∀ j, j ≤ i → |scores i j| ≤ 1)
    (hr : ∀ j, j ≤ i → |e j - expPolynomial (scores i j) 12| ≤ 1 / 10000000000)
    (hA : (1 / 2 : ℝ) ≤ ∑ j, if j ≤ i then e j else 0)
    (hv : ∀ j c, |v j c| ≤ 1)
    (hz : ∀ c, |z c - ∑ j, ((if j ≤ i then e j else 0) /
      (∑ k, if k ≤ i then e k else 0)) * v j c| ≤ 1 / 1000000) :
    ‖z - ∑ j, causalWeight scores i j • v j‖ ≤ 1 / 10000 := by
  let a : Fin 2 → ℝ := fun j => if j ≤ i then e j else 0
  have hpos : 0 < ∑ j, a j := by change 0 < ∑ j, if j ≤ i then e j else 0; linarith
  have hm : ∑ j, |a j - causalMass scores i j| ≤ 2 / 100000000 := by
    calc
      _ ≤ ∑ j : Fin 2, (1 / 100000000 : ℝ) := by
        apply Finset.sum_le_sum
        intro j hj
        by_cases hji : j ≤ i
        · simpa [a, causalMass, hji] using exp_checked (scores i j) (e j) (hx j hji) (hr j hji)
        · norm_num [a, causalMass, hji]
      _ = _ := by norm_num
  have hn : ∀ j, ‖v j‖ ≤ 2 := by
    intro j
    have h := norm_le_coordinates (v j) 1 (by norm_num) (hv j)
    norm_num at h
    exact h
  have hz' : ‖z - ∑ j, (a j / (∑ k, a k)) • v j‖ ≤ 2 / 1000000 := by
    have h := norm_le_coordinates (z - ∑ j, (a j / (∑ k, a k)) • v j)
      (1 / 1000000) (by norm_num) (by simpa [a] using hz)
    norm_num at h ⊢
    exact h
  have h := causal_average_residual scores i a v z hpos
    (2 / 100000000) 2 (2 / 1000000) (by norm_num) hm hn hz'
  have hd : (2 * (2 / 100000000 : ℝ)) / (∑ j, a j) ≤ 1 / 1000000 := by
    apply (div_le_iff₀ hpos).mpr
    change (1 / 2 : ℝ) ≤ ∑ j, a j at hA
    linarith
  linarith

example := silu_vector_checked (d := 4) (by decide) 0 0 (fun _ => 1)
  (by intro c; norm_num) (by intro c; norm_num)
  (by intro c; norm_num [expPolynomial, Finset.sum_range_succ]) (by intro c; norm_num)
example := average_checked (fun _ _ => 0) 1 (fun _ => 1)
  ![EuclideanSpace.single 0 1, EuclideanSpace.single 1 1]
  (WithLp.toLp 2 ![1 / 2, 1 / 2, 0, 0])
  (by intro j hj; norm_num) (by intro j hj; norm_num [expPolynomial, Finset.sum_range_succ])
  (by simp only [Fin.sum_univ_succ]; norm_num)
  (by intro j c; fin_cases j <;> fin_cases c <;> norm_num)
  (by intro c; fin_cases c <;> simp only [Fin.sum_univ_succ] <;> norm_num)

end Transformer.SequenceModel.Attention.Backend.Complete
