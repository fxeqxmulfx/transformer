module

public import Transformer.SequenceModel.Attention.Backend.Refined.Basic

/-!
# Sharper certificates for the frozen CUDA trace

Source: `review/flat_spectrum.md`, *Trace-dependent execution bounds*, M3.
Residual checks remain at each operation's own recorded inputs. Squared
scale residuals and polynomial remainders are independently checked in Lean.
-/

@[expose] public section
noncomputable section
namespace Transformer.SequenceModel.Attention.Backend.Refined
open GPTMini Stability Execution
open scoped BigOperators

/-- M3: a fixed-order polynomial supplies a uniform analytic exp remainder on [-1,1]. -/
theorem exp_local (x e : ℝ) (hx : |x| ≤ 1)
    (hr : |e - expPolynomial x 12| ≤ 1 / 10000000000) :
    |e - Real.exp x| ≤ 1 / 100000000 := by
  have h := exponential_error x e 12 hx (by decide) (1 / 10000000000) hr
  have hp : |x| ^ 12 ≤ 1 := pow_le_one₀ (abs_nonneg x) hx
  norm_num at h
  linarith

/-- M3: each SiLU coordinate is certified against its recorded gate input. -/
theorem silu_local (x z e : ℝ) (hx : |x| ≤ 1) (he : 0 ≤ e)
    (hr : |e - expPolynomial (-x) 12| ≤ 1 / 10000000000)
    (hz : |z - x / (1 + e)| ≤ 1 / 50000000) :
    |z - Landscape.Neural.silu x| ≤ 3 / 100000000 := by
  have hE := exp_local (-x) e (by simpa using hx) hr
  have h := silu_residual_error x z e (1 / 50000000) (1 / 100000000) he hE hz
  linarith

/-- M3: all SiLU coordinates, including the source width-eleven hidden layer. -/
theorem silu_vector_local {d : ℕ} (hd : d ≤ 16) (x z : EucSpace d) (e : Fin d → ℝ)
    (hx : ∀ c, |x c| ≤ 1) (he : ∀ c, 0 ≤ e c)
    (hr : ∀ c, |e c - expPolynomial (-x c) 12| ≤ 1 / 10000000000)
    (hz : ∀ c, |z c - x c / (1 + e c)| ≤ 1 / 50000000) :
    ‖z - WithLp.toLp 2 (fun c => Landscape.Neural.silu (x c))‖ ≤ 12 / 100000000 := by
  have hs : Real.sqrt (d : ℝ) ≤ 4 := Real.sqrt_le_iff.mpr
    ⟨by norm_num, by exact_mod_cast hd⟩
  have h := norm_le_coordinates (z - WithLp.toLp 2 (fun c => Landscape.Neural.silu (x c)))
    (3 / 100000000) (by norm_num)
    (fun c => silu_local (x c) (z c) (e c) (hx c) (he c) (hr c) (hz c))
  nlinarith

/-- M3: each causal row is certified without exposing hidden SDPA weights. -/
theorem average_local (scores : Fin 2 → Fin 2 → ℝ) (i : Fin 2)
    (e : Fin 2 → ℝ) (v : Fin 2 → EucSpace 4) (z : EucSpace 4)
    (hx : ∀ j, j ≤ i → |scores i j| ≤ 1)
    (hr : ∀ j, j ≤ i → |e j - expPolynomial (scores i j) 12| ≤ 1 / 10000000000)
    (hA : (1 / 2 : ℝ) ≤ ∑ j, if j ≤ i then e j else 0)
    (hv : ∀ j c, |v j c| ≤ 1)
    (hz : ∀ c, |z c - ∑ j, ((if j ≤ i then e j else 0) /
      (∑ k, if k ≤ i then e k else 0)) * v j c| ≤ 1 / 20000000) :
    ‖z - ∑ j, causalWeight scores i j • v j‖ ≤ 3 / 10000000 := by
  let a : Fin 2 → ℝ := fun j => if j ≤ i then e j else 0
  have hpos : 0 < ∑ j, a j := by change 0 < ∑ j, if j ≤ i then e j else 0; linarith
  have hm : ∑ j, |a j - causalMass scores i j| ≤ 2 / 100000000 := by
    calc
      _ ≤ ∑ j : Fin 2, (1 / 100000000 : ℝ) := by
        apply Finset.sum_le_sum
        intro j hj
        by_cases hji : j ≤ i
        · simpa [a, causalMass, hji] using exp_local (scores i j) (e j) (hx j hji) (hr j hji)
        · norm_num [a, causalMass, hji]
      _ = _ := by norm_num
  have hn : ∀ j, ‖v j‖ ≤ 2 := by
    intro j
    have h := norm_le_coordinates (v j) 1 (by norm_num) (hv j)
    norm_num at h
    exact h
  have hz' : ‖z - ∑ j, (a j / (∑ k, a k)) • v j‖ ≤ 1 / 10000000 := by
    have h := norm_le_coordinates (z - ∑ j, (a j / (∑ k, a k)) • v j)
      (1 / 20000000) (by norm_num) (by simpa [a] using hz)
    norm_num at h ⊢
    exact h
  have h := causal_average_residual scores i a v z hpos
    (2 / 100000000) 2 (1 / 10000000) (by norm_num) hm hn hz'
  have hd : (2 * (2 / 100000000 : ℝ)) / (∑ j, a j) ≤ 1 / 10000000 := by
    apply (div_le_iff₀ hpos).mpr
    change (1 / 2 : ℝ) ≤ ∑ j, a j at hA
    linarith
  linarith

/-- M3: every independently captured cosine and sine within its rational
polynomial box satisfies the stored-table contract with tau=1e-7. -/
theorem rope_local (c s : Fin 2 → ℝ)
    (hc : ∀ k, |c k - (rotationPolynomial (![1, 1 / 100] k) 12).re| ≤ 4 / 100000000)
    (hs : ∀ k, |s k - (rotationPolynomial (![1, 1 / 100] k) 12).im| ≤ 4 / 100000000) :
    ∀ k, |c k - Real.cos (invFreq 4 10000 k)| ≤ 1 / 10000000 ∧
      |s k - Real.sin (invFreq 4 10000 k)| ≤ 1 / 10000000 := by
  intro k
  rw [head4_phases]
  have h := stored_rotation_error (![1, 1 / 100] k) (c k) (s k) (4 / 100000000) 12
    (by fin_cases k <;> norm_num) (hc k) (hs k)
  have hb : (4 / 100000000 : ℝ) + |![1, 1 / 100] k| ^ 12 / (Nat.factorial 12) * 2 ≤
      1 / 10000000 := by fin_cases k <;> norm_num
  exact ⟨h.1.trans hb, h.2.trans hb⟩


example := exp_local 0 1 (by norm_num) (by norm_num [expPolynomial, Finset.sum_range_succ])
example := silu_local 0 0 1 (by norm_num) (by norm_num)
  (by norm_num [expPolynomial, Finset.sum_range_succ]) (by norm_num)
example := silu_vector_local (d := 4) (by decide) 0 0 (fun _ => 1)
  (by intro c; norm_num) (by intro c; norm_num)
  (by intro c; norm_num [expPolynomial, Finset.sum_range_succ]) (by intro c; norm_num)
example := average_local (fun _ _ => 0) 1 (fun _ => 1)
  (fun _ => (0 : EucSpace 4)) 0
  (by intro j hj; norm_num) (by intro j hj; norm_num [expPolynomial, Finset.sum_range_succ])
  (by simp only [Fin.sum_univ_succ]; norm_num) (by intro j c; simp)
  (by intro c; simp)
example := rope_local (fun k => (rotationPolynomial (![1, 1 / 100] k) 12).re)
  (fun k => (rotationPolynomial (![1, 1 / 100] k) 12).im)
  (by intro k; norm_num) (by intro k; norm_num)

end Transformer.SequenceModel.Attention.Backend.Refined
