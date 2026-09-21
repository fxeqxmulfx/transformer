import Transformer.AdamBeyond.Section5_Lemma
import Transformer.AdamBeyond.Section4_Abel

/-
# Adam and beyond — §5: Theorem 5

Theorem 5 of arXiv:1904.09237, the regret of AdamNC, and its proof (appendix,
§"Proof of Theorem 5"): the bound of Lemma 3.1 of arXiv:1904.03590
(`prepare_lem`), Abel summation of its first term by `abel_beta_le`
(`Section4_Abel`) with `a_t = √v_{t,i}/α_t`, non-decreasing by condition 2,
and the lemma of the proof (`adamNC_moment_sum`, `Section5_Lemma`) for its
second.

**What the source says and what is carried here.**

* AdamNC is the run with `β₂ = 0` and the rule `adamNCRule β₂`
  (`Section5_AdamNC`); `v_t` of the source is `v̂_t` of the run.

* Condition 1 of Theorem 5 reads `(1/α_T) √v_{t,i} ≥ (1/ζ) ‖g_{1:t,i}‖₂` for
  all `t ∈ [T]`.  The proof uses it at every `t` with `α_t` in place of `α_T`
  ("using similar argument for all time steps"), and with `α_T` it does not
  give that, since `1/α_t ≤ 1/α_T`; condition 1 is stated with `α_t`.  Its sum
  `Σ_j Π_k β_{2(t-k+1)}(1-β_{2j}) g²_{j,i}` is `v_{t,i}` (`adamNC_vhat_sum`).
  Implicit in the source and made explicit: `0 ≤ β_{1,t}`, `β₁ < 1`, `α > 0`.

* The source's `β₂ = 0`, which makes the run AdamNC, and its implicit
  `0 ≤ β_{2,t} ≤ 1` are not used by the proof: conditions 1 and 2 carry all it
  needs of `v_t`, including `m_{t,i} = 0` wherever `v_{t,i} = 0`, which
  Lemma 3.1 asks for.  The theorem is stated without them.

Not transcribed: "one can generalize the result to deal with the case where
[`Γ_t ⪰ 0`] is violated as long as the violation is not too large or
frequent", which names no statement.

Source: arXiv:1904.09237, §5, Theorem 5 (thm:adamnc-proof); appendix,
§"Proof of Theorem 5".
-/

open Finset

namespace Transformer
namespace AdamBeyond

open AMSGrad

variable {d : ℕ}

/-- **Theorem 5**, with condition 1 at `α_t`.  For AdamNC with `α_t = α/√t`,
`0 ≤ β_{1,t} ≤ β₁ = β_{1,1} < 1`, under the standing assumptions, if for some
`ζ > 0`
1. `√v_{t,i}/α_t ≥ ‖g_{1:t,i}‖₂/ζ` for all `t ∈ [T]` and `i`, and
2. `√v_{t,i}/α_t ≥ √v_{t-1,i}/α_{t-1}` for all `t ∈ {2, …, T}` and `i`,

then for every `x* ∈ F`

  `R_T ≤ D²/(2α(1-β₁)) Σᵢ √T √v_{T,i} + D²/(1-β₁)² Σ_{t=1}^T Σᵢ β_{1,t}√v_{t,i}/α_t`
  `     + 2ζ/(1-β₁)³ Σᵢ ‖g_{1:T,i}‖₂`.

Source: arXiv:1904.09237, §5, Theorem 5 (thm:adamnc-proof). -/
theorem adamNC_regret {S : Setup d} {F : Set (Vec d)} {D G : ℝ} (hS : IsOnlineConvex S F D G)
    {β₂ : ℕ → ℝ} {α : ℝ} (hα : 0 < α) (hαt : S.α = fun t : ℕ => α / Real.sqrt t)
    (hβ₁ : ∀ t, 1 ≤ t → 0 ≤ S.β₁ t ∧ S.β₁ t ≤ S.β₁ 1) (hβ₁' : S.β₁ 1 < 1)
    (T : ℕ) {ζ : ℝ} (hζ : 0 < ζ)
    (h₁ : ∀ t ∈ Icc 1 T, ∀ i,
      S.gnorm (adamNCRule β₂) t i / ζ ≤ Real.sqrt (S.vhat (adamNCRule β₂) t i) / S.α t)
    (h₂ : ∀ t ∈ Icc 2 T, ∀ i,
      Real.sqrt (S.vhat (adamNCRule β₂) (t - 1) i) / S.α (t - 1) ≤
        Real.sqrt (S.vhat (adamNCRule β₂) t i) / S.α t)
    {xstar : Vec d} (hxstar : xstar ∈ F) :
    S.regret (adamNCRule β₂) xstar T ≤
      D ^ 2 / (2 * α * (1 - S.β₁ 1)) *
          ∑ i, Real.sqrt T * Real.sqrt (S.vhat (adamNCRule β₂) T i)
      + D ^ 2 / (1 - S.β₁ 1) ^ 2 *
          ∑ t ∈ Icc 1 T, ∑ i, S.β₁ t * Real.sqrt (S.vhat (adamNCRule β₂) t i) / S.α t
      + 2 * ζ / (1 - S.β₁ 1) ^ 3 * ∑ i, S.gnorm (adamNCRule β₂) T i := by
  set R : Rule d := adamNCRule β₂
  rcases Nat.eq_zero_or_pos T with rfl | hT
  · simp [Setup.regret, Setup.gnorm]
  have hα' : ∀ t, 1 ≤ t → 0 < S.α t := fun t ht => by
    rw [hαt]; exact div_pos hα (Real.sqrt_pos.2 (by exact_mod_cast ht))
  have hB : 0 < 1 - S.β₁ 1 := by linarith
  -- `m_{t,i} = 0` wherever `v_{t,i} = 0`, by condition 1
  have hz : ∀ n ≤ T, ∀ i, Real.sqrt (S.vhat R n i) = 0 → S.m R n i = 0 := by
    intro n hn i h
    rcases Nat.eq_zero_or_pos n with rfl | hn0
    · rfl
    have hc := h₁ n (mem_Icc.2 ⟨hn0, hn⟩) i
    rw [h, zero_div, div_le_iff₀ hζ, zero_mul] at hc
    have hg : S.gnorm R n i = 0 := le_antisymm hc (Real.sqrt_nonneg _)
    have hm := m_sq_le_gnorm (R := R) hβ₁ hβ₁' i n
    rw [hg, zero_mul] at hm
    exact pow_eq_zero_iff two_ne_zero |>.1 (le_antisymm hm (sq_nonneg _))
  have hp := prepare_lem hS hα' hβ₁ hβ₁' hT hz hxstar
  have hi := fun i : Fin d => abel_beta_le
    (fun t => Real.sqrt (S.vhat R t i) / S.α t) S.β₁
    (fun t => (S.x R t i - xstar i) ^ 2) (E := D ^ 2) hT hβ₁'
    (fun t ht => div_nonneg (Real.sqrt_nonneg _) (hα' t (mem_Icc.1 ht).1).le)
    (fun t ht => h₂ t ht i)
    (fun t ht => hβ₁ t (mem_Icc.1 ht).1)
    (fun t _ => ⟨sq_nonneg _, by
      rw [← sq_abs]
      exact pow_le_pow_left₀ (abs_nonneg _) (hS.diam _ (x_mem hS _ t) _ hxstar i) 2⟩)
  have hmain : ∑ i, ∑ t ∈ Icc 1 T, Real.sqrt (S.vhat R t i) / (2 * S.α t * (1 - S.β₁ t)) *
          ((S.x R t i - xstar i) ^ 2 - (S.x R (t + 1) i - xstar i) ^ 2)
      + ∑ i, ∑ t ∈ Icc 2 T, S.β₁ t * Real.sqrt (S.vhat R (t - 1) i) /
          (2 * S.α (t - 1) * (1 - S.β₁ 1)) * (S.x R t i - xstar i) ^ 2 ≤
      ∑ i, D ^ 2 * (Real.sqrt (S.vhat R T i) / S.α T) / (2 * (1 - S.β₁ 1))
      + ∑ i, D ^ 2 / (1 - S.β₁ 1) * ∑ t ∈ Icc 2 T,
          S.β₁ t * (Real.sqrt (S.vhat R (t - 1) i) / S.α (t - 1)) := by
    rw [← sum_add_distrib, ← sum_add_distrib]
    refine sum_le_sum fun i _ => le_trans (le_of_eq ?_) (hi i)
    congr 1 <;> refine sum_congr rfl fun t _ => ?_
    · generalize 1 - S.β₁ t = q; ring
    · generalize 1 - S.β₁ 1 = q; ring
  have h1 : ∑ i, D ^ 2 * (Real.sqrt (S.vhat R T i) / S.α T) / (2 * (1 - S.β₁ 1)) =
      D ^ 2 / (2 * α * (1 - S.β₁ 1)) * ∑ i, Real.sqrt T * Real.sqrt (S.vhat R T i) := by
    rw [mul_sum]
    refine sum_congr rfl fun i _ => ?_
    rw [hαt]; simp only; rw [div_div_eq_mul_div]; generalize 1 - S.β₁ 1 = q; ring
  have hcoef : D ^ 2 / (1 - S.β₁ 1) ≤ D ^ 2 / (1 - S.β₁ 1) ^ 2 :=
    div_le_div_of_nonneg_left (sq_nonneg D) (by positivity) (by nlinarith [(hβ₁ 1 le_rfl).1])
  have h2 : ∑ i, D ^ 2 / (1 - S.β₁ 1) * ∑ t ∈ Icc 2 T,
        S.β₁ t * (Real.sqrt (S.vhat R (t - 1) i) / S.α (t - 1)) ≤
      D ^ 2 / (1 - S.β₁ 1) ^ 2 *
        ∑ t ∈ Icc 1 T, ∑ i, S.β₁ t * Real.sqrt (S.vhat R t i) / S.α t := by
    rw [sum_comm, mul_sum]
    refine sum_le_sum fun i _ => mul_le_mul hcoef ?_ ?_ (by positivity)
    · refine (sum_le_sum fun t ht => ?_).trans (sum_le_sum_of_subset_of_nonneg
        (Icc_subset_Icc_left one_le_two) fun t ht _ => ?_)
      · rw [mul_div_assoc]
        exact mul_le_mul_of_nonneg_left (h₂ t ht i)
          (hβ₁ t (by have := (mem_Icc.1 ht).1; omega)).1
      · exact div_nonneg (mul_nonneg (hβ₁ t (mem_Icc.1 ht).1).1 (Real.sqrt_nonneg _))
          (hα' t (mem_Icc.1 ht).1).le
    · refine sum_nonneg fun t ht => mul_nonneg
        (hβ₁ t (by have := (mem_Icc.1 ht).1; omega)).1 (div_nonneg (Real.sqrt_nonneg _) ?_)
      exact (hα' (t - 1) (by have := (mem_Icc.1 ht).1; omega)).le
  have h3 : ∑ i, ∑ t ∈ Icc 1 T, S.α t / (1 - S.β₁ 1) * S.m R t i ^ 2 /
      Real.sqrt (S.vhat R t i) ≤ 2 * ζ / (1 - S.β₁ 1) ^ 3 * ∑ i, S.gnorm R T i := by
    have e : ∑ i, ∑ t ∈ Icc 1 T, S.α t / (1 - S.β₁ 1) * S.m R t i ^ 2 /
        Real.sqrt (S.vhat R t i) = (∑ t ∈ Icc 1 T, S.α t *
          ∑ i, S.m R t i ^ 2 / Real.sqrt (S.vhat R t i)) / (1 - S.β₁ 1) := by
      rw [sum_comm, sum_div]
      refine sum_congr rfl fun t _ => ?_
      rw [mul_sum, sum_div]
      exact sum_congr rfl fun i _ => by ring
    rw [e, div_le_iff₀ hB]
    refine (adamNC_moment_sum hα' hβ₁ hβ₁' T hζ h₁).trans (le_of_eq ?_)
    field_simp
    rfl
  linarith

/-- The hypotheses of Theorem 5 are satisfiable: the zero cost on `[-1, 1]`,
`β_{2,t} = 1 - 1/t`, `α = 1`, `β_{1,t} = 0`, `ζ = 1`, `T = 2`, `x* = 0`; the
conditions hold by `adamNC_inv_cond`. -/
example :
    let S := zeroSetup (d := 1) (fun t : ℕ => 1 / Real.sqrt t) (fun _ => 0) 0
    let R : Rule 1 := adamNCRule fun t : ℕ => 1 - 1 / (t : ℝ)
    IsOnlineConvex S (Set.Icc (fun _ => -1) (fun _ => 1)) 2 0 ∧ (0 : ℝ) < 1 ∧
      S.α = (fun t : ℕ => 1 / Real.sqrt t) ∧
      (∀ t, 1 ≤ t → 0 ≤ S.β₁ t ∧ S.β₁ t ≤ S.β₁ 1) ∧ S.β₁ 1 < 1 ∧
      (∀ t ∈ Icc 1 2, ∀ i, S.gnorm R t i / 1 ≤ Real.sqrt (S.vhat R t i) / S.α t) ∧
      (∀ t ∈ Icc 2 2, ∀ i,
        Real.sqrt (S.vhat R (t - 1) i) / S.α (t - 1) ≤ Real.sqrt (S.vhat R t i) / S.α t) ∧
      (0 : Vec 1) ∈ Set.Icc (fun _ => -1) (fun _ => 1) := by
  intro S R
  have hc := fun i => adamNC_inv_cond (S := S) rfl one_pos rfl i
  exact ⟨isOnlineConvex_zero _ _ _, one_pos, rfl, fun _ _ => ⟨le_rfl, le_rfl⟩,
    by simp [S, zeroSetup], fun t ht i => (hc i).1 t (mem_Icc.mp ht).1,
    fun t ht i => (hc i).2 t (mem_Icc.mp ht).1, fun _ => by norm_num, fun _ => by norm_num⟩

end AdamBeyond
end Transformer
