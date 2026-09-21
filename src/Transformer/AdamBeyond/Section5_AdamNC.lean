import Transformer.AdamBeyond.Section2_Adam

/-
# Adam and beyond — §5: AdamNC

§5 of arXiv:1904.09237 (the extension after the experiments) and Algorithm 3
of the appendix: AdamNC, Adam with non-constant `β_{2,t}`,
`v_t = β_{2,t} v_{t-1} + (1 - β_{2,t}) g_t²`.

**What the source says and what is carried here.**

* AdamNC is the run of `Transformer.AMSGrad.Setup.state` with `β₂ = 0`, so
  that the moment `v_t` of the run is `g_t²`, and the rule
  `adamNCRule β₂`, `v̂_t = β_{2,t} v̂_{t-1} + (1 - β_{2,t}) v_t`: the `v_t` of
  Algorithm 3 is the `v̂_t` of the run, and the step is weighted by
  `√V_t`, as in Algorithm 3.

* The source writes `v_{t,i}` out in the first condition of Theorem 5 as
  `Σ_{j≤t} Π_{k=1}^{t-j} β_{2(t-k+1)} (1-β_{2j}) g²_{j,i}`: `adamNC_vhat_sum`.

* "`v_{t,i} = Σ_{j≤t} g²_{j,i}/t` when `β_{2t} = 1 - 1/t`": `adamNC_vhat_inv`.
  With `α_t = α/√t` this setting satisfies both conditions of Theorem 5 with
  `ζ = α`, `adamNC_inv_cond`; the source leaves `ζ` implicit in Corollary 2.

Source: arXiv:1904.09237, §5, "Extension: AdamNC algorithm", and Corollary 2
with the sentence after it; appendix, Algorithm 3 (alg:adamnc).
-/

open Finset

namespace Transformer
namespace AdamBeyond

open AMSGrad

variable {d : ℕ}

/-- AdamNC: `v̂_t = β_{2,t} v̂_{t-1} + (1 - β_{2,t}) v_t`.  Run with `β₂ = 0`, so
that `v_t = g_t²`.  arXiv:1904.09237, appendix, Algorithm 3 (alg:adamnc). -/
def adamNCRule (β₂ : ℕ → ℝ) : Rule d := fun t a b => β₂ t • a + (1 - β₂ t) • b

/-- One step of AdamNC, run with `β₂ = 0`. -/
theorem adamNC_vhat_succ {S : Setup d} (hβ₂ : S.β₂ = 0) (β₂ : ℕ → ℝ) (n : ℕ) (i : Fin d) :
    S.vhat (adamNCRule β₂) (n + 1) i =
      β₂ (n + 1) * S.vhat (adamNCRule β₂) n i
        + (1 - β₂ (n + 1)) * S.g (adamNCRule β₂) (n + 1) i ^ 2 := by
  change β₂ (n + 1) * S.vhat (adamNCRule β₂) n i
    + (1 - β₂ (n + 1)) * (S.β₂ * S.v (adamNCRule β₂) n i
      + (1 - S.β₂) * S.g (adamNCRule β₂) (n + 1) i ^ 2) = _
  rw [hβ₂]; ring

/-- `v_{t,i} = Σ_{j≤t} Π_{k=1}^{t-j} β_{2(t-k+1)} (1-β_{2j}) g²_{j,i}`, the
product running over `β_{2,l}`, `j < l ≤ t`.

Source: arXiv:1904.09237, §5, Theorem 5, condition 1. -/
theorem adamNC_vhat_sum {S : Setup d} (hβ₂ : S.β₂ = 0) (β₂ : ℕ → ℝ) (n : ℕ) (i : Fin d) :
    S.vhat (adamNCRule β₂) n i =
      ∑ j ∈ Icc 1 n, (∏ l ∈ Icc (j + 1) n, β₂ l) * (1 - β₂ j) * S.g (adamNCRule β₂) j i ^ 2 := by
  induction n with
  | zero => simp [Setup.vhat, Setup.state]
  | succ n ih =>
    rw [adamNC_vhat_succ hβ₂, ih, sum_Icc_succ_top (by omega),
      show Icc (n + 1 + 1) (n + 1) = ∅ from Icc_eq_empty (by omega), prod_empty, one_mul, mul_sum]
    congr 1
    refine sum_congr rfl fun j hj => ?_
    rw [prod_Icc_succ_top (by have := (mem_Icc.mp hj).2; omega)]
    ring

/-- With `β_{2t} = 1 - 1/t`, `v_{t,i} = Σ_{j≤t} g²_{j,i}/t`.

Source: arXiv:1904.09237, §5, the sentence after Corollary 2. -/
theorem adamNC_vhat_inv {S : Setup d} (hβ₂ : S.β₂ = 0) (n : ℕ) (i : Fin d) :
    S.vhat (adamNCRule fun t : ℕ => 1 - 1 / (t : ℝ)) n i =
      (∑ t ∈ Icc 1 n, S.g (adamNCRule fun t : ℕ => 1 - 1 / (t : ℝ)) t i ^ 2) / n := by
  induction n with
  | zero => simp [Setup.vhat, Setup.state]
  | succ n ih =>
    rw [adamNC_vhat_succ hβ₂, ih, sum_Icc_succ_top (by omega)]
    rcases Nat.eq_zero_or_pos n with rfl | hn
    · simp
    · have : (n : ℝ) ≠ 0 := by positivity
      push_cast
      field_simp
      ring

/-- With `β_{2t} = 1 - 1/t` and `α_t = α/√t`, the conditions of Theorem 5 hold
with `ζ = α`: `‖g_{1:t,i}‖₂/α ≤ √v_{t,i}/α_t` for `t ≥ 1`, and
`√v_{t,i}/α_t` is non-decreasing from `t = 1` on.

Source: arXiv:1904.09237, §5, Theorem 5, conditions 1 and 2, in the setting
of Corollary 2. -/
theorem adamNC_inv_cond {S : Setup d} (hβ₂ : S.β₂ = 0) {α : ℝ} (hα : 0 < α)
    (hαt : S.α = fun t : ℕ => α / Real.sqrt t) (i : Fin d) :
    (∀ t, 1 ≤ t → S.gnorm (adamNCRule fun t : ℕ => 1 - 1 / (t : ℝ)) t i / α ≤
      Real.sqrt (S.vhat (adamNCRule fun t : ℕ => 1 - 1 / (t : ℝ)) t i) / S.α t) ∧
    (∀ t, 2 ≤ t →
      Real.sqrt (S.vhat (adamNCRule fun t : ℕ => 1 - 1 / (t : ℝ)) (t - 1) i) / S.α (t - 1) ≤
        Real.sqrt (S.vhat (adamNCRule fun t : ℕ => 1 - 1 / (t : ℝ)) t i) / S.α t) := by
  set R : Rule d := adamNCRule fun t : ℕ => 1 - 1 / (t : ℝ)
  have key : ∀ t, 1 ≤ t → Real.sqrt (S.vhat R t i) / S.α t = S.gnorm R t i / α := by
    intro t ht
    have hs : 0 ≤ ∑ j ∈ Icc 1 t, S.g R j i ^ 2 := sum_nonneg fun _ _ => sq_nonneg _
    have hst : Real.sqrt t ≠ 0 := by positivity
    rw [adamNC_vhat_inv hβ₂, hαt, Real.sqrt_div hs, Setup.gnorm]
    field_simp
  refine ⟨fun t ht => (key t ht).ge, fun t ht => ?_⟩
  rw [key t (by omega), key (t - 1) (by omega)]
  gcongr
  unfold Setup.gnorm
  gcongr
  omega

end AdamBeyond
end Transformer
