import Transformer.GPTMini.AttentionBounds
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Tactic

/-!
# Prefix preservation through the actual causal head

Source: CausalMHA.forward in archived gpt_mini.py at f11b6e2, and
infrastructure/nn/attention.py at cbafbe9. A longer input may contain
arbitrary future queries, keys and values. Its old rows are unchanged.
The proof includes the softmax denominator, RoPE, QKNorm and XSA; it
does not assume that the weights select a correct Basis answer.
-/

namespace Transformer.GPTMini.Causality

open scoped BigOperators
noncomputable section

/-- Two finite arrays agree on the entire shorter prefix.
Source: the prefix restriction used by the reference causal mask. -/
def PrefixEq {A : Type*} {S R : ℕ} (short : Fin S → A) (long : Fin (S + R) → A) : Prop :=
  ∀ j, long (j.castAdd R) = short j

/-- A sum supported on the shorter prefix is independent of the added suffix.
Source: finite-sum reindexing for CausalMHA's masked softmax and value aggregation. -/
theorem sum_prefix {A : Type*} [AddCommMonoid A] {S R : ℕ}
    (short : Fin S → A) (long : Fin (S + R) → A)
    (hprefix : PrefixEq short long) (hsuffix : ∀ j : Fin R, long (Fin.natAdd S j) = 0) :
    (∑ j, long j) = ∑ j, short j := by
  rw [Fin.sum_univ_add]
  have hp : (∑ j : Fin S, long (j.castAdd R)) = ∑ j, short j := by
    apply Finset.sum_congr rfl
    intro j _
    exact hprefix j
  have hs : (∑ j : Fin R, long (Fin.natAdd S j)) = 0 := by
    apply Finset.sum_eq_zero
    intro j _
    exact hsuffix j
  rw [hp, hs, add_zero]

example : PrefixEq (fun j : Fin 2 => (j.val + 1 : ℕ))
      (fun j : Fin (2 + 1) => if j.val < 2 then j.val + 1 else 0) ∧
    (∀ j : Fin 1,
      (if (Fin.natAdd 2 j).val < 2 then (Fin.natAdd 2 j).val + 1 else 0 : ℕ) = 0) := by
  constructor
  · intro j
    simp only [Fin.castAdd, Fin.castLE]
    rw [ite_eq_left j.isLt]
  · intro j
    have hj : ¬(Fin.natAdd 2 j).val < 2 := by change ¬2 + j.val < 2; omega
    rw [ite_eq_right hj]

/-- The visible softmax denominator is exactly preserved by appending future keys.
Source: CausalMHA.causalAttnWeights, including every unmasked key in its denominator. -/
theorem denominator_prefix (cfg : Config) (alpha eps : ℝ) {S R : ℕ}
    (q k : Fin S → EucSpace cfg.head_dim)
    (ql kl : Fin (S + R) → EucSpace cfg.head_dim)
    (hq : PrefixEq q ql) (hk : PrefixEq k kl) (i : Fin S) :
    (∑ j : Fin (S + R), if j.val ≤ (i.castAdd R).val then
      Real.exp (preScore cfg alpha eps ql kl (i.castAdd R) j) else 0) =
    ∑ j : Fin S, if j.val ≤ i.val then Real.exp (preScore cfg alpha eps q k i j) else 0 := by
  apply sum_prefix
  · intro j
    change (if j.val ≤ i.val then
      Real.exp (preScore cfg alpha eps ql kl (i.castAdd R) (j.castAdd R)) else 0) = _
    unfold preScore
    rw [hq i, hk j]
  · intro j
    have hfuture : ¬(Fin.natAdd S j).val ≤ (i.castAdd R).val := by
      change ¬S + j.val ≤ i.val
      omega
    rw [ite_eq_right hfuture]

example (cfg : Config) :
    PrefixEq (fun _ : Fin 2 => (0 : EucSpace cfg.head_dim))
      (fun _ : Fin (2 + 1) => (0 : EucSpace cfg.head_dim)) ∧
    PrefixEq (fun _ : Fin 2 => (0 : EucSpace cfg.head_dim))
      (fun _ : Fin (2 + 1) => (0 : EucSpace cfg.head_dim)) := ⟨fun _ => rfl, fun _ => rfl⟩

/-- Every old attention coefficient remains unchanged in a longer context.
Source: the actual causal softmax, not merely a zero-above-diagonal abstraction. -/
theorem weights_prefix (cfg : Config) (alpha eps : ℝ) {S R : ℕ}
    (q k : Fin S → EucSpace cfg.head_dim)
    (ql kl : Fin (S + R) → EucSpace cfg.head_dim)
    (hq : PrefixEq q ql) (hk : PrefixEq k kl) (i j : Fin S) :
    causalAttnWeights cfg alpha eps ql kl (i.castAdd R) (j.castAdd R) =
      causalAttnWeights cfg alpha eps q k i j := by
  unfold causalAttnWeights
  rw [denominator_prefix cfg alpha eps q k ql kl hq hk i]
  change (if j.val > i.val then 0 else
    Real.exp (preScore cfg alpha eps ql kl (i.castAdd R) (j.castAdd R)) /
      (∑ j' : Fin S, if j'.val ≤ i.val then
        Real.exp (preScore cfg alpha eps q k i j') else 0)) = _
  unfold preScore
  rw [hq i, hk j]

example (cfg : Config) :
    PrefixEq (fun _ : Fin 1 => (0 : EucSpace cfg.head_dim))
      (fun _ : Fin (1 + 2) => (0 : EucSpace cfg.head_dim)) ∧
    PrefixEq (fun _ : Fin 1 => (0 : EucSpace cfg.head_dim))
      (fun _ : Fin (1 + 2) => (0 : EucSpace cfg.head_dim)) := ⟨fun _ => rfl, fun _ => rfl⟩

/-- New suffix values contribute zero and the complete old weighted sum is unchanged.
Source: CausalMHA's masked matrix multiplication attn @ v. -/
theorem output_prefix (cfg : Config) (alpha eps : ℝ) {S R : ℕ}
    (q k v : Fin S → EucSpace cfg.head_dim)
    (ql kl vl : Fin (S + R) → EucSpace cfg.head_dim)
    (hq : PrefixEq q ql) (hk : PrefixEq k kl) (hv : PrefixEq v vl) (i : Fin S) :
    attnOutput cfg alpha eps ql kl vl (i.castAdd R) = attnOutput cfg alpha eps q k v i := by
  unfold attnOutput
  apply sum_prefix
  · intro j
    dsimp only
    rw [weights_prefix cfg alpha eps q k ql kl hq hk i j, hv j]
  · intro j
    have hfuture : (i.castAdd R).val < (Fin.natAdd S j).val := by
      change i.val < S + j.val
      omega
    rw [causalAttnWeights_zero_above cfg alpha eps ql kl _ _ hfuture, zero_smul]

example (cfg : Config) :
    PrefixEq (fun _ : Fin 2 => (0 : EucSpace cfg.head_dim))
      (fun _ : Fin (2 + 1) => (0 : EucSpace cfg.head_dim)) ∧
    PrefixEq (fun _ : Fin 2 => (0 : EucSpace cfg.head_dim))
      (fun _ : Fin (2 + 1) => (0 : EucSpace cfg.head_dim)) ∧
    PrefixEq (fun _ : Fin 2 => (0 : EucSpace cfg.head_dim))
      (fun _ : Fin (2 + 1) => (0 : EucSpace cfg.head_dim)) :=
  ⟨fun _ => rfl, fun _ => rfl, fun _ => rfl⟩

/-- QKNorm, RoPE and XSA preserve the prefix identity of the complete actual head.
Source: every operation in CausalMHA.forward at f11b6e2. Old positions must also agree. -/
theorem head_prefix (cfg : Config) (alpha eps : ℝ) {S R : ℕ}
    (q k v : Fin S → EucSpace cfg.head_dim)
    (ql kl vl : Fin (S + R) → EucSpace cfg.head_dim)
    (positions : Fin S → ℝ) (pl : Fin (S + R) → ℝ)
    (hq : PrefixEq q ql) (hk : PrefixEq k kl) (hv : PrefixEq v vl)
    (hp : PrefixEq positions pl) (i : Fin S) :
    attentionHead cfg alpha eps ql kl vl pl (i.castAdd R) =
      attentionHead cfg alpha eps q k v positions i := by
  have hqr : PrefixEq
      (fun j => applyRope cfg.head_dim cfg.rope_theta (positions j) (q j))
      (fun j => applyRope cfg.head_dim cfg.rope_theta (pl j) (ql j)) := by
    intro j
    dsimp only
    rw [hp j, hq j]
  have hkr : PrefixEq
      (fun j => applyRope cfg.head_dim cfg.rope_theta (positions j) (k j))
      (fun j => applyRope cfg.head_dim cfg.rope_theta (pl j) (kl j)) := by
    intro j
    dsimp only
    rw [hp j, hk j]
  unfold attentionHead xsaProjection
  simp only [ite_true]
  rw [output_prefix cfg alpha eps _ _ v _ _ vl hqr hkr hv i, hv i]

example (cfg : Config) :
    PrefixEq (fun _ : Fin 2 => (0 : EucSpace cfg.head_dim))
      (fun _ : Fin (2 + 1) => (0 : EucSpace cfg.head_dim)) ∧
    PrefixEq (fun _ : Fin 2 => (0 : EucSpace cfg.head_dim))
      (fun _ : Fin (2 + 1) => (0 : EucSpace cfg.head_dim)) ∧
    PrefixEq (fun _ : Fin 2 => (0 : EucSpace cfg.head_dim))
      (fun _ : Fin (2 + 1) => (0 : EucSpace cfg.head_dim)) ∧
    PrefixEq (fun j : Fin 2 => (j.val : ℝ))
      (fun j : Fin (2 + 1) => (j.val : ℝ)) :=
  ⟨fun _ => rfl, fun _ => rfl, fun _ => rfl, fun _ => rfl⟩

end
end Transformer.GPTMini.Causality
