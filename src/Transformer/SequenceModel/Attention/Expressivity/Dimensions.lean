module

public import Transformer.SequenceModel.Attention.Expressivity.Interpolation

/-!
# Finite source-shaped widths suffice for every finite vocabulary

Source: `review/flat_spectrum.md`, *Transformer learning bridges*, E6.
Width 2(p+1)^2 plus arbitrary even padding has room for both token banks
and all quadratic features using the source's H=(8W+1)//3. Any positive
finite depth is allowed; later blocks can have zero residual branches.
These are sufficient sizes, not lower bounds or efficient circuit claims.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Expressivity

open GPTMini

/-- An unbounded family of finite residual widths. -/
def modelWidth (p padding : ℕ) : ℕ := 2 * (p + 1) ^ 2 + 2 * padding

/-- E6: every nonempty vocabulary fits two disjoint banks in width above three. -/
theorem width_room (p : ℕ) (hp : 0 < p) (padding : ℕ) :
    3 < modelWidth p padding ∧ p + p ≤ modelWidth p padding := by
  unfold modelWidth
  constructor <;> nlinarith

/-- E6: the source FFN width contains all opposed quadratic features. -/
theorem hidden_room (p padding : ℕ) :
    2 * (p + p * p) ≤ (8 * modelWidth p padding + 1) / 3 := by
  rw [Nat.le_div_iff_mul_le (by decide)]
  unfold modelWidth
  nlinarith

/-- One attention head, positive finite depth, and the exact source FFN ratio. -/
abbrev config (p : ℕ) (hp : 0 < p) (padding extraDepth : ℕ) : Config where
  vocab_size := p
  n_layers := extraDepth + 1
  n_heads := 1
  d_model := modelWidth p padding
  d_ff := (8 * modelWidth p padding + 1) / 3
  max_seq_len := 2
  rope_theta := 10000
  divides := one_dvd _
  head_even := by simp only [Nat.div_one]; exact ⟨(p + 1)^2 + padding, by unfold modelWidth; ring⟩
  n_heads_pos := by decide
  n_layers_pos := by omega
  d_model_pos := lt_trans (by decide) (width_room p hp padding).1
  d_ff_pos := by have := hidden_room p padding; nlinarith
  vocab_pos := hp
  max_seq_len_pos := by decide
  theta_pos := by norm_num

/-- Two fixed disjoint banks in the actual head coordinates. -/
def banks (p : ℕ) (hp : 0 < p) (padding extraDepth : ℕ) :
    (Fin p ⊕ Fin p) ↪ Fin (config p hp padding extraDepth).head_dim :=
  finSumFinEquiv.toEmbedding.trans (Fin.castLEEmb (by
    simpa only [Config.head_dim, config, Nat.div_one] using (width_room p hp padding).2))

/-- Fixed layout of all opposed gates and the remaining source hidden channels. -/
def hiddenLayout (p : ℕ) (hp : 0 < p) (padding extraDepth : ℕ) :
    Fin (config p hp padding extraDepth).d_ff ≃
      (Features p × Fin 2) ⊕ Fin ((config p hp padding extraDepth).d_ff - 2 * (p + p * p)) :=
  Fintype.equivOfCardEq (by
    simp only [Fintype.card_fin, Fintype.card_sum, Fintype.card_prod, Features]
    have h := hidden_room p padding
    omega)

example := width_room 97 (by decide) 0
example := width_room 193 (by decide) 1024
example : (config 3 (by decide) 0 4).d_model = 32 ∧
    (config 3 (by decide) 0 4).d_ff = 85 ∧ (config 3 (by decide) 0 4).n_layers = 5 := by decide

end Transformer.SequenceModel.Attention.Expressivity
