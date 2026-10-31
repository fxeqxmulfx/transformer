module

public import Transformer.SequenceModel.Attention.Model
public import Mathlib.Data.Fintype.BigOperators

/-!
# Depth, width and stored precision in the complete normalized Transformer

Source: `review/flat_spectrum.md`, *Depth, width and precision*, R1--R2.
The information bound uses every real source parameter coordinate, including
both branches and the single tied embedding. A fixed b-bit value alphabet
may be IEEE values, fixed point or another codebook. The theorem even allows
exact real evaluation after decoding, so internal rounding is not needed
for this worst-case obstruction.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Resources

/-- One parameter setting strictly classifies every row of the given table. -/
def Realizes {X V Θ : Type*} (score : Θ → X → V → ℝ) (θ : Θ) (f : X → V) : Prop :=
  ∀ x v, v ≠ f x → score θ x v < score θ x (f x)

/-- R1: a fixed predictor cannot realize two different complete target tables. -/
theorem realizes_unique {X V Θ : Type*} (score : Θ → X → V → ℝ) (θ : Θ)
    {f g : X → V} (hf : Realizes score θ f) (hg : Realizes score θ g) : f = g := by
  funext x
  by_contra h
  have hfg := hf x (g x) (Ne.symm h)
  have hgf := hg x (f x) h
  linarith

/-- R1: universal exact classification requires at least as many parameter
settings as target tables. Ties cannot encode additional labels. -/
theorem tables_le_parameters {X V Θ : Type*} [Fintype X] [Fintype V] [Fintype Θ]
    (score : Θ → X → V → ℝ) (h : ∀ f : X → V, ∃ θ, Realizes score θ f) :
    Fintype.card V ^ Fintype.card X ≤ Fintype.card Θ := by
  classical
  choose weights hw using h
  have hi : Function.Injective weights := by
    intro f g hfg
    exact realizes_unique score (weights f) (hw f) (by simpa only [hfg] using hw g)
  simpa only [Fintype.card_fun] using Fintype.card_le_of_injective weights hi

/-- R1: with b bits in each of M parameter coordinates, representing every
table on X with at least two labels needs at least card(X) stored bits.
The alphabet decoding and all other architecture choices are fixed. -/
theorem universal_bit_budget {X V I : Type*} [Fintype X] [Fintype V] [Fintype I]
    (b : ℕ) (hv : 2 ≤ Fintype.card V)
    (score : (I → Fin (2 ^ b)) → X → V → ℝ)
    (h : ∀ f : X → V, ∃ θ, Realizes score θ f) :
    Fintype.card X ≤ b * Fintype.card I := by
  classical
  have ht := tables_le_parameters score h
  have hpow : 2 ^ Fintype.card X ≤ 2 ^ (b * Fintype.card I) := by
    calc
      2 ^ Fintype.card X ≤ Fintype.card V ^ Fintype.card X := Nat.pow_le_pow_left hv _
      _ ≤ Fintype.card (I → Fin (2 ^ b)) := ht
      _ = 2 ^ (b * Fintype.card I) := by rw [Fintype.card_fun, Fintype.card_fin, Nat.pow_mul]
  exact (Nat.pow_le_pow_iff_right (by decide : 1 < 2)).mp hpow

/-- R1: below that budget, one fixed task defeats every available parameter
setting. The target is chosen before the weights, not after prediction. -/
theorem exists_unrealizable {X V I : Type*} [Fintype X] [Fintype V] [Fintype I]
    (b : ℕ) (hv : 2 ≤ Fintype.card V) (hbudget : b * Fintype.card I < Fintype.card X)
    (score : (I → Fin (2 ^ b)) → X → V → ℝ) :
    ∃ f : X → V, ∀ θ, ¬ Realizes score θ f := by
  by_contra h
  push Not at h
  have hb := universal_bit_budget b hv score h
  omega

private def lookup (θ : Fin 3 → Fin 2) (x : Fin 3) (v : Fin 2) : ℝ :=
  if v = θ x then 1 else 0
private theorem lookup_realizes (f : Fin 3 → Fin 2) : Realizes lookup f f := by
  intro x v hv
  simp [lookup, hv]

example := realizes_unique lookup (fun _ => 1)
  (lookup_realizes (fun _ => 1)) (lookup_realizes (fun _ => 1))
example := tables_le_parameters lookup (fun f => ⟨f, lookup_realizes f⟩)
example := universal_bit_budget (X := Fin 3) (V := Fin 2) (I := Fin 3) 1
  (by decide) lookup (fun f => ⟨f, lookup_realizes f⟩)
example := exists_unrealizable (X := Fin 3) (V := Fin 2) (I := Fin 2) 1
  (by decide) (by decide) (fun θ x v => if v = θ ⟨x.val % 2, by omega⟩ then 1 else 0)


open GPTMini Attention

/-- Scalar source parameter count; tied embeddings are stored once. -/
def parameterCount (cfg : Config) : ℕ :=
  cfg.vocab_size * cfg.d_model + cfg.n_layers *
    (4 * cfg.d_model ^ 2 + 3 * cfg.d_model * cfg.d_ff)

/-- R2: the information count includes all actual trainable coordinates. -/
theorem card_parameter_index (cfg : Config) : Fintype.card (Index cfg) = parameterCount cfg := by
  simp only [Index, BlockIndex, Fintype.card_sum, Fintype.card_prod, Fintype.card_fin, parameterCount]
  ring

/-- Decode a fixed b-bit value alphabet into the complete source weight vector.
The alphabet is fixed before choosing a target table. -/
def codedWeights (cfg : Config) (b : ℕ) (value : Fin (2 ^ b) → ℝ)
    (code : Index cfg → Fin (2 ^ b)) : Parameters cfg :=
  WithLp.toLp 2 (fun i => value (code i))

/-- R2: if stored parameter bits are fewer than input rows, some fixed task
cannot be strictly classified by any weights from that alphabet. This uses
the actual normalized RoPE/XSA/SwiGLU forward and allows any fixed phases. -/
theorem exists_source_failure (cfg : Config) (hv : 2 ≤ cfg.vocab_size) (N b : ℕ)
    (hbudget : b * parameterCount cfg < cfg.vocab_size ^ N)
    (value : Fin (2 ^ b) → ℝ) (eps xeps : ℝ)
    (angle : Fin cfg.n_layers → Fin N → Fin (cfg.head_dim / 2) → ℝ) (readout : Fin N) :
    ∃ f : (Fin N → Fin cfg.vocab_size) → Fin cfg.vocab_size,
      ∀ code : Index cfg → Fin (2 ^ b), ∃ x q, q ≠ f x ∧
        score cfg eps xeps angle readout (codedWeights cfg b value code) x (f x) ≤
        score cfg eps xeps angle readout (codedWeights cfg b value code) x q := by
  classical
  let S := fun code => score cfg eps xeps angle readout (codedWeights cfg b value code)
  obtain ⟨f, hf⟩ := exists_unrealizable (X := Fin N → Fin cfg.vocab_size)
    (V := Fin cfg.vocab_size) (I := Index cfg) b (by simpa only [Fintype.card_fin] using hv)
    (by simpa only [card_parameter_index, Fintype.card_fun, Fintype.card_fin] using hbudget) S
  refine ⟨f, ?_⟩
  intro code
  have h := hf code
  simp only [Realizes, not_forall, not_lt, exists_prop] at h
  exact h

/-- R2: the source FFN ratio bounds each layer's storage by 12W^2+W.
This is a storage bound, not a claim that its rank measures computation. -/
theorem source_parameter_bound (cfg : Config)
    (hH : cfg.d_ff = (8 * cfg.d_model + 1) / 3) :
    parameterCount cfg ≤ cfg.vocab_size * cfg.d_model +
      cfg.n_layers * (12 * cfg.d_model ^ 2 + cfg.d_model) := by
  have hh : 3 * cfg.d_ff ≤ 8 * cfg.d_model + 1 := by
    rw [hH, Nat.mul_comm]
    exact Nat.div_mul_le_self _ _
  have hm := Nat.mul_le_mul_left cfg.d_model hh
  unfold parameterCount
  apply Nat.add_le_add_left
  apply Nat.mul_le_mul_left
  nlinarith

/-- R2: a joint depth/width/precision obstruction at the source hidden ratio.
Increasing any budget can remove this obstruction; sufficiency does not follow. -/
theorem depth_width_precision_obstruction (cfg : Config) (hv : 2 ≤ cfg.vocab_size)
    (hH : cfg.d_ff = (8 * cfg.d_model + 1) / 3) (N b : ℕ)
    (hbudget : b * (cfg.vocab_size * cfg.d_model +
      cfg.n_layers * (12 * cfg.d_model ^ 2 + cfg.d_model)) < cfg.vocab_size ^ N)
    (value : Fin (2 ^ b) → ℝ) (eps xeps : ℝ)
    (angle : Fin cfg.n_layers → Fin N → Fin (cfg.head_dim / 2) → ℝ) (readout : Fin N) :
    ∃ f : (Fin N → Fin cfg.vocab_size) → Fin cfg.vocab_size,
      ∀ code : Index cfg → Fin (2 ^ b), ∃ x q, q ≠ f x ∧
        score cfg eps xeps angle readout (codedWeights cfg b value code) x (f x) ≤
        score cfg eps xeps angle readout (codedWeights cfg b value code) x q := by
  apply exists_source_failure cfg hv N b
    (lt_of_le_of_lt (Nat.mul_le_mul_left b (source_parameter_bound cfg hH)) hbudget)

private abbrev cfg : Config := { witnessConfig with max_seq_len := 8, max_seq_len_pos := by decide }
private def alphabet (a : Fin (2 ^ 8)) : ℝ := (a.val : ℝ) / 128 - 1
example := exists_source_failure cfg (by decide) 8 8 (by decide) alphabet
  (1 / 8388608) (1 / 1000000000000)
  (sourceAngles cfg (fun _ => true) (fun j => (j.val : ℝ))) (Fin.last 7)
example := source_parameter_bound cfg rfl
example := depth_width_precision_obstruction cfg (by decide) rfl 8 8 (by decide) alphabet
  (1 / 8388608) (1 / 1000000000000)
  (sourceAngles cfg (fun _ => true) (fun j => (j.val : ℝ))) (Fin.last 7)

end Transformer.SequenceModel.Resources
