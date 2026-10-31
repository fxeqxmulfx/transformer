module

public import Transformer.CRASP.Rotary.Program
public import Transformer.CRASP.Normalized.Input

/-!
# Exact source RoPE coefficients at unbounded natural positions

Source: `diff_gpt/domain/model/{block,mha,feed_forward,rope}.py` at b98c665b;
`review/flat_spectrum.md`, *Uniform nonperiodic RoPE*, U3.
The phases are exact reals and need not be periodic. All arithmetic/rounding
sites are those of `Normalized.sourceLayer`: fixed-point coefficients and
stored residuals, exact integer sums, real local maps. IEEE rounding remains
outside this model. The first-position flag replaces the finite position tag.
-/

@[expose] public section
noncomputable section

namespace Transformer.CRASP.Rotary

open GPTMini SequenceModel.Attention

/-- Finite source states, with no sequence-length or absolute-position field. -/
abbrev State (cfg : Config) (p s : ℕ) := Bool × (Fin cfg.d_model → Fx p s)

/-- Exact half-split rotary score at zero-based source positions i-1 and j-1. -/
def sourceWeight (cfg : Config) (B : BlockParams cfg) (eps : ℝ) {p s : ℕ}
    (angle : ℕ → Fin (cfg.head_dim / 2) → ℝ) (q a : State cfg p s)
    (h : Fin cfg.n_heads) (i j : ℕ) : ℝ :=
  Real.exp (headScore cfg.head_dim eps (angle (i - 1)) (angle (j - 1))
    (Normalized.component cfg B eps (qkvQ cfg) (Normalized.vector q.2) h)
    (Normalized.component cfg B eps (qkvK cfg) (Normalized.vector a.2) h))

/-- The same normalized multihead/XSA/SwiGLU update at every sequence length. -/
def sourceLayer (cfg : Config) (B : BlockParams cfg) (eps xeps : ℝ)
    (p s : ℕ) (angle : ℕ → Fin (cfg.head_dim / 2) → ℝ) :
    Layer (State cfg p s) (Normalized.Coordinate cfg) p s where
  denominator q c a i j := Fx.round p s (sourceWeight cfg B eps angle q a c.1 i j)
  numerator q c a i j := Fx.round p s (sourceWeight cfg B eps angle q a c.1 i j *
    Normalized.component cfg B eps (qkvV cfg) (Normalized.vector a.2) c.1 c.2)
  finish q y := (q.1, fun c => Fx.round p s
    (Normalized.localUpdate cfg B eps xeps q.1 (Normalized.vector q.2)
      (fun h => Normalized.vector (fun j => y (h, j))) c))

/-- Every source exponential supplies the required nonnegative denominator,
uniformly over all positions, phases and fixed real weights (U3). -/
theorem sourceLayer_nonnegative (cfg : Config) (B : BlockParams cfg) (eps xeps : ℝ)
    (p s : ℕ) (angle : ℕ → Fin (cfg.head_dim / 2) → ℝ) :
    (sourceLayer cfg B eps xeps p s angle).Nonnegative := by
  intro q c a i j
  exact Fx.m_round_nonneg _ (Real.exp_nonneg _)

/-- Erase an absolute position, retaining exactly the flag used by source XSA. -/
def tagged {cfg : Config} {T p s : ℕ} (i : Fin T) (x : Fin cfg.d_model → Fx p s) :
    State cfg p s := (decide (i.val = 0), x)

/-- The new denominator is exactly the prior source denominator at every
finite position. No approximate or periodic RoPE replaces the original (U3). -/
theorem denominator_restrict (cfg : Config) (B : BlockParams cfg) (eps xeps : ℝ)
    {T : ℕ} (p s : ℕ) (angle : ℕ → Fin (cfg.head_dim / 2) → ℝ)
    (i j : Fin T) (x z : Fin cfg.d_model → Fx p s) (c : Normalized.Coordinate cfg) :
    (sourceLayer cfg B eps xeps p s angle).denominator (tagged i x) c (tagged j z)
      (i.val + 1) (j.val + 1) =
    (Normalized.sourceLayer cfg B eps xeps p s (fun a : Fin T => angle a.val)).denominator
      (some (i, x)) c (some (j, z)) := by
  simp only [sourceLayer, sourceWeight, tagged, Normalized.sourceLayer, Normalized.weight,
    Nat.add_sub_cancel]

/-- The weighted values retain the same source normalization, reshape and
rounding, coordinate by coordinate (U3). -/
theorem numerator_restrict (cfg : Config) (B : BlockParams cfg) (eps xeps : ℝ)
    {T : ℕ} (p s : ℕ) (angle : ℕ → Fin (cfg.head_dim / 2) → ℝ)
    (i j : Fin T) (x z : Fin cfg.d_model → Fx p s) (c : Normalized.Coordinate cfg) :
    (sourceLayer cfg B eps xeps p s angle).numerator (tagged i x) c (tagged j z)
      (i.val + 1) (j.val + 1) =
    (Normalized.sourceLayer cfg B eps xeps p s (fun a : Fin T => angle a.val)).numerator
      (some (i, x)) c (some (j, z)) := by
  simp only [sourceLayer, sourceWeight, tagged, Normalized.sourceLayer, Normalized.weight,
    Nat.add_sub_cancel]

/-- After equal attention coordinates, the complete local normalized block
is unchanged; only the stored position tag is erased (U3). -/
theorem finish_restrict (cfg : Config) (B : BlockParams cfg) (eps xeps : ℝ)
    {T : ℕ} (p s : ℕ) (angle : ℕ → Fin (cfg.head_dim / 2) → ℝ)
    (i : Fin T) (x : Fin cfg.d_model → Fx p s)
    (y : Normalized.Coordinate cfg → Fx p s) :
    (Normalized.sourceLayer cfg B eps xeps p s (fun a : Fin T => angle a.val)).finish
      (some (i, x)) y =
      some (i, ((sourceLayer cfg B eps xeps p s angle).finish (tagged i x) y).2) := rfl

/-- Source geometric frequencies, with independently enabled layers, at
every natural zero-based position. No modulus is introduced. -/
def sourceAngles (cfg : Config) (enabled : ℕ → Bool) (l i : ℕ)
    (c : Fin (cfg.head_dim / 2)) : ℝ :=
  if enabled l then (i : ℝ) * invFreq cfg.head_dim cfg.rope_theta c else 0

/-- Restriction of the uniform grid is the exact existing source angle table. -/
theorem sourceAngles_restrict (cfg : Config) (enabled : ℕ → Bool) {T : ℕ}
    (l : Fin cfg.n_layers) (i : Fin T) (c : Fin (cfg.head_dim / 2)) :
    sourceAngles cfg enabled l.val i.val c =
      SequenceModel.Attention.sourceAngles cfg (fun a => enabled a.val)
        (fun a : Fin T => (a.val : ℝ)) l i c := rfl

/-- The residual-state count is independent of context length (U3). -/
theorem card_state (cfg : Config) (p s : ℕ) :
    Fintype.card (State cfg p s) = 2 * (2 * 2 ^ (p - 1)) ^ cfg.d_model := by
  simp only [State, Fintype.card_prod, Fintype.card_bool, Fintype.card_fun,
    Fintype.card_fin, Fx.card_eq]

/-- Concrete width and precision fix a finite state space for all word lengths. -/
example : Fintype.card (State witnessConfig 8 4) = 2 * 256 ^ 4 := by
  rw [card_state]
  norm_num [witnessConfig]

/-- Normalize and round a source token once, retaining its first-position flag. -/
def embed (cfg : Config) (θ : ModelParams cfg) (eps : ℝ) (p s : ℕ)
    (first : Bool) (a : Fin cfg.vocab_size) : State cfg p s :=
  (first, fun c => Fx.round p s (rmsNormEps eps (θ.embedding a) c))

/-- Extend the finite block table to natural layer indices. Only the first
cfg.n_layers entries are executed, so the cyclic extension is never reached. -/
def layers (cfg : Config) (θ : ModelParams cfg) (eps xeps : ℝ) (p s : ℕ)
    (enabled : Fin cfg.n_layers → Bool) (l : ℕ) :
    Layer (State cfg p s) (Normalized.Coordinate cfg) p s :=
  let idx : Fin cfg.n_layers := ⟨l % cfg.n_layers, Nat.mod_lt _ cfg.n_layers_pos⟩
  sourceLayer cfg (θ.blocks idx) eps xeps p s
    (fun i c => if enabled idx then (i : ℝ) * invFreq cfg.head_dim cfg.rope_theta c else 0)

/-- Within the actual stack, the extension selects precisely the configured
block and enabled flag, without sharing weights across executed layers. -/
theorem layers_at (cfg : Config) (θ : ModelParams cfg) (eps xeps : ℝ) (p s : ℕ)
    (enabled : Fin cfg.n_layers → Bool) (l : Fin cfg.n_layers) :
    layers cfg θ eps xeps p s enabled l.val = sourceLayer cfg (θ.blocks l) eps xeps p s
      (fun i c => if enabled l then (i : ℝ) * invFreq cfg.head_dim cfg.rope_theta c else 0) := by
  simp only [layers, Nat.mod_eq_of_lt l.isLt, Fin.eta]

open scoped Classical in
/-- Final source RMSNorm and the original tied raw embedding, with strict ties policy. -/
def readout (cfg : Config) (θ : ModelParams cfg) (eps : ℝ) {p s : ℕ}
    (target : Fin cfg.vocab_size) (q : State cfg p s) : Bool :=
  decide (∀ v, v ≠ target →
    inner (𝕜 := ℝ) (rmsNormEps eps (Normalized.vector q.2)) (θ.embedding v) <
      inner (𝕜 := ℝ) (rmsNormEps eps (Normalized.vector q.2)) (θ.embedding target))

/-- Removing the position tag leaves the existing source readout unchanged. -/
theorem readout_restrict (cfg : Config) (θ : ModelParams cfg) (eps : ℝ) {T p s : ℕ}
    (target : Fin cfg.vocab_size) (i : Fin T) (x : Fin cfg.d_model → Fx p s) :
    readout cfg θ eps target (tagged i x) =
      Normalized.tiedReadout cfg θ.embedding eps target (some (i, x)) := rfl


end Transformer.CRASP.Rotary
