/-
# Future-masked transformers with a position encoding

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), Appendix E (`app:pes`): the three position encodings the paper
treats — sinusoidal (`sec:sinusoidal_pes`, `eq:rotation`), RoPE
(`sec:rope_pes`) and ALiBi (`sec:alibi_pes`, `eq:alibi`).

Each modifies one line of `def:transformer`: sinusoidal adds a positional
vector to the embedding `eq:emb`, while RoPE and ALiBi change the attention
logit `eq:att_logit`.  `PosEnc` is therefore a single datatype selecting which
of the three (or none) is in force, and `PTfr` is `RTfr` with one such field;
the rest of the architecture — rounding, future masking, the empty-denominator
fallback, the residual — is unchanged, so `PTfr.layer` repeats
`RTfr.layer` with the logit replaced.

**Two modelling notes.**

`eq:rotation` rounds the entries of `R(θ)` and then takes powers of the
rounded matrix.  `rotate` below is instead the exact rotation by `i · θ_c` in
each two-dimensional block, with the rounding applied to the result; the
composition law `R(θ)^i = R(i θ)` used here is what the paper's own
computations use, and the accumulated rounding error of a matrix power is not
modelled.  Positions are the zero-based `Fin n` indices, so `R(θ)^{i-1}` of
the paper's one-based `i` is `rotate d θ i`.

The paper's `θ` is indexed by the blocks `c ∈ [d/2]`, so `θ (c / 2)` is the
angle governing the coordinate `c`.  It is carried as a parameter: the
theorems of Appendix E do not use `θ_c = 1000^{-2(c-1)/d}`, only that the
angles are rational multiples of `π` and the encoding is therefore periodic.
-/

import Transformer.CRASP.Transformers

namespace Transformer
namespace CRASP

universe u

variable {σ : Type u}

/-- The coordinate paired with `c` inside its `2 × 2` rotation block:
`2c ↔ 2c + 1`. -/
def pairIdx {d : ℕ} (c : Fin d) : Fin d :=
  if c.val % 2 = 0 then ⟨min (c.val + 1) (d - 1), by have := c.isLt; omega⟩
  else ⟨c.val - 1, by have := c.isLt; omega⟩

/-- `R(θ)^i v`, the block-diagonal rotation of `eq:rotation` applied to a real
vector: block `c` is rotated by the angle `i · θ_c`. -/
noncomputable def rotate (d : ℕ) (θ : ℕ → ℝ) (i : ℕ) (v : Fin d → ℝ) : Fin d → ℝ :=
  fun c =>
    if c.val % 2 = 0 then
      Real.cos ((i : ℝ) * θ (c.val / 2)) * v c
        - Real.sin ((i : ℝ) * θ (c.val / 2)) * v (pairIdx c)
    else
      Real.cos ((i : ℝ) * θ (c.val / 2)) * v c
        + Real.sin ((i : ℝ) * θ (c.val / 2)) * v (pairIdx c)

/-- The sinusoidal positional vector at position `i`: `R(θ)^i` applied to
`[sin 0, cos 0, …, sin 0, cos 0]`. -/
noncomputable def sinusoidalVec (d : ℕ) (θ : ℕ → ℝ) (i : ℕ) : Fin d → ℝ :=
  rotate d θ i fun c => if c.val % 2 = 0 then 0 else 1

/-- Which position encoding a transformer uses (Appendix E). -/
inductive PosEnc : Type where
  /-- No position encoding, as in `def:transformer` itself. -/
  | plain : PosEnc
  /-- Sinusoidal position encoding with angles `θ` (`sec:sinusoidal_pes`). -/
  | sinusoidal (θ : ℕ → ℝ) : PosEnc
  /-- RoPE with angles `θ` (`sec:rope_pes`). -/
  | rope (θ : ℕ → ℝ) : PosEnc
  /-- ALiBi with slope `a` (`sec:alibi_pes`). -/
  | alibi (a : ℝ) : PosEnc

/-- "All the angles `θ` are rational", i.e. rational multiples of `π`, which
is what makes the encoding periodic in the position. -/
def PosEnc.RationalAngles : PosEnc → Prop
  | .plain => True
  | .sinusoidal θ => ∃ q : ℕ → ℚ, ∀ c : ℕ, θ c = Real.pi * q c
  | .rope θ => ∃ q : ℕ → ℚ, ∀ c : ℕ, θ c = Real.pi * q c
  | .alibi _ => True

/-- The vector added to the word embedding at position `i` (`eq:emb`).  Only
the sinusoidal encoding contributes one. -/
noncomputable def PosEnc.emb (pe : PosEnc) (p s d i : ℕ) : Fin d → Fx p s :=
  match pe with
  | .sinusoidal θ => fun c => Fx.round p s (sinusoidalVec d θ i c)
  | _ => fun _ => 0

/-- The attention logit `s_ij` (`eq:att_logit`): the plain dot product, the
dot product of the rotated query and key for RoPE, or the dot product less
`a · (i - j)` for ALiBi. -/
noncomputable def PosEnc.logit (pe : PosEnc) {p s d : ℕ} (i j : ℕ)
    (q k : Fin d → Fx p s) : ℝ :=
  match pe with
  | .rope θ =>
      ∑ c : Fin d,
        rotate d θ i (fun c' => (q c').val) c * rotate d θ j (fun c' => (k c').val) c
  | .alibi a =>
      (∑ c : Fin d, (q c).val * (k c).val) - a * ((i : ℝ) - (j : ℝ))
  | _ => ∑ c : Fin d, (q c).val * (k c).val

/-- A future-masked rounded transformer carrying a position encoding
(Appendix E). -/
structure PTfr (σ : Type u) (p s d k : ℕ) extends RTfr σ p s d k where
  /-- The position encoding in force. -/
  pe : PosEnc

namespace PTfr

variable {p s d k : ℕ}

/-- One layer, as `RTfr.layer` but with the logit supplied by the position
encoding. -/
noncomputable def layer (T : PTfr σ p s d k) (ℓ : ℕ) {n : ℕ}
    (h : Fin n → Fin d → Fx p s) : Fin n → Fin d → Fx p s := fun i =>
  let score : Fin n → ℝ := fun j =>
    T.pe.logit i.val j.val (T.WQ ℓ (h i)) (T.WK ℓ (h j))
  let den : ℝ := ∑ j ∈ RTfr.masked i, (Fx.round p s (Real.exp (score j))).val
  let att : Fin d → Fx p s := fun c =>
    if den = 0 then
      Fx.round p s
        ((∑ j ∈ RTfr.masked i, (T.WV ℓ (h j) c).val) / ((RTfr.masked i).card : ℝ))
    else
      Fx.round p s
        ((∑ j ∈ RTfr.masked i,
            (Fx.round p s (Real.exp (score j) * (T.WV ℓ (h j) c).val)).val) / den)
  T.ff ℓ fun c => Fx.add (att c) (h i c)

/-- The activations, with the positional vector added at layer `0`. -/
noncomputable def act (T : PTfr σ p s d k) (w : List σ) :
    ℕ → Fin w.length → Fin d → Fx p s
  | 0 => fun i c => Fx.add (T.E w[i] c) (T.pe.emb p s d i.val c)
  | ℓ + 1 => T.layer ℓ (T.act w ℓ)

/-- `𝒯(w) = W_out(h^{(k)}_{|w|})`; the empty string is sent to `0`. -/
noncomputable def out (T : PTfr σ p s d k) (w : List σ) : Fx p s :=
  if h : 0 < w.length then T.Wout (T.act w k ⟨w.length - 1, by omega⟩) else 0

/-- "`𝒯` accepts `w` if `𝒯(w) > 0`." -/
def Accepts (T : PTfr σ p s d k) (w : List σ) : Prop := 0 < (T.out w).val

/-- `𝒯` recognizes `L` when it accepts exactly the strings `⊲ · w`, `w ∈ L`. -/
def Recognizes (T : PTfr (Option σ) p s d k) (L : Set (List σ)) : Prop :=
  ∀ w : List σ, T.Accepts (bos w) ↔ w ∈ L

end PTfr

end CRASP
end Transformer
