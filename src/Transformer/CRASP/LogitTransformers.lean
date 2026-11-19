module

public import Transformer.CRASP.PositionalTransformers
public import Transformer.KERPLE.Section4_Attention

/-!
# Rounded transformers whose logit is a map of the positions

In no paper. arXiv:2506.16055v3, Appendix B.1 (`def:transformer`), rounds every step of a
future-masked transformer, and Appendix F replaces its logit `eq:att_logit` for RoPE and ALiBi
(`Transformer.CRASP.PTfr`). Here the logit is any map `L i j x` of the position `i` of the query,
the position `j` of the key and their dot product `x = q_i · k_j`; the rest of `def:transformer`
is unchanged (`LTfr`). Positions are the zero-based indices of `⊲ · w`, as in `PTfr`.

* `alibiLogit a` is ALiBi, `eq:alibi`: the dot product less `a (i - j)`. A transformer with this
  logit is a `PTfr` with ALiBi of slope `a` (`LTfr.accepts_toAlibi`).
* `kerpleLogLogit r₁ r₂` and `kerplePowerLogit r₁ r₂` are KERPLE, arXiv:2205.09921v2, Eq. 1
  (`eq:krpe`): the dot product plus the logarithmic or the power kernel of §4 (Practical Choice,
  `Transformer.KERPLE.logKernel`, `Transformer.KERPLE.powerKernel`), under the future mask of
  `def:transformer`.
* `scaleLogit c L` is `L / c`. KERPLE's Eq. 1, like the standard attention, divides by `√d`, `d`
  the width; `def:transformer` and `eq:alibi` do not divide; dividing by `d` is a third variant.
* `ssmaxLogit s L` is SSMax, arXiv:2501.19399v1, Equation `eq:ssmax`: the logits of a query that
  sees `n` keys, multiplied by `s log n`. At the zero-based position `i` the query sees the
  `i + 1` keys `0, …, i`. SSMax maps the whole vector of logits, the positional term included;
  the paper's implementation (Equation `eq:ssmax_impl`) multiplies the query instead, which is the
  same thing when the logit is the dot product alone, and its experiments use RoPE. Scaling the
  dot product but not a positional term added after it is another model, not treated here.
-/

@[expose] public section

namespace Transformer.CRASP

universe u

/-- **A rounded transformer whose logit is `logit i j (q_i · k_j)`**, `def:transformer` with the
logit `eq:att_logit` replaced. -/
structure LTfr (σ : Type u) (p s d k : ℕ) extends RTfr σ p s d k where
  /-- The logit of the query at `i` and the key at `j`, from their dot product. -/
  logit : ℕ → ℕ → ℝ → ℝ

namespace LTfr

variable {σ : Type u} {p s d k : ℕ}

/-- The attention at the position `i` and the coordinate `c` (`eq:att`), with the logit
`logit i j (q_i · k_j)`; an empty denominator falls back on the average of the values. -/
noncomputable def attention (T : LTfr σ p s d k) (ℓ : ℕ) {n : ℕ} (h : Fin n → Fin d → Fx p s)
    (i : Fin n) (c : Fin d) : Fx p s :=
  let score : Fin n → ℝ := fun j =>
    T.logit i.val j.val (∑ t : Fin d, (T.WQ ℓ (h i) t).val * (T.WK ℓ (h j) t).val)
  let den : ℝ := ∑ j ∈ RTfr.masked i, (Fx.round p s (Real.exp (score j))).val
  if den = 0 then
    Fx.round p s ((∑ j ∈ RTfr.masked i, (T.WV ℓ (h j) c).val) / ((RTfr.masked i).card : ℝ))
  else
    Fx.round p s ((∑ j ∈ RTfr.masked i,
      (Fx.round p s (Real.exp (score j) * (T.WV ℓ (h j) c).val)).val) / den)

/-- One layer: attention, the residual connection and the feed-forward network. -/
noncomputable def layer (T : LTfr σ p s d k) (ℓ : ℕ) {n : ℕ} (h : Fin n → Fin d → Fx p s) :
    Fin n → Fin d → Fx p s := fun i => T.ff ℓ fun c => Fx.add (T.attention ℓ h i c) (h i c)

/-- The activations at every position. -/
noncomputable def act (T : LTfr σ p s d k) (w : List σ) : ℕ → Fin w.length → Fin d → Fx p s
  | 0 => fun i c => T.E w[i] c
  | ℓ + 1 => T.layer ℓ (T.act w ℓ)

/-- The output, read off the last position; the empty string is sent to `0`. -/
noncomputable def out (T : LTfr σ p s d k) (w : List σ) : Fx p s :=
  if h : 0 < w.length then T.Wout (T.act w k ⟨w.length - 1, by omega⟩) else 0

/-- `T` accepts `w` if `T(w) > 0`. -/
def Accepts (T : LTfr σ p s d k) (w : List σ) : Prop := 0 < (T.out w).val

/-- `T` recognizes `L` when it accepts exactly the strings `⊲ · w`, `w ∈ L`. -/
def Recognizes (T : LTfr (Option σ) p s d k) (L : Set (List σ)) : Prop :=
  ∀ w : List σ, T.Accepts (bos w) ↔ w ∈ L

end LTfr

/-- ALiBi, `eq:alibi`: the dot product less `a` times the distance `i - j`. -/
noncomputable def alibiLogit (a : ℝ) (i j : ℕ) (x : ℝ) : ℝ := x - a * ((i : ℝ) - j)

/-- The logit divided by `c`: by `√d`, `d` the width, in KERPLE's Eq. 1 and in the standard
attention, and by `d` in a variant. -/
noncomputable def scaleLogit (c : ℝ) (L : ℕ → ℕ → ℝ → ℝ) (i j : ℕ) (x : ℝ) : ℝ := L i j x / c

/-- SSMax, arXiv:2501.19399v1, Equation `eq:ssmax`: the logits of the query at `i`, which sees
the `i + 1` keys `0, …, i`, multiplied by `s log (i + 1)`. -/
noncomputable def ssmaxLogit (s : ℝ) (L : ℕ → ℕ → ℝ → ℝ) (i j : ℕ) (x : ℝ) : ℝ :=
  s * Real.log ((i : ℝ) + 1) * L i j x

/-- KERPLE, arXiv:2205.09921v2, Eq. 1, with the logarithmic kernel of §4:
`x - r₁ log (1 + r₂ |i - j|)`. -/
noncomputable def kerpleLogLogit (r₁ r₂ : ℝ) (i j : ℕ) (x : ℝ) : ℝ :=
  x + KERPLE.logKernel r₁ r₂ i j

/-- KERPLE, arXiv:2205.09921v2, Eq. 1, with the power kernel of §4: `x - r₁ |i - j| ^ r₂`. -/
noncomputable def kerplePowerLogit (r₁ r₂ : ℝ) (i j : ℕ) (x : ℝ) : ℝ :=
  x + KERPLE.powerKernel r₁ r₂ i j

@[simp] theorem scaleLogit_one (L : ℕ → ℕ → ℝ → ℝ) : scaleLogit 1 L = L := by
  funext i j x
  simp [scaleLogit]

namespace LTfr

variable {σ : Type u} {p s d k : ℕ}

/-- `T` with ALiBi of slope `a` in place of its logit. -/
def toAlibi (T : LTfr σ p s d k) (a : ℝ) : PTfr σ p s d k := { T.toRTfr with pe := .alibi a }

/-- When the logit of `T` is `alibiLogit a`, the layers of `T` are those of `T.toAlibi a`. -/
theorem layer_toAlibi (T : LTfr σ p s d k) {a : ℝ} (hT : T.logit = alibiLogit a) (ℓ : ℕ)
    {n : ℕ} (h : Fin n → Fin d → Fx p s) : (T.toAlibi a).layer ℓ h = T.layer ℓ h := by
  funext i
  simp only [layer, attention, hT]
  rfl

/-- The hypothesis of `layer_toAlibi` holds for a transformer built with the logit of ALiBi. -/
example (T : RTfr (Option Bool) 3 0 1 1) (h : Fin 2 → Fin 1 → Fx 3 0) :=
  layer_toAlibi ({ T with logit := alibiLogit 1 } : LTfr (Option Bool) 3 0 1 1) rfl 0 h

theorem act_toAlibi (T : LTfr σ p s d k) {a : ℝ} (hT : T.logit = alibiLogit a) (w : List σ) :
    ∀ ℓ, (T.toAlibi a).act w ℓ = T.act w ℓ
  | 0 => by
      funext i c
      simp [PTfr.act, act, toAlibi, PosEnc.emb]
  | ℓ + 1 => by rw [PTfr.act, act, act_toAlibi T hT w ℓ, layer_toAlibi T hT]

example (T : RTfr (Option Bool) 3 0 1 1) (w : List (Option Bool)) :=
  act_toAlibi ({ T with logit := alibiLogit 1 } : LTfr (Option Bool) 3 0 1 1) rfl w 1

/-- **A transformer whose logit is `alibiLogit a` is one with ALiBi of slope `a`.** -/
theorem accepts_toAlibi (T : LTfr σ p s d k) {a : ℝ} (hT : T.logit = alibiLogit a)
    (w : List σ) : (T.toAlibi a).Accepts w ↔ T.Accepts w := by
  simp only [PTfr.Accepts, Accepts, PTfr.out, out, act_toAlibi T hT]
  rfl

/-- The hypothesis of `accepts_toAlibi` holds for a transformer built with the logit of ALiBi. -/
example (T : RTfr (Option Bool) 3 0 1 1) (w : List (Option Bool)) :
    (({ T with logit := alibiLogit 1 } : LTfr (Option Bool) 3 0 1 1).toAlibi 1).Accepts w ↔
      ({ T with logit := alibiLogit 1 } : LTfr (Option Bool) 3 0 1 1).Accepts w :=
  accepts_toAlibi _ rfl w

end LTfr

end Transformer.CRASP
