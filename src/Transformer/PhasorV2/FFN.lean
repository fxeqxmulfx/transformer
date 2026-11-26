module

public import Transformer.PhasorV2.Attention

/-!
# Phasor v2: the complex feed-forward branch and the layers

The second sublayer of every layer. The formulas follow DiffGPT (`~/python_projects/gpt`,
`fe1d528`), `research/main_model/README.txt`, *Architecture*:

    z'_i = R_D(sum_s read'_i,s * h'_i,s)
    g_i = W_g z'_i; u_i = W_u z'_i
    f_i = R_D(W_F((g_i complex-multiply u_i)/sqrt(2)))
    h''_i,s = sum_t P'_i,s,t*r'_s*conj(r'_t)*h'_i,t + write'_i,s*alpha*f_i

"The FFN uses plain complex multiplication, with no radial sigmoid or SwiGLU", and its width
`round-up-to-even(floor((8D+1)/3))` gives "342 real/171 complex hidden units" at `D = 128`. As in
`Transformer.PhasorV2.Attention`, the projections are complex-linear maps, here to and from `m`
complex hidden channels (`FFN`). `z'_i = R_D(…)` and `f_i = R_D(…)` are the Peri-norm of
`Router.step`; the branch here is `z ↦ W_F((W_g z ⊙ W_u z)/√2)` at every position
(`FFN.branch`), `⊙` the product of the hidden channels.

* The branch is a quadratic form, `F(a z) = a² F(z)` for every complex `a` (`FFN.apply_smul`): it
  turns the phase `e^{iφ}` of its input into `e^{2iφ}`, and it is even, `F(−z) = F(z)`
  (`FFN.apply_neg`).
* It reads the stream at its own position only, so it is causal (`FFN.isCausal_branch`).
* A layer is an attention sublayer and then a feed-forward one (`layerBranch`). From the embedding
  of any tokens the streams at the position `i` depend on the tokens up to `i` only
  (`stack_layers_congr`), and they keep the RMS bound of `rms_stack_embed_le`.
-/

@[expose] public section

open Transformer.Sandwich

namespace Transformer.PhasorV2

variable {c m H p : ℕ}

/-- **The projections of a feed-forward sublayer** (README: "bias-free complex-linear maps"): the
gate `W_g` and the input `W_u` to `m` complex hidden channels, and the output `W_F`. -/
structure FFN (c m : ℕ) where
  Wg : CSpace c →ₗ[ℂ] Fin m → ℂ
  Wu : CSpace c →ₗ[ℂ] Fin m → ℂ
  WF : (Fin m → ℂ) →ₗ[ℂ] CSpace c

namespace FFN

variable (F : FFN c m)

/-- **The feed-forward map** at one position: `W_F((W_g z ⊙ W_u z)/√2)`. -/
noncomputable def apply (x : CSpace c) : CSpace c :=
  F.WF (((1 / Real.sqrt 2 : ℝ) : ℂ) • (F.Wg x * F.Wu x))

/-- **The feed-forward branch**: the feed-forward map at every position. -/
noncomputable def branch (z : ℕ → CSpace c) (i : ℕ) : CSpace c :=
  F.apply (z i)

/-- **The feed-forward map is a quadratic form**: `F(a z) = a² F(z)` for every complex `a`. -/
theorem apply_smul (a : ℂ) (x : CSpace c) : F.apply (a • x) = a ^ 2 • F.apply x := by
  simp only [apply, map_smul, smul_mul_smul_comm]
  rw [smul_comm, sq]

/-- The phase `e^{iφ}` of the input becomes `e^{2iφ}`: the main model's widths. -/
example (F : FFN 64 171) (φ : ℝ) (x : CSpace 64) :=
  F.apply_smul (Complex.exp (φ * Complex.I)) x

/-- **The feed-forward map is even.** -/
theorem apply_neg (x : CSpace c) : F.apply (-x) = F.apply x := by
  rw [← neg_one_smul ℂ x, apply_smul, neg_one_sq, one_smul]

/-- **The feed-forward branch is causal**: it reads the stream at its own position only. -/
theorem isCausal_branch : IsCausal F.branch := by
  intro x y i hxy
  simp only [branch, hxy i le_rfl]

end FFN

/-- **The branch of the sublayer `n`**: the attention of the layer `n / 2` at even `n`, its
feed-forward branch at odd `n`, with the LeRoPE scales `α` shared by all layers. -/
noncomputable def layerBranch (eps : ℝ) (α : Fin p → ℝ) (A : ℕ → Attention c H p)
    (F : ℕ → FFN c m) (n : ℕ) : (ℕ → CSpace c) → ℕ → CSpace c :=
  if n % 2 = 0 then (A (n / 2)).branch eps α else (F (n / 2)).branch

/-- **Every sublayer's branch is causal.** -/
theorem isCausal_layerBranch (eps : ℝ) (α : Fin p → ℝ) (A : ℕ → Attention c H p)
    (F : ℕ → FFN c m) (n : ℕ) : IsCausal (layerBranch eps α A F n) := by
  unfold layerBranch
  split
  · exact (A _).isCausal_branch eps α
  · exact (F _).isCausal_branch

/-- **The streams at the position `i` depend on the tokens up to `i` only**, after any number of
sublayers with the write scale `γ` (the README's `alpha`), for all weights. -/
theorem stack_layers_congr (eps γ : ℝ) (R : ℕ → Router c) (α : Fin p → ℝ)
    (A : ℕ → Attention c H p) (F : ℕ → FFN c m) (n : ℕ) {V : Type*} (E : V → CSpace c)
    {tok tok' : ℕ → V} {i : ℕ} (htok : ∀ j ≤ i, tok j = tok' j) :
    stack eps γ R (layerBranch eps α A F) n (embed eps E tok) i =
      stack eps γ R (layerBranch eps α A F) n (embed eps E tok') i :=
  isCausal_stack eps γ R (isCausal_layerBranch eps α A F) n _ _ i fun j hj => by
    unfold embed
    rw [htok j hj]

/-- Four layers of the main model: tokens that agree up to the position `5` give the same
streams there. -/
example {V : Type*} (E : V → CSpace 64) (R : ℕ → Router 64) (α : Fin 16 → ℝ)
    (A : ℕ → Attention 64 4 16) (F : ℕ → FFN 64 171) (tok : ℕ → V) (v : V) :=
  stack_layers_congr (1 / 2 ^ 23) (1 / Real.sqrt 8) R α A F 8 E (i := 5) (tok := tok)
    (tok' := fun j => if j ≤ 5 then tok j else v) fun j hj => by simp [hj]

/-- **Four layers of the main model keep every stream's RMS at most `1 + 2√8`** from the
embedding of any tokens: the bound `1 + 2 · 8 · (1/√8)` of eight sublayers at the source's write
scale `1/√8`. -/
example {V : Type*} (E : V → CSpace 64) (tok : ℕ → V) (R : ℕ → Router 64) (α : Fin 16 → ℝ)
    (A : ℕ → Attention 64 4 16) (F : ℕ → FFN 64 171) (i : ℕ) (s : Fin 4) :=
  rms_stack_embed_le (eps := 1 / 2 ^ 23) (by norm_num) (by norm_num) (1 / Real.sqrt 8) R
    (layerBranch (1 / 2 ^ 23) α A F) 8 E tok i s

end Transformer.PhasorV2
