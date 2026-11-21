module

public import Transformer.Phasor.OneStream
public import Transformer.Phasor.StreamSum

/-!
# The mean of the streams follows the ordinary residual

T3d of `review/residual_connections.md`, *What the extra streams can hold*. R0's plain routers keep
four equal streams equal. Here the write gates may differ, so the streams differ, and the *mean*
of the streams is still the one-stream state. A router reads the mean (`Router.IsBalanced`) when
its scales and phases are zero, its read biases are `−ln 3`, its four write gates sum to four and
the exponentials of its skip biases have equal row and column sums. Then:

* it reads `R_D` of the mean of any four streams (`Router.IsBalanced.read`), and the heads of an
  attention with it are the one-stream heads of the means (`Attention.heads_collapse`);
* a block adds to the mean the ordinary residual's update (`Router.IsBalanced.collapse_step`);
* so the main model with such routers has the one-stream state as the mean of its streams at every
  layer and position, and the same logits (`OneModel.logits_withRouters_of_isBalanced`).

Whatever the streams hold apart from their mean is never read (`ZeroSumPart`). `Router.uneven` is
such a router, with main's initial skip logits `2 I` and the write gates `3/2, 1/2, 1, 1`. Exact
real arithmetic, every `ε` and every depth scale.
-/

@[expose] public section

namespace Transformer.Phasor

open Transformer Transformer.MHC Transformer.MHC.PeriRMS Transformer.Residual
  Transformer.ComplexFFN

variable {D d H F V : ℕ}

/-- **A router that reads the mean**: zero scales and phases, read biases `−ln 3`, write gates
summing to four and skip biases whose exponentials have all row and column sums equal. -/
structure Router.IsBalanced (R : Router D) : Prop where
  scale : R.scale = 0
  readBias : R.readBias = fun _ => -Real.log 3
  phase : R.phase = 0
  write : ∑ i, writeGate (R.writeBias i) = 4
  skip : ∃ c ≠ 0, (∀ i, ∑ j, Real.exp (R.skipBias i j) = c) ∧
    ∀ j, ∑ i, Real.exp (R.skipBias i j) = c

/-- A router with the write gates `2σ(ln 3) = 3/2`, `2σ(−ln 3) = 1/2`, `1`, `1`, main's initial
skip logits `2 I`, and otherwise R0's plain router. -/
noncomputable def Router.uneven (D : ℕ) : Router D where
  readProj := 0
  writeProj := 0
  skipProj := 0
  scale := 0
  readBias := fun _ => -Real.log 3
  skipBias := 2 • 1
  phase := 0
  writeBias := ![Real.log 3, -Real.log 3, 0, 0]

/-- `Router.uneven` reads the mean. -/
theorem Router.uneven_isBalanced (D : ℕ) : (Router.uneven D).IsBalanced where
  scale := rfl
  readBias := rfl
  phase := rfl
  write := by
    simp [Router.uneven, Fin.sum_univ_four, writeGate, gate, Real.exp_neg, Real.exp_log]
    norm_num
  skip := ⟨Real.exp 2 + 3, by positivity, sum_exp_two_smul_one, sum_exp_two_smul_one'⟩

/-- The mean of the streams is a quarter of their sum. -/
theorem collapse_eq (x : State D) : collapse x = (1 / 4 : ℝ) • ∑ j, x j :=
  weightedSum_const_weights _ x

/-- A router that reads the mean reads `R_D` of the mean of any four streams. -/
theorem Router.IsBalanced.read {R : Router D} (hR : R.IsBalanced) (eps : ℝ) (x : State D) :
    R.read eps x = GPTMini.rmsNormEps eps (collapse x) := by
  rw [Router.read_of_const hR.scale hR.readBias, gate_neg_log_three, collapse_eq]

/-- `Router.uneven` at main's width reads the mean of any four streams. -/
example (x : State 128) :=
  (Router.uneven_isBalanced 128).read (1 / 8388608) x

/-- **A block with such a router adds the ordinary residual's update to the mean**:
`mean ↦ mean + α R_D(raw(R_D mean))`. -/
theorem Router.IsBalanced.collapse_step {R : Router D} (hR : R.IsBalanced) (eps scale : ℝ)
    (raw : State D → EucSpace D → EucSpace D) (x : State D) :
    collapse (step eps scale ⟨R.routing eps, raw⟩ x) =
      collapse x +
        scale • GPTMini.rmsNormEps eps (raw x (GPTMini.rmsNormEps eps (collapse x))) := by
  obtain ⟨c, hc, hr, hcol⟩ := hR.skip
  rw [collapse_eq (step eps scale _ x), Router.sum_step hR.scale hR.phase hc hr hcol, hR.write,
    hR.read, smul_add, smul_smul, collapse_eq x]
  norm_num

/-- `Router.uneven` at main's width and `ε`, with the depth scale `1/√8` of four layers. -/
example (raw : State 128 → EucSpace 128 → EucSpace 128) (x : State 128) :=
  (Router.uneven_isBalanced 128).collapse_step (1 / 8388608) (1 / Real.sqrt 8) raw x

/-- **The heads of an attention with such a router** are the one-stream heads of the means. -/
theorem Attention.heads_collapse {R : Router D} (hR : R.IsBalanced) (A : Attention D d H)
    (eps : ℝ) (alpha : Fin (d / 2) → ℝ) (x : ℕ → State D) (i : ℕ) :
    A.heads R eps alpha x i = A.oneHeads eps alpha (fun j => collapse (x j)) i := by
  simp only [Attention.heads, Attention.block, Attention.oneHeads, LeRoPEBlock.block,
    Attention.logit, Attention.oneLogit, LeRoPEBlock.leropeLogit, hR.read]

/-- The same at main's sizes. -/
example (A : Attention 128 32 4) (alpha : Fin (32 / 2) → ℝ) (x : ℕ → State 128) (i : ℕ) :=
  A.heads_collapse (Router.uneven_isBalanced 128) (1 / 8388608) alpha x i

/-- **The mean of the streams is the one-stream state**: with routers that read the mean, the main
model's streams have the one-stream state as their mean at every layer and position. -/
theorem OneModel.collapse_states_withRouters (M : OneModel D d H F V)
    {attnR ffnR : ℕ → Router D} (hA : ∀ l, (attnR l).IsBalanced) (hF : ∀ l, (ffnR l).IsBalanced)
    (eps scale : ℝ) (tokens : ℕ → Fin V) (l i : ℕ) :
    collapse ((M.withRouters attnR ffnR).states eps scale tokens l i) =
      M.states eps scale tokens l i := by
  induction l generalizing i with
  | zero =>
    change collapse (fun _ => GPTMini.rmsNormEps eps (M.embedding (tokens i))) = _
    rw [collapse, weightedSum_const]
    norm_num
    rfl
  | succ l ih =>
    change collapse (step eps scale ⟨(ffnR l).routing eps, fun _ z =>
        plainFFN (M.layer l).ffn.gate (M.layer l).ffn.value (M.layer l).ffn.outRe
          (M.layer l).ffn.outIm z⟩
      (step eps scale ⟨(attnR l).routing eps, fun _ _ => (M.layer l).attention.output
        ((M.layer l).attention.heads (attnR l) eps M.alpha
          ((M.withRouters attnR ffnR).states eps scale tokens l) i)⟩
        ((M.withRouters attnR ffnR).states eps scale tokens l i))) = _
    rw [(hF l).collapse_step, (hA l).collapse_step, Attention.heads_collapse (hA l)]
    simp only [ih]
    rfl

/-- Main's sizes with `Router.uneven` in every sublayer: the mean is the one-stream state. -/
example (M : OneModel 128 32 4 171 97) (tokens : ℕ → Fin 97) (l i : ℕ) :=
  M.collapse_states_withRouters (fun _ => Router.uneven_isBalanced 128)
    (fun _ => Router.uneven_isBalanced 128) (1 / 8388608) (1 / Real.sqrt 8) tokens l i

/-- **The logits are the one-stream logits**, though the four streams differ. -/
theorem OneModel.logits_withRouters_of_isBalanced (M : OneModel D d H F V)
    {attnR ffnR : ℕ → Router D} (hA : ∀ l, (attnR l).IsBalanced) (hF : ∀ l, (ffnR l).IsBalanced)
    (eps scale : ℝ) (tokens : ℕ → Fin V) (L i : ℕ) (v : Fin V) :
    (M.withRouters attnR ffnR).logits eps scale tokens L i v = M.logits eps scale tokens L i v := by
  rw [Model.logits, Model.feature, M.collapse_states_withRouters hA hF]
  rfl

/-- Main's sizes, `D = 128`, `d = 32`, four heads, `F = 171`, `ε = 2⁻²³`, `α = 1/√8`, with
`Router.uneven` in every sublayer: the one-stream logits. -/
example (M : OneModel 128 32 4 171 97) (tokens : ℕ → Fin 97) (L i : ℕ) (v : Fin 97) :
    (M.withRouters (fun _ => Router.uneven 128) (fun _ => Router.uneven 128)).logits (1 / 8388608)
        (1 / Real.sqrt 8) tokens L i v =
      M.logits (1 / 8388608) (1 / Real.sqrt 8) tokens L i v :=
  M.logits_withRouters_of_isBalanced (fun _ => Router.uneven_isBalanced 128)
    (fun _ => Router.uneven_isBalanced 128) _ _ tokens L i v

end Transformer.Phasor
