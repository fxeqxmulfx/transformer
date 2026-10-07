import Transformer.GPTMini.Convex.Structured.TensorObservationInference
import Transformer.GPTMini.Block

/-!
# Real two-residual blocks with the FFN deferred

Source: GPTMini.Block.ffnSubLayer/blockForward's original RMSNorm,
W_in/ReLU2/W_out and second residual, plus TensorStream's actual
intermediate attention residual at 09393dd. The proposed initial
convex prototype fixes both FFN matrices to zero, rather than claiming
convexity for a jointly learned nonlinear FFN. The actual original
FFN computation remains present in each block and is proved zero.

Nonfinal blocks retain the raw fields and overwrite only output axes;
the final block performs the verified raw cancellation. Both include
the original second RMSNorm/FFN/residual formula and are causal.
Their true computations are coupled to the previously checked residual
and observation laws. Repeated learned attention uses shared weights
in the subsequent stack; arbitrary independent deep weights and a
learned FFN are outside this prototype's convexity claim.
-/

namespace Transformer.GPTMini.Convex.Structured

open scoped Classical
noncomputable section

variable {C T : ℕ}

/-- Deferred FFN uses actual original matrix dimensions with both architectural matrices fixed to zero.
Source: GPTMini.FFNParams; training the FFN is intentionally outside the first convex prototype. -/
def tensorZeroFFN (cfg : Config) : FFNParams cfg := ⟨0, 0⟩

/-- Actual intermediate replacement retains the original second RMSNorm/FFN/residual operation.
Source: GPTMini.Block.blockForward, replacing only its attention computation by tensorStreamBlock. -/
def tensorDeferredBlock (cfg : Config) (eps : ℝ) (hwidth : 64 ≤ cfg.d_model) (ψ : TensorHeadParameters C)
    (x : Fin T → EucSpace cfg.d_model) (hcap : T ≤ C) : Fin T → EucSpace cfg.d_model :=
  let x1 := tensorStreamBlock eps hwidth ψ x hcap
  fun row => x1 row + ffnSubLayer cfg (tensorZeroFFN cfg) eps x1 row

/-- The actual final replacement clears raw fields through attention and applies the same deferred original FFN residual.
Source: tensorBlock's true cancellation and GPTMini.Block's unchanged second sublayer formula. -/
def tensorDeferredFinalBlock (cfg : Config) (eps : ℝ) (hwidth : 64 ≤ cfg.d_model) (ψ : TensorHeadParameters C)
    (x : Fin T → EucSpace cfg.d_model) (hcap : T ≤ C) : Fin T → EucSpace cfg.d_model :=
  let x1 := tensorBlock eps hwidth ψ x hcap
  fun row => x1 row + ffnSubLayer cfg (tensorZeroFFN cfg) eps x1 row

/-- The genuine original FFN sublayer with fixed zero output matrix evaluates to zero on every residual tensor.
Source: actual ffnSubLayer/relu2FFN matrix application, with no hidden-state correctness assumption. -/
theorem tensorZeroFFN_apply (cfg : Config) (eps : ℝ) (x : Fin T → EucSpace cfg.d_model) (row : Fin T) :
    ffnSubLayer cfg (tensorZeroFFN cfg) eps x row = 0 := by
  unfold ffnSubLayer relu2FFN tensorZeroFFN
  exact zero_apply _

/-- The actual deferred original FFN is zero on a concrete finite tensor input.
Source: the genuine original matrix/activation/RMS computation with the declared zero matrices. -/
example : ffnSubLayer Config.default (tensorZeroFFN Config.default) (1 / 100000)
    (fun _ : Fin 3 => (0 : EucSpace Config.default.d_model)) 1 = 0 :=
  tensorZeroFFN_apply _ _ _ _

/-- Nonzero residual inputs also pass through the genuine deferred original FFN with zero output.
Source: actual positive-epsilon prenorm and zero output matrix, independently of activation values. -/
example : ffnSubLayer Config.default (tensorZeroFFN Config.default) (1 / 100000)
    (fun _ : Fin 3 => WithLp.toLp 2 (fun _ : Fin Config.default.d_model => (1 : ℝ))) 1 = 0 :=
  tensorZeroFFN_apply _ _ _ _

/-- The actual second residual contributes zero and leaves the genuine intermediate attention computation.
Source: the proved original FFN evaluation, not an omitted second sublayer. -/
theorem tensorDeferredBlock_eq (cfg : Config) (eps : ℝ) (hwidth : 64 ≤ cfg.d_model) (ψ : TensorHeadParameters C)
    (x : Fin T → EucSpace cfg.d_model) (hcap : T ≤ C) :
    tensorDeferredBlock cfg eps hwidth ψ x hcap = tensorStreamBlock eps hwidth ψ x hcap := by
  funext row
  unfold tensorDeferredBlock
  rw [tensorZeroFFN_apply, add_zero]

example : (64 : ℕ) ≤ Config.default.d_model ∧ (3 : ℕ) ≤ 64 := by
  norm_num [Config.default]

/-- The genuine final second residual leaves the verified actual final attention output unchanged.
Source: the same real original zero-FFN evaluation at the final residual stream. -/
theorem tensorDeferredFinalBlock_eq (cfg : Config) (eps : ℝ) (hwidth : 64 ≤ cfg.d_model) (ψ : TensorHeadParameters C)
    (x : Fin T → EucSpace cfg.d_model) (hcap : T ≤ C) :
    tensorDeferredFinalBlock cfg eps hwidth ψ x hcap = tensorBlock eps hwidth ψ x hcap := by
  funext row
  unfold tensorDeferredFinalBlock
  rw [tensorZeroFFN_apply, add_zero]

example : (64 : ℕ) ≤ Config.default.d_model ∧ (3 : ℕ) ≤ 128 := by
  norm_num [Config.default]

/-- Every intermediate complete two-residual block retains the actual unit anchor.
Source: actual prenorm recovery, partial Euclidean write and the zero original FFN contribution. -/
theorem tensorDeferredBlock_anchor (cfg : Config) (eps : ℝ) (heps : 0 < eps) (hwidth : 64 ≤ cfg.d_model)
    (ψ : TensorHeadParameters C) (x : Fin T → EucSpace cfg.d_model) (hcap : T ≤ C) (row : Fin T)
    (hanchor : x row (tensorAnchorAxis hwidth) = 1) :
    tensorDeferredBlock cfg eps hwidth ψ x hcap row (tensorAnchorAxis hwidth) = 1 := by
  rw [tensorDeferredBlock_eq, tensorStreamBlock_write eps heps hwidth ψ x hcap row hanchor,
    tensorStreamWrite_anchor, hanchor]

example : (0 : ℝ) < 1 / 100000 ∧ (64 : ℕ) ≤ Config.default.d_model ∧ (3 : ℕ) ≤ 64 ∧
    (WithLp.toLp 2 (fun _ : Fin Config.default.d_model => (1 : ℝ)) : EucSpace Config.default.d_model)
      (tensorAnchorAxis (by norm_num [Config.default])) = 1 := by
  exact ⟨by norm_num, by norm_num [Config.default], by omega, rfl⟩

/-- Actual normalized observations survive the complete original second residual, not only the attention update.
Source: proved zero FFN and the real intermediate observation identity after new RMSNorm. -/
theorem tensorDeferredBlock_fieldsSame (cfg : Config) (eps : ℝ) (heps : 0 < eps) (hwidth : 64 ≤ cfg.d_model)
    (ψ : TensorHeadParameters C) (x : Fin T → EucSpace cfg.d_model) (hcap : T ≤ C) (row : Fin T)
    (hanchor : x row (tensorAnchorAxis hwidth) = 1) :
    tensorFieldsSame hwidth (rmsNormEps eps (tensorDeferredBlock cfg eps hwidth ψ x hcap row)) (rmsNormEps eps (x row)) := by
  rw [tensorDeferredBlock_eq]
  exact tensorFieldsSame_block eps heps hwidth ψ x hcap row hanchor

example : (0 : ℝ) < 1 / 100000 ∧ (64 : ℕ) ≤ Config.default.d_model ∧ (3 : ℕ) ≤ 64 ∧
    (WithLp.toLp 2 (fun _ : Fin Config.default.d_model => (1 : ℝ)) : EucSpace Config.default.d_model)
      (tensorAnchorAxis (by norm_num [Config.default])) = 1 := by
  exact ⟨by norm_num, by norm_num [Config.default], by omega, rfl⟩

/-- Actual intermediate complete blocks remain independent of all future tensor rows.
Source: true intermediate attention causality and the original zero FFN's row-local second residual. -/
theorem tensorDeferredBlock_causal (cfg : Config) (eps : ℝ) (hwidth : 64 ≤ cfg.d_model) (ψ : TensorHeadParameters C)
    (x y : Fin T → EucSpace cfg.d_model) (hcap : T ≤ C) (row : Fin T)
    (hprefix : ∀ position : Fin T, position.val ≤ row.val → x position = y position) :
    tensorDeferredBlock cfg eps hwidth ψ x hcap row = tensorDeferredBlock cfg eps hwidth ψ y hcap row := by
  rw [tensorDeferredBlock_eq, tensorDeferredBlock_eq]
  exact tensorStreamBlock_causal eps hwidth ψ x y hcap row hprefix

example : (64 : ℕ) ≤ Config.default.d_model ∧ (3 : ℕ) ≤ 64 ∧
    (∀ p : Fin 3, p.val ≤ 1 → (fun _ : Fin 3 => (0 : EucSpace Config.default.d_model)) p =
      (fun q : Fin 3 => if q.val ≤ 1 then (0 : EucSpace Config.default.d_model)
        else EuclideanSpace.single ⟨0, by norm_num [Config.default]⟩ 1) p) := by
  refine ⟨by norm_num [Config.default], by omega, ?_⟩
  intro p hp
  exact (ite_eq_left hp).symm

/-- The genuine final complete block also retains ordinary structural causal behavior.
Source: actual final attention causality with its zero original FFN residual discharged. -/
theorem tensorDeferredFinalBlock_causal (cfg : Config) (eps : ℝ) (hwidth : 64 ≤ cfg.d_model) (ψ : TensorHeadParameters C)
    (x y : Fin T → EucSpace cfg.d_model) (hcap : T ≤ C) (row : Fin T)
    (hprefix : ∀ position : Fin T, position.val ≤ row.val → x position = y position) :
    tensorDeferredFinalBlock cfg eps hwidth ψ x hcap row = tensorDeferredFinalBlock cfg eps hwidth ψ y hcap row := by
  rw [tensorDeferredFinalBlock_eq, tensorDeferredFinalBlock_eq]
  exact tensorBlock_causal eps hwidth ψ x y hcap row hprefix

example : (64 : ℕ) ≤ Config.default.d_model ∧ (3 : ℕ) ≤ 64 ∧
    (∀ p : Fin 3, p.val ≤ 1 → (fun _ : Fin 3 => (0 : EucSpace Config.default.d_model)) p =
      (fun q : Fin 3 => if q.val ≤ 1 then (0 : EucSpace Config.default.d_model)
        else EuclideanSpace.single ⟨0, by norm_num [Config.default]⟩ 1) p) := by
  refine ⟨by norm_num [Config.default], by omega, ?_⟩
  intro p hp
  exact (ite_eq_left hp).symm

end
end Transformer.GPTMini.Convex.Structured
