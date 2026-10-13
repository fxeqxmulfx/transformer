import Transformer.GPTMini.Convex.Structured.TensorDeferredFFN

/-!
# Actual shared-weight tensor stack and final tied decoder

Source: GPTMini.Model.hidden/forward's sequential residual stack and
TensorDeferredFFN's genuine two-residual replacement blocks. All
nonfinal layers use the same freely learned attention weights. The
final layer additionally clears raw fields before original final
RMSNorm/tied readout. The original FFN is present with fixed zero
matrices, as the initial embedding/attention prototype requires.

Induction derives the actual unit anchor and recovered input fields
after every complete block. Thus the true final learned head agrees
with the verified original mixed decoder for every free assignment,
at any positive configured depth and widths at least 64. This is a
shared-weight prototype, not a claim for independently trained deep
attention matrices. Repeated computations must all be charged. Raw
integer/Basis capability and full-stack training follow separately.
-/

namespace Transformer.GPTMini.Convex.Structured

open Transformer.GPTMini.TokenInterface
open scoped Classical
noncomputable section

variable {V C T : ℕ}

/-- Actual nonfinal layers are applied sequentially with shared learned weights and both real residuals.
Source: GPTMini.Model's chronological block loop, instantiated with the true deferred-FFN intermediate block. -/
def tensorStreamStack (cfg : Config) (eps : ℝ) (hwidth : 64 ≤ cfg.d_model) (ψ : TensorHeadParameters C)
    (x : Fin T → EucSpace cfg.d_model) (hcap : T ≤ C) : ℕ → Fin T → EucSpace cfg.d_model
  | 0 => x
  | layers + 1 => tensorDeferredBlock cfg eps hwidth ψ (tensorStreamStack cfg eps hwidth ψ x hcap layers) hcap

/-- The configured complete stack performs every nonfinal layer then its real final two-residual block.
Source: the positive original configured depth, with final raw-field cancellation before ordinary tied readout. -/
def tensorFullStack (cfg : Config) (eps : ℝ) (hwidth : 64 ≤ cfg.d_model) (ψ : TensorHeadParameters C)
    (x : Fin T → EucSpace cfg.d_model) (hcap : T ≤ C) : Fin T → EucSpace cfg.d_model :=
  tensorDeferredFinalBlock cfg eps hwidth ψ (tensorStreamStack cfg eps hwidth ψ x hcap (cfg.n_layers - 1)) hcap

/-- The actual number of nonfinal blocks plus the final block is exactly the original configured positive depth.
Source: Config.n_layers_pos; no implicit extra layer is introduced by final raw-field cancellation. -/
theorem tensorFullStack_layer_count (cfg : Config) : cfg.n_layers - 1 + 1 = cfg.n_layers := by
  have hl := cfg.n_layers_pos
  omega

/-- The actual complete intermediate stack preserves its unit anchor at every physical row and depth.
Source: induction through genuine attention prenorm, residual writes and the actual original zero FFN residual. -/
theorem tensorStreamStack_anchor (cfg : Config) (eps : ℝ) (heps : 0 < eps) (hwidth : 64 ≤ cfg.d_model)
    (ψ : TensorHeadParameters C) (x : Fin T → EucSpace cfg.d_model) (hcap : T ≤ C)
    (hanchor : ∀ row, x row (tensorAnchorAxis hwidth) = 1) (layers : ℕ) (row : Fin T) :
    tensorStreamStack cfg eps hwidth ψ x hcap layers row (tensorAnchorAxis hwidth) = 1 := by
  induction layers with
  | zero => exact hanchor row
  | succ layers ih =>
      exact tensorDeferredBlock_anchor cfg eps heps hwidth ψ _ hcap row ih

example : (0 : ℝ) < 1 / 100000 ∧ (64 : ℕ) ≤ Config.default.d_model ∧ (3 : ℕ) ≤ 64 ∧
    (∀ row : Fin 3, (fun _ : Fin 3 => WithLp.toLp 2 (fun _ : Fin Config.default.d_model => (1 : ℝ))) row
      (tensorAnchorAxis (by norm_num [Config.default])) = 1) := by
  exact ⟨by norm_num, by norm_num [Config.default], by omega, fun _ => rfl⟩

/-- Every genuine intermediate-stack prenorm has exactly the original learned head observations.
Source: derived anchors, actual new RMSNorm field preservation and chronological transitivity through all blocks. -/
theorem tensorStreamStack_fieldsSame (cfg : Config) (eps : ℝ) (heps : 0 < eps) (hwidth : 64 ≤ cfg.d_model)
    (ψ : TensorHeadParameters C) (x : Fin T → EucSpace cfg.d_model) (hcap : T ≤ C)
    (hanchor : ∀ row, x row (tensorAnchorAxis hwidth) = 1) (layers : ℕ) (row : Fin T) :
    tensorFieldsSame hwidth (rmsNormEps eps (tensorStreamStack cfg eps hwidth ψ x hcap layers row)) (rmsNormEps eps (x row)) := by
  induction layers with
  | zero => exact tensorFieldsSame_refl hwidth _
  | succ layers ih =>
      exact tensorFieldsSame_trans hwidth _ _ _
        (tensorDeferredBlock_fieldsSame cfg eps heps hwidth ψ _ hcap row
          (tensorStreamStack_anchor cfg eps heps hwidth ψ x hcap hanchor layers row)) ih

example : (0 : ℝ) < 1 / 100000 ∧ (64 : ℕ) ≤ Config.default.d_model ∧ (3 : ℕ) ≤ 128 ∧
    (∀ row : Fin 3, (fun _ : Fin 3 => WithLp.toLp 2 (fun _ : Fin Config.default.d_model => (1 : ℝ))) row
      (tensorAnchorAxis (by norm_num [Config.default])) = 1) := by
  exact ⟨by norm_num, by norm_num [Config.default], by omega, fun _ => rfl⟩

/-- Full intermediate-stack inference retains physical future independence at every visible row.
Source: chronological induction over the genuine complete block causality, retaining both residuals. -/
theorem tensorStreamStack_causal (cfg : Config) (eps : ℝ) (hwidth : 64 ≤ cfg.d_model) (ψ : TensorHeadParameters C)
    (x y : Fin T → EucSpace cfg.d_model) (hcap : T ≤ C) (row : Fin T)
    (hprefix : ∀ position : Fin T, position.val ≤ row.val → x position = y position) (layers : ℕ) :
    ∀ position : Fin T, position.val ≤ row.val → tensorStreamStack cfg eps hwidth ψ x hcap layers position =
      tensorStreamStack cfg eps hwidth ψ y hcap layers position := by
  induction layers with
  | zero => exact hprefix
  | succ layers ih =>
      intro position hp
      exact tensorDeferredBlock_causal cfg eps hwidth ψ _ _ hcap position
        (fun previous hprev => ih previous (le_trans hprev hp))

example : (64 : ℕ) ≤ Config.default.d_model ∧ (3 : ℕ) ≤ 64 ∧
    (∀ p : Fin 3, p.val ≤ 1 → (fun _ : Fin 3 => (0 : EucSpace Config.default.d_model)) p =
      (fun q : Fin 3 => if q.val ≤ 1 then (0 : EucSpace Config.default.d_model)
        else EuclideanSpace.single ⟨0, by norm_num [Config.default]⟩ 1) p) := by
  refine ⟨by norm_num [Config.default], by omega, ?_⟩
  intro p hp
  exact (ite_eq_left hp).symm

/-- The actual configured complete stack has no future-row dependency.
Source: proved intermediate-stack causal induction followed by the true final complete block. -/
theorem tensorFullStack_causal (cfg : Config) (eps : ℝ) (hwidth : 64 ≤ cfg.d_model) (ψ : TensorHeadParameters C)
    (x y : Fin T → EucSpace cfg.d_model) (hcap : T ≤ C) (row : Fin T)
    (hprefix : ∀ position : Fin T, position.val ≤ row.val → x position = y position) :
    tensorFullStack cfg eps hwidth ψ x hcap row = tensorFullStack cfg eps hwidth ψ y hcap row := by
  unfold tensorFullStack
  exact tensorDeferredFinalBlock_causal cfg eps hwidth ψ _ _ hcap row
    (tensorStreamStack_causal cfg eps hwidth ψ x y hcap row hprefix _)

example : (64 : ℕ) ≤ Config.default.d_model ∧ (3 : ℕ) ≤ 64 ∧
    (∀ p : Fin 3, p.val ≤ 1 → (fun _ : Fin 3 => (0 : EucSpace Config.default.d_model)) p =
      (fun q : Fin 3 => if q.val ≤ 1 then (0 : EucSpace Config.default.d_model)
        else EuclideanSpace.single ⟨0, by norm_num [Config.default]⟩ 1) p) := by
  refine ⟨by norm_num [Config.default], by omega, ?_⟩
  intro p hp
  exact (ite_eq_left hp).symm

/-- The true configured stack's final row is exactly the verified mixed head output on actual learned embeddings.
Source: derived raw anchors, full real-stack observation induction, final residual cancellation and physical final-prefix equality. -/
theorem tensorFullStack_last_input (cfg : Config) (hsize : V ≤ 1024) (hwidth : 64 ≤ cfg.d_model)
    (eps : ℝ) (heps : 0 < eps) (θ : BindingParameters V C) (head : Fin V) (tail : List (Fin V))
    (hcap : (head :: tail).length ≤ C) :
    tensorFullStack cfg eps hwidth (tensorHeadParameters θ) (tensorRawSequence hsize hwidth θ (head :: tail) hcap)
      hcap (bindingFinalPosition tail) = tensorCodeOutput hwidth
        (tensorMixedCoordinates hwidth (tensorHeadParameters θ) (tensorSequence hsize hwidth eps θ (head :: tail) hcap)
          hcap (bindingFinalPosition tail)) := by
  let x := tensorRawSequence hsize hwidth θ (head :: tail) hcap
  have ha : ∀ row, x row (tensorAnchorAxis hwidth) = 1 := fun _ => tensorInput_anchor hsize hwidth θ.1 _ _
  have hs := tensorStreamStack_fieldsSame cfg eps heps hwidth (tensorHeadParameters θ) x hcap ha (cfg.n_layers - 1)
  rw [tensorFullStack, tensorDeferredFinalBlock_eq,
    tensorBlock_output eps heps hwidth _ _ hcap _
      (tensorStreamStack_anchor cfg eps heps hwidth (tensorHeadParameters θ) x hcap ha _ _)]
  change tensorCodeOutput hwidth (tensorMixedCoordinates hwidth (tensorHeadParameters θ)
    (tensorPrefix (fun row => rmsNormEps eps (tensorStreamStack cfg eps hwidth (tensorHeadParameters θ) x hcap
      (cfg.n_layers - 1) row)) (Fin.last tail.length)) _ _) = _
  rw [tensorPrefix_last]
  change tensorCodeOutput hwidth (tensorMixedCoordinates hwidth (tensorHeadParameters θ)
    (fun row => rmsNormEps eps (tensorStreamStack cfg eps hwidth (tensorHeadParameters θ) x hcap (cfg.n_layers - 1) row))
      hcap (bindingFinalPosition tail)) = _
  rw [tensorMixedCoordinates_fieldsSame hwidth (tensorHeadParameters θ) _ _ hcap _ hs]
  rfl

example : (548 : ℕ) ≤ 1024 ∧ (64 : ℕ) ≤ Config.default.d_model ∧ (0 : ℝ) < 1 / 100000 ∧
    ([(1 : Fin 548), 36, 292, 36] : List (Fin 548)).length ≤ 64 := by
  exact ⟨by omega, by norm_num [Config.default], by norm_num, by decide⟩

/-- Actual complete-stack/final-RMS/tied greedy prediction equals the verified mixed decoder at every free assignment.
Source: full configured two-residual stack computation, genuine final normalization and unchanged tied inner product. -/
theorem tensorFullStack_last_best (cfg : Config) (hV : 0 < V) (hsize : V ≤ 1024) (hwidth : 64 ≤ cfg.d_model)
    (eps : ℝ) (heps : 0 < eps) (θ : BindingParameters V C) (head : Fin V) (tail : List (Fin V))
    (hcap : (head :: tail).length ≤ C) :
    bestToken hV (fun token => inner (𝕜 := ℝ) (rmsNormEps eps
      (tensorFullStack cfg eps hwidth (tensorHeadParameters θ) (tensorRawSequence hsize hwidth θ (head :: tail) hcap)
        hcap (bindingFinalPosition tail))) (tensorEmbedding hsize θ.1 token)) =
      mixedGreedy hV hsize θ (head :: tail) hcap (bindingFinalPosition tail) := by
  simp_rw [tensorFullStack_last_input cfg hsize hwidth eps heps]
  exact tensorMixedOutput_best hV hsize hwidth eps heps θ (head :: tail) hcap (bindingFinalPosition tail)

example : (0 : ℕ) < 68 ∧ (68 : ℕ) ≤ 1024 ∧ (64 : ℕ) ≤ Config.default.d_model ∧ (0 : ℝ) < 1 / 100000 ∧
    ([(1 : Fin 68), 22, 18] : List (Fin 68)).length ≤ 19 := by
  exact ⟨by omega, by omega, by norm_num [Config.default], by norm_num, by decide⟩

end
end Transformer.GPTMini.Convex.Structured
