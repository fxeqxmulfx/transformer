import Transformer.GPTMini.Convex.Structured.TensorReadout

/-!
# Causal tensor attention with the ordinary prenorm residual

Source: GPTMini.Block.forward's actual prenorm/residual formula,
TensorReadout's proved cancellation and the genuine tensor heads at
5bdff3c. The replacement has the same sequence-to-sequence Euclidean
interface. Each physical row builds its own self-inclusive prefix;
the head sees every pair in that prefix and its actual final query.
Future rows are excluded structurally, not by a semantic routing mask.

The head receives only prenorm tensors and compact learned global
weights. It returns ten inferred value axes minus the internally
recovered current raw row. The ordinary attention residual therefore
leaves the genuine code-supported output. For actual embeddings its
last-row tied/final-RMS greedy result is the verified mixed decoder
for all parameters. FFN and multilayer preservation are deferred.
-/

namespace Transformer.GPTMini.Convex.Structured

open Transformer.GPTMini.TokenInterface
open scoped Classical
noncomputable section

variable {V C d T : ℕ}

/-- The real visible tensor prefix at a physical query row, including that row.
Source: the usual self-inclusive causal attention interface, retaining chronological physical indices. -/
def tensorPrefix (x : Fin T → EucSpace d) (row : Fin T) (position : Fin (row.val + 1)) : EucSpace d :=
  x ⟨position.val, by have hp := position.isLt; have hr := row.isLt; omega⟩

/-- The physical query is exactly the final row of its own visible prefix.
Source: causal tensor indexing, without an external raw-token selector. -/
def tensorPrefixQuery (row : Fin T) : Fin (row.val + 1) := ⟨row.val, by omega⟩

/-- Every actual row prefix fits the checked tensor context cap.
Source: the same original finite physical sequence bound used at model entry. -/
theorem tensorPrefix_cap (hcap : T ≤ C) (row : Fin T) : row.val + 1 ≤ C := by
  have hr := row.isLt
  omega

example : (4 : ℕ) ≤ 64 := by omega

/-- The final tensor row sees exactly the whole supplied sequence, with no reordered or missing row.
Source: actual causal-prefix indexing at Fin.last. -/
theorem tensorPrefix_last {n : ℕ} (x : Fin (n + 1) → EucSpace d) : tensorPrefix x (Fin.last n) = x := by
  funext position
  rfl

/-- Replacement attention consumes prenorm tensors and returns an ordinary Euclidean residual contribution.
Source: TensorHeads' true inferred mixture and TensorReadout's internal raw-row subtraction. -/
def tensorAttention (hwidth : 64 ≤ d) (ψ : TensorHeadParameters C) (x : Fin T → EucSpace d)
    (hcap : T ≤ C) (row : Fin T) : EucSpace d :=
  tensorCodeOutput hwidth (tensorMixedCoordinates hwidth ψ (tensorPrefix x row)
    (tensorPrefix_cap hcap row) (tensorPrefixQuery row)) -
      tensorAnchorRecovery (tensorAnchorAxis hwidth) (x row)

/-- Actual prenorm attention residual, with the original sequence shape and FFN deferred.
Source: GPTMini.Block.forward's first residual, replacing only its embedding/attention computation. -/
def tensorBlock (eps : ℝ) (hwidth : 64 ≤ d) (ψ : TensorHeadParameters C) (x : Fin T → EucSpace d)
    (hcap : T ≤ C) (row : Fin T) : EucSpace d :=
  x row + tensorAttention hwidth ψ (fun position => rmsNormEps eps (x position)) hcap row

/-- No changed attention output depends on any tensor row later than its actual query.
Source: the complete physical-prefix restriction in the genuine state/all-pair tensor computation. -/
theorem tensorAttention_causal (hwidth : 64 ≤ d) (ψ : TensorHeadParameters C)
    (x y : Fin T → EucSpace d) (hcap : T ≤ C) (row : Fin T)
    (hprefix : ∀ position : Fin T, position.val ≤ row.val → x position = y position) :
    tensorAttention hwidth ψ x hcap row = tensorAttention hwidth ψ y hcap row := by
  have hp : tensorPrefix x row = tensorPrefix y row := by
    funext position
    apply hprefix
    change position.val ≤ row.val
    have hs := position.isLt
    omega
  have hr := hprefix row le_rfl
  unfold tensorAttention
  rw [hp, hr]

example : (64 : ℕ) ≤ 64 ∧ (3 : ℕ) ≤ 64 ∧
    (∀ position : Fin 3, position.val ≤ 1 → (fun _ : Fin 3 => (0 : EucSpace 64)) position =
      (fun p : Fin 3 => if p.val ≤ 1 then (0 : EucSpace 64) else EuclideanSpace.single 0 1) position) := by
  refine ⟨by omega, by omega, ?_⟩
  intro position hp
  exact (ite_eq_left hp).symm

/-- Actual RMSNorm and ordinary residual preserve the replacement's structural causality.
Source: row-local RMSNorm and the proved full tensor attention dependency, without encoder assumptions. -/
theorem tensorBlock_causal (eps : ℝ) (hwidth : 64 ≤ d) (ψ : TensorHeadParameters C)
    (x y : Fin T → EucSpace d) (hcap : T ≤ C) (row : Fin T)
    (hprefix : ∀ position : Fin T, position.val ≤ row.val → x position = y position) :
    tensorBlock eps hwidth ψ x hcap row = tensorBlock eps hwidth ψ y hcap row := by
  have hatt := tensorAttention_causal hwidth ψ
    (fun position => rmsNormEps eps (x position)) (fun position => rmsNormEps eps (y position)) hcap row
    (by intro position hp; rw [hprefix position hp])
  unfold tensorBlock
  rw [hatt, hprefix row le_rfl]

example : (64 : ℕ) ≤ 128 ∧ (3 : ℕ) ≤ 64 ∧
    (∀ position : Fin 3, position.val ≤ 1 → (fun _ : Fin 3 => (0 : EucSpace 128)) position =
      (fun p : Fin 3 => if p.val ≤ 1 then (0 : EucSpace 128) else EuclideanSpace.single 0 1) position) := by
  refine ⟨by omega, by omega, ?_⟩
  intro position hp
  exact (ite_eq_left hp).symm

/-- The actual prenorm residual leaves exactly the inferred ten-axis output whenever its genuine raw anchor is protected.
Source: TensorReadout's full real RMS/cancellation theorem, instantiated with the actual visible-prefix head computation. -/
theorem tensorBlock_output (eps : ℝ) (heps : 0 < eps) (hwidth : 64 ≤ d) (ψ : TensorHeadParameters C)
    (x : Fin T → EucSpace d) (hcap : T ≤ C) (row : Fin T) (hanchor : x row (tensorAnchorAxis hwidth) = 1) :
    tensorBlock eps hwidth ψ x hcap row = tensorCodeOutput hwidth
      (tensorMixedCoordinates hwidth ψ (tensorPrefix (fun position => rmsNormEps eps (x position)) row)
        (tensorPrefix_cap hcap row) (tensorPrefixQuery row)) := by
  unfold tensorBlock tensorAttention
  exact tensorFinalResidual hwidth eps heps (x row) hanchor _

example : (0 : ℝ) < 1 / 100000 ∧ (64 : ℕ) ≤ 64 ∧ (3 : ℕ) ≤ 64 ∧
    (WithLp.toLp 2 (fun _ : Fin 64 => (1 : ℝ)) : EucSpace 64) (tensorAnchorAxis (by omega)) = 1 := by
  exact ⟨by norm_num, by omega, by omega, rfl⟩

/-- Actual unnormalized residual input contains only genuine token-local embeddings and freely learned positions.
Source: TensorEmbedding's real table plus position addition, indexed by the unchanged raw prefix. -/
def tensorRawSequence (hsize : V ≤ 1024) (hwidth : 64 ≤ d) (θ : BindingParameters V C)
    (tokens : List (Fin V)) (hcap : tokens.length ≤ C) (position : Fin tokens.length) : EucSpace d :=
  tensorInput hsize hwidth θ.1 (tokens.get position) (rawBindingPosition tokens hcap position)

/-- The actual final-row block output is the verified genuine compact tensor-head output at the same free weights.
Source: derived raw embedding anchor, actual prenorm residual cancellation and complete causal final-prefix indexing. -/
theorem tensorBlock_last_input (hsize : V ≤ 1024) (hwidth : 64 ≤ d) (eps : ℝ) (heps : 0 < eps)
    (θ : BindingParameters V C) (head : Fin V) (tail : List (Fin V)) (hcap : (head :: tail).length ≤ C) :
    tensorBlock eps hwidth (tensorHeadParameters θ) (tensorRawSequence hsize hwidth θ (head :: tail) hcap)
      hcap (bindingFinalPosition tail) = tensorCodeOutput hwidth
        (tensorMixedCoordinates hwidth (tensorHeadParameters θ) (tensorSequence hsize hwidth eps θ (head :: tail) hcap)
          hcap (bindingFinalPosition tail)) := by
  rw [tensorBlock_output eps heps hwidth _ _ hcap _
    (tensorInput_anchor hsize hwidth θ.1 _ _)]
  change tensorCodeOutput hwidth (tensorMixedCoordinates hwidth (tensorHeadParameters θ)
    (tensorPrefix (tensorSequence hsize hwidth eps θ (head :: tail) hcap) (Fin.last tail.length)) _ _) = _
  rw [tensorPrefix_last]
  rfl

example : (548 : ℕ) ≤ 1024 ∧ (64 : ℕ) ≤ 128 ∧ (0 : ℝ) < 1 / 100000 ∧
    ([(1 : Fin 548), 36, 292, 36] : List (Fin 548)).length ≤ 64 := by
  exact ⟨by omega, by omega, by norm_num, by decide⟩

/-- Real causal block/residual/final-RMS/tied greedy decoding equals the verified mixed model for every free parameter vector.
Source: actual last-row block computation and TensorReadout's derived final normalization/order identity. -/
theorem tensorBlock_last_best (hV : 0 < V) (hsize : V ≤ 1024) (hwidth : 64 ≤ d) (eps : ℝ) (heps : 0 < eps)
    (θ : BindingParameters V C) (head : Fin V) (tail : List (Fin V)) (hcap : (head :: tail).length ≤ C) :
    bestToken hV (fun token => inner (𝕜 := ℝ) (rmsNormEps eps
      (tensorBlock eps hwidth (tensorHeadParameters θ) (tensorRawSequence hsize hwidth θ (head :: tail) hcap)
        hcap (bindingFinalPosition tail))) (tensorEmbedding hsize θ.1 token)) =
      mixedGreedy hV hsize θ (head :: tail) hcap (bindingFinalPosition tail) := by
  simp_rw [tensorBlock_last_input hsize hwidth eps heps]
  exact tensorMixedOutput_best hV hsize hwidth eps heps θ (head :: tail) hcap (bindingFinalPosition tail)

example : (0 : ℕ) < 68 ∧ (68 : ℕ) ≤ 1024 ∧ (64 : ℕ) ≤ 64 ∧ (0 : ℝ) < 1 / 100000 ∧
    ([(1 : Fin 68), 22, 18] : List (Fin 68)).length ≤ 19 := by
  exact ⟨by omega, by omega, by omega, by norm_num, by decide⟩

end
end Transformer.GPTMini.Convex.Structured
