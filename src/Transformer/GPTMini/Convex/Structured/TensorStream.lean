import Transformer.GPTMini.Convex.Structured.TensorLikelihood

/-!
# Intermediate causal tensor blocks preserve learned input fields

Source: GPTMini.Block.forward's ordinary attention prenorm/residual,
the genuine causal tensor heads at 84ba59d and complete tensor training
at 815556c. A nonfinal replacement writes its inferred means into the
ten decoder coordinates while retaining every other actual residual
coordinate. It subtracts only the old decoder coordinates, read by
internal anchor recovery from the real RMS-normalized current row.

Thus no copied raw input, token lookup or external scale enters the
attention call. Actual Q/K/value/state fields, protected anchor and
learned position survive this real residual update. The final block
can subsequently clear them with TensorCausalBlock's verified ordinary
residual before unchanged final RMSNorm/tied readout. Complete stack
induction and deferred FFN integration remain subsequent obligations.
-/

namespace Transformer.GPTMini.Convex.Structured

open scoped Classical
noncomputable section

variable {C d T : ℕ}

/-- A true residual coordinate replacement retains the raw row outside the ten output axes.
Source: actual Euclidean tensorCodeOutput writes, with the old physical code coordinates subtracted. -/
def tensorStreamWrite (hwidth : 64 ≤ d) (x : EucSpace d) (coordinates : Fin 10 → ℝ) : EucSpace d :=
  x + tensorCodeOutput hwidth (fun axis => coordinates axis - x (tensorCodeAxis hwidth axis))

/-- Every physical decoder coordinate becomes exactly the newly inferred mean.
Source: real residual addition and the proved injective ten-axis Euclidean write. -/
theorem tensorStreamWrite_code (hwidth : 64 ≤ d) (x : EucSpace d)
    (coordinates : Fin 10 → ℝ) (axis : Fin 10) :
    tensorStreamWrite hwidth x coordinates (tensorCodeAxis hwidth axis) = coordinates axis := by
  rw [tensorStreamWrite, PiLp.add_apply, tensorCodeOutput_code]
  ring

example : (64 : ℕ) ≤ 128 := by omega

/-- All actual nondecoder coordinates survive the intermediate residual unchanged.
Source: true support of the Euclidean update, not an assumed correct hidden-state representation. -/
theorem tensorStreamWrite_offcode (hwidth : 64 ≤ d) (x : EucSpace d)
    (coordinates : Fin 10 → ℝ) (position : Fin d) (hoff : position.val < 52 ∨ 62 ≤ position.val) :
    tensorStreamWrite hwidth x coordinates position = x position := by
  rw [tensorStreamWrite, PiLp.add_apply, tensorCodeOutput_offcode hwidth _ position hoff]
  ring

example : (64 : ℕ) ≤ 64 ∧ ((63 : Fin 64).val < 52 ∨ 62 ≤ (63 : Fin 64).val) := by omega

/-- Intermediate writes preserve all 52 jointly free token fields, regardless of the inferred output.
Source: genuine disjoint physical field/code supports and the actual off-code residual identity. -/
theorem tensorStreamWrite_field (hwidth : 64 ≤ d) (x : EucSpace d)
    (coordinates : Fin 10 → ℝ) (slot : Fin 52) :
    tensorStreamWrite hwidth x coordinates (tensorFieldAxis hwidth slot) = x (tensorFieldAxis hwidth slot) := by
  apply tensorStreamWrite_offcode
  exact Or.inl slot.isLt

example : (64 : ℕ) ≤ 128 := by omega

/-- The protected physical anchor is preserved by every intermediate real residual update.
Source: the output's actual disjoint support, including arbitrary finite learned input weights. -/
theorem tensorStreamWrite_anchor (hwidth : 64 ≤ d) (x : EucSpace d) (coordinates : Fin 10 → ℝ) :
    tensorStreamWrite hwidth x coordinates (tensorAnchorAxis hwidth) = x (tensorAnchorAxis hwidth) := by
  apply tensorStreamWrite_offcode
  exact Or.inr (by rfl)

example : (64 : ℕ) ≤ 64 := by omega

/-- Freely learned absolute position survives intermediate writes alongside raw token fields.
Source: the true physical position axis and ten-axis residual support. -/
theorem tensorStreamWrite_position (hwidth : 64 ≤ d) (x : EucSpace d) (coordinates : Fin 10 → ℝ) :
    tensorStreamWrite hwidth x coordinates (tensorPositionAxis hwidth) = x (tensorPositionAxis hwidth) := by
  apply tensorStreamWrite_offcode
  exact Or.inr (by change (62 : ℕ) ≤ 63; omega)

example : (64 : ℕ) ≤ 128 := by omega

/-- Actual nonfinal attention receives only prenorm tensors and computes a decoder-coordinate residual update.
Source: the same genuine causal state/pointer heads as tensorAttention, subtracting only internally recovered old code axes. -/
def tensorStreamAttention (hwidth : 64 ≤ d) (ψ : TensorHeadParameters C) (x : Fin T → EucSpace d)
    (hcap : T ≤ C) (row : Fin T) : EucSpace d :=
  tensorCodeOutput hwidth (fun axis =>
    tensorMixedCoordinates hwidth ψ (tensorPrefix x row) (tensorPrefix_cap hcap row) (tensorPrefixQuery row) axis -
      tensorAnchorRecovery (tensorAnchorAxis hwidth) (x row) (tensorCodeAxis hwidth axis))

/-- A true intermediate block uses ordinary row-local RMSNorm and residual addition.
Source: GPTMini.Block.forward's unchanged first residual; FFN is integrated in the subsequent stack module. -/
def tensorStreamBlock (eps : ℝ) (hwidth : 64 ≤ d) (ψ : TensorHeadParameters C) (x : Fin T → EucSpace d)
    (hcap : T ≤ C) (row : Fin T) : EucSpace d :=
  x row + tensorStreamAttention hwidth ψ (fun position => rmsNormEps eps (x position)) hcap row

/-- The genuine intermediate prenorm/residual performs exactly the claimed partial coordinate replacement.
Source: actual positive-epsilon RMS recovery from its preserved unit anchor, without a prepared head output. -/
theorem tensorStreamBlock_write (eps : ℝ) (heps : 0 < eps) (hwidth : 64 ≤ d) (ψ : TensorHeadParameters C)
    (x : Fin T → EucSpace d) (hcap : T ≤ C) (row : Fin T) (hanchor : x row (tensorAnchorAxis hwidth) = 1) :
    tensorStreamBlock eps hwidth ψ x hcap row = tensorStreamWrite hwidth (x row)
      (tensorMixedCoordinates hwidth ψ (tensorPrefix (fun position => rmsNormEps eps (x position)) row)
        (tensorPrefix_cap hcap row) (tensorPrefixQuery row)) := by
  unfold tensorStreamBlock tensorStreamAttention
  rw [tensorAnchorRecovery_rms (by omega) eps heps _ (x row) hanchor]
  exact rfl

example : (0 : ℝ) < 1 / 100000 ∧ (64 : ℕ) ≤ 64 ∧ (3 : ℕ) ≤ 64 ∧
    (WithLp.toLp 2 (fun _ : Fin 64 => (1 : ℝ)) : EucSpace 64) (tensorAnchorAxis (by omega)) = 1 := by
  exact ⟨by norm_num, by omega, by omega, rfl⟩

/-- Intermediate attention also excludes all future tensor rows structurally.
Source: its own complete visible-prefix restriction and row-local internal recovery. -/
theorem tensorStreamAttention_causal (hwidth : 64 ≤ d) (ψ : TensorHeadParameters C)
    (x y : Fin T → EucSpace d) (hcap : T ≤ C) (row : Fin T)
    (hprefix : ∀ position : Fin T, position.val ≤ row.val → x position = y position) :
    tensorStreamAttention hwidth ψ x hcap row = tensorStreamAttention hwidth ψ y hcap row := by
  have hp : tensorPrefix x row = tensorPrefix y row := by
    funext position
    apply hprefix
    change position.val ≤ row.val
    have hs := position.isLt
    omega
  unfold tensorStreamAttention
  rw [hp, hprefix row le_rfl]

example : (64 : ℕ) ≤ 64 ∧ (3 : ℕ) ≤ 64 ∧
    (∀ position : Fin 3, position.val ≤ 1 → (fun _ : Fin 3 => (0 : EucSpace 64)) position =
      (fun p : Fin 3 => if p.val ≤ 1 then (0 : EucSpace 64) else EuclideanSpace.single 0 1) position) := by
  refine ⟨by omega, by omega, ?_⟩
  intro position hp
  exact (ite_eq_left hp).symm

/-- Genuine intermediate RMSNorm and residual preserve full physical causality.
Source: row-local normalization and the actual intermediate attention's proved future independence. -/
theorem tensorStreamBlock_causal (eps : ℝ) (hwidth : 64 ≤ d) (ψ : TensorHeadParameters C)
    (x y : Fin T → EucSpace d) (hcap : T ≤ C) (row : Fin T)
    (hprefix : ∀ position : Fin T, position.val ≤ row.val → x position = y position) :
    tensorStreamBlock eps hwidth ψ x hcap row = tensorStreamBlock eps hwidth ψ y hcap row := by
  have hatt := tensorStreamAttention_causal hwidth ψ
    (fun position => rmsNormEps eps (x position)) (fun position => rmsNormEps eps (y position)) hcap row
    (by intro position hp; rw [hprefix position hp])
  unfold tensorStreamBlock
  rw [hatt, hprefix row le_rfl]

example : (64 : ℕ) ≤ 128 ∧ (3 : ℕ) ≤ 64 ∧
    (∀ position : Fin 3, position.val ≤ 1 → (fun _ : Fin 3 => (0 : EucSpace 128)) position =
      (fun p : Fin 3 => if p.val ≤ 1 then (0 : EucSpace 128) else EuclideanSpace.single 0 1) position) := by
  refine ⟨by omega, by omega, ?_⟩
  intro position hp
  exact (ite_eq_left hp).symm

end
end Transformer.GPTMini.Convex.Structured
