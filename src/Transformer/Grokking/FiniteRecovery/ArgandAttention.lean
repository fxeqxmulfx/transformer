module

public import Transformer.Grokking.FiniteRecovery.Basic
public import Transformer.PhasorV2.PhaseResidual

/-! The skip/write Argand model uses complex attention with learned rotary
frequencies. The layout supplies positions and the key mask so that the
same projections can also act on a grid of series and times. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open Transformer.PhasorV2
open scoped ComplexConjugate

variable {P : Type*} {c H d T : ℕ}

/-- The layout gives the rotation, admissible keys and the time variable
used in the logarithmic score multiplier. -/
structure AttentionGeometry (P : Type*) (d : ℕ) where
  rotate : P → (Fin d → ℂ) → Fin d → ℂ
  visible : P → P → Bool
  time : P → ℕ

/-- Original one-dimensional causal LeRoPE from ARGAND.md. -/
noncomputable def causalGeometry (alpha : Fin d → ℝ) : AttentionGeometry (Fin T) d where
  rotate i := turn (leropeFrequency 10000 alpha) i.val
  visible i j := decide (j ≤ i)
  time i := i.val

/-- A masked softmax takes its denominator over visible keys only. -/
noncomputable def maskedSoftmax [Fintype P] (keep : P → Bool) (logits : P → ℝ) (j : P) : ℝ :=
  if keep j then Real.exp (logits j) / ∑ k ∈ Finset.univ.filter (fun k => keep k), Real.exp (logits k)
  else 0

theorem maskedSoftmax_eq_zero [Fintype P] (keep : P → Bool) (logits : P → ℝ) (j : P)
    (h : keep j = false) : maskedSoftmax keep logits j = 0 := by
  simp [maskedSoftmax, h]

/-- A row of real softmax with the keys after the query masked out. -/
noncomputable def causalSoftmax (i : Fin T) (logits : Fin T → ℝ) (j : Fin T) : ℝ :=
  maskedSoftmax (fun k => decide (k ≤ i)) logits j

/-- The generic mask has exactly the original causal softmax formula. -/
theorem causalSoftmax_eq (i : Fin T) (logits : Fin T → ℝ) (j : Fin T) :
    causalSoftmax i logits j =
      if j ≤ i then Real.exp (logits j) / ∑ k ∈ Finset.univ.filter (· ≤ i), Real.exp (logits k)
      else 0 := by
  simp [causalSoftmax, maskedSoftmax]

example (logits : Fin 3 → ℝ) : causalSoftmax 1 logits 2 = 0 := by
  rw [causalSoftmax_eq, ite_eq_right (by decide)]

/-- Argand uses the norm-plus-offset denominator for each query and key head. -/
noncomputable def normalizeHead (eps : ℝ) (x : Fin d → ℂ) : Fin d → ℂ :=
  fun j => ((Real.sqrt d / (Real.sqrt (sqLen x) + eps) : ℝ) : ℂ) * x j

/-- Complex attention uses separate softmaxes for real and imaginary
scores. The same mask applies to their denominators. -/
noncomputable def attentionWith [Fintype P] (geometry : AttentionGeometry P d)
    (projections : Attention c H d) (eps : ℝ) (multiplier : ℂ) (x : P → CSpace c) (i : P) :
    CSpace c :=
  let query := fun j h => geometry.rotate j (normalizeHead eps (projections.WQ (x j) h))
  let key := fun j h => geometry.rotate j (normalizeHead eps (projections.WK (x j) h))
  let scaled := fun h j =>
    multiplier * ((Real.log ((geometry.time i : ℝ) + 1) / Real.sqrt d : ℝ) : ℂ) *
      ∑ m, conj (query i h m) * key j h m
  projections.WO fun h m => ∑ j,
    (⟨maskedSoftmax (geometry.visible i) (fun k => (scaled h k).re) j,
      maskedSoftmax (geometry.visible i) (fun k => (scaled h k).im) j⟩ : ℂ) *
        projections.WV (x j) h m

/-- The original one-dimensional attention is the time-only layout. -/
noncomputable def attention (projections : Attention c H d) (eps : ℝ) (multiplier : ℂ)
    (alpha : Fin d → ℝ) (x : Word (CSpace c) T) (i : Fin T) : CSpace c :=
  attentionWith (causalGeometry alpha) projections eps multiplier x i

end Transformer.Grokking.FiniteRecovery.Argand
