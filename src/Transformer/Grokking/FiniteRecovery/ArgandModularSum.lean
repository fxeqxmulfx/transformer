module

public import Transformer.Grokking.FiniteRecovery.ArgandParityPooling
public import Transformer.Grokking.FiniteRecovery.TransversalExamples

/-! Full native Argand weights for binary modular sums on up to 32 tokens.
ARGAND.md supplies the tied readout, four heads of width sixteen, the fourfold
FFN expansion, and the residual scale. One pass uses a pooling layer followed
by five product layers. The staircase is the one in Flat Spectrum, Theorem E. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open Transformer.PhasorV2 FlatSpectrum Set
open scoped InnerProductSpace

namespace Parity

/-- A fixed six-layer architecture. The pooling projection converts the fraction
of one-tokens to a count at the specified readout length. -/
noncomputable def model (length : ℕ) : Model (ZMod 2) 4 16 6 where
  embedding := embedding
  layer := ![poolLayer length, productLayer 16, productLayer 8, productLayer 4,
    productLayer 2, productLayer 1]
  alpha := 0
  eps := offset

private theorem root_band {n : ℕ} (x : Word (ZMod 2) (n+1)) :
    ∃ y : Stream, (model (n+1)).hidden 1 x (Fin.last n) = argandRMS offset y ∧
      Band 1 5 (ones x) y := by
  let scale : ℝ := 1 / Real.sqrt (2 * ((6 : ℕ) : ℝ) * ((1 : ℕ) : ℝ))
  have hs : 0 < scale := by dsimp [scale]; positivity
  let y := (poolLayer (n+1)).forward offset scale 0 (fun j => argandRMS offset (embedding (x j)))
  let step (b : ℕ) := fun z : Fin (n+1) → Stream => (productLayer b).forward offset scale 0 z
  let i := Fin.last n
  have h₀ : Band 32 0 (ones x) (y i) := poolLayer_band x hs
  have h₁ := productLayer_band (b := 16) (by decide) hs y i h₀
  have h₂ := productLayer_band (b := 8) (by decide) hs (step 16 y) i h₁
  have h₃ := productLayer_band (b := 4) (by decide) hs (step 8 (step 16 y)) i h₂
  have h₄ := productLayer_band (b := 2) (by decide) hs _ i h₃
  have h₅ := productLayer_band (b := 1) (by decide) hs _ i h₄
  refine ⟨_, ?_, h₅⟩
  simp only [Model.hidden, Model.hiddenWith, model, Function.iterate_one, List.ofFn_succ,
    List.ofFn_zero, List.foldl_cons, List.foldl_nil, Matrix.cons_val_succ, Matrix.cons_val_zero]
  rfl

/-- The actual tied readout gap has sign (-1) raised to the count of ones. -/
theorem logits_signed {n : ℕ} (hn : n+1 ≤ 32) (x : Word (ZMod 2) (n+1)) :
    ∃ g : ℝ, 0 < g ∧ (model (n+1)).logits 1 x (Fin.last n) 0 -
      (model (n+1)).logits 1 x (Fin.last n) 1 = (-1 : ℝ)^(ones x) * g := by
  obtain ⟨y, hy, hzero, w, hw, hroot⟩ := root_band x
  obtain ⟨a, ha, hv⟩ := root_signed (ones x) ((ones_le x).trans hn)
  have h0 := hzero 0 (by decide)
  have h1 := hroot 1 (by decide) (by decide)
  refine ⟨gain y * w * a / 2, by positivity [gain_pos y], ?_⟩
  simp only [Model.logits, Model.readout, hy]
  change (⟪EuclideanSpace.single 1 (1 : ℂ), argandRMS offset y⟫_ℂ).re / 2 -
    (⟪EuclideanSpace.single 0 (1 : ℂ), argandRMS offset y⟫_ℂ).re / 2 = _
  rw [EuclideanSpace.inner_single_left, EuclideanSpace.inner_single_left]
  simp only [map_one, one_mul, rms_coord, h0, h1, mul_zero, Complex.zero_re, zero_div, sub_zero,
    ← Complex.ofReal_mul, Complex.ofReal_re]
  change gain y * (w * value (ones x) 5 1) / 2 = _
  rw [hv]
  ring

/-- The exhibited vector is a vector of native trainable parameters. Every
nonempty binary word of the specified length is classified with a strict gap. -/
theorem native_realizes_ones {n : ℕ} (hn : n+1 ≤ 32) :
    (Model.fromWeights (model (n+1)).toWeights).realizes (T := n) 1
      (fun x => (ones x : ZMod 2)) := by
  rw [Model.fromWeights_toWeights _ rfl]
  intro x a hne
  dsimp only at hne ⊢
  obtain ⟨g, hg, h⟩ := logits_signed hn x
  have hb : ∀ b : ZMod 2, b = 0 ∨ b = 1 := by decide
  rcases Nat.even_or_odd (ones x) with he | ho
  · rw [he.natCast_zmod_two] at hne ⊢
    rw [he.neg_one_pow, one_mul] at h
    rw [(hb a).resolve_left hne]
    linarith
  · rw [ho.natCast_zmod_two] at hne ⊢
    rw [ho.neg_one_pow, neg_one_mul] at h
    rw [(hb a).resolve_right hne]
    linarith

end Parity

/-- At the same architecture and one pass, the precise targets used by the
Flat Spectrum examples have native representing weights. -/
theorem modularSum_native_realizes {k s : ℕ} (e : Fin (s+1) ≃ Word (ZMod 2) (k+1))
    (hk : (k+1)+(k+1) ≤ 32) :
    (Model.fromWeights (Parity.model ((k+1)+(k+1))).toWeights).realizes
      (T := (k+1)+k) 1 (modularSum e) := by
  rw [funext (Parity.modularSum_eq_ones e)]
  exact Parity.native_realizes_ones (n := (k+1)+k) hk

/-- From every native draw the combined full-gradient and rational search
stabilizes on the specified staircase, with a positive margin on every word. -/
theorem modularSum_staircase_settles {k s : ℕ} (e : Fin (s+1) ≃ Word (ZMod 2) (k+1))
    (hk : (k+1)+(k+1) ≤ 32) (draw : ComplexParameter (ZMod 2) 4 16 6 → ℂ) :
    let cut := enumeratedCut e
    let part := observe (inputsAtCells cut (staircase s)) (modularSum e)
    part.card = 2*s+1 ∧ ∃ N γ, 0 < γ ∧ SettlesWithMargin (T := (k+1)+k)
      (transversalSearch (T := (k+1)+k) true 1 cut ZMod.stdAddChar part (1/10) (wsdRate 20000) draw) 1
        (modularSum e) N γ := by
  dsimp only
  refine ⟨by rw [card_observe, card_inputsAtCells, ncard_staircase], ?_⟩
  let cut := enumeratedCut e
  let family : Set (Word (ZMod 2) ((k+1)+(k+1)) → ZMod 2) := {modularSum e}
  let inputs := inputsAtCells cut (staircase s)
  have hm : IsTransversal (recoveryHypergraph (targetKernel cut ZMod.stdAddChar '' family)
      univ nuclearNorm) (observedCells cut inputs) := by
    apply (isTransversal_iff_nuclearRecovers _ _).2
    rintro _ ⟨f, hf, rfl⟩
    have he : f = modularSum e := hf
    rw [he, observedCells_inputsAtCells]
    exact modularSum_nuclear_recovers e
  obtain ⟨N, γ, hγ, h⟩ := (transversalSearch_settles_iff (T := (k+1)+k)
    true 1 (1/10) (wsdRate 20000) cut ZMod.stdAddChar ZMod.injective_stdAddChar family inputs
    univ (subset_univ _) hm draw).2 (by
      intro f hf
      have he : f = modularSum e := hf
      exact he ▸ ⟨_, modularSum_native_realizes e hk⟩)
  exact ⟨N, γ, hγ, h _ (mem_singleton _)⟩

example := modularSum_staircase_settles (s := 15) (wordEnumeration 2 4) (by decide)
example := modularSum_staircase_settles (s := 255) (wordEnumeration 2 8) (by decide)
example := modularSum_staircase_settles (s := 65535) (wordEnumeration 2 16) (by decide)

end Transformer.Grokking.FiniteRecovery.Argand
