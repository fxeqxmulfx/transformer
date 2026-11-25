module

public import Transformer.Grokking.FlatSpectrum.ProductLoopGate

/-!
# One gated pass where products need two

Source: `review/flat_spectrum.md`, *Loops*, item LP5. Three copies of `x y` over `ZMod 2`, where
`ψ(t) = (−1)^t` (`stdAddChar_zmod_two`). The prelude is one leaf layer: leaf slot `i` gets
`cᵢ = [xᵢ yᵢ = 1]` and the root gets `0` (`parityPrelude_leaf`, `parityPrelude_root`). The core is
one gated layer of two units writing into the root (`parityCore`, `isGatedLayer_parityCore`):
`s · s` behind the gate `2σ(0) = 1`, and `1 · 1` behind the gate `2σ(−s ln 2)`, where
`s = ∑ᵢ cᵢ` and the constant `1` is the sum of the one-hot features of `x₀`. The readout
`root + 361 · 1 − 136 s` then holds `(−1)^s = ψ(∑ᵢ xᵢ yᵢ)` after one pass
(`parityReadout_loopState`), by `neg_one_pow_eq_gate`.

No product core does it (`not_productLoop_parity`): by LP1, a prelude layer, a core layer and
one pass would need `1 + ⌈log₂ 3⌉ = 3 ≤ 2`. The gate's logit reads `s`, of order `2`, and the
gate gives a function of all six operands in one layer, where a product unit only doubles the
order.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Finset Transformer.MHC

/-- On `ZMod 2` the standard character is `(−1)^t`. The value at `1` is proved as
`character_two` of CM20 (`Convex/CoupledMoments/FourierExtraction/Witness.lean`). -/
theorem stdAddChar_zmod_two (c : ZMod 2) : ZMod.stdAddChar c = (-1 : ℂ) ^ c.val := by
  have h1 : ZMod.stdAddChar (1 : ZMod 2) = -1 := by
    have hs := sum_stdAddChar_mul 2 (1 : ZMod 2)
    simp only [show (1 : ZMod 2) ≠ 0 by decide, ite_false, mul_one] at hs
    change (∑ x : Fin 2, ZMod.stdAddChar (show ZMod 2 from x)) = 0 at hs
    rw [Fin.sum_univ_two] at hs
    change ZMod.stdAddChar (0 : ZMod 2) + ZMod.stdAddChar (1 : ZMod 2) = 0 at hs
    simp only [AddChar.map_zero_eq_one] at hs
    linear_combination hs
  have hc : c = 0 ∨ c = 1 := by revert c; decide
  rcases hc with rfl | rfl
  · simp
  · rw [h1, show (1 : ZMod 2).val = 1 from rfl, pow_one]

/-- The features of three pairs over `ZMod 2`. -/
abbrev ParityFeature : Type := (Fin 3 × ZMod 2) ⊕ (Fin 3 × ZMod 2)

/-- The loop's state: the features, the leaf slots `some i` and the root `none`. -/
abbrev ParityState : Type := ParityFeature ⊕ Option (Fin 3)

/-- The root, which the core writes. -/
abbrev parityRoot : ParityState := Sum.inr none

/-- The pair that scratch slot `t` reads: pair `i` for the leaf `some i`, pair `0` for the
root. -/
def parityPair : Option (Fin 3) → Fin 3
  | some i => i
  | none => 0

/-- What the prelude writes into scratch slot `t` from the pair `(a, b)`: `[a b = 1]` into the
leaves, `0` into the root. -/
def parityValue : Option (Fin 3) → ZMod 2 → ZMod 2 → ℂ
  | some _, a, b => ((a * b).val : ℂ)
  | none, _, _ => 0

/-- **The prelude**: one leaf layer, with `8` units. -/
def parityPrelude : (ParityState → ℂ) → ParityState → ℂ := leafLayer parityPair parityValue

theorem isProductNet_parityPrelude : IsProductNet 8 1 parityPrelude := by
  have h := isProductLayer_leafLayer parityPair parityValue
  simp only [Fintype.card_option, Fintype.card_fin, ZMod.card] at h
  exact IsProductNet.of_layer h

lemma parityPrelude_leaf (x y : Fin 3 → ZMod 2) (i : Fin 3) :
    parityPrelude (Sum.elim (pairFeatures x y) 0) (Sum.inr (some i)) = ((x i * y i).val : ℂ) :=
  leafLayer_features parityPair parityValue x y (some i)

lemma parityPrelude_root (x y : Fin 3 → ZMod 2) :
    parityPrelude (Sum.elim (pairFeatures x y) 0) parityRoot = 0 :=
  leafLayer_features parityPair parityValue x y none

lemma parityPrelude_feature (z : ParityState → ℂ) (f : ParityFeature) :
    parityPrelude z (Sum.inl f) = z (Sum.inl f) := by
  simp [parityPrelude, leafLayer, Finset.sum_apply]

variable (ν : Type)

/-- `s = ∑ᵢ cᵢ`, the sum of the leaves. -/
def paritySum : (ParityState ⊕ ν → ℂ) →ₗ[ℂ] ℂ :=
  ∑ i : Fin 3, LinearMap.proj (Sum.inl (Sum.inr (some i)))

/-- The constant `1`: the one-hot features of `x₀` sum to one. -/
def parityOne : (ParityState ⊕ ν → ℂ) →ₗ[ℂ] ℂ :=
  ∑ a : ZMod 2, LinearMap.proj (Sum.inl (Sum.inl (Sum.inl (0, a))))

/-- The logit of the second gate, `−(ln 2) Re s`: a real-linear reading of the state. -/
noncomputable def parityLogit : (ParityState ⊕ ν → ℂ) →ₗ[ℝ] ℝ :=
  (-Real.log 2) • (Complex.reLm ∘ₗ (paritySum ν).restrictScalars ℝ)

/-- **The core**: one gated layer of two units writing into the root, `s · s` with weight `14`
behind the gate `2σ(0) = 1`, and `1 · 1` with weight `−360` behind the gate `2σ(−s ln 2)`. -/
noncomputable def parityCore : (ParityState ⊕ ν → ℂ) → ParityState ⊕ ν → ℂ :=
  gatedLayer ![0, parityLogit ν] 0 ![paritySum ν, parityOne ν] ![paritySum ν, parityOne ν]
    ![(14 : ℂ) • Sum.elim (Pi.single parityRoot 1) 0,
      (-360 : ℂ) • Sum.elim (Pi.single parityRoot 1) 0]

theorem isGatedLayer_parityCore : IsGatedLayer 2 (parityCore ν) :=
  ⟨Fin 2, inferInstance, (Fintype.card_fin 2).le, _, _, _, _, _, rfl⟩

/-- **The readout**: `root + 361 · 1 − 136 s`. -/
noncomputable def parityReadout : (ParityState → ℂ) →ₗ[ℂ] ℂ :=
  LinearMap.proj parityRoot + (361 : ℂ) • ∑ a : ZMod 2, LinearMap.proj (Sum.inl (Sum.inl (0, a)))
    - (136 : ℂ) • ∑ i : Fin 3, LinearMap.proj (Sum.inr (some i))

/-- The readout after one pass of the core on a state `u` and a message `m`. -/
lemma parityReadout_parityCore (u : ParityState → ℂ) (m : ν → ℂ) :
    parityReadout (parityCore ν (Sum.elim u m) ∘ Sum.inl) =
      u parityRoot + 361 * ∑ a : ZMod 2, u (Sum.inl (Sum.inl (0, a))) -
        136 * ∑ i, u (Sum.inr (some i)) + 14 * ((∑ i, u (Sum.inr (some i))) ^ 2) -
        360 * (writeGate (-Real.log 2 * (∑ i, u (Sum.inr (some i))).re) : ℂ) *
          (∑ a : ZMod 2, u (Sum.inl (Sum.inl (0, a)))) ^ 2 := by
  simp [parityReadout, parityCore, gatedLayer, paritySum, parityOne, parityLogit, Fin.sum_univ_two,
    Finset.sum_apply, Phasor.writeGate_zero]
  ring

/-- **One gated pass holds `ψ(∑ᵢ xᵢ yᵢ)` for three pairs over `ZMod 2`**, whatever the message. -/
theorem parityReadout_loopState (R : (ParityState → ℂ) →ₗ[ℂ] ν → ℂ) (x y : Fin 3 → ZMod 2) :
    parityReadout (loopState (parityCore ν) R (parityPrelude (Sum.elim (pairFeatures x y) 0)) 0) =
      ZMod.stdAddChar (∑ i, x i * y i) := by
  have hN : ∑ i, (x i * y i).val ≤ 3 := by
    have h := fun i => ZMod.val_lt (x i * y i)
    have h0 := h 0
    have h1 := h 1
    have h2 := h 2
    rw [Fin.sum_univ_three]
    omega
  have hleaf : ∑ i, parityPrelude (Sum.elim (pairFeatures x y) 0) (Sum.inr (some i)) =
      ((∑ i, (x i * y i).val : ℕ) : ℂ) := by
    simp only [parityPrelude_leaf, Nat.cast_sum]
  have hone :
      ∑ a : ZMod 2, parityPrelude (Sum.elim (pairFeatures x y) 0) (Sum.inl (Sum.inl (0, a))) = 1 :=
    by simp [parityPrelude_feature, pairFeatures, oneHot]
  have hψ : ZMod.stdAddChar (∑ i, x i * y i) = (-1 : ℂ) ^ ∑ i, (x i * y i).val := by
    rw [stdAddChar_sum, ← Finset.prod_pow_eq_pow_sum]
    exact Finset.prod_congr rfl fun i _ => stdAddChar_zmod_two _
  change parityReadout (parityCore ν (Sum.elim _ 0) ∘ Sum.inl) = _
  rw [parityReadout_parityCore, parityPrelude_root, hleaf, hone, Complex.natCast_re, hψ,
    neg_one_pow_eq_gate hN, show -Real.log 2 * ((∑ i, (x i * y i).val : ℕ) : ℝ) =
      -(((∑ i, (x i * y i).val : ℕ) : ℝ) * Real.log 2) by ring]
  ring

/-- Three products equal to one: the readout is `ψ(1) = −1`. -/
example : parityReadout (loopState (parityCore Unit) 0
    (parityPrelude (Sum.elim (pairFeatures (fun _ => 1) fun _ => 1) 0)) 0) = -1 := by
  rw [parityReadout_loopState, stdAddChar_zmod_two,
    show (∑ i : Fin 3, (fun _ => (1 : ZMod 2)) i * (fun _ => (1 : ZMod 2)) i).val = 1 by decide,
    pow_one]

/-- **No product core does it in one pass**: a product prelude and a product core of one layer
each, of any width, with any message and readout, do not hold `ψ(∑ᵢ xᵢ yᵢ)` for three pairs
over `ZMod 2` after one pass. By LP1 they would need `1 + ⌈log₂ 3⌉ = 3 ≤ 2`. -/
theorem not_productLoop_parity {T μ : Type} [Fintype T] [Fintype μ] {m₀ m₁ : ℕ}
    {P : (ParityFeature ⊕ T → ℂ) → ParityFeature ⊕ T → ℂ} (hP : IsProductNet m₀ 1 P)
    {L : ((ParityFeature ⊕ T) ⊕ μ → ℂ) → (ParityFeature ⊕ T) ⊕ μ → ℂ} (hL : IsProductNet m₁ 1 L)
    (R : (ParityFeature ⊕ T → ℂ) →ₗ[ℂ] μ → ℂ) (ℓ : (ParityFeature ⊕ T → ℂ) →ₗ[ℂ] ℂ) :
    ¬ ∀ x y : Fin 3 → ZMod 2,
      ℓ (loopState L R (P (Sum.elim (pairFeatures x y) 0)) 0) =
        ZMod.stdAddChar (∑ i, x i * y i) := by
  intro hval
  have h := le_passes_of_loop_copies (k := 3) (by norm_num) (fun a b : ZMod 2 => a * b)
    ⟨0, 1, 0, 1, by simpa using mixed_stdAddChar_ne_zero (p := 2) (by norm_num) fun _ => 0⟩
    hP hL R (IsProductNet.id_net 0 0) ℓ 0 hval
  have h3 : ¬ Nat.clog 2 3 ≤ 1 := by
    rw [Nat.clog_le_iff_le_pow one_lt_two]
    norm_num
  omega

/-- One pass of LP2's tree core on three copies of `x y` over `ZMod 2` is one of them. -/
example := not_productLoop_parity
  (IsProductNet.of_layer (isProductLayer_leafLayer (copiesPair (k := 3) (by norm_num))
    (copiesValue fun a b : ZMod 2 => a * b)))
  (IsProductNet.of_layer isProductLayer_treeCore) treeRead
  (LinearMap.proj (prodSlot ⟨0, by norm_num⟩))

end Transformer.Grokking.FlatSpectrum
