module

public import Transformer.Grokking.FlatSpectrum.ProductLoopSwiGLU
public import Transformer.Grokking.FlatSpectrum.ProductLoopGateParity
public import Transformer.Grokking.FlatSpectrum.Recall.Softmax

/-!
# Softmax attention in a layer of the loop

Source: `review/flat_spectrum.md`, *Loops*, item LP8. Softmax attention averages the values of the
keys with the weights `exp` of their scores (`Recall.softmaxAttention`). A *softmax layer* has
heads whose scores and values are real-linear readings of the state, and writes each head's
average along an output vector (`softmaxLayer`). Over two keys with the scores `0` and `t` and the
values `a` and `b`, the average is `a + (b − a) σ(t)`: the weight `σ(t)` of the second key is half
of Phasor's write gate `2σ(t)` (`softmaxAttention_two`).

Where the scores are constant on the inputs, the weights are numbers and a softmax layer is
affine in the state, so it keeps the order of every coordinate (`softmaxLayer_order`). Where they
read the state it does not: on LP5's prelude one head over two keys, with the scores `0` and
`−s ln 2` and the values `109 − 24 s` and `−107 − 66 s`, writes `(−1)^s` into the root
(`softmaxRoot_loopState`, by `neg_one_pow_eq_softmax`), so the head is no product layer of any
width (`not_isProductNet_softmaxCore`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Finset Transformer.MHC
open Transformer.Grokking.FlatSpectrum.Recall (softmaxAttention)

/-- **Softmax over two keys is Phasor's gate**: with the scores `0` and `t` and the values `a`
and `b`, the average is `a + (b − a) · 2σ(t)/2`. -/
theorem softmaxAttention_two (t a b : ℝ) :
    softmaxAttention ![0, t] ![a, b] = a + (b - a) * (writeGate t / 2) := by
  have h := Real.exp_pos t
  simp only [softmaxAttention, Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one,
    Real.exp_zero, writeGate, gate, Real.exp_neg]
  field_simp
  ring

/-- **One head over two keys gives the parity**: `(−1)^s` is the average of `109 − 24 s` and
`−107 − 66 s` with the scores `0` and `−s ln 2`, for `s ≤ 3`. -/
theorem neg_one_pow_eq_softmax {s : ℕ} (hs : s ≤ 3) :
    (-1 : ℝ) ^ s =
      softmaxAttention ![0, -((s : ℝ) * Real.log 2)] ![109 - 24 * s, -107 - 66 * s] := by
  rw [softmaxAttention_two, writeGate_neg_mul_log_two]
  interval_cases s <;> norm_num

example := neg_one_pow_eq_softmax (s := 3) le_rfl

variable {κ α β : Type} {ι : Type*}

/-- **A softmax layer**: head `h` averages the values `v_{h j} z` of the keys `j` with the weights
`exp` of the scores `σ_{h j} z`, both real-linear readings of the state, and writes the average
along `o_h`. -/
noncomputable def softmaxLayer {η K : Type*} [Fintype η] [Fintype K]
    (score value : η → K → (ι → ℂ) →ₗ[ℝ] ℝ) (out : η → ι → ℂ) : (ι → ℂ) → ι → ℂ :=
  fun z => z + ∑ h, softmaxAttention (fun j => score h j z) (fun j => value h j z) • out h

/-- **Constant scores keep the order**: where every score is constant on the inputs, a softmax
layer is affine in the state, and each coordinate keeps its order. -/
theorem softmaxLayer_order {η K : Type} [Fintype η] [Fintype K] [Fintype ι] {r : ℕ}
    (score value : η → K → (ι → ℂ) →ₗ[ℝ] ℝ) (out : η → ι → ℂ) {Z : (κ → α) → (κ → β) → ι → ℂ}
    (hZ : ∀ c, (fun x y => Z x y c) ∈ orderSpan r) {t : η → K → ℝ}
    (ht : ∀ h j x y, score h j (Z x y) = t h j) (c : ι) :
    (fun x y => softmaxLayer score value out (Z x y) c) ∈ orderSpan r := by
  have e : (fun x y => softmaxLayer score value out (Z x y) c) = (fun x y => Z x y c) +
      ∑ h, ∑ j, (((Real.exp (t h j) / ∑ k, Real.exp (t h k) : ℝ) : ℂ) * out h c) •
        fun x y => (value h j (Z x y) : ℂ) := by
    funext x y
    simp only [softmaxLayer, Pi.add_apply, Finset.sum_apply, Pi.smul_apply, Complex.real_smul,
      softmaxAttention, ht, smul_eq_mul]
    congr 1
    refine Finset.sum_congr rfl fun h _ => ?_
    rw [Finset.sum_div, Complex.ofReal_sum, Finset.sum_mul]
    refine Finset.sum_congr rfl fun j _ => ?_
    push_cast
    ring
  rw [e]
  exact add_mem (hZ c) (Submodule.sum_mem _ fun h _ => Submodule.sum_mem _ fun j _ =>
    Submodule.smul_mem _ _ (realLinear_mem_orderSpan hZ (value h j)))

/-- Scores that read the features of `x₀`, which sum to one, are constant. -/
example {η K : Type} [Fintype η] [Fintype K]
    (value : η → K → ((((Fin 3 × ZMod 2) ⊕ (Fin 3 × ZMod 2)) ⊕ Unit) → ℂ) →ₗ[ℝ] ℝ)
    (out : η → (((Fin 3 × ZMod 2) ⊕ (Fin 3 × ZMod 2)) ⊕ Unit) → ℂ)
    (c : ((Fin 3 × ZMod 2) ⊕ (Fin 3 × ZMod 2)) ⊕ Unit) :
    (fun x y => softmaxLayer (fun _ _ => Complex.reLm ∘ₗ (∑ a : ZMod 2,
      LinearMap.proj (R := ℂ) (φ := fun _ => ℂ) (Sum.inl (Sum.inl (0, a)))).restrictScalars ℝ)
        value out (Sum.elim (pairFeatures x y) 0) c) ∈ orderSpan 1 :=
  softmaxLayer_order _ value out features_mem_orderSpan (t := fun _ _ => 1)
    (fun _ _ x y => by simp [pairFeatures, oneHot]) c

variable (ν : Type)

/-- The scores of the two keys: `0` and LP5's logit `−(ln 2) Re s`. -/
noncomputable def softmaxScore : Fin 2 → (ParityState ⊕ ν → ℂ) →ₗ[ℝ] ℝ :=
  ![0, parityLogit ν]

/-- The values of the two keys: `109 − 24 s` and `−107 − 66 s`, the constant `1` being the sum of
the features of `x₀`. -/
noncomputable def softmaxValue : Fin 2 → (ParityState ⊕ ν → ℂ) →ₗ[ℝ] ℝ :=
  ![(109 : ℝ) • (Complex.reLm ∘ₗ (parityOne ν).restrictScalars ℝ) -
      (24 : ℝ) • (Complex.reLm ∘ₗ (paritySum ν).restrictScalars ℝ),
    (-107 : ℝ) • (Complex.reLm ∘ₗ (parityOne ν).restrictScalars ℝ) -
      (66 : ℝ) • (Complex.reLm ∘ₗ (paritySum ν).restrictScalars ℝ)]

/-- **The core**: one softmax head over two keys, writing into the root. -/
noncomputable def softmaxCore : (ParityState ⊕ ν → ℂ) → ParityState ⊕ ν → ℂ :=
  softmaxLayer ![softmaxScore ν] ![softmaxValue ν] ![Sum.elim (Pi.single parityRoot 1) 0]

/-- **One softmax pass writes `ψ(∑ᵢ xᵢ yᵢ)` into the root for three pairs over `ZMod 2`**,
whatever the message. -/
theorem softmaxRoot_loopState (R : (ParityState → ℂ) →ₗ[ℂ] ν → ℂ) (x y : Fin 3 → ZMod 2) :
    loopState (softmaxCore ν) R (parityPrelude (Sum.elim (pairFeatures x y) 0)) 0 parityRoot =
      ZMod.stdAddChar (∑ i, x i * y i) := by
  set s : ℕ := ∑ i, (x i * y i).val with hs
  have hN : s ≤ 3 := by
    rw [hs, Fin.sum_univ_three]
    linarith [ZMod.val_lt (x 0 * y 0), ZMod.val_lt (x 1 * y 1), ZMod.val_lt (x 2 * y 2)]
  have hleaf :
      ∑ i, parityPrelude (Sum.elim (pairFeatures x y) 0) (Sum.inr (some i)) = (s : ℂ) := by
    simp only [parityPrelude_leaf, hs, Nat.cast_sum]
  have hone :
      ∑ a : ZMod 2, parityPrelude (Sum.elim (pairFeatures x y) 0) (Sum.inl (Sum.inl (0, a))) = 1 :=
    by simp [parityPrelude_feature, pairFeatures, oneHot]
  have hψ : ZMod.stdAddChar (∑ i, x i * y i) = (-1 : ℂ) ^ s := by
    rw [stdAddChar_sum, hs, ← Finset.prod_pow_eq_pow_sum]
    exact Finset.prod_congr rfl fun i _ => stdAddChar_zmod_two _
  have hscore : (fun j => softmaxScore ν j (Sum.elim (parityPrelude
      (Sum.elim (pairFeatures x y) 0)) 0)) = ![0, -((s : ℝ) * Real.log 2)] := by
    funext j
    fin_cases j <;> simp [softmaxScore, parityLogit, paritySum, hleaf]
    ring
  have hvalue : (fun j => softmaxValue ν j (Sum.elim (parityPrelude
      (Sum.elim (pairFeatures x y) 0)) 0)) = ![109 - 24 * (s : ℝ), -107 - 66 * s] := by
    funext j
    fin_cases j <;> simp [softmaxValue, parityOne, paritySum, hleaf, hone]
  change softmaxCore ν (Sum.elim _ 0) (Sum.inl parityRoot) = _
  rw [hψ, show ((-1 : ℂ) ^ s) = (((-1 : ℝ) ^ s : ℝ) : ℂ) by push_cast; ring,
    neg_one_pow_eq_softmax hN]
  simp [softmaxCore, softmaxLayer, hscore, hvalue, parityPrelude_root]

/-- Three products equal to one: the root holds `ψ(1) = −1`. -/
example : loopState (softmaxCore Unit) 0 (parityPrelude (Sum.elim (pairFeatures 1 1) 0)) 0
    parityRoot = -1 := by
  rw [softmaxRoot_loopState, stdAddChar_zmod_two,
    show (∑ i : Fin 3, (1 : Fin 3 → ZMod 2) i * (1 : Fin 3 → ZMod 2) i).val = 1 by decide, pow_one]

/-- **The softmax head is no product layer**, of any width: with it, one pass of a product core
would hold the parity, against `not_productLoop_parity`. -/
theorem not_isProductNet_softmaxCore [Fintype ν] {m : ℕ} :
    ¬ IsProductNet m 1 (softmaxCore ν) := fun hL =>
  not_productLoop_parity isProductNet_parityPrelude hL 0 (LinearMap.proj parityRoot)
    fun x y => softmaxRoot_loopState ν 0 x y

example : ¬ IsProductNet 5 1 (softmaxCore Unit) := not_isProductNet_softmaxCore Unit

end Transformer.Grokking.FlatSpectrum
