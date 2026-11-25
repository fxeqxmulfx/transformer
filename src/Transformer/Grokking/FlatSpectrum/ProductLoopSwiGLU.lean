module

public import Transformer.Grokking.FlatSpectrum.ProductLoopGate
public import Transformer.ComplexFFN.Defs

/-!
# SiLU in a layer of the loop

Source: `review/flat_spectrum.md`, *Loops*, item LP7. The feed-forward block of DiffGPT's control
is a bias-free SwiGLU, `∑_u SiLU(a_u z) (b_u z) o_u` (`ComplexFFN.swigluFFN`), with
`SiLU(t) = t/(1 + e^{−t})` (`Landscape.Neural.silu`). A *SwiGLU layer* adds it to the state,
`F z = z + ∑_u SiLU(a_u z) (b_u z) o_u` (`swigluLayer`), with real-linear readings `a_u`, `b_u` of
the complex state. SiLU is a product with Phasor's write gate, `SiLU(t) = t · 2σ(t)/2`
(`silu_eq_mul_writeGate`): a SwiGLU unit is the product `(a_u z)(b_u z)` times half of LP5's
gate on the logit `a_u z`. On the logits `−s ln 2` it is `−s ln 2/(1 + 2^s)`
(`silu_neg_mul_log_two`), and two such units and a line give
`(−1)^s = 1 − 6s − 6s/(1 + 2^s) + 18s²/(1 + 2^s)` for `s ≤ 3` (`neg_one_pow_eq_silu`).

A real-linear reading reads the real and imaginary parts of the coordinates. Conjugation keeps
the order (`star_mem_orderSpan`), so real-linear readings keep it too (`realLinear_mem_orderSpan`).
Where the gate readings `a_u` are constant on the inputs, a SwiGLU layer is affine in the state and
keeps the order of every coordinate (`swigluLayer_order`). `ProductLoopSwiGLUParity` lets them read
the state.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Finset Transformer.MHC
open Transformer.Landscape.Neural (silu)

/-- **SiLU is a product with Phasor's write gate**: `SiLU(t) = t · 2σ(t)/2`. -/
theorem silu_eq_mul_writeGate (t : ℝ) : silu t = t * writeGate t / 2 := by
  rw [silu, writeGate, gate]
  ring

/-- On the logits `−s ln 2`, SiLU is `−s ln 2/(1 + 2^s)`. -/
theorem silu_neg_mul_log_two (s : ℕ) :
    silu (-((s : ℝ) * Real.log 2)) = -((s : ℝ) * Real.log 2) / (1 + 2 ^ s) := by
  rw [silu_eq_mul_writeGate, writeGate_neg_mul_log_two]
  ring

/-- **Two SiLU units and a line give the parity**:
`(−1)^s = 1 − 6s + (6/ln 2) SiLU(−s ln 2) − (18/ln 2) s SiLU(−s ln 2)` for `s ≤ 3`. -/
theorem neg_one_pow_eq_silu {s : ℕ} (hs : s ≤ 3) :
    ((-1 : ℂ) ^ s) = 1 - 6 * (s : ℂ) + (6 / Real.log 2 * silu (-((s : ℝ) * Real.log 2)) : ℝ) -
      (18 / Real.log 2 * (silu (-((s : ℝ) * Real.log 2)) * s) : ℝ) := by
  have h := (Real.log_pos one_lt_two).ne'
  rw [silu_neg_mul_log_two, show 6 / Real.log 2 * (-((s : ℝ) * Real.log 2) / (1 + 2 ^ s)) =
    -6 * s / (1 + 2 ^ s) by field_simp, show 18 / Real.log 2 * (-((s : ℝ) * Real.log 2) /
      (1 + 2 ^ s) * s) = -18 * s ^ 2 / (1 + 2 ^ s) by field_simp]
  interval_cases s <;> norm_num

example := neg_one_pow_eq_silu (s := 3) le_rfl

variable {κ α β : Type}

/-- **Conjugation keeps the order**: `φ̄` reads what `φ` reads. -/
lemma star_mem_orderSpan {r : ℕ} {φ : (κ → α) → (κ → β) → ℂ} (h : φ ∈ orderSpan r) :
    (fun x y => star (φ x y)) ∈ orderSpan r := by
  induction h using Submodule.span_induction with
  | mem φ hφ =>
    obtain ⟨S, hS, hφ⟩ := hφ
    exact mem_orderSpan_of_readsOnly hS fun x x' y y' hx hy => by rw [hφ x x' y y' hx hy]
  | zero =>
    convert (orderSpan r).zero_mem using 1
    funext x y
    simp
  | add φ ψ _ _ hφ hψ =>
    convert add_mem hφ hψ using 1
    funext x y
    simp
  | smul c φ _ hφ =>
    convert Submodule.smul_mem _ (star c) hφ using 1
    funext x y
    simp

example := star_mem_orderSpan (features_mem_orderSpan (κ := Unit) (α := Bool) (β := Bool)
  (T := Unit) (Sum.inr ()))

/-- A real-linear reading of a state whose coordinates have order at most `r` has order at most
`r`: it reads the real and imaginary parts of the coordinates. -/
lemma realLinear_mem_orderSpan {ι : Type*} [Fintype ι] {r : ℕ} {Z : (κ → α) → (κ → β) → ι → ℂ}
    (hZ : ∀ c, (fun x y => Z x y c) ∈ orderSpan r) (ℓ : (ι → ℂ) →ₗ[ℝ] ℝ) :
    (fun x y => (ℓ (Z x y) : ℂ)) ∈ orderSpan r := by
  classical
  have hz (z : ι → ℂ) : z = ∑ c, ((z c).re • (Pi.single c 1 : ι → ℂ) +
      (z c).im • (Pi.single c Complex.I : ι → ℂ)) := by
    funext j
    simp [Finset.sum_apply, Pi.single_apply, Finset.sum_add_distrib, Complex.re_add_im]
  have h : (fun x y => (ℓ (Z x y) : ℂ)) = ∑ c,
      (((ℓ (Pi.single c 1) : ℂ) / 2 + (ℓ (Pi.single c Complex.I) : ℂ) / (2 * Complex.I)) •
          (fun x y => Z x y c) +
        ((ℓ (Pi.single c 1) : ℂ) / 2 - (ℓ (Pi.single c Complex.I) : ℂ) / (2 * Complex.I)) •
          fun x y => star (Z x y c)) := by
    funext x y
    conv_lhs => rw [hz (Z x y)]
    simp only [map_sum, map_add, map_smul, smul_eq_mul, Complex.ofReal_sum, Complex.ofReal_add,
      Complex.ofReal_mul, Finset.sum_apply, Pi.add_apply, Pi.smul_apply]
    refine Finset.sum_congr rfl fun c _ => ?_
    rw [Complex.re_eq_add_conj, Complex.im_eq_sub_conj, Complex.star_def]
    field_simp
    ring
  rw [h]
  exact Submodule.sum_mem _ fun c _ =>
    add_mem (Submodule.smul_mem _ _ (hZ c)) (Submodule.smul_mem _ _ (star_mem_orderSpan (hZ c)))

example (ℓ : ((((Unit × Bool) ⊕ (Unit × Bool)) ⊕ Unit) → ℂ) →ₗ[ℝ] ℝ) :=
  realLinear_mem_orderSpan (features_mem_orderSpan (κ := Unit) (α := Bool) (β := Bool)) ℓ

variable {ι : Type*}

/-- **A SwiGLU layer**: `F z = z + ∑_u SiLU(a_u z) (b_u z) o_u`, the block of
`ComplexFFN.swigluFFN` added to the state, with real-linear readings of the complex state. -/
noncomputable def swigluLayer {η : Type*} [Fintype η] (a b : η → (ι → ℂ) →ₗ[ℝ] ℝ)
    (out : η → ι → ℂ) : (ι → ℂ) → ι → ℂ :=
  fun z => z + ComplexFFN.swigluFFN a b out z

/-- **Constant gate readings keep the order**: where every `a_u` is constant on the inputs, a
SwiGLU layer is affine in the state, and each coordinate keeps its order. -/
theorem swigluLayer_order {η : Type} [Fintype η] [Fintype ι] {r : ℕ}
    (a b : η → (ι → ℂ) →ₗ[ℝ] ℝ) (out : η → ι → ℂ) {Z : (κ → α) → (κ → β) → ι → ℂ}
    (hZ : ∀ c, (fun x y => Z x y c) ∈ orderSpan r) {t : η → ℝ}
    (ht : ∀ u x y, a u (Z x y) = t u) (c : ι) :
    (fun x y => swigluLayer a b out (Z x y) c) ∈ orderSpan r := by
  have h : (fun x y => swigluLayer a b out (Z x y) c) = (fun x y => Z x y c) +
      ∑ u, ((silu (t u) : ℂ) * out u c) • fun x y => (b u (Z x y) : ℂ) := by
    funext x y
    simp only [swigluLayer, ComplexFFN.swigluFFN, Pi.add_apply, Finset.sum_apply, Pi.smul_apply,
      Complex.real_smul, ht, smul_eq_mul, Complex.ofReal_mul]
    congr 1
    exact Finset.sum_congr rfl fun u _ => by ring
  rw [h]
  exact add_mem (hZ c) (Submodule.sum_mem _ fun u _ =>
    Submodule.smul_mem _ _ (realLinear_mem_orderSpan hZ (b u)))

/-- A gate that reads the features of `x₀`, which sum to one, is constant. -/
example {η : Type} [Fintype η]
    (b : η → ((((Fin 3 × ZMod 2) ⊕ (Fin 3 × ZMod 2)) ⊕ Unit) → ℂ) →ₗ[ℝ] ℝ)
    (out : η → (((Fin 3 × ZMod 2) ⊕ (Fin 3 × ZMod 2)) ⊕ Unit) → ℂ)
    (c : ((Fin 3 × ZMod 2) ⊕ (Fin 3 × ZMod 2)) ⊕ Unit) :
    (fun x y => swigluLayer (fun _ => Complex.reLm ∘ₗ (∑ a : ZMod 2,
      LinearMap.proj (R := ℂ) (φ := fun _ => ℂ) (Sum.inl (Sum.inl (0, a)))).restrictScalars ℝ)
        b out (Sum.elim (pairFeatures x y) 0) c) ∈ orderSpan 1 :=
  swigluLayer_order _ b out features_mem_orderSpan (t := fun _ => 1)
    (fun _ x y => by simp [pairFeatures, oneHot]) c

end Transformer.Grokking.FlatSpectrum
