module

public import Transformer.ComplexFFN.Defs
public import Transformer.GPTMini.RMSNorm

/-!
# Symmetries of the complex products and the gaps they force

Source: the blocks of `Defs` (DiffGPT `5a37c51`). Every complex product squares a
sign: its hidden vector at `-z` equals the one at `z`. So the plain, radial and
diagonal FFNs are even functions of their input, whatever the projections. When
the projections are complex-linear (`g (J z) = i g z` for the complex structure
`J`, multiplication by `i` on `ℂ^n`), plain and radial also satisfy
`f(J z) = -f(z)`: `(i g)(i u) = -g u`, and the radial gate sees only `|g|`.

`gap_of_invariant` and `gap_of_anti` turn such symmetries into lower bounds on
any uniform approximation error. The branch RMS keeps both: it is odd
(`rmsNormEps_neg`), so `R(f(J z)) = -R(f z)` and `R(f(-z)) = R(f z)` still hold.

SwiGLU has no such symmetry. Its even part is the quadratic product network
(`swigluFFN_add_neg`), because `SiLU(t) - SiLU(-t) = t`; its odd part is not zero.
-/

@[expose] public section

open scoped BigOperators

namespace Transformer.ComplexFFN

open Transformer.Landscape.Neural (silu silu_sub_neg)

section Gap

variable {V W : Type*} [NormedAddCommGroup W]

/-- An approximant invariant under `σ` misses the part of the target that `σ`
changes: an error bound `δ` on a set holding `z` and `σ z` is at least half of
`‖T z - T (σ z)‖`. -/
theorem gap_of_invariant {f T : V → W} {σ : V → V} (hf : ∀ z, f (σ z) = f z)
    {K : Set V} {δ : ℝ} (hδ : ∀ z ∈ K, ‖f z - T z‖ ≤ δ) {z : V} (hz : z ∈ K)
    (hσ : σ z ∈ K) : ‖T z - T (σ z)‖ ≤ 2 * δ := by
  have h : T z - T (σ z) = (T z - f z) + (f (σ z) - T (σ z)) := by
    rw [hf]
    abel
  rw [h, two_mul]
  exact (norm_add_le _ _).trans
    (add_le_add (by rw [norm_sub_rev]; exact hδ z hz) (hδ _ hσ))

/-- An approximant that `σ` negates misses the part of the target that `σ`
keeps: an error bound `δ` on a set holding `z` and `σ z` is at least half of
`‖T z + T (σ z)‖`. -/
theorem gap_of_anti {f T : V → W} {σ : V → V} (hf : ∀ z, f (σ z) = -f z)
    {K : Set V} {δ : ℝ} (hδ : ∀ z ∈ K, ‖f z - T z‖ ≤ δ) {z : V} (hz : z ∈ K)
    (hσ : σ z ∈ K) : ‖T z + T (σ z)‖ ≤ 2 * δ := by
  have h : T z + T (σ z) = (T z - f z) - (f (σ z) - T (σ z)) := by
    rw [hf]
    abel
  rw [h, two_mul]
  exact (norm_sub_le _ _).trans
    (add_le_add (by rw [norm_sub_rev]; exact hδ z hz) (hδ _ hσ))

example (z : ℝ) : ‖z ^ 2 - (-z) ^ 2‖ ≤ 2 * 0 :=
  gap_of_invariant (f := fun x : ℝ => x ^ 2) (T := fun x => x ^ 2) (σ := Neg.neg) (fun x => by ring)
    (K := Set.univ) (δ := 0) (fun x _ => by simp) (Set.mem_univ z) (Set.mem_univ _)

example (z : ℝ) : ‖z ^ 3 + (-z) ^ 3‖ ≤ 2 * 0 :=
  gap_of_anti (f := fun x : ℝ => x ^ 3) (T := fun x => x ^ 3) (σ := Neg.neg)
    (fun x => by ring) (K := Set.univ) (δ := 0) (fun x _ => by simp) (Set.mem_univ z) (Set.mem_univ _)

end Gap

/-- The branch RMS of the model, `x √d / √(‖x‖² + d ε)`, is odd; composing with it
keeps the symmetries of this file. -/
theorem rmsNormEps_neg {d : ℕ} (eps : ℝ) (x : EucSpace d) :
    GPTMini.rmsNormEps eps (-x) = -GPTMini.rmsNormEps eps x := by
  rw [GPTMini.rmsNormEps, GPTMini.rmsNormEps, norm_neg, smul_neg]

variable {V W ι κ : Type*} [Fintype ι] [Fintype κ]
variable [AddCommGroup V] [Module ℝ V] [AddCommGroup W] [Module ℝ W]

/-- SwiGLU's even part is a product network: `S(z) + S(-z) = Σ_j (a_j z)(b_j z) c_j`. -/
theorem swigluFFN_add_neg (a b : ι → V →ₗ[ℝ] ℝ) (c : ι → W) (z : V) :
    swigluFFN a b c z + swigluFFN a b c (-z) = productFFN a b c z := by
  simp only [swigluFFN, productFFN, ← Finset.sum_add_distrib, map_neg, ← add_smul]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [show silu (a j z) * b j z + silu (-a j z) * -b j z =
      (silu (a j z) - silu (-a j z)) * b j z by ring, silu_sub_neg]

/-- Every product network is even. -/
theorem productFFN_neg (a b : ι → V →ₗ[ℝ] ℝ) (c : ι → W) (z : V) :
    productFFN a b c (-z) = productFFN a b c z := by
  simp only [productFFN, map_neg, neg_mul_neg]

/-- The plain product squares the sign of its input. -/
theorem plainFFN_neg (g u : κ → V →ₗ[ℝ] ℂ) (c c' : κ → W) (z : V) :
    plainFFN g u c c' (-z) = plainFFN g u c c' z := by
  have h : plainHidden g u (-z) = plainHidden g u z := by
    funext k
    simp only [plainHidden, map_neg, neg_mul_neg]
  rw [plainFFN, plainFFN, h]

/-- The radial product squares the sign of its input; its gate sees `|g|`. -/
theorem radialFFN_neg (ε : ℝ) (g u : κ → V →ₗ[ℝ] ℂ) (c c' : κ → W) (z : V) :
    radialFFN ε g u c c' (-z) = radialFFN ε g u c c' z := by
  have h : radialHidden ε g u (-z) = radialHidden ε g u z := by
    funext k
    simp only [radialHidden, map_neg, neg_mul_neg,
      radialGate_of_normSq_eq ε (Complex.normSq_neg _)]
  rw [radialFFN, radialFFN, h]

/-- The coordinate product squares the sign of each coordinate of its input. -/
theorem diagonalFFN_neg (ε : ℝ) (g u : κ → V →ₗ[ℝ] ℂ) (c c' : κ → W) (z : V) :
    diagonalFFN ε g u c c' (-z) = diagonalFFN ε g u c c' z := by
  have h : diagonalHidden ε g u (-z) = diagonalHidden ε g u z := by
    funext k
    simp only [diagonalHidden, map_neg, Complex.neg_re, Complex.neg_im, neg_mul_neg,
      radialGate_of_normSq_eq ε (Complex.normSq_neg _)]
  rw [diagonalFFN, diagonalFFN, h]

/-- With complex-linear projections, `(i g)(i u) = -g u`: the plain FFN is
negated by the complex structure `J`. -/
theorem plainFFN_anti (J : V → V) (g u : κ → V →ₗ[ℝ] ℂ)
    (hg : ∀ k z, g k (J z) = Complex.I * g k z) (hu : ∀ k z, u k (J z) = Complex.I * u k z)
    (c c' : κ → W) (z : V) : plainFFN g u c c' (J z) = -plainFFN g u c c' z := by
  rw [plainFFN, plainFFN, ← complexReadout_neg]
  congr 1
  funext k
  simp only [plainHidden, Pi.neg_apply, hg, hu]
  linear_combination (g k z * u k z / (Real.sqrt 2 : ℂ)) * Complex.I_mul_I

/-- With complex-linear projections the radial FFN is also negated by `J`: its
gate is unchanged, since `|i g| = |g|`. -/
theorem radialFFN_anti (ε : ℝ) (J : V → V) (g u : κ → V →ₗ[ℝ] ℂ)
    (hg : ∀ k z, g k (J z) = Complex.I * g k z) (hu : ∀ k z, u k (J z) = Complex.I * u k z)
    (c c' : κ → W) (z : V) : radialFFN ε g u c c' (J z) = -radialFFN ε g u c c' z := by
  rw [radialFFN, radialFFN, ← complexReadout_neg]
  congr 1
  funext k
  have hs : radialGate ε (Complex.I * g k z) = radialGate ε (g k z) :=
    radialGate_of_normSq_eq ε (by rw [Complex.normSq_mul, Complex.normSq_I, one_mul])
  simp only [radialHidden, Pi.neg_apply, hg, hu, hs]
  linear_combination ((radialGate ε (g k z) : ℂ) * (g k z * u k z) / (Real.sqrt 2 : ℂ)) *
    Complex.I_mul_I

/-- The complex structure of `ℂ`, with the identity as gate and value projection. -/
example (z : ℂ) :
    plainFFN (fun _ : Unit => LinearMap.id) (fun _ => LinearMap.id) (fun _ => (1 : ℂ))
      (fun _ => Complex.I) (Complex.I * z) =
    -plainFFN (fun _ : Unit => LinearMap.id) (fun _ => LinearMap.id) (fun _ => (1 : ℂ))
      (fun _ => Complex.I) z :=
  plainFFN_anti (Complex.I * ·) _ _ (fun _ _ => rfl) (fun _ _ => rfl) _ _ z

example (z : ℂ) :
    radialFFN (2 ^ (-23 : ℤ)) (fun _ : Unit => LinearMap.id) (fun _ => LinearMap.id)
      (fun _ => (1 : ℂ)) (fun _ => Complex.I) (Complex.I * z) =
    -radialFFN (2 ^ (-23 : ℤ)) (fun _ : Unit => LinearMap.id) (fun _ => LinearMap.id)
      (fun _ => (1 : ℂ)) (fun _ => Complex.I) z :=
  radialFFN_anti _ (Complex.I * ·) _ _ (fun _ _ => rfl) (fun _ _ => rfl) _ _ z

end Transformer.ComplexFFN
