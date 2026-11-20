module

public import Transformer.ComplexFFN.Incomparable

/-!
# Signs survive the branch RMS

Source: DiffGPT (`~/python_projects/gpt`, `5a37c51`), `research/main_model/README.txt`:
the branch is `f = R_D(W_F (g ⊙_ℂ u)/√2)` with the parameter-free
`R_n(x) = x/√(mean(x²) + ε)`, which is `GPTMini.rmsNormEps`.

`R_D` multiplies its input by a nonnegative scalar. It erases magnitudes, and with them
the witnesses of `swigluFFN_gap_radial` and `swigluFFN_gap_diagonal`: with `eps = 0`, one
gated unit after `R_D` is its ungated product after `R_D` (`RMSErased`). It keeps the
sign of every linear functional, and signs suffice. If `R_D ∘ S` is within `δ` of an even
target `w` and `δ ‖ℓ‖ < ⟪ℓ, w z⟫`, then `⟪ℓ, S z⟫ > 0` and `⟪ℓ, S(-z)⟫ > 0`
(`pos_inner_of_rmsNormEps_near`), so the product network `P z = S z + S(-z)` of
`swigluFFN_add_neg` has `⟪ℓ, P z⟫ > 0`. Product networks
satisfy `P(αx + βy) - P(αx - βy) = P(βx + αy) - P(βx - αy)` (`productFFN_cross`). A
target whose signs along `ℓ` at these four points contradict the identity by a margin
`γ` is missed by `γ / ‖ℓ‖` at any width (`swigluFFN_rms_gap`).

`inner_sub_rmsNormEps_pair` reads the normalized `A e + B e'` along `e - e'`;
`RMSIncomparable` applies both to the gated products.
-/

@[expose] public section

open scoped BigOperators RealInnerProductSpace

namespace Transformer.ComplexFFN

variable {d : ℕ}

/-- `R_D` multiplies by a nonnegative scalar: a functional positive on the normalized
vector is positive on the vector. -/
theorem pos_inner_of_pos_inner_rmsNormEps (eps : ℝ) {ℓ x : EucSpace d}
    (h : 0 < ⟪ℓ, GPTMini.rmsNormEps eps x⟫) : 0 < ⟪ℓ, x⟫ := by
  rw [GPTMini.rmsNormEps, real_inner_smul_right] at h
  exact pos_of_mul_pos_right h (div_nonneg (Real.sqrt_nonneg _) (Real.sqrt_nonneg _))

/-- A normalized vector within `δ` of `w`, with `δ ‖ℓ‖ < ⟪ℓ, w⟫`, comes from a vector `x`
with `⟪ℓ, x⟫ > 0`. -/
theorem pos_inner_of_rmsNormEps_near (eps : ℝ) {ℓ w x : EucSpace d} {δ : ℝ}
    (hx : ‖GPTMini.rmsNormEps eps x - w‖ ≤ δ) (hw : δ * ‖ℓ‖ < ⟪ℓ, w⟫) : 0 < ⟪ℓ, x⟫ := by
  refine pos_inner_of_pos_inner_rmsNormEps eps ?_
  have h1 := real_inner_le_norm ℓ (w - GPTMini.rmsNormEps eps x)
  rw [inner_sub_right, norm_sub_rev] at h1
  have h2 := mul_le_mul_of_nonneg_left hx (norm_nonneg ℓ)
  linarith

/-- With `eps = 0`, `R_D` fixes a vector of norm `√d`, here `e₀` in `ℝ¹`. -/
theorem rmsNormEps_single_one :
    GPTMini.rmsNormEps 0 (EuclideanSpace.single (0 : Fin 1) (1 : ℝ)) =
      EuclideanSpace.single 0 1 := by
  simp [GPTMini.rmsNormEps]

example : 0 < ⟪EuclideanSpace.single (0 : Fin 1) (1 : ℝ), EuclideanSpace.single 0 1⟫ :=
  pos_inner_of_pos_inner_rmsNormEps 0 (by rw [rmsNormEps_single_one]; simp)

example : 0 < ⟪EuclideanSpace.single (0 : Fin 1) (1 : ℝ), EuclideanSpace.single 0 1⟫ :=
  pos_inner_of_rmsNormEps_near 0 (w := EuclideanSpace.single 0 1) (δ := 0)
    (by rw [rmsNormEps_single_one, sub_self, norm_zero]) (by simp)

/-- Product networks satisfy `P(αx + βy) - P(αx - βy) = P(βx + αy) - P(βx - αy)`: both
sides are `2αβ` times the polar form of `P` at `x, y`. -/
theorem productFFN_cross {V W ι : Type*} [Fintype ι] [AddCommGroup V] [Module ℝ V]
    [AddCommGroup W] [Module ℝ W] (a b : ι → V →ₗ[ℝ] ℝ) (c : ι → W) (x y : V) (α β : ℝ) :
    productFFN a b c (α • x + β • y) - productFFN a b c (α • x - β • y) =
      productFFN a b c (β • x + α • y) - productFFN a b c (β • x - α • y) := by
  simp only [productFFN, ← Finset.sum_sub_distrib, ← sub_smul, map_add, map_sub, map_smul,
    smul_eq_mul]
  refine Finset.sum_congr rfl fun j _ => ?_
  congr 1
  ring

variable {V ι : Type*} [Fintype ι] [AddCommGroup V] [Module ℝ V]

/-- After the branch RMS, a bias-free SwiGLU FFN of any width misses an even target `w`
by `γ / ‖ℓ‖` when `γ` bounds the signed readings `±⟪ℓ, w⟫` at `αx ± βy` and `βx ± αy`:
the signs `+, -, -, +` contradict `productFFN_cross`. -/
theorem swigluFFN_rms_gap {w : V → EucSpace d} (hw : ∀ z, w (-z) = w z) (eps : ℝ)
    (a b : ι → V →ₗ[ℝ] ℝ) (c : ι → EucSpace d) {K : Set V} (hK : ∀ z ∈ K, -z ∈ K) {δ : ℝ}
    (hδ : ∀ z ∈ K, ‖GPTMini.rmsNormEps eps (swigluFFN a b c z) - w z‖ ≤ δ)
    (ℓ : EucSpace d) {x y : V} {α β γ : ℝ}
    (h₁ : α • x + β • y ∈ K) (h₂ : α • x - β • y ∈ K) (h₃ : β • x + α • y ∈ K)
    (h₄ : β • x - α • y ∈ K)
    (hγ₁ : γ ≤ ⟪ℓ, w (α • x + β • y)⟫) (hγ₂ : γ ≤ -⟪ℓ, w (α • x - β • y)⟫)
    (hγ₃ : γ ≤ -⟪ℓ, w (β • x + α • y)⟫) (hγ₄ : γ ≤ ⟪ℓ, w (β • x - α • y)⟫) :
    γ ≤ δ * ‖ℓ‖ := by
  by_contra! hlt
  have key : ∀ (m : EucSpace d) (z : V), z ∈ K → δ * ‖m‖ < ⟪m, w z⟫ →
      0 < ⟪m, productFFN a b c z⟫ := by
    intro m z hz hm
    rw [← swigluFFN_add_neg, inner_add_right]
    exact add_pos (pos_inner_of_rmsNormEps_near eps (hδ z hz) hm)
      (pos_inner_of_rmsNormEps_near eps (hδ (-z) (hK z hz)) (by rwa [hw]))
  have k₁ := key ℓ _ h₁ (by linarith)
  have k₂ := key (-ℓ) _ h₂ (by rw [norm_neg, inner_neg_left]; linarith)
  have k₃ := key (-ℓ) _ h₃ (by rw [norm_neg, inner_neg_left]; linarith)
  have k₄ := key ℓ _ h₄ (by linarith)
  rw [inner_neg_left] at k₂ k₃
  have hc := congrArg (fun v => ⟪ℓ, v⟫) (productFFN_cross a b c x y α β)
  simp only [inner_sub_right] at hc
  linarith

/-- The zero target, met exactly by the zero SwiGLU FFN: the bound reads `0 ≤ 0`. -/
example (x y : ℝ) :
    (0 : ℝ) ≤ 0 * ‖(0 : EucSpace 1)‖ :=
  swigluFFN_rms_gap (w := fun _ : ℝ => 0) (fun _ => rfl) 0 (ι := Fin 0) (fun _ => 0)
    (fun _ => 0) (fun _ => 0) (K := Set.univ) (fun _ _ => Set.mem_univ _)
    (fun z _ => by simp [swigluFFN, GPTMini.rmsNormEps]) 0 (x := x) (y := y) (α := 1) (β := 1)
    (Set.mem_univ _) (Set.mem_univ _) (Set.mem_univ _) (Set.mem_univ _)
    (by simp) (by simp) (by simp) (by simp)

/-- Read along `e - e'`, the normalized `A e + B e'` of orthonormal `e, e'` is
`√d (A - B) / √(A² + B² + d ε)`. -/
theorem inner_sub_rmsNormEps_pair (eps : ℝ) {e e' : EucSpace d} (he : ‖e‖ = 1)
    (he' : ‖e'‖ = 1) (hee : ⟪e, e'⟫ = 0) (A B : ℝ) :
    ⟪e - e', GPTMini.rmsNormEps eps (A • e + B • e')⟫ =
      Real.sqrt d * (A - B) / Real.sqrt (A ^ 2 + B ^ 2 + d * eps) := by
  have he'e : ⟪e', e⟫ = 0 := (real_inner_comm e e').trans hee
  have hn : ‖A • e + B • e'‖ ^ 2 = A ^ 2 + B ^ 2 := by
    rw [norm_add_sq_real, norm_smul, norm_smul, real_inner_smul_left, real_inner_smul_right,
      hee, he, he', Real.norm_eq_abs, Real.norm_eq_abs]
    ring_nf
    rw [sq_abs, sq_abs]
  simp only [GPTMini.rmsNormEps, hn, real_inner_smul_right, inner_sub_left, inner_add_right,
    real_inner_self_eq_norm_sq, hee, he'e, he, he']
  ring

/-- With `ε > 0` two normalized vectors are within `2√d`, so every uniform error below
is at most `2√d`. -/
theorem norm_rmsNormEps_sub_le {eps : ℝ} (heps : 0 < eps) (hd : 0 < d) (u v : EucSpace d) :
    ‖GPTMini.rmsNormEps eps u - GPTMini.rmsNormEps eps v‖ ≤ 2 * Real.sqrt d := by
  have h1 := GPTMini.rmsNormEps_norm_le eps heps hd u
  have h2 := GPTMini.rmsNormEps_norm_le eps heps hd v
  linarith [norm_sub_le (GPTMini.rmsNormEps eps u) (GPTMini.rmsNormEps eps v)]

example (u v : EucSpace 1) := norm_rmsNormEps_sub_le one_pos one_pos u v

/-- Two orthonormal vectors are `√2` apart. -/
theorem norm_sub_of_orthonormal {e e' : EucSpace d} (he : ‖e‖ = 1) (he' : ‖e'‖ = 1)
    (hee : ⟪e, e'⟫ = 0) : ‖e - e'‖ = Real.sqrt 2 := by
  rw [← Real.sqrt_sq (norm_nonneg (e - e')), norm_sub_sq_real, he, he', hee]
  norm_num

/-- The coordinate vectors `e₀, e₁` of `ℝ²` are orthonormal. -/
theorem single_orthonormal :
    ‖EuclideanSpace.single (0 : Fin 2) (1 : ℝ)‖ = 1 ∧
      ‖EuclideanSpace.single (1 : Fin 2) (1 : ℝ)‖ = 1 ∧
      ⟪EuclideanSpace.single (0 : Fin 2) (1 : ℝ), EuclideanSpace.single 1 1⟫ = 0 := by
  simp [EuclideanSpace.inner_single_left]

example (eps A B : ℝ) := inner_sub_rmsNormEps_pair eps single_orthonormal.1
  single_orthonormal.2.1 single_orthonormal.2.2 A B

example := norm_sub_of_orthonormal single_orthonormal.1 single_orthonormal.2.1
  single_orthonormal.2.2

end Transformer.ComplexFFN
