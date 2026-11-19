module

public import Mathlib.Analysis.InnerProductSpace.Completion
public import Transformer.KERPLE.Section3_Powers

/-!
# §3.1 Fact 1: CPD kernels as negative squared distances in a Hilbert space

Chi, Fan, Ramadge, Rudnicky, arXiv:2205.09921v2, §3.1, Fact 1 (Berg, Christensen and Ressel
1984; Proposition 5 of Schölkopf 2000): "Let `k̃ : X × X → (−∞, 0]` be a CPD kernel with
`k̃(x, x) = 0` for all `x`. Then there exists a Hilbert space `H` and a mapping `φ : X → H` such
that `‖φ(x) − φ(x')‖² = −k̃(x, x')`."

* "PD kernels represent inner products" (§3.1): a PD kernel `K` is `⟪ψ(x), ψ(y)⟫` for a map `ψ`
  into a Hilbert space (`IsPDKernel.exists_feature_map`). The space is the completion of the
  finitely supported functions `X →₀ ℝ` under the semi-inner product `Σ_x Σ_y f(x) g(y) K(x, y)`,
  and `ψ(x)` is the indicator of `x`.
* Fact 1 (`IsCPDKernel.exists_embedding`): for a point `z`, the feature map of the PD kernel
  `(k̃(x, y) − k̃(x, z) − k̃(z, y) + k̃(z, z)) / 2` has `‖ψ(x) − ψ(x')‖² = −k̃(x, x')`. The range
  `(−∞, 0]` is not needed: it follows from the other hypotheses (`IsCPDKernel.nonpos_of_diag`).
* Conversely, a negative squared distance `−‖φ(x) − φ(x')‖²` into a real inner product space is
  CPD, vanishes on the diagonal and is nonpositive (`IsCPDKernel.of_embedding`).
-/

@[expose] public section

namespace Transformer.KERPLE

open Finset

universe u

variable {X : Type u}

/-- **PD kernels represent inner products** (§3.1): a PD kernel `K` on `X` is
`K(x, y) = ⟪ψ(x), ψ(y)⟫` for a map `ψ` from `X` into a Hilbert space. -/
theorem IsPDKernel.exists_feature_map {K : X → X → ℝ} (hK : IsPDKernel K) :
    ∃ (H : Type u) (_ : NormedAddCommGroup H) (_ : InnerProductSpace ℝ H) (_ : CompleteSpace H)
      (ψ : X → H), ∀ x y, inner ℝ (ψ x) (ψ y) = K x y := by
  classical
  let B : (X →₀ ℝ) → (X →₀ ℝ) → ℝ := fun f g => f.sum fun x a => a * g.sum fun y b => b * K x y
  have hexp : ∀ f g, B f g = f.sum fun x a => g.sum fun y b => a * (b * K x y) := fun f g => by
    simp only [B, Finsupp.mul_sum]
  have hsymm : ∀ f g, B g f = B f g := fun f g => by
    rw [hexp, hexp, Finsupp.sum_comm]
    refine Finsupp.sum_congr fun x _ => Finsupp.sum_congr fun y _ => ?_
    rw [hK.1 y x]
    ring
  have hnonneg : ∀ f, 0 ≤ B f f := fun f => by
    have h := hK.sum_nonneg (fun i : f.support => (i : X)) fun i => f i
    rw [hexp]
    simp only [Finsupp.sum]
    rw [← Finset.sum_coe_sort f.support]
    refine h.trans_eq (Finset.sum_congr rfl fun i _ => ?_)
    rw [← Finset.sum_coe_sort f.support]
    exact Finset.sum_congr rfl fun j _ => by ring
  let cd : PreInnerProductSpace.Core ℝ (X →₀ ℝ) :=
    { inner := B
      conj_inner_symm := fun f g => by simp [hsymm]
      re_inner_nonneg := fun f => by simpa using hnonneg f
      add_left := fun f₁ f₂ g =>
        Finsupp.sum_add_index' (fun _ => zero_mul _) fun _ _ _ => add_mul _ _ _
      smul_left := fun f g r => by
        simp only [B, conj_trivial]
        rw [Finsupp.sum_smul_index' fun _ => zero_mul _, Finsupp.mul_sum]
        exact Finsupp.sum_congr fun x _ => by rw [smul_eq_mul, mul_assoc] }
  let _ : SeminormedAddCommGroup (X →₀ ℝ) :=
    InnerProductSpace.Core.toSeminormedAddCommGroup (c := cd)
  let _ : InnerProductSpace ℝ (X →₀ ℝ) := InnerProductSpace.ofCore cd
  refine ⟨UniformSpace.Completion (X →₀ ℝ), inferInstance, inferInstance, inferInstance,
    fun x => ((Finsupp.single x 1 : X →₀ ℝ) : UniformSpace.Completion (X →₀ ℝ)), fun x y => ?_⟩
  rw [UniformSpace.Completion.inner_coe]
  show B (Finsupp.single x 1) (Finsupp.single y 1) = K x y
  simp [B, Finsupp.sum_single_index]

/-- The hypothesis of `IsPDKernel.exists_feature_map` holds for the Gaussian kernel on `ℝ`. -/
example : ∃ (H : Type) (_ : NormedAddCommGroup H) (_ : InnerProductSpace ℝ H) (_ : CompleteSpace H)
    (ψ : ℝ → H), ∀ x y, inner ℝ (ψ x) (ψ y) = Real.exp (1 * -‖x - y‖ ^ 2) :=
  (isCPDKernel_neg_norm_sub_sq.isPDKernel_exp_mul zero_le_one).exists_feature_map

/-- **Fact 1.** A CPD kernel `k̃` with `k̃(x, x) = 0` is a negative squared distance:
`‖φ(x) − φ(x')‖² = −k̃(x, x')` for a map `φ` from `X` into a Hilbert space. The paper also asks
`k̃ ≤ 0`, which follows (`IsCPDKernel.nonpos_of_diag`). -/
theorem IsCPDKernel.exists_embedding {k : X → X → ℝ} (hk : IsCPDKernel k)
    (hdiag : ∀ x, k x x = 0) :
    ∃ (H : Type u) (_ : NormedAddCommGroup H) (_ : InnerProductSpace ℝ H) (_ : CompleteSpace H)
      (φ : X → H), ∀ x x', ‖φ x - φ x'‖ ^ 2 = -k x x' := by
  rcases isEmpty_or_nonempty X with hX | ⟨⟨z⟩⟩
  · obtain ⟨H, i₁, i₂, i₃, ψ, -⟩ := (isPDKernel_const (X := X) le_rfl).exists_feature_map
    exact ⟨H, i₁, i₂, i₃, ψ, fun x => isEmptyElim x⟩
  obtain ⟨H, i₁, i₂, i₃, ψ, hψ⟩ :=
    ((hk.isPDKernel_sub z).smul (by norm_num : (0 : ℝ) ≤ 1 / 2)).exists_feature_map
  refine ⟨H, i₁, i₂, i₃, ψ, fun x x' => ?_⟩
  rw [norm_sub_sq_real, ← real_inner_self_eq_norm_sq, ← real_inner_self_eq_norm_sq, hψ, hψ, hψ]
  simp only [hdiag]
  rw [hk.1 z x, hk.1 z x']
  ring

/-- The hypotheses of `IsCPDKernel.exists_embedding` hold for `−‖x − y‖²` on `ℝ`. -/
example : ∃ (H : Type) (_ : NormedAddCommGroup H) (_ : InnerProductSpace ℝ H) (_ : CompleteSpace H)
    (φ : ℝ → H), ∀ x x', ‖φ x - φ x'‖ ^ 2 = -(-‖x - x'‖ ^ 2) :=
  isCPDKernel_neg_norm_sub_sq.exists_embedding fun x => by simp

/-- **The converse of Fact 1.** A negative squared distance, `k̃(x, x') = −‖φ(x) − φ(x')‖²` for a
map `φ` into a real inner product space, is CPD (Fact 4 pulled back along `φ`), vanishes on the
diagonal and takes values in `(−∞, 0]`. -/
theorem IsCPDKernel.of_embedding {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    {k : X → X → ℝ} (φ : X → E) (h : ∀ x x', ‖φ x - φ x'‖ ^ 2 = -k x x') :
    IsCPDKernel k ∧ (∀ x, k x x = 0) ∧ ∀ x x', k x x' ≤ 0 := by
  refine ⟨(isCPDKernel_neg_norm_sub_sq.comp φ).congr fun x x' => ?_, fun x => ?_,
    fun x x' => ?_⟩
  · rw [h, neg_neg]
  · have := h x x
    rw [sub_self, norm_zero] at this
    linarith
  · linarith [h x x', sq_nonneg ‖φ x - φ x'‖]

/-- The hypothesis of `IsCPDKernel.of_embedding` holds for `−(x − x')²` on `ℝ` and `φ = id`. -/
example : IsCPDKernel (fun x x' : ℝ => -(x - x') ^ 2) ∧ (∀ x : ℝ, -(x - x) ^ 2 = 0) ∧
    ∀ x x' : ℝ, -(x - x') ^ 2 ≤ 0 :=
  IsCPDKernel.of_embedding (fun x : ℝ => x) fun x x' => by rw [Real.norm_eq_abs, sq_abs, neg_neg]

end Transformer.KERPLE
