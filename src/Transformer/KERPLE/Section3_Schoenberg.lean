module

public import Mathlib.Analysis.Matrix.Order
public import Mathlib.Analysis.SpecialFunctions.Exponential
public import Transformer.KERPLE.Section3_Facts

/-!
# §3.2 Schoenberg's theorem: `exp(t k̃)` is PD when `k̃` is CPD

Chi, Fan, Ramadge, Rudnicky, arXiv:2205.09921v2, §3.2. The paper cites Facts 3, 5 and 6 from
Berg, Christensen and Ressel (1984), Schölkopf (2000) and Schoenberg (1938); their proofs go
through the theorem proved here.

* Products, powers and exponentials of PD kernels are PD (`IsPDKernel.mul`, Mathlib's Schur
  product theorem; `IsPDKernel.pow`; `IsPDKernel.exp`, through the exponential series), and so
  is `f(x) k(x, y) f(y)` (`IsPDKernel.mul_left_right`).
* For a CPD `k̃` and a point `z`, `k̃(x, y) − k̃(x, z) − k̃(z, y) + k̃(z, z)` is PD
  (`IsCPDKernel.isPDKernel_sub`, Lemma 3.2.1 of Berg, Christensen and Ressel).
* Schoenberg's theorem: if `k̃` is CPD then `exp(t k̃)` is PD for every `t ≥ 0`
  (`IsCPDKernel.isPDKernel_exp_mul`).
* CPD kernels are closed under pointwise limits (`IsCPDKernel.of_tendsto`).
-/

@[expose] public section

namespace Transformer.KERPLE

open Finset Filter Topology

variable {X : Type*}

/-- A kernel equal to a PD kernel is PD. -/
theorem IsPDKernel.congr {k k' : X → X → ℝ} (hk : IsPDKernel k) (h : ∀ x y, k x y = k' x y) :
    IsPDKernel k' := by
  have : k = k' := funext fun x => funext (h x)
  exact this ▸ hk

/-- **Schur product.** The product of two PD kernels is PD. -/
theorem IsPDKernel.mul {k₁ k₂ : X → X → ℝ} (h₁ : IsPDKernel k₁) (h₂ : IsPDKernel k₂) :
    IsPDKernel fun x y => k₁ x y * k₂ x y := by
  rw [isPDKernel_iff_posSemidef] at h₁ h₂ ⊢
  exact h₁.hadamard h₂

/-- Powers of a PD kernel are PD. -/
theorem IsPDKernel.pow {k : X → X → ℝ} (hk : IsPDKernel k) (n : ℕ) :
    IsPDKernel fun x y => k x y ^ n := by
  induction n with
  | zero => simpa using isPDKernel_const (X := X) zero_le_one
  | succ n ih => exact (ih.mul hk).congr fun x y => (pow_succ _ _).symm

/-- The exponential of a PD kernel is PD: it is the sum of the PD kernels `k^n / n!`. -/
theorem IsPDKernel.exp {k : X → X → ℝ} (hk : IsPDKernel k) :
    IsPDKernel fun x y => Real.exp (k x y) := by
  refine ⟨fun x y => congrArg Real.exp (hk.1 x y), fun N x c => ?_⟩
  have hs : HasSum (fun n : ℕ => ∑ i, ∑ j, c i * c j * ((n.factorial : ℝ)⁻¹ * k (x i) (x j) ^ n))
      (∑ i, ∑ j, c i * c j * Real.exp (k (x i) (x j))) := by
    refine hasSum_sum fun i _ => hasSum_sum fun j _ => ?_
    rw [Real.exp_eq_exp_ℝ]
    refine ((NormedSpace.expSeries_div_hasSum_exp (k (x i) (x j))).mul_left (c i * c j)).congr_fun
      ?_
    intro n
    simp only [div_eq_inv_mul]
  refine hs.nonneg fun n => ?_
  rw [sum_sum_smul (fun x y => k x y ^ n)]
  exact mul_nonneg (by positivity) ((hk.pow n).2 N x c)

/-- `f(x) k(x, y) f(y)` is PD when `k` is: its form is that of `k` with the coefficients
`c_i f(x_i)`. -/
theorem IsPDKernel.mul_left_right {k : X → X → ℝ} (hk : IsPDKernel k) (f : X → ℝ) :
    IsPDKernel fun x y => f x * k x y * f y := by
  refine ⟨fun x y => by simp only [hk.1 x y]; ring, fun N x c => ?_⟩
  have h := hk.2 N x fun i => c i * f (x i)
  calc (0 : ℝ) ≤ _ := h
    _ = _ := sum_congr rfl fun i _ => sum_congr rfl fun j _ => by ring

/-- The form at the points `z, x_1, …, x_N` with coefficients `d, c_1, …, c_N`. -/
theorem sum_sum_cons {N : ℕ} (k : X → X → ℝ) (z : X) (x : Fin N → X) (d : ℝ) (c : Fin N → ℝ) :
    ∑ i : Fin (N + 1), ∑ j : Fin (N + 1), (Fin.cons d c : Fin (N + 1) → ℝ) i *
        (Fin.cons d c : Fin (N + 1) → ℝ) j *
        k ((Fin.cons z x : Fin (N + 1) → X) i) ((Fin.cons z x : Fin (N + 1) → X) j) =
      d * d * k z z + d * ∑ j, c j * k z (x j) + d * ∑ i, c i * k (x i) z +
        ∑ i, ∑ j, c i * c j * k (x i) (x j) := by
  simp only [Fin.sum_univ_succ, Fin.cons_zero, Fin.cons_succ, sum_add_distrib, mul_sum]
  ring_nf

/-- **Berg, Christensen and Ressel, Lemma 3.2.1.** For a CPD `k̃` and a point `z`,
`k̃(x, y) − k̃(x, z) − k̃(z, y) + k̃(z, z)` is PD: its form at `x_1, …, x_N` with coefficients
`c` is the form of `k̃` at `z, x_1, …, x_N` with the zero-sum coefficients `−Σ_i c_i, c`. -/
theorem IsCPDKernel.isPDKernel_sub {k : X → X → ℝ} (hk : IsCPDKernel k) (z : X) :
    IsPDKernel fun x y => k x y - k x z - k z y + k z z := by
  refine ⟨fun x y => by simp only [hk.1 x y, hk.1 x z, hk.1 z y]; ring, fun N x c => ?_⟩
  have h := hk.2 (N + 1) (Fin.cons z x) (Fin.cons (-∑ i, c i) c) (by simp [Fin.sum_univ_succ])
  rw [sum_sum_cons] at h
  calc (0 : ℝ) ≤ _ := h
    _ = _ := by
      simp only [mul_add, mul_sub, sum_add_distrib, sum_sub_distrib]
      rw [sum_sum_mul_left (fun i => k (x i) z), sum_sum_mul_right (fun j => k z (x j)),
        sum_sum_const]
      ring

/-- The hypothesis of `IsCPDKernel.isPDKernel_sub` holds for `−‖x − y‖²` on `ℝ`. -/
example :
    IsPDKernel fun x y : ℝ => -‖x - y‖ ^ 2 - -‖x - 0‖ ^ 2 - -‖0 - y‖ ^ 2 + -‖(0 : ℝ) - 0‖ ^ 2 :=
  isCPDKernel_neg_norm_sub_sq.isPDKernel_sub 0

/-- **Schoenberg's theorem, `t = 1`.** If `k̃` is CPD then `exp k̃` is PD: with any point `z`,
`exp k̃(x, y) = f(x) exp(k̃(x, y) − k̃(x, z) − k̃(z, y) + k̃(z, z)) f(y)` for
`f(x) = exp(k̃(x, z) − k̃(z, z) / 2)`. -/
theorem IsCPDKernel.isPDKernel_exp {k : X → X → ℝ} (hk : IsCPDKernel k) :
    IsPDKernel fun x y => Real.exp (k x y) := by
  rcases isEmpty_or_nonempty X with hX | ⟨⟨z⟩⟩
  · exact ⟨fun x => isEmptyElim x, fun N x c => by
      rcases N with _ | N
      · simp
      · exact isEmptyElim (x 0)⟩
  · refine ((hk.isPDKernel_sub z).exp.mul_left_right
      fun x => Real.exp (k x z - k z z / 2)).congr fun x y => ?_
    rw [← Real.exp_add, ← Real.exp_add, hk.1 z y]
    congr 1
    ring

/-- **Schoenberg's theorem.** If `k̃` is CPD then `exp(t k̃)` is PD for every `t ≥ 0`. -/
theorem IsCPDKernel.isPDKernel_exp_mul {k : X → X → ℝ} (hk : IsCPDKernel k) {t : ℝ}
    (ht : 0 ≤ t) : IsPDKernel fun x y => Real.exp (t * k x y) :=
  (hk.smul ht).isPDKernel_exp

/-- The hypotheses of `IsCPDKernel.isPDKernel_exp_mul` hold for `−‖x − y‖²` on `ℝ` and `t = 1`:
the Gaussian kernel is PD. -/
example : IsPDKernel fun x y : ℝ => Real.exp (1 * -‖x - y‖ ^ 2) :=
  isCPDKernel_neg_norm_sub_sq.isPDKernel_exp_mul zero_le_one

/-- CPD kernels are closed under pointwise limits. -/
theorem IsCPDKernel.of_tendsto {ι : Type*} {l : Filter ι} [l.NeBot] {K : ι → X → X → ℝ}
    {k : X → X → ℝ} (hK : ∀ᶠ i in l, IsCPDKernel (K i))
    (hlim : ∀ x y, Tendsto (fun i => K i x y) l (𝓝 (k x y))) : IsCPDKernel k := by
  refine ⟨fun x y => tendsto_nhds_unique (hlim x y) ?_, fun N x c hc => ?_⟩
  · exact (hlim y x).congr' (hK.mono fun i hi => (hi.1 x y).symm)
  · refine ge_of_tendsto (tendsto_finsetSum _ fun i _ => tendsto_finsetSum _ fun j _ =>
      (hlim _ _).const_mul _) ?_
    exact hK.mono fun i hi => hi.2 N x c hc

/-- The hypotheses of `IsCPDKernel.of_tendsto` hold for the constant family `−‖x − y‖²`. -/
example : IsCPDKernel fun x y : ℝ => -‖x - y‖ ^ 2 :=
  IsCPDKernel.of_tendsto (l := atTop) (K := fun (_ : ℕ) (x y : ℝ) => -‖x - y‖ ^ 2)
    (Eventually.of_forall fun _ => isCPDKernel_neg_norm_sub_sq) fun _ _ => tendsto_const_nhds

end Transformer.KERPLE
