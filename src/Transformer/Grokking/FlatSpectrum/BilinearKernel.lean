module

public import Transformer.Grokking.FlatSpectrum.FlatCapacity
public import Transformer.Grokking.FlatSpectrum.FlatSupport

/-!
# Functions of two groups of variables coupled by a bilinear form

Source: `review/flat_spectrum.md`, *Functions of n variables* (added), item NV1. A function of
`n` variables over `ℤ/p`, cut into a group `u` of `k` variables and a group `v` of `l`, is a
`p^k × p^l` table. When the groups meet only through a bilinear form,
`f(u, v) = g(u) + h(v) + uᵀ M v`, its kernel `K = [ψ(f(u, v))]` has orthogonal rows:

  `(K Kᴴ)(u, u') = ψ(g u − g u') ∑_v ψ((u − u')ᵀ M · v) = p^l [(u − u')ᵀ M = 0]`

by the orthogonality of the characters of `(ℤ/p)^l` (`sum_stdAddChar_dotProduct`). So
`K Kᴴ = p^l I` as soon as `w ↦ wᵀ M` is injective (`bilinearKernel_mul_conjTranspose`), and `K`
is flat with `c = p^l` (`isFlat_bilinearKernel`). Theorem A then holds for `k + l` variables:
`K` is the minimum of M from no proper set of cells (`not_isMinOn_bilinearKernel`), and R3
gives every product through width `d` an error of at least `p^l (p^k − d)`.

For `k = l` and an invertible `M` the table is square with `N = p^k` rows, of rank `N` and
nuclear norm `N √N`, the largest of an `N × N` table with entries of modulus one
(`rank_bilinearKernel`, `nuclearNorm_bilinearKernel`). The example is `k` copies of
`x³ + x y`, `f = ∑ᵢ (xᵢ³ + xᵢ yᵢ)`, cut between the `x`s and the `y`s: `M = I`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix

variable (p : ℕ) [NeZero p] {ι κ : Type*} [Fintype ι] [Fintype κ] [DecidableEq ι] [DecidableEq κ]

/-- The kernel of `f(u, v) = g u + h v + uᵀ M v` (`review/flat_spectrum.md`, NV1) on the
groups `u : ι → ℤ/p` and `v : κ → ℤ/p`, with `ψ = exp (2πi · / p)`. -/
noncomputable def bilinearKernel (g : (ι → ZMod p) → ZMod p) (h : (κ → ZMod p) → ZMod p)
    (M : Matrix ι κ (ZMod p)) : Matrix (ι → ZMod p) (κ → ZMod p) ℂ :=
  Matrix.of fun u v => ZMod.stdAddChar (g u + h v + u ⬝ᵥ M *ᵥ v)

/-- `ψ` turns a sum into a product. -/
lemma stdAddChar_sum {α : Type*} (s : Finset α) (f : α → ZMod p) :
    ZMod.stdAddChar (∑ a ∈ s, f a) = ∏ a ∈ s, ZMod.stdAddChar (f a) := by
  classical
  induction s using Finset.induction_on with
  | empty => simp
  | insert a s ha ih =>
    rw [Finset.sum_insert ha, AddChar.map_add_eq_mul, ih, Finset.prod_insert ha]

/-- **Orthogonality of the characters of `(ℤ/p)^κ`** (`review/flat_spectrum.md`, NV1):
`∑_v ψ(w · v) = p^{|κ|} [w = 0]`, the product of the one-variable sums `p [wᵢ = 0]`. -/
lemma sum_stdAddChar_dotProduct (w : κ → ZMod p) :
    ∑ v : κ → ZMod p, ZMod.stdAddChar (w ⬝ᵥ v) =
      if w = 0 then (p : ℂ) ^ Fintype.card κ else 0 := by
  have h1 (v : κ → ZMod p) :
      ZMod.stdAddChar (w ⬝ᵥ v) = ∏ i, ZMod.stdAddChar (v i * w i) := by
    rw [dotProduct, stdAddChar_sum]
    exact Finset.prod_congr rfl fun i _ => by rw [mul_comm]
  simp_rw [h1]
  rw [← Fintype.prod_sum (fun i b => ZMod.stdAddChar (b * w i))]
  simp_rw [sum_stdAddChar_mul, Fintype.prod_ite_zero, Finset.prod_const, Finset.card_univ]
  simp [funext_iff]

example : ∑ v : Fin 2 → ZMod 3, ZMod.stdAddChar (0 ⬝ᵥ v) = (3 : ℂ) ^ 2 := by
  rw [sum_stdAddChar_dotProduct]
  simp

omit [DecidableEq ι] [DecidableEq κ] in
/-- One summand of `K Kᴴ`: `ψ(f(u, v)) conj ψ(f(u', v)) = ψ(g u − g u') ψ((u − u')ᵀ M · v)`. -/
lemma bilinearKernel_mul_star (g : (ι → ZMod p) → ZMod p) (h : (κ → ZMod p) → ZMod p)
    (M : Matrix ι κ (ZMod p)) (u u' : ι → ZMod p) (v : κ → ZMod p) :
    bilinearKernel p g h M u v * star (bilinearKernel p g h M u' v) =
      ZMod.stdAddChar (g u - g u') * ZMod.stdAddChar ((u - u') ᵥ* M ⬝ᵥ v) := by
  simp only [bilinearKernel, of_apply]
  rw [mul_comm, star_stdAddChar_mul, ← AddChar.map_add_eq_mul, sub_vecMul, sub_dotProduct,
    dotProduct_mulVec, dotProduct_mulVec]
  ring_nf

omit [DecidableEq ι] in
/-- **The rows of a bilinearly coupled kernel are orthogonal** (`review/flat_spectrum.md`, NV1):
if `w ↦ wᵀ M` is injective, `K Kᴴ = p^{|κ|} I`. -/
theorem bilinearKernel_mul_conjTranspose (g : (ι → ZMod p) → ZMod p)
    (h : (κ → ZMod p) → ZMod p) {M : Matrix ι κ (ZMod p)} (hM : Function.Injective M.vecMul) :
    bilinearKernel p g h M * (bilinearKernel p g h M)ᴴ = ((p : ℂ) ^ Fintype.card κ) • 1 := by
  ext u u'
  simp only [Matrix.mul_apply, conjTranspose_apply, Matrix.smul_apply, Matrix.one_apply,
    smul_eq_mul, bilinearKernel_mul_star, ← Finset.mul_sum, sum_stdAddChar_dotProduct]
  by_cases huu : u = u'
  · subst huu
    simp
  · have hw : (u - u') ᵥ* M ≠ 0 := fun h0 =>
      huu (sub_eq_zero.1 (hM (by rw [h0, zero_vecMul] : M.vecMul (u - u') = M.vecMul 0)))
    simp [hw, huu]

/-- **A bilinearly coupled kernel is flat** (`review/flat_spectrum.md`, NV1), with
`c = p^{|κ|}`. -/
theorem isFlat_bilinearKernel (g : (ι → ZMod p) → ZMod p) (h : (κ → ZMod p) → ZMod p)
    {M : Matrix ι κ (ZMod p)} (hM : Function.Injective M.vecMul) :
    IsFlat (bilinearKernel p g h M) ((p : ℝ) ^ Fintype.card κ) :=
  Or.inr (by rw [bilinearKernel_mul_conjTranspose p g h hM]; push_cast; rfl)

/-- A cubic in each of two variables coupled by `x y` is flat with `c = p`. -/
example := isFlat_bilinearKernel 5 (ι := Fin 1) (κ := Fin 1) (fun x => x 0 ^ 3) (fun y => y 0)
  (M := 1) (vecMul_injective_of_isUnit isUnit_one)

omit [DecidableEq ι] [DecidableEq κ] in
/-- The entries of `bilinearKernel` have modulus one, so none is zero. -/
lemma bilinearKernel_ne_zero (g : (ι → ZMod p) → ZMod p) (h : (κ → ZMod p) → ZMod p)
    (M : Matrix ι κ (ZMod p)) (u : ι → ZMod p) (v : κ → ZMod p) :
    bilinearKernel p g h M u v ≠ 0 :=
  norm_ne_zero_iff.mp (by simp [bilinearKernel])

/-- **Theorem A for `k + l` variables** (`review/flat_spectrum.md`, NV1): if `w ↦ wᵀ M` is
injective, the kernel of `g u + h v + uᵀ M v` is the minimum of M from no proper set of cells. -/
theorem not_isMinOn_bilinearKernel (g : (ι → ZMod p) → ZMod p) (h : (κ → ZMod p) → ZMod p)
    {M : Matrix ι κ (ZMod p)} (hM : Function.Injective M.vecMul)
    {Ω : Set ((ι → ZMod p) × (κ → ZMod p))} (hΩ : Ω ≠ Set.univ) :
    ¬ IsMinOn nuclearNorm {X | ∀ ij ∈ Ω, X ij.1 ij.2 = bilinearKernel p g h M ij.1 ij.2}
      (bilinearKernel p g h M) :=
  (isFlat_bilinearKernel p g h hM).not_isMinOn (pow_pos (Nat.cast_pos.2 (NeZero.pos p)) _)
    (bilinearKernel_ne_zero p g h M) hΩ

/-- With no cell seen, two copies of `x³ + x y` are not a minimum of M. -/
example := not_isMinOn_bilinearKernel 3 (ι := Fin 2) (κ := Fin 2) (fun x => ∑ i, x i ^ 3)
  (fun _ => 0) (M := 1) (vecMul_injective_of_isUnit isUnit_one) (Ω := ∅) Set.empty_ne_univ

/-- **R3 for `k + l` variables** (`review/flat_spectrum.md`, NV1): every `X` misses the kernel
of `g u + h v + uᵀ M v`, with `w ↦ wᵀ M` injective, by `p^l (min(p^k, p^l) − rank X)`. -/
theorem mul_sub_rank_le_frobeniusSq_sub_bilinearKernel (g : (ι → ZMod p) → ZMod p)
    (h : (κ → ZMod p) → ZMod p) {M : Matrix ι κ (ZMod p)} (hM : Function.Injective M.vecMul)
    (X : Matrix (ι → ZMod p) (κ → ZMod p) ℂ) :
    (p : ℝ) ^ Fintype.card κ * ((min (p ^ Fintype.card ι) (p ^ Fintype.card κ) : ℕ) - X.rank) ≤
      frobeniusSq (bilinearKernel p g h M - X) := by
  have h1 := (isFlat_bilinearKernel p g h hM).mul_sub_rank_le_frobeniusSq_sub
    (pow_pos (Nat.cast_pos.2 (NeZero.pos p)) _) X
  simpa [Fintype.card_fun, ZMod.card] using h1

/-- With `X = 0`, two copies of `x³ + x y` have `‖K‖_S2² ≥ 3² · 3²`. -/
example := mul_sub_rank_le_frobeniusSq_sub_bilinearKernel 3 (ι := Fin 2) (κ := Fin 2)
  (fun x => ∑ i, x i ^ 3) (fun _ => 0) (M := 1) (vecMul_injective_of_isUnit isUnit_one) 0

variable {p}

/-- **Rank `N = p^k`** (`review/flat_spectrum.md`, NV1): for `k = l` and an invertible `M`. -/
theorem rank_bilinearKernel (g h : (ι → ZMod p) → ZMod p) {M : Matrix ι ι (ZMod p)}
    (hM : IsUnit M) : (bilinearKernel p g h M).rank = p ^ Fintype.card ι := by
  rw [(isFlat_bilinearKernel p g h (vecMul_injective_of_isUnit hM)).rank_eq
    (pow_ne_zero _ (Nat.cast_ne_zero.2 (NeZero.ne p)))]
  simp [ZMod.card]

example := rank_bilinearKernel (p := 2) (ι := Fin 3) (fun x => ∑ i, x i ^ 3) 0 isUnit_one

/-- **Nuclear norm `N √N`** (`review/flat_spectrum.md`, NV1): for `k = l` and an invertible
`M`, with `N = p^k`. -/
theorem nuclearNorm_bilinearKernel (g h : (ι → ZMod p) → ZMod p) {M : Matrix ι ι (ZMod p)}
    (hM : IsUnit M) :
    nuclearNorm (bilinearKernel p g h M) =
      (p : ℝ) ^ Fintype.card ι * √((p : ℝ) ^ Fintype.card ι) := by
  rw [(isFlat_bilinearKernel p g h (vecMul_injective_of_isUnit hM)).nuclearNorm_eq
    (pow_ne_zero _ (Nat.cast_ne_zero.2 (NeZero.ne p)))]
  simp [ZMod.card]

example := nuclearNorm_bilinearKernel (p := 2) (ι := Fin 3) (fun x => ∑ i, x i ^ 3) 0 isUnit_one

/-- `k` copies of `x³ + x y` are the kernel of `g x = ∑ᵢ xᵢ³`, `h = 0` and `M = I`. -/
theorem bilinearKernel_cube_add_mul (k : ℕ) :
    bilinearKernel p (fun x : Fin k → ZMod p => ∑ i, x i ^ 3) (fun _ => 0) 1 =
      Matrix.of fun x y => ZMod.stdAddChar (∑ i, (x i ^ 3 + x i * y i)) := by
  ext x y
  simp [bilinearKernel, dotProduct, Finset.sum_add_distrib]

end Transformer.Grokking.FlatSpectrum
