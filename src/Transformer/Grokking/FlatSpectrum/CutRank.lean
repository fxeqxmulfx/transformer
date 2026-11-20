module

public import Mathlib.FieldTheory.Finiteness
public import Transformer.Grokking.FlatSpectrum.BilinearKernel

/-!
# The rank of a function of n variables across a cut

Source: `review/flat_spectrum.md`, *Functions of n variables* (added), item NV5.

Let `f(u, v) = g(u) + h(v) + uᵀ M v` with any `k × l` matrix `M` over `ℤ/p`, not only an
injective one as in NV1. The row `u` of the kernel `K = [ψ(f(u, v))]` is `ψ(g u)` times a row that
depends on `u` only through `uᵀ M`: two rows with the same `uᵀ M` differ by a factor of modulus one
(`bilinearKernel_row_eq`), and two rows with different `uᵀ M` are orthogonal, by the orthogonality
of the characters (`sum_bilinearKernel_mul_star`). So the rank of `K` is the number of distinct
`uᵀ M` (`rank_bilinearKernel_eq_card`), and over a prime `p`, where they form a subspace of
dimension `rank M`, it is `p^{rank M}` (`rank_bilinearKernel_eq_pow`).

For `n` variables the cut decides `M`. Take `k` pairs coupled by `xᵢ yᵢ`, as in `k` copies of
`x³ + x y`: a cut that separates `s` of the pairs has a coupling of rank `s`, and the table across
it has rank `p^s`. Between the `x`s and the `y`s that is `p^k`; in the order `x₁ y₁ x₂ y₂ …`
every cut into a prefix and a suffix separates at most one pair, and the rank is at most `p`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix

variable {p : ℕ} [NeZero p] {ι κ : Type*} [Fintype ι] [Fintype κ] [DecidableEq κ]

/-- **Rows of a bilinearly coupled kernel** (`review/flat_spectrum.md`, NV5): two rows are
orthogonal unless their couplings agree, `∑_v K(u, v) conj K(u', v) = ψ(g u − g u') p^l` when
`uᵀ M = u'ᵀ M` and `0` otherwise. -/
lemma sum_bilinearKernel_mul_star (g : (ι → ZMod p) → ZMod p) (h : (κ → ZMod p) → ZMod p)
    (M : Matrix ι κ (ZMod p)) (u u' : ι → ZMod p) :
    ∑ v, bilinearKernel p g h M u v * star (bilinearKernel p g h M u' v) =
      if u ᵥ* M = u' ᵥ* M then ZMod.stdAddChar (g u - g u') * (p : ℂ) ^ Fintype.card κ
      else 0 := by
  simp only [bilinearKernel_mul_star, ← Finset.mul_sum, sum_stdAddChar_dotProduct, sub_vecMul,
    sub_eq_zero, mul_ite, mul_zero]

example (u : Fin 1 → ZMod 3) :=
  sum_bilinearKernel_mul_star (κ := Fin 1) (fun _ => 0) (fun _ => 0) 0 u u

omit [DecidableEq κ] in
/-- Two rows with the same coupling `uᵀ M = u'ᵀ M` differ by the factor `ψ(g u − g u')`. -/
lemma bilinearKernel_row_eq (g : (ι → ZMod p) → ZMod p) (h : (κ → ZMod p) → ZMod p)
    (M : Matrix ι κ (ZMod p)) {u u' : ι → ZMod p} (huu : u ᵥ* M = u' ᵥ* M) :
    bilinearKernel p g h M u = ZMod.stdAddChar (g u - g u') • bilinearKernel p g h M u' := by
  funext v
  simp only [bilinearKernel, of_apply, Pi.smul_apply, smul_eq_mul, ← AddChar.map_add_eq_mul,
    dotProduct_mulVec, huu]
  ring_nf

example (u : Fin 1 → ZMod 3) :=
  bilinearKernel_row_eq (κ := Fin 1) (fun _ => 0) (fun _ => 0) 0 (u := u) (u' := 0) (by simp)

/-- **The rank across a cut is the number of couplings** (`review/flat_spectrum.md`, NV5): the
kernel of `g u + h v + uᵀ M v` has rank `|{uᵀ M}|`, for every `M`. -/
theorem rank_bilinearKernel_eq_card (g : (ι → ZMod p) → ZMod p) (h : (κ → ZMod p) → ZMod p)
    (M : Matrix ι κ (ZMod p)) :
    (bilinearKernel p g h M).rank = Nat.card (Set.range M.vecMul) := by
  classical
  choose σ hσ using fun w : Set.range M.vecMul => w.2
  have hli : LinearIndependent ℂ fun w => bilinearKernel p g h M (σ w) := by
    rw [Fintype.linearIndependent_iff]
    intro c hc w₀
    have h1 := congrArg
      (fun x : (κ → ZMod p) → ℂ => ∑ v, x v * star (bilinearKernel p g h M (σ w₀) v)) hc
    simp only [Finset.sum_apply, Pi.smul_apply, smul_eq_mul, Pi.zero_apply, zero_mul,
      Finset.sum_const_zero, Finset.sum_mul, mul_assoc] at h1
    rw [Finset.sum_comm] at h1
    simp only [← Finset.mul_sum, sum_bilinearKernel_mul_star, hσ, Subtype.coe_inj, mul_ite,
      mul_zero, Finset.sum_ite_eq', Finset.mem_univ, ite_true, sub_self,
      AddChar.map_zero_eq_one, one_mul] at h1
    exact (mul_eq_zero.1 h1).resolve_right (pow_ne_zero _ (Nat.cast_ne_zero.2 (NeZero.ne p)))
  have hspan : Submodule.span ℂ (Set.range (bilinearKernel p g h M).row) =
      Submodule.span ℂ (Set.range fun w => bilinearKernel p g h M (σ w)) := by
    refine le_antisymm (Submodule.span_le.2 ?_) (Submodule.span_mono ?_)
    · rintro _ ⟨u, rfl⟩
      rw [row_apply', bilinearKernel_row_eq g h M (hσ ⟨u ᵥ* M, u, rfl⟩).symm]
      exact Submodule.smul_mem _ _ (Submodule.subset_span ⟨_, rfl⟩)
    · rintro _ ⟨w, rfl⟩
      exact ⟨σ w, rfl⟩
  rw [rank_eq_finrank_span_row, hspan, finrank_span_eq_card hli, Nat.card_eq_fintype_card]

/-- The coupling `0` has one value: the kernel has rank one. -/
example := rank_bilinearKernel_eq_card (p := 3) (ι := Fin 2) (κ := Fin 2) (fun _ => 0)
  (fun _ => 0) 0

/-- **Over a prime the rank across a cut is `p^{rank M}`** (`review/flat_spectrum.md`, NV5): the
couplings `uᵀ M` form a subspace of dimension `rank M`. -/
theorem rank_bilinearKernel_eq_pow [Fact p.Prime] (g : (ι → ZMod p) → ZMod p)
    (h : (κ → ZMod p) → ZMod p) (M : Matrix ι κ (ZMod p)) :
    (bilinearKernel p g h M).rank = p ^ M.rank := by
  classical
  have hS : Set.range M.vecMul = (LinearMap.range Mᵀ.mulVecLin : Set (κ → ZMod p)) := by
    rw [LinearMap.coe_range]
    exact congrArg Set.range (funext fun u => (mulVec_transpose M u).symm)
  rw [rank_bilinearKernel_eq_card, hS, ← rank_transpose M, Matrix.rank, SetLike.coe_sort_coe,
    Nat.card_eq_fintype_card, Module.card_eq_pow_finrank (K := ZMod p), ZMod.card]

/-- Two pairs coupled by `x₁ y₁` alone: the rank is `3` to the rank of that coupling. -/
example := rank_bilinearKernel_eq_pow (p := 3) (ι := Fin 2) (κ := Fin 2) (fun _ => 0)
  (fun _ => 0) (Matrix.single 0 0 1)

end Transformer.Grokking.FlatSpectrum
