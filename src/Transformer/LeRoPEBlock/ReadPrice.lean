module

public import Transformer.LeRoPEBlock.ReadDFT
public import Mathlib.LinearAlgebra.FiniteDimensional.Basic
public import Mathlib.LinearAlgebra.Dimension.Constructions
public import Mathlib.LinearAlgebra.StdBasis

/-!
# The price of reading keys with several heads

Source: `review/lerope_block.md`, *Reading a slot* (added), RS4.

**Errors.** The heads turned by `i − s` for all keys `s` of the query position `i ≤ 2R` put `p` on
their key and `q` on the others (`depthWeight_dftScores`), so their averages move by at least
`p − q` times as much as the values, `∑_s ‖o_s‖² ≥ (p − q)² ∑_j ‖v_j‖²` (`dft_read_stable`):
`p − q` is the smallest singular value of their weights, and the reading at most multiplies an
error by `1 / (p − q)`. On `N = 2R + 1` keys `p − q = (P − 1)/(P + 2R)` with
`P = N · N^{1/(2R)} > N + 1`, since `ln N > 1 − 1/N`, so `p − q > 1/2` (`half_lt_dftGap`).

**Count.** A combination of heads reads the key `t` exactly when it combines their weight vectors
into the indicator `e_t` (`sum_smul_attend_eq`). Softmax weights are positive, so no weight vector
vanishes anywhere. If `H` heads read every key of a set `T` of `H` keys, the indicators of `T` span
an `H`-dimensional subspace inside the span of the `H` weight vectors, hence all of it, and every
weight vector vanishes off `T`. So when some key lies outside `T`, `T` has at most `H − 1` keys
(`card_lt_of_read`). One uniform head and `H − 1` turned DFT heads read `H − 1` keys
(`dft_read_uniform`): `H − 1` is what a layer of `H` heads reads exactly from a longer context.
-/

@[expose] public section

open Finset Module
open Transformer.AttentionResiduals Transformer.Residual

namespace Transformer.LeRoPEBlock

variable {F : Type*} [NormedAddCommGroup F] [InnerProductSpace ℝ F]

/-- **The reading is stable**: the averages of the DFT heads turned by `i − s` for all keys `s`
have `∑_s ‖o_s‖² ≥ (p − q)² ∑_j ‖v_j‖²`, `p − q = (P − 1)/(P + i)`. -/
theorem dft_read_stable {X : Type*} {R i : ℕ} (hR : 1 ≤ R) (hi : i ≤ 2 * R) (x : ℕ → X)
    (v : Fin (i + 1) → F) :
    dftGap R i ^ 2 * ∑ j, ‖v j‖ ^ 2 ≤
      ∑ s : Fin (i + 1), ‖attend (dftScores R (i - s) i x) v‖ ^ 2 := by
  have hP : 1 ≤ dftPow R i := Real.one_le_rpow (by linarith [(i.cast_nonneg : (0 : ℝ) ≤ i)])
    (by positivity)
  have hpos : 0 < dftPow R i + i := by positivity
  simp only [attend, fun s => funext (depthWeight_dftScores hR hi x s), dftGap_eq]
  exact sum_norm_sq_weightedSum_ite_ge (by positivity) (div_le_div_of_nonneg_right hP hpos.le) v

/-- Three keys, one band. -/
example (v : Fin 3 → ℝ) := dft_read_stable (X := Unit) (R := 1) (i := 2) le_rfl le_rfl
  (fun _ => ()) v

/-- **At `i = 2R` the gap is more than `1/2`**: `P = N · N^{1/(2R)} > N + 1` for `N = 2R + 1`,
since `ln N > 1 − 1/N`. -/
theorem half_lt_dftGap {R : ℕ} (hR : 1 ≤ R) : 1 / 2 < dftGap R (2 * R) := by
  have hR0 : (1 : ℝ) ≤ R := by exact_mod_cast hR
  have hN : (0 : ℝ) < 2 * R + 1 := by positivity
  have hlog : 1 - (2 * (R : ℝ) + 1)⁻¹ < Real.log (2 * R + 1) := by
    have h := Real.log_lt_sub_one_of_pos (inv_pos.2 hN) (by
      rw [Ne, inv_eq_one]; linarith)
    rw [Real.log_inv] at h
    linarith
  have hP : dftPow R (2 * R) = (2 * R + 1) * Real.exp (Real.log (2 * R + 1) / (2 * R)) := by
    rw [dftPow, Real.rpow_add (by positivity), Real.rpow_one, Real.rpow_def_of_pos (by positivity)]
    push_cast
    ring_nf
  have hexp := Real.add_one_le_exp (Real.log (2 * R + 1) / (2 * R))
  have h1 : (2 * (R : ℝ) + 1) * (Real.log (2 * R + 1) / (2 * R) + 1) > 2 * R + 2 := by
    have : (2 * (R : ℝ) + 1)⁻¹ < Real.log (2 * R + 1) / (2 * R) := by
      rw [lt_div_iff₀ (by positivity)]
      have : (2 * (R : ℝ) + 1)⁻¹ * (2 * R) = 1 - (2 * (R : ℝ) + 1)⁻¹ := by field_simp; ring
      linarith
    have h2 : (2 * (R : ℝ) + 1) * ((2 * (R : ℝ) + 1)⁻¹ + 1) = 2 * R + 2 := by field_simp; ring
    nlinarith
  have h3 : 2 * (R : ℝ) + 2 < dftPow R (2 * R) := by
    rw [hP]
    nlinarith
  rw [dftGap, lt_div_iff₀ (by push_cast; linarith)]
  push_cast
  linarith

/-- The main model's `16` bands: `p − q > 1/2` on `33` keys. -/
example := half_lt_dftGap (R := 16) (by norm_num)

/-- **`H` heads with positive weights read fewer than `H` keys** of a context with a key left
over: if coefficients `b` combine the weights `A h` into the indicator of every key of `T`, and
`T` is not every key, then `|T| < H`. -/
theorem card_lt_of_read {κ ι : Type*} [Fintype κ] [Nonempty κ] [Fintype ι] [DecidableEq ι]
    (A : κ → ι → ℝ) (hA : ∀ h j, 0 < A h j) (T : Finset ι) (hT : T ≠ univ)
    (hb : ∀ t ∈ T, ∃ b : κ → ℝ, ∀ j, ∑ h, b h * A h j = if j = t then 1 else 0) :
    T.card < Fintype.card κ := by
  by_contra hle
  replace hle := not_lt.1 hle
  set W := Submodule.span ℝ (Set.range A)
  have hW : finrank ℝ W ≤ Fintype.card κ := finrank_range_le_card (R := ℝ) A
  set e : T → (ι → ℝ) := fun t => Pi.basisFun ℝ ι t
  have he : LinearIndependent ℝ e :=
    (Pi.basisFun ℝ ι).linearIndependent.comp _ Subtype.val_injective
  set U := Submodule.span ℝ (Set.range e)
  have hU : finrank ℝ U = T.card := by rw [finrank_span_eq_card he, Fintype.card_coe]
  have hUW : U ≤ W := by
    rw [Submodule.span_le]
    rintro _ ⟨t, rfl⟩
    obtain ⟨b, hb⟩ := hb t t.2
    have ht : e t = ∑ h, b h • A h := by
      funext j
      simp only [e, Pi.basisFun_apply, Finset.sum_apply, Pi.smul_apply, smul_eq_mul, hb j,
        Pi.single_apply]
    rw [ht]
    exact Submodule.sum_mem _ fun h _ => Submodule.smul_mem _ _ (Submodule.subset_span ⟨h, rfl⟩)
  have hUW' : U = W := Submodule.eq_of_le_of_finrank_le hUW (by omega)
  obtain ⟨j, hj⟩ : ∃ j, j ∉ T := by
    by_contra h
    exact hT (eq_univ_of_forall fun x => by_contra fun hx => h ⟨x, hx⟩)
  have hvan : ∀ f ∈ U, f j = 0 := by
    intro f hf
    induction hf using Submodule.span_induction with
    | mem x hx =>
      obtain ⟨t, rfl⟩ := hx
      simp only [e, Pi.basisFun_apply, Pi.single_apply]
      exact ite_eq_right fun (h : j = (t : ι)) => hj (h ▸ t.2)
    | zero => rfl
    | add x y _ _ hx hy => simp [hx, hy]
    | smul a x _ hx => simp [hx]
  obtain ⟨h0⟩ := ‹Nonempty κ›
  have hA0 : A h0 ∈ U := hUW' ▸ Submodule.subset_span ⟨h0, rfl⟩
  exact (hA h0 j).ne' (hvan _ hA0)

/-- Two heads with positive weights on three keys cannot read two of them: `|T| = 2` is not
`< 2`, so no `b` reads both. -/
example (A : Fin 2 → Fin 3 → ℝ) (hA : ∀ h j, 0 < A h j)
    (hb : ∀ t ∈ ({0, 1} : Finset (Fin 3)), ∃ b : Fin 2 → ℝ,
      ∀ j, ∑ h, b h * A h j = if j = t then 1 else 0) : False := by
  have := card_lt_of_read A hA {0, 1} (by decide) hb
  simp at this

end Transformer.LeRoPEBlock
