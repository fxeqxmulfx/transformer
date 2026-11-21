module

public import Mathlib.FieldTheory.Finite.Basic
public import Mathlib.LinearAlgebra.FiniteDimensional.Lemmas
public import Mathlib.LinearAlgebra.Dimension.Constructions
public import Mathlib.Data.Matrix.Mul
public import Transformer.Grokking.FlatSpectrum.TablePart.Threshold

/-!
# PT3: the flat class and its dimension

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT3. Over a field `F`,
with `U = F^k` and `V = F^l`, the flat class `H` (`flatClass`) is the space of the functions
`g(u) + h(v) + uᵀ M v`, for arbitrary `g`, `h` and a matrix `M`. The source takes `F = F_p`, and
nothing here needs a prime field. The mixed differences of a member are `(u − u')ᵀ M (v − v')`
(`flatEval_mixed`), so a zero member has `M = 0`, and then `g = t`, `h = −t` for a constant `t`:

  `dim H = q^k + q^l − 1 + k l` (`finrank_flatClass`).

`flatEval` writes the same class through one potential `x` on the rows and the columns,
`x(u) − x(v) + uᵀ M v` (`range_flatEval`), the form that the count of `FlatCount` uses.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.TablePart

open Matrix Module

variable (F : Type*) [Field F] (ι κ : Type*) [Fintype ι] [Fintype κ]

/-- The parametrization of the flat class (`review/flat_spectrum.md`, PT3):
`(g, h, M) ↦ g(u) + h(v) + uᵀ M v`. -/
noncomputable def flatParam :
    (((ι → F) → F) × ((κ → F) → F) × Matrix ι κ F) →ₗ[F] ((ι → F) × (κ → F) → F) where
  toFun p c := p.1 c.1 + p.2.1 c.2 + c.1 ⬝ᵥ p.2.2 *ᵥ c.2
  map_add' p p' := by
    funext c
    simp only [Prod.fst_add, Prod.snd_add, Pi.add_apply, add_mulVec, dotProduct_add]
    ring
  map_smul' t p := by
    funext c
    simp only [Prod.smul_fst, Prod.smul_snd, Pi.smul_apply, smul_eq_mul, RingHom.id_apply,
      smul_mulVec, dotProduct_smul]
    ring

/-- The flat class `H` (`review/flat_spectrum.md`, PT3): the functions
`g(u) + h(v) + uᵀ M v` on `F^k × F^l`. -/
noncomputable def flatClass : Submodule F ((ι → F) × (κ → F) → F) :=
  LinearMap.range (flatParam F ι κ)

/-- The flat class through one potential on the rows and the columns:
`(x, M) ↦ x(u) − x(v) + uᵀ M v`. -/
noncomputable def flatEval :
    (((ι → F) ⊕ (κ → F) → F) × Matrix ι κ F) →ₗ[F] ((ι → F) × (κ → F) → F) where
  toFun p c := p.1 (.inl c.1) - p.1 (.inr c.2) + c.1 ⬝ᵥ p.2 *ᵥ c.2
  map_add' p p' := by
    funext c
    simp only [Prod.fst_add, Prod.snd_add, Pi.add_apply, add_mulVec, dotProduct_add]
    ring
  map_smul' t p := by
    funext c
    simp only [Prod.smul_fst, Prod.smul_snd, Pi.smul_apply, smul_eq_mul, RingHom.id_apply,
      smul_mulVec, dotProduct_smul]
    ring

variable {F ι κ}

/-- A member of `H` is `g(u) + h(v) + uᵀ M v`. -/
lemma mem_flatClass {f : (ι → F) × (κ → F) → F} :
    f ∈ flatClass F ι κ ↔ ∃ (g : (ι → F) → F) (h : (κ → F) → F) (M : Matrix ι κ F),
      ∀ u v, f (u, v) = g u + h v + u ⬝ᵥ M *ᵥ v := by
  constructor
  · rintro ⟨⟨g, h, M⟩, rfl⟩
    exact ⟨g, h, M, fun _ _ => rfl⟩
  · rintro ⟨g, h, M, hf⟩
    exact ⟨(g, h, M), funext fun c => (hf c.1 c.2).symm⟩

example : (fun c : (Fin 1 → ℚ) × (Fin 1 → ℚ) => c.1 0 * c.2 0) ∈ flatClass ℚ (Fin 1) (Fin 1) :=
  mem_flatClass.2 ⟨0, 0, 1, fun u v => by simp [dotProduct, mulVec]⟩

@[simp] lemma flatEval_apply (p : ((ι → F) ⊕ (κ → F) → F) × Matrix ι κ F) (u : ι → F)
    (v : κ → F) : flatEval F ι κ p (u, v) = p.1 (.inl u) - p.1 (.inr v) + u ⬝ᵥ p.2 *ᵥ v := rfl

/-- The two parametrizations have one range. -/
lemma range_flatEval : LinearMap.range (flatEval F ι κ) = flatClass F ι κ := by
  ext f
  rw [mem_flatClass]
  constructor
  · rintro ⟨⟨x, M⟩, rfl⟩
    exact ⟨fun u => x (.inl u), fun v => -x (.inr v), M, fun u v => by simp [sub_eq_add_neg]⟩
  · rintro ⟨g, h, M, hf⟩
    refine ⟨(Sum.elim g fun v => -h v, M), funext fun c => ?_⟩
    obtain ⟨u, v⟩ := c
    simp [hf]

/-- The mixed difference of a member (`review/flat_spectrum.md`, PT3): `g` and `h` cancel. -/
lemma flatEval_mixed (p : ((ι → F) ⊕ (κ → F) → F) × Matrix ι κ F) (u u' : ι → F) (v v' : κ → F) :
    flatEval F ι κ p (u, v) - flatEval F ι κ p (u', v) - flatEval F ι κ p (u, v') +
      flatEval F ι κ p (u', v') = (u - u') ⬝ᵥ p.2 *ᵥ (v - v') := by
  simp only [flatEval_apply, mulVec_sub, dotProduct_sub, sub_dotProduct]
  ring

/-- `flatEval` is zero exactly at `M = 0` with a constant potential. -/
lemma flatEval_eq_zero_iff [DecidableEq ι] [DecidableEq κ]
    (p : ((ι → F) ⊕ (κ → F) → F) × Matrix ι κ F) :
    flatEval F ι κ p = 0 ↔ p.2 = 0 ∧ ∀ a, p.1 a = p.1 (.inl 0) := by
  constructor
  · intro h
    have hc (u : ι → F) (v : κ → F) : flatEval F ι κ p (u, v) = 0 := congrFun h (u, v)
    have hM : p.2 = 0 := by
      ext i j
      have hmix := flatEval_mixed p (Pi.single i 1) 0 (Pi.single j 1) 0
      rw [hc, hc, hc, hc] at hmix
      simpa [mulVec_single_one] using hmix.symm
    refine ⟨hM, ?_⟩
    have hx (u : ι → F) (v : κ → F) : p.1 (.inl u) = p.1 (.inr v) := by
      have hcuv := hc u v
      rw [flatEval_apply, hM] at hcuv
      simpa [sub_eq_zero] using hcuv
    rintro (u | v)
    · rw [hx u 0, ← hx 0 0]
    · exact (hx 0 v).symm
  · rintro ⟨hM, hx⟩
    funext ⟨u, v⟩
    simp [hM, hx (.inl u), hx (.inr v)]

/-- The kernel of `flatEval` is the line of the constants `(t, 0)`, the kernel `(t, −t, 0)` of the
parametrization in the source. -/
lemma ker_flatEval [DecidableEq ι] [DecidableEq κ] :
    LinearMap.ker (flatEval F ι κ) =
      F ∙ ((fun _ => 1, 0) : ((ι → F) ⊕ (κ → F) → F) × Matrix ι κ F) := by
  ext p
  rw [LinearMap.mem_ker, flatEval_eq_zero_iff, Submodule.mem_span_singleton]
  constructor
  · rintro ⟨hM, hx⟩
    refine ⟨p.1 (.inl 0), Prod.ext (funext fun a => ?_) ?_⟩
    · simp [hx a]
    · simp [hM]
  · rintro ⟨t, rfl⟩
    simp

/-- The kernel of `flatEval` is one line. -/
lemma finrank_ker_flatEval [DecidableEq ι] [DecidableEq κ] :
    finrank F (LinearMap.ker (flatEval F ι κ)) = 1 := by
  rw [ker_flatEval]
  refine finrank_span_singleton fun h => one_ne_zero (α := F) ?_
  exact congrFun (congrArg Prod.fst h) (.inl 0)

/-- **The dimension of the flat class** (`review/flat_spectrum.md`, PT3):
`dim H = q^k + q^l − 1 + k l`, written without subtraction. -/
theorem finrank_flatClass [Fintype F] [DecidableEq ι] [DecidableEq κ] :
    finrank F (flatClass F ι κ) + 1 =
      Fintype.card F ^ Fintype.card ι + Fintype.card F ^ Fintype.card κ +
        Fintype.card ι * Fintype.card κ := by
  have h := LinearMap.finrank_range_add_finrank_ker (flatEval F ι κ)
  rw [range_flatEval, finrank_ker_flatEval, finrank_prod, finrank_fintype_fun_eq_card,
    finrank_matrix, Fintype.card_sum, Fintype.card_fun, Fintype.card_fun] at h
  simpa using h

example : finrank (ZMod 2) (flatClass (ZMod 2) (Fin 1) (Fin 1)) + 1 = 2 ^ 1 + 2 ^ 1 + 1 * 1 := by
  simpa [ZMod.card] using finrank_flatClass (F := ZMod 2) (ι := Fin 1) (κ := Fin 1)

end Transformer.Grokking.FlatSpectrum.TablePart
