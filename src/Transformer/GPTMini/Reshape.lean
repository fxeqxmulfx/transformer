/-
# The coordinate layout between `d_model` and the heads

`reference/model.py` moves between three shapes with `view` and `chunk`:

```python
q, k, v = self.c_attn(x).chunk(3, dim=-1)          # (T, 3·d_model) → 3 × (T, d_model)
q = q.view(T, self.n_heads, self.head_dim)         # (T, d_model)   → (T, n_heads, head_dim)
y = y.transpose(1, 2).contiguous().view(T, C)      # and back
```

Both are relabelings of the coordinate index, so both are `Equiv`s on `Fin`:
`qkvSplit` cuts `Fin (3 · d_model)` into three blocks and `headSplit` reads
`Fin d_model` as a head index paired with a within-head index, in the
row-major order `i ↦ (i / head_dim, i % head_dim)` that `view` uses.

`headSlice` and `headMerge` are the two directions of the reshape on vectors,
and they are mutually inverse — `headSlice_headMerge`, `headMerge_headSlice`.
-/

import Transformer.Basic
import Transformer.GPTMini.Config

namespace Transformer
namespace GPTMini

variable (cfg : Config)

/-- `d_model = n_heads · head_dim`: the head split tiles the model dimension.
This is `head_dim = d_model / n_heads` together with `n_heads ∣ d_model`. -/
theorem d_model_eq (cfg : Config) : cfg.d_model = cfg.n_heads * cfg.head_dim :=
  (Nat.mul_div_cancel' cfg.divides).symm

/-- The cut that `chunk(3, dim=-1)` performs on the output of the fused QKV
projection: `Fin (3 · d_model)` is the query block, then the key block, then
the value block. -/
def qkvSplit (cfg : Config) :
    Fin (3 * cfg.d_model) ≃ (Fin cfg.d_model ⊕ Fin cfg.d_model) ⊕ Fin cfg.d_model :=
  (finCongr (by omega : 3 * cfg.d_model = cfg.d_model + cfg.d_model + cfg.d_model)).trans
    (finSumFinEquiv.symm.trans (Equiv.sumCongr finSumFinEquiv.symm (Equiv.refl _)))

/-- The coordinate of the query block that `chunk` sends to `i`. -/
def qkvQ (cfg : Config) (i : Fin cfg.d_model) : Fin (3 * cfg.d_model) :=
  (qkvSplit cfg).symm (Sum.inl (Sum.inl i))

/-- The coordinate of the key block that `chunk` sends to `i`. -/
def qkvK (cfg : Config) (i : Fin cfg.d_model) : Fin (3 * cfg.d_model) :=
  (qkvSplit cfg).symm (Sum.inl (Sum.inr i))

/-- The coordinate of the value block that `chunk` sends to `i`. -/
def qkvV (cfg : Config) (i : Fin cfg.d_model) : Fin (3 * cfg.d_model) :=
  (qkvSplit cfg).symm (Sum.inr i)

/-- The three blocks are disjoint: no coordinate of `Fin (3 · d_model)` is
read by two of `qkvQ`, `qkvK`, `qkvV`. -/
theorem qkv_disjoint (cfg : Config) (i j : Fin cfg.d_model) :
    qkvQ cfg i ≠ qkvK cfg j ∧ qkvQ cfg i ≠ qkvV cfg j ∧ qkvK cfg i ≠ qkvV cfg j := by
  refine ⟨fun h => ?_, fun h => ?_, fun h => ?_⟩ <;>
    · have := (qkvSplit cfg).symm.injective.eq_iff.mp h
      simp at this

/-- `view(T, n_heads, head_dim)`: the model coordinate `i` is the pair
`(i / head_dim, i % head_dim)`. -/
def headSplit (cfg : Config) : Fin cfg.d_model ≃ Fin cfg.n_heads × Fin cfg.head_dim :=
  (finCongr (d_model_eq cfg)).trans finProdFinEquiv.symm

/-- The block of `u : ℝ^{3 d_model}` that the index map `e` selects.
`qkvSlice cfg (qkvQ cfg) u` is the `q` of `chunk(3, dim=-1)`. -/
noncomputable def qkvSlice (cfg : Config)
    (e : Fin cfg.d_model → Fin (3 * cfg.d_model)) (u : EucSpace (3 * cfg.d_model)) :
    EucSpace cfg.d_model :=
  (EuclideanSpace.equiv (Fin cfg.d_model) ℝ).symm (fun i => u (e i))

@[simp] theorem qkvSlice_apply (cfg : Config)
    (e : Fin cfg.d_model → Fin (3 * cfg.d_model)) (u : EucSpace (3 * cfg.d_model))
    (i : Fin cfg.d_model) :
    qkvSlice cfg e u i = u (e i) := rfl

/-- The `head_dim` coordinates of head `h` inside a `d_model` vector. -/
noncomputable def headSlice (cfg : Config) (u : EucSpace cfg.d_model)
    (h : Fin cfg.n_heads) : EucSpace cfg.head_dim :=
  (EuclideanSpace.equiv (Fin cfg.head_dim) ℝ).symm
    (fun c => u ((headSplit cfg).symm (h, c)))

@[simp] theorem headSlice_apply (cfg : Config) (u : EucSpace cfg.d_model)
    (h : Fin cfg.n_heads) (c : Fin cfg.head_dim) :
    headSlice cfg u h c = u ((headSplit cfg).symm (h, c)) := rfl

/-- The inverse reshape: `n_heads` head-dimensional vectors concatenated back
into one `d_model` vector, in the same order. -/
noncomputable def headMerge (cfg : Config)
    (y : Fin cfg.n_heads → EucSpace cfg.head_dim) : EucSpace cfg.d_model :=
  (EuclideanSpace.equiv (Fin cfg.d_model) ℝ).symm
    (fun i => y (headSplit cfg i).1 (headSplit cfg i).2)

@[simp] theorem headMerge_apply (cfg : Config)
    (y : Fin cfg.n_heads → EucSpace cfg.head_dim) (i : Fin cfg.d_model) :
    headMerge cfg y i = y (headSplit cfg i).1 (headSplit cfg i).2 := rfl

/-- Slicing the merge of `y` at head `h` returns `y h`. -/
@[simp] theorem headSlice_headMerge (cfg : Config)
    (y : Fin cfg.n_heads → EucSpace cfg.head_dim) (h : Fin cfg.n_heads) :
    headSlice cfg (headMerge cfg y) h = y h := by
  ext c
  simp

/-- Merging the slices of `u` returns `u`: no coordinate is lost or duplicated
by the reshape. -/
@[simp] theorem headMerge_headSlice (cfg : Config) (u : EucSpace cfg.d_model) :
    headMerge cfg (headSlice cfg u) = u := by
  ext i
  simp

/-! ### Norms across the reshape

The three reshapes read each coordinate at most once, so none of them can
increase the Euclidean norm, and `headMerge` preserves the total energy
exactly.  These are the bounds the sub-layer growth estimates rest on. -/

/-- Reading coordinates along an injective index map cannot increase the
Euclidean norm: the selected coordinates are a sub-family of the original. -/
theorem norm_comp_injective_le {m n : ℕ} (e : Fin m → Fin n)
    (he : Function.Injective e) (u : EucSpace n) :
    ‖(EuclideanSpace.equiv (Fin m) ℝ).symm (fun i => u (e i))‖ ≤ ‖u‖ := by
  rw [EuclideanSpace.norm_eq, EuclideanSpace.norm_eq]
  refine Real.sqrt_le_sqrt ?_
  classical
  rw [show (∑ i : Fin m, ‖((EuclideanSpace.equiv (Fin m) ℝ).symm (fun i => u (e i))) i‖ ^ 2)
      = ∑ i : Fin m, ‖u (e i)‖ ^ 2 from rfl,
    ← Finset.sum_image (f := fun j : Fin n => ‖u j‖ ^ 2)
      (fun i _ j _ h => he h)]
  exact Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ _)
    fun _ _ _ => by positivity

/-- The `q`/`k`/`v` cut selects a block of coordinates, so it is norm-decreasing. -/
theorem qkvSlice_norm_le (cfg : Config)
    (e : Fin cfg.d_model → Fin (3 * cfg.d_model)) (he : Function.Injective e)
    (u : EucSpace (3 * cfg.d_model)) :
    ‖qkvSlice cfg e u‖ ≤ ‖u‖ :=
  norm_comp_injective_le e he u

/-- `qkvV` is injective, being `Sum.inr` followed by an equivalence. -/
theorem qkvV_injective (cfg : Config) : Function.Injective (qkvV cfg) := by
  intro i j h
  simpa using (qkvSplit cfg).symm.injective h

/-- `qkvQ` is injective, being two `Sum.inl`s followed by an equivalence. -/
theorem qkvQ_injective (cfg : Config) : Function.Injective (qkvQ cfg) := by
  intro i j h
  simpa using (qkvSplit cfg).symm.injective h

/-- `qkvK` is injective, being `Sum.inl ∘ Sum.inr` followed by an equivalence. -/
theorem qkvK_injective (cfg : Config) : Function.Injective (qkvK cfg) := by
  intro i j h
  simpa using (qkvSplit cfg).symm.injective h

/-- A head reads `head_dim` distinct coordinates, so it is norm-decreasing. -/
theorem headSlice_norm_le (cfg : Config) (u : EucSpace cfg.d_model)
    (h : Fin cfg.n_heads) :
    ‖headSlice cfg u h‖ ≤ ‖u‖ :=
  norm_comp_injective_le _
    (fun c₁ c₂ hc => by simpa using (headSplit cfg).symm.injective hc) u

/-- `headMerge` is an isometry onto its image: the merged vector carries
exactly the energy of its heads. -/
theorem headMerge_norm_sq (cfg : Config)
    (y : Fin cfg.n_heads → EucSpace cfg.head_dim) :
    ‖headMerge cfg y‖ ^ 2 = ∑ h, ‖y h‖ ^ 2 := by
  rw [EuclideanSpace.norm_eq, Real.sq_sqrt (by positivity)]
  rw [Fintype.sum_equiv (headSplit cfg)
      (fun i => ‖headMerge cfg y i‖ ^ 2)
      (fun p : Fin cfg.n_heads × Fin cfg.head_dim => ‖y p.1 p.2‖ ^ 2)
      (fun i => by rw [headMerge_apply]),
    Fintype.sum_prod_type]
  exact Finset.sum_congr rfl fun h _ => by
    rw [EuclideanSpace.norm_eq, Real.sq_sqrt (by positivity)]

/-- A uniform bound on the heads bounds the merged vector by `√n_heads`. -/
theorem headMerge_norm_le (cfg : Config)
    (y : Fin cfg.n_heads → EucSpace cfg.head_dim) (B : ℝ)
    (hB : ∀ h, ‖y h‖ ≤ B) :
    ‖headMerge cfg y‖ ≤ Real.sqrt (cfg.n_heads : ℝ) * B := by
  have hB0 : 0 ≤ B := le_trans (norm_nonneg _) (hB ⟨0, cfg.n_heads_pos⟩)
  have hsq : ‖headMerge cfg y‖ ^ 2 ≤ (cfg.n_heads : ℝ) * B ^ 2 := by
    rw [headMerge_norm_sq]
    calc ∑ h, ‖y h‖ ^ 2 ≤ ∑ _h : Fin cfg.n_heads, B ^ 2 :=
          Finset.sum_le_sum fun h _ => by gcongr; exact hB h
      _ = (cfg.n_heads : ℝ) * B ^ 2 := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  have := Real.sqrt_le_sqrt hsq
  rwa [Real.sqrt_sq (norm_nonneg _), Real.sqrt_mul (Nat.cast_nonneg _),
    Real.sqrt_sq hB0] at this

end GPTMini
end Transformer
