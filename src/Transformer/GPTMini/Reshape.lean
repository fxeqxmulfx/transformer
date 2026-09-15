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

end GPTMini
end Transformer
