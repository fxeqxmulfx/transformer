module

public import Transformer.SequenceModel.Attention.Expressivity.XSA

/-!
# The construction uses the actual fused-QKV and head layouts

Source: `review/flat_spectrum.md`, *Transformer learning bridges*, E5.
A one-head model has an isometric reshape between its head and residual
coordinates. Packing values in the fused QKV matrix leaves Q/K identically
zero and preserves the complete value map. No forward operator is changed.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Expressivity

open GPTMini
open scoped BigOperators

/-- Merge a single head using the source layout, proved isometric. -/
def merge (cfg : Config) (h1 : cfg.n_heads = 1) :
    EucSpace cfg.head_dim →ₗᵢ[ℝ] EucSpace cfg.d_model where
  toFun x := headMerge cfg (fun _ => x)
  map_add' x y := by ext i; simp [headMerge_apply]
  map_smul' c x := by ext i; simp [headMerge_apply]
  norm_map' x := by
    change ‖headMerge cfg (fun _ => x)‖ = ‖x‖
    have h := headMerge_norm_sq cfg (fun _ => x)
    simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, h1, one_nsmul] at h
    nlinarith [norm_nonneg (headMerge cfg (fun _ => x)), norm_nonneg x]

/-- Select the source's first head, as an ordinary linear matrix. -/
def slice (cfg : Config) : EucSpace cfg.d_model →L[ℝ] EucSpace cfg.head_dim :=
  LinearMap.toContinuousLinearMap {
    toFun := fun x => headSlice cfg x ⟨0, cfg.n_heads_pos⟩
    map_add' := by intro x y; ext i; rfl
    map_smul' := by intro c x; ext i; rfl }

/-- E5: every head sees the constructed vector through the source reshape. -/
theorem headSlice_merge (cfg : Config) (h1 : cfg.n_heads = 1)
    (x : EucSpace cfg.head_dim) (h : Fin cfg.n_heads) :
    headSlice cfg (merge cfg h1 x) h = x := by
  exact headSlice_headMerge cfg (fun _ => x) h

/-- E5: the first-head map exactly recovers the constructed coordinates. -/
theorem slice_merge (cfg : Config) (h1 : cfg.n_heads = 1) (x : EucSpace cfg.head_dim) :
    slice cfg (merge cfg h1 x) = x := headSlice_merge cfg h1 x _

/-- Pack zero Q/K and a full value vector in the source's fused layout. -/
def packValue (cfg : Config) : EucSpace cfg.d_model →L[ℝ] EucSpace (3 * cfg.d_model) :=
  LinearMap.toContinuousLinearMap {
    toFun := fun x => WithLp.toLp 2 fun i => match qkvSplit cfg i with
      | .inl _ => 0 | .inr c => x c
    map_add' := by
      intro x y; ext i
      cases h : qkvSplit cfg i <;> simp [h]
    map_smul' := by
      intro c x; ext i
      cases h : qkvSplit cfg i <;> simp [h] }

/-- E5: the actual query slice of the packed linear map is zero. -/
theorem qkvQ_pack (cfg : Config) (x : EucSpace cfg.d_model) :
    qkvSlice cfg (qkvQ cfg) (packValue cfg x) = 0 := by
  ext i
  change (match qkvSplit cfg ((qkvSplit cfg).symm (.inl (.inl i))) with
    | .inl _ => (0 : ℝ) | .inr c => x c) = 0
  rw [Equiv.apply_symm_apply]

/-- E5: the actual key slice of the packed linear map is zero. -/
theorem qkvK_pack (cfg : Config) (x : EucSpace cfg.d_model) :
    qkvSlice cfg (qkvK cfg) (packValue cfg x) = 0 := by
  ext i
  change (match qkvSplit cfg ((qkvSplit cfg).symm (.inl (.inr i))) with
    | .inl _ => (0 : ℝ) | .inr c => x c) = 0
  rw [Equiv.apply_symm_apply]

/-- E5: the actual value slice is exactly the requested full vector. -/
theorem qkvV_pack (cfg : Config) (x : EucSpace cfg.d_model) :
    qkvSlice cfg (qkvV cfg) (packValue cfg x) = x := by
  ext i
  change (match qkvSplit cfg ((qkvSplit cfg).symm (.inr i)) with
    | .inl _ => (0 : ℝ) | .inr c => x c) = x i
  rw [Equiv.apply_symm_apply]

/-- Copy the token bank into the disjoint context bank by a fixed matrix. -/
def shift {p d : ℕ} (e : (Fin p ⊕ Fin p) ↪ Fin d) : EucSpace d →L[ℝ] EucSpace d :=
  LinearMap.toContinuousLinearMap {
    toFun := fun x => ∑ a, x (e (.inl a)) • unit (e (.inr a))
    map_add' := by intro x y; simp only [PiLp.add_apply, add_smul, Finset.sum_add_distrib]
    map_smul' := by intro c x; simp only [PiLp.smul_apply, smul_eq_mul, mul_smul,
      RingHom.id_apply, Finset.smul_sum] }

/-- E5: the context-copy matrix sends every token basis vector to its workspace. -/
theorem shift_unit {p d : ℕ} (e : (Fin p ⊕ Fin p) ↪ Fin d) (a : Fin p) :
    shift e (unit (e (.inl a))) = unit (e (.inr a)) := by
  change (∑ b, unit (e (.inl a)) (e (.inl b)) • unit (e (.inr b))) = _
  simp [unit, EuclideanSpace.single, e.injective.eq_iff, ite_smul]

example := headSlice_merge witnessConfig (by rfl) (WithLp.toLp 2 ![1, 2, 3, 4]) 0
example := slice_merge witnessConfig (by rfl) (WithLp.toLp 2 ![1, 2, 3, 4])

end Transformer.SequenceModel.Attention.Expressivity
