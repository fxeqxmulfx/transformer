import Transformer.GPTMini.Semantics.DepthEmbedding
import Transformer.GPTMini.Reshape

/-!
# Two simultaneous uniform depth heads in the actual fused QKV

Source: GPTMini CausalMHA.forward at f11b6e2, with the unchanged
64-to-192 or 128-to-384 bias-free fused matrix and four-head reshape.
Both query and key chunks are zero. Two ordinary value rows read
specified residual coordinates into the first coordinates of heads
zero and one. The other two heads have zero values.

The source coordinates are shared parameters of the block, not a
function of a prefix or a desired ordered occurrence. This module
evaluates the complete actual Q/K/V chunk and head view at arbitrary
real residual vectors. Normalization, true uniform-head bounds,
output projection and the feature recurrence are coupled separately.

No rotary or QKNorm operator is removed: their original zero-input
behavior supplies uniform scores in the subsequent genuine heads.
This gives ordinary baseline parameters, not a convex architecture.
-/

namespace Transformer.GPTMini.Semantics

open Transformer.Basis

/-- The detector branches use heads zero and one of the existing four-head model.
Source: the original head count, without adding a head. -/
def depthHead (mode : Mode) (branch : Fin 2) : Fin (depthConfig mode).n_heads :=
  ⟨branch.val, by have hb := branch.isLt; change branch.val < 4; omega⟩

/-- The first actual value coordinate exists in either original head width.
Source: the positive head dimension of the unchanged well-formed configuration. -/
def depthHeadZero (mode : Mode) : Fin (depthConfig mode).head_dim :=
  ⟨0, (depthConfig mode).head_dim_pos⟩

/-- A genuine unit value direction in the actual head space.
Source: the original V head's first coordinate; values themselves are not rotated. -/
noncomputable def depthHeadDirection (mode : Mode) : EucSpace (depthConfig mode).head_dim :=
  EuclideanSpace.single (depthHeadZero mode) 1

/-- The actual unit head direction has norm one in both widths.
Source: the original L2 head norm and single-coordinate insertion. -/
theorem depthHeadDirection_norm (mode : Mode) : ‖depthHeadDirection mode‖ = 1 := by
  simp only [depthHeadDirection, PiLp.norm_single, Real.norm_eq_abs, abs_one]

/-- Each assigned value row is the original chunk/view inverse coordinate, not a new reshape.
Source: GPTMini.Reshape.qkvV and headSplit from the real fused matrix. -/
noncomputable def depthValueIndex (mode : Mode) (branch : Fin 2) : Fin (3 * (depthConfig mode).d_model) :=
  qkvV (depthConfig mode) ((headSplit (depthConfig mode)).symm (depthHead mode branch, depthHeadZero mode))

/-- A genuine V slice reads this assigned row exactly at its own head and first coordinate.
Source: injectivity of the original chunk and head reshape, retaining their actual indices. -/
theorem depthValueIndex_pair_eq (mode : Mode) (head : Fin (depthConfig mode).n_heads)
    (c : Fin (depthConfig mode).head_dim) (branch : Fin 2) :
    qkvV (depthConfig mode) ((headSplit (depthConfig mode)).symm (head, c)) = depthValueIndex mode branch ↔
      head.val = branch.val ∧ c.val = 0 := by
  unfold depthValueIndex
  rw [(qkvV_injective (depthConfig mode)).eq_iff,
    (headSplit (depthConfig mode)).symm.injective.eq_iff]
  constructor
  · intro h
    have hh := congrArg Prod.fst h
    have hc := congrArg Prod.snd h
    exact ⟨congrArg Fin.val hh, congrArg Fin.val hc⟩
  · rintro ⟨hh, hc⟩
    have hhead : head = depthHead mode branch := Fin.ext hh
    have hcoord : c = depthHeadZero mode := Fin.ext hc
    rw [hhead, hcoord]

/-- Every ordinary fused-matrix row is either one of the two real coordinate projections or zero.
Source: simultaneous assignment of two V rows; all Q/K and other V rows stay zero. -/
noncomputable def depthValueRow (mode : Mode) (source : Fin 2 → Fin (depthConfig mode).d_model)
    (row : Fin (3 * (depthConfig mode).d_model)) : EucSpace (depthConfig mode).d_model →L[ℝ] ℝ :=
  if row = depthValueIndex mode 0 then EuclideanSpace.proj (source 0)
  else if row = depthValueIndex mode 1 then EuclideanSpace.proj (source 1) else 0

/-- A single genuine bias-free linear matrix assembles both assigned V projections simultaneously.
Source: the original W_qkv type and ordinary pi assembly of its real linear rows. -/
noncomputable def depthValueQKV (mode : Mode) (source : Fin 2 → Fin (depthConfig mode).d_model) :
    EucSpace (depthConfig mode).d_model →L[ℝ] EucSpace (3 * (depthConfig mode).d_model) :=
  (EuclideanSpace.equiv (Fin (3 * (depthConfig mode).d_model)) ℝ).symm.toContinuousLinearMap.comp
    (ContinuousLinearMap.pi (depthValueRow mode source))

/-- Each true matrix coordinate is its declared ordinary row applied to the same residual vector.
Source: the complete pi matrix and original Euclidean coordinate equivalence. -/
theorem depthValueQKV_coordinate (mode : Mode) (source : Fin 2 → Fin (depthConfig mode).d_model)
    (x : EucSpace (depthConfig mode).d_model) (row : Fin (3 * (depthConfig mode).d_model)) :
    depthValueQKV mode source x row = depthValueRow mode source row x := by
  rfl

/-- Every real head has zero query and key slices in this single simultaneous projection matrix.
Source: the actual chunk disjointness, so assigned V rows cannot leak into either Q or K. -/
theorem depthValueQKV_qk (mode : Mode) (source : Fin 2 → Fin (depthConfig mode).d_model)
    (x : EucSpace (depthConfig mode).d_model) (head : Fin (depthConfig mode).n_heads) :
    headSlice (depthConfig mode) (qkvSlice (depthConfig mode) (qkvQ (depthConfig mode))
      (depthValueQKV mode source x)) head = 0 ∧
    headSlice (depthConfig mode) (qkvSlice (depthConfig mode) (qkvK (depthConfig mode))
      (depthValueQKV mode source x)) head = 0 := by
  constructor <;> ext c
  · change depthValueQKV mode source x
      (qkvQ (depthConfig mode) ((headSplit (depthConfig mode)).symm (head, c))) = 0
    have h0 := (qkv_disjoint (depthConfig mode) ((headSplit (depthConfig mode)).symm (head, c))
      ((headSplit (depthConfig mode)).symm (depthHead mode 0, depthHeadZero mode))).2.1
    have h1 := (qkv_disjoint (depthConfig mode) ((headSplit (depthConfig mode)).symm (head, c))
      ((headSplit (depthConfig mode)).symm (depthHead mode 1, depthHeadZero mode))).2.1
    rw [depthValueQKV_coordinate, depthValueRow, depthValueIndex, ite_eq_right h0, depthValueIndex, ite_eq_right h1]
    rfl
  · change depthValueQKV mode source x
      (qkvK (depthConfig mode) ((headSplit (depthConfig mode)).symm (head, c))) = 0
    have h0 := (qkv_disjoint (depthConfig mode) ((headSplit (depthConfig mode)).symm (head, c))
      ((headSplit (depthConfig mode)).symm (depthHead mode 0, depthHeadZero mode))).2.2
    have h1 := (qkv_disjoint (depthConfig mode) ((headSplit (depthConfig mode)).symm (head, c))
      ((headSplit (depthConfig mode)).symm (depthHead mode 1, depthHeadZero mode))).2.2
    rw [depthValueQKV_coordinate, depthValueRow, depthValueIndex, ite_eq_right h0, depthValueIndex, ite_eq_right h1]
    rfl

/-- The two genuine V slices read their own selected residual scalars; the other heads are exactly zero.
Source: evaluated full matrix, real chunk/view indices and a true one-coordinate value direction. -/
theorem depthValueQKV_value (mode : Mode) (source : Fin 2 → Fin (depthConfig mode).d_model)
    (x : EucSpace (depthConfig mode).d_model) (head : Fin (depthConfig mode).n_heads) :
    headSlice (depthConfig mode) (qkvSlice (depthConfig mode) (qkvV (depthConfig mode))
      (depthValueQKV mode source x)) head =
      if head.val = 0 then x (source 0) • depthHeadDirection mode
      else if head.val = 1 then x (source 1) • depthHeadDirection mode else 0 := by
  ext c
  change depthValueQKV mode source x
    (qkvV (depthConfig mode) ((headSplit (depthConfig mode)).symm (head, c))) =
    (if head.val = 0 then x (source 0) • depthHeadDirection mode
      else if head.val = 1 then x (source 1) • depthHeadDirection mode else 0) c
  rw [depthValueQKV_coordinate]
  have he0 := depthValueIndex_pair_eq mode head c 0
  have he1 := depthValueIndex_pair_eq mode head c 1
  simp only [depthValueRow, he0, he1]
  have hc : c = depthHeadZero mode ↔ c.val = 0 := by rw [Fin.ext_iff]; rfl
  by_cases h0 : head.val = 0 <;> by_cases h1 : head.val = 1 <;> by_cases hz : c.val = 0 <;>
    simp [h0, h1, hz, depthHeadDirection, PiLp.smul_apply, PiLp.single_apply, hc, EuclideanSpace.proj, PiLp.proj_apply]

/-- Each actual active head reads exactly its own branch's scalar without a conditional supplied to the head input.
Source: both assigned V slices of the same simultaneous fused matrix. -/
theorem depthValueQKV_branch (mode : Mode) (source : Fin 2 → Fin (depthConfig mode).d_model)
    (x : EucSpace (depthConfig mode).d_model) (branch : Fin 2) :
    headSlice (depthConfig mode) (qkvSlice (depthConfig mode) (qkvV (depthConfig mode))
      (depthValueQKV mode source x)) (depthHead mode branch) = x (source branch) • depthHeadDirection mode := by
  rw [depthValueQKV_value mode source x (depthHead mode branch)]
  fin_cases branch <;> norm_num [depthHead] <;> rfl

/-- The remaining original heads carry no value in the two-branch depth construction.
Source: their genuine fused V rows are zero, with no removal of a head from the model. -/
theorem depthValueQKV_unused (mode : Mode) (source : Fin 2 → Fin (depthConfig mode).d_model)
    (x : EucSpace (depthConfig mode).d_model) (head : Fin (depthConfig mode).n_heads) (hunused : 2 ≤ head.val) :
    headSlice (depthConfig mode) (qkvSlice (depthConfig mode) (qkvV (depthConfig mode))
      (depthValueQKV mode source x)) head = 0 := by
  have h0 : head.val ≠ 0 := by omega
  have h1 : head.val ≠ 1 := by omega
  rw [depthValueQKV_value, ite_eq_right h0, ite_eq_right h1]

example : 2 ≤ (2 : Fin 4).val := by decide

end Transformer.GPTMini.Semantics
