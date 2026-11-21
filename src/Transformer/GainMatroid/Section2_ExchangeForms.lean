module

public import Mathlib.Tactic.LinearCombination
public import Transformer.GainMatroid.Section2_ExchangeMatrices

/-!
# Bringing a triangle of a matrix to the columns of `I(K₄)`

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4: "Given a matrix `A′` such that `M(A′)` contains
a triangle `X`, `A′` is projectively equivalent to a matrix `A` in which the columns corresponding to
`X` are as the last three columns of `I(K₄)`, perhaps with one row omitted or with zero rows added."
`exists_isTriangleForm` proves this, with the fourth row of `I(K₄)` omitted.

A relation among columns is a relation on the row space (`sum_smul_col_eq_zero_iff`). So projectively
equivalent matrices have the same independent sets of columns (`ProjEquiv.linearIndepOn_iff`) and
represent the same matroid (`IsMatrixRepresentation.of_projEquiv`). On an independent set of columns
the row space takes every value (`exists_mem_rowSpace_eq_on`).
-/

@[expose] public section

namespace Transformer.GainMatroid

variable {R R' E F : Type*} [Field F]

/-- A linear relation among the columns of `A` is a linear relation on its row space. -/
theorem sum_smul_col_eq_zero_iff (A : Matrix R E F) (s : Finset E) (f : E → F) :
    ∑ e ∈ s, f e • (fun r => A r e) = 0 ↔ ∀ w ∈ rowSpace A, ∑ e ∈ s, f e * w e = 0 := by
  constructor
  · intro h w hw
    have hle : rowSpace A ≤
        LinearMap.ker (∑ e ∈ s, f e • (LinearMap.proj e : (E → F) →ₗ[F] F)) := by
      refine Submodule.span_le.2 ?_
      rintro _ ⟨r, rfl⟩
      have := congrFun h r
      simpa [Finset.sum_apply, LinearMap.sum_apply, Matrix.row] using this
    simpa [LinearMap.sum_apply] using hle hw
  · intro h
    funext r
    simpa [Finset.sum_apply] using h _ (row_mem_rowSpace A r)

private theorem linearIndepOn_of_projEquiv [Finite E] {A : Matrix R E F} {B : Matrix R' E F}
    (h : ProjEquiv A B) {I : Set E} (hA : LinearIndepOn F (fun e r => A r e) I) :
    LinearIndepOn F (fun e r => B r e) I := by
  obtain ⟨s, rfl⟩ := I.toFinite.exists_finset_coe
  obtain ⟨t, ht⟩ := h
  rw [linearIndepOn_finset_iff] at hA ⊢
  intro g hg e he
  have hrow := (sum_smul_col_eq_zero_iff B s g).1 hg
  have hscale := (sum_smul_col_eq_zero_iff (scaleCols t A) s g).2 (by rw [ht]; exact hrow)
  have hrel : ∑ i ∈ s, (g i * t i) • (fun r => A r i) = 0 := by
    funext r
    have := congrFun hscale r
    simp only [Finset.sum_apply, Pi.smul_apply, smul_eq_mul, Pi.zero_apply, scaleCols,
      Matrix.of_apply] at this ⊢
    rw [← this]
    exact Finset.sum_congr rfl fun i _ => by ring
  exact (mul_eq_zero.1 (hA _ hrel e he)).resolve_right (t e).ne_zero

/-- Projectively equivalent matrices have the same independent sets of columns. -/
theorem ProjEquiv.linearIndepOn_iff [Finite E] {A : Matrix R E F} {B : Matrix R' E F}
    (h : ProjEquiv A B) (I : Set E) :
    LinearIndepOn F (fun e r => A r e) I ↔ LinearIndepOn F (fun e r => B r e) I :=
  ⟨linearIndepOn_of_projEquiv h, linearIndepOn_of_projEquiv h.symm⟩

/-- A matrix projectively equivalent to a representation of `M` represents `M`. -/
theorem IsMatrixRepresentation.of_projEquiv [Finite E] {M : Matroid E} {A : Matrix R E F}
    {B : Matrix R' E F} (hA : IsMatrixRepresentation M A) (h : ProjEquiv A B) :
    IsMatrixRepresentation M B :=
  ⟨hA.1, fun I => (hA.2 I).trans (h.linearIndepOn_iff I)⟩

/-- On an independent set of columns, the row space takes every value. -/
theorem exists_mem_rowSpace_eq_on {A : Matrix R E F} {s : Finset E}
    (h : LinearIndepOn F (fun e r => A r e) (s : Set E)) (g : E → F) :
    ∃ w ∈ rowSpace A, ∀ e ∈ s, w e = g e := by
  classical
  let res : (E → F) →ₗ[F] (s → F) := LinearMap.funLeft F F (Subtype.val : s → E)
  have htop : (rowSpace A).map res = ⊤ := by
    by_contra hne
    obtain ⟨f, hf0, hle⟩ := Submodule.exists_le_ker_of_lt_top _ (lt_top_iff_ne_top.2 hne)
    have hzero := Fintype.linearIndependent_iff.1 h
      (fun i : s => f fun j => if i = j then 1 else 0) (by
        funext r
        have hr : f (res (A r)) = 0 := hle (Submodule.mem_map_of_mem (row_mem_rowSpace A r))
        rw [LinearMap.pi_apply_eq_sum_univ] at hr
        simpa [Finset.sum_apply, res, LinearMap.funLeft_apply, mul_comm] using hr)
    exact hf0 (LinearMap.ext fun u => by rw [LinearMap.pi_apply_eq_sum_univ]; simp [hzero])
  obtain ⟨w, hw, hres⟩ := Submodule.mem_map.1
    (htop ▸ Submodule.mem_top : (fun i : s => g i) ∈ (rowSpace A).map res)
  exact ⟨w, hw, fun e he => congrFun hres ⟨e, he⟩⟩

/-- In a represented triangle, two columns are independent and the third is a combination of them
with nonzero coefficients. -/
theorem exists_col_eq_of_isCircuit {M : Matroid E} {A : Matrix R E F}
    (hA : IsMatrixRepresentation M A) {a b c : E} (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c)
    (ht : M.IsCircuit {a, b, c}) :
    LinearIndepOn F (fun e r => A r e) {a, c} ∧ ∃ p q : F, p ≠ 0 ∧ q ≠ 0 ∧
      (fun r => A r b) = p • (fun r => A r a) + q • fun r => A r c := by
  set v : E → R → F := fun e r => A r e with hv
  obtain ⟨hdep, hind⟩ := Matroid.isCircuit_iff_dep_forall_sdiff_singleton_indep.1 ht
  have hpair (u w d : E) (hd : d ∈ ({a, b, c} : Set E))
      (hset : ({a, b, c} : Set E) \ {d} = {u, w}) : LinearIndepOn F v {u, w} :=
    (hA.2 _).1 (hset ▸ hind d hd)
  have hset (u w d : E) (h : ∀ e, ((e = a ∨ e = b ∨ e = c) ∧ e ≠ d) ↔ (e = u ∨ e = w)) :
      ({a, b, c} : Set E) \ {d} = {u, w} := by
    ext e
    simpa only [Set.mem_sdiff, Set.mem_insert_iff, Set.mem_singleton_iff] using h e
  have hac' := hpair a c b (by simp) (hset a c b fun e => by grind)
  have hbc' := hpair b c a (by simp) (hset b c a fun e => by grind)
  have hab' := hpair a b c (by simp) (hset a b c fun e => by grind)
  have hnot : ¬LinearIndepOn F v (insert b {a, c}) := fun h =>
    hdep.1 ((hA.2 _).2 (by rw [Set.insert_comm]; exact h))
  rw [linearIndepOn_insert (by simp [hab.symm, hbc])] at hnot
  have hspan : v b ∈ Submodule.span F (v '' {a, c}) := by
    by_contra hn
    exact hnot ⟨hac', hn⟩
  rw [Set.image_pair, Submodule.mem_span_pair] at hspan
  obtain ⟨p, q, hpq⟩ := hspan
  refine ⟨hac', p, q, ?_, ?_, hpq.symm⟩
  · rintro rfl
    have := ((LinearIndepOn.pair_iff v hbc).1 hbc' 1 (-q) (by rw [← hpq]; simp)).1
    exact one_ne_zero this
  · rintro rfl
    have := ((LinearIndepOn.pair_iff v hab).1 hab' (-p) 1 (by rw [← hpq]; simp)).2
    exact one_ne_zero this

/-- **§2.4.** A represented triangle can be brought to the last three columns of `I(K₄)`: the rows
`x`, `y`, `z` are `inl 0`, `inl 1`, `inl 2`, and the other rows vanish on `a`, `b` and `c`. -/
theorem exists_isTriangleForm [DecidableEq E] [Finite E] {M : Matroid E} {A' : Matrix R E F}
    (hA : IsMatrixRepresentation M A') {a b c : E} (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c)
    (ht : M.IsCircuit {a, b, c}) :
    ∃ (n : ℕ) (A : Matrix (Fin 3 ⊕ Fin n) E F), ProjEquiv A' A ∧
      IsTriangleForm A a b c (.inl 0) (.inl 1) (.inl 2) := by
  obtain ⟨hind, p, q, hp, hq, hrel⟩ := exists_col_eq_of_isCircuit hA hab hac hbc ht
  let s : E → Fˣ := fun e => if e = a then Units.mk0 p hp else if e = c then Units.mk0 q hq else 1
  set B := scaleCols s A' with hB
  have hrow : ∀ w ∈ rowSpace B, w b = w a + w c := by
    intro w hw
    have hle : rowSpace B ≤ LinearMap.ker
        ((LinearMap.proj b : (E → F) →ₗ[F] F) - LinearMap.proj a - LinearMap.proj c) := by
      refine Submodule.span_le.2 ?_
      rintro _ ⟨r, rfl⟩
      have h := congrFun hrel r
      simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul] at h
      simp [hB, scaleCols, s, Matrix.row, hab.symm, hac.symm, hbc, h, mul_comm]
    have := hle hw
    simp only [LinearMap.mem_ker, LinearMap.sub_apply, LinearMap.proj_apply] at this
    linear_combination this
  have hindB : LinearIndepOn F (fun e r => B r e) (({a, c} : Finset E) : Set E) := by
    rw [Finset.coe_insert, Finset.coe_singleton]
    exact (ProjEquiv.linearIndepOn_iff ⟨s, rfl⟩ _).1 hind
  obtain ⟨w₁, hw₁, h₁⟩ := exists_mem_rowSpace_eq_on hindB fun e => if e = a then 1 else 0
  obtain ⟨w₂, hw₂, h₂⟩ := exists_mem_rowSpace_eq_on hindB fun e => if e = c then 1 else 0
  have h1a : w₁ a = 1 := by simpa using h₁ a (by simp)
  have h1c : w₁ c = 0 := by simpa [hac.symm] using h₁ c (by simp)
  have h2a : w₂ a = 0 := by simpa [hac] using h₂ a (by simp)
  have h2c : w₂ c = 1 := by simpa using h₂ c (by simp)
  have h1b : w₁ b = 1 := by rw [hrow w₁ hw₁, h1a, h1c, add_zero]
  have h2b : w₂ b = 1 := by rw [hrow w₂ hw₂, h2a, h2c, zero_add]
  let W₀ : Submodule F (E → F) := rowSpace B ⊓
    (LinearMap.ker (LinearMap.proj a : (E → F) →ₗ[F] F) ⊓ LinearMap.ker (LinearMap.proj c))
  obtain ⟨n, gen, hgen⟩ :=
    Submodule.fg_iff_exists_fin_generating_family.1 (IsNoetherian.noetherian W₀)
  have hgenW (j : Fin n) : gen j ∈ W₀ := hgen ▸ Submodule.subset_span ⟨j, rfl⟩
  have hga (j : Fin n) : gen j a = 0 := (hgenW j).2.1
  have hgc (j : Fin n) : gen j c = 0 := (hgenW j).2.2
  have hgb (j : Fin n) : gen j b = 0 := by rw [hrow _ (hgenW j).1, hga, hgc, add_zero]
  let rows : Fin 3 ⊕ Fin n → E → F := Sum.elim ![w₂, w₁ - w₂, -w₁] gen
  have h3 : ∀ i : Fin 3, i = 0 ∨ i = 1 ∨ i = 2 := by decide
  refine ⟨n, Matrix.of rows, ⟨s, le_antisymm ?_ (rowSpace_le_of_forall_mem ?_)⟩, ?_⟩
  · set S := rowSpace (Matrix.of rows) with hS
    have hx : w₂ ∈ S := row_mem_rowSpace (Matrix.of rows) (.inl 0)
    have hz : -w₁ ∈ S := row_mem_rowSpace (Matrix.of rows) (.inl 2)
    have hw₁S : w₁ ∈ S := by simpa using S.neg_mem hz
    intro w hw
    have hrest : w - w a • w₁ - w c • w₂ ∈ W₀ :=
      ⟨Submodule.sub_mem _ (Submodule.sub_mem _ hw (Submodule.smul_mem _ _ hw₁))
        (Submodule.smul_mem _ _ hw₂), by simp [h1a, h2a], by simp [h1c, h2c]⟩
    rw [← hgen] at hrest
    have hrestS := (Submodule.span_le.2 (by
      rintro _ ⟨j, rfl⟩
      exact row_mem_rowSpace (Matrix.of rows) (.inr j)) : Submodule.span F _ ≤ S) hrest
    have heq : w - w a • w₁ - w c • w₂ + w a • w₁ + w c • w₂ = w := by abel
    exact heq ▸ S.add_mem (S.add_mem hrestS (S.smul_mem (w a) hw₁S)) (S.smul_mem (w c) hx)
  · rintro (i | j)
    · rcases h3 i with rfl | rfl | rfl
      · exact hw₂
      · exact Submodule.sub_mem _ hw₁ hw₂
      · exact Submodule.neg_mem _ hw₁
    · exact (hgenW j).1
  · refine ⟨by simp, by simp, by simp, ?_, ?_, ?_⟩ <;> rintro (i | j)
    · rcases h3 i with rfl | rfl | rfl <;> simp [rows, h1a, h2a]
    · simp [rows, hga]
    · rcases h3 i with rfl | rfl | rfl <;> simp [rows, h1b, h2b]
    · simp [rows, hgb]
    · rcases h3 i with rfl | rfl | rfl <;> simp [rows, h1c, h2c]
    · simp [rows, hgc]

end Transformer.GainMatroid
