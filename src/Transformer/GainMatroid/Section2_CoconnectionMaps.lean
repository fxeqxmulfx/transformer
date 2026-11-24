module

public import Transformer.GainMatroid.Section2_CoconnectionMatrix

/-!
# Linear maps between the columns of `A` and of `coconnectionMatrix`

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4, citing Brylawski. Let `A` be in the form
`IsTriadForm` on the triad `Y = {a, b, c}`, and let `Q = coconnectionMatrix A a b c x y z`: the rows
of `A` are `R`, those of `Q` are `R ⊕ Fin 3`. Four linear maps relate their columns.

* `mergeNew v` adds the new rows to the row `v`. It sends the column of `Q` at an element of `E` to
  its column in `A`, and the new columns to zero (`IsTriadForm.mergeNew_inl`, `mergeNew_inr`).
* `liftNew v j` extends a vector by zeros on the new rows and moves its entry at `v` to the new row
  `j`. It sends the column of `A` at an element outside `Y` to its column in `Q`
  (`IsTriadForm.liftNew_inl`), and at the `i`-th element of `Y` to its column in `Q` plus the
  difference of the new rows `i` and `j` (`IsTriadForm.liftNew_triad`).
* `newRows` restricts a vector to the new rows. There the columns of `Q` vanish off `Y`, and on the
  edges of `K₄` they are `k4NewCol` (`newRows_inl`, `newRows_deltaEmbed`).
* `contractNew x y z` adds the new rows `x', y', z'` to the rows `x, y, z`. It kills the columns of
  `Y`, and sends the other columns of `Q`, relabelled by `deltaRelabel`, to the columns of `∇_Y A`
  (`contractNew_triad`, `contractNew_deltaRelabel`). A vector differs from the extension of its
  image by a combination of the columns of `Y` (`sub_extendNew_contractNew_mem_span`).
-/

@[expose] public section

namespace Transformer.GainMatroid

open Submodule

variable {E R F : Type*} [Field F]

/-- Extending a vector by zeros on the new rows. -/
def extendNew : (R → F) →ₗ[F] R ⊕ Fin 3 → F where
  toFun w := Sum.elim w 0
  map_add' w w' := by
    rw [← Sum.elim_add_add, add_zero]
  map_smul' s w := by
    funext r
    cases r <;> simp

theorem extendNew_apply (w : R → F) : extendNew w = Sum.elim w 0 := rfl

/-- Restricting a vector to the new rows. -/
def newRows : (R ⊕ Fin 3 → F) →ₗ[F] Fin 3 → F :=
  LinearMap.funLeft F F Sum.inr

theorem newRows_apply (w : R ⊕ Fin 3 → F) : newRows w = w ∘ Sum.inr := rfl

variable [DecidableEq R]

/-- Adding the new rows to the row `v`. -/
def mergeNew (v : R) : (R ⊕ Fin 3 → F) →ₗ[F] R → F :=
  LinearMap.funLeft F F Sum.inl +
    ((LinearMap.proj (Sum.inr 0) : (R ⊕ Fin 3 → F) →ₗ[F] F) +
      (LinearMap.proj (Sum.inr 1) : (R ⊕ Fin 3 → F) →ₗ[F] F) +
      (LinearMap.proj (Sum.inr 2) : (R ⊕ Fin 3 → F) →ₗ[F] F)).smulRight (Pi.single v 1)

theorem mergeNew_apply (v : R) (w : R ⊕ Fin 3 → F) :
    mergeNew v w = w ∘ Sum.inl + (w (.inr 0) + w (.inr 1) + w (.inr 2)) • Pi.single v 1 := rfl

/-- Extending a vector by zeros on the new rows, and moving its entry at `v` to the new row `j`. -/
def liftNew (v : R) (j : Fin 3) : (R → F) →ₗ[F] R ⊕ Fin 3 → F :=
  extendNew +
    (LinearMap.proj v : (R → F) →ₗ[F] F).smulRight (Sum.elim (-Pi.single v 1) (Pi.single j 1))

theorem liftNew_apply (v : R) (j : Fin 3) (w : R → F) :
    liftNew v j w = Sum.elim w 0 + w v • Sum.elim (-Pi.single v 1) (Pi.single j 1) := rfl

/-- Adding the new rows `x', y', z'` to the rows `x, y, z`. -/
def contractNew (x y z : R) : (R ⊕ Fin 3 → F) →ₗ[F] R → F :=
  LinearMap.funLeft F F Sum.inl +
    (LinearMap.proj (.inr 0) : (R ⊕ Fin 3 → F) →ₗ[F] F).smulRight (Pi.single x 1) +
    (LinearMap.proj (.inr 1) : (R ⊕ Fin 3 → F) →ₗ[F] F).smulRight (Pi.single y 1) +
    (LinearMap.proj (.inr 2) : (R ⊕ Fin 3 → F) →ₗ[F] F).smulRight (Pi.single z 1)

theorem contractNew_apply (x y z : R) (w : R ⊕ Fin 3 → F) :
    contractNew x y z w = w ∘ Sum.inl + w (.inr 0) • Pi.single x 1 + w (.inr 1) • Pi.single y 1 +
      w (.inr 2) • Pi.single z 1 := rfl

variable [DecidableEq E] {A : Matrix R E F} {v : R} {a b c : E} {x y z : R}

theorem newRows_inl {f : E} (ha : f ≠ a) (hb : f ≠ b) (hc : f ≠ c) :
    newRows (coconnectionCol A a b c x y z (.inl f)) = 0 := by
  rw [newRows_apply, coconnectionCol_inl ha hb hc, Sum.elim_comp_inr]

theorem newRows_deltaEmbed (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) (g : Fin 3 ⊕ Fin 3) :
    newRows (coconnectionCol A a b c x y z (deltaEmbed a b c g)) = k4NewCol g := by
  rcases g with i | k
  · change newRows (coconnectionCol A a b c x y z (.inl (![a, b, c] i))) = _
    rw [newRows_apply, coconnectionCol_triad hab hac hbc, Sum.elim_comp_inr]
  · rfl

theorem mergeNew_inr (k : Fin 3) : mergeNew v (coconnectionCol A a b c x y z (.inr k)) = 0 := by
  rw [mergeNew_apply, coconnectionCol_inr, Sum.elim_comp_inl]
  fin_cases k <;> simp [k4NewCol]

theorem contractNew_triad (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) (i : Fin 3) :
    contractNew x y z (coconnectionCol A a b c x y z (.inl (![a, b, c] i))) = 0 := by
  rw [coconnectionCol_triad hab hac hbc, contractNew_apply, Sum.elim_comp_inl]
  fin_cases i <;> simp [k4NewCol]

theorem contractNew_deltaRelabel (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) (e : E) :
    contractNew x y z (coconnectionCol A a b c x y z (deltaRelabel a b c e)) =
      fun r => nablaMatrix A a b c x y z r e := by
  funext r
  rcases eq_or_eq_triad (a := a) (b := b) (c := c) e with ⟨ha, hb, hc⟩ | ⟨i, rfl⟩
  · simp [deltaRelabel, nablaMatrix, ha, hb, hc, contractNew_apply, coconnectionCol_inl ha hb hc]
  · fin_cases i <;>
      simp [deltaRelabel, nablaMatrix, hab.symm, hac.symm, hbc.symm, contractNew_apply,
        coconnectionCol_inr, k4NewCol, Pi.single_apply, sub_eq_add_neg]

/-- A vector differs from the extension of its image under `contractNew` by a combination of the
columns of `Y`. -/
theorem sub_extendNew_contractNew_mem_span (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c)
    (u : R ⊕ Fin 3 → F) :
    u - extendNew (contractNew x y z u) ∈
      span F (coconnectionCol A a b c x y z '' Set.range fun i => .inl (![a, b, c] i)) := by
  have hY (i : Fin 3) : coconnectionCol A a b c x y z (.inl (![a, b, c] i)) ∈
      span F (coconnectionCol A a b c x y z '' Set.range fun i => .inl (![a, b, c] i)) :=
    subset_span ⟨_, ⟨i, rfl⟩, rfl⟩
  have h : u - extendNew (contractNew x y z u) =
      -(u (.inr 0) • coconnectionCol A a b c x y z (.inl (![a, b, c] 0)) +
        u (.inr 1) • coconnectionCol A a b c x y z (.inl (![a, b, c] 1)) +
        u (.inr 2) • coconnectionCol A a b c x y z (.inl (![a, b, c] 2))) := by
    simp only [coconnectionCol_triad hab hac hbc]
    funext r
    rcases r with r | k
    · simp [extendNew_apply, contractNew_apply]
      ring
    · fin_cases k <;> simp [extendNew_apply, contractNew_apply, k4NewCol]
  rw [h]
  exact neg_mem (add_mem (add_mem (smul_mem _ _ (hY 0)) (smul_mem _ _ (hY 1)))
    (smul_mem _ _ (hY 2)))

namespace IsTriadForm

theorem mergeNew_inl (hform : IsTriadForm A v a b c x y z) (e : E) :
    mergeNew v (coconnectionCol A a b c x y z (.inl e)) = fun r => A r e := by
  rcases eq_or_eq_triad (a := a) (b := b) (c := c) e with ⟨ha, hb, hc⟩ | ⟨i, rfl⟩
  · rw [mergeNew_apply, coconnectionCol_inl ha hb hc, Sum.elim_comp_inl]
    simp
  · rw [mergeNew_apply, coconnectionCol_triad hform.a_ne_b hform.a_ne_c hform.b_ne_c,
      Sum.elim_comp_inl]
    funext r
    rw [hform.col_triad]
    fin_cases i <;> simp [k4NewCol, sub_eq_add_neg]

theorem liftNew_inl (hform : IsTriadForm A v a b c x y z) (j : Fin 3) {f : E} (ha : f ≠ a)
    (hb : f ≠ b) (hc : f ≠ c) :
    liftNew v j (fun r => A r f) = coconnectionCol A a b c x y z (.inl f) := by
  simp only [liftNew_apply, hform.row_v f ha hb hc, zero_smul, add_zero,
    coconnectionCol_inl ha hb hc]

theorem liftNew_triad (hform : IsTriadForm A v a b c x y z) (i j : Fin 3) :
    liftNew v j (fun r => A r (![a, b, c] i)) =
      coconnectionCol A a b c x y z (.inl (![a, b, c] i)) +
      Sum.elim (0 : R → F) (Pi.single i 1 - Pi.single j 1) := by
  rw [coconnectionCol_triad hform.a_ne_b hform.a_ne_c hform.b_ne_c]
  funext r
  rcases r with r | k
  · simp only [liftNew_apply, Pi.add_apply, Pi.smul_apply, Sum.elim_inl, hform.triad_v i,
      hform.col_triad i r]
    simp
  · simp only [liftNew_apply, Pi.add_apply, Pi.smul_apply, Sum.elim_inr, hform.triad_v i]
    simp [k4NewCol]
    ring

end IsTriadForm

example := (iK4_isTriadForm (F := ℚ)).mergeNew_inl 3

example := (iK4_isTriadForm (F := ℚ)).liftNew_inl 0 (f := 3) (by decide) (by decide) (by decide)

example := (iK4_isTriadForm (F := ℚ)).liftNew_triad 0 1

example := newRows_inl (A := (iK4 : Matrix (Fin 4) (Fin 6) ℚ)) (a := 0) (b := 1) (c := 2)
  (x := 0) (y := 1) (z := 2) (f := 3) (by decide) (by decide) (by decide)

example := newRows_deltaEmbed (A := (iK4 : Matrix (Fin 4) (Fin 6) ℚ)) (a := 0) (b := 1) (c := 2)
  (x := 0) (y := 1) (z := 2) (by decide) (by decide) (by decide) (.inl 0)

example := contractNew_triad (A := (iK4 : Matrix (Fin 4) (Fin 6) ℚ)) (a := 0) (b := 1) (c := 2)
  (x := 0) (y := 1) (z := 2) (by decide) (by decide) (by decide) 0

example := contractNew_deltaRelabel (A := (iK4 : Matrix (Fin 4) (Fin 6) ℚ)) (a := 0) (b := 1)
  (c := 2) (x := 0) (y := 1) (z := 2) (by decide) (by decide) (by decide) 3

example := sub_extendNew_contractNew_mem_span (A := (iK4 : Matrix (Fin 4) (Fin 6) ℚ)) (a := 0)
  (b := 1) (c := 2) (x := 0) (y := 1) (z := 2) (by decide) (by decide) (by decide) 0

end Transformer.GainMatroid
