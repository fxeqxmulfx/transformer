module

public import Transformer.GainMatroid.Section2_TriadForms

/-!
# Which triads take the columns of `I(K₄)`

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4: "Similarly if `M(A′)` contains a triad `Y`,
then `A′` is projectively equivalent to a matrix `A` in which the columns corresponding to `Y` are
as the first three columns of `I(K₄)`, possibly with a row omitted or additional zero rows."

Those columns are independent, also with a row omitted (`IsTriadColumns.linearIndepOn`). So the
claim fails for a dependent triad: `triadColumns_fails` takes `U_{2,4}` over `𝔽₃`, whose triad
`{0, 1, 2}` is dependent. `Section2_K4Forms` proves the claim for an independent triad.
-/

@[expose] public section

namespace Transformer.GainMatroid

open OrientedGraph

variable {R E F : Type*} [Field F]

/-- The columns `a`, `b`, `c` of `A` are the first three columns of `I(K₄)`, with at most one row
omitted (`S` is the set of rows kept) and with zero rows added. -/
def IsTriadColumns (A : Matrix R E F) (a b c : E) : Prop :=
  ∃ (S : Set (Fin 4)) (ρ : S → R), Sᶜ.Subsingleton ∧ Function.Injective ρ ∧
    (∀ i : S, A (ρ i) a = iK4 (i : Fin 4) 0 ∧ A (ρ i) b = iK4 (i : Fin 4) 1 ∧
      A (ρ i) c = iK4 (i : Fin 4) 2) ∧
    ∀ r, r ∉ Set.range ρ → A r a = 0 ∧ A r b = 0 ∧ A r c = 0

/-- The first three columns of `I(K₄)` stay independent when one row is omitted. -/
theorem IsTriadColumns.linearIndepOn [DecidableEq E] {A : Matrix R E F} {a b c : E}
    (h : IsTriadColumns A a b c) (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) :
    LinearIndepOn F (fun e r => A r e) {a, b, c} := by
  obtain ⟨S, ρ, hS, -, hcol, -⟩ := h
  have hset : ({a, b, c} : Set E) = (({a, b, c} : Finset E) : Set E) := by simp
  rw [hset, linearIndepOn_finset_iff]
  intro f hf e he
  rw [Finset.sum_insert (by simp [hab, hac]), Finset.sum_pair hbc] at hf
  have heq (i : Fin 4) (hi : i ∈ S) :
      f a * iK4 i 0 + (f b * iK4 i 1 + f c * iK4 i 2) = 0 := by
    have := congrFun hf (ρ ⟨i, hi⟩)
    obtain ⟨h1, h2, h3⟩ := hcol ⟨i, hi⟩
    simpa [h1, h2, h3] using this
  have e0 : (0 : Fin 4) ∈ S → f a = 0 := fun h => by simpa [iK4] using heq 0 h
  have e1 : (1 : Fin 4) ∈ S → f b = 0 := fun h => by simpa [iK4] using heq 1 h
  have e2 : (2 : Fin 4) ∈ S → f c = 0 := fun h => by simpa [iK4] using heq 2 h
  have e3 : (3 : Fin 4) ∈ S → f a + (f b + f c) = 0 := fun h => by
    have := heq 3 h
    simp only [iK4, Matrix.of_apply, Matrix.cons_val', Matrix.cons_val_zero, Matrix.cons_val_one,
      Matrix.cons_val, Matrix.cons_val_fin_one] at this
    linear_combination -this
  have hor (i j : Fin 4) (hij : i ≠ j) : i ∈ S ∨ j ∈ S := by
    by_contra hn
    push Not at hn
    exact hij (hS hn.1 hn.2)
  have hall : f a = 0 ∧ f b = 0 ∧ f c = 0 := by
    by_cases h0 : (0 : Fin 4) ∈ S
    · by_cases h1 : (1 : Fin 4) ∈ S
      · by_cases h2 : (2 : Fin 4) ∈ S
        · exact ⟨e0 h0, e1 h1, e2 h2⟩
        · have h3 := e3 ((hor 2 3 (by decide)).resolve_left h2)
          exact ⟨e0 h0, e1 h1, by linear_combination h3 - e0 h0 - e1 h1⟩
      · have h2 := e2 ((hor 1 2 (by decide)).resolve_left h1)
        have h3 := e3 ((hor 1 3 (by decide)).resolve_left h1)
        exact ⟨e0 h0, by linear_combination h3 - e0 h0 - h2, h2⟩
    · have h1 := e1 ((hor 0 1 (by decide)).resolve_left h0)
      have h2 := e2 ((hor 0 2 (by decide)).resolve_left h0)
      have h3 := e3 ((hor 0 3 (by decide)).resolve_left h0)
      exact ⟨by linear_combination h3 - h1 - h2, h1, h2⟩
  simp only [Finset.mem_insert, Finset.mem_singleton] at he
  rcases he with rfl | rfl | rfl
  exacts [hall.1, hall.2.1, hall.2.2]

/-- `U_{2,4}` over `𝔽₃`. -/
def uniformTwoFourTernary : Matrix (Fin 2) (Fin 4) (ZMod 3) := !![1, 0, 1, 1; 0, 1, 1, 2]

set_option maxRecDepth 100000 in
theorem uniformTwoFourTernary_represents :
    IsMatrixRepresentation (finsetUniform (Fin 4) 2) uniformTwoFourTernary := by
  refine ⟨rfl, fun I => ?_⟩
  obtain ⟨I, rfl⟩ := I.toFinite.exists_finset_coe
  refine (finsetUniform_indep 2 I).trans (Iff.trans ?_ linearIndepOn_finset_iff.symm)
  revert I
  decide +kernel

set_option maxRecDepth 100000 in
/-- `{0, 1, 2}` is a triad of `U_{2,4}`. -/
theorem uniformTwoFour_triad : (finsetUniform (Fin 4) 2)✶.IsCircuit {0, 1, 2} := by
  have h : (finsetUniform (Fin 4) 2)✶.IsCircuit (({0, 1, 2} : Finset (Fin 4)) : Set (Fin 4)) :=
    (matroidOfFinset_dual_isCircuit_iff _).2 (by decide +kernel)
  simpa using h

/-- **The triad claim of §2.4 fails as printed.** `{0, 1, 2}` is a triad of `U_{2,4}`, and no matrix
projectively equivalent to a representation has the columns of `I(K₄)` on it, because they would
be independent. -/
theorem triadColumns_fails :
    ∃ (M : Matroid (Fin 4)) (A' : Matrix (Fin 2) (Fin 4) (ZMod 3)),
      IsMatrixRepresentation M A' ∧ M✶.IsCircuit {0, 1, 2} ∧
        ∀ (R' : Type) (A : Matrix R' (Fin 4) (ZMod 3)), ProjEquiv A' A →
          ¬IsTriadColumns A 0 1 2 := by
  refine ⟨_, _, uniformTwoFourTernary_represents, uniformTwoFour_triad, fun R' A hP hform => ?_⟩
  have hind := (uniformTwoFourTernary_represents.2 _).2
    ((hP.linearIndepOn_iff _).2 (hform.linearIndepOn (by decide) (by decide) (by decide)))
  have hc : (finsetUniform (Fin 4) 2).Indep (({0, 1, 2} : Finset (Fin 4)) : Set (Fin 4)) := by
    simpa using hind
  exact absurd ((finsetUniform_indep 2 _).1 hc) (by decide)

/-- The first three columns of `I(K₄)`, with no row omitted. -/
theorem iK4_isTriadColumns : IsTriadColumns (iK4 : Matrix (Fin 4) (Fin 6) F) 0 1 2 :=
  ⟨Set.univ, Subtype.val, by simp, Subtype.val_injective, fun _ => ⟨rfl, rfl, rfl⟩,
    fun r hr => (hr ⟨⟨r, trivial⟩, rfl⟩).elim⟩

/-- A matrix in triad form has the first three columns of `I(K₄)`, with no row omitted. -/
theorem IsTriadForm.isTriadColumns [DecidableEq R] {A : Matrix R E F} {v : R} {a b c : E}
    {x y z : R} (h : IsTriadForm A v a b c x y z) : IsTriadColumns A a b c := by
  obtain ⟨hxy, hxz, hyz, hvx, hvy, hvz, ha, hb, hc, -⟩ := h
  have hinj : Function.Injective ![x, y, z, v] := by
    intro i j hij
    fin_cases i <;> fin_cases j <;>
      simp_all [hxy.symm, hxz.symm, hyz.symm, hvx.symm, hvy.symm, hvz.symm]
  refine ⟨Set.univ, fun i => ![x, y, z, v] i, by simp, fun i j hij => Subtype.ext (hinj hij), ?_,
    fun r hr => ?_⟩
  · rintro ⟨i, -⟩
    fin_cases i <;> simp [ha, hb, hc, iK4, hxy, hxz, hyz, hvx, hvy, hvz, hxy.symm, hxz.symm,
      hyz.symm, hvx.symm, hvy.symm, hvz.symm]
  · have hne (i : Fin 4) : r ≠ ![x, y, z, v] i := fun heq => hr ⟨⟨i, trivial⟩, heq.symm⟩
    have h0 : r ≠ x := hne 0
    have h1 : r ≠ y := hne 1
    have h2 : r ≠ z := hne 2
    have h3 : r ≠ v := hne 3
    simp [ha, hb, hc, h0, h1, h2, h3]

example := (iK4_isTriadColumns (F := ℚ)).linearIndepOn (by decide) (by decide) (by decide)

example := (iK4_isTriadForm (F := ℚ)).isTriadColumns

end Transformer.GainMatroid
