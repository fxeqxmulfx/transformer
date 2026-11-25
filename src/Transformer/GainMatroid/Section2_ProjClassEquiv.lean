module

public import Transformer.GainMatroid.Section2_DeltaTriad
public import Transformer.GainMatroid.Section2_ExchangeExamples
public import Transformer.GainMatroid.Section2_TriadRowSpace

/-!
# Projective classes across a Δ-Y exchange

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4, Proposition 2.9 (ii), credited to Whittle,
for a coindependent triangle `X = {a, b, c}` of `M` (`IsDeltaMatroid.nonempty_projClass_equiv`).
A representation of `M` is projectively equivalent to one in triangle form `A`
(`exists_isTriangleForm`), and then `Δ_X A` represents `Δ_X M`
(`IsDeltaMatroid.deltaMatrix_represents`, by `isDeltaMatroid_of_isTriangleForm` and the uniqueness
of `Δ_X M`). Relate a representation of `M` to the representations of `Δ_X M` projectively
equivalent to `Δ_X A` for such an `A`. A relation between representatives that is total and
surjective, and under which related pairs are equivalent on one side exactly when they are on the
other, gives a bijection of the quotients (`nonempty_quotient_equiv`).

* Projectively equivalent representations of `M` have projectively equivalent exchanges
  (`IsTriangleForm.projEquiv_deltaMatrix`).
* Conversely, projectively equivalent exchanges come from projectively equivalent representations
  (`IsTriangleForm.projEquiv_of_deltaMatrix`), since a coindependent triangle supports no nonzero
  vector of the row space (`IsMatrixRepresentation.eq_zero_of_dual_indep`).
* Every representation `B'` of `Δ_X M` is related to one of `M`. The triangle `X` is an independent
  triad of `Δ_X M` (`IsDeltaMatroid.indep_triangle`, `IsDeltaMatroid.dual_isCircuit`), so `B'` is
  projectively equivalent to some `B` in triad form (`exists_isTriadForm`). Then `∇_X B` represents
  `M` (`IsDeltaMatroid.nablaMatrix_represents`), is in triangle form, and `Δ_X ∇_X B` has the row
  space of `B` (`IsTriadForm.rowSpace_deltaMatrix_nablaMatrix`).

A representation with a finite type of rows is projectively equivalent to one with rows `Fin m`
(`exists_repMatrix`).
-/

@[expose] public section

namespace Transformer.GainMatroid

/-- **Two quotients are equivalent** when a relation between representatives is total and
surjective, and related pairs are equivalent on one side exactly when they are on the other. -/
theorem nonempty_quotient_equiv {α β : Type*} {s : Setoid α} {t : Setoid β} (r : α → β → Prop)
    (htot : ∀ a, ∃ b, r a b) (hsurj : ∀ b, ∃ a, r a b)
    (hcompat : ∀ a a' b b', r a b → r a' b' → (s.r a a' ↔ t.r b b')) :
    Nonempty (Quotient s ≃ Quotient t) := by
  choose f hf using htot
  let g : Quotient s → Quotient t := Quotient.lift (fun a => Quotient.mk t (f a)) fun a a' h =>
    Quotient.sound ((hcompat a a' (f a) (f a') (hf a) (hf a')).1 h)
  refine ⟨Equiv.ofBijective g ⟨fun p q hpq => ?_, fun q => ?_⟩⟩
  · revert hpq
    refine Quotient.inductionOn₂ p q fun a a' hpq => Quotient.sound ?_
    exact (hcompat a a' (f a) (f a') (hf a) (hf a')).2 (Quotient.exact hpq)
  · refine Quotient.inductionOn q fun b => ?_
    obtain ⟨a, ha⟩ := hsurj b
    exact ⟨Quotient.mk s a, Quotient.sound ((hcompat a a (f a) b (hf a) ha).1 (s.iseqv.refl a))⟩

variable {E R F : Type*} [Field F]

/-- A representation with a finite type of rows is projectively equivalent to one with rows
`Fin m`. -/
theorem exists_repMatrix [Finite E] [Fintype R] {M : Matroid E} {A : Matrix R E F}
    (hA : IsMatrixRepresentation M A) : ∃ B : RepMatrix F M, ProjEquiv A B.2.1 := by
  set σ := (Fintype.equivFin R).symm
  have h : ProjEquiv A (A.submatrix σ id) := ⟨1, by
    rw [scaleCols_one, rowSpace, rowSpace]
    exact congrArg (Submodule.span F) (σ.surjective.range_comp A.row).symm⟩
  exact ⟨⟨Fintype.card R, A.submatrix σ id, hA.of_projEquiv h⟩, h⟩

variable [DecidableEq E] [Finite E] [DecidableEq R] {M M' : Matroid E} {a b c : E}

/-- `Δ_X A` represents `Δ_X M` when `A` represents `M` in triangle form. -/
theorem IsDeltaMatroid.deltaMatrix_represents (h : IsDeltaMatroid M a b c M') {A : Matrix R E F}
    {x y z : R} (hA : IsMatrixRepresentation M A) (hform : IsTriangleForm A a b c x y z) :
    IsMatrixRepresentation M' (deltaMatrix A a b c x y z) := by
  have hP := matrixMatroid_isMatrixRepresentation (deltaMatrix A a b c x y z)
  rwa [(isDeltaMatroid_of_isTriangleForm hA hform hP).eq h] at hP

/-- `∇_X B` represents `M` when `B` represents `Δ_X M` in triad form and `X` is a triad of
`Δ_X M`. -/
theorem IsDeltaMatroid.nablaMatrix_represents (h : IsDeltaMatroid M a b c M')
    (ht : M'✶.IsCircuit {a, b, c}) {B : Matrix R E F} {v x y z : R}
    (hB : IsMatrixRepresentation M' B) (hform : IsTriadForm B v a b c x y z) :
    IsMatrixRepresentation M (nablaMatrix B a b c x y z) := by
  have hQ := matrixMatroid_isMatrixRepresentation (nablaMatrix B a b c x y z)
  rwa [h.eq_of_isNablaMatroid ht (isNablaMatroid_of_isTriadForm hB hform ht hQ)] at hQ

variable (F) in
/-- **Proposition 2.9 (ii)** for a coindependent triangle: the projective classes of
`F`-representations of `M` and of `Δ_X M` correspond. -/
theorem IsDeltaMatroid.nonempty_projClass_equiv (h : IsDeltaMatroid M a b c M')
    (hco : M✶.Indep {a, b, c}) : Nonempty (ProjClass F M ≃ ProjClass F M') := by
  refine nonempty_quotient_equiv (fun (A' : RepMatrix F M) (B' : RepMatrix F M') =>
    ∃ (n : ℕ) (A : Matrix (Fin 3 ⊕ Fin n) E F), ProjEquiv A'.2.1 A ∧
      IsTriangleForm A a b c (.inl 0) (.inl 1) (.inl 2) ∧
      ProjEquiv (deltaMatrix A a b c (.inl 0) (.inl 1) (.inl 2)) B'.2.1) ?_ ?_ ?_
  · rintro ⟨m, A', hA'⟩
    obtain ⟨n, A, hpe, hform⟩ := exists_isTriangleForm hA' h.ne_ab h.ne_ac h.ne_bc h.triangle
    obtain ⟨B', hB'⟩ := exists_repMatrix (h.deltaMatrix_represents (hA'.of_projEquiv hpe) hform)
    exact ⟨B', n, A, hpe, hform, hB'⟩
  · rintro ⟨m, B', hB'⟩
    have ht := h.dual_isCircuit hco
    obtain ⟨n, B, hpe, hform⟩ :=
      exists_isTriadForm hB' h.ne_ab h.ne_ac h.ne_bc ht h.indep_triangle
    obtain ⟨A', hA'⟩ :=
      exists_repMatrix (h.nablaMatrix_represents ht (hB'.of_projEquiv hpe) hform)
    obtain ⟨n', A, hpe', hform'⟩ :=
      exists_isTriangleForm A'.2.2 h.ne_ab h.ne_ac h.ne_bc h.triangle
    refine ⟨A', n', A, hpe', hform', ?_⟩
    exact ((hform'.projEquiv_deltaMatrix hform.isTriangleForm_nablaMatrix
      (hpe'.symm.trans hA'.symm)).trans ⟨1, by
        rw [scaleCols_one, hform.rowSpace_deltaMatrix_nablaMatrix]⟩).trans hpe.symm
  · rintro ⟨m₁, A'₁, hA'₁⟩ ⟨m₂, A'₂, hA'₂⟩ B'₁ B'₂ ⟨n₁, A₁, p₁, f₁, q₁⟩ ⟨n₂, A₂, p₂, f₂, q₂⟩
    change ProjEquiv A'₁ A'₂ ↔ ProjEquiv B'₁.2.1 B'₂.2.1
    constructor
    · intro hA
      exact q₁.symm.trans ((f₁.projEquiv_deltaMatrix f₂ (p₁.symm.trans (hA.trans p₂))).trans q₂)
    · intro hB
      refine p₁.trans ((f₁.projEquiv_of_deltaMatrix f₂ (fun w hw h₀ => ?_)
        (q₁.trans (hB.trans q₂.symm))).trans p₂.symm)
      exact (hA'₂.of_projEquiv p₂).eq_zero_of_dual_indep hco hw h₀

/-- `Δ_X` of the triangle of `U_{2,3}` is the free matroid, read off `triangleMatrix`. -/
theorem twoOfThree_isDeltaMatroid : IsDeltaMatroid twoOfThree 1 2 0 freeThree :=
  isDeltaMatroid_of_isTriangleForm triangleMatrix_represents triangleMatrix_isTriangleForm
    deltaMatrix_triangleMatrix_represents

example := nonempty_quotient_equiv (s := ⟨Eq, eq_equivalence⟩) (t := ⟨Eq, eq_equivalence⟩)
  (fun a b : Bool => a = b) (fun a => ⟨a, rfl⟩) (fun b => ⟨b, rfl⟩) fun _ _ _ _ h h' => by
    subst h h'
    rfl

example := exists_repMatrix iK4_represents_binary

example := twoOfThree_isDeltaMatroid.eq twoOfThree_isDeltaMatroid

example := twoOfThree_isDeltaMatroid.indep_triangle

example := twoOfThree_isDeltaMatroid.deltaMatrix_represents triangleMatrix_represents
  triangleMatrix_isTriangleForm

example := nablaMatroid_isDeltaMatroid.nablaMatrix_represents clawMatroid_triad
  clawMatrix_represents clawMatrix_isTriadForm

example := nablaMatroid_isDeltaMatroid.eq_of_isNablaMatroid clawMatroid_triad
  (isNablaMatroid_of_isTriadForm clawMatrix_represents clawMatrix_isTriadForm clawMatroid_triad
    nablaMatrix_clawMatrix_represents)

end Transformer.GainMatroid
