module

public import Transformer.GainMatroid.Section2_TriadColumns

/-!
# Bringing a triad to the columns of `I(K₄)`

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4: "Similarly if `M(A′)` contains a triad `Y`,
then `A′` is projectively equivalent to a matrix `A` in which the columns corresponding to `Y` are
as the first three columns of `I(K₄)`, possibly with a row omitted or additional zero rows."

This fails for a dependent triad (`triadColumns_fails`). `exists_isTriadForm` proves it for an
independent triad. The row of the fourth vertex is the vector of the row space supported on `Y`.
-/

@[expose] public section

namespace Transformer.GainMatroid

open OrientedGraph

variable {R E F : Type*} [Field F]

/-- **§2.4, corrected.** An independent triad of a represented matroid can be brought to the first
three columns of `I(K₄)`, whose fourth row vanishes on every other column. -/
theorem exists_isTriadForm [DecidableEq E] [Finite E] {M : Matroid E} {A' : Matrix R E F}
    (hA : IsMatrixRepresentation M A') {a b c : E} (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c)
    (ht : M✶.IsCircuit {a, b, c}) (hi : M.Indep {a, b, c}) :
    ∃ (n : ℕ) (A : Matrix (Fin 4 ⊕ Fin n) E F), ProjEquiv A' A ∧
      IsTriadForm A (.inl 3) a b c (.inl 0) (.inl 1) (.inl 2) := by
  classical
  obtain ⟨w₀, hw₀, hsupp, hoff⟩ := hA.exists_mem_rowSpace_of_cocircuit ht
  let s : E → Fˣ := fun e =>
    if h : w₀ e = 0 then 1 else Units.mk0 (-(w₀ e)⁻¹) (neg_ne_zero.2 (inv_ne_zero h))
  set B := scaleCols s A' with hB
  let v : E → F := fun e => w₀ e * s e
  have hv : v ∈ rowSpace B := by
    rw [hB, rowSpace_scaleCols]
    exact Submodule.mem_map_of_mem hw₀
  have hvY : ∀ e ∈ ({a, b, c} : Set E), v e = -1 := fun e he => by
    have h0 := hsupp e he
    simp [v, s, h0]
  have hvoff : ∀ e, e ≠ a → e ≠ b → e ≠ c → v e = 0 := fun e h1 h2 h3 => by
    have h0 : w₀ e = 0 := hoff e (by simp [h1, h2, h3])
    simp [v, h0]
  have hind : LinearIndepOn F (fun e r => B r e) (({a, b, c} : Finset E) : Set E) := by
    simp only [Finset.coe_insert, Finset.coe_singleton]
    exact (ProjEquiv.linearIndepOn_iff ⟨s, rfl⟩ _).1 ((hA.2 _).1 hi)
  obtain ⟨wx, hwx, hx⟩ := exists_mem_rowSpace_eq_on hind fun e => if e = a then 1 else 0
  obtain ⟨wy, hwy, hy⟩ := exists_mem_rowSpace_eq_on hind fun e => if e = b then 1 else 0
  have hxa : wx a = 1 := by simpa using hx a (by simp)
  have hxb : wx b = 0 := by simpa [hab.symm] using hx b (by simp)
  have hxc : wx c = 0 := by simpa [hac.symm] using hx c (by simp)
  have hya : wy a = 0 := by simpa [hab] using hy a (by simp)
  have hyb : wy b = 1 := by simpa using hy b (by simp)
  have hyc : wy c = 0 := by simpa [hbc.symm] using hy c (by simp)
  have hva := hvY a (by simp)
  have hvb := hvY b (by simp)
  have hvc := hvY c (by simp)
  let wz : E → F := -v - wx - wy
  have hwz : wz ∈ rowSpace B :=
    Submodule.sub_mem _ (Submodule.sub_mem _ (Submodule.neg_mem _ hv) hwx) hwy
  have hza : wz a = 0 := by simp [wz, hva, hxa, hya]
  have hzb : wz b = 0 := by simp [wz, hvb, hxb, hyb]
  have hzc : wz c = 1 := by simp [wz, hvc, hxc, hyc]
  let W₀ : Submodule F (E → F) := rowSpace B ⊓ (LinearMap.ker (LinearMap.proj a : (E → F) →ₗ[F] F) ⊓
    (LinearMap.ker (LinearMap.proj b) ⊓ LinearMap.ker (LinearMap.proj c)))
  obtain ⟨n, gen, hgen⟩ :=
    Submodule.fg_iff_exists_fin_generating_family.1 (IsNoetherian.noetherian W₀)
  have hgenW (j : Fin n) : gen j ∈ W₀ := hgen ▸ Submodule.subset_span ⟨j, rfl⟩
  have hga (j : Fin n) : gen j a = 0 := (hgenW j).2.1
  have hgb (j : Fin n) : gen j b = 0 := (hgenW j).2.2.1
  have hgc (j : Fin n) : gen j c = 0 := (hgenW j).2.2.2
  let rows : Fin 4 ⊕ Fin n → E → F := Sum.elim ![wx, wy, wz, v] gen
  have h4 : ∀ i : Fin 4, i = 0 ∨ i = 1 ∨ i = 2 ∨ i = 3 := by decide
  refine ⟨n, Matrix.of rows, ⟨s, le_antisymm ?_ (rowSpace_le_of_forall_mem ?_)⟩, ?_⟩
  · set S := rowSpace (Matrix.of rows) with hS
    have hxS : wx ∈ S := row_mem_rowSpace (Matrix.of rows) (.inl 0)
    have hyS : wy ∈ S := row_mem_rowSpace (Matrix.of rows) (.inl 1)
    have hzS : wz ∈ S := row_mem_rowSpace (Matrix.of rows) (.inl 2)
    intro w hw
    have hrest : w - w a • wx - w b • wy - w c • wz ∈ W₀ :=
      ⟨Submodule.sub_mem _ (Submodule.sub_mem _ (Submodule.sub_mem _ hw
        (Submodule.smul_mem _ _ hwx)) (Submodule.smul_mem _ _ hwy)) (Submodule.smul_mem _ _ hwz),
        by simp [hxa, hya, hza], by simp [hxb, hyb, hzb], by simp [hxc, hyc, hzc]⟩
    rw [← hgen] at hrest
    have hrestS := (Submodule.span_le.2 (by
      rintro _ ⟨j, rfl⟩
      exact row_mem_rowSpace (Matrix.of rows) (.inr j)) : Submodule.span F _ ≤ S) hrest
    have heq : w - w a • wx - w b • wy - w c • wz + w a • wx + w b • wy + w c • wz = w := by abel
    exact heq ▸ S.add_mem (S.add_mem (S.add_mem hrestS (S.smul_mem _ hxS)) (S.smul_mem _ hyS))
      (S.smul_mem _ hzS)
  · rintro (i | j)
    · rcases h4 i with rfl | rfl | rfl | rfl
      exacts [hwx, hwy, hwz, hv]
    · exact (hgenW j).1
  · refine ⟨by simp, by simp, by simp, by simp, by simp, by simp, ?_, ?_, ?_, fun e h1 h2 h3 => ?_⟩
    · rintro (i | j)
      · rcases h4 i with rfl | rfl | rfl | rfl <;> simp [rows, hxa, hya, hza, hva]
      · simp [rows, hga]
    · rintro (i | j)
      · rcases h4 i with rfl | rfl | rfl | rfl <;> simp [rows, hxb, hyb, hzb, hvb]
      · simp [rows, hgb]
    · rintro (i | j)
      · rcases h4 i with rfl | rfl | rfl | rfl <;> simp [rows, hxc, hyc, hzc, hvc]
      · simp [rows, hgc]
    · simpa [rows] using hvoff e h1 h2 h3

example := exists_isTriadForm clawMatrix_represents (by decide) (by decide) (by decide)
  clawMatroid_triad clawMatroid_star_indep

end Transformer.GainMatroid
