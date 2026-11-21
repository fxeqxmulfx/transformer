module

public import Transformer.Grokking.FlatSpectrum.TablePart.DegreeHalf
public import Transformer.Grokking.FlatSpectrum.TablePart.FlatCycles

/-!
# PT4: the best part

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT4, *The best*. A part
that determines the flat class has at least `dim H = q^k + q^l − 1 + k l` cells
(`add_le_card_of_determines`). The cross `{(u, 0)} ∪ {(0, v)}` (`cross`) with the `k l` cells
`(e_i, e_j)` (`crossPart`) has exactly that many (`card_crossPart`) and determines the class
(`determines_crossPart`). The cross alone is connected (`connected_cross`), so it fixes
`g(a) + h(b)` (`rigid_cross`). At `p = 97`, `k = l = 1` the best part has 194 cells
(`exists_determines_card_194`, `le_card_of_determines_97`), one more than the 193 of the cross
(`card_cross_97`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.TablePart

open Finset Matrix Module SimpleGraph

variable (F ι κ : Type*) [Field F] [Fintype F] [DecidableEq F]
  [Fintype ι] [Fintype κ] [DecidableEq ι] [DecidableEq κ]

/-- The cross `{(u, 0)} ∪ {(0, v)}` (`review/flat_spectrum.md`, PT4). -/
def cross : Finset ((ι → F) × (κ → F)) :=
  (univ.image fun u => (u, 0)) ∪ univ.image fun v => (0, v)

/-- The best part (`review/flat_spectrum.md`, PT4): the cross with the `k l` cells `(e_i, e_j)`. -/
def crossPart : Finset ((ι → F) × (κ → F)) :=
  cross F ι κ ∪ univ.image fun ij : ι × κ => (Pi.single ij.1 1, Pi.single ij.2 1)

variable {F ι κ}

lemma row_mem_cross (u : ι → F) : (u, (0 : κ → F)) ∈ cross F ι κ :=
  mem_union_left _ (mem_image_of_mem _ (mem_univ u))

lemma col_mem_cross (v : κ → F) : ((0 : ι → F), v) ∈ cross F ι κ :=
  mem_union_right _ (mem_image_of_mem _ (mem_univ v))

/-- The cross has `q^k + q^l − 1` cells, written without subtraction. -/
lemma card_cross :
    #(cross F ι κ) + 1 = Fintype.card F ^ Fintype.card ι + Fintype.card F ^ Fintype.card κ := by
  have hinter : (univ.image fun u : ι → F => (u, (0 : κ → F))) ∩
      (univ.image fun v : κ → F => ((0 : ι → F), v)) = {(0, 0)} := by
    ext ⟨u, v⟩
    simp only [mem_inter, mem_image, mem_univ, true_and, Prod.mk.injEq, mem_singleton]
    constructor
    · rintro ⟨⟨u', rfl, rfl⟩, ⟨v', hu, hv⟩⟩
      exact ⟨hu.symm, rfl⟩
    · rintro ⟨rfl, rfl⟩
      exact ⟨⟨0, rfl, rfl⟩, ⟨0, rfl, rfl⟩⟩
  have h := card_union_add_card_inter (univ.image fun u : ι → F => (u, (0 : κ → F)))
    (univ.image fun v : κ → F => ((0 : ι → F), v))
  rw [hinter, card_singleton, card_image_of_injective _ fun _ _ h => (Prod.ext_iff.1 h).1,
    card_image_of_injective _ fun _ _ h => (Prod.ext_iff.1 h).2, card_univ, card_univ,
    Fintype.card_fun, Fintype.card_fun] at h
  exact h

/-- The graph of the cross is connected (`review/flat_spectrum.md`, PT4). -/
theorem connected_cross : (patternGraph (cross F ι κ : Set ((ι → F) × (κ → F)))).Connected := by
  have hrow (v : κ → F) : (patternGraph (cross F ι κ : Set ((ι → F) × (κ → F)))).Adj
      (.inl 0) (.inr v) := patternGraph_adj_inl_inr.2 (mem_coe.2 (col_mem_cross v))
  have hcol (u : ι → F) : (patternGraph (cross F ι κ : Set ((ι → F) × (κ → F)))).Adj
      (.inr 0) (.inl u) := patternGraph_adj_inr_inl.2 (mem_coe.2 (row_mem_cross u))
  refine (connected_iff_exists_forall_reachable _).2 ⟨.inl 0, ?_⟩
  rintro (u | v)
  · exact (hrow 0).reachable.trans (hcol u).reachable
  · exact (hrow v).reachable

/-- The cross determines `g(a) + h(b)` (`review/flat_spectrum.md`, PT4, by *Rigidity is
connectivity*): if `r(u) + s(v) = 0` on the cross, it is zero everywhere. -/
theorem rigid_cross (r : (ι → F) → F) (s : (κ → F) → F)
    (h : ∀ c ∈ cross F ι κ, r c.1 + s c.2 = 0) (u : ι → F) (v : κ → F) : r u + s v = 0 :=
  rigid_of_preconnected connected_cross.preconnected r s (fun c hc => h c (mem_coe.1 hc)) u v

example (u : ι → F) (v : κ → F) : (0 : F) + 0 = 0 :=
  rigid_cross (F := F) (ι := ι) (κ := κ) (fun _ => 0) (fun _ => 0) (fun _ _ => add_zero 0) u v

/-- **The cross with the cells `(e_i, e_j)` determines the flat class**
(`review/flat_spectrum.md`, PT4). A member that vanishes there has `M_ij` as a mixed difference,
so `M = 0`, and then the cross kills the rest. -/
theorem determines_crossPart :
    Determines (flatClass F ι κ : Set ((ι → F) × (κ → F) → F)) (crossPart F ι κ : Set _) := by
  refine (determines_submodule_iff _ _).2 fun f hf h0 => ?_
  rw [← range_flatEval] at hf
  obtain ⟨p, rfl⟩ := hf
  have hr (u : ι → F) : flatEval F ι κ p (u, 0) = 0 :=
    h0 _ (mem_coe.2 (mem_union_left _ (row_mem_cross u)))
  have hc (v : κ → F) : flatEval F ι κ p (0, v) = 0 :=
    h0 _ (mem_coe.2 (mem_union_left _ (col_mem_cross v)))
  have hM : p.2 = 0 := by
    ext i j
    have he : flatEval F ι κ p (Pi.single i 1, Pi.single j 1) = 0 :=
      h0 _ (mem_coe.2 (mem_union_right _ (mem_image_of_mem _ (mem_univ (i, j)))))
    have hmix := flatEval_mixed p (Pi.single i 1) 0 (Pi.single j 1) 0
    rw [he, hc, hr, hr] at hmix
    simpa [mulVec_single_one] using hmix.symm
  funext ⟨u, v⟩
  have hmix := flatEval_mixed p u 0 v 0
  rw [hc, hr, hr, hM] at hmix
  simpa using hmix

omit [DecidableEq F] in
/-- **A part that determines the flat class has at least `dim H` cells**
(`review/flat_spectrum.md`, PT4), written without subtraction. -/
theorem add_le_card_of_determines {Ω : Finset ((ι → F) × (κ → F))}
    (h : Determines (flatClass F ι κ : Set ((ι → F) × (κ → F) → F)) Ω) :
    Fintype.card F ^ Fintype.card ι + Fintype.card F ^ Fintype.card κ +
      Fintype.card ι * Fintype.card κ ≤ #Ω + 1 := by
  have h1 := finrank_le_card_of_determines _ h
  have h2 := finrank_flatClass (F := F) (ι := ι) (κ := κ)
  omega

example := add_le_card_of_determines (determines_crossPart (F := ZMod 2) (ι := Fin 1) (κ := Fin 1))

/-- **The best part has exactly `dim H` cells** (`review/flat_spectrum.md`, PT4):
`q^k + q^l − 1 + k l`, written without subtraction. -/
theorem card_crossPart :
    #(crossPart F ι κ) + 1 = Fintype.card F ^ Fintype.card ι + Fintype.card F ^ Fintype.card κ +
      Fintype.card ι * Fintype.card κ := by
  refine le_antisymm ?_ (add_le_card_of_determines determines_crossPart)
  have h1 := card_union_le (cross F ι κ)
    (univ.image fun ij : ι × κ => ((Pi.single ij.1 1 : ι → F), (Pi.single ij.2 1 : κ → F)))
  have h2 : #(univ.image fun ij : ι × κ => ((Pi.single ij.1 1 : ι → F),
      (Pi.single ij.2 1 : κ → F))) ≤ Fintype.card ι * Fintype.card κ :=
    card_image_le.trans (by simp)
  have h3 := card_cross (F := F) (ι := ι) (κ := κ)
  unfold crossPart
  omega

section Numbers

attribute [local instance] fact_prime_97

/-- **At `p = 97` the best part has 194 cells** (`review/flat_spectrum.md`, PT4). -/
theorem exists_determines_card_194 :
    ∃ Ω : Finset ((Fin 1 → ZMod 97) × (Fin 1 → ZMod 97)), #Ω = 194 ∧
      Determines (flatClass (ZMod 97) (Fin 1) (Fin 1) :
        Set ((Fin 1 → ZMod 97) × (Fin 1 → ZMod 97) → ZMod 97)) Ω := by
  refine ⟨crossPart _ _ _, ?_, determines_crossPart⟩
  have h := card_crossPart (F := ZMod 97) (ι := Fin 1) (κ := Fin 1)
  simp only [ZMod.card, Fintype.card_fin] at h
  omega

/-- **No smaller part works at `p = 97`** (`review/flat_spectrum.md`, PT4). -/
theorem le_card_of_determines_97 {Ω : Finset ((Fin 1 → ZMod 97) × (Fin 1 → ZMod 97))}
    (h : Determines (flatClass (ZMod 97) (Fin 1) (Fin 1) :
      Set ((Fin 1 → ZMod 97) × (Fin 1 → ZMod 97) → ZMod 97)) Ω) : 194 ≤ #Ω := by
  have hle := add_le_card_of_determines h
  simp only [ZMod.card, Fintype.card_fin] at hle
  omega

example : 194 ≤ #(crossPart (ZMod 97) (Fin 1) (Fin 1)) :=
  le_card_of_determines_97 determines_crossPart

/-- The cross at `p = 97` has the 193 cells that determine `g(a) + h(b)`
(`review/flat_spectrum.md`, PT4). -/
theorem card_cross_97 : #(cross (ZMod 97) (Fin 1) (Fin 1)) = 193 := by
  have h := card_cross (F := ZMod 97) (ι := Fin 1) (κ := Fin 1)
  simp only [ZMod.card, Fintype.card_fin] at h
  omega

end Numbers

end Transformer.Grokking.FlatSpectrum.TablePart
