module

public import Transformer.Grokking.FiniteRecovery.Series
public import Transformer.Grokking.FlatSpectrum.PatternGraph

/-! Sample size for several aligned series with an additive row–time law.
Flat Spectrum's bipartite observation graph now has the series on one side
and the times on the other (review/flat_spectrum.md, Setting and Scope). -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery

open SimpleGraph Transformer.Grokking.FlatSpectrum
open Transformer.Grokking.FlatSpectrum.TablePart

variable {R T G : Type*}

/-- The additive class on a grid consists of a shared time profile plus
a separate offset for each series. -/
def additiveSeriesFamily (R T G : Type*) [Add G] : Set (R × T → G) :=
  {f | ∃ row : R → G, ∃ time : T → G, f = fun p => row p.1 + time p.2}

/-- The additive class is determined exactly when the observed pairs connect
all series and time positions. -/
theorem determines_additiveSeries_iff [Nonempty R] [Nonempty T]
    [AddCommGroup G] [Nontrivial G] (sites : Set (R × T)) :
    Determines (additiveSeriesFamily R T G) sites ↔ (patternGraph sites).Connected := by
  constructor
  · intro h
    apply connected_of_rigid (G := G)
    intro row time hzero r t
    have hf : (fun c : R × T => row c.1 + time c.2) ∈ additiveSeriesFamily R T G :=
      ⟨row, time, rfl⟩
    have h0 : (fun _ : R × T => (0 : G)) ∈ additiveSeriesFamily R T G :=
      ⟨fun _ => 0, fun _ => 0, by simp⟩
    exact congrFun (h _ hf _ h0 hzero) (r, t)
  · intro h
    rintro _ ⟨r, t, rfl⟩ _ ⟨r', t', rfl⟩ hagree
    funext c
    have hzero : ∀ p ∈ sites, (r p.1 - r' p.1) + (t p.2 - t' p.2) = 0 := by
      intro p hp
      rw [← add_sub_add_comm, sub_eq_zero]
      exact hagree p hp
    have hf := rigid_of_preconnected h.preconnected (fun r₀ => r r₀ - r' r₀)
      (fun t₀ => t t₀ - t' t₀) hzero c.1 c.2
    rwa [← add_sub_add_comm, sub_eq_zero] at hf

/-- An additive law gives a rank-one phase table across series and time. -/
theorem additiveSeries_phase_factor [AddCommGroup G] (code : AddChar G ℂ) (r : R → G) (t : T → G) :
    (Matrix.of fun i j => code (r i + t j)) = Matrix.vecMulVec (code ∘ r) (code ∘ t) := by
  ext i j
  exact code.map_add_eq_mul (r i) (t j)

/-- A row and a column attain the bound. The corner is counted once. -/
theorem isLeast_additiveSeries_observations [Fintype R] [Fintype T] [Nonempty R] [Nonempty T]
    (G : Type*) [AddCommGroup G] [Nontrivial G] :
    IsLeast {n : ℕ | ∃ sites : Finset (R × T), sites.card = n ∧
      Determines (additiveSeriesFamily R T G) sites} (Fintype.card R + Fintype.card T - 1) := by
  classical
  obtain ⟨r₀⟩ := ‹Nonempty R›
  obtain ⟨t₀⟩ := ‹Nonempty T›
  let row : Finset (R × T) := Finset.univ.image fun t => (r₀, t)
  let column : Finset (R × T) := (Finset.univ.erase r₀).image fun r => (r, t₀)
  let cross := row ∪ column
  have hrow (t : T) : (r₀, t) ∈ cross := by simp [cross, row]
  have hcol (r : R) : (r, t₀) ∈ cross := by
    by_cases hr : r = r₀
    · exact hr.symm ▸ hrow t₀
    · simp [cross, column, hr]
  refine ⟨⟨cross, ?_, (determines_additiveSeries_iff _).2 ?_⟩, ?_⟩
  · have hdisjoint : Disjoint row column := by
      rw [Finset.disjoint_left]
      rintro ⟨r, t⟩ hr hc
      have hEq : r₀ = r := by simpa [row] using hr
      have hNe : r ≠ r₀ ∧ t₀ = t := by simpa [column] using hc
      exact hNe.1 hEq.symm
    have hRowInj : Function.Injective (fun t : T => (r₀, t)) := fun _ _ h => congrArg Prod.snd h
    have hColInj : Function.Injective (fun r : R => (r, t₀)) := fun _ _ h => congrArg Prod.fst h
    rw [show cross = row ∪ column from rfl, Finset.card_union_of_disjoint hdisjoint]
    simp only [row, column, Finset.card_image_of_injective _ hRowInj,
      Finset.card_image_of_injective _ hColInj, Finset.card_erase_of_mem (Finset.mem_univ r₀),
      Finset.card_univ]
    have := Fintype.card_pos (α := R)
    omega
  · let K := patternGraph (cross : Set (R × T))
    have hleft (t : T) : K.Adj (.inl r₀) (.inr t) := patternGraph_adj_inl_inr.2 (hrow t)
    have hright (r : R) : K.Adj (.inr t₀) (.inl r) := patternGraph_adj_inr_inl.2 (hcol r)
    refine (connected_iff_exists_forall_reachable K).2 ⟨.inl r₀, ?_⟩
    rintro (r | t)
    · exact (hleft t₀).reachable.trans (hright r).reachable
    · exact (hleft t).reachable
  · rintro _ ⟨sites, rfl, hdet⟩
    have h := card_add_card_le_ncard_add_one_of_connected
      ((determines_additiveSeries_iff (G := G) _).1 hdet)
    rw [Nat.card_eq_fintype_card, Nat.card_eq_fintype_card, Set.ncard_coe_finset] at h
    omega

/-- Thirty-two series sampled at 128 common times need 159 well-placed
labels for additive recovery. All 4096 sites are available to the mask. -/
example : IsLeast {n : ℕ | ∃ sites : Finset (Fin 32 × Fin 128), sites.card = n ∧
    Determines (additiveSeriesFamily (Fin 32) (Fin 128) (ZMod 2)) sites} 159 := by
  simpa using isLeast_additiveSeries_observations (R := Fin 32) (T := Fin 128) (ZMod 2)

end Transformer.Grokking.FiniteRecovery
