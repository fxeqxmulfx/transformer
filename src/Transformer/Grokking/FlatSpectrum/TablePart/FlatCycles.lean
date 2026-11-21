module

public import Transformer.Grokking.FlatSpectrum.TablePart.FlatRank

/-!
# PT3: four-cycles

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT3. A four-cycle on the
rows `u, u'` and the columns `v, v'` has the twist `(u − u')(v − v')ᵀ`
(`exists_fourCycle_twist`). So at `k = l = 1` every connected part with a four-cycle
determines the flat class (`determines_of_connected_of_fourCycle`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.TablePart

open Matrix Module SimpleGraph

variable {F ι κ : Type*} [Field F]

/-- A four-cycle has the twist `(u − u')(v − v')ᵀ` (`review/flat_spectrum.md`, PT3). -/
theorem exists_fourCycle_twist {Ω : Set ((ι → F) × (κ → F))} {u u' : ι → F} {v v' : κ → F}
    (h₁ : (u, v) ∈ Ω) (h₂ : (u', v) ∈ Ω) (h₃ : (u', v') ∈ Ω) (h₄ : (u, v') ∈ Ω) :
    ∃ w : (patternGraph Ω).Walk (.inl u) (.inl u),
      w.length = 4 ∧ twist w = vecMulVec (u - u') (v - v') := by
  refine ⟨.cons (patternGraph_adj_inl_inr.2 h₁) <| .cons (patternGraph_adj_inr_inl.2 h₂) <|
    .cons (patternGraph_adj_inl_inr.2 h₃) <| .cons (patternGraph_adj_inr_inl.2 h₄) .nil, rfl, ?_⟩
  simp only [twist_row_step, twist_nil, add_zero]
  ext i j
  simp [vecMulVec_apply]
  ring

example : ∃ w : (patternGraph (Set.univ : Set ((Fin 1 → ℚ) × (Fin 1 → ℚ)))).Walk (.inl 0) (.inl 0),
    w.length = 4 ∧ twist w = vecMulVec ((0 : Fin 1 → ℚ) - 1) ((0 : Fin 1 → ℚ) - 1) :=
  exists_fourCycle_twist (Set.mem_univ _) (Set.mem_univ _) (Set.mem_univ _) (Set.mem_univ _)

/-- The twist of a four-cycle lies in the span of the twists. -/
theorem vecMulVec_sub_mem_twistSpan {Ω : Set ((ι → F) × (κ → F))} {u u' : ι → F} {v v' : κ → F}
    (h₁ : (u, v) ∈ Ω) (h₂ : (u', v) ∈ Ω) (h₃ : (u', v') ∈ Ω) (h₄ : (u, v') ∈ Ω) :
    vecMulVec (u - u') (v - v') ∈ twistSpan Ω := by
  obtain ⟨w, -, hw⟩ := exists_fourCycle_twist h₁ h₂ h₃ h₄
  exact hw ▸ twist_mem_twistSpan w

example : vecMulVec ((fun _ => 1 : Fin 1 → ℚ) - fun _ => 0) ((fun _ => 1 : Fin 1 → ℚ) - fun _ => 0)
    ∈ twistSpan (Set.univ : Set ((Fin 1 → ℚ) × (Fin 1 → ℚ))) :=
  vecMulVec_sub_mem_twistSpan (Set.mem_univ _) (Set.mem_univ _) (Set.mem_univ _) (Set.mem_univ _)

/-- Among `1 × 1` matrices, one nonzero twist spans everything. -/
lemma twistSpan_eq_top_of_ne_zero {Ω : Set ((Fin 1 → F) × (Fin 1 → F))}
    {Z : Matrix (Fin 1) (Fin 1) F} (hZ : Z ∈ twistSpan Ω) (h0 : Z ≠ 0) : twistSpan Ω = ⊤ := by
  have hz : Z 0 0 ≠ 0 := fun h => h0 (by
    ext i j
    rw [Subsingleton.elim i 0, Subsingleton.elim j 0]
    simpa using h)
  refine (Submodule.eq_top_iff').2 fun A => ?_
  have hA : A = (A 0 0 / Z 0 0) • Z := by
    ext i j
    rw [Subsingleton.elim i 0, Subsingleton.elim j 0]
    simp [hz]
  rw [hA]
  exact Submodule.smul_mem _ _ hZ

example : twistSpan (Set.univ : Set ((Fin 1 → ℚ) × (Fin 1 → ℚ))) = ⊤ :=
  twistSpan_eq_top_of_ne_zero
    (vecMulVec_sub_mem_twistSpan (u := 1) (u' := 0) (v := 1) (v' := 0)
      (Set.mem_univ _) (Set.mem_univ _) (Set.mem_univ _) (Set.mem_univ _))
    fun h => by simpa [vecMulVec_apply] using congrFun (congrFun h 0) 0

/-- The whole table has a connected graph. -/
lemma connected_patternGraph_univ {m n : Type*} [Nonempty m] [Nonempty n] :
    (patternGraph (Set.univ : Set (m × n))).Connected := by
  obtain ⟨i₀⟩ := ‹Nonempty m›
  obtain ⟨j₀⟩ := ‹Nonempty n›
  have hrow (j : n) : (patternGraph (Set.univ : Set (m × n))).Adj (.inl i₀) (.inr j) :=
    patternGraph_adj_inl_inr.2 (Set.mem_univ _)
  have hcol (i : m) : (patternGraph (Set.univ : Set (m × n))).Adj (.inr j₀) (.inl i) :=
    patternGraph_adj_inr_inl.2 (Set.mem_univ _)
  refine (connected_iff_exists_forall_reachable _).2 ⟨.inl i₀, ?_⟩
  rintro (i | j)
  · exact (hrow j₀).reachable.trans (hcol i).reachable
  · exact (hrow j).reachable

/-- **A connected part with a four-cycle at `k = l = 1`** (`review/flat_spectrum.md`, PT3): it
determines the flat class. -/
theorem determines_of_connected_of_fourCycle [Fintype F] {Ω : Set ((Fin 1 → F) × (Fin 1 → F))}
    (hΩ : (patternGraph Ω).Connected) {u u' v v' : Fin 1 → F} (hu : u ≠ u') (hv : v ≠ v')
    (h₁ : (u, v) ∈ Ω) (h₂ : (u', v) ∈ Ω) (h₃ : (u', v') ∈ Ω) (h₄ : (u, v') ∈ Ω) :
    Determines (flatClass F (Fin 1) (Fin 1) : Set ((Fin 1 → F) × (Fin 1 → F) → F)) Ω := by
  refine (determines_flatClass_iff Ω).2 ⟨hΩ,
    twistSpan_eq_top_of_ne_zero (vecMulVec_sub_mem_twistSpan h₁ h₂ h₃ h₄) fun h => ?_⟩
  have h00 : (u 0 - u' 0) * (v 0 - v' 0) = 0 := by
    simpa [vecMulVec_apply] using congrFun (congrFun h 0) 0
  rcases mul_eq_zero.1 h00 with h | h
  · exact hu (funext fun i => by rw [Subsingleton.elim i 0]; exact sub_eq_zero.1 h)
  · exact hv (funext fun i => by rw [Subsingleton.elim i 0]; exact sub_eq_zero.1 h)

example : Determines (flatClass (ZMod 2) (Fin 1) (Fin 1) :
    Set ((Fin 1 → ZMod 2) × (Fin 1 → ZMod 2) → ZMod 2)) Set.univ :=
  determines_of_connected_of_fourCycle (u := fun _ => 0) (u' := fun _ => 1) (v := fun _ => 0)
    (v' := fun _ => 1) connected_patternGraph_univ (fun h => zero_ne_one (congrFun h 0))
    (fun h => zero_ne_one (congrFun h 0)) (Set.mem_univ _) (Set.mem_univ _) (Set.mem_univ _)
    (Set.mem_univ _)

end Transformer.Grokking.FlatSpectrum.TablePart
