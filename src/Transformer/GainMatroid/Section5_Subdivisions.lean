module

public import Transformer.GainMatroid.Section1_Particular
public import Transformer.GainMatroid.Section2_ExchangeClasses
public import Transformer.GainMatroid.Section2_LiftMatroid

/-!
# Lemma 5.7: subdivisions

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §5.2. Subdividing edges does not change the
projective classes of representations or the classes of realizations: "a pair of edges incident to a
vertex of degree two form a series pair in the matroid". For a spanning subdivision `m` of `S` in
`Ω`, the bijections send the class of the canonical matrix of a realization `γ` of `Ω` to the class
of the canonical matrix of the gain function `e' ↦ γ(P_{e'})` that `γ` induces along the paths
`P_{e'}` of `m`.

Every biased graph is a spanning subdivision of itself (`SubdivisionMap.refl`).
-/

@[expose] public section

namespace Transformer.GainMatroid

universe uV uE uV' uE'

variable {V : Type uV} {E : Type uE} {V' : Type uV'} {E' : Type uE'}

open OrientedGraph

namespace BiasedGraph

/-- Every biased graph is a subdivision of itself, each edge its own walk. -/
def SubdivisionMap.refl (Ω : BiasedGraph V E) : Ω.SubdivisionMap Ω where
  vertex := id
  vertex_injective := Function.injective_id
  path e := .cons e (Or.inl ⟨rfl, rfl⟩) (.nil _)
  isPath e he := by simp [Walk.IsPath, Walk.support, he]
  isLoop _ _ := ⟨by simp [Walk.edges], by simp [Walk.edges], by simp [Walk.support]⟩
  ends e v hv _ := by
    simp only [Walk.support, List.mem_cons, List.not_mem_nil, or_false] at hv
    exact hv
  disjoint _ _ _ v _ _ := ⟨v, rfl⟩
  edge_disjoint e₁ e₂ hne f hf₁ hf₂ := by
    simp only [Walk.edges, List.mem_cons, List.not_mem_nil, or_false] at hf₁ hf₂
    exact hne (hf₁.symm.trans hf₂)
  balanced_iff C _ := by
    have h : (⋃ e ∈ C, (Walk.cons e (show Ω.graph.Joins e (Ω.graph.tail e) (Ω.graph.head e) from
        Or.inl ⟨rfl, rfl⟩) (.nil _)).edgeSet) = C := by
      ext f
      simp [Walk.edgeSet, Walk.edges]
    simp only [id] at h ⊢
    rw [h]

theorem SubdivisionMap.refl_isSpanning (Ω : BiasedGraph V E) :
    (SubdivisionMap.refl Ω).IsSpanning :=
  ⟨fun f => ⟨f, by simp [refl, Walk.edges]⟩, fun v => Or.inl ⟨v, rfl⟩⟩

variable {S : BiasedGraph V' E'} {Ω : BiasedGraph V E}

/-- **Lemma 5.7 (i).** "Let `(H, S)` be a subdivision of `(G, B)`. … The `𝔽`-matrix representations
of `F(H, S)` are in one-to-one correspondence with the `𝔽`-matrix representations of `F(G, B)` up to
projective equivalence." The bijection matches canonical frame matrices of `γ` and of the gains `γ`
induces along the paths of the subdivision. -/
theorem SubdivisionMap.exists_frame_projClass_equiv [DecidableEq V] [DecidableEq V']
    (m : S.SubdivisionMap Ω) (hm : m.IsSpanning) {M : Matroid E} {M' : Matroid E'}
    (hM : Ω.IsFrameMatroid M) (hM' : S.IsFrameMatroid M') (F : Type*) [Field F] :
    ∃ e : ProjClass F M ≃ ProjClass F M', ∀ (A : RepMatrix F M) (A' : RepMatrix F M')
      (γ : E → Fˣ), Ω.Realizes γ → ProjEquiv A.2.1 (frameMatrix Ω.graph γ) →
        ProjEquiv A'.2.1 (frameMatrix S.graph fun e' => (m.path e').gain γ) →
          e (Quotient.mk _ A) = Quotient.mk _ A' := by
  sorry

/-- **Lemma 5.7 (ii).** "The `𝔽`-matrix representations of `L(H, S)` are in one-to-one
correspondence with the `𝔽`-matrix representations of `L(G, B)` up to projective equivalence."
The bijection matches canonical lift matrices of `γ` and of the gains `γ` induces along the paths of
the subdivision. -/
theorem SubdivisionMap.exists_lift_projClass_equiv [DecidableEq V] [DecidableEq V']
    (m : S.SubdivisionMap Ω) (hm : m.IsSpanning) {M : Matroid E} {M' : Matroid E'}
    (hM : Ω.IsLiftMatroid M) (hM' : S.IsLiftMatroid M') (F : Type*) [Field F] :
    ∃ e : ProjClass F M ≃ ProjClass F M', ∀ (A : RepMatrix F M) (A' : RepMatrix F M')
      (γ : E → F), Ω.AddRealizes γ → ProjEquiv A.2.1 (liftMatrix Ω.graph γ) →
        ProjEquiv A'.2.1 (liftMatrix S.graph fun e' => (m.path e').addGain γ) →
          e (Quotient.mk _ A) = Quotient.mk _ A' := by
  sorry

/-- **Lemma 5.7 (iii)**, multiplicative. "The `Γ`-realizations of `(H, S)` are in one-to-one
correspondence with the `Γ`-realizations of `(G, B)` up to switching", for `Γ = 𝔽ˣ`; stated here for
any group. The bijection sends `γ` to the gains it induces along the paths of the subdivision. -/
theorem SubdivisionMap.exists_realizationClass_equiv [DecidableEq V] [DecidableEq V']
    (m : S.SubdivisionMap Ω) (hm : m.IsSpanning) (Γ : Type*) [Group Γ] :
    ∃ e : RealizationClass Ω Γ ≃ RealizationClass S Γ, ∀ (γ : E → Γ) (hγ : Ω.Realizes γ)
      (hγ' : S.Realizes fun e' => (m.path e').gain γ),
        e (Quotient.mk _ ⟨γ, hγ⟩) = Quotient.mk _ ⟨_, hγ'⟩ := by
  sorry

/-- **Lemma 5.7 (iii)**, additive: for `Γ = 𝔽⁺`, "up to … switching-and-scaling". -/
theorem SubdivisionMap.exists_addRealizationClass_equiv [DecidableEq V] [DecidableEq V']
    (m : S.SubdivisionMap Ω) (hm : m.IsSpanning) (F : Type*) [Field F] :
    ∃ e : AddRealizationClass Ω F ≃ AddRealizationClass S F, ∀ (γ : E → F)
      (hγ : Ω.AddRealizes γ) (hγ' : S.AddRealizes fun e' => (m.path e').addGain γ),
        e (Quotient.mk _ ⟨γ, hγ⟩) = Quotient.mk _ ⟨_, hγ'⟩ := by
  sorry

end BiasedGraph

open BiasedGraph

/-- The hypotheses of Lemma 5.7 hold for `2C₃` with the gains `twoC3Gain`, a spanning subdivision
of itself, and its frame and lift matroids. -/
example : (SubdivisionMap.refl (twoC3.gainBiased twoC3Gain)).IsSpanning ∧
    (∃ M, (twoC3.gainBiased twoC3Gain).IsFrameMatroid M) ∧
    ∃ M, (twoC3.gainBiased twoC3Gain).IsLiftMatroid M :=
  ⟨SubdivisionMap.refl_isSpanning _, exists_isFrameMatroid _, exists_isLiftMatroid _⟩

end Transformer.GainMatroid
