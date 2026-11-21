module

public import Transformer.GainMatroid.Section2_InducedMinors

/-!
# Proposition 2.4 for the contractions of a joint

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), Proposition 2.4 for the two contractions of a joint
(§2.1, §2.3) and for the minors of the proof of Theorem 6, which use one frame-type contraction.
In a frame-type contraction at `v`, an edge at `v` is a loop, so a cycle through it has no other
edge; every other cycle avoids `v` and is a cycle of `Ω` with the same gains
(`IsFrameContraction.realizes`). The lift-type contraction has the identity gain.

Two loops at one vertex are the worked example: contracting one as a frame-type joint leaves the
other as a balanced loop.
-/

@[expose] public section

namespace Transformer.GainMatroid

open OrientedGraph

universe uV uE uV' uE'

variable {V : Type uV} {E : Type uE} {V' : Type uV'} {E' : Type uE'} {Γ : Type*}

namespace BiasedGraph

variable {Ω : BiasedGraph V E} {Ω' : BiasedGraph V' E'} {e : E} {v : V}

/-- **Proposition 2.4** for a frame-type contraction of a joint. -/
theorem IsFrameContraction.realizes [DecidableEq V] [Group Γ]
    {Ω₂ : BiasedGraph V ↥({e}ᶜ : Set E)} (h : Ω.IsFrameContraction e v Ω₂) {γ : E → Γ}
    {ι : ↥({e}ᶜ : Set E) → Γ} (hγ : Ω.Realizes γ) (hι : Ω.IsFrameInducedGain e v γ ι) :
    Ω₂.Realizes ι := by
  have hne : γ e ≠ 1 := (hγ.isJoint_iff h.isJoint.1).1 h.isJoint
  have hoff : ∀ f : ↥({e}ᶜ : Set E), Ω.graph.tail f ≠ v → Ω.graph.head f ≠ v →
      Ω₂.graph.tail f = Ω.graph.tail f ∧ Ω₂.graph.head f = Ω.graph.head f := fun f h₁ h₂ => by
    rw [h.graph_eq]
    exact frameContract_of_off h₁ h₂
  have hloop : ∀ f : ↥({e}ᶜ : Set E), (Ω.graph.tail f = v ∨ Ω.graph.head f = v) →
      Ω₂.graph.tail f = Ω₂.graph.head f := fun f hf => by
    rw [h.graph_eq]
    exact frameContract_loop_of_touch hf
  have key : ∀ C, Ω₂.graph.IsCycle C → (C ∈ Ω₂.graph.balancedCycles ι ↔
      (∃ f : ↥({e}ᶜ : Set E), Ω.graph.tail f = v ∧ Ω.graph.head f = v ∧ C = {f}) ∨
        (Subtype.val '' C ∈ Ω.balanced ∧ v ∉ Ω.graph.verts (Subtype.val '' C))) := by
    intro C hC
    by_cases htouch : ∃ f ∈ C, Ω.graph.tail f = v ∨ Ω.graph.head f = v
    · obtain ⟨f, hfC, hf⟩ := htouch
      have hCs : C = {f} := by
        obtain ⟨u, W, hW, rfl⟩ := hC
        exact hW.edgeSet_eq_singleton_of_loop hfC (hloop f hf)
      subst hCs
      rw [singleton_mem_balancedCycles_iff ι (hloop f hf)]
      rcases hι f with ⟨h₁, h₂, -⟩ | ⟨h₁, h₂, hf₁⟩ | ⟨hlink, -, hf₁⟩
      · exact absurd hf (not_or.2 ⟨h₁, h₂⟩)
      · exact ⟨fun _ => Or.inl ⟨f, h₁, h₂, rfl⟩, fun _ => hf₁⟩
      · rw [hf₁]
        refine ⟨fun hc => absurd hc hne, ?_⟩
        rintro (⟨g, h₁, h₂, hg⟩ | ⟨-, hv⟩)
        · obtain rfl : f = g := Set.singleton_eq_singleton_iff.1 hg
          exact absurd (h₁.trans h₂.symm) hlink
        · exact (hv (mem_verts.2 ⟨(f : E), Set.mem_image_of_mem _ (Set.mem_singleton f), hf⟩)).elim
    · have hoffC : ∀ f ∈ C, Ω.graph.tail f ≠ v ∧ Ω.graph.head f ≠ v := fun f hf =>
        not_or.1 fun hc => htouch ⟨f, hf, hc⟩
      have hmap : ∀ f ∈ C, Ω.graph.tail (Subtype.val f) = Ω₂.graph.tail f ∧
          Ω.graph.head (Subtype.val f) = Ω₂.graph.head f := fun f hf => by
        obtain ⟨h₁, h₂⟩ := hoff f (hoffC f hf).1 (hoffC f hf).2
        exact ⟨h₁.symm, h₂.symm⟩
      have hι' : ∀ f ∈ C, ι f = γ f := fun f hf => by
        rcases hι f with ⟨-, -, h₁⟩ | ⟨h₁, -⟩ | ⟨-, h₁, -⟩
        · exact h₁
        · exact absurd h₁ (hoffC f hf).1
        · exact absurd h₁ (not_or.2 (hoffC f hf))
      have hsw : ∀ (ξ : V → Γ), ∀ f ∈ C,
          switch Ω₂.graph ξ ι f = switch Ω.graph ξ γ f := fun ξ f hf => by
        obtain ⟨h₁, h₂⟩ := hmap f hf
        rw [switch_apply, switch_apply, ← h₁, ← h₂, hι' f hf]
      have hcyc : Ω.graph.IsCycle (Subtype.val '' C) :=
        hC.map Subtype.val Subtype.val_injective hmap
      have hv : v ∉ Ω.graph.verts (Subtype.val '' C) := fun hv => by
        obtain ⟨_, ⟨f, hf, rfl⟩, h₁ | h₁⟩ := mem_verts.1 hv
        · exact (hoffC f hf).1 h₁
        · exact (hoffC f hf).2 h₁
      constructor
      · intro hb
        obtain ⟨-, ξ, hξ⟩ := mem_balancedCycles_iff.1 hb
        refine Or.inr ⟨?_, hv⟩
        rw [← show Ω.graph.balancedCycles γ = Ω.balanced from hγ]
        refine mem_balancedCycles_iff.2 ⟨hcyc, ξ, ?_⟩
        rintro _ ⟨f, hf, rfl⟩
        rw [← hsw ξ f hf]
        exact hξ f hf
      · rintro (⟨f, h₁, -, rfl⟩ | ⟨hb, -⟩)
        · exact absurd h₁ (hoffC f rfl).1
        · rw [← show Ω.graph.balancedCycles γ = Ω.balanced from hγ] at hb
          obtain ⟨-, ξ, hξ⟩ := mem_balancedCycles_iff.1 hb
          refine mem_balancedCycles_iff.2 ⟨hC, ξ, fun f hf => ?_⟩
          rw [hsw ξ f hf]
          exact hξ f (Set.mem_image_of_mem _ hf)
  change Ω₂.graph.balancedCycles ι = Ω₂.balanced
  ext C
  constructor
  · intro hC
    have hcyc := (mem_balancedCycles_iff.1 hC).1
    exact (h.balanced_iff C hcyc).2 ((key C hcyc).1 hC)
  · intro hC
    have hcyc := Ω₂.isCycle_of_mem _ hC
    exact (key C hcyc).2 ((h.balanced_iff C hcyc).1 hC)

/-- **Proposition 2.4** for a lift-type contraction of a joint: the identity gain realizes it. -/
theorem IsLiftContraction.realizes_one [DecidableEq V] [Group Γ]
    {Ω₂ : BiasedGraph V ↥({e}ᶜ : Set E)} (h : Ω.IsLiftContraction e Ω₂) :
    Ω₂.Realizes (1 : ↥({e}ᶜ : Set E) → Γ) := by
  change Ω₂.graph.balancedCycles 1 = Ω₂.balanced
  ext C
  constructor
  · exact fun hC => h.balanced C (mem_balancedCycles_iff.1 hC).1
  · exact fun hC => mem_balancedCycles_iff.2
      ⟨Ω₂.isCycle_of_mem _ hC, 1, fun f _ => by simp [switch]⟩

/-- **Proposition 2.4** for a minor with one frame-type contraction of a joint. -/
theorem FrameMinorMap.realizes_of_isInducedGain [DecidableEq V] [DecidableEq V'] [Group Γ]
    (m : Ω.FrameMinorMap Ω') {γ : E → Γ} {ι : E' → Γ} (hγ : Ω.Realizes γ)
    (hι : m.IsInducedGain γ ι) : Ω'.Realizes ι := by
  classical
  obtain ⟨γ₁, γ₂, h₁, h₂, h₃⟩ := hι
  exact m.second.realizes_of_isInducedGain
    (m.contraction.realizes (m.first.realizes_of_isInducedGain hγ h₁) h₂) h₃

/-- A gain function induces itself on the trivial link minor. -/
theorem LinkMinorMap.isInducedGain_refl [Group Γ] (Ω : BiasedGraph V E) (γ : E → Γ) :
    (LinkMinorMap.refl Ω).IsInducedGain γ γ :=
  ⟨1, fun f hf => absurd hf (Set.notMem_empty f), fun e => Or.inl ⟨rfl, by rw [switch_one]; rfl⟩⟩

example [DecidableEq V] [Group Γ] (G : OrientedGraph V E) (γ : E → Γ) :
    (G.gainBiased γ).Realizes γ :=
  (LinkMinorMap.refl (G.gainBiased γ)).realizes_of_isInducedGain rfl
    (LinkMinorMap.isInducedGain_refl _ γ)

/-- The frame-type contraction of the joint `true` of two loops: one balanced loop. -/
noncomputable def twoLoopsContract : BiasedGraph Unit ↥({true}ᶜ : Set Bool) :=
  ((twoLoops.frameContract ()).delete {true}).gainBiased (1 : ↥({true}ᶜ : Set Bool) → ℚˣ)

theorem twoLoops_isFrameContraction :
    twoLoops.contrabalanced.IsFrameContraction true () twoLoopsContract where
  isJoint := ⟨rfl, Set.notMem_empty _⟩
  tail_eq := rfl
  graph_eq := rfl
  balanced_iff C hC := by
    refine ⟨fun _ => ?_, fun _ => mem_balancedCycles_iff.2 ⟨hC, 1, fun f _ => by simp [switch]⟩⟩
    obtain ⟨u, W, hW, rfl⟩ := hC
    obtain ⟨f, hf⟩ := List.exists_mem_of_ne_nil _ hW.edges_ne_nil
    exact Or.inl ⟨f, rfl, rfl, hW.edgeSet_eq_singleton_of_loop hf (Subsingleton.elim _ _)⟩

/-- The constant gain `2` makes both loops unbalanced. -/
theorem twoLoops_realizes : twoLoops.contrabalanced.Realizes (fun _ : Bool => ratTwo) := by
  change twoLoops.balancedCycles _ = ∅
  refine Set.eq_empty_of_forall_notMem fun C hC => ?_
  obtain ⟨u, W, hW, rfl⟩ := (mem_balancedCycles_iff.1 hC).1
  obtain ⟨f, hf⟩ := List.exists_mem_of_ne_nil _ hW.edges_ne_nil
  rw [hW.edgeSet_eq_singleton_of_loop hf rfl, singleton_mem_balancedCycles_iff _ rfl] at hC
  exact ratTwo_ne_one hC

example : twoLoopsContract.Realizes (1 : ↥({true}ᶜ : Set Bool) → ℚˣ) :=
  twoLoops_isFrameContraction.realizes twoLoops_realizes fun _ => Or.inr (Or.inl ⟨rfl, rfl, rfl⟩)

example : ((twoLoops.delete {true}).gainBiased (1 : ↥({true}ᶜ : Set Bool) → ℚˣ)).Realizes
    (1 : ↥({true}ᶜ : Set Bool) → ℚˣ) :=
  IsLiftContraction.realizes_one (Ω := twoLoops.contrabalanced) (e := true)
    ⟨⟨rfl, Set.notMem_empty _⟩, rfl,
      fun _ hC => mem_balancedCycles_iff.2 ⟨hC, 1, fun f _ => by simp [switch]⟩⟩

/-- Two loops have the balanced loop as a minor with one frame-type contraction. -/
noncomputable def twoLoopsFrameMinor :
    twoLoops.contrabalanced.FrameMinorMap twoLoopsContract where
  V₁ := Unit
  E₁ := Bool
  Ω₁ := twoLoops.contrabalanced
  first := LinkMinorMap.refl _
  joint := true
  vertex := ()
  Ω₂ := twoLoopsContract
  contraction := twoLoops_isFrameContraction
  second := LinkMinorMap.refl _

example : twoLoopsContract.Realizes (1 : ↥({true}ᶜ : Set Bool) → ℚˣ) :=
  twoLoopsFrameMinor.realizes_of_isInducedGain twoLoops_realizes
    ⟨fun _ => ratTwo, 1, LinkMinorMap.isInducedGain_refl _ _,
      fun _ => Or.inr (Or.inl ⟨rfl, rfl, rfl⟩), LinkMinorMap.isInducedGain_refl _ _⟩

end BiasedGraph

end Transformer.GainMatroid
