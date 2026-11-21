module

public import Transformer.Grokking.FlatSpectrum.TablePart.Twist

/-!
# PT3: the count of the functions that vanish on a part

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT3. Let `c` be the number
of components of the graph of `Ω`, isolated vertices included, and `s` the dimension of the span of
the twists. The members of the flat class that vanish on `Ω` (`vanishing`) form a space of dimension

  `(c − 1) + (k l − s)` (`finrank_vanishing`).

The proof follows the source on the parameters `(x, M)` of `flatEval`. Along a walk, a parameter
that vanishes on `Ω` telescopes: `⟨M, twist w⟩ = x(b) − x(a)` (`pair_twist_eq`), so `M` is
orthogonal to the twists. Conversely, for such an `M`, carrying the values along walks from a root
of each component gives a potential (`exists_potential`). For one `M` two potentials differ by a
function constant on the components (`finrank_potentials`). So the parameters that vanish on `Ω`
have dimension `(k l − s) + c`, and dividing by the kernel of the parametrization removes one.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.TablePart

open Matrix Module SimpleGraph

section Potentials

variable (F : Type*) [Field F] {α : Type*} (G : SimpleGraph α)

/-- The functions that agree across every edge of `G`. -/
def potentials : Submodule F (α → F) where
  carrier := {x | ∀ a b, G.Adj a b → x a = x b}
  add_mem' hx hy a b h := by simp [hx a b h, hy a b h]
  zero_mem' _ _ _ := rfl
  smul_mem' t x hx a b h := by simp [hx a b h]

variable {F G}

/-- A potential is constant along a walk. -/
lemma potentials_eq_of_walk {x : α → F} (hx : x ∈ potentials F G) {a b : α} (w : G.Walk a b) :
    x a = x b := by
  induction w with
  | nil => rfl
  | cons h _ ih => exact (hx _ _ h).trans ih

variable (F G)

/-- The potentials are the functions on the components: one free constant per component. -/
theorem finrank_potentials [Finite α] :
    finrank F (potentials F G) = Nat.card G.ConnectedComponent := by
  classical
  have := Fintype.ofFinite α
  have := Fintype.ofFinite G.ConnectedComponent
  let Φ : (G.ConnectedComponent → F) →ₗ[F] (α → F) := LinearMap.funLeft F F G.connectedComponentMk
  have hinj : Function.Injective Φ :=
    LinearMap.funLeft_injective_of_surjective F F _ Quot.mk_surjective
  have hrange : LinearMap.range Φ = potentials F G := by
    ext x
    constructor
    · rintro ⟨y, rfl⟩ a b h
      exact congrArg y (ConnectedComponent.sound h.reachable)
    · intro hx
      exact ⟨ConnectedComponent.lift x fun v w p _ => potentials_eq_of_walk hx p, rfl⟩
  rw [← hrange, LinearMap.finrank_range_of_inj hinj, finrank_fintype_fun_eq_card,
    Nat.card_eq_fintype_card]

example : finrank ℚ (potentials ℚ (⊤ : SimpleGraph (Fin 2))) =
    Nat.card (⊤ : SimpleGraph (Fin 2)).ConnectedComponent := finrank_potentials ℚ _

end Potentials

variable {F ι κ : Type*} [Field F] [Fintype ι] [Fintype κ]

/-- The members of the flat class that vanish on `Ω` (`review/flat_spectrum.md`, PT3). -/
noncomputable def vanishing (Ω : Set ((ι → F) × (κ → F))) :
    Submodule F ((ι → F) × (κ → F) → F) where
  carrier := {f | f ∈ flatClass F ι κ ∧ ∀ c ∈ Ω, f c = 0}
  add_mem' hf hf' := ⟨Submodule.add_mem _ hf.1 hf'.1, fun c hc => by simp [hf.2 c hc, hf'.2 c hc]⟩
  zero_mem' := ⟨Submodule.zero_mem _, fun _ _ => rfl⟩
  smul_mem' t _ hf := ⟨Submodule.smul_mem _ t hf.1, fun c hc => by simp [hf.2 c hc]⟩

lemma mem_vanishing {Ω : Set ((ι → F) × (κ → F))} {f : (ι → F) × (κ → F) → F} :
    f ∈ vanishing Ω ↔ f ∈ flatClass F ι κ ∧ ∀ c ∈ Ω, f c = 0 := Iff.rfl

/-- On an edge of the graph of `Ω`, the pairing of `M` with the gain is the step of the potential
of a parameter that vanishes on `Ω`. -/
lemma pair_gain_eq {Ω : Set ((ι → F) × (κ → F))} {p : ((ι → F) ⊕ (κ → F) → F) × Matrix ι κ F}
    (hp : ∀ c ∈ Ω, flatEval F ι κ p c = 0) {a b : (ι → F) ⊕ (κ → F)}
    (h : (patternGraph Ω).Adj a b) : pair p.2 (gain a b) = p.1 b - p.1 a := by
  rcases a with u | v <;> rcases b with u' | v'
  · exact absurd h not_patternGraph_adj_inl_inl
  · have := hp (u, v') (patternGraph_adj_inl_inr.1 h)
    rw [flatEval_apply] at this
    simp only [gain, pair_vecMulVec]
    linear_combination this
  · have := hp (u', v) (patternGraph_adj_inr_inl.1 h)
    rw [flatEval_apply] at this
    simp only [gain, map_neg, pair_vecMulVec]
    linear_combination -this
  · exact absurd h not_patternGraph_adj_inr_inr

/-- **Telescoping** (`review/flat_spectrum.md`, PT3): along a walk, a parameter that vanishes on
`Ω` has `⟨M, twist w⟩ = x(b) − x(a)`. -/
lemma pair_twist_eq {Ω : Set ((ι → F) × (κ → F))} {p : ((ι → F) ⊕ (κ → F) → F) × Matrix ι κ F}
    (hp : ∀ c ∈ Ω, flatEval F ι κ p c = 0) {a b : (ι → F) ⊕ (κ → F)}
    (w : (patternGraph Ω).Walk a b) : pair p.2 (twist w) = p.1 b - p.1 a := by
  induction w with
  | nil => simp
  | cons h q ih =>
    rw [twist_cons, map_add, ih, pair_gain_eq hp h]
    ring

/-- A parameter that vanishes on `Ω` has its `M` orthogonal to the twists. -/
lemma snd_mem_ann {Ω : Set ((ι → F) × (κ → F))} {p : ((ι → F) ⊕ (κ → F) → F) × Matrix ι κ F}
    (hp : ∀ c ∈ Ω, flatEval F ι κ p c = 0) : p.2 ∈ ann (twistSpan Ω) := by
  refine mem_ann.2 fun Z hZ => ?_
  have hle : twistSpan Ω ≤ LinearMap.ker (pair p.2) := by
    refine Submodule.span_le.2 ?_
    rintro _ ⟨u, w, rfl⟩
    simp [pair_twist_eq hp w]
  exact hle hZ

/-- **Carrying the values along walks** (`review/flat_spectrum.md`, PT3): a matrix orthogonal to the
twists has a potential, read off walks from a root of each component. -/
lemma exists_potential {Ω : Set ((ι → F) × (κ → F))} {M : Matrix ι κ F}
    (hM : M ∈ ann (twistSpan Ω)) :
    ∃ x : (ι → F) ⊕ (κ → F) → F, ∀ c ∈ Ω, flatEval F ι κ (x, M) c = 0 := by
  classical
  let G := patternGraph Ω
  let root : (ι → F) ⊕ (κ → F) → (ι → F) ⊕ (κ → F) := fun a => (G.connectedComponentMk a).out
  have hreach (a : (ι → F) ⊕ (κ → F)) : G.Reachable (root a) a :=
    ConnectedComponent.exact (Quot.out_eq _)
  let w (a : (ι → F) ⊕ (κ → F)) : G.Walk (root a) a := (hreach a).some
  refine ⟨fun a => pair M (twist (w a)), ?_⟩
  rintro ⟨u, v⟩ hc
  have hab : G.Adj (.inl u) (.inr v) := patternGraph_adj_inl_inr.2 hc
  have hroot : root (.inl u) = root (.inr v) :=
    congrArg Quot.out (ConnectedComponent.sound hab.reachable)
  let C : G.Walk (root (.inr v)) (root (.inr v)) :=
    ((w (.inl u)).copy hroot rfl).append (Walk.cons hab (w (.inr v)).reverse)
  have hC : pair M (twist C) = 0 := mem_ann.1 hM _ (twist_mem_twistSpan C)
  simp only [C, twist_append, twist_copy, twist_cons, twist_reverse, map_add, map_neg] at hC
  simp only [gain, pair_vecMulVec] at hC
  rw [flatEval_apply]
  linear_combination hC

end Transformer.Grokking.FlatSpectrum.TablePart
