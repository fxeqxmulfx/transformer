module

public import Mathlib.Logic.Relation
public import Transformer.Phasor.EmbeddingInput

/-!
# Transport of a third hidden slot along learned partial maps

Source: “Triple transport” in `experiments/phasor_value_transport/README.md`.
Three distinct matches belonging to one context relate two ordered triples.
Taking equivalence closure combines evidence from different contexts. The
input consists of partial maps between hidden slots. The slots have no order,
arithmetic, or interpreted names.
-/

@[expose] public section

namespace Transformer.Phasor.TripleTransport

variable {I X Y Z : Type*}

/-- Two anchors and the slot whose image is to be inferred. -/
structure Triple (X : Type*) where
  first : X
  second : X
  third : X

def Triple.map (f : X → Y) (t : Triple X) : Triple Y :=
  ⟨f t.first, f t.second, f t.third⟩

def Triple.distinct (t : Triple X) : Prop :=
  t.first ≠ t.second ∧ t.first ≠ t.third ∧ t.second ≠ t.third

/-- All three correspondences come from the same partial map. -/
def Edge (P : I → X → Option X) (s t : Triple X) : Prop :=
  s.distinct ∧ t.distinct ∧ ∃ i,
    P i s.first = some t.first ∧ P i s.second = some t.second ∧ P i s.third = some t.third

/-- Hidden triples connected by finite chains of matches in either direction. -/
abbrev Connected (P : I → X → Option X) := Relation.EqvGen (Edge P)

/-- A quantity preserved by every trusted triple match is preserved along any
chain, including chains that switch between different partial maps. -/
theorem invariant {P : I → X → Option X} {F : Triple X → Z}
    (preserves : ∀ s t, Edge P s t → F s = F t) {s t : Triple X}
    (path : Connected P s t) : F s = F t := by
  induction path with
  | rel x y h => exact preserves x y h
  | refl => rfl
  | symm _ _ _ ih => exact ih.symm
  | trans _ _ _ _ _ ih₁ ih₂ => exact ih₁.trans ih₂

/-- If the values of a preserved invariant distinguish the third output
when both output anchors are fixed, every connected candidate agrees with
the true image. No completeness assumption on the observed graph is needed. -/
theorem third_eq {P : I → X → Option X} {F : Triple X → Z}
    (preserves : ∀ s t, Edge P s t → F s = F t)
    (s : Triple X) (a b actual candidate : X)
    (separates : Function.Injective fun y => F ⟨a, b, y⟩)
    (completion : F s = F ⟨a, b, actual⟩)
    (path : Connected P s ⟨a, b, candidate⟩) : candidate = actual :=
  separates ((invariant preserves path).symm.trans completion)

/-- Embedding slot names injectively into a larger dictionary preserves
every inferred connection when its learned maps agree on the old slots. -/
theorem rename {P : I → X → Option X} {Q : I → Y → Option Y}
    (f : X → Y) (inj : Function.Injective f)
    (agrees : ∀ i x, Q i (f x) = (P i x).map f)
    {s t : Triple X} (path : Connected P s t) : Connected Q (s.map f) (t.map f) := by
  induction path with
  | rel x y h =>
    obtain ⟨hx, hy, i, h₁, h₂, h₃⟩ := h
    apply Relation.EqvGen.rel
    exact ⟨⟨inj.ne hx.1, inj.ne hx.2.1, inj.ne hx.2.2⟩,
      ⟨inj.ne hy.1, inj.ne hy.2.1, inj.ne hy.2.2⟩, i,
      by simp only [Triple.map, agrees, h₁, Option.map_some],
      by simp only [Triple.map, agrees, h₂, Option.map_some],
      by simp only [Triple.map, agrees, h₃, Option.map_some]⟩
  | refl => exact Relation.EqvGen.refl _
  | symm _ _ _ ih => exact Relation.EqvGen.symm _ _ ih
  | trans _ _ _ _ _ ih₁ ih₂ => exact Relation.EqvGen.trans _ _ _ ih₁ ih₂

private def identityBank (_ : Unit) (x : Fin 4) : Option (Fin 4) := some x

private theorem identity_preserves (s t : Triple (Fin 4)) (h : Edge identityBank s t) :
    s.third = t.third := by
  obtain ⟨_, _, i, _, _, hm⟩ := h
  exact Option.some.inj hm

private theorem sampleEdge : Edge identityBank ⟨0, 1, 3⟩ ⟨0, 1, 3⟩ :=
  ⟨by unfold Triple.distinct; decide, by unfold Triple.distinct; decide, (), rfl, rfl, rfl⟩

example : (⟨0, 1, 3⟩ : Triple (Fin 4)).third = 3 :=
  invariant (F := Triple.third) identity_preserves (.rel _ _ sampleEdge)

example (y : Fin 4) (path : Connected identityBank ⟨0, 1, 3⟩ ⟨0, 1, y⟩) : y = 3 :=
  third_eq (F := Triple.third) identity_preserves _ 0 1 3 y (fun _ _ h => h) rfl path

example : Connected identityBank ((Triple.mk 0 1 3).map id) ((Triple.mk 0 1 3).map id) :=
  rename id (fun _ _ h => h) (P := identityBank) (fun _ _ => rfl) (.rel _ _ sampleEdge)

section Confidence

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- A damped cosine above one over the square root of two. The squared
form also rejects a zero prototype without requiring division. -/
def trusted (ε : ℝ) (x k : E) : Prop :=
  0 < inner ℝ x k ∧ (‖x‖ ^ 2 + ε) * ‖k‖ ^ 2 < 2 * (inner ℝ x k) ^ 2

/-- A trusted value has squared norm strictly above the damping energy.
Consequently, for positive damping an open neighborhood of the unwritten
value has no outgoing trusted edge, in every direction. -/
theorem energy_of_trusted {ε : ℝ} {x k : E} (h : trusted ε x k) : ε < ‖x‖ ^ 2 := by
  have cs := (sq_le_sq₀ (abs_nonneg _) (mul_nonneg (norm_nonneg x) (norm_nonneg k))).2
    (abs_real_inner_le_norm x k)
  rw [sq_abs, mul_pow] at cs
  by_contra! he
  have bound := mul_le_mul_of_nonneg_right he (sq_nonneg ‖k‖)
  nlinarith [h.2]

example : (1 / 4 : ℝ) < ‖(1 : ℝ)‖ ^ 2 :=
  energy_of_trusted (k := (1 : ℝ)) (by norm_num [trusted, real_inner_self_eq_norm_sq])

end Confidence

/-- A learned gate reads the strict-prefix hidden vector. Its four values
weight the current contextual feature, a direct association, a row prototype
and the transported prototype, all in the same dense output space. -/
def mix {D : ℕ} (gate : EucSpace D → Fin 4 → ℝ) (h : EucSpace D)
    (reads : Fin 4 → EucSpace D) : EucSpace D :=
  ∑ i, gate h i • reads i

/-- At fixed routing and prefix, the adjoint for each value is its own
learned gate times the incoming dense cotangent. -/
theorem mix_pairing {D : ℕ} (gate : EucSpace D → Fin 4 → ℝ) (h v : EucSpace D)
    (reads : Fin 4 → EucSpace D) :
    inner ℝ (mix gate h reads) v = ∑ i, inner ℝ (reads i) (gate h i • v) := by
  simp only [mix, sum_inner, real_inner_smul_left, real_inner_smul_right]

/-- Source: the prototype addresses in the experiment's “Triple transport” section.
Carry the preceding dense key's winning slot as the model scans one sequence.
An ambiguous or missing winner at either end gives no direct address. -/
noncomputable def scanAddresses {D : ℕ} (nearest : EucSpace D → Option X) (scale : ℝ) :
    Option X → List (EucSpace D × EucSpace D) → List (Option (X × X))
  | _, [] => []
  | prior, (x, h) :: rest =>
      (prior.bind fun a => (nearest x).map (a, ·)) ::
        scanAddresses nearest scale (nearest (x + scale • h)) rest

/-- Source: “Triple transport”, parallel prototype lookup. Each cell can search
its own input and the preceding position's key independently. Zipping these
searches agrees with the carried scan for every dense input, contextual state,
and tie decision. At a sequence boundary the incoming address is `none`. -/
theorem scanAddresses_eq_zip {D : ℕ} (nearest : EucSpace D → Option X) (scale : ℝ)
    (prior : Option X) (rows : List (EucSpace D × EucSpace D)) :
    scanAddresses nearest scale prior rows =
      List.zipWith (fun p x => p.bind fun a => (nearest x).map (a, ·))
        (prior :: rows.map (fun r => nearest (r.1 + scale • r.2))) (rows.map Prod.fst) := by
  induction rows generalizing prior with
  | nil => rfl
  | cons row rest ih =>
    obtain ⟨x, h⟩ := row
    simp only [scanAddresses, List.map_cons, List.zipWith_cons_cons]
    rw [ih]

end Transformer.Phasor.TripleTransport
