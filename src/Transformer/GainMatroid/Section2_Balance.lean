module

public import Transformer.GainMatroid.Section2_Cycles

/-!
# Gains of walks

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.3. The gain of a walk is the product of the gains
of its steps, a step along a link against its orientation contributing the inverse. On a loop the
source's two ordered triples coincide, so a loop has the single gain `γ(e)`.

The source chooses a closed Eulerian walk `W_C` around each cycle `C`, calls `C` balanced when
`γ(W_C)` is the identity, and observes that any two such walks have conjugate gains. That holds
only up to inversion, since reversing a walk inverts its gain. What the definition needs is
`IsCycleWalk.gain_eq_one_iff`: a closed walk around a cycle has gain one exactly when some switching
function makes every edge of the cycle have gain one, which is a condition on the edge set alone.
`gain_eq_one_iff_of_edgeSet_eq` is the resulting independence of the walk, for any group.
-/

@[expose] public section

namespace Transformer.GainMatroid

universe uV uE

variable {V : Type uV} {E : Type uE} {Γ : Type*} [DecidableEq V]

namespace OrientedGraph

/-- The gain of the step along `e` starting at its end `u` (§2.3): `γ(e)` from the tail, and
`γ(e)⁻¹` from the head of a link. -/
@[to_additive addDart
  /-- The additive gain of the step along `e` starting at its end `u` (§2.3): `γ(e)` from the tail,
  and `-γ(e)` from the head of a link. -/]
def dart [Group Γ] (G : OrientedGraph V E) (γ : E → Γ) (e : E) (u : V) : Γ :=
  if G.tail e = u then γ e else (γ e)⁻¹

@[to_additive addDart_of_tail]
theorem dart_of_tail [Group Γ] {G : OrientedGraph V E} (γ : E → Γ) {e : E} {u : V}
    (h : G.tail e = u) : G.dart γ e u = γ e := by
  simp [dart, h]

@[to_additive addDart_of_ne]
theorem dart_of_ne [Group Γ] {G : OrientedGraph V E} (γ : E → Γ) {e : E} {u : V}
    (h : G.tail e ≠ u) : G.dart γ e u = (γ e)⁻¹ := by
  simp [dart, h]

/-- For a link the two directions have inverse gains, the source's `γ(e⁻¹) = γ(e)⁻¹`. -/
@[to_additive addDart_symm
  /-- For a link the two directions have opposite gains, the source's `γ(e⁻¹) = -γ(e)`. -/]
theorem dart_symm [Group Γ] {G : OrientedGraph V E} (γ : E → Γ) {e : E} {u v : V}
    (h : G.Joins e u v) (huv : u ≠ v) : G.dart γ e v = (G.dart γ e u)⁻¹ := by
  rcases h with ⟨h₁, -⟩ | ⟨h₁, -⟩
  · rw [dart_of_ne γ (h₁ ▸ huv), dart_of_tail γ h₁]
  · rw [dart_of_tail γ h₁, dart_of_ne γ (h₁ ▸ huv.symm), inv_inv]

/-- A switching function makes a step's edge have gain one exactly when it follows the step. -/
@[to_additive addSwitch_eq_zero_iff
  /-- A switching function makes a step's edge have gain zero exactly when it follows the step. -/]
theorem switch_eq_one_iff [Group Γ] {G : OrientedGraph V E} {γ : E → Γ} (η : V → Γ) {e : E}
    {u v : V} (h : G.Joins e u v) :
    switch G η γ e = 1 ↔ η v = (G.dart γ e u)⁻¹ * η u := by
  rw [switch_apply]
  by_cases ht : G.tail e = u
  · have hh : G.head e = v := by
      rcases h with ⟨-, h₂⟩ | ⟨h₁, h₂⟩
      · exact h₂
      · rw [h₂, ← ht, h₁]
    rw [dart_of_tail γ ht, ht, hh, mul_assoc, inv_mul_eq_one, eq_inv_mul_iff_mul_eq, eq_comm]
  · obtain ⟨h₁, h₂⟩ : G.tail e = v ∧ G.head e = u := h.resolve_left fun hc => ht hc.1
    rw [dart_of_ne γ ht, h₁, h₂, mul_assoc, inv_mul_eq_one, inv_inv]

namespace Walk

variable {G : OrientedGraph V E}

/-- The steps of a walk: each edge with the vertex it starts from. -/
def steps : ∀ {u v : V}, G.Walk u v → List (E × V)
  | _, _, nil _ => []
  | u, _, cons e _ p => (e, u) :: p.steps

/-- The **gain** `γ(W)` of a walk (§2.3). -/
@[to_additive addGain /-- The additive **gain** `γ(W)` of a walk (§2.3). -/]
def gain [Group Γ] (γ : E → Γ) {u v : V} (p : G.Walk u v) : Γ :=
  (p.steps.map fun s => G.dart γ s.1 s.2).prod

@[to_additive (attr := simp) addGain_nil]
theorem gain_nil [Group Γ] (γ : E → Γ) (u : V) : (nil u : G.Walk u u).gain γ = 1 := rfl

@[to_additive addGain_cons]
theorem gain_cons [Group Γ] (γ : E → Γ) {u v w : V} (e : E) (h : G.Joins e u v) (p : G.Walk v w) :
    (cons e h p).gain γ = G.dart γ e u * p.gain γ := by
  simp [gain, steps]

@[to_additive addGain_append]
theorem gain_append [Group Γ] (γ : E → Γ) {u v w : V} (p : G.Walk u v) (q : G.Walk v w) :
    (p.append q).gain γ = p.gain γ * q.gain γ := by
  induction p with
  | nil => simp [append]
  | cons e h p ih => simp [append, gain_cons, ih, mul_assoc]

/-- A path traversed backwards has the inverse gain. -/
@[to_additive addGain_reverse /-- A path traversed backwards has the opposite gain. -/]
theorem gain_reverse [Group Γ] (γ : E → Γ) {u v : V} {p : G.Walk u v} (hp : p.IsPath) :
    p.reverse.gain γ = (p.gain γ)⁻¹ := by
  induction p with
  | nil => simp [reverse]
  | @cons a b c e h p ih =>
    have hcons : a ∉ p.support ∧ p.support.Nodup := by
      simpa [IsPath, support] using hp
    have hab : a ≠ b := fun hab => hcons.1 (hab ▸ p.start_mem_support)
    simp [reverse, gain_append, gain_cons, ih hcons.2, dart_symm γ h hab]

/-- Along a walk whose edges are all trivialized by `η`, the gain telescopes. -/
@[to_additive addGain_eq_of_potential
  /-- Along a walk whose edges are all trivialized by `η`, the additive gain telescopes. -/]
theorem gain_eq_of_potential [Group Γ] {γ : E → Γ} (η : V → Γ) {u v : V} (p : G.Walk u v)
    (h : ∀ e ∈ p.edges, switch G η γ e = 1) : p.gain γ = η u * (η v)⁻¹ := by
  induction p with
  | nil => simp
  | @cons a b c e hj p ih =>
    have he := (switch_eq_one_iff η hj).1 (h e List.mem_cons_self)
    rw [gain_cons, ih fun f hf => h f (List.mem_cons_of_mem _ hf), he]
    simp [mul_assoc]

/-- On a path, a switching function with any prescribed value at the start trivializes every
edge; its value at the end is then determined by the gain of the path. -/
@[to_additive exists_addPotential_of_isPath
  /-- On a path, a switching function with any prescribed value at the start trivializes every
  edge; its value at the end is then determined by the additive gain of the path. -/]
theorem exists_potential_of_isPath [Group Γ] (γ : E → Γ) {u v : V} (p : G.Walk u v)
    (hp : p.IsPath) (a : Γ) :
    ∃ η : V → Γ, η u = a ∧ η v = (p.gain γ)⁻¹ * a ∧ ∀ e ∈ p.edges, switch G η γ e = 1 := by
  induction p generalizing a with
  | nil u => exact ⟨fun _ => a, rfl, by simp, by simp [edges]⟩
  | @cons x y z e h p ih =>
    obtain ⟨hx, hnd⟩ : x ∉ p.support ∧ p.support.Nodup := by simpa [IsPath, support] using hp
    obtain ⟨η, h₁, h₂, h₃⟩ := ih hnd ((G.dart γ e x)⁻¹ * a)
    have hyx : y ≠ x := fun hc => hx (hc ▸ p.start_mem_support)
    have hzx : z ≠ x := fun hc => hx (hc ▸ p.end_mem_support)
    refine ⟨Function.update η x a, by simp, ?_, fun f hf => ?_⟩
    · rw [Function.update_of_ne hzx, h₂, gain_cons, mul_inv_rev, mul_assoc]
    · simp only [edges, List.mem_cons] at hf
      rcases hf with rfl | hf
      · refine (switch_eq_one_iff _ h).2 ?_
        rw [Function.update_of_ne hyx, Function.update_self, h₁]
      · obtain ⟨ht, hh⟩ := p.ends_mem_support hf
        have hne₁ : G.tail f ≠ x := fun hc => hx (hc ▸ ht)
        have hne₂ : G.head f ≠ x := fun hc => hx (hc ▸ hh)
        have := h₃ f hf
        rw [switch_apply] at this ⊢
        rwa [Function.update_of_ne hne₁, Function.update_of_ne hne₂]

/-- A closed walk around a cycle has gain one exactly when some switching function makes every
edge of the cycle have gain one. -/
@[to_additive addGain_eq_zero_iff
  /-- A closed walk around a cycle has additive gain zero exactly when some switching function
  makes every edge of the cycle have gain zero. -/]
theorem IsCycleWalk.gain_eq_one_iff [Group Γ] (γ : E → Γ) {u : V} {W : G.Walk u u}
    (hW : W.IsCycleWalk) :
    W.gain γ = 1 ↔ ∃ η : V → Γ, ∀ e ∈ W.edges, switch G η γ e = 1 := by
  refine ⟨fun hg => ?_, fun ⟨η, h⟩ => by rw [gain_eq_of_potential η W h, mul_inv_cancel]⟩
  cases W with
  | nil => exact absurd rfl hW.edges_ne_nil
  | @cons _ b _ e hj p =>
    have hp : p.IsPath := hW.support_tail_nodup
    obtain ⟨η, h₁, h₂, h₃⟩ := exists_potential_of_isPath γ p hp 1
    have hpg : (p.gain γ)⁻¹ = G.dart γ e u := by
      rw [gain_cons] at hg
      rw [eq_inv_of_mul_eq_one_right hg, inv_inv]
    refine ⟨η, fun f hf => ?_⟩
    simp only [edges, List.mem_cons] at hf
    rcases hf with rfl | hf
    · refine (switch_eq_one_iff η hj).2 ?_
      rw [h₁, h₂, hpg, mul_one, inv_mul_cancel]
    · exact h₃ f hf

/-- Whether a closed walk around a cycle has gain one depends only on the cycle: the source's
well-definedness of `B_γ` (§2.3), for any group. -/
@[to_additive addGain_eq_zero_iff_of_edgeSet_eq
  /-- Whether a closed walk around a cycle has additive gain zero depends only on the cycle. -/]
theorem gain_eq_one_iff_of_edgeSet_eq [Group Γ] (γ : E → Γ) {u v : V} {W : G.Walk u u}
    {W' : G.Walk v v} (hW : W.IsCycleWalk) (hW' : W'.IsCycleWalk) (h : W.edgeSet = W'.edgeSet) :
    W.gain γ = 1 ↔ W'.gain γ = 1 := by
  have hmem : ∀ e, e ∈ W.edges ↔ e ∈ W'.edges := fun e => Set.ext_iff.1 h e
  rw [hW.gain_eq_one_iff, hW'.gain_eq_one_iff]
  simp only [hmem]

end Walk

end OrientedGraph

end Transformer.GainMatroid
