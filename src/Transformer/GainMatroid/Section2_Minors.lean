module

public import Transformer.GainMatroid.Section2_Normalized

/-!
# Deletion, contraction and Proposition 2.3

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.3, paragraph *Minors and induced gain functions*.
`G.delete D` keeps the vertices and the edges outside `D`. `G.contract F` identifies the vertices
of each component of `F` and keeps the edges outside `F`. For a set `F` of links this is the
source's contraction, and a link minor is obtained by contracting a forest.

`IsInducedGain G F γ ι` says that `ι` is the restriction to the edges outside `F` of a switching of
`γ` that is the identity on `F`. The source normalizes on a maximal forest containing `F`, which is
one such switching (`IsInducedGain.of_normalized`), and one exists for every forest
(`exists_isInducedGain`, by Proposition 2.1).

Proposition 2.3 (`switchEquiv_iff_contract`, `switchScaleEquiv_iff_contract`) says that two gain
functions are equivalent exactly when their induced gain functions on `G / F` are. The source takes
`F` to be a forest and `Γ` to be abelian. The proof below uses neither: it only needs both gain
functions to be normalized on `F`, because a switching function between two such functions is
constant on the components of `F`, so it descends to `G / F`, and every switching function on
`G / F` pulls back. It therefore also holds for every choice of normalization, including two
different maximal forests for `φ` and `ψ`.
-/

@[expose] public section

namespace Transformer.GainMatroid

universe uV uE

variable {V : Type uV} {E : Type uE} {Γ : Type*}

namespace OrientedGraph

variable {G : OrientedGraph V E}

/-- The **deletion** `G \ D` (§2.3): the same vertices and the edges outside `D`. -/
def delete (G : OrientedGraph V E) (D : Set E) : OrientedGraph V ↥Dᶜ :=
  ⟨fun e => G.tail e, fun e => G.head e⟩

/-- The **contraction** `G / F` (§2.3): the vertices are the components of `F`, and the edges are
those outside `F`, with the classes of their ends. -/
def contract (G : OrientedGraph V E) (F : Set E) :
    OrientedGraph (Quotient (G.reachableSetoid F)) ↥Fᶜ :=
  ⟨fun e => Quotient.mk _ (G.tail e), fun e => Quotient.mk _ (G.head e)⟩

/-- The two ends of an edge of `F` lie in one vertex of `G / F`. -/
theorem contract_class_eq {F : Set E} {f : E} (hf : f ∈ F) :
    Quotient.mk (G.reachableSetoid F) (G.tail f) = Quotient.mk _ (G.head f) :=
  Quotient.sound ((reachableIn_refl F _).step hf (Or.inl ⟨rfl, rfl⟩))

/-- `ι` is an **induced gain function** on `G / F` (§2.3): the restriction to the edges outside `F`
of a switching `γ^η` of `γ` that is the identity on `F`. -/
@[to_additive IsAddInducedGain
  /-- `ι` is an additive **induced gain function** on `G / F` (§2.3). -/]
def IsInducedGain [Group Γ] (G : OrientedGraph V E) (F : Set E) (γ : E → Γ) (ι : ↥Fᶜ → Γ) : Prop :=
  ∃ η : V → Γ, (∀ f ∈ F, switch G η γ f = 1) ∧ ∀ e : ↥Fᶜ, ι e = switch G η γ e

/-- The source's induced gain function: normalize on a forest `F'` containing `F` (§2.3). -/
@[to_additive
  /-- The source's additive induced gain function: normalize on a forest containing `F`. -/]
theorem IsInducedGain.of_normalized [Group Γ] {F F' : Set E} (hsub : F ⊆ F') {γ : E → Γ}
    {η : V → Γ} (h : ∀ f ∈ F', switch G η γ f = 1) :
    G.IsInducedGain F γ fun e => switch G η γ e :=
  ⟨η, fun f hf => h f (hsub hf), fun _ => rfl⟩

/-- A gain function already normalized on `F` induces its own restriction. -/
@[to_additive
  /-- An additive gain function already normalized on `F` induces its own restriction. -/]
theorem IsInducedGain.self [Group Γ] {F : Set E} {γ : E → Γ} (h : ∀ f ∈ F, γ f = 1) :
    G.IsInducedGain F γ fun e => γ e :=
  ⟨1, fun f hf => by rw [switch_one]; exact h f hf, fun e => by rw [switch_one]⟩

/-- Every gain function has an induced gain function on the contraction of a forest. -/
@[to_additive exists_isAddInducedGain
  /-- Every additive gain function has an induced gain function on the contraction of a forest. -/]
theorem exists_isInducedGain [Group Γ] {F : Set E} (hF : G.IsForest F) (γ : E → Γ) :
    ∃ ι, G.IsInducedGain F γ ι := by
  obtain ⟨η, hη⟩ := exists_switch_eq_one_of_isForest hF γ
  exact ⟨fun e => switch G η γ e, η, hη, fun _ => rfl⟩

/-- Proposition 2.3 for gain functions already normalized on `F`. -/
@[to_additive addSwitchEquiv_contract_of_normalized
  /-- Proposition 2.3 for additive gain functions already normalized on `F`. -/]
theorem switchEquiv_contract_of_normalized [Group Γ] {F : Set E} {φ ψ : E → Γ}
    (hφ : ∀ f ∈ F, φ f = 1) (hψ : ∀ f ∈ F, ψ f = 1) :
    SwitchEquiv G φ ψ ↔
      SwitchEquiv (G.contract F) (fun e : ↥Fᶜ => φ e) (fun e : ↥Fᶜ => ψ e) := by
  constructor
  · rintro ⟨ξ, hξ⟩
    have hconst : ∀ f ∈ F, ξ (G.tail f) = ξ (G.head f) := by
      intro f hf
      have h₀ := congrFun hξ f
      rw [switch_apply, hφ f hf, hψ f hf, mul_one, inv_mul_eq_one] at h₀
      exact h₀
    refine ⟨Quotient.lift ξ fun u v huv => ?_, funext fun e => congrFun hξ e⟩
    obtain ⟨p, hp⟩ := huv
    exact Walk.eq_of_forall_mem_edges ξ p fun e he => hconst e (hp e he)
  · rintro ⟨ξ, hξ⟩
    refine ⟨fun v => ξ (Quotient.mk _ v), funext fun e => ?_⟩
    by_cases he : e ∈ F
    · simp [switch_apply, hφ e he, hψ e he, contract_class_eq he]
    · exact congrFun hξ ⟨e, he⟩

/-- **Proposition 2.3** (§2.3), switching form. Two gain functions are switching equivalent exactly
when their induced gain functions on `G / F` are. The source's hypotheses that `F` is a forest and
`Γ` is abelian are not used; see the module docstring. -/
@[to_additive addSwitchEquiv_iff_contract
  /-- **Proposition 2.3** (§2.3), switching form, for an additive group. -/]
theorem switchEquiv_iff_contract [Group Γ] {F : Set E} {φ ψ : E → Γ} {φ' ψ' : ↥Fᶜ → Γ}
    (hφ : G.IsInducedGain F φ φ') (hψ : G.IsInducedGain F ψ ψ') :
    SwitchEquiv G φ ψ ↔ SwitchEquiv (G.contract F) φ' ψ' := by
  obtain ⟨η₁, h₁, hφ'⟩ := hφ
  obtain ⟨η₂, h₂, hψ'⟩ := hψ
  obtain rfl : φ' = fun e : ↥Fᶜ => switch G η₁ φ e := funext hφ'
  obtain rfl : ψ' = fun e : ↥Fᶜ => switch G η₂ ψ e := funext hψ'
  refine Iff.trans ?_ (switchEquiv_contract_of_normalized h₁ h₂)
  have r := switchEquiv_equivalence (Γ := Γ) G
  have a : SwitchEquiv G φ (switch G η₁ φ) := ⟨η₁, rfl⟩
  have b : SwitchEquiv G ψ (switch G η₂ ψ) := ⟨η₂, rfl⟩
  exact ⟨fun h => r.trans (r.symm a) (r.trans h b), fun h => r.trans a (r.trans h (r.symm b))⟩

/-- Proposition 2.3 for additive gain functions normalized on `F`, with scaling. -/
theorem switchScaleEquiv_contract_of_normalized {K : Type*} [Field K] {F : Set E} {φ ψ : E → K}
    (hφ : ∀ f ∈ F, φ f = 0) (hψ : ∀ f ∈ F, ψ f = 0) :
    SwitchScaleEquiv G φ ψ ↔
      SwitchScaleEquiv (G.contract F) (fun e : ↥Fᶜ => φ e) (fun e : ↥Fᶜ => ψ e) := by
  constructor
  · rintro ⟨ξ, a, hξ⟩
    have hconst : ∀ f ∈ F, ξ (G.tail f) = ξ (G.head f) := by
      intro f hf
      have h₀ := congrFun hξ f
      simp only [Pi.smul_apply, addSwitch_apply, smul_eq_mul, hφ f hf, hψ f hf, add_zero] at h₀
      exact neg_add_eq_zero.1 ((mul_eq_zero.1 h₀).resolve_left a.ne_zero)
    refine ⟨Quotient.lift ξ fun u v huv => ?_, a, funext fun e => congrFun hξ e⟩
    obtain ⟨p, hp⟩ := huv
    exact Walk.eq_of_forall_mem_edges ξ p fun e he => hconst e (hp e he)
  · rintro ⟨ξ, a, hξ⟩
    refine ⟨fun v => ξ (Quotient.mk _ v), a, funext fun e => ?_⟩
    by_cases he : e ∈ F
    · simp [addSwitch_apply, hφ e he, hψ e he, contract_class_eq he]
    · exact congrFun hξ ⟨e, he⟩

/-- **Proposition 2.3** (§2.3), switching-and-scaling form, for the additive group of a field. The
source's hypothesis that `F` is a forest is not used; see the module docstring. -/
theorem switchScaleEquiv_iff_contract {K : Type*} [Field K] {F : Set E} {φ ψ : E → K}
    {φ' ψ' : ↥Fᶜ → K} (hφ : G.IsAddInducedGain F φ φ') (hψ : G.IsAddInducedGain F ψ ψ') :
    SwitchScaleEquiv G φ ψ ↔ SwitchScaleEquiv (G.contract F) φ' ψ' := by
  obtain ⟨η₁, h₁, hφ'⟩ := hφ
  obtain ⟨η₂, h₂, hψ'⟩ := hψ
  obtain rfl : φ' = fun e : ↥Fᶜ => addSwitch G η₁ φ e := funext hφ'
  obtain rfl : ψ' = fun e : ↥Fᶜ => addSwitch G η₂ ψ e := funext hψ'
  refine Iff.trans ?_ (switchScaleEquiv_contract_of_normalized h₁ h₂)
  have r := switchScaleEquiv_equivalence (F := K) G
  have a : SwitchScaleEquiv G φ (addSwitch G η₁ φ) := ⟨η₁, 1, by simp⟩
  have b : SwitchScaleEquiv G ψ (addSwitch G η₂ ψ) := ⟨η₂, 1, by simp⟩
  exact ⟨fun h => r.trans (r.symm a) (r.trans h b), fun h => r.trans a (r.trans h (r.symm b))⟩

/-- Contracting one link of `pumpkin` turns the other two into loops. -/
theorem pumpkin_contract_loop (e : ↥({0} : Set (Fin 3))ᶜ) :
    (pumpkin.contract {0}).tail e = (pumpkin.contract {0}).head e :=
  contract_class_eq (G := pumpkin) (f := 0) (Set.mem_singleton 0)

example : Quotient.mk (pumpkin.reachableSetoid {0}) (pumpkin.tail 0) =
    Quotient.mk _ (pumpkin.head 0) :=
  contract_class_eq (Set.mem_singleton 0)

example [Group Γ] (γ : Fin 3 → Γ) : ∃ ι, pumpkin.IsInducedGain {0} γ ι :=
  exists_isInducedGain pumpkin_isForest γ

example (γ : Fin 3 → ℚ) : ∃ ι, pumpkin.IsAddInducedGain {0} γ ι :=
  exists_isAddInducedGain pumpkin_isForest γ

example [Group Γ] (g h : Γ) : pumpkin.IsInducedGain {0} ![1, g, h]
    fun e => switch pumpkin 1 ![1, g, h] e :=
  IsInducedGain.of_normalized (F' := {0}) subset_rfl (by simp [switch_one])

example [Group Γ] (g h g' h' : Γ) :
    SwitchEquiv pumpkin ![1, g, h] ![1, g', h'] ↔ SwitchEquiv (pumpkin.contract {0})
      (fun e => (![1, g, h] : Fin 3 → Γ) e) (fun e => (![1, g', h'] : Fin 3 → Γ) e) :=
  switchEquiv_iff_contract (IsInducedGain.self (by simp)) (IsInducedGain.self (by simp))

example (g h g' h' : ℚ) :
    SwitchScaleEquiv pumpkin ![0, g, h] ![0, g', h'] ↔ SwitchScaleEquiv (pumpkin.contract {0})
      (fun e => (![0, g, h] : Fin 3 → ℚ) e) (fun e => (![0, g', h'] : Fin 3 → ℚ) e) :=
  switchScaleEquiv_iff_contract (IsAddInducedGain.self (by simp)) (IsAddInducedGain.self (by simp))

end OrientedGraph

end Transformer.GainMatroid
