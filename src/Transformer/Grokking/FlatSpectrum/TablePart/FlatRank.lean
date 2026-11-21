module

public import Transformer.Grokking.FlatSpectrum.TablePart.FlatCount

/-!
# PT3: the theorem

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT3, **Theorem**. For a
part `Ω` of `F^k × F^l`, with `c` components and a span of twists of dimension `s`:

* the members of the flat class that vanish on `Ω` have dimension `(c − 1) + (k l − s)`
  (`finrank_vanishing`);
* `Ω` puts `n − c + s` independent conditions on the class, for `n = q^k + q^l`
  (`finrank_map_restrict_flatClass`);
* `Ω` determines the class exactly when its graph is connected and its twists span all `k × l`
  matrices (`determines_flatClass_iff`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.TablePart

open Matrix Module SimpleGraph

variable {F ι κ : Type*} [Field F] [Fintype ι] [Fintype κ]

/-- The parameters `(x, M)` whose function vanishes on `Ω`. -/
noncomputable def vanishingParams (Ω : Set ((ι → F) × (κ → F))) :
    Submodule F (((ι → F) ⊕ (κ → F) → F) × Matrix ι κ F) :=
  (vanishing Ω).comap (flatEval F ι κ)

lemma mem_vanishingParams {Ω : Set ((ι → F) × (κ → F))}
    {p : ((ι → F) ⊕ (κ → F) → F) × Matrix ι κ F} :
    p ∈ vanishingParams Ω ↔ ∀ c ∈ Ω, flatEval F ι κ p c = 0 := by
  simp only [vanishingParams, Submodule.mem_comap, mem_vanishing]
  exact ⟨fun h => h.2, fun h => ⟨by rw [← range_flatEval]; exact LinearMap.mem_range_self _ p, h⟩⟩

/-- With `M = 0`, a parameter that vanishes on `Ω` is a potential. -/
lemma mem_potentials_of_snd_eq_zero {Ω : Set ((ι → F) × (κ → F))}
    {p : ((ι → F) ⊕ (κ → F) → F) × Matrix ι κ F} (hp : ∀ c ∈ Ω, flatEval F ι κ p c = 0)
    (h0 : p.2 = 0) : p.1 ∈ potentials F (patternGraph Ω) := by
  intro a b hab
  have h := pair_gain_eq hp hab
  rw [h0, map_zero, LinearMap.zero_apply] at h
  exact (sub_eq_zero.1 h.symm).symm

/-- A potential with `M = 0` vanishes on `Ω`. -/
lemma mem_vanishingParams_of_potentials {Ω : Set ((ι → F) × (κ → F))}
    {y : (ι → F) ⊕ (κ → F) → F} (hy : y ∈ potentials F (patternGraph Ω)) :
    ((y, 0) : ((ι → F) ⊕ (κ → F) → F) × Matrix ι κ F) ∈ vanishingParams Ω := by
  refine mem_vanishingParams.2 ?_
  rintro ⟨u, v⟩ hc
  simp [hy (.inl u) (.inr v) (patternGraph_adj_inl_inr.2 hc)]

example : (0 : (Fin 1 → ℚ) ⊕ (Fin 1 → ℚ) → ℚ) ∈
    potentials ℚ (patternGraph (Set.univ : Set ((Fin 1 → ℚ) × (Fin 1 → ℚ)))) :=
  mem_potentials_of_snd_eq_zero (p := 0) (fun c _ => by simp) rfl

example : ((0, 0) : ((Fin 1 → ℚ) ⊕ (Fin 1 → ℚ) → ℚ) × Matrix (Fin 1) (Fin 1) ℚ) ∈
    vanishingParams (Set.univ : Set ((Fin 1 → ℚ) × (Fin 1 → ℚ))) :=
  mem_vanishingParams_of_potentials (Submodule.zero_mem _)

variable [Fintype F] [DecidableEq ι] [DecidableEq κ]

/-- Dividing the parameters by the kernel of the parametrization removes one dimension. -/
lemma finrank_vanishing_add_one (Ω : Set ((ι → F) × (κ → F))) :
    finrank F (vanishing Ω) + 1 = finrank F (vanishingParams Ω) := by
  have hle : LinearMap.ker (flatEval F ι κ) ≤ vanishingParams Ω := fun p hp =>
    mem_vanishingParams.2 fun c _ => by rw [LinearMap.mem_ker.1 hp]; rfl
  have h := LinearMap.finrank_range_add_finrank_ker
    ((flatEval F ι κ).domRestrict (vanishingParams Ω))
  rw [LinearMap.range_domRestrict, LinearMap.ker_domRestrict,
    (Submodule.comapSubtypeEquivOfLe hle).finrank_eq, finrank_ker_flatEval] at h
  have hmap : (vanishingParams Ω).map (flatEval F ι κ) = vanishing Ω :=
    Submodule.map_comap_eq_of_le (by rw [range_flatEval]; exact fun f hf => hf.1)
  rwa [hmap] at h

omit [DecidableEq ι] [DecidableEq κ] in
/-- The parameters that vanish on `Ω` have dimension `(k l − s) + c`: a matrix orthogonal to the
twists, and one free constant per component. -/
lemma finrank_vanishingParams (Ω : Set ((ι → F) × (κ → F))) :
    finrank F (vanishingParams Ω) =
      finrank F (ann (twistSpan Ω)) + Nat.card (patternGraph Ω).ConnectedComponent := by
  let π := (LinearMap.snd F ((ι → F) ⊕ (κ → F) → F) (Matrix ι κ F)).domRestrict
    (vanishingParams Ω)
  have hrange : LinearMap.range π = ann (twistSpan Ω) := by
    rw [LinearMap.range_domRestrict]
    ext M
    constructor
    · rintro ⟨p, hp, rfl⟩
      exact snd_mem_ann (mem_vanishingParams.1 hp)
    · intro hM
      obtain ⟨x, hx⟩ := exists_potential hM
      exact ⟨(x, M), mem_vanishingParams.2 hx, rfl⟩
  let e : LinearMap.ker π ≃ₗ[F] potentials F (patternGraph Ω) :=
    { toFun := fun z => ⟨z.1.1.1, mem_potentials_of_snd_eq_zero (mem_vanishingParams.1 z.1.2)
        (LinearMap.mem_ker.1 z.2)⟩
      map_add' := fun _ _ => rfl
      map_smul' := fun _ _ => rfl
      invFun := fun y => ⟨⟨(y.1, 0), mem_vanishingParams_of_potentials y.2⟩, rfl⟩
      left_inv := fun z =>
        Subtype.ext (Subtype.ext (Prod.ext rfl (LinearMap.mem_ker.1 z.2).symm))
      right_inv := fun _ => rfl }
  have hrn := LinearMap.finrank_range_add_finrank_ker π
  rw [hrange, e.finrank_eq, finrank_potentials] at hrn
  exact hrn.symm

omit [DecidableEq ι] [DecidableEq κ] in
/-- The graph of a part has at least one component: the rows and the columns are not empty. -/
lemma one_le_card_components (Ω : Set ((ι → F) × (κ → F))) :
    1 ≤ Nat.card (patternGraph Ω).ConnectedComponent := by
  have : Nonempty (patternGraph Ω).ConnectedComponent :=
    ⟨(patternGraph Ω).connectedComponentMk (.inl 0)⟩
  exact Nat.card_pos

omit [Fintype F] in
/-- The twists span at most the `k l` dimensions of the matrices. -/
lemma finrank_twistSpan_le (Ω : Set ((ι → F) × (κ → F))) :
    finrank F (twistSpan Ω) ≤ Fintype.card ι * Fintype.card κ := by
  have := finrank_ann (twistSpan Ω)
  omega

/-- **The count of PT3** (`review/flat_spectrum.md`, Theorem): the members of the flat class that
vanish on `Ω` form a space of dimension `(c − 1) + (k l − s)`. -/
theorem finrank_vanishing (Ω : Set ((ι → F) × (κ → F))) :
    finrank F (vanishing Ω) =
      (Nat.card (patternGraph Ω).ConnectedComponent - 1) +
        (Fintype.card ι * Fintype.card κ - finrank F (twistSpan Ω)) := by
  have h1 := finrank_vanishing_add_one Ω
  have h2 := finrank_vanishingParams Ω
  have h3 := finrank_ann (twistSpan Ω)
  have hc := one_le_card_components Ω
  omega

example : finrank (ZMod 2) (vanishing (Set.univ : Set ((Fin 1 → ZMod 2) × (Fin 1 → ZMod 2)))) =
    (Nat.card (patternGraph (Set.univ : Set ((Fin 1 → ZMod 2) × (Fin 1 → ZMod 2)))).ConnectedComponent
      - 1) + (Fintype.card (Fin 1) * Fintype.card (Fin 1) -
        finrank (ZMod 2) (twistSpan (Set.univ : Set ((Fin 1 → ZMod 2) × (Fin 1 → ZMod 2))))) :=
  finrank_vanishing _

/-- **The determination of PT3** (`review/flat_spectrum.md`, Theorem): `Ω` determines the flat
class exactly when its graph is connected and its twists span all `k × l` matrices. -/
theorem determines_flatClass_iff (Ω : Set ((ι → F) × (κ → F))) :
    Determines (flatClass F ι κ : Set ((ι → F) × (κ → F) → F)) Ω ↔
      (patternGraph Ω).Connected ∧ twistSpan Ω = ⊤ := by
  have hV : Determines (flatClass F ι κ : Set ((ι → F) × (κ → F) → F)) Ω ↔ vanishing Ω = ⊥ := by
    rw [determines_submodule_iff, Submodule.eq_bot_iff]
    exact ⟨fun h f hf => h f hf.1 hf.2, fun h f hf h0 => h f ⟨hf, h0⟩⟩
  have hc := one_le_card_components Ω
  have hs := finrank_twistSpan_le Ω
  have hfin : finrank F (Matrix ι κ F) = Fintype.card ι * Fintype.card κ := by
    simp only [finrank_matrix, finrank_self, mul_one]
  rw [hV, ← Submodule.finrank_eq_zero, finrank_vanishing, ← card_connectedComponent_eq_one_iff]
  constructor
  · intro h
    exact ⟨by omega, Submodule.eq_top_of_finrank_eq (by omega)⟩
  · rintro ⟨h1, h2⟩
    rw [h1, h2, finrank_top, hfin]
    simp

/-- **The rank of PT3** (`review/flat_spectrum.md`, Theorem): `Ω` puts `n − c + s` independent
conditions on the flat class, for `n = q^k + q^l`; written without subtraction. -/
theorem finrank_map_restrict_flatClass (Ω : Set ((ι → F) × (κ → F))) :
    finrank F ((flatClass F ι κ).map (LinearMap.funLeft F F ((↑) : Ω → (ι → F) × (κ → F)))) +
        Nat.card (patternGraph Ω).ConnectedComponent =
      Fintype.card F ^ Fintype.card ι + Fintype.card F ^ Fintype.card κ +
        finrank F (twistSpan Ω) := by
  let r := (LinearMap.funLeft F F ((↑) : Ω → (ι → F) × (κ → F))).domRestrict (flatClass F ι κ)
  have hker : (LinearMap.ker r).map (flatClass F ι κ).subtype = vanishing Ω := by
    ext f
    constructor
    · rintro ⟨⟨g, hg⟩, hgk, rfl⟩
      exact ⟨hg, fun c hc => congrFun (LinearMap.mem_ker.1 hgk) ⟨c, hc⟩⟩
    · rintro ⟨hf, h0⟩
      exact ⟨⟨f, hf⟩, LinearMap.mem_ker.2 (funext fun c => h0 c.1 c.2), rfl⟩
  have hrn := LinearMap.finrank_range_add_finrank_ker r
  rw [LinearMap.range_domRestrict, ← Submodule.finrank_map_subtype_eq, hker] at hrn
  have hH := finrank_flatClass (F := F) (ι := ι) (κ := κ)
  have h1 := finrank_vanishing_add_one Ω
  have h2 := finrank_vanishingParams Ω
  have h3 := finrank_ann (twistSpan Ω)
  omega

end Transformer.Grokking.FlatSpectrum.TablePart
