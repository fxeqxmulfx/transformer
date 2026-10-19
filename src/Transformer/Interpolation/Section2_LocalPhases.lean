module

public import Transformer.Interpolation.Section1_PhaseSequenceControls

/-!
# Perceptron phases acting inside a region

ArXiv:2411.04551v3, §2.2, proof of `prop: compression`: "when we act on
the `i`-th measure, all the other measures remain invariant". A local
phase is a proved autonomous phase with `𝐕 = 0` and `𝐁 = 0` whose unit
clock law is the pushforward of every initial law by one continuous
bijection of the sphere, the identity off a set `D`.

Such a phase fixes every law without mass in `D`, does not decrease the
mass of a set disjoint from `D`, and keeps every law that gives no mass
outside a region containing `D` inside that region. Finite sequences of
phases in one region inherit all three properties.
-/

@[expose] public section

open Real MeasureTheory Set

namespace Transformer.Interpolation

variable {d : ℕ}

/-- A proved perceptron phase whose unit clock law is the pushforward by
the continuous bijection `Φ`, the identity off `D`. Source auxiliary:
arXiv:2411.04551v3, §2.2, proof of `prop: compression`. -/
def IsLocalPhase (P : AutonomousPhase d) (D : Set (SSphere d)) (Φ : SSphere d → SSphere d) :
    Prop :=
  IsAutonomousPhase P ∧ P.1.B = 0 ∧ P.1.V = 0 ∧ Continuous Φ ∧ Function.Bijective Φ ∧
    (∀ x, x ∉ D → Φ x = x) ∧
    ∀ ρ : Perspective.ProbSphere d, (P.2 ρ 1 : Measure (SSphere d)) = Measure.map Φ ρ

/-- A local phase is local in every larger region. Source auxiliary:
arXiv:2411.04551v3, §2.2. -/
theorem IsLocalPhase.mono {P : AutonomousPhase d} {D E : Set (SSphere d)}
    {Φ : SSphere d → SSphere d} (hP : IsLocalPhase P D Φ) (hDE : D ⊆ E) :
    IsLocalPhase P E Φ :=
  ⟨hP.1, hP.2.1, hP.2.2.1, hP.2.2.2.1, hP.2.2.2.2.1,
    fun x hx => hP.2.2.2.2.2.1 x fun h => hx (hDE h), hP.2.2.2.2.2.2⟩

/-- A local phase fixes every law that gives no mass to its domain.
Source auxiliary: arXiv:2411.04551v3, §2.2, the invariance of the
other measures. -/
theorem IsLocalPhase.law_eq_self {P : AutonomousPhase d} {D : Set (SSphere d)}
    {Φ : SSphere d → SSphere d} (hP : IsLocalPhase P D Φ) (ρ : Perspective.ProbSphere d)
    (hρ : (ρ : Measure (SSphere d)) D = 0) : P.2 ρ 1 = ρ := by
  apply Subtype.ext
  change (P.2 ρ 1 : Measure (SSphere d)) = (ρ : Measure (SSphere d))
  rw [hP.2.2.2.2.2.2 ρ]
  have hae : Φ =ᵐ[(ρ : Measure (SSphere d))] id := by
    refine (measure_eq_zero_iff_ae_notMem.mp hρ).mono fun x hx => ?_
    exact hP.2.2.2.2.2.1 x hx
  rw [Measure.map_congr hae, Measure.map_id]

/-- A local phase does not decrease the mass of a set disjoint from its
domain. Source auxiliary: arXiv:2411.04551v3, §2.2, the clusters that
another phase leaves in place. -/
theorem IsLocalPhase.le_of_disjoint {P : AutonomousPhase d} {D : Set (SSphere d)}
    {Φ : SSphere d → SSphere d} (hP : IsLocalPhase P D Φ) (ρ : Perspective.ProbSphere d)
    (E : Set (SSphere d)) (hED : Disjoint E D) :
    (ρ : Measure (SSphere d)) E ≤ (P.2 ρ 1 : Measure (SSphere d)) E := by
  rw [hP.2.2.2.2.2.2 ρ]
  refine (measure_mono fun x hx => ?_).trans
    (Measure.le_map_apply hP.2.2.2.1.measurable.aemeasurable E)
  change Φ x ∈ E
  rwa [hP.2.2.2.2.2.1 x (Set.disjoint_left.mp hED hx)]

/-- The map of a local phase fixes the complement of every region
containing its domain, as a set. Source auxiliary: arXiv:2411.04551v3,
§2.2, the transport maps of the clustering phases. -/
theorem IsLocalPhase.preimage_compl {P : AutonomousPhase d} {D : Set (SSphere d)}
    {Φ : SSphere d → SSphere d} (hP : IsLocalPhase P D Φ) (V : Set (SSphere d))
    (hDV : D ⊆ V) : Φ ⁻¹' Vᶜ = Vᶜ := by
  have hfix (x : SSphere d) (hx : x ∈ Vᶜ) : Φ x = x :=
    hP.2.2.2.2.2.1 x fun h => hx (hDV h)
  ext x
  constructor
  · intro hx
    have h := hfix (Φ x) hx
    rw [← hP.2.2.2.2.1.1 h]
    exact hx
  · intro hx
    change Φ x ∈ Vᶜ
    rwa [hfix x hx]

/-- A local phase keeps a law without mass outside a region containing
its domain without mass there. Source auxiliary: arXiv:2411.04551v3,
§2.2, the supports stay in their own neighbourhoods. -/
theorem IsLocalPhase.compl_eq_zero {P : AutonomousPhase d} {D : Set (SSphere d)}
    {Φ : SSphere d → SSphere d} (hP : IsLocalPhase P D Φ) (V : Set (SSphere d))
    (hV : MeasurableSet V) (hDV : D ⊆ V) (ρ : Perspective.ProbSphere d)
    (hρ : (ρ : Measure (SSphere d)) Vᶜ = 0) : (P.2 ρ 1 : Measure (SSphere d)) Vᶜ = 0 := by
  rw [hP.2.2.2.2.2.2 ρ, Measure.map_apply hP.2.2.2.1.measurable hV.compl,
    hP.preimage_compl V hDV, hρ]

/-- Every phase of the list is local in the region `V`. Source auxiliary:
arXiv:2411.04551v3, §2.2, the phases acting on one measure. -/
def AllLocalIn (L : List (AutonomousPhase d)) (V : Set (SSphere d)) : Prop :=
  ∀ P ∈ L, ∃ Φ : SSphere d → SSphere d, IsLocalPhase P V Φ

/-- A list of local phases consists of proved perceptron phases.
Source auxiliary: arXiv:2411.04551v3, §2.2, `𝐕 ≡ 0`. -/
theorem AllLocalIn.valid {L : List (AutonomousPhase d)} {V : Set (SSphere d)}
    (hL : AllLocalIn L V) : ∀ P ∈ L, IsAutonomousPhase P ∧ P.1.B = 0 ∧ P.1.V = 0 := by
  intro P hP
  obtain ⟨Φ, hΦ⟩ := hL P hP
  exact ⟨hΦ.1, hΦ.2.1, hΦ.2.2.1⟩

/-- Local phases fix a law without mass in their region. Source
auxiliary: arXiv:2411.04551v3, §2.2, `eq: init.invariant`. -/
theorem AllLocalIn.law_eq_self {L : List (AutonomousPhase d)} {V : Set (SSphere d)}
    (hL : AllLocalIn L V) (ρ : Perspective.ProbSphere d)
    (hρ : (ρ : Measure (SSphere d)) V = 0) : phaseSequenceLaw L ρ = ρ := by
  induction L with
  | nil => rfl
  | cons P L ih =>
    obtain ⟨Φ, hΦ⟩ := hL P List.mem_cons_self
    rw [phaseSequenceLaw_cons, hΦ.law_eq_self ρ hρ]
    exact ih fun Q hQ => hL Q (List.mem_cons_of_mem P hQ)

/-- Local phases keep a law without mass outside their region there.
Source auxiliary: arXiv:2411.04551v3, §2.2. -/
theorem AllLocalIn.compl_eq_zero {L : List (AutonomousPhase d)} {V : Set (SSphere d)}
    (hL : AllLocalIn L V) (hV : MeasurableSet V) (ρ : Perspective.ProbSphere d)
    (hρ : (ρ : Measure (SSphere d)) Vᶜ = 0) :
    (phaseSequenceLaw L ρ : Measure (SSphere d)) Vᶜ = 0 := by
  induction L generalizing ρ with
  | nil => exact hρ
  | cons P L ih =>
    obtain ⟨Φ, hΦ⟩ := hL P List.mem_cons_self
    rw [phaseSequenceLaw_cons]
    exact ih (fun Q hQ => hL Q (List.mem_cons_of_mem P hQ)) _
      (hΦ.compl_eq_zero V hV subset_rfl ρ hρ)

/-- Local phases do not decrease the mass of a set disjoint from their
region. Source auxiliary: arXiv:2411.04551v3, §2.2. -/
theorem AllLocalIn.le_of_disjoint {L : List (AutonomousPhase d)} {V : Set (SSphere d)}
    (hL : AllLocalIn L V) (ρ : Perspective.ProbSphere d) (E : Set (SSphere d))
    (hEV : Disjoint E V) :
    (ρ : Measure (SSphere d)) E ≤ (phaseSequenceLaw L ρ : Measure (SSphere d)) E := by
  induction L generalizing ρ with
  | nil => exact le_rfl
  | cons P L ih =>
    obtain ⟨Φ, hΦ⟩ := hL P List.mem_cons_self
    rw [phaseSequenceLaw_cons]
    exact (hΦ.le_of_disjoint ρ E hEV).trans
      (ih (fun Q hQ => hL Q (List.mem_cons_of_mem P hQ)) _)

/-- Concatenated lists local in one region are local in it. Source
auxiliary: arXiv:2411.04551v3, §2.2. -/
theorem AllLocalIn.append {L M : List (AutonomousPhase d)} {V : Set (SSphere d)}
    (hL : AllLocalIn L V) (hM : AllLocalIn M V) : AllLocalIn (L ++ M) V := by
  intro P hP
  rcases List.mem_append.mp hP with h | h
  · exact hL P h
  · exact hM P h

/-- Lists local in a region are local in every larger region. Source
auxiliary: arXiv:2411.04551v3, §2.2. -/
theorem AllLocalIn.mono {L : List (AutonomousPhase d)} {V W : Set (SSphere d)}
    (hL : AllLocalIn L V) (hVW : V ⊆ W) : AllLocalIn L W := by
  intro P hP
  obtain ⟨Φ, hΦ⟩ := hL P hP
  exact ⟨Φ, hΦ.mono hVW⟩

/-- The zero ridge is a local phase in the empty region. -/
example (i : Fin 2) : IsLocalPhase (ridgePhase i 0 0 0) (∅ : Set (SSphere 2)) id := by
  refine ⟨isAutonomousPhase_ridge i 0 0 0, rfl, rfl, continuous_id, Function.bijective_id,
    fun _ _ => rfl, fun ρ => ?_⟩
  rw [Measure.map_id]
  change (ridgeLaw 0 0 0 ρ 1 : Measure (SSphere 2)) = _
  rw [ridgeLaw_eq_of_inactive_ae 0 0 0 ρ (ae_of_all _ fun _ => by simp) 1]

end Transformer.Interpolation
