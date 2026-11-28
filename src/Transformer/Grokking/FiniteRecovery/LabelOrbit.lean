module

public import Transformer.Grokking.FiniteRecovery.FirstHitSymmetry
public import Transformer.Grokking.FiniteRecovery.NormalizedRecovery

/-! Hidden label assignments under permutations that preserve the
observations and the hidden prompts. The cardinality of their orbit is
a necessary restriction on eventual generalization for an equivariant
training procedure. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery

open Set MeasureTheory Filter
open scoped NNReal

/-- A uniform bound on success at every update also bounds the probability
of success at every sufficiently late update. No independence is needed. -/
theorem eventual_success_mass_le {D : Type*} [MeasurableSpace D] (μ : Measure D) [IsFiniteMeasure μ]
    (E : ℕ → Set D) (p : ℝ) (hp : 0 ≤ p) (h : ∀ t, μ.real (E t) ≤ p) :
    μ.real {z | ∀ᶠ t in atTop, z ∈ E t} ≤ p := by
  let tail (n : ℕ) := {z | ∀ t ≥ n, z ∈ E t}
  have mono : Monotone tail := fun _ _ hnm _ hz t hmt => hz t (hnm.trans hmt)
  have he : {z | ∀ᶠ t in atTop, z ∈ E t} = ⋃ n, tail n := by
    ext z
    simp only [mem_ofPred_eq,eventually_atTop,mem_iUnion,tail]
  have bound : μ (⋃ n, tail n) ≤ ENNReal.ofReal p := by
    rw [mono.measure_iUnion]
    refine iSup_le fun n => (measure_mono (show tail n ⊆ E n from fun _ hz => hz n le_rfl)).trans ?_
    rw [← ofReal_measureReal (μ := μ) (s := E n)]
    exact ENNReal.ofReal_le_ofReal (h n)
  rw [he,measureReal_def]
  exact (ENNReal.toReal_mono ENNReal.ofReal_ne_top bound).trans (ENNReal.toReal_ofReal hp).le

variable {A X D P : Type*}

/-- Use any finite indexing of distinct hidden answer assignments.
The draw transport may be different for each competing assignment. -/
theorem strict_assignment_first_hit [MeasurableSpace D] (μ : Measure D) [IsProbabilityMeasure μ]
    (grid : Finset ℕ) (C : Finset P) (base : P) (hb : base ∈ C) (label : P → X → A)
    (inj : InjOn label (C : Set P)) (score : D → ℕ → X → A → ℝ)
    (meas : ∀ t p, MeasurableSet {z | StrictOn (score z t) (label p)})
    (symm : ∀ p ∈ C, ∃ (e : Equiv.Perm A) (move : D → D),
      (∀ x, e (label base x) = label p x) ∧ (∀ U, μ U = μ (move ⁻¹' U)) ∧
      ∀ z t x a, score (move z) t x (e a) = score z t x a) :
    μ.real {z | ∃ t ∈ grid, StrictOn (score z t) (label base)} ≤ (grid.card : ℝ)/C.card := by
  let W t p := {z | StrictOn (score z t) (label p)}
  apply first_hit_mass_le_checks_div_card μ grid C ⟨base,hb⟩ base W meas
  · intro t p hp q hq hne
    apply Set.disjoint_left.mpr
    intro z h g
    apply hne (inj hp hq ?_)
    funext x
    by_contra hn
    exact lt_asymm (h x _ (Ne.symm hn)) (g x _ hn)
  · intro t p hp
    obtain ⟨e,move,target,law,perm⟩ := symm p hp
    have pre : move ⁻¹' W t p = W t base := by
      ext z
      change (∀ x b, b ≠ label p x → score (move z) t x b < score (move z) t x (label p x)) ↔ _
      constructor
      · intro h x b hb
        have v := h x (e b) (fun he => hb (e.injective (he.trans (target x).symm)))
        simpa only [← target,perm] using v
      · intro h x c hc
        obtain ⟨b,rfl⟩ := e.surjective c
        simp only [← target,perm]
        exact h x b (fun he => hc ((congrArg e he).trans (target x)))
    exact (law _).trans (congrArg μ pre)

/-- Distinct assignments obtained by relabeling the answers, counted
once even when several group elements induce the same assignment. -/
noncomputable def hiddenLabelOrbit [Finite A] [Finite X] (G : Subgroup (Equiv.Perm A)) (f : X → A) : Finset (X → A) :=
  (Set.toFinite (range fun e : G => e.val ∘ f)).toFinset

theorem mem_hiddenLabelOrbit [Finite A] [Finite X] {G : Subgroup (Equiv.Perm A)} {f p : X → A} :
    p ∈ hiddenLabelOrbit G f ↔ ∃ e : G, e.val ∘ f = p := by
  simp only [hiddenLabelOrbit,Set.Finite.mem_toFinset,mem_range]

theorem hiddenLabelOrbit_singleton_iff [Finite A] [Finite X] (G : Subgroup (Equiv.Perm A)) (f : X → A) :
    (hiddenLabelOrbit G f).card = 1 ↔ ∀ (e : G) x, e.val (f x) = f x := by
  classical
  have hf : f ∈ hiddenLabelOrbit G f := mem_hiddenLabelOrbit.mpr ⟨1,rfl⟩
  constructor
  · intro h e x
    obtain ⟨p,hp⟩ := Finset.card_eq_one.mp h
    have he : e.val ∘ f ∈ hiddenLabelOrbit G f := mem_hiddenLabelOrbit.mpr ⟨e,rfl⟩
    rw [hp,Finset.mem_singleton] at hf he
    exact congrFun (he.trans hf.symm) x
  · intro h
    have eq : hiddenLabelOrbit G f = {f} := by
      ext p
      simp only [mem_hiddenLabelOrbit,Finset.mem_singleton]
      exact ⟨fun ⟨e,he⟩ => he.symm.trans (funext (h e)),fun hp => ⟨1,hp.symm⟩⟩
    simp [eq]

/-- Success probability above one half forces every data-preserving
permutation to fix each hidden answer when the orbit upper bound applies. -/
theorem hidden_labels_fixed_of_above_half [Finite A] [Finite X] (G : Subgroup (Equiv.Perm A)) (f : X → A)
    {p : ℝ} (hp : 1/2 < p) (bound : p ≤ 1/(hiddenLabelOrbit G f).card) :
    ∀ (e : G) x, e.val (f x) = f x := by
  apply (hiddenLabelOrbit_singleton_iff G f).mp
  have pos := Finset.card_pos.mpr (show (hiddenLabelOrbit G f).Nonempty from ⟨f,mem_hiddenLabelOrbit.mpr ⟨1,rfl⟩⟩)
  by_contra hn
  have h : 2 ≤ (hiddenLabelOrbit G f).card := by omega
  exact (not_lt_of_ge (bound.trans (one_div_le_one_div_of_le (by norm_num) (by exact_mod_cast h)))) hp

/-- A symmetry group of the training law bounds both the first hit on
a finite grid and eventual correct generalization on the infinite path. -/
theorem hidden_orbit_bounds [Finite A] [Finite X] [MeasurableSpace D] (μ : Measure D) [IsProbabilityMeasure μ]
    (G : Subgroup (Equiv.Perm A)) (f : X → A) (score : D → ℕ → X → A → ℝ)
    (meas : ∀ t p, MeasurableSet {z | StrictOn (score z t) p}) (move : G → D → D)
    (law : ∀ e U, μ U = μ (move e ⁻¹' U))
    (perm : ∀ (e : G) z t x a, score (move e z) t x (e.val a) = score z t x a) :
    (∀ grid : Finset ℕ, μ.real {z | ∃ t ∈ grid, StrictOn (score z t) f} ≤ (grid.card : ℝ)/(hiddenLabelOrbit G f).card) ∧
    μ.real {z | ∀ᶠ t in atTop, StrictOn (score z t) f} ≤ 1/(hiddenLabelOrbit G f).card := by
  have finiteGrid (grid : Finset ℕ) : μ.real {z | ∃ t ∈ grid, StrictOn (score z t) f} ≤
      (grid.card : ℝ)/(hiddenLabelOrbit G f).card := by
    apply strict_assignment_first_hit μ grid (hiddenLabelOrbit G f) f (mem_hiddenLabelOrbit.mpr ⟨1,rfl⟩) id
      Function.injective_id.injOn score meas
    intro p hp
    obtain ⟨e,rfl⟩ := mem_hiddenLabelOrbit.mp hp
    exact ⟨e.val,move e,fun _ => rfl,law e,perm e⟩
  refine ⟨finiteGrid,eventual_success_mass_le μ (fun t => {z | StrictOn (score z t) f}) _ (by positivity) ?_⟩
  intro t
  simpa using finiteGrid {t}

namespace Argand

variable [Fintype A] [Finite X] {H d L T : ℕ}
local notation "Draw" => (ComplexParameter A H d L → ℂ)

/-- For stock and recursively normalized AdamW it suffices that the
permutations fix each minibatch and every hidden prompt. All native
weights, gates and projections continue to learn. -/
theorem normalizedAdam_hidden_orbit (G : Subgroup (Equiv.Perm A)) (r : ℕ) (rate : ℕ → ℝ)
    (batch : ℕ → List (LabeledPosition A)) (prompt : X → Word A (T+1)) (f : X → A)
    (batches : ∀ (e : G) t, (batch t).map (relabelPosition e.val) = batch t)
    (prompts : ∀ (e : G) x, e.val ∘ prompt x = prompt x) (variance : ℝ≥0) :
    let score (z : Draw) t x := (Model.fromWeights (normalizedAdamRun r batch rate z t).weights).logits r (prompt x) (Fin.last T)
    let μ := independentGaussianDraws (fun _ : ComplexParameter A H d L => variance)
    (∀ grid : Finset ℕ, μ.real {z | ∃ t ∈ grid, StrictOn (score z t) f} ≤ (grid.card : ℝ)/(hiddenLabelOrbit G f).card) ∧
    μ.real {z | ∀ᶠ t in atTop, StrictOn (score z t) f} ≤ 1/(hiddenLabelOrbit G f).card := by
  intro score μ
  apply hidden_orbit_bounds μ G f score _ (fun e => relabelDraw e.val)
  · intro e U
    exact gaussian_relabel_preimage e.val (fun _ => variance) U
  · intro e z t x a
    simpa only [score,normalizedAdam_weights,funext (batches e),prompts] using
      adamW_logits_relabel e.val r t batch rate z (prompt x) (Fin.last T) a
  · intro t p
    simpa only [score,normalizedAdam_weights] using (adamW_strict_open (H := H) (d := d) (L := L) r t rate batch prompt p).measurableSet

/-- On every identifying division mask in the factor family, observing
each denominator pins every unit answer token. One observed zero-row
answer also pins the (possibly nonnumeric) constant token. -/
theorem factor_hidden_orbit_singleton (S : Set DivisionInput)
    (connected : (FlatSpectrum.patternGraph (unitMask S)).Preconnected) (hz : ∃ j, (none,j) ∈ S)
    (G : Subgroup (Equiv.Perm (Fin 239))) (f : DivisionInput → Fin 239) (hf : f ∈ factorClass unitToken)
    (inputs : ∀ (e : G) x, x ∈ S → e.val ∘ divisionPrompt x = divisionPrompt x)
    (answers : ∀ (e : G) x, x ∈ S → e.val (f x) = f x) (hidden : X → DivisionInput) :
    (hiddenLabelOrbit G (f ∘ hidden)).card = 1 := by
  obtain ⟨r,c,z,rfl⟩ := hf
  apply (hiddenLabelOrbit_singleton_iff _ _).mpr
  intro e x
  have units (u : DivisionUnit) : e.val (residueToken u.val) = residueToken u.val := by
    obtain ⟨v,adj⟩ := connected.exists_adj_of_nontrivial (Sum.inr u)
    cases v with
    | inl i =>
      have observed : (i,u) ∈ unitMask S := FlatSpectrum.patternGraph_adj_inr_inl.mp adj
      exact congrFun (inputs e (some i,u) observed) (3 : Fin 5)
    | inr _ => exact False.elim (FlatSpectrum.not_patternGraph_adj_inr_inr adj)
  obtain ⟨j,hj⟩ := hz
  change e.val (factorTable unitToken r c z (hidden x)) = factorTable unitToken r c z (hidden x)
  obtain ⟨i,b⟩ := hidden x
  cases i with
  | none => exact answers e (none,j) hj
  | some i => exact units (r i+c b).toMul

end Argand
namespace Argand.HiddenCorner

/-- The same three-observation corner obstructs permanent correctness for
every second-moment decay, including the 0.95 uncorrected validation rule. -/
theorem uncorrected_eventual_bound (stream : ℕ → List (Fin 3)) (second : ℝ) :
    let μ := independentGaussianDraws (fun _ : ComplexParameter (Fin 64) 4 16 6 => (1/50 : ℝ≥0)^2)
    let score (z : ComplexParameter (Fin 64) 4 16 6 → ℂ) t := (Model.fromWeights (uncorrectedAdamRun 1
      (9/10) second (1/100000000) (1/10) (fun k => (stream k).map observation)
      (fun k => 3*wsdRate 20000 k) z t).weights).logits 1 (prompt (1,0)) 0
    μ.real {z | ∀ᶠ t in atTop, ∀ b, b ≠ label (1,0) → score z t b < score z t (label (1,0))} ≤ 1/58 := by
  intro μ score
  apply eventual_success_mass_le μ (fun t => {z | ∀ b, b ≠ label (1,0) → score z t b < score z t (label (1,0))}) _ (by norm_num)
  intro t
  simpa [cornerFirstHitProbability,μ,score] using corner_first_hit_bound stream second {t}

example (stream : ℕ → List (Fin 3)) := uncorrected_eventual_bound stream (19/20)

end Argand.HiddenCorner
end Transformer.Grokking.FiniteRecovery
