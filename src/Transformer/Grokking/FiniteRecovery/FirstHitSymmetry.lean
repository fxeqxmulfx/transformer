module

public import Transformer.Grokking.FiniteRecovery.HiddenCorner

/-! Finite-horizon first hits can visit competing tokens at different times.
If M answer tokens are interchangeable during ordinary training, the chance
to hit a given one at any of k scheduled checks is at most k/M. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery

open MeasureTheory Set

/-- No independence between checkpoints is needed. At each step at most
one token can strictly win and each interchangeable token has the same mass. -/
theorem first_hit_mass_le_checks_div_card {D A : Type*} [MeasurableSpace D]
    (μ : Measure D) [IsProbabilityMeasure μ] (grid : Finset ℕ) (C : Finset A) (hC : C.Nonempty)
    (a : A) (W : ℕ → A → Set D) (hmeas : ∀ t c, MeasurableSet (W t c))
    (hdis : ∀ t, (C : Set A).PairwiseDisjoint (W t))
    (heq : ∀ t, ∀ c ∈ C, μ (W t c) = μ (W t a)) :
    μ.real {z | ∃ t ∈ grid, z ∈ W t a} ≤ (grid.card : ℝ)/C.card := by
  have hp : 0 < (C.card : ℝ) := by exact_mod_cast Finset.card_pos.mpr hC
  have hone (t : ℕ) : μ.real (W t a) ≤ 1/C.card := by
    apply (le_div_iff₀ hp).2
    calc
      μ.real (W t a)*C.card = ∑ _c ∈ C, μ.real (W t a) := by simp [mul_comm]
      _ = ∑ c ∈ C, μ.real (W t c) :=
        Finset.sum_congr rfl fun c hc => congrArg ENNReal.toReal (heq t c hc).symm
      _ = μ.real (⋃ c ∈ C, W t c) := (measureReal_biUnion_finset (hdis t) (fun c _ => hmeas t c)).symm
      _ ≤ 1 := measureReal_le_one
  rw [show {z | ∃ t ∈ grid, z ∈ W t a} = ⋃ t ∈ grid, W t a by ext; simp]
  calc
    _ ≤ ∑ t ∈ grid, μ.real (W t a) := measureReal_biUnion_finset_le _ _
    _ ≤ ∑ _t ∈ grid, (1:ℝ)/C.card := Finset.sum_le_sum fun t _ => hone t
    _ = _ := by simp [div_eq_mul_inv]

namespace Argand

open scoped NNReal

variable {A : Type*} [Fintype A] {H d L T : ℕ}
local notation "Draw" => (ComplexParameter A H d L → ℂ)

/-- For any vocabulary and any two uncorrected moment decays, permutations
fixing the available data and probe bound the chance of a strict first hit. -/
theorem uncorrectedAdam_first_hit_orbit (r : ℕ) (first second ε decay : ℝ) (hε : 0 < ε)
    (rate : ℕ → ℝ) (batch : ℕ → List (LabeledPosition A)) (x : Word A T) (i : Fin T)
    (a : A) (C : Finset A) (hC : C.Nonempty)
    (hs : ∀ c ∈ C, ∃ e : Equiv.Perm A, e a = c ∧ e ∘ x = x ∧
      ∀ t, (batch t).map (relabelPosition e) = batch t)
    (grid : Finset ℕ) (variance : ℝ≥0) :
    let μ := independentGaussianDraws (fun _ : ComplexParameter A H d L => variance)
    let score (z : Draw) t := (Model.fromWeights (uncorrectedAdamRun r first second ε decay batch rate z t).weights).logits r x i
    μ.real {z | ∃ t ∈ grid, ∀ b, b ≠ a → score z t b < score z t a} ≤ (grid.card : ℝ)/C.card := by
  intro μ score
  let W (t : ℕ) (c : A) := {z : Draw | ∀ b, b ≠ c → score z t b < score z t c}
  have hcont (t : ℕ) (b : A) : Continuous (fun z => score z t b) :=
    (continuous_logits_fromWeights r x i b).comp
      ((EqualMoment.continuous_uncorrectedRun decay rate (fun k => batchGradient r (batch k))
        first second hε (fun k => continuous_batchGradient r (batch k)) t).1.comp (continuous_sgdInitialWeights true))
  apply first_hit_mass_le_checks_div_card μ grid C hC a W
  · intro t c
    simp only [W,Set.ofPred_forall]
    exact (isOpen_iInter_of_finite fun b => isOpen_iInter_of_finite fun _ =>
      isOpen_lt (hcont t b) (hcont t c)).measurableSet
  · intro t b _ c _ hbc
    exact Set.disjoint_left.mpr fun _ hb hc => lt_asymm (hb c hbc.symm) (hc b hbc)
  · intro t c hc
    obtain ⟨e,hea,hx,hb⟩ := hs c hc
    have perm (z : Draw) (b : A) : score (relabelDraw e z) t (e b) = score z t b := by
      simpa only [funext hb,hx,score] using uncorrectedAdam_logits_relabel e r t first second ε decay rate batch z x i b
    have pre : (relabelDraw e) ⁻¹' W t c = W t a := by
      ext z
      change (∀ b, b ≠ c → score (relabelDraw e z) t b < score (relabelDraw e z) t c) ↔ _
      rw [← hea]
      constructor
      · intro h d hd
        simpa only [perm] using h (e d) (fun he => hd (e.injective he))
      · intro h d hd
        obtain ⟨b,rfl⟩ := e.surjective d
        simpa only [perm] using h b (fun he => hd (congrArg e he))
    exact (gaussian_relabel_preimage e (fun _ => variance) (W t c)).trans (congrArg μ pre)

namespace HiddenCorner

def usedTokens : Finset (Fin 64) := {0,1,8,9,10,11}
def unseenTokens : Finset (Fin 64) := Finset.univ \ usedTokens

/-- Chance of hitting the strict hidden answer on at least one evaluation. -/
noncomputable def cornerFirstHitProbability (stream : ℕ → List (Fin 3)) (second : ℝ) (grid : Finset ℕ) : ℝ :=
  let μ := independentGaussianDraws (fun _ : ComplexParameter (Fin 64) 4 16 6 => (1/50 : ℝ≥0)^2)
  let score (z : ComplexParameter (Fin 64) 4 16 6 → ℂ) t := (Model.fromWeights (uncorrectedAdamRun 1
    (9/10) second (1/100000000) (1/10) (fun k => (stream k).map observation)
    (fun k => 3*wsdRate 20000 k) z t).weights).logits 1 (prompt (1,0)) 0
  μ.real {z | ∃ t ∈ grid, ∀ b, b ≠ label (1,0) → score z t b < score z t (label (1,0))}

/-- All 58 unused answer tokens are exchangeable at each ordinary step. -/
theorem corner_first_hit_bound (stream : ℕ → List (Fin 3)) (second : ℝ) (grid : Finset ℕ) :
    cornerFirstHitProbability stream second grid ≤ (grid.card : ℝ)/58 := by
  unfold cornerFirstHitProbability
  have nh : unseenTokens.Nonempty := ⟨2,by decide⟩
  have hcard : unseenTokens.card = 58 := by decide
  rw [← show (unseenTokens.card : ℝ) = 58 by exact_mod_cast hcard]
  apply uncorrectedAdam_first_hit_orbit _ _ _ _ _ (by norm_num) _ _ _ _ _ unseenTokens nh
  intro c hc
  let e := Equiv.swap (2 : Fin 64) c
  have hc' : c ∉ usedTokens := (Finset.mem_sdiff.mp hc).2
  have fix {n : Fin 64} (hn : n ∈ usedTokens) : e n = n :=
    Equiv.swap_apply_of_ne_of_ne (fun h => (by decide : (2 : Fin 64) ∉ usedTokens) (h ▸ hn)) (fun h => hc' (h ▸ hn))
  have hi (j : Fin 3) : relabelPosition e (observation j) = observation j := by
    have entry (j : Fin 3) : prompt (observedCell j) 0 ∈ usedTokens ∧ label (observedCell j) ∈ usedTokens := by
      fin_cases j <;> decide
    have hp : e ∘ prompt (observedCell j) = prompt (observedCell j) :=
      funext fun _ => fix (entry j).1
    simp only [observation,relabelPosition,hp,fix (entry j).2]
  refine ⟨e,Equiv.swap_apply_left ..,funext (fun _ => fix (n := 10) (by decide)),fun t => ?_⟩
  rw [List.map_map,show relabelPosition e ∘ observation = observation from funext hi]

/-- The numerical qualification budget allows 17 evaluations including zero.
For either second moment, at most 17/58 of ordinary starts can generalize
by that cutoff on this identifiable and strictly representable example. -/
theorem first_hit_4096 (stream : ℕ → List (Fin 3)) (second : ℝ) :
    cornerFirstHitProbability stream second (evaluationGrid 256 4096) ≤ 17/58 := by
  have hcard : (evaluationGrid 256 4096).card = 17 := by decide +kernel
  simpa only [hcard,Nat.cast_ofNat] using corner_first_hit_bound stream second (evaluationGrid 256 4096)

example (stream : ℕ → List (Fin 3)) : cornerFirstHitProbability stream (19/20) (evaluationGrid 256 4096) < 1/2 :=
  (first_hit_4096 stream (19/20)).trans_lt (by norm_num)

end HiddenCorner
end Argand
end Transformer.Grokking.FiniteRecovery
