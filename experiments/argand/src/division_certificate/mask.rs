//! Export the observed prompt mask as literals with ordinary Lean kernel
//! proofs. Neither held-out equations nor neural checkpoints are inputs.
use super::{Completion, N};
use crate::{Result, division, division_equations as eq};
use std::fmt::{Display, Write};

fn array<T: Display>(values: &[T]) -> String {
    values
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

pub(crate) fn lean_source() -> Result<String> {
    let open = eq::split(0, 4656).0;
    if eq::fingerprint(&open) != division::TRAIN_HASH {
        return Err("training fingerprint mismatch".into());
    }
    let completion = Completion::from_observations(&open)?;
    let mut observed = [0_u128; eq::P];
    for e in &open {
        observed[e.num as usize] |= 1_u128 << (e.den - 1);
    }
    let mut bridge = Vec::with_capacity(N);
    let mut cover = Vec::with_capacity(N);
    for i in 0..N {
        let common = observed[1] & observed[i + 1];
        if common == 0 {
            return Err("unit row does not meet the anchor in a column".into());
        }
        bridge.push(common.trailing_zeros());
        cover.push(
            observed[1..]
                .iter()
                .position(|row| row >> i & 1 == 1)
                .ok_or("unobserved unit column")?,
        );
    }
    // Choose one hidden input for each quotient whose token we will permute.
    // Both operands stay outside the entire set of permuted quotients.
    let free = 10..=82;
    let mut probes = Vec::new();
    for q in free.clone() {
        let (a, b) = (1..=N)
            .filter(|b| !free.contains(b))
            .find_map(|b| {
                (1..=N)
                    .find(|&a| {
                        !free.contains(&a)
                            && observed[a] >> (b - 1) & 1 == 0
                            && completion.label(a, b) == q
                    })
                    .map(|a| (a, b))
            })
            .ok_or("missing hidden probe for output permutation")?;
        probes.push(format!("({}, {})", a - 1, b - 1));
    }
    let matching = completion
        .graph
        .perfect_matching
        .ok_or("no perfect unit matching")?;
    let mut inverse = [0; N];
    for (i, &j) in matching.iter().enumerate() {
        inverse[j] = i;
    }
    let mut out = String::from(
        "import Transformer.Grokking.FiniteRecovery.FactorFirstNoise\nimport Transformer.Grokking.FiniteRecovery.ProjectedSoftsign\nimport Transformer.Grokking.FiniteRecovery.ArgandRmsProp\nimport Transformer.Grokking.FiniteRecovery.SoftsignLimit\nimport Transformer.Grokking.FiniteRecovery.LabelOrbit\nimport Transformer.Grokking.FiniteRecovery.DelayedCheckpoints\nimport Transformer.Grokking.FiniteRecovery.ProposalMass\nimport Transformer.Grokking.FiniteRecovery.OrdinaryDivisionValidation\nimport Transformer.Grokking.FiniteRecovery.ArgandObservedJacobian\nimport Transformer.Grokking.FiniteRecovery.ArgandHiddenLearnability\n\n",
    );
    writeln!(
        out,
        "/- Observed equations SHA-256: {} -/",
        division::TRAIN_HASH
    )?;
    out.push_str("namespace OriginalDivisionMask\nopen Transformer.Grokking.FiniteRecovery Transformer.Grokking.FiniteRecovery.Argand\nset_option maxRecDepth 8192\nset_option maxHeartbeats 2000000\n");
    out.push_str(
        "open Transformer.Grokking.FlatSpectrum Matrix MeasureTheory\nopen scoped NNReal ENNReal\n",
    );
    writeln!(out, "def observed : Array Nat := #[{}]", array(&observed))?;
    for (name, values) in [
        ("bridges", array(&bridge)),
        ("covers", array(&cover)),
        ("matchCols", array(&matching)),
        ("inverseRows", array(&inverse)),
    ] {
        writeln!(out, "def {name} : Array (Fin 96) := #[{values}]")?;
    }
    out.push_str("def rows (i : Fin 97) := observed[i.val]!\ndef bridge (i : Fin 96) := bridges[i.val]!\ndef cover (i : Fin 96) := covers[i.val]!\n");
    writeln!(
        out,
        "def massProbes : Array (Fin 96 × Fin 96) := #[{}]",
        array(&probes)
    )?;
    out.push_str(r#"
def probeIndices (i : Fin 73) := massProbes[i.val]!
def probe (i : Fin 73) : DivisionInput :=
  (some (divisionUnitIndex.symm (probeIndices i).1), divisionUnitIndex.symm (probeIndices i).2)

theorem original_probe_hidden (i : Fin 73) : probe i ∉ bitDivisionMask rows := by
  have h : ∀ j : Fin 73, (rows (probeIndices j).1.succ).testBit (probeIndices j).2.val = false := by decide
  simpa [bitDivisionMask,probe] using h i

theorem original_probe_answers (i : Fin 73) : (divisionAnswer (probe i)).val = 32+i.val := by
  let q : Fin 96 := ⟨9+i.val,by omega⟩
  have table : ∀ j : Fin 73, (probeIndices j).1.val+1 =
      ((10+j.val)*((probeIndices j).2.val+1))%97 := by decide
  have product : divisionUnitIndex.symm (probeIndices i).1 =
      divisionUnitIndex.symm q * divisionUnitIndex.symm (probeIndices i).2 := by
    apply Units.ext
    apply ZMod.val_injective 97
    change (probeIndices i).1.val+1 = ((9+i.val+1)*((probeIndices i).2.val+1))%97
    simpa only [show 9+i.val+1 = 10+i.val by omega] using table i
  change 22+((divisionUnitIndex.symm (probeIndices i).1 / divisionUnitIndex.symm (probeIndices i).2).val.val) = _
  rw [div_eq_iff_eq_mul.mpr product]
  change 22+(9+i.val+1) = 32+i.val
  omega

theorem original_zero_proposal_ceiling (variance : ℝ≥0) :
    let law := Measure.pi (fun _ : Parameter (Fin 239) 4 16 6 => ProbabilityTheory.gaussianReal 0 variance)
    law.real {z | ∀ x ∈ (bitDivisionMask rows)ᶜ, ∀ b, b ≠ divisionAnswer x →
      (Model.fromWeights z).logits 1 (divisionPrompt x) (Fin.last 4) b <
        (Model.fromWeights z).logits 1 (divisionPrompt x) (Fin.last 4) (divisionAnswer x)} ≤
          1/(Nat.factorial 73 : ℝ) :=
  division_zero_proposal_ceiling _ probe original_probe_hidden original_probe_answers (by decide) variance

#print axioms original_zero_proposal_ceiling

theorem original_uniform_proposal_range (R : ℝ) (hR : 0 ≤ R) :
    let law := Measure.pi (fun _ : Parameter (Fin 239) 4 16 6 =>
      ProbabilityTheory.gaussianReal 0 ((1/1000000 : ℝ≥0)^2))
    let U := {w : Weights (Fin 239) 4 16 6 | ∀ x ∈ (bitDivisionMask rows)ᶜ, ∀ b, b ≠ divisionAnswer x →
      (Model.fromWeights w).logits 1 (divisionPrompt x) (Fin.last 4) b <
        (Model.fromWeights w).logits 1 (divisionPrompt x) (Fin.last 4) (divisionAnswer x)}
    let p := uniformProposalMass law (Metric.closedBall 0 R) U
    0 < p ∧ p ≤ ENNReal.ofReal (1/(Nat.factorial 73 : ℝ)) :=
  division_uniform_proposal_range (bitDivisionMask rows)ᶜ probe original_probe_hidden
    original_probe_answers (by decide) ((1/1000000 : ℝ≥0)^2) (by norm_num) R hR

#print axioms original_uniform_proposal_range

theorem original_ordinary_finite_hit (B : ℕ) (batch : OrdinaryDivision.Draw → OrdinaryDivision.Batches) :
    let μ := independentGaussianDraws (fun _ : ComplexParameter (Fin 239) 4 16 6 => (1/50 : ℝ≥0)^2)
    let p := μ (OrdinaryDivision.hit B (fun t => 3*wsdRate 20000 t) batch (bitDivisionMask rows)ᶜ)
    0 < p ∧ p < 1 :=
  OrdinaryDivision.tuned_wsd_finite_hit B batch _ (probe 0) (original_probe_hidden 0)
    (Fin.ext (by simpa using original_probe_answers 0))

example (batch : OrdinaryDivision.Draw → OrdinaryDivision.Batches) := original_ordinary_finite_hit 8192 batch
#print axioms original_ordinary_finite_hit
theorem original_ordinary_learned_mass (B : ℕ) (batch : OrdinaryDivision.Draw → OrdinaryDivision.Batches) :
    let μ := independentGaussianDraws (fun _ : ComplexParameter (Fin 239) 4 16 6 => (1/50 : ℝ≥0)^2)
    let E := OrdinaryDivision.hit B (fun t => 3*wsdRate 20000 t) batch (bitDivisionMask rows)ᶜ
    let I := OrdinaryDivision.initialSuccess (bitDivisionMask rows)ᶜ
    μ.real (E \ I) = μ.real E - μ.real I ∧ μ.real I ≤ 1/(Nat.factorial 73 : ℝ) :=
  ⟨OrdinaryDivision.learned_mass _ _ _ _ _, OrdinaryDivision.initial_success_ceiling _ probe
    original_probe_hidden original_probe_answers (by decide) _⟩
#print axioms original_ordinary_learned_mass
"#);
    out.push_str("def matching (i : Fin 96) := matchCols[i.val]!\ndef inverse (j : Fin 96) := inverseRows[j.val]!\n");
    out.push_str("theorem observed_matching : (∀ i, (rows i.succ).testBit (matching i).val = true) ∧ (∀ i, inverse (matching i) = i) ∧ (∀ j, matching (inverse j) = j) := by decide\n");
    out.push_str("theorem original_graph : (patternGraph (unitMask (bitDivisionMask rows))).Preconnected ∧ ∃ j, (none,j) ∈ bitDivisionMask rows :=\n");
    writeln!(
        out,
        "  bitMask_identifying_graph rows bridge cover {} (by decide) (by decide) (by decide)",
        observed[0].trailing_zeros()
    )?;
    out.push_str(r#"
theorem original_mask_certifies (score : DivisionInput → Fin 239 → ℝ) :
    classCertificate (bitDivisionMask rows) (factorClass unitToken) (fun x => divisionAnswer x) score ↔
      StrictOn score divisionAnswer :=
  division_certificate_iff _ original_graph.1 original_graph.2 score

noncomputable def riskSites : Finset DivisionInput := (bitDivisionMask rows).toFinite.toFinset
def quotientRule (_ : Unit) : Set (DivisionInput → Fin 239) := factorClass unitToken

theorem original_rule_identified : Set.InjOn (fun g : DivisionInput → Fin 239 => fun x : riskSites => g x)
    (⋂ j, quotientRule j) := by
  simp only [quotientRule,Set.iInter_const]
  intro f hf g hg eq
  apply factorClass_identified unitToken (bitDivisionMask rows) original_graph.1 original_graph.2 hf hg
  exact funext fun x => congrFun eq ⟨x.val, (bitDivisionMask rows).toFinite.mem_toFinset.mpr x.property⟩

section
-- Keep specialization from reducing the entire six-layer forward pass.
attribute [local irreducible] Model.logits ObservedDrift.marginalBound

/-- The quotient-factor class contributes a single structural rule. -/
theorem original_marginal_certificate (w : Weights (Fin 239) 4 16 6) :
    ObservedDrift.marginalBound (H := 4) (d := 16) (L := 6) 1 (Fintype.card Unit) riskSites
      (fun x => divisionAnswer x.val) divisionAnswer divisionPrompt (fun _ => Fin.last 4) w < (1/2 : ℝ) →
      StrictOn (fun x => (Model.fromWeights w).logits 1 (divisionPrompt x) (Fin.last 4)) divisionAnswer :=
  ObservedDrift.strict_of_marginalBound (H := 4) (d := 16) (L := 6) 1 riskSites divisionAnswer
    quotientRule (fun _ => divisionAnswer_mem) original_rule_identified divisionPrompt (fun _ => Fin.last 4) w
#print axioms original_marginal_certificate
end

/-- With the original mask, Gaussian start and WSD, the remaining hypothesis
is on the Jacobian along the actual observed minibatch steps and its excess. -/
example (correct : Bool) (batches : OrdinaryDivision.Draw → ℕ → List riskSites) (B : ℕ)
    {ρ r : ℝ} (hρ : 0 ≤ ρ) (hlt : ρ < 1) (hr : r < 1/2) (e : OrdinaryDivision.Draw → ℝ)
    (hm : Measurable e) (hi : Integrable e (independentGaussianDraws (fun _ => (1/50 : ℝ≥0)^2)))
    (hp : ∀ᵐ z ∂independentGaussianDraws (fun _ => (1/50 : ℝ≥0)^2), 0 ≤ e z)
    (mean : ∫ z, e z ∂independentGaussianDraws (fun _ => (1/50 : ℝ≥0)^2) < (1-ρ)*(1/2-r)/4) :=
  ObservedDrift.three_quarters_of_marginal_jacobian _ correct 1 riskSites (fun x => divisionAnswer x)
    quotientRule divisionAnswer (fun _ => divisionAnswer_mem) (fun _ => rfl) original_rule_identified
    divisionPrompt (fun _ => Fin.last 4) batches (fun t => 3*wsdRate 20000 t) B hρ hlt hr e hm hi hp mean

theorem riskSites_coe : (riskSites : Set DivisionInput) = bitDivisionMask rows :=
  (bitDivisionMask rows).toFinite.coe_toFinset

/-- Each answer of each member of the class is an observed denominator token
or the observed zero-row answer, so no relabeling symmetry of the run moves it. -/
theorem original_answers_seen {f : DivisionInput → Fin 239} (hf : f ∈ factorClass unitToken)
    (x : DivisionInput) : f x ∈ divisionSeen (bitDivisionMask rows) f :=
  division_answers_seen _ original_graph.1 original_graph.2 hf x
#print axioms original_answers_seen

/-- **Three quarters on the original split, conditionally.** Any law of the
draw and the minibatch order, ordinary AdamW in real arithmetic at the
registered WSD, every member of the class. The Jacobian bound along the
actual updates and the mean of its excess are hypotheses. -/
theorem original_division_three_quarters (μ : Measure OrdinaryDivision.Draw) [IsProbabilityMeasure μ]
    {F : DivisionInput → Fin 239} (hF : F ∈ factorClass unitToken)
    (batches : OrdinaryDivision.Draw → ℕ → List riskSites) (B : ℕ) {ρ floor : ℝ}
    (hρ : 0 ≤ ρ) (hlt : ρ < 1) (hr : floor < 1/2) (e : OrdinaryDivision.Draw → ℝ)
    (hm : Measurable e) (hi : Integrable e μ) (hp : ∀ᵐ z ∂μ, 0 ≤ e z)
    (mean : ∫ z, e z ∂μ < (1-ρ)*(1/2-floor)/4) :
    let s z := ObservedDrift.state true 1 (fun t => ObservedDrift.labeledBatch riskSites (fun x => F x)
      divisionPrompt (fun _ => Fin.last 4) (batches z t)) (fun t => 3*wsdRate 20000 t) z
    let U := ObservedDrift.marginalBound (H := 4) (d := 16) (L := 6) 1 1 riskSites (fun x => F x) F
      divisionPrompt (fun _ => Fin.last 4)
    (∀ z t, B ≤ t → ∀ u ∈ Set.Ioo (0:ℝ) 1,
      let w := (s z t).weights
      let Δ := ObservedDrift.displacement true (3*wsdRate 20000 t) t (s z t) (s z (t+1))
      fderiv ℝ U (w + u • Δ) Δ ≤ -(1-ρ)*(U w-floor)+e z) →
    3/4 < μ.real {z | HiddenSuccess (bitDivisionMask rows) (fun t x =>
      (Model.fromWeights (s z t).weights).logits 1 (divisionPrompt x) (Fin.last 4)) F} := by
  intro s U jac
  rw [← riskSites_coe]
  exact ObservedDrift.division_three_quarters μ riskSites (by rw [riskSites_coe]; exact original_graph.1)
    (original_graph.2.imp fun _ h => (Set.Finite.mem_toFinset _).mpr h) hF batches B hρ hlt hr e hm hi hp
    mean jac
#print axioms original_division_three_quarters

/-- **Learned, conditionally.** Under the stock Gaussian law, the runs that
become strict on every hidden input for good although their initial hidden
table is not strict have mass above 3/4 - 1/73!. -/
theorem original_learned_three_quarters (batches : OrdinaryDivision.Draw → ℕ → List riskSites) (B : ℕ)
    {ρ floor : ℝ} (hρ : 0 ≤ ρ) (hlt : ρ < 1) (hr : floor < 1/2) (e : OrdinaryDivision.Draw → ℝ)
    (hm : Measurable e) (hi : Integrable e (independentGaussianDraws (fun _ => (1/50 : ℝ≥0)^2)))
    (hp : ∀ᵐ z ∂independentGaussianDraws (fun _ => (1/50 : ℝ≥0)^2), 0 ≤ e z)
    (mean : ∫ z, e z ∂independentGaussianDraws (fun _ => (1/50 : ℝ≥0)^2) < (1-ρ)*(1/2-floor)/4) :
    let s z := ObservedDrift.state true 1 (fun t => ObservedDrift.labeledBatch riskSites
      (fun x => divisionAnswer x) divisionPrompt (fun _ => Fin.last 4) (batches z t)) (fun t => 3*wsdRate 20000 t) z
    let U := ObservedDrift.marginalBound (H := 4) (d := 16) (L := 6) 1 1 riskSites (fun x => divisionAnswer x)
      divisionAnswer divisionPrompt (fun _ => Fin.last 4)
    (∀ z t, B ≤ t → ∀ u ∈ Set.Ioo (0:ℝ) 1,
      let w := (s z t).weights
      let Δ := ObservedDrift.displacement true (3*wsdRate 20000 t) t (s z t) (s z (t+1))
      fderiv ℝ U (w + u • Δ) Δ ≤ -(1-ρ)*(U w-floor)+e z) →
    3/4 - 1/(Nat.factorial 73 : ℝ) < (independentGaussianDraws (fun _ => (1/50 : ℝ≥0)^2)).real
      ({z | HiddenSuccess (bitDivisionMask rows) (fun t x =>
        (Model.fromWeights (s z t).weights).logits 1 (divisionPrompt x) (Fin.last 4)) divisionAnswer} \
        OrdinaryDivision.initialSuccess (bitDivisionMask rows)ᶜ) := by
  intro s U jac
  have hP := original_division_three_quarters _ divisionAnswer_mem batches B hρ hlt hr e hm hi hp mean jac
  have hI := OrdinaryDivision.initial_success_ceiling (bitDivisionMask rows)ᶜ probe original_probe_hidden
    original_probe_answers (by decide) ((1/50 : ℝ≥0)^2)
  exact ((sub_lt_sub_right hP _).trans_le (sub_le_sub_left hI _)).trans_le le_measureReal_sdiff
#print axioms original_learned_three_quarters

example (batches : OrdinaryDivision.Draw → ℕ → List riskSites) :=
  original_learned_three_quarters batches 20000 (ρ := 0) (floor := 0) le_rfl one_pos (by norm_num)
    (fun _ => 0) measurable_const (integrable_const 0) (ae_of_all _ fun _ => le_rfl) (by norm_num)

theorem original_factor_label_orbit {X : Type*} [Finite X] (G : Subgroup (Equiv.Perm (Fin 239)))
    (f : DivisionInput → Fin 239) (hf : f ∈ factorClass unitToken)
    (inputs : ∀ (e : G) x, x ∈ bitDivisionMask rows → e.val ∘ divisionPrompt x = divisionPrompt x)
    (answers : ∀ (e : G) x, x ∈ bitDivisionMask rows → e.val (f x) = f x) (hidden : X → DivisionInput) :
    (hiddenLabelOrbit G (f ∘ hidden)).card = 1 :=
  factor_hidden_orbit_singleton _ original_graph.1 original_graph.2 G f hf inputs answers hidden

theorem original_nuclear_completion {u v : DivisionUnit → ℂ}
    (h : ∀ i j, ‖vecMulVec u v i j‖ = 1) :
    let Ω := unitMask (bitDivisionMask rows)
    ∀ X ∈ {X' : Matrix DivisionUnit DivisionUnit ℂ | ∀ ij ∈ Ω, X' ij.1 ij.2 = vecMulVec u v ij.1 ij.2},
      (IsMinOn nuclearNorm {X' | ∀ ij ∈ Ω, X' ij.1 ij.2 = vecMulVec u v ij.1 ij.2} X ↔ X = vecMulVec u v) := by
  let σ : Equiv.Perm (Fin 96) := ⟨matching,inverse,observed_matching.2.1,observed_matching.2.2⟩
  exact (forall_isMinOn_vecMulVec_iff_connected_of_entries h).2
    ⟨⟨original_graph.1⟩,bitMask_unit_matching rows σ observed_matching.1⟩

theorem original_factor_class_stabilizes (stockRuns : ℕ) (K : ℕ → ℕ)
    (variance : ComplexParameter (Fin 239) 4 16 6 → ℝ≥0) (hv : ∀ i, 0 < variance i) :
    let P := Measure.infinitePi (fun _ : ℕ => independentGaussianDraws variance)
    ∀ᵐ w ∂P, ∀ f : (factorClass unitToken : Set (DivisionInput → Fin 239)),
      ∀ batch : ℕ → ℕ → List (LabeledPosition (Fin 239)),
        let scores := restartScores 1 K (restartWsdRate stockRuns K) divisionPrompt w batch
        let ok := fun j => classCertificate (bitDivisionMask rows) (factorClass unitToken) (fun x => f.val x) (scores j)
        ∃ j, StrictOn (scores j) f.val ∧ ∀ k, retained ok id (j+1+k) = some j :=
  Division.Factors.class_restartWsd_stabilizes stockRuns K _ original_graph.1 original_graph.2 variance hv

theorem original_firstNoise_factor_recovery (n : ℕ) (f : DivisionInput → Fin 239)
    (hf : f ∈ factorClass unitToken) (batch : ℕ → List (LabeledPosition (Fin 239)))
    (draw : ComplexParameter (Fin 239) 4 16 6 → ℂ) :
    let μ := Measure.pi (fun _ : Parameter (Fin 239) 4 16 6 => ProbabilityTheory.gaussianReal 0 ((1/1000000 : ℝ≥0)^2))
    let scores := firstNoiseScores 1 (n+1) (wsdRate (n+1)) batch divisionPrompt draw
    let ok z := classCertificate (bitDivisionMask rows) (factorClass unitToken) (fun x => f x) (scores z)
    ∀ᵐ z ∂Measure.infinitePi (fun _ : ℕ => μ), ∃ j, StrictOn (scores (z j)) f ∧
      ∀ k, retained ok z (j+1+k) = some (z j) :=
  (Division.Factors.class_firstNoise_wsd_restarts n _ original_graph.1 original_graph.2 f hf batch draw).2.2

/-- The frozen 4,656 observed equations identify the class. After the first
8,192 ordinary updates, independent real proposals at every 256th step
give almost-sure certified recovery on the complementary inputs. -/
theorem original_softsign_checkpoint_recovery (f : DivisionInput → Fin 239)
    (hf : f ∈ factorClass unitToken) (batch : ℕ → List (LabeledPosition (Fin 239)))
    (draw : ComplexParameter (Fin 239) 4 16 6 → ℂ) (decay scale : ℝ)
    (hdc : 0 < decay) (hs : 0 ≤ scale) (hcap : scale*decay ≤ 1000) :
    let S := bitDivisionMask rows
    let hidden : {x : DivisionInput // x ∉ S} → DivisionInput := Subtype.val
    PathCheckpointRecovery 1 (fun j =>
      (softsignRun 1 decay batch (fun t => scale*wsdRate 20000 t) draw (256*(32+j))).weights)
      (divisionPrompt ∘ hidden) (f ∘ hidden) (hiddenCertificate S (factorClass unitToken) (fun x => f x) hidden)
      ((1/1000000 : ℝ≥0)^2) :=
  factor_softsign_checkpoint_recovers 20000 32 decay scale hdc hs hcap _ _ original_graph.1 original_graph.2 f hf batch draw

example (batch : ℕ → List (LabeledPosition (Fin 239))) (draw : ComplexParameter (Fin 239) 4 16 6 → ℂ) :=
  original_softsign_checkpoint_recovery divisionAnswer divisionAnswer_mem batch draw (2/3) (3/10)
    (by norm_num) (by norm_num) (by norm_num)

#print axioms original_softsign_checkpoint_recovery

/-- The exact observed mask also identifies the target for the projected
softsign and RMSProp checkpoint searches at their pilot hyperparameters. -/
theorem original_projected_and_rmsProp_recovery (f : DivisionInput → Fin 239)
    (hf : f ∈ factorClass unitToken) (batch : ℕ → List (LabeledPosition (Fin 239)))
    (draw : ComplexParameter (Fin 239) 4 16 6 → ℂ) :
    let S := bitDivisionMask rows
    let hidden : {x : DivisionInput // x ∉ S} → DivisionInput := Subtype.val
    let recover := fun (path : ℕ → Weights (Fin 239) 4 16 6) =>
      PathCheckpointRecovery 1 path (divisionPrompt ∘ hidden) (f ∘ hidden)
        (hiddenCertificate S (factorClass unitToken) (fun x => f x) hidden) ((1/1000000 : ℝ≥0)^2)
    recover (fun j => (projectedSoftsignRun 1 (7/4) (1/10) batch (fun t => 3/10*wsdRate 20000 t)
      draw (256*(32+j))).weights) ∧
    recover (fun j => (rmsPropRun 1 (1/5) batch (wsdRate 20000) draw (256*(32+j))).weights) :=
  ⟨factor_projectedSoftsign_checkpoint_recovers _ _ original_graph.1 original_graph.2 f hf batch draw,
   factor_rmsProp_checkpoint_recovers _ _ original_graph.1 original_graph.2 f hf batch draw⟩

example := original_projected_and_rmsProp_recovery divisionAnswer divisionAnswer_mem
#print axioms original_projected_and_rmsProp_recovery

theorem original_equalMoment_checkpoint_recovery (scale : ℝ) (hs : 0 ≤ scale) (hcap : scale ≤ 10000)
    (f : DivisionInput → Fin 239) (hf : f ∈ factorClass unitToken)
    (batch : ℕ → List (LabeledPosition (Fin 239))) (draw : ComplexParameter (Fin 239) 4 16 6 → ℂ) :
    let S := bitDivisionMask rows
    let hidden : {x : DivisionInput // x ∉ S} → DivisionInput := Subtype.val
    PathCheckpointRecovery 1 (fun j => (equalMomentRun 1 (1/10) batch (fun t => scale*wsdRate 20000 t)
      draw (256*(32+j))).weights) (divisionPrompt ∘ hidden) (f ∘ hidden)
      (hiddenCertificate S (factorClass unitToken) (fun x => f x) hidden) ((1/1000000 : ℝ≥0)^2) :=
  factor_equalMoment_checkpoint_recovers scale hs hcap _ _ original_graph.1 original_graph.2 f hf batch draw

example := original_equalMoment_checkpoint_recovery 3 (by norm_num) (by norm_num) divisionAnswer divisionAnswer_mem
#print axioms original_equalMoment_checkpoint_recovery

/-- For both ordinary optimizers the graph hypotheses are discharged on
the observed split. The common-gradient and structural limit hypotheses
remain explicit in the returned implications. -/
example (batch : ℕ → List (LabeledPosition (Fin 239))) (draw : ComplexParameter (Fin 239) 4 16 6 → ℂ) :=
  equalMoment_wsd_generalizes 1 3 (by norm_num) (by norm_num) batch draw divisionPrompt (bitDivisionMask rows)
    (factorClass unitToken) (factorClass_identified _ _ original_graph.1 original_graph.2)
    divisionAnswer divisionAnswer_mem

example (batch : ℕ → List (LabeledPosition (Fin 239))) :=
  factor_equalMoment_ordinary_restarts 20000 (bitDivisionMask rows) original_graph.1 original_graph.2
    (fun x : ((bitDivisionMask rows)ᶜ : Set DivisionInput) => x.val) divisionAnswer divisionAnswer_mem batch

example (batch : ℕ → List (LabeledPosition (Fin 239))) :=
  factor_uncorrectedAdam_ordinary_restarts 20000 (bitDivisionMask rows) original_graph.1 original_graph.2
    (fun x : ((bitDivisionMask rows)ᶜ : Set DivisionInput) => x.val) divisionAnswer divisionAnswer_mem batch (19/20)

example (batch : ℕ → List (LabeledPosition (Fin 239))) :=
  factor_normalizedAdam_restarts 20000 (bitDivisionMask rows) original_graph.1 original_graph.2
    (fun x : ((bitDivisionMask rows)ᶜ : Set DivisionInput) => x.val) divisionAnswer divisionAnswer_mem batch

theorem original_delayed_normalizedAdam (f : DivisionInput → Fin 239) (hf : f ∈ factorClass unitToken)
    (batch : ℕ → List (LabeledPosition (Fin 239))) (draw : ComplexParameter (Fin 239) 4 16 6 → ℂ) :
    let hidden := fun x : ((bitDivisionMask rows)ᶜ : Set DivisionInput) => x.val
    let score (w : Weights (Fin 239) 4 16 6) x := (Model.fromWeights w).logits 1 (divisionPrompt (hidden x)) (Fin.last 4)
    let center j := (normalizedAdamRun 1 batch (fun k => 3*wsdRate 20000 k) draw (256*j)).weights
    DelayedRecovery 32 (Measure.infinitePi (fun _ : ℕ => Measure.pi (fun _ : Parameter (Fin 239) 4 16 6 =>
      ProbabilityTheory.gaussianReal 0 ((1/1000000 : ℝ≥0)^2)))) center
      (fun w => hiddenCertificate (bitDivisionMask rows) (factorClass unitToken) (fun x => f x) hidden (score w))
      (fun w => StrictOn (score w) (f ∘ hidden)) :=
  factor_delayed_normalizedAdam 32 _ _ original_graph.1 original_graph.2 f hf batch draw

theorem original_delayed_uncorrectedAdam (f : DivisionInput → Fin 239) (hf : f ∈ factorClass unitToken)
    (batch : ℕ → List (LabeledPosition (Fin 239))) (draw : ComplexParameter (Fin 239) 4 16 6 → ℂ) :
    let hidden := fun x : ((bitDivisionMask rows)ᶜ : Set DivisionInput) => x.val
    DelayedHiddenRecovery 1 32 (fun j => (uncorrectedAdamRun 1 (9/10) (19/20) (1/100000000) (1/10)
      batch (fun k => 3*wsdRate 20000 k) draw (256*j)).weights) (divisionPrompt ∘ hidden) hidden
      (bitDivisionMask rows) (factorClass unitToken) f ((1/1000000 : ℝ≥0)^2) :=
  factor_delayed_uncorrectedAdam 32 _ _ original_graph.1 original_graph.2 f hf batch draw

example := equalMoment_uniform_observed_confidence 1 (evaluationGrid 256 20000) (by simp [evaluationGrid])
  divisionPrompt id (bitDivisionMask rows) (factorClass unitToken)
  (factorClass_identified _ _ original_graph.1 original_graph.2)
  (fun _ : ComplexParameter (Fin 239) 4 16 6 => (1/50 : ℝ≥0)^2) (by intro; norm_num)
  Division.Factors.class_native_realization (failure := (1/4 : ℝ≥0∞)) (by norm_num)

example (batch : ℕ → List (LabeledPosition (Fin 239))) (draw : ComplexParameter (Fin 239) 4 16 6 → ℂ) :=
  softsign_wsd_generalizes 1 (3/10) (2/3) (by norm_num) (by norm_num) (by norm_num)
    batch draw divisionPrompt (bitDivisionMask rows) (factorClass unitToken)
    (factorClass_identified _ _ original_graph.1 original_graph.2) divisionAnswer divisionAnswer_mem

example := original_nuclear_completion (u := fun _ => 1) (v := fun _ => 1) (by simp [vecMulVec_apply])
example := original_factor_class_stabilizes 12 (fun _ => 100000) (fun _ => 1) (by intro; norm_num)
example {D : Type*} [MeasurableSpace D] [MeasurableSingletonClass D] [Countable D]
    (π : Measure D) [IsProbabilityMeasure π] (batch : D → ℕ → List (LabeledPosition (Fin 239)))
    (draw : D → ComplexParameter (Fin 239) 4 16 6 → ℂ) :=
  Division.Factors.class_firstNoise_seeded_wsd π 99999 _ original_graph.1 original_graph.2
    divisionAnswer divisionAnswer_mem batch draw

#print axioms original_mask_certifies
#print axioms original_factor_label_orbit
#print axioms original_delayed_normalizedAdam
#print axioms original_delayed_uncorrectedAdam
#print axioms original_nuclear_completion
#print axioms original_factor_class_stabilizes
#print axioms original_firstNoise_factor_recovery
#print axioms observed_matching
end OriginalDivisionMask
"#);
    Ok(out)
}
