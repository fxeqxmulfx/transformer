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
        "import Transformer.Grokking.FiniteRecovery.FactorFirstNoise\nimport Transformer.Grokking.FiniteRecovery.ProjectedSoftsign\nimport Transformer.Grokking.FiniteRecovery.ArgandRmsProp\nimport Transformer.Grokking.FiniteRecovery.SoftsignLimit\nimport Transformer.Grokking.FiniteRecovery.LabelOrbit\nimport Transformer.Grokking.FiniteRecovery.DelayedCheckpoints\nimport Transformer.Grokking.FiniteRecovery.ProposalMass\n\n",
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
