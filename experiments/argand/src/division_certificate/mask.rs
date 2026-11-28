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
    let matching = completion
        .graph
        .perfect_matching
        .ok_or("no perfect unit matching")?;
    let mut inverse = [0; N];
    for (i, &j) in matching.iter().enumerate() {
        inverse[j] = i;
    }
    let mut out = String::from("import Transformer.Grokking.FiniteRecovery.FactorFirstNoise\n\n");
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

example := original_nuclear_completion (u := fun _ => 1) (v := fun _ => 1) (by simp [vecMulVec_apply])
example := original_factor_class_stabilizes 12 (fun _ => 100000) (fun _ => 1) (by intro; norm_num)
example {D : Type*} [MeasurableSpace D] [MeasurableSingletonClass D] [Countable D]
    (π : Measure D) [IsProbabilityMeasure π] (batch : D → ℕ → List (LabeledPosition (Fin 239)))
    (draw : D → ComplexParameter (Fin 239) 4 16 6 → ℂ) :=
  Division.Factors.class_firstNoise_seeded_wsd π 99999 _ original_graph.1 original_graph.2
    divisionAnswer divisionAnswer_mem batch draw

#print axioms original_mask_certifies
#print axioms original_nuclear_completion
#print axioms original_factor_class_stabilizes
#print axioms original_firstNoise_factor_recovery
#print axioms observed_matching
end OriginalDivisionMask
"#);
    Ok(out)
}
