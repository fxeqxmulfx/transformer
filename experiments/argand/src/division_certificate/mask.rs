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
    let mut out = String::from("import Transformer.Grokking.FiniteRecovery.DivisionMask\n\n");
    writeln!(
        out,
        "/- Observed equations SHA-256: {} -/",
        division::TRAIN_HASH
    )?;
    out.push_str("namespace OriginalDivisionMask\nopen Transformer.Grokking.FiniteRecovery Transformer.Grokking.FiniteRecovery.Argand\nset_option maxRecDepth 8192\nset_option maxHeartbeats 2000000\n");
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
    out.push_str("theorem original_mask_certifies (score : DivisionInput → Fin 239 → ℝ) :\n    classCertificate (bitDivisionMask rows) (factorClass unitToken) (fun x => divisionAnswer x) score ↔ StrictOn score divisionAnswer :=\n");
    writeln!(
        out,
        "  bitMask_certifies_division rows bridge cover {} (by decide) (by decide) (by decide) score",
        observed[0].trailing_zeros()
    )?;
    out.push_str("#print axioms original_mask_certifies\n#print axioms observed_matching\nend OriginalDivisionMask\n");
    Ok(out)
}
