//! Export every answer-position logit bit for independent Lean kernel checks.
//! Inputs are in canonical numerator/denominator order. Unknown labels are
//! reconstructed from observations; the generated proof checks division again.
use super::{Completion, N, bitplanes::Planes, checkpoint_with_batch};
use crate::{Result, division, division_equations as eq, hash, margins, metrics, suite::publish};
use serde_json::{Value, json};
use std::{fmt::Write, fs, path::Path};

pub(crate) fn export(dir: &Path, first: bool) -> Result<Value> {
    let (record, mut engine) = checkpoint_with_batch(dir, first, 32)?;
    let vocab = engine.config.vocab;
    let completion = Completion::original_split()?;
    let examples = completion.examples();
    let mut packed = metrics::Packed::default();
    let kind = if first { "first-strict" } else { "terminal" };
    let modules = format!("recorded-{kind}-lean");
    let root = dir.join(&modules);
    let blocks = root.join("RecordedDivisionBlocks");
    fs::create_dir_all(&blocks)?;
    let lock = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(root.join("export.lock"))?;
    lock.lock()?;
    let mut header = String::new();
    writeln!(
        header,
        "/- Weights SHA-256: {}. Training equations SHA-256: {}. -/",
        record["weights_sha256"]
            .as_str()
            .ok_or("missing checkpoint digest")?,
        division::TRAIN_HASH
    )?;
    header.push_str("namespace RecordedDivision\nopen Transformer.Grokking.FiniteRecovery.Binary32Scores\nset_option maxRecDepth 8192\nset_option maxHeartbeats 16000000\n\n");
    let mut source = String::new();
    let mut hashes = Vec::new();
    let mut raw = Vec::with_capacity(4 * vocab * examples.len());
    for a in 0..eq::P {
        let mut block =
            String::from("import Transformer.Grokking.FiniteRecovery.Binary32Scores\n\n");
        block.push_str(&header);
        let indices: Vec<_> = ((a * N) as u32..((a + 1) * N) as u32).collect();
        for batch in indices.chunks(engine.config.max_batch) {
            packed.fill(&examples, batch);
            let logits = margins::masked_logits(&mut engine, &packed)?;
            for (&i, seq) in batch.iter().zip(logits.chunks_exact(vocab * packed.length)) {
                let b = (i as usize) % N + 1;
                let quotient = completion.label(a, b);
                let z = &seq[eq::ANSWER_POS * vocab..(eq::ANSWER_POS + 1) * vocab];
                let y = eq::NUM0 + quotient;
                if z.iter().any(|x| !x.is_finite())
                    || z.iter().enumerate().any(|(j, x)| j != y && *x >= z[y])
                {
                    return Err(format!(
                        "strict table certificate fails at numerator {a}, denominator {b}"
                    )
                    .into());
                }
                raw.extend(z.iter().flat_map(|x| x.to_le_bytes()));
                write!(block, "private noncomputable def row{a}_{b} : Row := ")?;
                Planes::encode(z)?.write_row(&mut block, quotient)?;
                block.push('\n');
            }
        }
        write!(block, "noncomputable def data{a} : Array Row := #[")?;
        for b in 1..=N {
            if b > 1 {
                block.push(',');
            }
            write!(block, "row{a}_{b}")?;
        }
        writeln!(
            block,
            "]\ntheorem checked{a} : goodBlock {a} data{a} = true := by decide +kernel\nend RecordedDivision"
        )?;
        let file = format!("B{a:02}.lean");
        hashes.push(json!({"file":file,"sha256":hash(block.as_bytes())}));
        publish(&blocks, &file, block.as_bytes())?;
        writeln!(source, "import RecordedDivisionBlocks.B{a:02}")?;
    }
    source.push_str(&header);
    source.push_str("noncomputable def blocks : (a : Fin 97) → Block a.val\n");
    for a in 0..eq::P {
        writeln!(source, "  | ⟨{a}, _⟩ => ⟨data{a}, checked{a}⟩")?;
    }
    source.push_str("  | ⟨n+97, h⟩ => by omega\n\ntheorem all_recorded_quotients (a : Fin 97) (b : Fin 96) :\n  let r := (blocks a).val[b.val]!\n  let q := fieldQuotient a b\n  ∀ j : Fin 239, j.val ≠ 22+q → rowNumerator r j.val < rowNumerator r (22+q) :=\n  quotient_wins blocks a b\n\n#print axioms all_recorded_quotients\nend RecordedDivision\n");
    let file = format!("recorded-{kind}-division.lean");
    let report = json!({"weights_sha256":record["weights_sha256"],"step":record["step"],
        "numerators":eq::P,"denominators":N,"vocab":vocab,"recorded_logits":raw.len()/4,
        "logits_little_endian_sha256":hash(&raw),"lean_file":file,"lean_sha256":hash(source.as_bytes()),
        "modules":modules,"blocks":hashes});
    publish(dir, &file, source.as_bytes())?;
    publish(
        dir,
        &format!("recorded-{kind}-division.json"),
        &serde_json::to_vec_pretty(&report)?,
    )?;
    Ok(report)
}
