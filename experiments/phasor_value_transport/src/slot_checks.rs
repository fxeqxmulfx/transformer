//! Gradient fixtures and independent ablations of saved slot-memory weights.
use super::*;

// Keep the content dictionaries separated while exercising the continuous
// read parameters. Random directions across a hard routing boundary do not
// estimate the local adjoint on either side.
pub(super) fn fixture(cfg: &Config, w: &mut [f32]) {
    slot_memory::initialize(cfg, 907, w);
    let l = slot_memory::Layout::new(cfg);
    let w = &mut w[architecture::memory_start(cfg)..];
    let mut rng = random::Random::new(137);
    for x in &mut w[l.values..l.gates] {
        *x = rng.gauss(0., 0.002) as f32;
    }
    for x in &mut w[l.gates..l.bias + 4] {
        *x = rng.gauss(0., 0.08) as f32;
    }
}

struct Scores {
    logits: Vec<f32>,
    written: usize,
    correct: usize,
}

fn score(e: &mut Engine, rows: &[[u32; 3]], names: &[usize]) -> Result<Scores> {
    let mut written = 0;
    let scored = dense_audit::score(e, rows, names, |e| {
        written += e.written_slot_reads()?;
        Ok(())
    })?;
    Ok(Scores {
        logits: scored.logits,
        written,
        correct: scored.correct,
    })
}

fn counts(e: &mut Engine, splits: [&[[u32; 3]]; 2], names: &[usize]) -> Result<[usize; 2]> {
    let mut c = [0; 2];
    for (i, rows) in splits.into_iter().enumerate() {
        c[i] = score(e, rows, names)?.correct;
    }
    Ok(c)
}

pub(super) fn audit(path: &Path) -> Result<()> {
    if !architecture::TRIPLE_TRANSPORT {
        return Err("this audit requires triple-transport".into());
    }
    let bytes = fs::read(path)?;
    let digest = hash(&bytes);
    let trained =
        snapshot::recorded_nv11(path, &digest)?.ok_or("expected named training checkpoint")?;
    let mut cfg = config(BATCH)?;
    cfg.head = architecture::HEAD;
    let mut engine = snapshot::load(&bytes, cfg.clone())?;
    let w = engine.parameters.read()?;
    let l = slot_memory::Layout::new(&cfg);
    let offset = architecture::memory_start(&cfg);
    let data = cubic_data()?;
    let splits = [data.training.rows.as_slice(), data.validation.as_slice()];
    let names: Vec<_> = (0..cfg.vocab).collect();
    let training = score(&mut engine, splits[0], &names)?;
    let held_out = score(&mut engine, splits[1], &names)?;
    let written = [training.written, held_out.written];
    let full = [training.correct, held_out.correct];
    let reference = [training.logits, held_out.logits];
    if trained["train"]["correct"] != full[0] || trained["held_out"]["correct"] != full[1] {
        return Err("audit reproduction differs from the saved observation".into());
    }
    let mut rng = random::Random::new(73);
    let mut slots: Vec<_> = (0..l.slots).collect();
    let mut vocabulary = names.clone();
    rng.shuffle(&mut slots);
    rng.shuffle(&mut vocabulary);
    let mut results = serde_json::Map::new();
    for (name, change_slots, change_words) in [
        ("permuted_slots", true, false),
        ("permuted_vocabulary", false, true),
    ] {
        let mut changed = w.clone();
        if change_slots {
            for (i, &s) in slots.iter().enumerate() {
                let start = offset + l.dictionary + s * l.d;
                changed[start..start + l.d].copy_from_slice(l.prototype(&w[offset..], i));
                for (j, &t) in slots.iter().enumerate() {
                    let at = offset + l.values + (s * l.slots + t) * l.d;
                    changed[at..at + l.d].copy_from_slice(l.row(&w[offset..], i * l.slots + j));
                }
            }
        }
        if change_words {
            for (i, &s) in vocabulary.iter().enumerate() {
                changed[s * l.d..(s + 1) * l.d].copy_from_slice(&w[i * l.d..(i + 1) * l.d]);
            }
        }
        engine.parameters.write(&changed)?;
        let rename = if change_words { &vocabulary } else { &names };
        for (i, rows) in splits.iter().enumerate() {
            let changed = score(&mut engine, rows, rename)?;
            if !changed
                .logits
                .iter()
                .zip(&reference[i])
                .all(|(a, b)| a.to_bits() == b.to_bits())
            {
                return Err(format!("{name} changed a logit on split {i}").into());
            }
        }
        results.insert(
            name.into(),
            json!({"bit_identical_logits":reference.iter().map(Vec::len).sum::<usize>()}),
        );
    }
    for (name, disabled) in [
        ("disable_transport", &[3][..]),
        ("disable_both_structural_reads", &[2, 3][..]),
        ("only_structural_reads", &[0, 1][..]),
    ] {
        let mut changed = w.clone();
        for &i in disabled {
            let g = offset + l.gates + i * l.d;
            changed[g..g + l.d].fill(0.);
            changed[offset + l.bias + i] = if i < 2 { -1. } else { 0. };
        }
        engine.parameters.write(&changed)?;
        results.insert(name.into(), json!(counts(&mut engine, splits, &names)?));
    }
    let cell_count = w[offset + l.values..offset + l.gates]
        .chunks_exact(l.d)
        .filter(|v| v.iter().any(|&x| x != 0.))
        .count();
    println!(
        "{}",
        json!({"model":architecture::MODEL,"weights_sha256":digest,"step":trained["step"],
        "baseline":full,"nonzero_memory_vectors":cell_count,"written_direct_reads":written,
        "slots":l.slots,"checks":results})
    );
    Ok(())
}
