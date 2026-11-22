//! Replay trained weights and evaluate the sufficient margin in TransportedValues.lean.
use super::*;

pub fn save(e: &Engine, path: &Path) -> Result<String> {
    let weights = e.parameters.read()?;
    layout::validate(&e.config, &weights)?;
    let pending = path.with_extension("f32.tmp");
    let bytes: Vec<_> = weights.iter().flat_map(|v| v.to_le_bytes()).collect();
    fs::write(&pending, &bytes)?;
    fs::rename(pending, path)?;
    Ok(hash(&bytes))
}

pub(super) fn load(bytes: &[u8], cfg: Config) -> Result<Engine> {
    if bytes.len() != 4 * layout::total_count(&cfg) {
        return Err("wrong checkpoint length".into());
    }
    let weights: Vec<_> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .copied()
        .map(f32::from_le_bytes)
        .collect();
    Engine::new(cfg, &weights)
}

pub fn predict(path: &Path, vocabulary: Option<usize>, a: u32, b: u32) -> Result<()> {
    let mut cfg = config(1)?;
    if architecture::RAW || vocabulary.is_some() {
        cfg.head = architecture::HEAD;
    }
    if let Some(vocab) = vocabulary {
        if vocab == 0
            || vocab > u32::MAX as usize
            || vocab as u64 > fs::metadata(path)?.len() / (4 * cfg.width as u64)
        {
            return Err("vocabulary outside the checkpoint's embedding capacity".into());
        }
        cfg.vocab = vocab;
    }
    let mut e = load(&fs::read(path)?, cfg)?;
    e.forward(&[a, b], &[u32::MAX, u32::MAX], 1, 2)?;
    if vocabulary.is_some() || architecture::RAW {
        println!(
            "{}",
            json!({"a":a,"b":b,"prediction":e.predictions.read()?[1]})
        );
        return Ok(());
    }
    let address = e.predictions.read()?[0];
    ValueLayer::new(&e)?.forward(&e, 1, true)?;
    println!(
        "{}",
        json!({"a":a,"b":b,"address":address,"prediction":e.predictions.read()?[1]})
    );
    Ok(())
}

// Bind a named training checkpoint to the observation that saved it. Several
// learned architectures have the same weight count and distinct forward maps.
pub(super) fn recorded_nv11(path: &Path, digest: &str) -> Result<Option<Value>> {
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let Some((arm, slot)) = ["nv11-train", "shared-values"]
        .into_iter()
        .find_map(|arm| stem.strip_prefix(arm)?.strip_prefix('-').map(|s| (arm, s)))
    else {
        return Ok(None);
    };
    let log = path.with_file_name(format!("{arm}.jsonl"));
    if !log.try_exists()? {
        return Ok(None);
    }
    let step = if slot == "first-exact" {
        None
    } else if let Ok(n) = slot.parse::<u64>() {
        Some(n)
    } else {
        return Ok(None);
    };
    use std::io::{BufRead, BufReader};
    for line in BufReader::new(fs::File::open(&log)?).lines() {
        let row: Value = serde_json::from_str(&line?)?;
        let selected = match step {
            Some(n) => row["step"] == n,
            None => row["train"]["correct"] == 4704 && row["held_out"]["correct"] == 4705,
        };
        if !selected {
            continue;
        }
        if row["arm"] != arm || row["weights_sha256"] != digest {
            return Err("NV11 weight digest differs from the named training observation".into());
        }
        if (arm == "nv11-train" || row.get("model").is_some())
            && row["model"] != architecture::MODEL
        {
            return Err("NV11 checkpoint was trained with a different architecture".into());
        }
        return Ok(Some(row));
    }
    Err("missing named NV11 checkpoint in the training observations".into())
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Nv11Detail {
    Margins,
    Shifts,
    Rows,
}

/// Score all 9,409 pairs using just the two input tokens in the forward pass.
/// Keep the smallest correct-class advantage over every wrong class, including
/// a negative margin when the checkpoint misclassifies an input.
pub fn nv11_scores(path: &Path, detail: Nv11Detail) -> Result<()> {
    let bytes = fs::read(path)?;
    let digest = hash(&bytes);
    let recorded = recorded_nv11(path, &digest)?;
    let data = cubic_data()?;
    let mut cfg = config(BATCH)?;
    cfg.head = architecture::HEAD;
    let p = cfg.vocab;
    if detail != Nv11Detail::Margins && p != 97 {
        return Err("NV11 diagnostics require the 97-class source table".into());
    }
    let mut e = load(&bytes, cfg)?;
    let mut split = |rows: &[[u32; 3]]| -> Result<Value> {
        let mut residuals =
            (detail != Nv11Detail::Margins).then(shift_diagnostics::Residuals::<97>::new);
        let mut correct = 0usize;
        let mut minimum = f64::INFINITY;
        let mut closest = None;
        for batch in rows.chunks(BATCH) {
            let tokens: Vec<_> = batch.iter().flat_map(|&[a, b, _]| [a, b]).collect();
            e.forward(&tokens, &vec![u32::MAX; tokens.len()], batch.len(), 2)?;
            let logits = e.logits.read()?;
            let predictions = e.predictions.read()?;
            for (row, &[a, b, y]) in batch.iter().enumerate() {
                let pos = 2 * row + 1;
                let scores = &logits[p * pos..p * (pos + 1)];
                if scores.iter().any(|s| !s.is_finite()) {
                    return Err("nonfinite NV11 logits".into());
                }
                correct += usize::from(predictions[pos] == y);
                if let Some(diagnostic) = &mut residuals {
                    diagnostic.record(a, b, predictions[pos], y)?;
                }
                let rival = (0..p)
                    .filter(|&j| j != y as usize)
                    .max_by(|&i, &j| scores[i].total_cmp(&scores[j]))
                    .unwrap();
                let margin = f64::from(scores[y as usize]) - f64::from(scores[rival]);
                if margin < minimum {
                    minimum = margin;
                    closest = Some([a, b, y, rival as u32]);
                }
            }
        }
        let mut report = json!({"correct":correct,"examples":rows.len(),"minimum_margin":minimum,
            "closest_pair_target_rival":closest});
        if let Some(diagnostic) = residuals {
            if detail == Nv11Detail::Rows {
                report["row_accuracy"] = diagnostic.row_accuracy();
            } else {
                report["shift_consistency"] = diagnostic.report();
            }
        }
        Ok(report)
    };
    let training = split(&data.training.rows)?;
    let held_out = split(&data.validation)?;
    if let Some(row) = recorded
        && (training["correct"] != row["train"]["correct"]
            || held_out["correct"] != row["held_out"]["correct"]
            || training["examples"] != row["train"]["examples"]
            || held_out["examples"] != row["held_out"]["examples"])
    {
        return Err("reloaded NV11 scores differ from the named training observation".into());
    }
    println!(
        "{}",
        json!({"training":training,"held_out":held_out,
            "weights_sha256":digest,"scored_with_model":architecture::MODEL,
            "scored_with_kernite":KERNITE_VERSION})
    );
    Ok(())
}

/// For each ordered target/rival pair, compute the good-address gap and the
/// smallest gap across all other value codes. Translation permutes the codes.
fn gram_gaps(e: &[f32], p: usize, d: usize) -> Vec<Vec<(f64, f64)>> {
    let gram: Vec<Vec<f64>> = e
        .chunks_exact(d)
        .take(p)
        .map(|u| {
            e.chunks_exact(d)
                .take(p)
                .map(|v| {
                    u.iter()
                        .zip(v)
                        .map(|(&x, &y)| f64::from(x) * f64::from(y))
                        .sum()
                })
                .collect()
        })
        .collect();
    (0..p)
        .map(|t| {
            (0..p)
                .filter(|&c| c != t)
                .map(|c| {
                    let good = gram[t][t] - gram[c][t];
                    let other = (0..p)
                        .filter(|&k| k != t)
                        .map(|k| gram[t][k] - gram[c][k])
                        .fold(f64::INFINITY, f64::min);
                    (good, other)
                })
                .collect()
        })
        .collect()
}

pub fn inspect(path: &Path) -> Result<()> {
    let data = cubic_data()?;
    let mut e = load(&fs::read(path)?, config(BATCH)?)?;
    let (p, d) = (e.config.vocab, e.config.width);
    let layer = ValueLayer::new(&e)?;
    let metrics = evaluate(&mut e, Some((&layer, true)), &data.validation)?;
    let gaps = gram_gaps(&e.parameters.read()?, p, d);
    // Position zero only depends on a; at b=0 the stored value index is s.
    let tokens: Vec<_> = (0..p as u32).flat_map(|a| [a, 0]).collect();
    e.forward(&tokens, &vec![u32::MAX; 2 * p], p, 2)?;
    let scores = e.logits.read()?;
    let mut mass = vec![0.; p];
    for (a, m) in mass.iter_mut().enumerate() {
        let s = &scores[2 * a * p..(2 * a + 1) * p];
        let max = f64::from(s.iter().copied().fold(f32::NEG_INFINITY, f32::max));
        let z: f64 = s.iter().map(|&x| (f64::from(x) - max).exp()).sum();
        *m = (f64::from(s[a * a * a % p]) - max).exp() / z;
    }
    let mut positive = 0;
    let mut zero_products = 0;
    let mut worst = f64::INFINITY;
    for &[a, b, target] in &data.validation {
        zero_products += usize::from((a as usize * b as usize).is_multiple_of(p));
        let m = mass[a as usize];
        let bound = gaps[target as usize]
            .iter()
            .map(|&(good, other)| m * good + (1. - m) * other)
            .fold(f64::INFINITY, f64::min);
        positive += usize::from(bound > 0.);
        worst = worst.min(bound);
    }
    println!(
        "{}",
        json!({"held_out":metrics,"positive_gram_bound":positive,"minimum_gram_bound":worst,
        "zero_product_pairs":zero_products,
        "minimum_cubic_address_mass":mass.into_iter().fold(1.,f64::min)})
    );
    Ok(())
}
