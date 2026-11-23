//! Reload a synthetic word-tokenized lookup checkpoint and score its generated answer letters.
use super::*;

mod coefficients;
mod data;
pub mod fit;
pub mod isolated;
use data::vocabulary;
pub(super) use data::{NAME, data};

fn checkpoint(seed: u32, out: &Path, row: &Value, cfg: &Config) -> Result<(Vec<f32>, Value)> {
    let dir = out.join(NAME).join(format!("seed{seed}"));
    let reader = match fs::File::open(dir.join("cell.lock")) {
        Ok(file) => {
            file.lock_shared()?;
            Some(file)
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => return Err(e.into()),
    };
    let (weights, point) = if dir.join("result.json").try_exists()? {
        let (result, weights) = embedding::verified_weights(row, seed, cfg, &dir)?;
        (weights, result["observation"].clone())
    } else {
        progress::Progress::frozen(&dir, row, seed, cfg)?
    };
    drop(reader);
    Ok((weights, point))
}

fn weight_digest(weights: &[f32]) -> String {
    hash(
        &weights
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}

pub fn score(seed: u32, out: &Path, ablations: bool) -> Result<()> {
    score_with(seed, out, ablations, None)
}

pub fn score_with(
    seed: u32,
    out: &Path,
    ablations: bool,
    transform: Option<(&str, &str)>,
) -> Result<()> {
    let (row, task) = data()?;
    let cfg = task_config(&row, &task, true)?;
    let (mut weights, point) = checkpoint(seed, out, &row, &cfg)?;
    let hash = weight_digest(&weights);
    let mut intervention = None;
    if let Some((selection, gain)) = transform {
        let range = coefficients::table_range(&cfg)?;
        let mut info = coefficients::transform(&mut weights[range], selection, gain)?;
        info["effective_weights_sha256"] = json!(weight_digest(&weights));
        intervention = Some(info);
    }
    let words = vocabulary();
    let mut controls = vec![("full", cfg.head)];
    if ablations && let Some(read) = cfg.head.peer_read() {
        controls.extend(
            read.induction_controls()
                .into_iter()
                .filter(|(_, variant)| {
                    read.path.is_none() || variant.path != read.path
                })
                .map(|(name, read)| (name, head::Head::PeerTriplePhase(read))),
        );
    }
    let mut results = BTreeMap::new();
    for (name, head) in controls {
        let mut config = cfg.clone();
        config.head = head;
        let mut engine = Engine::for_scoring(config, &weights)?;
        let mut scores = BTreeMap::new();
        for (split, examples) in task.splits.iter().filter(|(s, _)| s.as_str() != "train") {
            let mut response = None;
            let counts =
                metrics::evaluate_with_completions(&mut engine, examples, true, |_, tokens| {
                    if split == "test/mira" {
                        response = Some(
                            tokens
                                .iter()
                                .map(|i| words[*i as usize].as_str())
                                .collect::<Vec<_>>()
                                .join(" "),
                        );
                    }
                    Ok(())
                })?;
            let mut entry = json!({"correct":counts["exact"],"total":examples.len()});
            if split == "test/mira" {
                entry["prediction"] = json!(response.ok_or("fixture produced no response")?);
            }
            scores.insert(split, entry);
        }
        if transform.is_none()
            && name == "full"
            && scores[&task.selection]["correct"] != point["correct"]
        {
            return Err("held-out replay changed".into());
        }
        results.insert(name, scores);
    }
    let mut report = json!({"study":NAME,"model":MODEL,"seed":seed,"step":point["step"],"weights_sha256":hash,
        "protocol_sha256":row["catalog_sha256"],"kernite":KERNITE_VERSION,"scores":results});
    if let Some(info) = intervention {
        report["coefficients"] = info;
    }
    println!("{report}");
    Ok(())
}

pub fn interventions(seed: u32, out: &Path) -> Result<()> {
    let (row, task) = data()?;
    let cfg = task_config(&row, &task, true)?;
    let (weights, point) = checkpoint(seed, out, &row, &cfg)?;
    let digest = weight_digest(&weights);
    let mut e = Engine::for_scoring(cfg, &weights)?;
    let paired = data::paired(&task)?;
    let letter = |token: i64| paired.letters.iter().position(|&v| v == token);
    let mut totals = BTreeMap::new();
    for (hops, group) in paired.groups {
        let mut output = Vec::new();
        for rows in &group {
            let mut labels = Vec::with_capacity(rows.len());
            let metric = metrics::evaluate_with_completions(&mut e, rows, true, |i, tokens| {
                if i != labels.len() {
                    return Err("paired predictions changed example order".into());
                }
                labels.push(match tokens {
                    [v] => letter(i64::from(*v)),
                    _ => None,
                });
                Ok(())
            })?;
            let count = labels
                .iter()
                .zip(rows)
                .filter(|(p, row)| p.is_some() && **p == letter(row.answer[0]))
                .count();
            if rows.len() != labels.len() || metric["exact"] != count {
                return Err("paired metric disagrees with original exact scorer".into());
            }
            output.push(labels);
        }
        let base = &output[0];
        let mut scores = BTreeMap::new();
        for (arm, name) in [
            "original",
            "rows_reversed",
            "first_table_rewired",
            "choices_rotated",
        ]
        .into_iter()
        .enumerate()
        {
            let mut correct = 0;
            let mut both_correct = 0;
            let mut relation = 0;
            let mut valid_pairs = 0;
            for (i, &prediction) in output[arm].iter().enumerate() {
                let ok = prediction.is_some() && prediction == letter(group[arm][i].answer[0]);
                correct += usize::from(ok);
                both_correct += usize::from(ok && base[i] == letter(group[0][i].answer[0]));
                if let (Some(a), Some(b)) = (base[i], prediction) {
                    valid_pairs += 1;
                    relation += usize::from(match arm {
                        2 => a != b,
                        3 => (a + 4) % 5 == b,
                        _ => a == b,
                    });
                }
            }
            scores.insert(name,json!({"correct":correct,"both_correct":both_correct,"expected_relation":relation,"valid_letter_pairs":valid_pairs,"total":base.len()}));
        }
        totals.insert(hops, scores);
    }
    println!(
        "{}",
        json!({"model":MODEL,"seed":seed,"step":point["step"],"weights_sha256":digest,"suite_sha256":row["catalog_sha256"],"paired_sha256":paired.digest,"scored_with_kernite":KERNITE_VERSION,"by_hop":totals})
    );
    Ok(())
}
