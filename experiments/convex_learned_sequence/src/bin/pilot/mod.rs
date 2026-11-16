//! Prospective two-mode pilot protocol: all-training seal, independent trace
//! replay, original-object scoring and deterministic independent-run comparison.

mod settings;
mod wire;

use convex_learned_sequence::model::ComplexModel;
use serde::{Deserialize, Serialize};
use settings::*;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;
use wire::Step;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Seal {
    version: usize,
    binary_sha: String,
    records: usize,
    updates_per_mode: usize,
    heldout_pools_scored: usize,
    files: BTreeMap<String, String>,
    endpoints: BTreeMap<String, String>,
}
fn step_name(mode: &str, index: usize) -> String {
    format!("{mode}-{index:04}.json")
}
fn expected_files(count: usize) -> BTreeSet<String> {
    let mut names = BTreeSet::from(["config.json".into(), "training.json".into()]);
    for mode in MODES {
        for index in 0..EPOCHS * count.div_ceil(BATCH) {
            names.insert(step_name(mode, index));
        }
    }
    names
}
fn check_seal(out: &Path) -> Result<(Seal, Config, Vec<Input>)> {
    guard()?;
    let seal: Seal = load(&out.join("training-seal.json"))?;
    let config: Config = load(&out.join("config.json"))?;
    if config != Config::new(&config.revision, &config.source_sha)? {
        return Err("changed pilot configuration".into());
    }
    let inputs: Vec<Input> = load(&out.join("training.json"))?;
    records(&inputs)?;
    let updates = EPOCHS * inputs.len().div_ceil(BATCH);
    if seal.version != 1
        || !hex(&seal.binary_sha, 64)
        || seal.records != inputs.len()
        || seal.updates_per_mode != updates
        || seal.heldout_pools_scored != 0
        || seal.endpoints.len() != MODES.len()
        || seal.files.keys().cloned().collect::<BTreeSet<_>>() != expected_files(inputs.len())
    {
        return Err("incomplete all-training seal".into());
    }
    for mode in MODES {
        if seal.endpoints.get(mode) != Some(&step_name(mode, updates - 1)) {
            return Err("missing mode endpoint".into());
        }
    }
    for (name, expected) in &seal.files {
        if !hex(expected, 64)
            || fs::symlink_metadata(out.join(name))?
                .file_type()
                .is_symlink()
            || hash(&out.join(name))? != *expected
        {
            return Err(format!("sealed artifact mismatch: {name}").into());
        }
    }
    let mut allowed = expected_files(inputs.len());
    allowed.extend([
        "training-seal.json".into(),
        "evaluation.json".into(),
        "score.json".into(),
    ]);
    for entry in fs::read_dir(out)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if !entry.file_type()?.is_file() || !allowed.contains(&name) {
            return Err("unexpected run artifact".into());
        }
    }
    Ok((seal, config, inputs))
}

fn train(input: &Path, out: &Path, revision: &str, source_sha: &str) -> Result<()> {
    guard()?;
    let config = Config::new(revision, source_sha)?;
    let inputs: Vec<Input> = serde_json::from_slice(&load_bytes(input, 1 << 16)?)?;
    let dataset = records(&inputs)?;
    fs::create_dir(out)?;
    save(&out.join("config.json"), &config)?;
    save(&out.join("training.json"), &inputs)?;
    let mut endpoints = BTreeMap::new();
    let updates = EPOCHS * dataset.len().div_ceil(BATCH);
    for mode in MODES {
        let mut current = model()?;
        let mut index = 0;
        for epoch in 0..EPOCHS {
            for (chunk, batch) in dataset.chunks(BATCH).enumerate() {
                guard()?;
                let start = chunk * BATCH;
                let trace = if mode == "mini" {
                    let update =
                        current.mini_step(batch, Some(&rename()), &ratio(1, 32), mini_limits())?;
                    Step::mini(
                        index,
                        epoch,
                        start,
                        batch.len(),
                        &update,
                        mean_loss(&update.projected.model, batch)?,
                    )
                } else {
                    let update = current.forward_mini_step(
                        batch,
                        Some(&rename()),
                        &forward_config(),
                        forward_limits(),
                    )?;
                    Step::forward(index, epoch, start, batch.len(), &update)
                };
                let path = out.join(step_name(mode, index));
                save(&path, &trace)?;
                // Replay the persisted representation, including every forward
                // query, before accepting the next training state.
                let persisted: Step = load(&path)?;
                current = persisted.replay(&current, batch)?;
                index += 1;
                println!(
                    "{mode} {index}/{updates} batch_ce={:.10}",
                    f64::from_bits(trace.final_loss)
                );
            }
        }
        endpoints.insert(mode.into(), step_name(mode, updates - 1));
    }
    let mut files = BTreeMap::new();
    for name in expected_files(dataset.len()) {
        files.insert(name.clone(), hash(&out.join(name))?);
    }
    let seal = Seal {
        version: 1,
        binary_sha: hash(&std::env::current_exe()?)?,
        records: dataset.len(),
        updates_per_mode: updates,
        heldout_pools_scored: 0,
        files,
        endpoints,
    };
    save(&out.join("training-seal.json"), &seal)?;
    check_seal(out)?;
    println!("sealed both complete training modes; heldout pools scored: 0");
    Ok(())
}

fn replay_training(out: &Path) -> Result<(Seal, Config, BTreeMap<String, ComplexModel>)> {
    let (seal, config, inputs) = check_seal(out)?;
    let dataset = records(&inputs)?;
    let mut endpoints = BTreeMap::new();
    for mode in MODES {
        let mut current = model()?;
        let mut index = 0;
        for epoch in 0..EPOCHS {
            for (chunk, batch) in dataset.chunks(BATCH).enumerate() {
                guard()?;
                let trace: Step = load(&out.join(step_name(mode, index)))?;
                if trace.mode != mode
                    || trace.index != index
                    || trace.epoch != epoch
                    || trace.start != chunk * BATCH
                    || trace.count != batch.len()
                {
                    return Err("changed indexed minibatch sequence".into());
                }
                current = trace.replay(&current, batch)?;
                index += 1;
            }
        }
        endpoints.insert(mode.into(), current);
    }
    Ok((seal, config, endpoints))
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Raw {
    tokens: Vec<usize>,
    target: usize,
    predicted: usize,
    ce_bits: u64,
    probabilities_bits: Vec<u64>,
    complex_scores_bits: Vec<[u64; 2]>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Score {
    version: usize,
    training_seal_sha: String,
    evaluation_sha: String,
    modes: BTreeMap<String, Vec<Raw>>,
}
fn evaluate(
    models: &BTreeMap<String, ComplexModel>,
    inputs: &[Input],
) -> Result<BTreeMap<String, Vec<Raw>>> {
    let dataset = records(inputs)?;
    let mut result = BTreeMap::new();
    for mode in MODES {
        let model = models.get(mode).ok_or("missing trained mode")?;
        let mut raw = Vec::with_capacity(dataset.len());
        for record in &dataset {
            let output = model.paired_forward(&record.tokens, &rename())?;
            raw.push(Raw {
                tokens: record.tokens.clone(),
                target: record.target,
                predicted: output.object(),
                ce_bits: output.cross_entropy(record.target)?.to_bits(),
                probabilities_bits: output.probabilities().iter().map(|p| p.to_bits()).collect(),
                complex_scores_bits: output
                    .complex_scores()
                    .iter()
                    .map(|z| [z.re.to_bits(), z.im.to_bits()])
                    .collect(),
            });
        }
        result.insert(mode.into(), raw);
    }
    Ok(result)
}
fn score(out: &Path, evaluation: &Path) -> Result<()> {
    // This precedes even opening the heldout data path. Both full training
    // trajectories and every persisted query/certificate must replay first.
    let (_, _, models) = replay_training(out)?;
    let inputs: Vec<Input> = serde_json::from_slice(&load_bytes(evaluation, 1 << 16)?)?;
    let raw = evaluate(&models, &inputs)?;
    save(&out.join("evaluation.json"), &inputs)?;
    let score = Score {
        version: 1,
        training_seal_sha: hash(&out.join("training-seal.json"))?,
        evaluation_sha: hash(&out.join("evaluation.json"))?,
        modes: raw,
    };
    save(&out.join("score.json"), &score)?;
    println!(
        "scored all {} original evaluation records in both modes",
        inputs.len()
    );
    Ok(())
}
fn replay_score(out: &Path) -> Result<(Seal, Config, Score)> {
    let (seal, config, models) = replay_training(out)?;
    let inputs: Vec<Input> = load(&out.join("evaluation.json"))?;
    let score: Score = load(&out.join("score.json"))?;
    if score.version != 1
        || score.training_seal_sha != hash(&out.join("training-seal.json"))?
        || score.evaluation_sha != hash(&out.join("evaluation.json"))?
        || score.modes != evaluate(&models, &inputs)?
    {
        return Err("raw prediction/CE replay mismatch".into());
    }
    Ok((seal, config, score))
}
#[derive(Serialize)]
struct Count {
    correct: usize,
    total: usize,
    mean_ce: f64,
}
#[derive(Serialize)]
struct Verified {
    version: usize,
    revision: String,
    source_sha: String,
    original_binary_sha: String,
    repeat_binary_sha: String,
    matched_training_artifacts: usize,
    updates_per_mode: usize,
    full_training_and_raw_replay: bool,
    modes: BTreeMap<String, Count>,
}
fn verify(original: &Path, repeat: &Path, report: &Path) -> Result<()> {
    let (a, ac, ar) = replay_score(original)?;
    let (b, bc, br) = replay_score(repeat)?;
    if ac != bc
        || a.files != b.files
        || a.endpoints != b.endpoints
        || ar.evaluation_sha != br.evaluation_sha
        || ar.modes != br.modes
    {
        return Err("independent run mismatch".into());
    }
    let mut modes = BTreeMap::new();
    for (mode, raw) in &ar.modes {
        let correct = raw.iter().filter(|r| r.predicted == r.target).count();
        let total = raw.len();
        let mean_ce = raw.iter().map(|r| f64::from_bits(r.ce_bits)).sum::<f64>() / total as f64;
        modes.insert(
            mode.clone(),
            Count {
                correct,
                total,
                mean_ce,
            },
        );
        println!("verified {mode}: {correct}/{total}, mean CE={mean_ce:.10}");
    }
    save(
        report,
        &Verified {
            version: 1,
            revision: ac.revision,
            source_sha: ac.source_sha,
            original_binary_sha: a.binary_sha,
            repeat_binary_sha: b.binary_sha,
            matched_training_artifacts: a.files.len(),
            updates_per_mode: a.updates_per_mode,
            full_training_and_raw_replay: true,
            modes,
        },
    )
}

pub fn run(args: Vec<String>) -> Result<()> {
    match args.as_slice() {
        [cmd, input, out, revision, source] if cmd == "train" => train(Path::new(input), Path::new(out), revision, source),
        [cmd, out, input] if cmd == "score" => score(Path::new(out), Path::new(input)),
        [cmd, original, repeat, report] if cmd == "verify" => verify(Path::new(original), Path::new(repeat), Path::new(report)),
        _ => Err("usage: train TRAIN OUT REVISION SOURCE_SHA | score OUT EVAL | verify ORIGINAL REPEAT REPORT".into()),
    }
}

#[cfg(test)]
mod tests;
