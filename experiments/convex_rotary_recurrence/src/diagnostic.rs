//! Training-only replay of frozen ff00f921, followed by feasible frequency probes.
//! CM98 permits every finite real frequency without changing content capacities.
//! No probe is accepted as a trained endpoint and no held-out labels are inspected.
use crate::{
    Result,
    atom::Family,
    experiment,
    train::{self, Config},
};
use baseline::random::Random;
use bilinear::{experiment as original, guard};
use serde_json::{Value, json};
use std::{fs, path::Path};
mod probes;

pub const PRODUCER: &str = "f75861a34af01cca344cc7576c77d1f1f0070fa97d127beb8ed249218d581280";
pub fn fingerprint() -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    for source in [
        include_bytes!("diagnostic.rs").as_slice(),
        include_bytes!("diagnostic/probes.rs"),
        include_bytes!("bin/diagnose.rs"),
        experiment::fingerprint().as_bytes(),
    ] {
        h.update((source.len() as u64).to_le_bytes());
        h.update(source);
    }
    h.finalize().iter().map(|x| format!("{x:02x}")).collect()
}

pub fn run(
    workspace: &Path,
    storage: &Path,
    cohort: &Path,
    output: &Path,
    mode: &str,
) -> Result<()> {
    guard::admit(0, workspace, storage)?;
    let text = experiment::CAUTIOUS_PROTOCOL;
    let (protocol, config) = experiment::protocol(mode, 1, text)?;
    let name = "gpt-probe-key-value-binding";
    let (case, original_row, task) = original::case_in(name, text)?;
    let cell = experiment::score::cell(cohort, name, mode, 1, Family::Rotary);
    let metadata = original::score::read(&cell.join("metadata.json"))?;
    let saved = original::score::read(&cell.join("training_receipt.json"))?;
    let endpoint = fs::read(cell.join("endpoint.json"))?;
    if metadata["source_fingerprint"] != PRODUCER
        || saved["source_fingerprint"] != PRODUCER
        || metadata["protocol"] != protocol
        || metadata["original"] != original_row
        || metadata["case"] != case
        || metadata["mode"] != mode
        || metadata["seed"] != 1
        || metadata["family"] != "rotary"
        || saved["status"] != "sealed"
        || saved["endpoint_sha256"] != original::hash(&endpoint)
    {
        return Err("diagnostic input is not the declared sealed cautious producer".into());
    }
    let trace: Vec<Value> = fs::read_to_string(cell.join("training.jsonl"))?
        .lines()
        .map(serde_json::from_str)
        .collect::<std::result::Result<_, _>>()?;
    let updates = case["maximum_updates"].as_u64().ok_or("updates")? as usize;
    let batch = original_row["case"]["batch"].as_u64().ok_or("batch")? as usize;
    if trace.len() != updates || updates != 128 {
        return Err("incomplete recorded trajectory".into());
    }
    fs::create_dir(output)?;
    let training = &task.splits["train"];
    let mut sampler = Random::new(10_001);
    let mut component_rng = Random::new(20_001);
    let mut order: Vec<_> = (0..training.len()).collect();
    let mut cursor = order.len();
    let mut model = experiment::initial_model(task.vocab, &protocol, Family::Rotary)?;
    let mut last_centre = None;
    let mut last_indices = vec![];
    for (update, expected) in trace.iter().enumerate() {
        if cursor == order.len() {
            sampler.shuffle(&mut order);
            cursor = 0;
        }
        let end = (cursor + batch).min(order.len());
        let indices: Vec<_> = order[cursor..end]
            .iter()
            .flat_map(|&i| (i as u64).to_le_bytes())
            .collect();
        if expected["batch_indices_sha256"] != original::hash(&indices) {
            return Err("replayed original batch changed".into());
        }
        let records = original::records(training, &order[cursor..end])?;
        guard::admit(model.reservation(records.len())?, workspace, storage)?;
        if update + 1 == updates {
            let mut centre = model.clone();
            centre.shrink_for_queries(config.query_step)?;
            last_centre = Some(centre);
            last_indices = order[cursor..end].to_vec();
        }
        let step_config = Config {
            step_ceiling: experiment::step_ceiling(&protocol, update)?,
            ..config
        };
        let receipt = train::step(&mut model, &records, step_config, &mut component_rng)?;
        if serde_json::to_value(&receipt)? != expected["receipt"] {
            return Err(format!(
                "deterministic training receipt differs at update{}",
                update + 1
            )
            .into());
        }
        cursor = end;
        if update == 0 || (update + 1) % 25 == 0 {
            println!("replay mode={mode} update={} receipt=identical", update + 1);
        }
    }
    let mut replayed = serde_json::to_vec(&model)?;
    replayed.push(b'\n');
    if replayed != endpoint {
        return Err("reconstructed checkpoint is not byte-identical".into());
    }
    original::write_json(
        &output.join("replay.json"),
        &json!({"status":"byte-identical",
        "diagnostic_fingerprint":fingerprint(),"producer_fingerprint":PRODUCER,"mode":mode,
        "endpoint_sha256":original::hash(&endpoint),"receipts_matched":trace.len(),
        "original_training_examples":training.len(),"heldout_labels_used":false}),
    )?;
    let centre = last_centre.ok_or("missing final centre")?;
    let records = original::records(training, &last_indices)?;
    let atom = model.atoms.last().ok_or("missing selected component")?;
    let baseline = trace.last().ok_or("missing final receipt")?["receipt"]["component_trial_ce"]
        .as_f64()
        .ok_or("component comparison CE")?;
    probes::run(&centre, atom, &records, config, baseline, output)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{C, atom::Atom, model::Model};
    use bilinear::train::{Mode, Record};
    #[test]
    fn probes_execute_feasible_models_and_retain_finer_perturbations() {
        let mut centre = Model::zero(3, 16., Family::Rotary).unwrap();
        centre.shrink_for_queries(1e-4).unwrap();
        let mut atom = Atom::identity_start(3, Family::Rotary);
        atom.frequency = Some([1., 0.01]);
        atom.transition[1] = [
            [C::default(), C::new(1., 0.)],
            [C::new(1., 0.), C::default()],
        ];
        let records = [
            Record {
                prefix: &[0, 1],
                target: 1,
            },
            Record {
                prefix: &[0, 0, 1],
                target: 2,
            },
        ];
        let config = Config {
            mode: Mode::ForwardMini,
            query_step: 1e-4,
            sweeps: 6,
            line_iterations: 64,
            step_ceiling: 0.1,
            frequency_start: Family::Fixed.initial_frequency(),
        };
        let report = probes::measure(&centre, &atom, &records, config).unwrap();
        assert_eq!(report["directions"].as_array().unwrap().len(), 88);
        assert_eq!(
            report["scope"],
            "last selected component on its original final minibatch"
        );
        let directions = report["directions"].as_array().unwrap();
        assert!(
            directions
                .iter()
                .any(|d| d["radius"].as_f64().unwrap() < 1e-5)
        );
        assert!(
            directions
                .iter()
                .any(|d| d["maximum_state_change"].as_f64().unwrap() > 1e-3)
        );
        assert!(
            directions
                .iter()
                .all(|d| d["model_ce_discrepancy"].as_f64().unwrap() < 2e-12)
        );
    }
}
