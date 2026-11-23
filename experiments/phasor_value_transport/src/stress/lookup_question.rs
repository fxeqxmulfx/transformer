//! Reload a synthetic word-tokenized lookup checkpoint and score its generated answer letters.
use super::*;

mod data;
use data::vocabulary;
pub(super) use data::{NAME, data};

pub fn score(seed: u32, out: &Path, ablations: bool) -> Result<()> {
    let (row, task) = data()?;
    let cfg = task_config(&row, &task, true)?;
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
        let (result, weights) = embedding::verified_weights(&row, seed, &cfg, &dir)?;
        (weights, result["observation"].clone())
    } else {
        let mut engine = Engine::new(cfg.clone(), &layout::initialize_model(&cfg, seed))?;
        let state = progress::Progress::load(&dir, &row, seed, &mut engine)?
            .ok_or("missing question checkpoint")?;
        (
            engine.parameters.read()?,
            state.observation(state.completed)?,
        )
    };
    drop(reader);
    let hash = hash(
        &weights
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    let words = vocabulary();
    let mut controls = vec![("full", cfg.head)];
    if ablations && let Some(read) = cfg.head.peer_read() {
        controls.extend(
            read.induction_controls()
                .into_iter()
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
        if name == "full" && scores[&task.selection]["correct"] != point["correct"] {
            return Err("held-out replay changed".into());
        }
        results.insert(name, scores);
    }
    println!(
        "{}",
        json!({"study":NAME,"model":MODEL,"seed":seed,"step":point["step"],"weights_sha256":hash,
        "protocol_sha256":row["catalog_sha256"],"kernite":KERNITE_VERSION,"scores":results})
    );
    Ok(())
}
