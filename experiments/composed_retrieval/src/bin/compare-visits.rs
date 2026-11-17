//! Compare the sealed sparse control and the matched-visit cohort, including
//! per-object decisions from both trained modes and OS-measured fit costs.
use baseline::{channel::argmax, hash};
use bilinear::guard;
use composed_retrieval::{Result, data};
use learned_retrieval::cohort::{raw_file, read};
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs, path::Path};

#[derive(Default)]
struct Decisions {
    rows: usize,
    different: usize,
    max_logit_delta: f64,
}
impl Decisions {
    fn add(&mut self, a: &[u8], b: &[u8], v: usize) -> Result<()> {
        if v == 0 || a.is_empty() || a.len() != b.len() || !a.len().is_multiple_of(8 * v) {
            return Err("unaligned raw logits".into());
        }
        let decode = |row: &[u8]| {
            row.as_chunks::<8>()
                .0
                .iter()
                .map(|b| f64::from_le_bytes(*b))
                .collect::<Vec<_>>()
        };
        for (a, b) in a.chunks_exact(v * 8).zip(b.chunks_exact(v * 8)) {
            let (a, b) = (decode(a), decode(b));
            if a.iter().chain(&b).any(|x| !x.is_finite()) {
                return Err("nonfinite raw score".into());
            }
            self.rows += 1;
            self.different += usize::from(argmax(&a) != argmax(&b));
            for (x, y) in a.iter().zip(b) {
                self.max_logit_delta = self.max_logit_delta.max((x - y).abs());
            }
        }
        Ok(())
    }
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let [workspace, storage, sparse, dense] = args.as_slice() else {
        return Err("expected WORKSPACE STORAGE SPARSE_OUTPUT DENSE_OUTPUT".into());
    };
    guard::admit(1 << 30, Path::new(workspace), Path::new(storage))?;
    let (sparse, dense) = (Path::new(sparse), Path::new(dense));
    let p = data::protocol()?;
    if p.coordinates_per_table.is_some() {
        return Err("compile with the dense-forward feature".into());
    }
    let mut report = BTreeMap::new();
    for (id, spec) in &p.cases {
        let (task, _) = data::original(&p, id)?;
        let mut decisions = Decisions::default();
        let mut resources = BTreeMap::<String, Vec<Value>>::new();
        let mut queries = 0;
        for arm in ["original", "repeat"] {
            let first = sparse.join(arm).join("run");
            let second = dense.join(arm).join("run");
            let seals = [
                read(&first.join("sealed/manifest.json"))?,
                read(&second.join("sealed/manifest.json"))?,
            ];
            if seals[0]["identity"]["protocol_sha256"] != hash(data::BASE_PROTOCOL.as_bytes())
                || seals[1]["identity"]["protocol_sha256"] != hash(data::PROTOCOL.as_bytes())
            {
                return Err("incorrect comparison protocols".into());
            }
            for mode in &p.modes {
                for seed in &p.seeds {
                    let cell = format!("{mode}-{seed}");
                    let key = format!("{id}/{cell}");
                    let receipts = [
                        fs::read(first.join(&key).join("receipt.json"))?,
                        fs::read(second.join(&key).join("receipt.json"))?,
                    ];
                    let mut training = vec![];
                    for (i, bytes) in receipts.iter().enumerate() {
                        let r: Value = serde_json::from_slice(bytes)?;
                        if seals[i]["cells"][&key] != hash(bytes)
                            || r["identity"] != seals[i]["identity"]
                        {
                            return Err("unsealed control receipt".into());
                        }
                        training.push(r["training"].clone());
                    }
                    if training[0]["batch_sha256"] != training[1]["batch_sha256"]
                        || training[0]["before"] != training[1]["before"]
                        || training[1]["min_visits"] != p.updates
                        || training[1]["max_visits"] != p.updates
                    {
                        return Err(format!(
                            "batch order, initializer or clocks differ at {arm}/{key}"
                        )
                        .into());
                    }
                    let model = fs::read(second.join(&key).join("model.json"))?;
                    if mode == "mini" {
                        if training[0] != training[1]
                            || fs::read(first.join(&key).join("model.json"))? != model
                        {
                            return Err("minibatch control differs".into());
                        }
                    } else {
                        let m: composed_retrieval::model::Model = serde_json::from_slice(&model)?;
                        queries = 2 * m.weights.len() * p.updates;
                        if training[1]["forward_loss_queries"] != queries {
                            return Err("dense loss query count differs".into());
                        }
                    }
                    let timing =
                        fs::read_to_string(dense.join(arm).join(format!("fit-{id}-{cell}.time")))?;
                    let t = timing
                        .split_whitespace()
                        .map(str::parse::<f64>)
                        .collect::<std::result::Result<Vec<_>, _>>()?;
                    let [elapsed, user, system, rss] = t.as_slice() else {
                        return Err("missing OS resource measurement".into());
                    };
                    resources.entry(mode.clone()).or_default().push(json!({"arm":arm,"seed":seed,"elapsed_s":elapsed,"cpu_s":user+system,"rss_kib":rss}));
                    let old_scores = first.join(id).join("scores");
                    let new_scores = second.join(id).join("scores");
                    for split in task.splits.keys() {
                        let initial = format!("{cell}-initial");
                        if fs::read(raw_file(&old_scores, &initial, split))?
                            != fs::read(raw_file(&new_scores, &initial, split))?
                        {
                            return Err("initial raw logits differ".into());
                        }
                        if mode == "mini" {
                            let end = format!("{cell}-final");
                            let a = fs::read(raw_file(&new_scores, &end, split))?;
                            if a != fs::read(raw_file(&old_scores, &end, split))? {
                                return Err("standard minibatch raw logits differ".into());
                            }
                            if arm == "original" {
                                let b = fs::read(raw_file(
                                    &new_scores,
                                    &format!("forward-{seed}-final"),
                                    split,
                                ))?;
                                decisions.add(&a, &b, spec.vocabulary)?;
                            }
                        }
                    }
                }
            }
        }
        report.insert(id,json!({"mode_comparison_rows":decisions.rows,"differing_predictions":decisions.different,
            "max_logit_delta":decisions.max_logit_delta,"forward_loss_queries_per_fit":queries,"resources":resources}));
    }
    serde_json::to_writer_pretty(std::io::stdout().lock(), &report)?;
    println!();
    Ok(())
}
