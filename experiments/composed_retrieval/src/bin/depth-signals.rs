//! Training-only replay of the original sparse two-hop experiment. The
//! observer checks the executed depth estimate without changing any update.
#[path = "depth-signals/measurement.rs"]
mod measurement;
use baseline::hash;
use bilinear::{experiment::write_json, guard};
use composed_retrieval::{
    Result, data,
    model::Model,
    operator::Tables,
    train::{self, Observer, Update},
};
use learned_retrieval::cohort::{publish, read};
use measurement::measure;
use serde_json::{Value, json};
use std::{collections::BTreeSet, fs, ops::Range, path::Path};

const PRODUCER: &str = "0f0aadbda8092a98ff740d53589be55d1f89e1cc";
const CASE: &str = "suite-lookup";
struct Trace<'a> {
    workspace: &'a Path,
    storage: &'a Path,
    mode: &'a str,
    eps: f64,
    steps: Vec<Value>,
}
fn rms(u: &Update<'_>, group: Range<usize>) -> Result<f64> {
    let mut n = 0;
    let mut ss = 0.;
    for (&i, &g) in u.coordinates.iter().zip(u.gradient) {
        if group.contains(&i) {
            n += 1;
            ss += g * g;
        }
    }
    if n == 0 {
        return Err("coordinate group was not visited".into());
    }
    Ok((ss / f64::from(n)).sqrt())
}
impl Observer for &mut Trace<'_> {
    fn admit(&mut self) -> Result<()> {
        guard::admit(1 << 30, self.workspace, self.storage)?;
        Ok(())
    }
    fn before_update(&mut self, u: Update<'_>) -> Result<()> {
        let l = &u.tables.layout;
        let view = measure(u.model, u.tables, u.batch, self.eps)?;
        let j = u
            .coordinates
            .iter()
            .position(|&i| i == l.depth)
            .ok_or("missing depth update")?;
        let executed = u.gradient[j];
        let expected = if self.mode == "mini" {
            view.slope
        } else {
            view.feasible_secant
        };
        if (executed - expected).abs() > 2e-9 || !executed.is_finite() || !expected.is_finite() {
            return Err(format!(
                "depth estimate disagrees at update {}: {executed} vs {expected}",
                u.number
            )
            .into());
        }
        self.steps.push(json!({"update":u.number,"executed_depth_estimate":executed,
            "visited_key_rms":rms(&u,l.key.clone())?,"visited_relative_rms":rms(&u,l.relation.clone())?,"batch":view}));
        Ok(())
    }
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let [workspace, storage, baseline, arm, output, mode, seed] = args.as_slice() else {
        return Err("expected WORKSPACE STORAGE SPARSE_OUTPUT ARM FRESH_CELL MODE SEED".into());
    };
    let seed: u32 = seed.parse()?;
    let (workspace, storage, baseline, output) = (
        Path::new(workspace),
        Path::new(storage),
        Path::new(baseline),
        Path::new(output),
    );
    guard::admit(1 << 30, workspace, storage)?;
    let protocol = data::protocol()?;
    if !["original", "repeat"].contains(&arm.as_str())
        || data::PROTOCOL != data::BASE_PROTOCOL
        || !protocol.modes.contains(mode)
        || !protocol.seeds.contains(&seed)
        || fs::read_to_string(baseline.join("producer.commit"))?.trim() != PRODUCER
    {
        return Err("expected frozen original sparse producer, mode and seed".into());
    }
    let root = baseline.join(arm).join("run");
    let seal = read(&root.join("sealed/manifest.json"))?;
    let identity = &seal["identity"];
    let key = format!("{CASE}/{mode}-{seed}");
    let cell = root.join(&key);
    let bytes = fs::read(cell.join("receipt.json"))?;
    let receipt: Value = serde_json::from_slice(&bytes)?;
    let old_model = fs::read(cell.join("model.json"))?;
    let (task, pins) = data::original(&protocol, CASE)?;
    if identity["producer"] != PRODUCER
        || identity["protocol_sha256"] != hash(data::PROTOCOL.as_bytes())
        || identity["binary_sha256"]
            != hash(&fs::read(baseline.join(arm).join("composed-retrieval"))?)
        || seal["cells"][&key] != hash(&bytes)
        || receipt["identity"] != *identity
        || receipt["data_fingerprints"] != pins
        || receipt["endpoint_sha256"] != hash(&old_model)
        || receipt["heldout_scored"] != false
        || receipt["case"] != CASE
        || receipt["mode"] != *mode
        || receipt["seed"] != seed
    {
        return Err("invalid sealed training evidence".into());
    }
    let spec = &protocol.cases[CASE];
    let source = &task.splits["train"];
    let mut trace = Trace {
        workspace,
        storage,
        mode,
        eps: protocol.query_step,
        steps: Vec::with_capacity(protocol.updates),
    };
    publish(output, |out| {
        let mut m = Model::new(task.vocab, spec.context, &protocol, seed)?;
        let training = train::fit(
            &mut m, source, spec.batch, mode, seed, &protocol, &mut trace,
        )?;
        let mut model_bytes = serde_json::to_vec(&m)?;
        model_bytes.push(b'\n');
        if receipt["training"] != training
            || model_bytes != old_model
            || trace.steps.len() != protocol.updates
        {
            return Err(
                "diagnostics changed earlier model, batches or optimizer trajectory".into(),
            );
        }
        let rows = train::samples(&m, source)?;
        let rows: Vec<_> = rows.iter().flatten().collect();
        let lengths: BTreeSet<_> = rows
            .iter()
            .map(|s| s.prefix.len().saturating_sub(1))
            .collect();
        let tables = Tables::new(&m, &lengths.into_iter().collect::<Vec<_>>())?;
        let final_train = measure(&m, &tables, &rows, protocol.query_step)?;
        write_json(&out.join("updates.json"), &trace.steps)?;
        write_json(
            &out.join("endpoint.json"),
            &json!({"source":PRODUCER,"case":CASE,"mode":mode,"seed":seed,
            "endpoint_sha256":receipt["endpoint_sha256"],"training":training,"final_train":final_train}),
        )
    })
}
