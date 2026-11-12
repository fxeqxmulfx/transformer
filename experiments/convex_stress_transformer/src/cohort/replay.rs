//! Every observation's predictions and the final native AdamW update are replayed.
use super::*;
use std::{collections::BTreeMap, io::BufRead};
pub(super) fn lines(path: &Path, mut visit: impl FnMut(Value) -> Result<()>) -> Result<()> {
    if fs::metadata(path)?.len() > 64 * 1024 * 1024 {
        return Err("oversize transformer log".into());
    }
    for line in std::io::BufReader::new(fs::File::open(path)?).lines() {
        let line = line?;
        if line.is_empty() || line.len() > 1024 * 1024 {
            return Err("invalid bounded transformer log record".into());
        }
        visit(serde_json::from_str(&line)?)?;
    }
    Ok(())
}
fn loaded(
    out: &Path,
    snapshot: &Value,
    x: &Inputs,
    binding: &str,
    filename: &str,
) -> Result<checkpoint::Restored> {
    if snapshot["file"] != filename || snapshot["file_sha256"] != file_hash(&out.join(filename))? {
        return Err("checkpoint filename or file digest differs".into());
    }
    let state = checkpoint::load(&out.join(filename), binding)?;
    if snapshot["body_sha256"] != state.body_sha256
        || snapshot["progress"] != serde_json::to_value(&state.progress)?
        || state.model.dims != x.dims
        || state.model.hyper != x.hyper
        || state.model.microbatch != x.microbatch
    {
        return Err("checkpoint configuration, progress or digest differs".into());
    }
    Ok(state)
}
fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}
fn check_progress(
    p: &checkpoint::Progress,
    step: u64,
    totals: &BTreeMap<u64, (u64, u64, Value)>,
) -> Result<()> {
    let (examples, targets, _) = totals
        .get(&step)
        .ok_or("missing observed progress totals")?;
    if p.completed_steps != step
        || p.examples_presented != *examples
        || p.targets_presented != *targets
    {
        return Err("checkpoint counters differ from complete update log".into());
    }
    Ok(())
}
pub fn replay(root: &Path, id: &str) -> Result<Value> {
    let c = case(id)?;
    let inv = inventory(root)?;
    let x = inputs(&c)?;
    replay_with(root, c, inv, x)
}
pub(super) fn replay_with(root: &Path, c: Case, inv: Value, x: Inputs) -> Result<Value> {
    let spec = check_spec(&inv, &c, &x)?;
    let binding = hash(&serde_json::to_vec(&spec)?);
    let expected = json!({"specification":spec,"binding":binding,"producer_commit":inv["producer_commit"],"parent":inv["parent"]});
    let out = root.join(c.id());
    let r = read(&out.join("result.json"))?;
    if read(&out.join("metadata.json"))? != expected
        || r["metadata"] != expected
        || r["status"] != "complete"
        || r["case"] != serde_json::to_value(&c)?
        || r["id"] != c.id()
        || r["primary_split"] != x.task.selection
        || r["historical_seed1"] != x.row["historical_seed1"]
        || r["updates_sha256"] != file_hash(&out.join("updates.jsonl"))?
        || r["observations_sha256"] != file_hash(&out.join("observations.jsonl"))?
    {
        return Err("transformer provenance, specification or logs changed".into());
    }
    let completed = r["progress"]["completed_steps"]
        .as_u64()
        .ok_or("missing completed updates")?;
    if completed == 0 || completed > x.cap || !observe(completed, x.cap)? {
        return Err("invalid transformer stopping step".into());
    }
    let mut totals = BTreeMap::from([(0, (0, 0, Value::Null))]);
    let (mut step, mut examples, mut targets) = (0u64, 0u64, 0u64);
    lines(&out.join("updates.jsonl"), |u| {
        step += 1;
        let n = integer(&u, "targets")?;
        if u["step"] != step
            || u["examples"] != x.batch
            || n < x.batch
            || n > x.batch * x.dims.ctx
            || number(&u, "batch_ce")? < -1e-8
            || u["learning_rate"] != x.hyper.lr(step)
        {
            return Err("invalid transformer update trace".into());
        }
        examples += x.batch as u64;
        targets += n as u64;
        if observe(step, x.cap)? || observe(step + 1, x.cap)? {
            totals.insert(step, (examples, targets, u));
        }
        Ok(())
    })?;
    if step != completed {
        return Err("incomplete transformer update trace".into());
    }
    let mut count = 0;
    let mut previous = 0;
    let mut previous_seconds = 0.;
    let mut endpoint = Value::Null;
    lines(&out.join("observations.jsonl"), |o| {
        let at = o["step"].as_u64().ok_or("missing observation step")?;
        let wanted = if count == 0 {
            0
        } else {
            (previous + protocol()?["observation_every_updates"].as_u64().unwrap()).min(x.cap)
        };
        if at != wanted || at > completed {
            return Err("missing or unscheduled transformer observation".into());
        }
        let mut state = loaded(
            &out,
            &o["checkpoint"],
            &x,
            &binding,
            &format!("at-{at:06}.bin"),
        )?;
        check_progress(&state.progress, at, &totals)?;
        if state.progress.optimizer_seconds < previous_seconds {
            return Err("optimizer time moved backwards".into());
        }
        previous_seconds = state.progress.optimizer_seconds;
        if at == 0 {
            if !same_bits(&state.model.p, &engine::initialize(x.dims, c.seed)?)
                || state.model.m.iter().chain(&state.model.v).any(|&v| v != 0.)
                || state.rng.words() != rng(c.seed).words()
                || !o["before"].is_null()
                || !o["last_update"].is_null()
            {
                return Err("initial transformer state differs".into());
            }
        } else {
            let before = loaded(
                &out,
                &o["before"],
                &x,
                &binding,
                &format!("before-{at:06}.bin"),
            )?;
            check_progress(&before.progress, at - 1, &totals)?;
            if before.progress.optimizer_seconds > state.progress.optimizer_seconds
                || o["last_update"] != totals[&at].2
            {
                return Err("previous-step checkpoint or final update log differs".into());
            }
        }
        let primary = evaluate(&mut state.model, &x.task.splits[&x.task.selection])?;
        if primary != o["primary"] {
            return Err("original primary metrics did not replay".into());
        }
        if at > 0 && number(&primary, "sequence_accuracy")? >= 0.99 && at != completed {
            return Err("training continued past the first successful observation".into());
        }
        previous = at;
        count += 1;
        endpoint = o;
        Ok(())
    })?;
    if previous != completed || r["observations"] != count || r["endpoint"] != endpoint {
        return Err("selected endpoint does not match the complete observation sequence".into());
    }
    let hit = number(&endpoint["primary"], "sequence_accuracy")? >= 0.99;
    let reason = if hit {
        "primary_target"
    } else {
        "original_update_cap"
    };
    if r["stop_reason"] != reason || (!hit && completed != x.cap) {
        return Err("original transformer stopping rule differs".into());
    }
    let mut actual = loaded(
        &out,
        &endpoint["checkpoint"],
        &x,
        &binding,
        &format!("at-{completed:06}.bin"),
    )?;
    let mut before = loaded(
        &out,
        &endpoint["before"],
        &x,
        &binding,
        &format!("before-{completed:06}.bin"),
    )?;
    let samples = batch(&x, &mut before.rng);
    let (loss, target_count) = before.model.step(&samples, completed)?;
    if endpoint["last_update"]["batch_ce"] != loss
        || endpoint["last_update"]["targets"] != target_count
        || before.rng.words() != actual.rng.words()
        || !same_bits(&before.model.p, &actual.model.p)
        || !same_bits(&before.model.m, &actual.model.m)
        || !same_bits(&before.model.v, &actual.model.v)
    {
        return Err("final AdamW update or sampling RNG does not reproduce bit for bit".into());
    }
    drop(before);
    let metrics = metrics_all(&mut actual.model, &x.task)?;
    if metrics != r["metrics"] || r["progress"] != serde_json::to_value(&actual.progress)? {
        return Err("endpoint original metrics or work counts differ".into());
    }
    Ok(
        json!({"replayed":true,"id":c.id(),"source_fingerprint":fingerprint(),"result_sha256":file_hash(&out.join("result.json"))?,
        "updates_sha256":r["updates_sha256"],"observations_sha256":r["observations_sha256"],"endpoint_body_sha256":actual.body_sha256,
        "observations_replayed":count,"final_update_bit_exact":true,"original_metrics_replayed":true,"complete_training_rerun":false}),
    )
}
