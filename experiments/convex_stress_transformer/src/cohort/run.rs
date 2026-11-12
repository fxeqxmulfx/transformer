//! Original-cap minibatch training; validation stopping never reads test labels.
use super::*;
use std::time::Instant;
fn line(writer: &mut impl Write, value: &Value) -> Result<()> {
    serde_json::to_writer(&mut *writer, value)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}
fn snapshot(
    out: &Path,
    name: &str,
    net: &engine::Transformer,
    rng: &Mt,
    p: &checkpoint::Progress,
    binding: &str,
) -> Result<Value> {
    disk_reserve(out)?;
    let digest = checkpoint::save_new(&out.join(name), net, rng, p.clone(), binding)?;
    Ok(
        json!({"file":name,"body_sha256":digest,"file_sha256":file_hash(&out.join(name))?,"progress":p}),
    )
}
pub fn run(root: &Path, id: &str) -> Result<()> {
    let started = Instant::now();
    let c = case(id)?;
    let inv = inventory(root)?;
    let x = inputs(&c)?;
    execute(root, c, inv, x, started)
}
pub(super) fn execute(root: &Path, c: Case, inv: Value, x: Inputs, started: Instant) -> Result<()> {
    let spec = check_spec(&inv, &c, &x)?;
    let binding = hash(&serde_json::to_vec(&spec)?);
    let out = root.join(c.id());
    fs::create_dir(&out)?;
    let metadata = json!({"specification":spec,"binding":binding,"producer_commit":inv["producer_commit"],"parent":inv["parent"]});
    write_new(&out.join("metadata.json"), &metadata)?;
    let mut net = engine::Transformer::new(x.dims, c.seed, x.microbatch, x.hyper)?;
    let mut rng = rng(c.seed);
    let mut progress = checkpoint::Progress {
        completed_steps: 0,
        examples_presented: 0,
        targets_presented: 0,
        optimizer_seconds: 0.,
    };
    let mut history = std::io::BufWriter::new(fs::File::create_new(out.join("updates.jsonl"))?);
    let mut observations =
        std::io::BufWriter::new(fs::File::create_new(out.join("observations.jsonl"))?);
    let initial = snapshot(&out, "at-000000.bin", &net, &rng, &progress, &binding)?;
    let initial_primary = evaluate(&mut net, &x.task.splits[&x.task.selection])?;
    line(
        &mut observations,
        &json!({"step":0,"checkpoint":initial,"before":null,"primary":initial_primary,"last_update":null}),
    )?;
    let mut observation_count = 1;
    let mut selected = Value::Null;
    let mut reason = "original_update_cap";
    for step in 1..=x.cap {
        let observe_now = observe(step, x.cap)?;
        let before = if observe_now {
            snapshot(
                &out,
                &format!("before-{step:06}.bin"),
                &net,
                &rng,
                &progress,
                &binding,
            )?
        } else {
            Value::Null
        };
        let step_started = Instant::now();
        let examples = batch(&x, &mut rng);
        let (loss, targets) = net.step(&examples, step)?;
        progress.completed_steps = step;
        progress.examples_presented += examples.len() as u64;
        progress.targets_presented += targets as u64;
        progress.optimizer_seconds += step_started.elapsed().as_secs_f64();
        let update = json!({"step":step,"batch_ce":loss,"examples":examples.len(),"targets":targets,"learning_rate":x.hyper.lr(step)});
        line(&mut history, &update)?;
        if observe_now {
            let saved = snapshot(
                &out,
                &format!("at-{step:06}.bin"),
                &net,
                &rng,
                &progress,
                &binding,
            )?;
            let primary = evaluate(&mut net, &x.task.splits[&x.task.selection])?;
            let record = json!({"step":step,"checkpoint":saved,"before":before,"primary":primary,"last_update":update});
            line(&mut observations, &record)?;
            observation_count += 1;
            let hit =
                number(&primary, "sequence_accuracy")? >= number(&protocol()?, "primary_target")?;
            selected = record;
            if hit {
                reason = "primary_target";
                break;
            }
        }
    }
    history.flush()?;
    history.get_ref().sync_all()?;
    observations.flush()?;
    observations.get_ref().sync_all()?;
    let metrics_started = Instant::now();
    let all = metrics_all(&mut net, &x.task)?;
    if all[&x.task.selection] != selected["primary"] {
        return Err("endpoint primary differs from stopping observation".into());
    }
    write_new(
        &out.join("result.json"),
        &json!({"status":"complete","case":c,"id":c.id(),"metadata":metadata,
        "stop_reason":reason,"progress":progress,"observations":observation_count,"endpoint":selected,
        "metrics":all,"primary_split":x.task.selection,"metrics_seconds":metrics_started.elapsed().as_secs_f64(),
        "updates_sha256":file_hash(&out.join("updates.jsonl"))?,"observations_sha256":file_hash(&out.join("observations.jsonl"))?,
        "total_seconds_including_data":started.elapsed().as_secs_f64(),"historical_seed1":x.row["historical_seed1"]}),
    )
}
