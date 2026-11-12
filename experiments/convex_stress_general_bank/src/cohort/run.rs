//! All candidate endpoints and their CE choice precede every held-out evaluation.
use super::*;
use std::{io::Write, time::Instant};

pub(super) struct Fit {
    pub model: Model,
    pub state: State,
    pub report: Value,
}

/// The replay executes every update again, with the same frozen input order.
/// `observe` receives training-only snapshots and can persist them immediately.
pub(super) fn fit(
    c: &Case,
    x: &Inputs,
    verified: &Verified,
    k: usize,
    mut observe: impl FnMut(&Value) -> Result<()>,
) -> Result<Fit> {
    let p = protocol()?;
    let s = schedule(c, x)?;
    let mode = mode(c)?;
    let mut solver = Solver::new(verified, k, mode)?;
    let rounds = count(&p, "repair_rounds")?;
    let mut order: Vec<_> = (0..s.count).collect();
    let mut rng = Random::new(count(&p, "seed")? as u32 + 10_000);
    let mut presented = if mode == Mode::Forward { s.count } else { 0 };
    let (mut updates, mut norms, mut snapshots) = (0, 0, 0);
    let (endpoint, stop, train_ce) = loop {
        if updates % s.cadence == 0 || updates == s.limit {
            let endpoint = solver.snapshot(rounds)?;
            let train = evaluate(&endpoint.model, &x.task.splits["train"])?;
            let ce = number(&train, "loss")?;
            let excess = ce - ideal_loss(x.shape.vocab);
            if excess < -1e-10 {
                return Err("CE below its proved lower bound".into());
            }
            let diagnostic = solver.state();
            norms += endpoint.norm_evaluations;
            snapshots += 1;
            observe(
                &json!({"update":updates,"train":train,"ideal_ce_excess":excess,
                "iterate_weight":endpoint.iterate_weight,"norm_evaluations":endpoint.norm_evaluations,
                "last_split_residual":diagnostic.residual,"costs":diagnostic.costs}),
            )?;
            if updates > 0 && excess <= number(&p, "ideal_ce_tolerance")? {
                break (endpoint, "ideal_ce_excess", ce);
            }
            if updates == s.limit {
                break (endpoint, "budget", ce);
            }
        }
        let selected = batch(x, &s, mode, updates, &mut rng, &mut order);
        solver.step(&selected)?;
        if mode != Mode::Forward {
            presented += selected.len();
        }
        updates += 1;
    };
    endpoint.model.certified(verified)?;
    let state = solver.state();
    let report = json!({"decoder_index":k,"status":"complete","updates":updates,"stop_reason":stop,
        "train_ce":train_ce,"ideal_ce_excess":train_ce-ideal_loss(x.shape.vocab),"costs":state.costs,
        "optimizer_examples_presented":presented,"epochs":presented as f64/s.count as f64,
        "setup_observation_checks":s.count,"setup_row_projections":1,"setup_norm_evaluations":1,
        "snapshot_count":snapshots,"snapshot_norm_evaluations":norms,
        "last_split_residual":state.residual,"iterate_weight":endpoint.iterate_weight,"moment_norms":endpoint.norms,
        "fit_defect":fit_defect(&endpoint.model,&x.problem)});
    Ok(Fit {
        model: endpoint.model,
        state,
        report,
    })
}

pub fn run(root: &Path, id: &str) -> Result<()> {
    let started = Instant::now();
    let c = case(id)?;
    execute(root, c.clone(), inventory(root)?, inputs(&c)?, started)
}
pub(super) fn execute(root: &Path, c: Case, inv: Value, x: Inputs, started: Instant) -> Result<()> {
    check_task(&inv, &c, &x)?;
    let out = root.join(c.id());
    fs::create_dir(&out)?;
    let mut r = metadata(&c, &inv, &x)?;
    write_new(&out.join("metadata.json"), &r)?;
    let search_started = Instant::now();
    let limits: search::Limits = serde_json::from_value(protocol()?["search"].clone())?;
    let found = search::discover(&x.problem, limits)?;
    r["discovery_seconds"] = search_started.elapsed().as_secs_f64().into();
    write_new(&out.join("discovery.json"), &found.record)?;
    r["discovery_sha256"] = hash(&bytes(&out.join("discovery.json"), x.explicit_bytes)?).into();
    r["search_stop"] = serde_json::to_value(found.record.stop)?;
    r["producer_decoder_checks"] = found.record.producer_checks.into();
    r["checker_decoder_checks"] = found.record.checker_checks.into();
    r["covered_cells"] = found
        .record
        .selected
        .as_ref()
        .map_or(0, |cert| {
            cert.answers.iter().filter(|a| a.len() == 1).count()
        })
        .into();
    let mut candidates = vec![];
    for k in 0..x.problem.decoders.len() {
        if let Some(reason) = candidate_refusal(found.verified.as_ref(), k) {
            candidates.push(json!({"decoder_index":k,"status":"refused","reason":reason}));
            continue;
        }
        let verified = found
            .verified
            .as_ref()
            .ok_or("missing verified certificate")?;
        let dir = out.join(format!("d{k}"));
        fs::create_dir(&dir)?;
        let mut trace = fs::File::create_new(dir.join("trace.jsonl"))?;
        let train_started = Instant::now();
        let fitted = fit(&c, &x, verified, k, |v| {
            writeln!(trace, "{v}")?;
            trace.flush()?;
            eprintln!(
                "decoder={k} update={} train_ce_excess={} iterate_weight={}",
                v["update"], v["ideal_ce_excess"], v["iterate_weight"]
            );
            Ok(())
        })?;
        let mut report = fitted.report;
        report["training_seconds"] = train_started.elapsed().as_secs_f64().into();
        write_new(&dir.join("model.json"), &fitted.model)?;
        write_new(&dir.join("state.json"), &fitted.state)?;
        for file in ["model", "state", "trace"] {
            let suffix = if file == "trace" { "jsonl" } else { "json" };
            report[format!("{file}_sha256")] = hash(&bytes(
                &dir.join(format!("{file}.{suffix}")),
                x.explicit_bytes,
            )?)
            .into();
        }
        candidates.push(report);
    }
    // Only training CE has been evaluated at this point, for every candidate.
    let selected = choose(&candidates, x.shape.vocab)?;
    r["selected_decoder"] = serde_json::to_value(selected)?;
    for (k, report) in candidates.iter_mut().enumerate() {
        if report["status"] == "complete" {
            let model: Model = serde_json::from_slice(&bytes(
                &out.join(format!("d{k}/model.json")),
                x.explicit_bytes,
            )?)?;
            report["metrics"] = metrics_all(&model, &x.task)?;
        }
    }
    r["status"] = if selected.is_some() {
        "complete"
    } else if found.verified.as_ref().is_some_and(Verified::inconsistent) {
        "bank_inconsistent"
    } else {
        "no_candidate"
    }
    .into();
    r["candidates"] = serde_json::to_value(candidates)?;
    r["total_seconds_including_data"] = started.elapsed().as_secs_f64().into();
    write_new(&out.join("result.json"), &r)
}
