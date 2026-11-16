//! Original full-split scoring only after all required training endpoints seal.
//! Checkpoint reconstruction rechecks the complete training dictionary, complex
//! coefficient digest, inverse/budget audit and observed CE before prediction.
use crate::{
    resources::Guard,
    schema::{Endpoint, ModeScore, Replayed, Scored, Stage},
    *,
};
use joint::records::Source;
use probes::dataset::Dataset;
use std::{
    fs::{self, File},
    io::BufReader,
    path::Path,
};
fn reconstruct(
    compiled: &model::model::Compiled,
    endpoint: &Endpoint,
    path: &Path,
    work: &mut Work,
) -> Result<joint::head::Head> {
    let room = compiled
        .budget
        .model_room(&compiled.domain, &compiled.records)?;
    let (domain, q) = joint::persistence::read(
        BufReader::with_capacity(262144, File::open(path)?),
        io::digest(&endpoint.payload_sha256)?,
        endpoint.checkpoint.bytes,
        room,
        work,
    )?;
    if domain != compiled.domain
        || endpoint.logical_parameters != domain.parameters()
        || endpoint.unknown_rows != domain.unknown.len()
    {
        return Err("saved domain differs from ALL original training constraints".into());
    }
    drop(domain);
    let head_room = room
        .checked_sub(q.allocation_bytes())
        .ok_or("complete head construction exceeds room")?;
    let head = joint::head::Head::build(&compiled.domain, &q, head_room, work)?;
    if head.audit(&compiled.domain, &q, work)? != endpoint.audit
        || stage::head_hash(&head, work)? != endpoint.complete_head_sha256
    {
        return Err("complete reconstructed complex head differs".into());
    }
    let ce =
        joint::training::loss_minibatches(&head, Source::Packed(&compiled.records), 127, work)?;
    if ce.to_bits() != endpoint.final_ce.to_bits() {
        return Err("original observed endpoint CE differs".into());
    }
    Ok(head)
}
pub fn run(workspace: &Path, root: &Path, case: &str, replay: bool) -> Result<()> {
    let inputs = inputs::load(root)?;
    let entry = inputs::entry(&inputs, case)?;
    let barrier = seal::require(root, &inputs)?;
    let report = stage::read(root, entry)?;
    let data = inputs::dataset(root, entry)?;
    let guard = Guard::new(workspace, root)?;
    if replay {
        replay_case(root, &report, data.as_ref(), &barrier, &guard)?;
    } else {
        score_case(root, &report, data.as_ref(), barrier, &guard)?;
    }
    Ok(())
}
pub(crate) fn score_case(
    root: &Path,
    stage: &Stage,
    data: Option<&Dataset>,
    barrier: io::Artifact,
    guard: &Guard,
) -> Result<()> {
    let directory = root.join(&stage.name).join("scores");
    fs::create_dir(&directory)?;
    let mut report = Scored {
        name: stage.name.clone(),
        seal: barrier,
        refusal: stage.refusal.clone(),
        modes: vec![],
    };
    if stage.refusal.is_none() {
        let data = data.ok_or("accepted stage has no source dataset")?;
        let mut work = Work::new(WORK_LIMIT)?;
        let prepared = inputs::training(data, &mut work)?;
        if Some(stage::family(&prepared)) != stage.family {
            return Err("complete training family changed since seal".into());
        }
        let compiled = model::model::compile(&prepared, &mut work)?
            .ok_or("sealed complete support disappeared")?;
        for endpoint in &stage.endpoints {
            guard.check()?;
            let mode = endpoint.config.mode;
            let head = reconstruct(
                &compiled,
                endpoint,
                &root.join(&stage.name).join(format!("{}.bin", name(mode)?)),
                &mut work,
            )?;
            let mut scored = ModeScore {
                mode,
                splits: vec![],
            };
            for (index, split) in data.splits.keys().enumerate() {
                guard.check()?;
                let predictor =
                    model::model::Forward::new(&prepared, &compiled.domain, &head, &mut work)?;
                scored.splits.push(trace::record(
                    data,
                    split,
                    predictor,
                    &directory.join(format!("{}-{index}.raw", name(mode)?)),
                    guard,
                )?);
            }
            report.modes.push(scored);
        }
    }
    validate_scores(&report, stage, data)?;
    io::json(&directory.join("scores.json"), &report, guard)?;
    Ok(())
}
pub(crate) fn validate_scores(
    report: &Scored,
    stage: &Stage,
    data: Option<&Dataset>,
) -> Result<()> {
    if report.name != stage.name || report.refusal != stage.refusal {
        return Err("original outcome changed".into());
    }
    if stage.refusal.is_some() {
        if !report.modes.is_empty() {
            return Err("refused source or support was scored".into());
        }
        return Ok(());
    }
    let data = data.ok_or("missing scored data")?;
    if report.modes.len() != MODES.len() {
        return Err("missing required scored mode".into());
    }
    for (mode, expected) in report.modes.iter().zip(MODES) {
        if mode.mode != expected || mode.splits.len() != data.splits.len() {
            return Err("mode or original split omitted".into());
        }
        for (score, (name, split)) in mode.splits.iter().zip(&data.splits) {
            if score.split != *name
                || score.calls == 0
                || score.calls > trace::maximum_calls(split, name)?
            {
                return Err("scored original split or complete call inventory differs".into());
            }
        }
    }
    Ok(())
}
pub(crate) fn replay_case(
    root: &Path,
    stage: &Stage,
    data: Option<&Dataset>,
    barrier: &io::Artifact,
    guard: &Guard,
) -> Result<()> {
    let directory = root.join(&stage.name).join("scores");
    let path = directory.join("scores.json");
    let report: Scored = io::read(&path)?;
    validate_scores(&report, stage, data)?;
    if report.seal != *barrier {
        return Err("scores refer to a different training seal".into());
    }
    let count = recompute(&directory, stage, &report, data, guard)?;
    io::json(
        &root.join(&stage.name).join("replay.json"),
        &Replayed {
            name: stage.name.clone(),
            score: io::artifact(&path)?,
            raw_calls: count,
        },
        guard,
    )?;
    Ok(())
}
/// The final verifier performs this computation again; a replay receipt alone
/// cannot establish correct counts, CE, free generation or unquantized MSE.
pub(crate) fn recompute(
    directory: &Path,
    stage: &Stage,
    report: &Scored,
    data: Option<&Dataset>,
    guard: &Guard,
) -> Result<u64> {
    validate_scores(report, stage, data)?;
    let mut count = 0u64;
    if stage.refusal.is_none() {
        let data = data.ok_or("missing replay data")?;
        for mode in &report.modes {
            for (i, split) in mode.splits.iter().enumerate() {
                guard.check()?;
                let mut raw = trace::Replay::new(
                    &directory.join(format!("{}-{i}.raw", name(mode.mode)?)),
                    data.vocabulary,
                    split.calls,
                    &split.trace,
                )?;
                let actual = probe(probes::scoring::score(data, &split.split, &mut raw))?;
                raw.finish()?;
                if actual != split.score {
                    return Err("raw ordinary CE/generation/exact/MSE replay differs".into());
                }
                count = count
                    .checked_add(split.calls)
                    .ok_or("raw replay count overflow")?;
            }
        }
    }
    Ok(count)
}
