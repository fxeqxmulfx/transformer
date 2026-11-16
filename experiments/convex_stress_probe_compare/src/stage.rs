//! Training-only endpoints. Complete model states and every logical complex
//! coefficient are saved/audited before the global heldout seal exists.
use crate::{
    resources::Guard,
    schema::{Endpoint, Family, Stage},
    *,
};
use joint::{head::Head, training::Mode};
use model::input::Prepared;
use sha2::{Digest, Sha256};
use std::{fs, path::Path};
pub(crate) fn family(input: &Prepared<'_>) -> Family {
    Family {
        alphabet: input.alphabet().clone(),
        programs: input.programs().to_vec(),
        partition: input.partition().clone(),
        observations: input.observations().to_vec(),
    }
}
pub(crate) fn head_hash(head: &Head, work: &mut Work) -> Result<String> {
    let sh = head.shape();
    work.charge(32 * sh.cells() as u128)?;
    let mut hash = Sha256::new();
    for p in 0..sh.programs {
        for s in 0..sh.states {
            for c in 0..sh.classes {
                let (re, im) = head.coefficient(p, s, c)?;
                hash.update(re.to_bits().to_le_bytes());
                hash.update(im.to_bits().to_le_bytes());
            }
        }
    }
    Ok(io::hex(&hash.finalize()))
}
pub fn run(workspace: &Path, root: &Path, case: &str) -> Result<()> {
    let inputs = inputs::load(root)?;
    let entry = inputs::entry(&inputs, case)?;
    let guard = Guard::new(workspace, root)?;
    if root.join("seal.json").exists() {
        return Err("training after global seal is forbidden".into());
    }
    let data = inputs::dataset(root, entry)?;
    let report = build(root, case, data.as_ref(), entry.refusal.as_deref(), &guard)?;
    io::json(&root.join(case).join("stage.json"), &report, &guard)?;
    Ok(())
}
pub(crate) fn build(
    root: &Path,
    case: &str,
    data: Option<&probes::dataset::Dataset>,
    refusal: Option<&str>,
    guard: &Guard,
) -> Result<Stage> {
    fs::create_dir(root.join(case))?;
    let mut report = Stage {
        name: case.into(),
        refusal: refusal.map(str::to_owned),
        family: None,
        endpoints: vec![],
    };
    let Some(data) = data else {
        if refusal.is_none_or(str::is_empty) {
            return Err("missing source dataset and explicit refusal".into());
        }
        return Ok(report);
    };
    if refusal.is_some() {
        return Err("accepted dataset also has a source refusal".into());
    }
    let mut work = Work::new(WORK_LIMIT)?;
    let prepared = inputs::training(data, &mut work)?;
    report.family = Some(family(&prepared));
    let Some(compiled) = model::model::compile(&prepared, &mut work)? else {
        report.refusal = Some("empty complete training support".into());
        return Ok(report);
    };
    for mode in MODES {
        guard.check()?;
        let trained = compiled.train(mode, &mut work)?;
        let path = root.join(case).join(format!("{}.bin", name(mode)?));
        let length = joint::persistence::bytes(&compiled.domain);
        let mut writer = io::Writer::new(&path, length, guard)?;
        let hash = joint::persistence::write(
            &compiled.domain,
            &trained.parameters,
            &mut writer,
            &mut work,
        )?;
        let artifact = writer.finish()?;
        if artifact.bytes != length {
            return Err("complete endpoint checkpoint length differs".into());
        }
        report.endpoints.push(Endpoint {
            config: trained.config,
            checkpoint: artifact,
            payload_sha256: io::hex(&hash),
            complete_head_sha256: head_hash(&trained.head, &mut work)?,
            initial_ce: trained.initial_ce,
            final_ce: trained.final_ce,
            audit: trained.audit,
            steps: trained.steps,
            operations: trained.operations,
            logical_parameters: compiled.domain.parameters(),
            unknown_rows: compiled.domain.unknown.len(),
            target_realizability_certified: trained.target_realizability_certified,
            exact_real_projection_certified: trained.exact_real_projection_certified,
        });
        // Parameters and full head drop before the next policy is initialized.
    }
    Ok(report)
}
pub(crate) fn read(root: &Path, entry: &probes::inventory::Entry) -> Result<Stage> {
    let id = inputs::id(entry)?;
    let report: Stage = io::read(&root.join(id).join("stage.json"))?;
    validate(&report, id, entry.refusal.as_deref())?;
    for endpoint in &report.endpoints {
        let actual = io::artifact(
            &root
                .join(id)
                .join(format!("{}.bin", name(endpoint.config.mode)?)),
        )?;
        if actual != endpoint.checkpoint {
            return Err("complete checkpoint differs from training record".into());
        }
    }
    Ok(report)
}
pub(crate) fn validate(report: &Stage, id: &str, source_refusal: Option<&str>) -> Result<()> {
    if report.name != id {
        return Err("stage identity changed".into());
    }
    if let Some(reason) = source_refusal {
        if report.refusal.as_deref() != Some(reason)
            || report.family.is_some()
            || !report.endpoints.is_empty()
        {
            return Err("original source refusal was changed or scored".into());
        }
        return Ok(());
    }
    let family = report
        .family
        .as_ref()
        .ok_or("missing complete training family")?;
    family.alphabet.validate()?;
    if family.programs != model::features::catalog()
        || family.partition.decisions.len() != model::features::PROGRAMS
        || family.observations.is_empty()
        || family.observations.len() > 65536
    {
        return Err("complete family or original observation inventory differs".into());
    }
    let retained: Vec<_> = family
        .partition
        .decisions
        .iter()
        .enumerate()
        .filter_map(|(p, d)| (*d == model::partition::Decision::Keep).then_some(p))
        .collect();
    if retained != family.partition.retained {
        return Err("compatible family was truncated".into());
    }
    if retained.is_empty() {
        if report.refusal.as_deref() != Some("empty complete training support")
            || !report.endpoints.is_empty()
        {
            return Err("empty complete support status differs".into());
        }
        return Ok(());
    }
    if report.refusal.is_some() || report.endpoints.len() != MODES.len() {
        return Err("missing required mode endpoints".into());
    }
    for (endpoint, mode) in report.endpoints.iter().zip(MODES) {
        if endpoint.config != model::model::policy(mode)?
            || endpoint.target_realizability_certified
            || endpoint.exact_real_projection_certified
            || !endpoint.initial_ce.is_finite()
            || !endpoint.final_ce.is_finite()
            || endpoint.final_ce > endpoint.initial_ce + 1e-8
            || endpoint.audit.column_l1.len() != family.alphabet.vocabulary
        {
            return Err("endpoint policy or audit scope differs".into());
        }
        io::digest(&endpoint.payload_sha256)?;
        io::digest(&endpoint.complete_head_sha256)?;
        let n = family.observations.len();
        let expected = if mode == Mode::Forward {
            0
        } else {
            n.div_ceil(127)
        };
        if endpoint.steps.len() != expected {
            return Err("original minibatch count changed".into());
        }
        for (i, step) in endpoint.steps.iter().enumerate() {
            if step.epoch != 0
                || step.first_record != i * 127
                || step.end_record_exclusive != ((i + 1) * 127).min(n)
            {
                return Err("original minibatch identity or tail changed".into());
            }
        }
    }
    Ok(())
}
