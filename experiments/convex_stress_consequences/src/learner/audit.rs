use super::*;

pub fn check(input: &Request, out: &Report) -> Result<()> {
    let (partition, bound) = input.validate()?;
    if out.input != *input || out.partition != partition || out.buffer_preflight_bytes != bound {
        return Err("learner original categorical partition, input or forecast differs".into());
    }
    let q = partition.classes.len();
    if !(2..=257).contains(&q) {
        if out.construction.is_some()
            || !out.attempts.is_empty()
            || !out.candidates.is_empty()
            || out.selected.is_some()
            || out.reason != "unsupported_latent_domain"
            || out.work != Work::default()
        {
            return Err("unsupported domain emitted a model or certificate".into());
        }
        return Ok(());
    }
    let construction = out
        .construction
        .as_ref()
        .ok_or("missing original code construction")?;
    composite::check(&input.problem(&partition), construction)?;
    if out.attempts.len() != construction.candidates.len()
        || out.candidates.len() > input.policy.models
    {
        return Err("learner omitted a completed code attempt or exceeded its model cap".into());
    }
    let mut work = Work {
        composite: Some(construction.work.clone()),
        ..Work::default()
    };
    let mut admitted = 0;
    for (attempt, &branch) in out.attempts.iter().zip(&construction.candidates) {
        if attempt.branch != branch {
            return Err("learner branch order differs".into());
        }
        let point = &construction.branches[branch].point;
        let n = input.n;
        let codes = &point[2 * n..];
        let mut conflict = None;
        for (index, (&[cell, _], &class)) in input
            .observations
            .iter()
            .zip(&partition.class_indices)
            .enumerate()
        {
            work.observation_checks += 1;
            let equation =
                point[cell % n] as i128 + point[n + cell / n] as i128 - codes[class] as i128;
            if equation.rem_euclid(q as i128) != 0 {
                conflict = Some(index);
                break;
            }
        }
        if let Some(observation) = conflict {
            if attempt.decision != (Decision::ObservationConflict { observation }) {
                return Err("learner original-observation conflict differs".into());
            }
            continue;
        }
        // Derive all fields from actual codes independently of Model::construct.
        let decoder: Vec<_> = (0..q)
            .map(|z| {
                codes
                    .iter()
                    .position(|&c| c as usize == z)
                    .map(|class| partition.classes[class])
                    .ok_or("checked code lacks an actual decoder state")
            })
            .collect::<std::result::Result<_, _>>()?;
        let latent_labels: Vec<_> = (0..n * n)
            .map(|cell| (point[cell % n] as usize + point[n + cell / n] as usize) % q)
            .collect();
        let model = Model {
            n,
            q,
            vocab: input.vocabulary,
            rows: point[..n].iter().map(|&x| x as usize).collect(),
            columns: point[n..2 * n].iter().map(|&x| x as usize).collect(),
            output_labels: latent_labels.iter().map(|&z| decoder[z]).collect(),
            decoder,
            latent_labels,
        };
        work.model_constructions += 1;
        work.decoder_coordinates += q;
        work.prediction_cells += n * n;
        let mut duplicate = None;
        for index in 0..admitted {
            work.duplicate_tests += 1;
            if out.candidates[index].model.output_labels == model.output_labels {
                duplicate = Some(index);
                break;
            }
        }
        let decision = if let Some(candidate) = duplicate {
            Decision::Duplicate { candidate }
        } else if work.raw_audits == input.policy.models {
            Decision::ModelCap
        } else {
            work.raw_audits += 1;
            work.raw_frequency_extents.push(None);
            match model.audit(&input.samples()) {
                Err(e) => Decision::RawRefusal {
                    error: e.to_string(),
                },
                Ok(raw_audit) => {
                    let c = out
                        .candidates
                        .get(admitted)
                        .ok_or("missing admitted actual model")?;
                    if c.branch != branch
                        || c.model != model
                        || c.raw_audit != raw_audit
                        || raw_audit["training_errors"] != 0
                    {
                        return Err("learner actual latent/output table, raw spectra or ordinary CE differs".into());
                    }
                    let request = input.certificate(q, model.output_labels.clone());
                    decoded::check(&request, &c.certificate)?;
                    work.certificates.push(c.certificate.work.clone());
                    work.completed_raw_frequencies += q;
                    *work
                        .raw_frequency_extents
                        .last_mut()
                        .expect("current checked audit") = Some(q);
                    let candidate = admitted;
                    admitted += 1;
                    Decision::Admitted { candidate }
                }
            }
        };
        if attempt.decision != decision {
            return Err("learner duplicate, cap, raw audit or admission differs".into());
        }
    }
    if admitted != out.candidates.len()
        || out.work != work
        || out.selected != plan::select(&out.candidates)?
        || out.reason != plan::reason(&out.candidates)
    {
        return Err(
            "learner actual work, candidate extent or observed-CE selection differs".into(),
        );
    }
    Ok(())
}
