use super::*;

pub fn run(input: Request) -> Result<Report> {
    let (partition, bound) = input.validate()?;
    let q = partition.classes.len();
    let mut out = Report {
        input,
        partition,
        buffer_preflight_bytes: bound,
        construction: None,
        attempts: vec![],
        candidates: vec![],
        selected: None,
        reason: "unsupported_latent_domain".into(),
        work: Work::default(),
    };
    if !(2..=257).contains(&q) {
        return Ok(out);
    }
    let construction = composite::run(out.input.problem(&out.partition))?;
    out.work.composite = Some(construction.work.clone());
    let n = out.input.n;
    for &branch in &construction.candidates {
        let point = &construction.branches[branch].point;
        let codes = &point[2 * n..];
        let conflict = out
            .input
            .observations
            .iter()
            .zip(&out.partition.class_indices)
            .enumerate()
            .find_map(|(index, (&[cell, _], &class))| {
                out.work.observation_checks += 1;
                let value = (point[cell % n] as u32 + point[n + cell / n] as u32) % q as u32;
                (value != codes[class] as u32).then_some(index)
            });
        if let Some(observation) = conflict {
            out.attempts.push(Attempt {
                branch,
                decision: Decision::ObservationConflict { observation },
            });
            continue;
        }
        let mut decoder = vec![0; q];
        for (class, &code) in codes.iter().enumerate() {
            decoder[code as usize] = out.partition.classes[class];
        }
        let model = Model::construct(
            out.input.vocabulary,
            point[..n].iter().map(|&x| x as usize).collect(),
            point[n..2 * n].iter().map(|&x| x as usize).collect(),
            decoder,
        )?;
        out.work.model_constructions += 1;
        out.work.decoder_coordinates += q;
        out.work.prediction_cells += n * n;
        let duplicate = out.candidates.iter().enumerate().find_map(|(i, c)| {
            out.work.duplicate_tests += 1;
            (model.output_labels == c.model.output_labels).then_some(i)
        });
        let decision = if let Some(candidate) = duplicate {
            Decision::Duplicate { candidate }
        } else if out.work.raw_audits == out.input.policy.models {
            Decision::ModelCap
        } else {
            out.work.raw_audits += 1;
            out.work.raw_frequency_extents.push(None);
            match model.audit(&out.input.samples()) {
                Err(e) => Decision::RawRefusal {
                    error: e.to_string(),
                },
                Ok(raw_audit) => {
                    if raw_audit["training_errors"] != 0 {
                        return Err(
                            "actual phase decoder violates the checked original class equations"
                                .into(),
                        );
                    }
                    out.work.completed_raw_frequencies += model.q;
                    *out.work
                        .raw_frequency_extents
                        .last_mut()
                        .expect("current raw audit") = Some(model.q);
                    let request = out.input.certificate(q, model.output_labels.clone());
                    let certificate = decoded::run(request.clone())?;
                    decoded::check(&request, &certificate)?;
                    out.work.certificates.push(certificate.work.clone());
                    let candidate = out.candidates.len();
                    out.candidates.push(Candidate {
                        branch,
                        model,
                        raw_audit,
                        certificate,
                    });
                    Decision::Admitted { candidate }
                }
            }
        };
        out.attempts.push(Attempt { branch, decision });
    }
    out.selected = plan::select(&out.candidates)?;
    out.reason = plan::reason(&out.candidates);
    out.construction = Some(construction);
    Ok(out)
}
