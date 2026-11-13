//! Independent CM23 step checks from raw saved probability-table coordinates.
//! Integer assignment/decoder checks are exact; DFT/SVD and CE are numerical.
use crate::{
    Result, assignment_certificate as assignment,
    refinement::{Input, Kind, Outcome},
};
use serde_json::{Value, json};

pub fn check(input: &Input, out: &Outcome) -> Result<Value> {
    let samples = crate::refinement::validate(input)?;
    let (n, vocab) = (input.seed.n, input.seed.vocab);
    if &out.input != input || out.steps.is_empty() || out.steps.len() > 1 + 5 * input.rounds {
        return Err("refinement audit input or trace bound failed".into());
    }
    let (mut previous, mut best) = (None, None);
    let (mut comparisons, mut dual_checks, mut decoder_checks) = (0usize, 0usize, 0usize);
    for (i, step) in out.steps.iter().enumerate() {
        let s = &step.state;
        if !assignment::permutation(&s.rows, n)
            || !assignment::permutation(&s.columns, n)
            || s.decoder.len() != n
        {
            return Err("refinement state left the declared permutation family".into());
        }
        let model = s.model(vocab)?;
        let errors = samples
            .iter()
            .filter(|o| model.output_labels[o.cell] != o.label)
            .count();
        let mut counts = vec![0usize; n * vocab];
        for o in &samples {
            counts[model.latent_labels[o.cell] * vocab + o.label] += 1;
        }
        let covered = counts
            .chunks(vocab)
            .all(|row| row.iter().sum::<usize>() > 0);
        if errors != s.errors || covered != s.covered {
            return Err("raw refinement errors or coverage mismatch".into());
        }
        let expected = if i == 0 {
            Kind::Seed
        } else {
            [
                Kind::Decoder,
                Kind::Row,
                Kind::Decoder,
                Kind::Column,
                Kind::Decoder,
            ][(i - 1) % 5]
        };
        if step.kind != expected {
            return Err("refinement stage order mismatch".into());
        }
        if i == 0 {
            if model != input.seed || step.certificate.is_some() {
                return Err("refinement seed changed".into());
            }
        } else {
            let old = &out.steps[i - 1].state;
            if errors > old.errors {
                return Err("raw observed CE increased".into());
            }
            match step.kind {
                Kind::Seed => unreachable!(),
                Kind::Decoder => {
                    if s.rows != old.rows || s.columns != old.columns || step.certificate.is_some()
                    {
                        return Err("decoder update changed phases or certificate".into());
                    }
                    for l in 0..n {
                        for c in 0..vocab {
                            decoder_checks += 1;
                            if counts[l * vocab + c] > counts[l * vocab + s.decoder[l]] {
                                return Err("decoder count maximum failed".into());
                            }
                        }
                    }
                }
                Kind::Row | Kind::Column => {
                    let row = step.kind == Kind::Row;
                    if s.decoder != old.decoder
                        || (row && s.columns != old.columns)
                        || (!row && s.rows != old.rows)
                    {
                        return Err("assignment changed fixed coordinates".into());
                    }
                    let cert = step.certificate.as_ref().ok_or("missing assignment dual")?;
                    if cert.permutation
                        != if row {
                            s.rows.clone()
                        } else {
                            s.columns.clone()
                        }
                    {
                        return Err("assignment certificate does not describe the state".into());
                    }
                    let mut cost = vec![0; n * n];
                    for a in 0..n {
                        for z in 0..n {
                            for o in &samples {
                                comparisons += 1;
                                let (r, c) = (o.cell % n, o.cell / n);
                                if (if row { r } else { c }) == a {
                                    let l = if row {
                                        (z + old.columns[c]) % n
                                    } else {
                                        (old.rows[r] + z) % n
                                    };
                                    cost[a * n + z] += usize::from(old.decoder[l] != o.label);
                                }
                            }
                        }
                    }
                    assignment::check(&cost, n, cert)?;
                    dual_checks += n * n;
                    if cert.objective != errors {
                        return Err("assignment count differs from raw errors".into());
                    }
                }
            }
        }
        if covered && best.is_none_or(|b: usize| errors < out.steps[b].state.errors) {
            best = Some(i);
        }
        previous = Some(model);
    }
    if best != out.best {
        return Err("retained refinement endpoint mismatch".into());
    }
    let endpoint = if let Some(i) = best {
        let audit = out.steps[i].state.model(vocab)?.audit(&samples)?;
        let ideal = ((vocab - 1) as f64 + 1f64.exp()).ln() - 1.;
        let expected = ideal + out.steps[i].state.errors as f64 / samples.len() as f64;
        if (audit["train_ce"].as_f64().ok_or("missing CE")? - expected).abs() > 1e-10 {
            return Err("ordinary CE and exact error identity disagree".into());
        }
        audit
    } else {
        Value::Null
    };
    // Keep validation of the final raw construction explicit even without a
    // covered selected endpoint. No selected model is inferred from a cap.
    previous
        .ok_or("missing final refinement state")?
        .validate_shape()?;
    Ok(
        json!({"steps_checked":out.steps.len(),"state_observations":out.steps.len()*samples.len(),
        "assignment_scan_visits":comparisons,"dual_inequalities":dual_checks,
        "decoder_count_comparisons":decoder_checks,"endpoint":endpoint,
        "integer_assignment_certificates":true,"floating_spectra_are_exact_certificates":false}),
    )
}
