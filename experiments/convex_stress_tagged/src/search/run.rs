use super::*;

fn domain_bank(view: &views::View) -> Vec<usize> {
    let mut classes = vec![];
    for &[_, c] in &view.observations {
        if !classes.contains(&c) {
            classes.push(c);
        }
    }
    let mut qs = vec![view.n];
    if classes.len() != view.n {
        qs.push(classes.len());
    }
    qs
}
fn seed_model(
    view: &views::View,
    sketch: &sketch::Sketch,
    q: usize,
    vocab: usize,
    work: &mut Work,
) -> Result<Model> {
    let rows = sketch::quantize(&sketch.left, &sketch.observed_rows, q)?;
    let columns = sketch::quantize(&sketch.conjugate_right, &sketch.observed_columns, q)?;
    let mut classes = vec![];
    let mut counts = vec![0; q * vocab];
    for &[cell, c] in &view.observations {
        if !classes.contains(&c) {
            classes.push(c);
        }
        counts[((rows[cell % view.n] + columns[cell / view.n]) % q) * vocab + c] += 1;
    }
    work.decoder_observations += view.observations.len();
    let mut decoder = vec![classes[0]; q];
    for (z, d) in decoder.iter_mut().enumerate() {
        for &c in &classes {
            work.decoder_comparisons += 1;
            if counts[z * vocab + c] > counts[z * vocab + *d] {
                *d = c;
            }
        }
    }
    Model::construct(vocab, rows, columns, decoder)
}
pub fn run(input: Input) -> Result<Report> {
    let buffer_bound_bytes = input.validate()?;
    let samples = input.samples();
    let views = views::propose(
        input.n,
        input.vocab,
        &samples,
        input.policy.minimum_strip_support,
    )?;
    let p = input.policy.clone();
    let mut out = Report {
        input,
        buffer_bound_bytes,
        views,
        sketches: vec![],
        attempts: vec![],
        work: Work {
            strip_observations: samples.len(),
            ..Work::default()
        },
        selected: None,
    };
    // Round-robin across views/domains prevents a first view taking every seed
    // before a categorical constant-strip proposal receives an initial attempt.
    for trial in 0..p.code_trials {
        for vi in 0..out.views.len() {
            let view = &out.views[vi];
            let skip = if vi >= p.max_views {
                Some("view_budget_exhausted")
            } else if view.reason == "empty_or_small_interior" {
                Some("empty_or_small_interior")
            } else if out.work.seeds >= p.seed_budget {
                Some("seed_budget_exhausted")
            } else {
                None
            };
            let mut slot = SketchSlot {
                view: vi,
                trial,
                reason: skip.unwrap_or("power_complete").into(),
                sketch: None,
            };
            if skip.is_none() {
                let sketch = sketch::build(
                    view.n,
                    out.input.vocab,
                    &view.samples(),
                    p.seed.wrapping_add((trial as u32).wrapping_mul(104729)),
                    p.power_steps,
                )?;
                out.work.sketches += 1;
                out.work.sketch_observations += 2 * view.observations.len();
                out.work.power_matvecs += sketch.matvecs;
                out.work.power_products += sketch.products;
                slot.reason = sketch.reason.clone();
                slot.sketch = Some(sketch);
            }
            for q in domain_bank(view) {
                let mut attempt = Attempt {
                    view: vi,
                    trial,
                    q,
                    reason: slot.reason.clone(),
                    refinement: None,
                    certificate_audit: None,
                    model: None,
                    audit: None,
                };
                if !(2..=257).contains(&q) {
                    attempt.reason = "latent_domain_out_of_bounds".into();
                } else if out.work.seeds >= p.seed_budget {
                    attempt.reason = "seed_budget_exhausted".into();
                } else if slot.reason == "power_complete" {
                    let seed = seed_model(
                        view,
                        slot.sketch.as_ref().unwrap(),
                        q,
                        out.input.vocab,
                        &mut out.work,
                    );
                    match seed {
                        Err(e) => attempt.reason = format!("phase_clock_refused: {e}"),
                        Ok(seed) => {
                            out.work.seeds += 1;
                            let refined = refinement::run(refinement::Input {
                                seed,
                                observations: view.observations.clone(),
                                rounds: p.refinement_rounds,
                                axis_budget: (p.axis_budget - out.work.axis_steps).min(256),
                            })?;
                            let checked = refinement::check(&refined.input, &refined)?;
                            out.work.axis_steps += refined.cost.axis_steps;
                            out.work.axis_observation_comparisons +=
                                refined.cost.axis_observation_comparisons;
                            out.work.coordinate_states += refined.steps.len();
                            out.work.decoder_observations += refined.cost.decoder_observations;
                            out.work.decoder_comparisons += refined.cost.decoder_comparisons;
                            attempt.reason = refined.reason.clone();
                            if let Some(best) = refined.best {
                                out.work.raw_base_frequencies += q;
                                let model =
                                    view.lift(&refined.steps[best].state.model(out.input.vocab)?)?;
                                let audit = model.audit(&samples)?;
                                out.work.raw_full_frequencies += model.q;
                                // Every stripped observed label was checked against
                                // the proposed constant. Verify the actual full table.
                                if audit["training_errors"].as_u64()
                                    != Some(refined.steps[best].state.errors as u64)
                                {
                                    return Err("actual full-table errors differ from the observed interior partition".into());
                                }
                                attempt.model = Some(model);
                                attempt.audit = Some(audit);
                            }
                            attempt.certificate_audit = Some(checked);
                            attempt.refinement = Some(refined);
                        }
                    }
                }
                if let Some(audit) = &attempt.audit {
                    let ce = audit["train_ce"]
                        .as_f64()
                        .ok_or("missing new candidate CE")?;
                    if out.selected.is_none_or(|i| {
                        ce < out.attempts[i].audit.as_ref().unwrap()["train_ce"]
                            .as_f64()
                            .unwrap()
                            - 1e-12
                    }) {
                        out.selected = Some(out.attempts.len());
                    }
                }
                out.attempts.push(attempt);
            }
            out.sketches.push(slot);
        }
    }
    Ok(out)
}
