use crate::*;
mod gradient;
mod mutations;

fn fixture() -> Result<(kernel::discovery::Report, grouped::discovery::Report)> {
    let request = grouped::discovery::Request {
        input: kernel::discovery::Input {
            n: 3,
            vocabulary: 4,
            // Names and order are categorical. This order makes the raw
            // first-class fallback differ from the grouped missing corner.
            observations: vec![[1, 2], [0, 0], [3, 3]],
            policy: kernel::discovery::Policy::default(),
        },
        readout: vec![0, 0, 1, 1],
        answers: 2,
        prime: 2,
    };
    let raw = kernel::discovery::run(kernel::discovery::Request {
        input: request.input.clone(),
        prime: request.prime,
    })?;
    Ok((raw, grouped::discovery::run(request)?))
}
fn initialized() -> Result<(Model, kernel::discovery::Report, grouped::discovery::Report)> {
    let (raw, grouped) = fixture()?;
    let model = Model::new(
        kernel::initialize(&raw)?.model,
        grouped::construct::initialize(&grouped)?.model,
        &raw,
        &grouped,
        raw.request.input.observations.clone(),
    )?;
    Ok((model, raw, grouped))
}
fn best(values: &[f64]) -> usize {
    (1..values.len()).fold(0, |a, b| if values[b] > values[a] { b } else { a })
}
fn bits(values: Vec<f64>) -> Vec<u64> {
    values.into_iter().map(f64::to_bits).collect()
}

#[test]
fn actual_complex_forward_extends_only_new_coverage_and_preserves_other_bits() -> Result<()> {
    let (model, _, _) = initialized()?;
    assert_eq!(model.coverage.prior_supported, 3);
    assert_eq!(model.coverage.newly_covered, 1);
    assert_eq!(model.coverage.jointly_refused, 5);
    assert_eq!(model.coverage.unknown_prior_axes, 5);
    assert_eq!(model.coverage.unmatched_prior_cells, 1);
    assert_eq!(best(&model.prior.logits_cell(4)), 2);
    assert_eq!(best(&model.logits_cell(4)?), 0);
    // Raw aliases0 and1 have the same declared answer. This construction
    // cannot identify which unobserved raw label is the target.
    assert_ne!(best(&model.logits_cell(4)?), 1);
    assert_eq!(model.route(4)?.source, Source::Supplement);
    assert!(model.route(4)?.prior_axes_encoded);
    assert!(!model.route(4)?.exact_observed_input);
    assert_eq!(model.route(8)?.source, Source::Prior);
    assert!(!model.route(8)?.prior_axes_encoded);
    for cell in 0..9 {
        let route = model.route(cell)?;
        let selected = if route.source == Source::Prior {
            &model.prior
        } else {
            &model.supplement
        };
        assert_eq!(
            bits(model.logits_cell(cell)?),
            bits(selected.logits_cell(cell))
        );
        let independent = kernel::audit::complex_logits(selected, selected.location(cell));
        assert!(
            model
                .logits_cell(cell)?
                .iter()
                .zip(independent)
                .all(|(&a, b)| (a - b).abs() < 1e-12)
        );
    }
    assert!(
        model
            .supplement
            .coefficients
            .iter()
            .enumerate()
            .any(|(i, &x)| i % 8 >= 4 && x.abs() > 1.)
    );
    for c in 0..4 {
        let combined: f64 = [&model.prior, &model.supplement]
            .into_iter()
            .flat_map(|m| m.coefficients.iter().skip(c).step_by(4))
            .map(|x| x.abs())
            .sum();
        assert!(combined <= model.bounds.radius + 1e-8);
    }
    assert_eq!(
        model.bounds.real_coefficients,
        model.prior.coefficients.len() + model.supplement.coefficients.len()
    );
    assert_eq!(
        model.bounds.stored_float_scalars,
        [&model.prior, &model.supplement]
            .into_iter()
            .map(|m| m.coefficients.len()
                + m.table.len()
                + 2 * m.blocks.iter().map(|b| b.code.len()).sum::<usize>())
            .sum::<usize>()
    );
    Ok(())
}

#[test]
fn actual_full_mini_and_forward_endpoints_retain_raw_ce_and_histories() -> Result<()> {
    let (raw, grouped) = fixture()?;
    let policies = kernel::training::policies(3, 2)?;
    let a = kernel::binding::run(&raw, policies.clone())?;
    let b = grouped::binding::run(&grouped, policies.clone())?;
    kernel::binding::replay(&raw, &policies, &a)?;
    grouped::binding::replay(&grouped, &policies, &b)?;
    let saved = (digest(&a)?, digest(&b)?);
    for mode in 0..3 {
        let prior = &a.endpoints[mode].training.as_ref().unwrap().model;
        let supplement = &b.endpoints[mode].training.as_ref().unwrap().model;
        let model = Model::new(
            prior.clone(),
            supplement.clone(),
            &raw,
            &grouped,
            raw.request.input.observations.clone(),
        )?;
        for ids in [&[0, 1, 2][..], &[0, 1][..], &[2][..]] {
            let samples: Vec<_> = ids.iter().map(|&i| model.records[i]).collect();
            let old = kernel::training::gradient(prior, &prior.encode(&samples)?)?;
            let new = model.batch_gradient(ids)?;
            assert_eq!(old.loss.to_bits(), new.loss.to_bits());
            assert_eq!(bits(old.head), bits(new.prior));
        }
        assert_eq!(bits(model.logits_cell(4)?), bits(supplement.logits_cell(4)));
    }
    let full = a.endpoints[0].training.as_ref().unwrap();
    let mini = a.endpoints[1].training.as_ref().unwrap();
    let forward = a.endpoints[2].training.as_ref().unwrap();
    assert_eq!(full.work.gradient_batches, 1);
    assert_eq!(
        mini.steps.iter().map(|s| s.count).collect::<Vec<_>>(),
        vec![2, 1]
    );
    assert_eq!(forward.work.gradient_batches, 0);
    assert_eq!(saved, (digest(&a)?, digest(&b)?));
    Ok(())
}

#[test]
fn load_reconstructs_support_and_preserves_long_prefix_prediction_domain() -> Result<()> {
    let (model, raw, grouped) = initialized()?;
    let loaded = Model::load(
        serde_json::to_value(&model)?,
        &raw,
        &grouped,
        model.binding(),
    )?;
    assert_eq!(loaded.binding(), model.binding());
    for cell in 0..9 {
        assert_eq!(loaded.route(cell)?, model.route(cell)?);
        assert_eq!(
            bits(loaded.logits_cell(cell)?),
            bits(model.logits_cell(cell)?)
        );
    }
    for length in [2, 65, 4096] {
        let prefix = vec![1; length];
        assert_eq!(bits(model.logits(&prefix)?), bits(model.logits_cell(4)?));
    }
    for prefix in [
        vec![],
        vec![0],
        vec![1; 4097],
        vec![-1, 0],
        vec![0, 4],
        vec![0, 3],
    ] {
        assert!(model.logits(&prefix).is_err());
    }
    assert!(model.route(9).is_err());
    assert!(model.logits_cell(usize::MAX).is_err());
    Ok(())
}
