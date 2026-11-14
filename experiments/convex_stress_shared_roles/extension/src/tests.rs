//! Synthetic CM40 witnesses; no original held-out examples are inspected.
use crate::*;
mod gradient;
mod mutations;

type Certificates = (kernel::discovery::Report, shared::discovery::Report);
fn fixture() -> Result<Certificates> {
    let request = shared::discovery::Request {
        input: kernel::discovery::Input {
            n: 4,
            vocabulary: 4,
            observations: vec![[5, 1], [0, 0], [10, 3]],
            policy: Default::default(),
        },
        readout: vec![0, 1, 2, 0],
        answers: 3,
        prime: 3,
    };
    let raw = kernel::discovery::run(kernel::discovery::Request {
        input: request.input.clone(),
        prime: request.prime,
    })?;
    Ok((raw, shared::discovery::run(request)?))
}
fn samples(input: &kernel::discovery::Input) -> Vec<records::Record> {
    input
        .observations
        .iter()
        .enumerate()
        .map(|(id, &[cell, label])| records::Record {
            id: 10 + 3 * id,
            cell,
            label,
        })
        .collect()
}
fn initialized() -> Result<(Model, Certificates)> {
    let certificates = fixture()?;
    let (raw, shared) = &certificates;
    let model = Model::new(
        kernel::initialize(raw)?.model,
        shared::construct::initialize(shared)?.model,
        raw,
        shared,
        samples(&raw.request.input),
    )?;
    Ok((model, certificates))
}
fn bits(values: Vec<f64>) -> Vec<u64> {
    values.into_iter().map(f64::to_bits).collect()
}
fn best(values: &[f64]) -> usize {
    (1..values.len()).fold(0, |a, b| if values[b] > values[a] { b } else { a })
}

#[test]
fn actual_augmented_kernel_extends_coverage_and_preserves_every_prior_bit() -> Result<()> {
    let (model, (raw, shared)) = initialized()?;
    assert_eq!(model.coverage.prior_supported, 3);
    assert_eq!(model.coverage.newly_covered, 2);
    assert_eq!(model.coverage.jointly_refused, 11);
    assert_eq!(model.coverage.unknown_prior_axes, 7);
    assert_eq!(model.coverage.unmatched_prior_cells, 6);
    assert!(raw.code.as_ref().unwrap().dimension > shared.raw_code.as_ref().unwrap().dimension);
    assert_eq!(model.route(2)?.source, Source::Supplement);
    assert_eq!(best(&model.prior.logits_cell(2)), 1);
    let new_label = best(&model.logits_cell(2)?);
    assert_eq!(shared.request.readout[new_label], 0);
    assert!(!model.route(2)?.exact_observed_input);
    assert!(!shared.raw_label_identification);
    assert!(!shared.shared_role_target_proved);
    for cell in 0..16 {
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
        if route.source == Source::Supplement {
            assert!(!route.prior_supported && route.supplement_supported);
        }
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
        let actual: f64 = [&model.prior, &model.supplement]
            .into_iter()
            .flat_map(|m| m.coefficients.iter().skip(c).step_by(4))
            .map(|x| x.abs())
            .sum();
        assert!(actual <= model.bounds.radius + 1e-8);
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
fn full_mini_forward_histories_are_bound_without_new_updates() -> Result<()> {
    let (raw, shared) = fixture()?;
    let records = samples(&raw.request.input);
    let policies = shared::training::policies(records.len(), 2)?;
    let prior_policies = kernel::training::policies(records.len(), 2)?;
    let old = kernel::binding::run(&raw, prior_policies.clone())?;
    let new = shared::binding::run(&shared, &records, policies.clone())?;
    kernel::binding::replay(&raw, &prior_policies, &old)?;
    shared::binding::replay(&shared, &records, &policies, &new)?;
    let hashes = (digest(&old)?, digest(&new)?);
    for (i, batches) in [1, 2, 0].into_iter().enumerate() {
        let a = old.endpoints[i].training.as_ref().unwrap();
        let b = new.endpoints[i].training.as_ref().unwrap();
        assert_eq!(a.work.gradient_batches, batches);
        assert_eq!(b.work.gradient_batches, batches);
        let model = Model::new(
            a.model.clone(),
            b.model.clone(),
            &raw,
            &shared,
            records.clone(),
        )?;
        let result = model.batch_gradient(&[0, 1, 2])?;
        assert_eq!(result.loss.to_bits(), a.final_ce.to_bits());
        assert!(result.supplement.iter().all(|&x| x == 0.));
        for record in &records {
            assert_eq!(
                bits(model.logits_cell(record.cell)?),
                bits(a.model.logits_cell(record.cell))
            );
        }
    }
    assert_eq!(
        old.endpoints[1]
            .training
            .as_ref()
            .unwrap()
            .steps
            .iter()
            .map(|s| (s.start, s.count))
            .collect::<Vec<_>>(),
        vec![(0, 2), (2, 1)]
    );
    assert_eq!(hashes, (digest(&old)?, digest(&new)?));
    Ok(())
}

#[test]
fn original_prefix_order_raw_aliases_and_complete_length_domain_are_preserved() -> Result<()> {
    let request = shared::discovery::Request {
        input: kernel::discovery::Input {
            n: 3,
            vocabulary: 4,
            observations: vec![[1, 0], [3, 1]],
            policy: Default::default(),
        },
        readout: vec![0, 0, 1, 2],
        answers: 3,
        prime: 3,
    };
    let raw = kernel::discovery::run(kernel::discovery::Request {
        input: request.input.clone(),
        prime: 3,
    })?;
    let shared = shared::discovery::run(request)?;
    let model = Model::new(
        kernel::initialize(&raw)?.model,
        shared::construct::initialize(&shared)?.model,
        &raw,
        &shared,
        samples(&raw.request.input),
    )?;
    assert_eq!(best(&model.logits(&[0, 1])?), 0);
    assert_eq!(best(&model.logits(&[1, 0])?), 1);
    for length in [2, 65, 4096] {
        let mut prefix = vec![0; length];
        prefix[length - 1] = 1;
        assert_eq!(
            bits(model.logits(&prefix)?),
            bits(model.prior.logits_cell(1))
        );
    }
    assert!(model.logits(&[0]).is_err());
    assert!(model.logits(&vec![0; 4097]).is_err());
    assert!(model.logits(&[4, 0, 1]).is_err());
    assert!(model.logits(&[-1, 0, 1]).is_err());
    assert!(model.logits(&[0, 3]).is_err());
    assert!(model.logits_cell(9).is_err());
    Ok(())
}
