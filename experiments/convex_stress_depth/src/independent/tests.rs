use super::*;
use crate::{
    data,
    features::{Mask, Transport},
};

fn fixture() -> Result<(Independent, Cache)> {
    let bank = Bank {
        vocab: 3,
        depth: 2,
        history_slots: 8,
        masks: vec![Mask::All, Mask::Pair(0, 2)],
        transport: Transport::Fourier,
        execution: Execution::Composed,
    };
    let mut model = Independent::new(bank, 30.)?;
    model.probabilities = vec![0.3, 0.6];
    for (i, d) in model.decoder.iter_mut().enumerate() {
        *d = 0.01 * (i as f64 - 30.);
    }
    let e = data::Example {
        task: "arbitrary".into(),
        tokens: vec![0, 1, 2, 0, 2],
        targets: vec![1, 2, 0, 1, 0],
        changes: vec![false; 5],
        prompt: vec![],
        answer: vec![],
        limit: 0,
        field: None,
        row_index: None,
    };
    let cache = Cache::build(&model.bank, &[e])?;
    Ok((model, cache))
}
#[test]
fn factor_gradients_match_finite_differences_including_the_shared_decoder() -> Result<()> {
    let (mut model, cache) = fixture()?;
    let mut decoder = vec![0.; model.decoder.len()];
    let mut router = vec![0.; model.bank.depth];
    objective(&model, &cache, &[0], Some(&mut decoder), Some(&mut router))?;
    for (i, &g) in decoder.iter().enumerate() {
        let old = model.decoder[i];
        model.decoder[i] = old + 1e-5;
        let plus = objective(&model, &cache, &[0], None, None)?.0;
        model.decoder[i] = old - 1e-5;
        let minus = objective(&model, &cache, &[0], None, None)?.0;
        model.decoder[i] = old;
        assert!((g - (plus - minus) / 2e-5).abs() < 1e-9);
    }
    for (i, &g) in router.iter().enumerate() {
        let old = model.probabilities[i];
        model.probabilities[i] = old + 1e-5;
        let plus = objective(&model, &cache, &[0], None, None)?.0;
        model.probabilities[i] = old - 1e-5;
        let minus = objective(&model, &cache, &[0], None, None)?.0;
        model.probabilities[i] = old;
        assert!((g - (plus - minus) / 2e-5).abs() < 1e-9);
    }
    Ok(())
}
#[test]
fn product_router_embeds_with_equal_logits_and_the_same_column_budget() -> Result<()> {
    let (model, cache) = fixture()?;
    let head = model.lift()?;
    assert!((model.routes().iter().sum::<f64>() - 1.).abs() < 1e-14);
    assert!((model.validate()? - head.validate()?).abs() < 1e-12);
    assert!(
        (objective(&model, &cache, &[0], None, None)?.0
            - head::objective(&head, &cache, &[0], None)?.0)
            .abs()
            < 1e-14
    );
    for prefix in [&[0, 1, 2, 0][..], &[2, 2, 0, 1, 2]] {
        for (x, y) in model.logits(prefix).iter().zip(head.logits(prefix)) {
            assert!((x - y).abs() < 1e-14);
        }
    }
    Ok(())
}
#[test]
fn forward_vertex_factorization_is_exact_and_routing_derivatives_handle_zero_weights() -> Result<()>
{
    let (model, cache) = fixture()?;
    let mut head = Head::zero(model.bank, model.radius)?;
    let mut counts = vec![0.; head.coefficients.len()];
    let channel = head::forward_initialize(&mut head, &cache, &mut counts)?;
    let factor = Independent::from_vertex(&head, channel)?;
    assert_eq!(factor.lift()?.coefficients, head.coefficients);
    let derivatives = factor.route_derivatives();
    for layer in 0..factor.bank.depth {
        assert_eq!(derivatives.iter().map(|d| d[layer]).sum::<f64>(), 0.);
        assert!(derivatives.iter().any(|d| d[layer] != 0.));
    }
    head.coefficients[0] += 0.01;
    // Changing a different path must invalidate the claimed vertex factorization.
    let other = (channel / 2 % head.bank.paths_per_view() + 1) % head.bank.paths_per_view();
    head.coefficients[other * head.bank.stride() * head.bank.vocab] = 0.01;
    assert!(Independent::from_vertex(&head, channel).is_err());
    Ok(())
}
#[test]
fn block_coordinate_step_decreases_batch_ce_and_preserves_the_domain() -> Result<()> {
    let (mut model, cache) = fixture()?;
    let initial = objective(&model, &cache, &[0], None, None)?.0;
    let mut gradient = vec![0.; model.decoder.len()];
    let (prefixes, evaluations) = step(&mut model, &cache, &[0], &mut gradient)?;
    assert_eq!(prefixes, 2 * cache.labels.len());
    assert!((1..=8).contains(&evaluations));
    assert!(objective(&model, &cache, &[0], None, None)?.0 < initial);
    model.validate()?;
    model.probabilities[0] = -0.01;
    assert!(model.validate().is_err());
    Ok(())
}
