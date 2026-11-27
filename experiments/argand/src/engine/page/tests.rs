use super::*;
use crate::suite::Shape;

fn close(actual: f32, expected: f32) -> bool {
    (actual - expected).abs() <= 2e-5 * (1.0 + expected.abs())
}

#[test]
fn corrections_keep_the_context_and_weight_only_the_sampled_positions() {
    let train = Observations {
        tokens: &[0, 1, 2, 3, 2, 1],
        labels: &[u32::MAX, 0, 3, 1, u32::MAX, 3],
        length: 3,
    };
    let packed = train.sample(&mut Pcg::new(&[41]), 4, 4, 4);
    assert_eq!(packed.tokens, [0, 1, 2, 0, 1, 2, 3, 2, 1, 3, 2, 1]);
    assert_eq!(
        packed.labels,
        [
            u32::MAX,
            0,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            3,
            1,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            3
        ]
    );
}

#[test]
#[ignore = "requires CUDA"]
fn accepted_weights_and_both_estimator_branches_match_a_coordinate_reference() -> Result<()> {
    let train = Observations {
        tokens: &[1, 2, 3, 3, 2, 1],
        labels: &[u32::MAX, u32::MAX, 0, u32::MAX, u32::MAX, 1],
        length: 3,
    };
    for refresh in [false, true] {
        let seed = (0..100)
            .find(|s| (Pcg::new(&[*s]).below(3) == 0) == refresh)
            .unwrap();
        let mut rng = Pcg::new(&[seed]);
        let _decision = rng.below(3);
        let sample = train.sample(&mut rng, 2, 1, 4);
        let row = Observations::from(&sample);
        let mut engine = Engine::new(
            Shape {
                width: 16,
                heads: 2,
                layers: 2,
                vocab: 4,
                max_length: 3,
                max_batch: 2,
            },
            1,
            9,
        )?;
        let x = engine.weights()?;
        let mut page = AdaptivePage::new(&mut engine, train, seed, PageRule::Single)?;
        let estimate = page.estimate.read()?;
        let start_objective = page.objective;
        let norm_squared: f64 = estimate.iter().map(|&g| f64::from(g).powi(2)).sum();
        row.forward(&mut engine)?;
        regularized_backward(&mut engine, page.penalty)?;
        let old = engine.parameters.grad.read()?;
        assert!(page.step(&mut engine, train, false)?);
        assert_eq!(page.refreshes, usize::from(refresh));
        assert_eq!(page.corrections, usize::from(!refresh));
        assert!(
            page.objective <= start_objective - 0.5 * f64::from(page.rate) * norm_squared + 1e-7
        );
        for ((w, x), g) in engine.weights()?.into_iter().zip(x).zip(&estimate) {
            assert!(close(w, (-*g).mul_add(page.rate, x)));
        }
        (if refresh { train } else { row }).forward(&mut engine)?;
        regularized_backward(&mut engine, page.penalty)?;
        let after = engine.parameters.grad.read()?;
        let actual = page.estimate.read()?;
        for i in 0..actual.len() {
            let reference = if refresh {
                after[i]
            } else {
                estimate[i] + (after[i] - old[i])
            };
            assert!(close(actual[i], reference));
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn switched_reference_uses_its_own_initial_objective_and_mean_loss() -> Result<()> {
    let tokens = [1, 2, 3, 0, 1, 3];
    let first = Observations {
        tokens: &tokens,
        labels: &[0, 1, 2, 3, 0, 1],
        length: 3,
    };
    let next = Observations {
        tokens: &tokens,
        labels: &[u32::MAX, 3, 0, 1, u32::MAX, u32::MAX],
        length: 3,
    };
    let mut engine = Engine::new(
        Shape {
            width: 16,
            heads: 2,
            layers: 2,
            vocab: 4,
            max_length: 3,
            max_batch: 2,
        },
        1,
        9,
    )?;
    let mut page = AdaptivePage::new(&mut engine, first, 77, PageRule::SquareRoot)?;
    assert!(page.step(&mut engine, first, false)?);
    next.forward(&mut engine)?;
    let before = next.mean_loss(&engine, 3)?
        + 0.5 * f64::from(page.penalty) * norm_squared(&engine, &engine.parameters.value)?;
    assert!(page.step(&mut engine, next, true)?);
    assert!(page.objective < before);
    next.forward(&mut engine)?;
    assert_eq!(page.loss, next.mean_loss(&engine, 3)?);
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn an_accepted_estimated_retry_uses_the_original_snapshot_and_previous_rate() -> Result<()> {
    let train = Observations {
        tokens: &[1, 2, 3, 3, 2, 1, 0, 1, 3, 1, 0, 1],
        labels: &[
            u32::MAX,
            u32::MAX,
            0,
            u32::MAX,
            u32::MAX,
            1,
            u32::MAX,
            u32::MAX,
            3,
            u32::MAX,
            u32::MAX,
            2,
        ],
        length: 3,
    };
    let mut engine = Engine::new(
        Shape {
            width: 16,
            heads: 2,
            layers: 2,
            vocab: 4,
            max_batch: 4,
            max_length: 3,
        },
        1,
        9,
    )?;
    let mut page = AdaptivePage::new(&mut engine, train, 7, PageRule::Retry)?;
    for _ in 0..100 {
        let old = engine.weights()?;
        let direction = page.estimate.read()?;
        let rate = page.rate;
        let retried = page.retries_accepted;
        if !page.step(&mut engine, train, false)? {
            break;
        }
        if page.retries_accepted > retried {
            assert_eq!(page.rate, rate);
            for ((w, x), g) in engine.weights()?.iter().zip(old).zip(direction) {
                assert!(close(*w, (-g).mul_add(rate, x)));
            }
            return Ok(());
        }
    }
    Err("the test stream did not exercise an accepted estimated retry".into())
}
