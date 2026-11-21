use super::*;
use arithmetic::{dataset::Training, reference::SplitMix64};

/// A determining cross and one off-cross cell (PT4), with shifts unrelated to NV11.
fn fixture() -> (Training, Vec<u32>, Vec<u32>) {
    let p = P as u32;
    let mut g: Vec<_> = (0..p).collect();
    let mut h = g.clone();
    SplitMix64(17).shuffle(&mut g);
    SplitMix64(31).shuffle(&mut h);
    let mut rows = vec![];
    let mut add = |a, b| {
        rows.push([a, b, (g[a as usize] + h[b as usize] + 43 * a * b) % p]);
    };
    for t in 0..p {
        add(0, t);
        if t != 0 {
            add(t, 0);
        }
    }
    add(1, 1);
    (
        Training {
            vocabulary: P,
            rows,
        },
        g,
        h,
    )
}

#[test]
fn fitted_phases_reach_the_native_logits() -> Result<()> {
    let (training, g, h) = fixture();
    let mut weights = circuit::fit_class(&training)?;
    let inputs = [
        [0, 0],
        [96, 0],
        [0, 96],
        [13, 1],
        [19, 2],
        [47, 4],
        [3, 8],
        [18, 16],
        [29, 32],
        [67, 64],
        [63, 63],
        [43, 95],
        [96, 96],
    ];
    let tokens = inputs.concat();
    for shifted in [true, false] {
        if !shifted {
            circuit::set_shifts(&mut weights, &[0; P], &[0; P]);
        }
        let logits = oracle::main_model(
            &config(),
            &forward_weights(&weights),
            &tokens,
            inputs.len(),
            2,
        );
        for (logits, [a, b]) in answer_logits(&logits).zip(inputs) {
            let offset = if shifted {
                g[a as usize] + h[b as usize]
            } else {
                0
            };
            let label = (43 * a * b + offset) % P as u32;
            let mut score = Score::empty();
            score.observe(logits, label);
            assert!(
                score.min_margin > 0.0,
                "pair ({a}, {b}), shift {shifted}, margin {}",
                score.min_margin
            );
        }
    }
    Ok(())
}

#[test]
fn fit_requires_consistent_and_determining_training_labels() {
    let (mut training, g, h) = fixture();
    training
        .rows
        .push([2, 3, (g[2] + h[3] + 43 * 6 + 1) % P as u32]);
    assert!(
        circuit::fit_class(&training)
            .unwrap_err()
            .to_string()
            .contains("contradict")
    );
    training.rows.truncate(training.rows.len() - 2);
    assert!(
        circuit::fit_class(&training)
            .unwrap_err()
            .to_string()
            .contains("determine")
    );
    training.rows.push([0, P as u32, 0]);
    assert!(circuit::fit_class(&training).is_err());
}

#[test]
fn saved_parameter_block_replays_and_rejects_truncation() -> Result<()> {
    let weights = circuit::fit_class(&fixture().0)?;
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("parameters.f32");
    save(&path, &weights)?;
    assert_eq!(weights, load(&path)?);
    fs::write(&path, [0; 4])?;
    assert!(load(&path).is_err());
    Ok(())
}

#[test]
fn likelihood_fits_an_arbitrary_row_shift_from_observed_logits() -> Result<()> {
    let (_, offsets, _) = fixture();
    let p = P as u32;
    let rows: Vec<_> = (0..p)
        .map(|a| {
            let b = (a + 12) % p;
            [a, b, (a * b + offsets[a as usize]) % p]
        })
        .collect();
    let mut training = Training {
        vocabulary: P,
        rows,
    };
    let fitted = shifts::fit(&training)?;
    let mut expected = circuit::parameters(1);
    circuit::set_shifts(&mut expected, &offsets, &[0; P]);
    assert_eq!(fitted, expected);
    training.rows.pop();
    assert!(
        shifts::fit(&training)
            .unwrap_err()
            .to_string()
            .contains("unobserved")
    );
    Ok(())
}
