use crate::{
    coordinates::{Fit, Record},
    head::Head,
    training::{Mode, Policy},
    *,
};

fn rows() -> Vec<Record> {
    [(0., 1), (0., 1), (0.5, 2)]
        .into_iter()
        .enumerate()
        .map(|(id, (feature, label))| Record { id, feature, label })
        .collect()
}
fn coordinates() -> Vec<f64> {
    (0..5).map(|i| i as f64 / 4.).collect()
}
fn policy(mode: Mode) -> Policy {
    Policy {
        mode,
        iterations: if mode == Mode::Forward { 0 } else { 8 },
        batch: 2,
        learning_rate: 8.,
        initial_margin: 4.,
        minimum_margin: 1.,
        radius: 128.,
    }
}

#[test]
fn two_observed_anchors_identify_an_unseen_raw_class_in_the_complete_product() -> Result<()> {
    let fit = Fit::new(coordinates(), &rows())?;
    assert_eq!(fit.anchor_ids, [0, 2]);
    assert_eq!((fit.slope, fit.intercept), (0.5, 0.25));
    let h = Head::new(fit, 4., 32.)?;
    assert_eq!(h.coefficients.len(), 40);
    assert_eq!(h.coefficients[3 * 5 + 4], 1.); // Nonzero imaginary coordinate.
    for x in [-3., 0., 0.5, 1., 4.] {
        let z = h.logits(x);
        let independent = h.independent_logits(x);
        for (c, (&a, &b)) in z.iter().zip(&independent).enumerate() {
            let e = h.fit.coordinates[c];
            let expected = 4. * (2. * e * (0.5 * x + 0.25) - e * e) / (x.abs() + 1.);
            assert!((a - b).abs() < 1e-12 && (a - expected).abs() < 1e-12);
        }
        assert!(head::features(x).iter().all(|v| v.abs() <= 1.));
    }
    assert_eq!(argmax(&h.logits(1.)), 3);
    assert!(!rows().iter().any(|r| r.label == 3));
    assert!(h.max_column_l1() <= 32.);
    assert!(h.max_column_l1() > 0.);
    Ok(())
}

#[test]
fn transported_class_coordinates_preserve_predictions() -> Result<()> {
    let permutation = [4, 0, 3, 1, 2];
    let mut coords = vec![0.; 5];
    for (c, e) in coordinates().into_iter().enumerate() {
        coords[permutation[c]] = e;
    }
    let mapped: Vec<_> = rows()
        .into_iter()
        .map(|mut r| {
            r.label = permutation[r.label];
            r
        })
        .collect();
    let a = Head::new(Fit::new(coordinates(), &rows())?, 4., 32.)?;
    let b = Head::new(Fit::new(coords, &mapped)?, 4., 32.)?;
    for x in [0., 0.5, 1., 1.5] {
        for (c, &renamed) in permutation.iter().enumerate() {
            assert_eq!(a.logits(x)[c], b.logits(x)[renamed]);
        }
    }
    Ok(())
}

#[test]
fn all_modes_replay_and_minibatch_retains_repeated_inputs_and_short_tail() -> Result<()> {
    for mode in [Mode::Forward, Mode::Full, Mode::Mini] {
        let out = training::train(Fit::new(coordinates(), &rows())?, &rows(), policy(mode))?;
        assert_eq!(argmax(&out.head.logits(1.)), 3);
        assert!(out.final_ce <= out.initial_ce + 1e-12);
        assert_eq!(
            out.work.gradient_records,
            if mode == Mode::Forward { 0 } else { 24 }
        );
        if mode == Mode::Forward {
            assert_eq!(out.work.effective_updates, 0);
        } else {
            assert!(out.work.effective_updates > 0);
            assert!(out.final_ce < out.initial_ce);
        }
        if mode == Mode::Mini {
            assert_eq!(out.steps.len(), 16);
            assert_eq!(out.steps[0].record_ids, vec![0, 1]);
            assert_eq!(out.steps[1].record_ids, vec![2]);
        }
        let saved: training::Outcome = serde_json::from_str(&serde_json::to_string(&out)?)?;
        training::replay(&saved, &rows())?;
        let mut bad = saved.clone();
        bad.head.coefficients[0] += 1.;
        assert!(training::replay(&bad, &rows()).is_err());
        if !saved.steps.is_empty() {
            let mut bad = saved;
            bad.steps[0].record_ids[0] = 999;
            assert!(training::replay(&bad, &rows()).is_err());
        }
    }
    Ok(())
}

#[test]
fn actual_ce_derivative_and_short_batch_denominator_agree() -> Result<()> {
    let fit = Fit::new(coordinates(), &rows())?;
    let h = Head::new(fit.clone(), 4., 128.)?;
    let u = Head::new(fit.clone(), 1., 128.)?;
    for batch in [&rows()[..], &rows()[2..]] {
        let (loss, g) = training::gradient(&h, &u, batch);
        assert!((loss - training::score(&h, batch)).abs() < 1e-12);
        let eps = 1e-4;
        let lo = Head::new(fit.clone(), 4. - eps, 128.)?;
        let hi = Head::new(fit.clone(), 4. + eps, 128.)?;
        let numerical = (training::score(&hi, batch) - training::score(&lo, batch)) / (2. * eps);
        assert!((g - numerical).abs() < 1e-9);
    }
    Ok(())
}

#[test]
fn missing_prior_and_invalid_or_excessive_inputs_refuse() -> Result<()> {
    let mut inconsistent = rows();
    inconsistent.push(Record {
        id: 3,
        feature: 1.,
        label: 4,
    });
    assert!(Fit::new(coordinates(), &inconsistent).is_err());
    let mut duplicate = rows();
    duplicate[1].id = 0;
    assert!(Fit::new(coordinates(), &duplicate).is_err());
    assert!(Fit::new(vec![0.; 5], &rows()).is_err());
    assert!(Fit::new(vec![f64::NAN; 5], &rows()).is_err());
    let fit = Fit::new(coordinates(), &rows())?;
    assert!(Head::new(fit.clone(), 4., 1.).is_err());
    let mut p = policy(Mode::Full);
    p.iterations = usize::MAX;
    assert!(p.forecast(3, 5).is_err());
    assert!(p.forecast(usize::MAX, usize::MAX).is_err());
    p = policy(Mode::Full);
    p.iterations = 1024;
    assert!(p.forecast(200000, 1024).is_err());
    p = policy(Mode::Forward);
    p.radius = 1.;
    assert!(training::train(fit, &rows(), p).is_err());
    Ok(())
}
