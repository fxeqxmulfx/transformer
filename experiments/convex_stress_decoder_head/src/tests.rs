use crate::*;
use serde_json::json;

fn source() -> Result<tagged::Model> {
    tagged::Model::construct(
        7,
        vec![0, 0, 2, 5],
        vec![1, 3, 1, 4],
        vec![3, 1, 3, 1, 6, 6],
    )
}
fn endpoint(mode: Mode) -> Endpoint {
    Endpoint {
        mode,
        source_reference: "saved-training-fixture:0".into(),
        source_gradient_training_run: false,
        source_optimizer_updates: Some(0),
    }
}
fn fixture(wrong: bool, mode: Mode) -> Result<Model> {
    let source = source()?;
    let records = vec![
        records::Record {
            id: 10,
            cell: 0,
            label: source.output_labels[0],
        },
        records::Record {
            id: 11,
            cell: 0,
            label: source.output_labels[0],
        },
        records::Record {
            id: 12,
            cell: 15,
            label: if wrong { 6 } else { source.output_labels[15] },
        },
    ];
    let (a, b) = (digest(&source)?, digest(&records)?);
    Model::new(source, records, endpoint(mode), &a, &b)
}
fn loss_at(m: &Model, coefficients: &[f64], positions: &[usize]) -> f64 {
    positions
        .iter()
        .map(|&i| {
            let r = &m.records[i];
            tagged::loss(
                &model::product(m.features(r.cell).unwrap(), coefficients, m.source.vocab),
                r.label,
            )
        })
        .sum::<f64>()
        / positions.len() as f64
}
fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol, "{a} differs from {b}");
}

#[test]
fn original_errors_select_confidence_without_changing_declared_labels() -> Result<()> {
    let correct = fixture(false, Mode::Forward)?;
    let wrong = fixture(true, Mode::Forward)?;
    let p: serde_json::Value = serde_json::from_str(PROTOCOL)?;
    assert_eq!(p["margins"], json!(BANK));
    assert_eq!(p["prediction_limit"], json!(PREDICTION_LIMIT));
    assert_eq!(p["maximum_records"], json!(records::MAX_RECORDS));
    assert_eq!(p["scalar_work_limit"], json!(WORK_CAP));
    assert_eq!(correct.selection.selected_margin, 24.);
    assert_eq!(wrong.selection.selected_margin, 2.);
    assert_eq!(wrong.selection.raw_errors, 1);
    assert_eq!(wrong.selection.original_records, 3);
    assert_eq!(wrong.records[0].cell, wrong.records[1].cell);
    assert_ne!(wrong.records[0].id, wrong.records[1].id);
    for m in [&correct, &wrong] {
        for c in &m.selection.candidates {
            let coeff = model::initializer(&m.source, &m.code, c.margin);
            close(loss_at(m, &coeff, &[0, 1, 2]), c.mean_ce, 1e-12);
        }
        for cell in 0..m.source.n * m.source.n {
            let z = m.logits_cell(cell)?;
            let label = m.source.output_labels[cell];
            assert!(
                z.iter()
                    .enumerate()
                    .all(|(c, &v)| c == label || v < z[label])
            );
        }
    }
    let mut all_wrong = wrong.records.clone();
    for r in &mut all_wrong {
        r.label = 0;
    }
    assert_eq!(
        selection::select(&wrong.source, &all_wrong).selected_margin,
        1.
    );
    Ok(())
}

#[test]
fn every_margin_uses_actual_complex_products_and_complete_budget() -> Result<()> {
    let m = fixture(true, Mode::Full)?;
    let (q, v) = (m.source.q, m.source.vocab);
    assert!(m.code.iter().any(|p| p.im.abs() > 0.1));
    assert!(
        m.coefficients
            .iter()
            .enumerate()
            .any(|(i, &w)| (i / v) % 2 == 1 && w.abs() > 0.1)
    );
    for i in 0..q {
        for j in 0..q {
            let inner: C = (0..q)
                .map(|f| m.code[i * q + f] * m.code[j * q + f].conj())
                .sum();
            close(inner.re, f64::from(i == j), 1e-12);
            close(inner.im, 0., 1e-12);
        }
    }
    for margin in BANK {
        let w = model::initializer(&m.source, &m.code, margin);
        for c in 0..v {
            let charged: f64 = (0..2 * q).map(|k| w[k * v + c].abs()).sum();
            assert!(charged <= m.radius);
        }
        for s in 0..q {
            let phi = &m.code[s * q..(s + 1) * q];
            let z = model::product(phi, &w, v);
            for c in 0..v {
                let independent: C = (0..q)
                    .map(|f| phi[f] * C::new(w[2 * f * v + c], w[2 * f * v + v + c]))
                    .sum();
                close(z[c], independent.re, 1e-12);
                close(
                    z[c],
                    if m.source.decoder[s] == c { margin } else { 0. },
                    1e-11,
                );
                close(
                    tagged::loss(&z, c),
                    selection::margin_ce(v, margin)
                        + if m.source.decoder[s] == c { 0. } else { margin },
                    1e-11,
                );
            }
        }
    }
    for cell in 0..m.source.n * m.source.n {
        assert_eq!(
            m.logits_cell(cell)?,
            model::product(m.features(cell)?, m.coefficients(), v)
        );
    }
    Ok(())
}

#[test]
fn full_real_imaginary_gradient_preserves_repetitions_and_tail_denominators() -> Result<()> {
    let m = fixture(true, Mode::Mini)?;
    let full = m.batch_gradient(&[0, 1, 2])?;
    let repeated = m.batch_gradient(&[0, 1])?;
    let tail = m.batch_gradient(&[2])?;
    close(tail.loss, selection::margin_ce(7, 2.) + 2., 1e-12);
    close(full.loss, (2. * repeated.loss + tail.loss) / 3., 1e-12);
    assert_eq!(full.coefficients.len(), 2 * m.source.q * m.source.vocab);
    for (k, &g) in full.coefficients.iter().enumerate() {
        close(
            g,
            (2. * repeated.coefficients[k] + tail.coefficients[k]) / 3.,
            1e-12,
        );
    }
    for positions in [&[0, 1, 2][..], &[0, 1][..], &[2][..]] {
        let analytic = m.batch_gradient(positions)?;
        for (k, &g) in analytic.coefficients.iter().enumerate() {
            let mut w = m.coefficients.clone();
            w[k] += 1e-5;
            let plus = loss_at(&m, &w, positions);
            w[k] -= 2e-5;
            close(g, (plus - loss_at(&m, &w, positions)) / 2e-5, 2e-9);
        }
    }
    let mut other = m.coefficients.clone();
    for (i, w) in other.iter_mut().enumerate() {
        *w += (i as f64).sin();
    }
    for t in [0., 0.25, 0.5, 0.75, 1.] {
        let mix: Vec<_> = m
            .coefficients
            .iter()
            .zip(&other)
            .map(|(a, b)| t * a + (1. - t) * b)
            .collect();
        assert!(
            loss_at(&m, &mix, &[0, 1, 2])
                <= t * full.loss + (1. - t) * loss_at(&m, &other, &[0, 1, 2]) + 1e-12
        );
    }
    Ok(())
}

#[test]
fn saved_artifacts_reject_source_record_coefficient_cache_and_metadata_mutations() -> Result<()> {
    let m = fixture(true, Mode::Full)?;
    let saved = serde_json::to_value(&m)?;
    assert_eq!(Model::load(saved.clone(), m.binding())?, m);
    for path in [
        "/source/rows/0",
        "/source/output_labels/0",
        "/source/decoder/0",
        "/records/0/label",
        "/records/1/id",
        "/endpoint/source_optimizer_updates",
        "/selection/selected_margin",
        "/selection/candidates/0/mean_ce",
        "/bounds/buffer_bytes",
        "/coefficients/3",
        "/table/0",
        "/radius",
    ] {
        let mut bad = saved.clone();
        let value = bad.pointer_mut(path).unwrap();
        *value = if let Some(n) = value.as_u64() {
            json!(n + 1)
        } else {
            json!(value.as_f64().unwrap() + 1.)
        };
        assert!(Model::load(bad, m.binding()).is_err(), "accepted {path}");
    }
    let mut bad = saved.clone();
    bad["code"][0] = serde_json::to_value(C::new(0., 0.))?;
    assert!(Model::load(bad, m.binding()).is_err());
    for path in [
        "/binding/source_sha256",
        "/binding/records_sha256",
        "/binding/protocol_sha256",
        "/binding/endpoint_sha256",
        "/endpoint/source_reference",
    ] {
        let mut bad = saved.clone();
        *bad.pointer_mut(path).unwrap() = json!("changed");
        assert!(Model::load(bad, m.binding()).is_err());
    }
    let mut bad = saved;
    bad["unaccounted"] = json!(true);
    assert!(Model::load(bad, m.binding()).is_err());
    Ok(())
}

#[test]
fn every_mode_retains_inherited_metadata_and_adds_zero_optimizer_updates() -> Result<()> {
    let mut previous = None;
    for mode in [Mode::Full, Mode::Mini, Mode::Forward] {
        let m = fixture(false, mode)?;
        assert_eq!(m.endpoint().mode, mode);
        assert!(!m.endpoint().source_gradient_training_run);
        assert_eq!(m.bounds().new_optimizer_updates, 0);
        assert_eq!(m.bounds().new_gradient_batches, 0);
        if let Some(w) = previous {
            assert_eq!(m.coefficients, w);
        }
        previous = Some(m.coefficients.clone());
        assert_eq!(Model::load(serde_json::to_value(&m)?, m.binding())?, m);
    }
    Ok(())
}

#[test]
fn invalid_inputs_and_excessive_work_fail_before_construction() -> Result<()> {
    let m = fixture(true, Mode::Forward)?;
    assert!(forecast(193, 576, 193, 18528)?.buffer_bytes < 2 * 1024 * 1024 * 1024);
    for (n, q, v, r) in [
        (258, 6, 7, 3),
        (4, 772, 7, 3),
        (4, 6, 1025, 3),
        (4, 6, 7, 0),
        (4, 6, 7, records::MAX_RECORDS + 1),
        (257, 771, 1024, records::MAX_RECORDS),
    ] {
        assert!(forecast(n, q, v, r).is_err());
    }
    for prefix in [&[][..], &[0], &[-1, 0], &[0, 4], &[7, 0, 0]] {
        assert!(m.logits(prefix).is_err());
    }
    assert!(m.logits(&vec![0; PREDICTION_LIMIT]).is_ok());
    assert!(m.logits(&vec![0; PREDICTION_LIMIT + 1]).is_err());
    assert!(m.logits_cell(16).is_err());
    for positions in [&[][..], &[0, 0], &[3]] {
        assert!(m.batch_gradient(positions).is_err());
    }
    let mut duplicate = m.records.clone();
    duplicate[1].id = duplicate[0].id;
    assert!(
        Model::new(
            m.source.clone(),
            duplicate.clone(),
            m.endpoint.clone(),
            &digest(&m.source)?,
            &digest(&duplicate)?
        )
        .is_err()
    );
    assert!(
        Model::new(
            m.source.clone(),
            m.records.clone(),
            m.endpoint.clone(),
            "wrong",
            &digest(&m.records)?
        )
        .is_err()
    );
    let mut source = m.source.clone();
    source.latent_labels[0] += 1;
    assert!(
        Model::new(
            source.clone(),
            m.records.clone(),
            m.endpoint.clone(),
            &digest(&source)?,
            &digest(&m.records)?
        )
        .is_err()
    );
    Ok(())
}
