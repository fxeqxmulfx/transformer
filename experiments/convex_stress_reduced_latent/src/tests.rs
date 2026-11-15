use super::*;
use domain::Domain;
use head::{Head, PredictionBudget};
use projection::Target;
use training::{Config, Mode};

fn work() -> Work {
    Work::new(dense::MAX_WORK).unwrap()
}
fn fixture() -> (Domain, Vec<Record>) {
    let records = vec![
        Record {
            states: vec![0, 0],
            label: 0,
        },
        Record {
            states: vec![1, 0],
            label: 0,
        },
        Record {
            states: vec![2, 1],
            label: 2,
        },
        Record {
            states: vec![3, 2],
            label: 1,
        },
    ];
    let d = Domain::new(
        Shape {
            programs: 2,
            states: 8,
            classes: 5,
        },
        &records,
        1.,
        32.,
        &mut work(),
    )
    .unwrap();
    (d, records)
}
fn dense_shape(d: &Domain) -> dense::Shape {
    dense::Shape {
        programs: d.shape.programs,
        states: d.shape.states,
        classes: d.shape.classes,
    }
}
fn dense_domain(d: &Domain) -> dense::domain::Domain {
    dense::domain::Domain {
        shape: dense_shape(d),
        labels: d.labels.clone(),
        minimum_mass: d.minimum_mass,
        maximum_mass: d.maximum_mass,
    }
}
fn expand(d: &Domain, q: &Parameters) -> dense::Parameters {
    let sh = dense_shape(d);
    let mut values = vec![0.; sh.parameters()];
    values[..sh.programs].copy_from_slice(&q.values[..sh.programs]);
    for p in 0..sh.programs {
        for s in 0..sh.states {
            for c in 0..sh.classes {
                values[sh.at(p, s, c)] = q.gamma(d, p, s, c);
            }
        }
    }
    dense::Parameters { values }
}
fn target(d: &Domain, v: &dense::Parameters) -> Target {
    let sh = dense_shape(d);
    let mut linear = v.values[..sh.programs].to_vec();
    for (p, l) in linear.iter_mut().enumerate() {
        for s in 0..sh.states {
            if let Some(c) = d.labels[p * sh.states + s] {
                *l += v.values[sh.at(p, s, c)];
            }
        }
    }
    let mut free = vec![];
    for &(p, s) in &d.unknown {
        for c in 0..sh.classes {
            free.push(v.values[sh.at(p, s, c)]);
        }
    }
    Target { linear, free }
}
fn close(a: f64, b: f64, tolerance: f64) {
    assert!((a - b).abs() <= tolerance, "{a} differs from {b}");
}
fn nonuniform(d: &Domain) -> Parameters {
    let mut q = d.initial(4., &mut work()).unwrap();
    let n = d.shape.classes;
    let sum = (n * (n + 1) / 2) as f64;
    for &(p, s) in &d.unknown {
        let at = d.offsets[p * d.shape.states + s];
        for c in 0..n {
            q.values[at + c] = q.values[p] * (c + 1) as f64 / sum;
        }
    }
    d.check(&q, 1e-10, &mut work()).unwrap();
    q
}

#[test]
fn all_complete_coefficients_logits_and_certificates_match_dense() {
    let (d, _) = fixture();
    let q = nonuniform(&d);
    let mut w = work();
    let sparse = Head::build(&d, &q, &mut w).unwrap();
    let full = expand(&d, &q);
    let reference = dense::head::Head::build(dense_shape(&d), &full, &mut w).unwrap();
    let a = sparse.audit(&d, &q, &mut w).unwrap();
    let b = reference.audit(&full, &mut w).unwrap();
    assert_eq!(a.logical_coefficients, b.coefficients);
    assert!(a.stored_coefficients < a.logical_coefficients);
    assert!(
        sparse
            .blocks
            .iter()
            .flat_map(|b| &b.imaginary)
            .flatten()
            .any(|&x| x != 0.)
    );
    for p in 0..d.shape.programs {
        for s in 0..d.shape.states {
            for c in 0..d.shape.classes {
                let (re, im) = sparse.coefficient(p, s, c).unwrap();
                let at = reference.at(p, s, c);
                assert_eq!(re.to_bits(), reference.real[at].to_bits());
                assert_eq!(im.to_bits(), reference.imaginary[at].to_bits());
            }
        }
    }
    for (a, b) in a.column_l1.iter().zip(&b.column_l1) {
        close(*a, *b, 1e-12);
    }
    for x in 0..8 {
        for y in 0..8 {
            let states = [x, y];
            assert_eq!(
                sparse.logits(&states, &mut w).unwrap(),
                reference.logits(&states, &mut w).unwrap()
            );
            let a = sparse
                .predict(&states, &mut PredictionBudget::new(8).unwrap(), &mut w)
                .unwrap();
            let b = reference
                .predict(
                    &states,
                    &mut dense::head::PredictionBudget::new(8).unwrap(),
                    &mut w,
                )
                .unwrap();
            assert_eq!(a, b);
        }
    }
}

#[test]
fn arbitrary_off_domain_projection_matches_full_reference() {
    let (d, _) = fixture();
    let sh = dense_shape(&d);
    let mut w = work();
    assert_ne!(d.known[0], d.known[1]);
    let mut seed = 71u64;
    for case in 0..20 {
        let values = (0..sh.parameters())
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                if case == 0 {
                    0.
                } else if case == 1 {
                    50.
                } else {
                    (seed % 4001) as f64 / 137. - 12.
                }
            })
            .collect();
        let v = dense::Parameters { values };
        let t = target(&d, &v);
        let a = projection::project(&d, &t, &mut w).unwrap();
        let b = dense::projection::project(&dense_domain(&d), &v, &mut w).unwrap();
        let full = expand(&d, &a.parameters);
        for (x, y) in full.values.iter().zip(&b.parameters.values) {
            close(*x, *y, 3e-8);
        }
        close(a.multiplier, b.multiplier, 3e-8);
        assert!(a.maximum_kkt_residual <= 1e-7);
        let mut constant = 0.;
        for p in 0..sh.programs {
            constant += v.values[p].powi(2) - t.linear[p].powi(2) / (1 + d.known[p]) as f64;
            for s in 0..sh.states {
                if d.labels[p * sh.states + s].is_some() {
                    for c in 0..sh.classes {
                        constant += v.values[sh.at(p, s, c)].powi(2);
                    }
                }
            }
        }
        let full_objective: f64 = full
            .values
            .iter()
            .zip(&v.values)
            .map(|(x, y)| (x - y).powi(2))
            .sum();
        close(full_objective, a.reduced_objective + constant, 1e-7);
        close(full_objective, b.squared_distance, 1e-7);
        for total in [1., 4., 16., 32.] {
            let alt = d.initial(total, &mut w).unwrap();
            assert!(a.reduced_objective <= t.objective(&d, &alt).unwrap() + 1e-7);
        }
    }
}

#[test]
fn reduced_gradient_is_the_full_pullback_and_matches_finite_differences() {
    let (d, mut rows) = fixture();
    rows.push(rows[0].clone());
    let mut w = work();
    let q = nonuniform(&d);
    let head = Head::build(&d, &q, &mut w).unwrap();
    let full = expand(&d, &q);
    let dh = dense::head::Head::build(dense_shape(&d), &full, &mut w).unwrap();
    let (loss, g) = training::gradient(&d, &head, &rows, &mut w).unwrap();
    let (old_loss, old_g) = dense::training::gradient(&dh, &rows, &mut w).unwrap();
    close(loss, old_loss, 1e-14);
    let pullback = target(&d, &old_g);
    for (a, b) in g.values[..d.shape.programs].iter().zip(&pullback.linear) {
        close(*a, *b, 1e-14);
    }
    assert!(g.values[d.shape.programs..].iter().all(|&x| x == 0.));
    for k in 0..q.values.len() {
        let eps = 1e-5;
        let mut left = q.clone();
        let mut right = q.clone();
        left.values[k] -= eps;
        right.values[k] += eps;
        let a = training::loss(&Head::build(&d, &left, &mut w).unwrap(), &rows, &mut w).unwrap();
        let b = training::loss(&Head::build(&d, &right, &mut w).unwrap(), &rows, &mut w).unwrap();
        close((b - a) / (2. * eps), g.values[k], 2e-8);
    }
    let fake = [Record {
        states: vec![7, 7],
        label: 1,
    }];
    assert!(training::gradient(&d, &head, &fake, &mut w).is_err());
}

#[test]
fn actual_forward_full_and_minibatch_tail_match_dense_training() {
    let (d, base) = fixture();
    let rows: Vec<_> = (0..130).map(|i| base[i % base.len()].clone()).collect();
    for (mode, rate) in [Mode::Forward, Mode::Full, Mode::Mini]
        .into_iter()
        .flat_map(|mode| [3., 1e6].map(|rate| (mode, rate)))
    {
        let config = Config {
            mode,
            epochs: match mode {
                Mode::Forward => 0,
                Mode::Full => 4,
                Mode::Mini => 1,
            },
            batch_size: 127,
            learning_rate: rate,
            initial_mass: 8.,
        };
        let a = training::train(&d, &rows, config.clone(), &mut work()).unwrap();
        let b = dense::training::train(&dense_domain(&d), &rows, config, &mut work()).unwrap();
        assert_eq!(a.steps.len(), b.steps.len());
        for (x, y) in a.steps.iter().zip(&b.steps) {
            assert_eq!(
                (x.epoch, x.first_record, x.end_record_exclusive),
                (y.epoch, y.first_record, y.end_record_exclusive)
            );
            close(x.before_ce, y.before_ce, 1e-10);
            close(x.after_ce, y.after_ce, 1e-10);
            close(x.parameter_change_l2, y.parameter_change_l2, 1e-8);
        }
        if mode == Mode::Mini {
            assert_eq!(
                a.steps.last().unwrap().end_record_exclusive - a.steps.last().unwrap().first_record,
                3
            );
        }
        if mode != Mode::Forward {
            assert!(a.final_ce < a.initial_ce);
        }
        for (x, y) in expand(&d, &a.parameters)
            .values
            .iter()
            .zip(&b.parameters.values)
        {
            close(*x, *y, 1e-8);
        }
        close(a.final_ce, b.final_ce, 1e-10);
        assert!(!a.exact_real_projection_certified && !a.target_realizability_certified);
    }
}

#[test]
fn genuine_ties_use_exact_products_and_budgets_remain_refusals() {
    let mut w = work();
    let rows = [Record {
        states: vec![0],
        label: 0,
    }];
    let d = Domain::new(
        Shape {
            programs: 1,
            states: 4,
            classes: 4,
        },
        &rows,
        1.,
        8.,
        &mut w,
    )
    .unwrap();
    let mut q = d.initial(2., &mut w).unwrap();
    let at = d.offsets[1];
    q.values[at..at + 4].copy_from_slice(&[1., 1., 0., 0.]);
    let h = Head::build(&d, &q, &mut w).unwrap();
    let a = h
        .predict(&[1], &mut PredictionBudget::new(1).unwrap(), &mut w)
        .unwrap();
    assert_eq!(a.maximizers, vec![0, 1]);
    assert_eq!(a.certificate, head::Certificate::ExactDyadics);
    assert!(
        h.predict(&[1], &mut PredictionBudget::new(0).unwrap(), &mut w)
            .is_err()
    );
    q.values[at] = 1f64.next_up();
    let h = Head::build(&d, &q, &mut w).unwrap();
    let b = dense::head::Head::build(dense_shape(&d), &expand(&d, &q), &mut w).unwrap();
    assert_eq!(
        h.predict(&[1], &mut PredictionBudget::new(1).unwrap(), &mut w)
            .unwrap(),
        b.predict(
            &[1],
            &mut dense::head::PredictionBudget::new(1).unwrap(),
            &mut w
        )
        .unwrap()
    );
    assert!(
        h.predict(&[4], &mut PredictionBudget::new(1).unwrap(), &mut w)
            .is_err()
    );
}

#[test]
fn persistence_preserves_bits_and_rejects_corruption_even_after_rehashing() {
    use sha2::{Digest, Sha256};
    let (d, _) = fixture();
    let mut q = d.initial(4., &mut work()).unwrap();
    let at = d.offsets[4];
    q.values[at..at + 5].copy_from_slice(&[2., -0., f64::from_bits(1), 0., 0.]);
    let mut bytes = vec![];
    let hash = persistence::write(&d, &q, &mut bytes, &mut work()).unwrap();
    assert_eq!(bytes.len() as u64, persistence::bytes(&d));
    let (loaded, r) =
        persistence::read(bytes.as_slice(), hash, bytes.len() as u64, &mut work()).unwrap();
    assert_eq!(d, loaded);
    assert_eq!(
        q.values.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        r.values.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(
        Head::build(&d, &q, &mut work()).unwrap(),
        Head::build(&loaded, &r, &mut work()).unwrap()
    );
    for pos in [0, 8, 16, 32, 56, 56 + 4 * d.shape.rows(), bytes.len() - 1] {
        let mut bad = bytes.clone();
        bad[pos] ^= 1;
        assert!(persistence::read(bad.as_slice(), hash, bad.len() as u64, &mut work()).is_err());
    }
    let mut bad = bytes.clone();
    let end = bad.len() - 32;
    bad[56 + 4 * d.shape.rows()..64 + 4 * d.shape.rows()].copy_from_slice(&f64::NAN.to_le_bytes());
    let new: [u8; 32] = Sha256::digest(&bad[..end]).into();
    bad[end..].copy_from_slice(&new);
    assert!(persistence::read(bad.as_slice(), hash, bad.len() as u64, &mut work()).is_err());
    assert!(persistence::read(bad.as_slice(), new, bad.len() as u64, &mut work()).is_err());
    let mut extra = bytes.clone();
    extra.push(0);
    assert!(persistence::read(extra.as_slice(), hash, extra.len() as u64, &mut work()).is_err());
    assert!(
        persistence::read(
            &bytes[..bytes.len() - 1],
            hash,
            (bytes.len() - 1) as u64,
            &mut work()
        )
        .is_err()
    );
}

#[test]
fn large_fully_anchored_head_stores_only_exact_nonzero_class_support() {
    let sh = Shape {
        programs: 128,
        states: 196,
        classes: 193,
    };
    let mut w = work();
    let labels = (0..sh.rows()).map(|i| Some(i % 3)).collect();
    let d = Domain::from_labels(sh, labels, 1., 32., &mut w).unwrap();
    assert_eq!(d.parameters(), 128);
    let q = d.initial(8., &mut w).unwrap();
    let h = Head::build(&d, &q, &mut w).unwrap();
    let audit = h.audit(&d, &q, &mut w).unwrap();
    assert_eq!(audit.logical_coefficients, 9_683_968);
    assert_eq!(audit.stored_coefficients, 150_528);
    assert!(audit.stored_coefficients * 50 < audit.logical_coefficients);
    assert!(persistence::bytes(&d) < 200_000);
    assert_eq!(persistence::bytes(&d), 101_464);
    assert!(audit.column_l1.iter().all(|&x| x <= 2. * 32. * 196.));
}

#[test]
fn allocation_work_dictionary_and_invalid_parameter_admission_are_enforced() {
    assert!(
        Shape {
            programs: usize::MAX,
            states: 4,
            classes: 2
        }
        .admit(0, 0, 0)
        .is_err()
    );
    assert!(
        Shape {
            programs: 1_000_000,
            states: 4096,
            classes: 1024
        }
        .admit(0, 0, 0)
        .is_err()
    );
    let (d, rows) = fixture();
    assert!(Domain::new(d.shape, &rows, 1., 32., &mut Work::new(1).unwrap()).is_err());
    let mut bad = rows.clone();
    bad.push(Record {
        states: rows[0].states.clone(),
        label: 1,
    });
    assert!(Domain::new(d.shape, &bad, 1., 32., &mut work()).is_err());
    let mut extra = d.clone();
    extra.labels[7] = Some(4);
    assert!(extra.validate(&mut work()).is_err());
    let mut q = d.initial(4., &mut work()).unwrap();
    q.values[0] = f64::INFINITY;
    assert!(Head::build(&d, &q, &mut work()).is_err());
    assert!(
        d.shape
            .admit(65537, d.unknown.len(), d.head_columns_bound)
            .is_err()
    );
}

#[test]
fn missing_columns_and_modified_complex_coefficients_fail_the_full_audit() {
    let (d, _) = fixture();
    let q = nonuniform(&d);
    let head = Head::build(&d, &q, &mut work()).unwrap();
    let mut missing = head.clone();
    missing.blocks[0].classes.pop();
    missing.blocks[0].real.pop();
    missing.blocks[0].imaginary.pop();
    assert!(missing.audit(&d, &q, &mut work()).is_err());
    let mut changed = head;
    changed.blocks[0].imaginary[0][1] += 1e-3;
    assert!(changed.audit(&d, &q, &mut work()).is_err());
}

#[test]
fn both_disks_and_shared_filesystem_reserves_are_required() {
    use resources::sufficient;
    let g = 1024u64.pow(3);
    assert!(sufficient(100 * g, 16 * g - 1, false, 0).is_err());
    assert!(sufficient(2 * g - 1, 100 * g, false, 0).is_err());
    assert!(sufficient(18 * g, 18 * g, true, 1).is_err());
    assert!(sufficient(18 * g + 1, 18 * g + 1, true, 1).is_ok());
    assert!(sufficient(u64::MAX, u64::MAX, true, u64::MAX).is_err());
}

#[test]
fn guarded_checkpoint_write_stops_before_open_and_never_overwrites() {
    let (d, _) = fixture();
    let q = d.initial(4., &mut work()).unwrap();
    let root = std::env::temp_dir().join(format!("reduced-latent-check-{}", std::process::id()));
    std::fs::create_dir(&root).unwrap();
    let path = root.join("checkpoint.bin");
    assert!(persistence::write_file(&d, &q, &root, &root, &path, u64::MAX, &mut work()).is_err());
    assert!(!path.exists());
    let hash = persistence::write_file(&d, &q, &root, &root, &path, 0, &mut work()).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    persistence::read(bytes.as_slice(), hash, bytes.len() as u64, &mut work()).unwrap();
    assert!(persistence::write_file(&d, &q, &root, &root, &path, 0, &mut work()).is_err());
    assert_eq!(bytes, std::fs::read(&path).unwrap());
    std::fs::remove_file(path).unwrap();
    std::fs::remove_dir(root).unwrap();
}
