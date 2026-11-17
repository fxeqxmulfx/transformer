use crate::{
    C,
    block::Block,
    cache::Cache,
    estimate,
    geometry::{self, Geometry},
    history::{self, Entry},
    model::Model,
};
use bilinear::train::{Mode, Record};
use rotary::atom::{Family, operator};

#[test]
fn actual_complete_norm_matches_independent_complex_svd() {
    let mut rng = baseline::random::Random::new(1703);
    let mut matrices = vec![
        [[C::default(); 2]; 2],
        [
            [C::new(3., 4.), C::default()],
            [C::default(), C::new(0., 2.)],
        ],
        [
            [C::new(1., 2.), C::new(3., 4.)],
            [C::new(2., 4.), C::new(6., 8.)],
        ],
    ];
    for _ in 0..128 {
        let mut m = [[C::default(); 2]; 2];
        for z in m.iter_mut().flatten() {
            *z = C::new(4. * rng.random() - 2., 4. * rng.random() - 2.);
        }
        matrices.push(m);
    }
    for m in matrices {
        let svd = bilinear::algebra::svd(2, 2, &[m[0][0], m[1][0], m[0][1], m[1][1]]).unwrap();
        let norm = geometry::hilbert_norm(m).unwrap();
        assert!((norm - svd.values[0]).abs() < 2e-12 * (1. + norm));
        assert_eq!(svd.values.len(), 2);
        let projected = Geometry::Hilbert.matrix(m).unwrap();
        assert!(geometry::hilbert_norm(projected).unwrap() <= 1. + 2e-12);
        if norm <= 1. {
            assert_eq!(projected, m);
        }
    }
    assert!(geometry::hilbert_norm([[C::new(f64::NAN, 0.); 2]; 2]).is_err());
}

#[test]
fn mixing_retains_energy_old_projection_decays_and_order_survives() {
    let r = [
        [C::new(0.6, 0.), C::new(-0.8, 0.)],
        [C::new(0.8, 0.), C::new(0.6, 0.)],
    ];
    let q = Geometry::Old.matrix(r).unwrap();
    let h = Geometry::Hilbert.matrix(r).unwrap();
    assert!(Geometry::Old.matrix_norm(r).unwrap() > 1.3);
    for (x, y) in q.iter().flatten().zip([0.4, -0.6, 0.6, 0.4]) {
        assert!((x.re - y).abs() < 1e-15);
    }
    let mut x = [C::new(0., 1.), C::default()];
    let mut y = x;
    for n in 1..=12 {
        x = operator(h, x);
        y = operator(q, y);
        assert!((Geometry::Hilbert.state_norm(x) - 1.).abs() < 2e-14);
        assert!((Geometry::Hilbert.state_norm(y).powi(2) - (13f64 / 25.).powi(n)).abs() < 2e-14);
    }
    let d = [
        [C::new(0., 1.), C::default()],
        [C::default(), C::new(0., -1.)],
    ];
    let a = operator(d, operator(r, [C::new(0., 1.), C::default()]));
    let b = operator(r, operator(d, [C::new(0., 1.), C::default()]));
    assert!(a.iter().zip(b).map(|(x, y)| (*x - y).norm()).sum::<f64>() > 1.);
    let broadcast = [[C::new(1., 0.), C::default()]; 2];
    assert_eq!(Geometry::Old.matrix_norm(broadcast).unwrap(), 1.);
    assert!(geometry::hilbert_norm(broadcast).unwrap() > 1.4);
}

fn block(g: Geometry) -> Block {
    let mut b = Block::initial(3, g, 901, 0.75, 1. / 64.).unwrap();
    b.atom.output = vec![
        [C::new(0.2, -0.1), C::new(-0.15, 0.12)],
        [C::new(-0.3, 0.05), C::new(0.1, -0.1)],
        [C::new(0.05, 0.2), C::new(0.15, 0.2)],
    ];
    b.validate().unwrap();
    b
}

#[test]
fn every_real_imaginary_coordinate_is_queried_and_modes_agree_locally() {
    let records = [
        Record {
            prefix: &[0, 1, 0],
            target: 2,
        },
        Record {
            prefix: &[2],
            target: 0,
        },
    ];
    let bases = vec![vec![0.3, -0.2, 0.1], vec![-0.1, 0.2, 0.4]];
    for geometry in [Geometry::Old, Geometry::Hilbert] {
        let b = block(geometry);
        assert_eq!(b.dimensions(), 40);
        for k in 0..b.dimensions() {
            for sign in [-1., 1.] {
                let p = b.perturb(k, sign * 1e-5).unwrap();
                p.validate().unwrap();
                assert_ne!(p.coordinates(), b.coordinates());
            }
        }
        let mini = estimate::batch(&b, &bases, &records, 2., Mode::Mini, 1e-6).unwrap();
        let forward = estimate::batch(&b, &bases, &records, 2., Mode::ForwardMini, 1e-6).unwrap();
        assert_eq!(mini.coordinate_pairs, [8, 24, 24]);
        assert_eq!(mini.logit_queries, 112);
        assert_eq!(mini.ce_evaluations, 2);
        assert_eq!(forward.ce_evaluations, 114);
        assert!(
            mini.values
                .iter()
                .flatten()
                .zip(forward.values.iter().flatten())
                .all(|(a, b)| (a - b).abs() < 2e-6)
        );
        for k in 4 + 16..4 + 24 {
            assert_eq!(mini.values[0][k], 0.);
        }
        assert!(mini.values.iter().flatten().any(|x| x.abs() > 1e-3));
    }
}

#[test]
fn paired_initial_logits_match_and_no_content_signal_is_supplied() {
    let parent = rotary::model::Model::zero(3, 16., Family::Rotary).unwrap();
    let mut old = Model {
        parent: parent.clone(),
        block: Block::initial(3, Geometry::Old, 901, 0.75, 1. / 64.).unwrap(),
        fraction: 0.125,
    };
    old.parent
        .mix(
            rotary::atom::Atom::bias(3, Family::Rotary, 1, 1.).unwrap(),
            0.25,
        )
        .unwrap();
    let hilbert = Model {
        parent: old.parent.clone(),
        block: Block::initial(3, Geometry::Hilbert, 901, 0.75, 1. / 64.).unwrap(),
        fraction: old.fraction,
    };
    let r = Record {
        prefix: &[2, 1, 0, 2],
        target: 0,
    };
    assert_eq!(
        old.logits(r.prefix).unwrap(),
        hilbert.logits(r.prefix).unwrap()
    );
    for model in [old, hilbert] {
        let estimates = estimate::batch(
            &model.block,
            &[model.parent_logits(r.prefix).unwrap()],
            std::slice::from_ref(&r),
            model.mass(),
            Mode::ForwardMini,
            1e-4,
        )
        .unwrap();
        assert!(estimates.values[0][..28].iter().all(|&x| x == 0.));
        assert!(estimates.values[0][28..].iter().any(|x| x.abs() > 0.01));
        let bytes = serde_json::to_vec(&model).unwrap();
        let restored: Model = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            model.logits(r.prefix).unwrap(),
            restored.logits(r.prefix).unwrap()
        );
    }
}

#[test]
fn duplicate_occurrences_tail_refresh_and_wrong_versions_are_detected() {
    let records = [
        Record {
            prefix: &[0],
            target: 1,
        },
        Record {
            prefix: &[0],
            target: 1,
        },
        Record {
            prefix: &[1, 0],
            target: 0,
        },
    ];
    let mut c = Cache::new(&records, 2).unwrap();
    c.observe(&[0, 1], &records[..2], &[vec![1., 2.], vec![1., 2.]], 0)
        .unwrap();
    c.observe(&[2], &records[2..], &[vec![-2., -4.]], 0)
        .unwrap();
    assert!(c.mean.iter().all(|x| x.abs() < 1e-15));
    c.observe(&[0], &records[..1], &[vec![4., 2.]], 1).unwrap();
    assert!((c.mean[0] - 1.).abs() < 1e-15);
    assert!(c.error().unwrap() < 1e-15);
    assert_eq!(c.coverage(), 3);
    assert_eq!(c.refreshes, 4);
    assert!(c.observe(&[0], &records[2..], &[vec![0., 0.]], 2).is_err());
    assert!(c.observe(&[0], &records[..1], &[vec![0., 0.]], 0).is_err());
    assert!(c.observe(&[5], &records[..1], &[vec![0., 0.]], 2).is_err());
    c.mean[1] += 0.01;
    assert!(c.error().unwrap() > 0.009);
}

#[test]
fn every_cached_estimate_replays_from_its_actual_history() {
    let records = [
        Record {
            prefix: &[0, 1],
            target: 2,
        },
        Record {
            prefix: &[1],
            target: 0,
        },
    ];
    let bases = vec![vec![0.1, 0.2, -0.1]; 2];
    for g in [Geometry::Old, Geometry::Hilbert] {
        for mode in [Mode::Mini, Mode::ForwardMini] {
            let initial = block(g);
            let mut b = initial.clone();
            let mut c = Cache::new(&records, b.dimensions()).unwrap();
            let mut h = vec![Entry::new(0, &b).unwrap()];
            let first = estimate::batch(&b, &bases, &records, 2., mode, 1e-4).unwrap();
            c.observe(&[0, 1], &records, &first.values, 0).unwrap();
            b.update(&c.mean, 0.5).unwrap();
            h.push(Entry::new(1, &b).unwrap());
            let second = estimate::batch(&b, &bases[..1], &records[..1], 2., mode, 1e-4).unwrap();
            c.observe(&[0], &records[..1], &second.values, 1).unwrap();
            b.update(&c.mean, 0.5).unwrap();
            h.push(Entry::new(2, &b).unwrap());
            history::validate(&h, &initial, &b).unwrap();
            assert!(c.error().unwrap() < 1e-14);
            for i in 0..2 {
                history::replay_record(&c, &h, i, &records[i], &bases[i], (2., mode, 1e-4))
                    .unwrap();
            }
            c.values[0][1] += 0.001;
            assert!(
                history::replay_record(&c, &h, 0, &records[0], &bases[0], (2., mode, 1e-4))
                    .is_err()
            );
            h[1].block_sha256 = "changed".into();
            assert!(history::validate(&h, &initial, &b).is_err());
        }
    }
}
