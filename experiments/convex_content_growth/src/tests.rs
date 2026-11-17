use crate::{C, diagnostic, estimate, experiment, model::Model, query::Centre};
use bilinear::{
    model::cross_entropy,
    train::{Mode, Record},
};
use hilbert::{
    block::Block,
    cache::Cache,
    geometry::Geometry,
    history::{self, Entry},
};
use rotary::atom::Family;
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
fn source(g: Geometry) -> hilbert::model::Model {
    let mut parent = rotary::model::Model::zero(3, 16., Family::Rotary).unwrap();
    parent
        .mix(
            rotary::atom::Atom::bias(3, Family::Rotary, 1, 1.).unwrap(),
            0.25,
        )
        .unwrap();
    hilbert::model::Model {
        parent,
        block: block(g),
        fraction: 0.125,
    }
}
#[test]
fn finite_one_class_ce_matches_full_loss_with_asymmetry_and_max_crossing() {
    for logits in [vec![0., 0., 0.], vec![12., -9., 4.], vec![-10., -11., -12.]] {
        for target in 0..3 {
            let centre = Centre::new(&logits, target).unwrap();
            for c in 0..3 {
                for (dp, dm) in [(0.001, -0.0002), (5., -2.), (-0.1, 0.3)] {
                    let mut lp = logits.clone();
                    lp[c] += dp;
                    let mut lm = logits.clone();
                    lm[c] += dm;
                    assert!(
                        (centre.loss + centre.increment(c, dp).unwrap()
                            - cross_entropy(&lp, target).unwrap())
                        .abs()
                            < 2e-13
                    );
                    let full =
                        cross_entropy(&lp, target).unwrap() - cross_entropy(&lm, target).unwrap();
                    assert!((centre.difference(c, dp, dm).unwrap() - full).abs() < 3e-13);
                }
            }
        }
    }
    let c = Centre::new(&[1., -1.], 0).unwrap();
    let finite = c.difference(1, 0.3, -0.3).unwrap() / 0.6;
    assert!((finite - c.shares[1]).abs() > 1e-4);
    assert!(c.increment(2, 0.1).is_err());
    assert!(c.increment(0, f64::NAN).is_err());
}
#[test]
fn all_output_pairs_equal_complete_projected_queries() {
    for geometry in [Geometry::Old, Geometry::Hilbert] {
        let mut b = block(geometry);
        b.atom.output[1] = geometry.output([C::new(2., 3.), C::new(-4., 1.)]).unwrap();
        for k in 4 + 8 * b.objects()..b.dimensions() {
            let (c, p, m) = estimate::output_pair(&b, k, 1e-4).unwrap();
            assert_eq!(p, b.perturb(k, 1e-4).unwrap().atom.output[c]);
            assert_eq!(m, b.perturb(k, -1e-4).unwrap().atom.output[c]);
        }
    }
}
#[test]
fn complete_estimates_match_reference_and_single_record_replay_bits() {
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
    for g in [Geometry::Old, Geometry::Hilbert] {
        for mode in [Mode::Mini, Mode::ForwardMini] {
            let b = block(g);
            let fast = estimate::batch(&b, &bases, &records, 2., mode, 1e-4, true).unwrap();
            let full = hilbert::estimate::batch(&b, &bases, &records, 2., mode, 1e-4).unwrap();
            assert_eq!(fast.coordinate_pairs, [8, 24, 24]);
            assert_eq!(fast.coordinate_pairs, full.coordinate_pairs);
            assert_eq!(fast.finite_logit_queries, full.logit_queries);
            assert!(fast.audit_comparisons > 0);
            for (a, b) in fast
                .values
                .iter()
                .flatten()
                .zip(full.values.iter().flatten())
            {
                assert!((a - b).abs() < 2e-8, "{a} vs {b}");
            }
            for i in 0..2 {
                let single = estimate::batch(
                    &b,
                    &bases[i..i + 1],
                    &records[i..i + 1],
                    2.,
                    mode,
                    1e-4,
                    false,
                )
                .unwrap();
                assert!(
                    single.values[0]
                        .iter()
                        .zip(&fast.values[i])
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                );
            }
            for k in 20..28 {
                assert_eq!(fast.values[0][k], 0.);
            }
        }
    }
}
#[test]
fn complete_convex_append_preserves_old_components_and_initial_logits() {
    for g in [Geometry::Old, Geometry::Hilbert] {
        let old = source(g);
        let mut m = Model::from_source(old.clone()).unwrap();
        let prefix = &[0, 1, 2, 0];
        assert_eq!(old.logits(prefix).unwrap(), m.logits(prefix).unwrap());
        let prior = m.logits(prefix).unwrap();
        let old_bytes = serde_json::to_vec(&m.parent).unwrap();
        let old_components = m.components();
        let b = block(g);
        let state = b.atom.state(prefix).unwrap();
        let output: Vec<_> = b
            .atom
            .output
            .iter()
            .map(|&u| rotary::atom::row(u, state).re)
            .collect();
        let active = m.append(b, 0.125).unwrap();
        assert_eq!(m.used_mass(), 16.);
        assert_eq!(m.components(), old_components + 1);
        assert_eq!(serde_json::to_vec(&m.parent).unwrap(), old_bytes);
        for ((a, b), c) in m.logits(prefix).unwrap().iter().zip(prior).zip(output) {
            assert!((a - (0.875 * b + 2. * c)).abs() < 2e-14);
        }
        let hash = m.base_hash(active).unwrap();
        m.blocks[active].block = m.blocks[active].block.perturb(0, 0.01).unwrap();
        assert_eq!(m.base_hash(active).unwrap(), hash);
        m.blocks[0].block = m.blocks[0].block.perturb(0, 0.01).unwrap();
        assert_ne!(m.base_hash(active).unwrap(), hash);
    }
}
#[test]
fn both_arms_retain_real_multistage_histories_with_duplicate_occurrences() {
    let records = [
        Record {
            prefix: &[0, 1],
            target: 2,
        },
        Record {
            prefix: &[0, 1],
            target: 2,
        },
        Record {
            prefix: &[2],
            target: 0,
        },
    ];
    let p = experiment::protocol().unwrap();
    for arm in ["continue", "grow"] {
        for geometry in [Geometry::Old, Geometry::Hilbert] {
            for mode in [Mode::Mini, Mode::ForwardMini] {
                let g = if geometry == Geometry::Old {
                    "old"
                } else {
                    "hilbert"
                };
                let m = if mode == Mode::Mini {
                    "mini"
                } else {
                    "forward"
                };
                let cell = experiment::Cell::new(arm, g, "cautious", m).unwrap();
                let mut model = Model::from_source(source(geometry)).unwrap();
                for stage in 1..=2 {
                    let active = experiment::prepare(&mut model, &cell, stage, &p).unwrap();
                    let initial = model.blocks[active].block.clone();
                    let mass = model.blocks[active].weight;
                    let hash = model.base_hash(active).unwrap();
                    let bases: Vec<_> = records
                        .iter()
                        .map(|r| model.base_logits(Some(active), r.prefix).unwrap())
                        .collect();
                    let mut cache = Cache::new(&records, initial.dimensions()).unwrap();
                    let mut history = vec![Entry::new(0, &initial).unwrap()];
                    for (version, (start, end)) in [(0, 3), (0, 1), (1, 3)].into_iter().enumerate()
                    {
                        let ids: Vec<_> = (start..end).collect();
                        let e = estimate::batch(
                            &model.blocks[active].block,
                            &bases[start..end],
                            &records[start..end],
                            mass,
                            mode,
                            1e-4,
                            true,
                        )
                        .unwrap();
                        cache
                            .observe(&ids, &records[start..end], &e.values, version)
                            .unwrap();
                        model.blocks[active].block.update(&cache.mean, 1.).unwrap();
                        history.push(Entry::new(version + 1, &model.blocks[active].block).unwrap());
                    }
                    history::validate(&history, &initial, &model.blocks[active].block).unwrap();
                    assert_eq!(model.base_hash(active).unwrap(), hash);
                    assert!(cache.error().unwrap() < 1e-14);
                    assert_eq!(cache.coverage(), 3);
                    assert_eq!(cache.refreshes, 6);
                    for i in 0..3 {
                        diagnostic::replay_record(
                            &cache,
                            &history,
                            i,
                            &records[i],
                            &bases[i],
                            (mass, mode, 1e-4),
                        )
                        .unwrap();
                    }
                    cache.values[0][0] += 0.001;
                    assert!(
                        diagnostic::replay_record(
                            &cache,
                            &history,
                            0,
                            &records[0],
                            &bases[0],
                            (mass, mode, 1e-4)
                        )
                        .is_err()
                    );
                }
                assert_eq!(model.blocks.len(), if arm == "grow" { 3 } else { 1 });
            }
        }
    }
}
