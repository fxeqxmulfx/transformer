use crate::{C, ablation};
use bilinear::train::Record;
use hilbert::{block::Block, geometry::Geometry, model::Model};
use rotary::atom::{Atom, Family};
fn example(g: Geometry) -> Model {
    let mut a = Atom::identity_start(3, Family::Plain);
    a.initial = [C::new(0., 1.), C::default()];
    a.transition[1] = [
        [C::new(-1., 0.), C::default()],
        [C::default(), C::new(-1., 0.)],
    ];
    a.output[0] = [C::new(0., -1.), C::default()];
    a.output[1] = [C::new(0., 1.), C::default()];
    Model {
        parent: rotary::model::Model::zero(3, 16., Family::Rotary).unwrap(),
        block: Block {
            geometry: g,
            atom: a,
        },
        fraction: 0.125,
    }
}
fn predicted(model: &Model, prefix: &[i64]) -> usize {
    let logits = model.logits(prefix).unwrap();
    (1..logits.len()).fold(0, |best, c| if logits[c] > logits[best] { c } else { best })
}
#[test]
fn actual_input_dependence_disappears_under_mean_and_zero_ablations() {
    let records = [
        Record {
            prefix: &[0],
            target: 0,
        },
        Record {
            prefix: &[1],
            target: 1,
        },
    ];
    for geometry in [Geometry::Old, Geometry::Hilbert] {
        let m = example(geometry);
        let mean = ablation::training_mean(&m, &records).unwrap();
        assert_eq!(mean.state, [C::default(); 2]);
        assert_eq!(mean.records, 2);
        assert!((mean.mean_squared_euclidean_deviation - 1.).abs() < 1e-15);
        assert_eq!(predicted(&m, &[0]), 0);
        assert_eq!(predicted(&m, &[1]), 1);
        for kind in ["zero-state", "training-mean-state"] {
            let a = ablation::make(&m, &mean, kind).unwrap();
            assert_eq!(predicted(&a, &[0]), 0);
            assert_eq!(predicted(&a, &[1]), 0);
            assert_eq!(a.parent.atoms.len(), m.parent.atoms.len());
            assert_eq!(a.block.atom.output.len(), 3);
        }
        let intact = ablation::make(&m, &mean, "intact").unwrap();
        assert_eq!(
            serde_json::to_vec(&intact).unwrap(),
            serde_json::to_vec(&m).unwrap()
        );
    }
}
#[test]
fn mean_uses_original_prefixes_and_occurrence_counts_but_no_labels() {
    for geometry in [Geometry::Old, Geometry::Hilbert] {
        let m = example(geometry);
        let a = [
            Record {
                prefix: &[0],
                target: 0,
            },
            Record {
                prefix: &[0],
                target: 0,
            },
            Record {
                prefix: &[1],
                target: 1,
            },
        ];
        let b = [
            Record {
                prefix: &[0],
                target: 2,
            },
            Record {
                prefix: &[0],
                target: 1,
            },
            Record {
                prefix: &[1],
                target: 0,
            },
        ];
        let ma = ablation::training_mean(&m, &a).unwrap();
        let mb = ablation::training_mean(&m, &b).unwrap();
        assert_eq!(
            serde_json::to_vec(&ma).unwrap(),
            serde_json::to_vec(&mb).unwrap()
        );
        assert!((ma.state[0] - C::new(0., 1. / 3.)).norm() < 1e-15);
        assert_eq!(ma.state[1], C::default());
        assert!(ablation::training_mean(&m, &[]).is_err());
        let constant = ablation::make(&m, &ma, "training-mean-state").unwrap();
        for prefix in [vec![], vec![0], vec![2, 1, 0, 1]] {
            assert_eq!(constant.block.atom.state(&prefix).unwrap(), ma.state);
        }
    }
}
#[test]
fn constant_bias_is_preserved_and_new_block_isolated_without_deleting_components() {
    let records = [
        Record {
            prefix: &[0],
            target: 0,
        },
        Record {
            prefix: &[1, 2],
            target: 1,
        },
    ];
    for geometry in [Geometry::Old, Geometry::Hilbert] {
        let mut m = example(geometry);
        m.block.atom.transition[1] = m.block.atom.transition[0];
        m.parent
            .mix(Atom::bias(3, Family::Rotary, 2, 1.).unwrap(), 1.)
            .unwrap();
        let mean = ablation::training_mean(&m, &records).unwrap();
        let constant = ablation::make(&m, &mean, "training-mean-state").unwrap();
        for prefix in [vec![0], vec![2, 1], vec![1, 0, 0]] {
            assert_eq!(
                m.logits(&prefix).unwrap(),
                constant.logits(&prefix).unwrap()
            );
        }
        assert_eq!(predicted(&m, &[0]), 2);
        let alone = ablation::make(&m, &mean, "new-block-only").unwrap();
        assert_eq!(predicted(&alone, &[0]), 0);
        assert_eq!(alone.parent.atoms.len(), m.parent.atoms.len());
        assert_eq!(
            serde_json::to_vec(&alone.parent.atoms).unwrap(),
            serde_json::to_vec(&m.parent.atoms).unwrap()
        );
        assert_eq!(alone.parent.weights.iter().sum::<f64>(), 16.);
        assert_eq!(alone.mass(), 2.);
        let mut invalid = mean;
        invalid.state[0] = C::new(2., 0.);
        assert!(ablation::make(&m, &invalid, "training-mean-state").is_err());
    }
}
