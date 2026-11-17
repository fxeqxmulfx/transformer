use crate::{
    C, experiment,
    model::{Atom, Model},
    proposal::Search,
    train::{self, Config},
};
use baseline::{data::Example, random::Random};
use bilinear::{
    experiment as original,
    model::cross_entropy,
    train::{Mode, Record},
};
use serde_json::json;

fn fixture() -> Atom {
    Atom {
        initial: C::new(0.2, 0.7),
        transition: vec![C::new(0.3, -0.8), C::new(-0.6, 0.2), C::new(0.4, 0.5)],
        output: vec![C::new(0.7, 0.3), C::new(-0.1, -0.8), C::new(0.5, -0.6)],
    }
}

#[test]
fn complete_complex_execution_matches_real_matrix_composition() {
    let atom = fixture();
    atom.validate(3).unwrap();
    let prefix = [2, 0, 2, 1];
    let (mut x, mut y) = (atom.initial.re, atom.initial.im);
    for o in prefix {
        let z = atom.transition[o as usize];
        (x, y) = (z.re * x - z.im * y, z.im * x + z.re * y);
    }
    for (z, u) in atom.complex(&prefix).unwrap().iter().zip(&atom.output) {
        assert!((z.re - (u.re * x - u.im * y)).abs() < 1e-15);
        assert!((z.im - (u.im * x + u.re * y)).abs() < 1e-15);
    }
    let mut real = atom.clone();
    for z in std::iter::once(&mut real.initial)
        .chain(&mut real.transition)
        .chain(&mut real.output)
    {
        z.im = 0.;
    }
    assert_ne!(
        real.complex(&prefix).unwrap(),
        atom.complex(&prefix).unwrap()
    );
    assert!(atom.state(&[-1]).is_err());
    assert!(atom.state(&[3]).is_err());
}

#[test]
fn capacity_queries_are_actual_feasible_models_at_all_lengths() {
    let mut model = Model::zero(3, 16.).unwrap();
    model.mix(fixture(), 0.8).unwrap();
    let h = 1e-4;
    model.shrink_for_queries(h).unwrap();
    for prefix in [vec![], vec![0, 1, 2], vec![2; 1000]] {
        let base = model.logits(&prefix).unwrap();
        assert!(base.iter().all(|v| v.abs() <= 16.));
        for class in 0..3 {
            for sign in [-1., 1.] {
                let mut query = model.clone();
                query.weights[0] -= h;
                query.atoms.push(Atom::bias(3, class, sign).unwrap());
                query.weights.push(h);
                query.validate().unwrap();
                let actual = query.logits(&prefix).unwrap();
                for c in 0..3 {
                    assert!(
                        (actual[c] - base[c] - if c == class { sign * h } else { 0. }).abs()
                            < 1e-14
                    );
                }
            }
        }
    }
    let mut invalid = fixture();
    invalid.transition[0] = C::new(1., 1.);
    assert!(model.mix(invalid, 0.1).is_err());
}

#[test]
fn opaque_object_renaming_and_scalar_order_limitation() {
    let atom = fixture();
    let rename = [2, 0, 1];
    let mut changed = atom.clone();
    for (i, &renamed) in rename.iter().enumerate() {
        changed.transition[renamed] = atom.transition[i];
        changed.output[renamed] = atom.output[i];
    }
    let prefix = [0, 2, 0, 1];
    let renamed: Vec<_> = prefix.iter().map(|&o| rename[o as usize] as i64).collect();
    let x = atom.complex(&prefix).unwrap();
    let y = changed.complex(&renamed).unwrap();
    for c in 0..3 {
        assert_eq!(x[c], y[rename[c]]);
    }
    let a = atom.complex(&[0, 1]).unwrap();
    let b = atom.complex(&[1, 0]).unwrap();
    for (a, b) in a.iter().zip(b) {
        assert!((*a - b).norm() < 1e-15);
    }
}

#[test]
fn complete_finite_ce_signals_include_every_original_competitor() {
    let logits = [0.1, -0.7, 1.8, -2.1, 0.3];
    let analytic = train::scalars(&logits, 3, Mode::Mini, 1e-4).unwrap();
    let forward = train::scalars(&logits, 3, Mode::ForwardMini, 1e-4).unwrap();
    assert_eq!(forward.len(), logits.len());
    for (&a, &f) in analytic.iter().zip(&forward) {
        assert!((a - f).abs() < 3e-9);
        assert_ne!(f, 0.);
    }
    assert!(forward[3] < 0.);
    assert!(train::scalars(&logits, 5, Mode::Mini, 1e-4).is_err());
}

#[test]
fn absent_transition_skip_matches_complete_coordinate_search() {
    let records = [
        Record {
            prefix: &[0, 2, 0],
            target: 1,
        },
        Record {
            prefix: &[2, 2],
            target: 0,
        },
    ];
    let base = vec![vec![0.; 4]; 2];
    let signals: Vec<_> = records
        .iter()
        .map(|r| train::scalars(&base[0], r.target, Mode::ForwardMini, 1e-4).unwrap())
        .collect();
    let mut sparse = Search {
        base: &base,
        signals: &signals,
        records: &records,
        budget: 16.,
        loss_queries: 0,
    };
    let mut dense = Search {
        base: &base,
        signals: &signals,
        records: &records,
        budget: 16.,
        loss_queries: 0,
    };
    let mut initial = Atom::identity_start(4);
    initial.transition[3] = C::new(1e-100, -1e-100);
    let (a, la) = sparse.improve(initial.clone(), 3, true).unwrap();
    let (b, lb) = dense.improve(initial.clone(), 3, false).unwrap();
    assert_eq!(
        serde_json::to_value(&a).unwrap(),
        serde_json::to_value(&b).unwrap()
    );
    assert_eq!(la, lb);
    assert_eq!(a.transition[3], initial.transition[3]);
    assert!(sparse.loss_queries < dense.loss_queries);
}

#[test]
fn duplicate_records_and_short_tails_keep_original_masks() {
    let e = Example {
        task: "opaque".into(),
        tokens: vec![0, 2, 1],
        targets: vec![-100, 1, 2],
        changes: vec![false, true, true],
        prompt: vec![],
        answer: vec![],
        limit: 0,
        field: None,
        row_index: None,
    };
    let examples = vec![e];
    let duplicate = original::records(&examples, &[0, 0]).unwrap();
    let tail = original::records(&examples, &[0]).unwrap();
    assert_eq!(duplicate.len(), 4);
    assert_eq!(tail.len(), 2);
    for (a, b) in duplicate[..2].iter().zip(&duplicate[2..]) {
        assert_eq!(a.prefix, b.prefix);
        assert_eq!(a.target, b.target);
    }
    assert_eq!(tail[0].prefix, &[0, 2]);
    assert_eq!(tail[1].prefix, &[0, 2, 1]);
    assert_eq!(tail[0].target, 1);
    assert_eq!(tail[1].target, 2);
}

#[test]
fn both_modes_learn_transitions_without_a_supplied_rule_and_replay() {
    let prefixes: Vec<Vec<i64>> = (0..8)
        .map(|n| {
            (0..3)
                .map(|i| if n & (1 << i) == 0 { 2 } else { 4 })
                .collect()
        })
        .collect();
    // A test-only target oracle; the learner receives only opaque prefixes/labels.
    let records: Vec<_> = prefixes
        .iter()
        .enumerate()
        .map(|(n, p)| Record {
            prefix: p,
            target: if n.count_ones() % 2 == 0 { 3 } else { 5 },
        })
        .collect();
    for mode in [Mode::Mini, Mode::ForwardMini] {
        let mut model = Model::zero(7, 16.).unwrap();
        let mut rng = Random::new(31);
        let config = Config {
            mode,
            query_step: 1e-4,
            sweeps: 6,
            line_iterations: 64,
        };
        for _ in 0..6 {
            let receipt = train::step(&mut model, &records, config, &mut rng).unwrap();
            let actual: f64 = records
                .iter()
                .map(|r| cross_entropy(&model.logits(r.prefix).unwrap(), r.target).unwrap())
                .sum::<f64>()
                / 8.;
            assert!((actual - receipt.after_ce).abs() < 1e-12);
            assert!(receipt.after_ce <= receipt.centre_ce + 1e-12);
            assert_eq!(
                receipt.bias_loss_queries,
                if mode == Mode::ForwardMini { 112 } else { 0 }
            );
        }
        let loss = records
            .iter()
            .map(|r| cross_entropy(&model.logits(r.prefix).unwrap(), r.target).unwrap())
            .sum::<f64>()
            / 8.;
        assert!(loss < 0.5, "mode={mode:?} loss={loss}");
        assert!(
            model.atoms[1..]
                .iter()
                .any(|a| a.transition[2] != a.transition[4])
        );
        assert_eq!(model.atoms.len(), 7);
        let copied: Model = serde_json::from_slice(&serde_json::to_vec(&model).unwrap()).unwrap();
        for r in &records {
            assert_eq!(
                copied.logits(r.prefix).unwrap(),
                model.logits(r.prefix).unwrap()
            );
        }
    }
}

#[test]
fn sealing_rejects_incomplete_and_changed_producers() {
    let path = std::env::temp_dir().join(format!("recurrent-seal-{}", std::process::id()));
    std::fs::create_dir(&path).unwrap();
    assert!(experiment::score::sealed(&path).is_err());
    let protocol: serde_json::Value = serde_json::from_str(experiment::PROTOCOL).unwrap();
    let binary = original::hash(&std::fs::read(std::env::current_exe().unwrap()).unwrap());
    for case in protocol["cases"].as_array().unwrap() {
        for mode in ["mini", "forward"] {
            let cell = original::score::cell(&path, case["name"].as_str().unwrap(), mode, 1);
            std::fs::create_dir_all(&cell).unwrap();
            original::write_json(&cell.join("metadata.json"),&json!({"source_fingerprint":experiment::fingerprint(),"protocol":protocol,"case":case,"binary_sha256":binary,"mode":mode,"seed":1})).unwrap();
            original::write_json(&cell.join("training_receipt.json"),&json!({"status":"sealed","heldout_scored":false,"source_fingerprint":experiment::fingerprint(),"updates":case["maximum_updates"],"case":case["name"],"mode":mode,"seed":1,"endpoint_sha256":original::hash(b"{}\n")})).unwrap();
            original::write_json(&cell.join("endpoint.json"), &json!({})).unwrap();
        }
    }
    experiment::score::sealed(&path).unwrap();
    let cell = original::score::cell(&path, "basis-parity16", "forward", 1);
    std::fs::write(cell.join("metadata.json"), b"{}").unwrap();
    assert!(experiment::score::sealed(&path).is_err());
    std::fs::remove_dir_all(path).unwrap();
}
