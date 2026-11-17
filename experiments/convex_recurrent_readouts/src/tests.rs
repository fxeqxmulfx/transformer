use crate::{
    C, experiment, score, source,
    update::{self, Config},
};
use bilinear::{
    experiment as original,
    model::cross_entropy,
    train::{Mode, Record},
};
use rotary::{
    atom::{Atom, Family},
    model::Model,
};
use serde_json::json;

fn config(mode: Mode) -> Config {
    Config {
        mode,
        query_step: 1e-4,
        learning_rate: 4.,
        line_iterations: 64,
        projection: crate::projection::Rule::Radial,
    }
}
fn model() -> Model {
    // A nonconstant order witness is used only in this execution control.
    let mut atom = Atom::identity_start(5, Family::Rotary);
    atom.initial = [C::new(0., 1.); 2];
    atom.transition[0][1][1] = C::new(-1., 0.);
    atom.transition[1] = [
        [C::default(), C::new(1., 0.)],
        [C::new(1., 0.), C::default()],
    ];
    let mut model = Model::zero(5, 16., Family::Rotary).unwrap();
    model.mix(atom, 0.7).unwrap();
    model
}

#[test]
fn both_modes_learn_complete_imaginary_readouts_without_changing_learned_content() {
    let records = [
        Record {
            prefix: &[0, 1],
            target: 3,
        },
        Record {
            prefix: &[1, 0],
            target: 4,
        },
    ];
    let initial = model();
    let content = source::content_hash(&initial).unwrap();
    let mut models = vec![];
    for mode in [Mode::Mini, Mode::ForwardMini] {
        let mut model = initial.clone();
        for _ in 0..4 {
            let receipt = update::step(&mut model, &records, config(mode)).unwrap();
            assert!(receipt.after_ce <= receipt.centre_ce + 1e-12);
            assert_eq!(receipt.row_coordinates_processed, 40);
            assert_eq!(
                receipt.total_ce_evaluations,
                2 * 71 + if mode == Mode::ForwardMini { 20 } else { 0 }
            );
            assert!(receipt.maximum_actual_logit_error < 1e-12);
            assert_eq!(source::content_hash(&model).unwrap(), content);
        }
        for r in &records {
            assert_eq!(model.next_object(r.prefix).unwrap(), r.target);
            assert!(cross_entropy(&model.logits(r.prefix).unwrap(), r.target).unwrap() < 0.01);
        }
        assert!(model.atoms[1].output.iter().flatten().any(|z| z.im != 0.));
        let copied: Model = serde_json::from_slice(&serde_json::to_vec(&model).unwrap()).unwrap();
        for r in &records {
            assert_eq!(
                copied.logits(r.prefix).unwrap(),
                model.logits(r.prefix).unwrap()
            );
        }
        models.push(model);
    }
    for r in &records {
        for (a, b) in models[0]
            .logits(r.prefix)
            .unwrap()
            .iter()
            .zip(models[1].logits(r.prefix).unwrap())
        {
            assert!((a - b).abs() < 1e-6);
        }
    }
}

#[test]
fn every_class_and_both_complex_coefficients_enter_the_gradient_and_logits() {
    let mut atom = Atom::identity_start(5, Family::Rotary);
    atom.frequency = Some([0.31, -0.72]);
    atom.initial = [C::new(0.3, 0.4), C::new(-0.2, 0.1)];
    let mut model = Model::zero(5, 16., Family::Rotary).unwrap();
    model.mix(atom, 0.6).unwrap();
    let content = source::content_hash(&model).unwrap();
    let records = [
        Record {
            prefix: &[0, 1, 2],
            target: 3,
        },
        Record {
            prefix: &[0, 1, 2],
            target: 3,
        },
        Record {
            prefix: &[2],
            target: 3,
        },
    ];
    let receipt = update::step(&mut model, &records, config(Mode::Mini)).unwrap();
    assert_eq!(receipt.records, 3);
    assert_eq!(source::content_hash(&model).unwrap(), content);
    for row in &model.atoms[1].output {
        assert!(row.iter().all(|z| z.re != 0. && z.im != 0.));
    }
    for r in &records {
        let states: Vec<_> = model
            .atoms
            .iter()
            .map(|a| a.state(r.prefix).unwrap())
            .collect();
        assert_eq!(
            update::logits(&model, &states).unwrap(),
            model.logits(r.prefix).unwrap()
        );
    }
    assert!(update::logits(&model, &[]).is_err());
    assert!(update::step(&mut model, &[], config(Mode::Mini)).is_err());
    assert!(
        update::step(
            &mut model,
            &[Record {
                prefix: &[-1],
                target: 3
            }],
            config(Mode::Mini)
        )
        .is_err()
    );
}

#[test]
fn radial_rows_and_actual_signed_queries_preserve_complete_capacity() {
    let small = [C::new(0.1, 0.2), C::new(-0.2, 0.1)];
    assert_eq!(update::project_row(small).unwrap(), small);
    let large = [C::new(3., 4.), C::new(-8., 6.)];
    let projected = update::project_row(large).unwrap();
    assert_eq!(projected, [large[0] / 15., large[1] / 15.]);
    assert!((projected[0].norm() + projected[1].norm() - 1.).abs() < 1e-15);
    assert!(update::project_row([C::new(f64::INFINITY, 0.), C::default()]).is_err());
    let mut model = model();
    model.shrink_for_queries(1e-4).unwrap();
    let prefix = [0, 1];
    let centre = model.logits(&prefix).unwrap();
    for c in 0..model.objects {
        for sign in [-1., 1.] {
            let mut point = model.clone();
            point.weights[0] -= 1e-4;
            point.weights.push(1e-4);
            point
                .atoms
                .push(Atom::bias(5, Family::Rotary, c, sign).unwrap());
            point.validate().unwrap();
            for (j, (&a, b)) in centre
                .iter()
                .zip(point.logits(&prefix).unwrap())
                .enumerate()
            {
                assert!((b - a - if j == c { 1e-4 * sign } else { 0. }).abs() < 1e-12);
            }
        }
    }
}

#[test]
fn no_scoring_until_all_four_endpoints_are_sealed_and_unchanged() {
    let root = std::env::temp_dir().join(format!("readout-seal-{}", std::process::id()));
    std::fs::create_dir(&root).unwrap();
    let binary = original::hash(&std::fs::read(std::env::current_exe().unwrap()).unwrap());
    for variant in experiment::VARIANTS {
        for mode in experiment::MODES {
            assert!(score::sealed(&root).is_err());
            let (protocol, _) = experiment::settings(variant, mode).unwrap();
            let path = experiment::cell(&root, variant, mode);
            std::fs::create_dir_all(&path).unwrap();
            original::write_json(&path.join("endpoint.json"), &json!({})).unwrap();
            original::write_json(&path.join("metadata.json"),&json!({"source_fingerprint":experiment::fingerprint(),
                "protocol":protocol,"binary_sha256":binary,"variant":variant,"mode":mode,"input":{"content_sha256":"unchanged"}})).unwrap();
            original::write_json(&path.join("training_receipt.json"),&json!({"source_fingerprint":experiment::fingerprint(),
                "variant":variant,"mode":mode,"updates":512,"status":"sealed","heldout_scored":false,
                "content_sha256":"unchanged","endpoint_sha256":original::hash(b"{}\n")})).unwrap();
        }
    }
    score::sealed(&root).unwrap();
    let path = experiment::cell(&root, "refined", "forward");
    std::fs::write(path.join("endpoint.json"), b"{\"altered\":true}").unwrap();
    assert!(score::sealed(&root).is_err());
    std::fs::remove_dir_all(root).unwrap();
}
