use super::{measure, trace};
use crate::{
    C,
    atom::{Atom, Family},
    model::Model,
};
use bilinear::train::Record;

#[test]
fn constant_and_contracting_states_are_distinguished_without_label_filtering() {
    let records = [
        Record {
            prefix: &[0, 1],
            target: 0,
        },
        Record {
            prefix: &[1, 0],
            target: 1,
        },
    ];
    let mut a = Atom::identity_start(2, Family::Rotary);
    a.frequency = Some([0.7, -0.2]);
    let mut model = Model::zero(2, 16., Family::Rotary).unwrap();
    model.mix(a.clone(), 1.).unwrap();
    let report = measure::run(&model, &records, || Ok(())).unwrap();
    assert_eq!(report["summary"]["mixed_label_feature_groups"], 1);
    assert_eq!(
        report["summary"]["records_in_mixed_label_feature_groups"],
        2
    );
    assert_eq!(
        report["summary"]["exactly_constant_components_within_every_length"],
        2
    );
    for m in &mut a.transition {
        for row in m {
            for z in row {
                *z *= 0.5;
            }
        }
    }
    let states = trace::states(&a, &[0, 1]).unwrap();
    for (i, &state) in states.iter().enumerate() {
        assert!((trace::norm(state) - 0.5f64.powi(i as i32)).abs() < 1e-15);
    }
    let v = trace::variation(&[*states.last().unwrap(); 3]);
    assert_eq!(v["exactly_constant"], true);
    assert_eq!(trace::bits(-0.), trace::bits(0.));
    assert_ne!(trace::bits(1e-100), trace::bits(0.));
}

#[test]
fn order_and_imaginary_information_survive_complete_state_inspection() {
    let mut atom = Atom::identity_start(2, Family::Plain);
    atom.initial = [C::new(0., 1.); 2];
    atom.transition[0][1][1] = C::new(-1., 0.);
    atom.transition[1] = [
        [C::default(), C::new(1., 0.)],
        [C::new(1., 0.), C::default()],
    ];
    let records = [
        Record {
            prefix: &[0, 1],
            target: 0,
        },
        Record {
            prefix: &[1, 0],
            target: 1,
        },
    ];
    let mut model = Model::zero(2, 16., Family::Plain).unwrap();
    model.mix(atom, 1.).unwrap();
    let report = measure::run(&model, &records, || Ok(())).unwrap();
    assert_eq!(report["summary"]["mixed_label_feature_groups"], 0);
    assert_eq!(report["summary"]["complete_feature_groups"], 2);
    assert_eq!(report["summary"]["maximum_independent_real_error"], 0.);
    assert!(
        report["components"][1]["within_length_variance"]
            .as_f64()
            .unwrap()
            > 1.
    );
    assert_eq!(report["summary"]["state_coordinate_values_compared"], 16);
    assert!(measure::run(&model, &[], || Ok(())).is_err());
}
