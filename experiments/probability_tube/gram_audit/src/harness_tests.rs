//! Inventory and numerical-interface checks without trained-state access.
use crate::{domain::data::Plan, observe, sources};
use serde_json::json;
#[test]
fn exact_inventory_preserves_initialization_terminal_states_and_all_horizons() {
    let cases = sources::inventory();
    assert_eq!(cases.len(), 144);
    let mut endpoints = 0;
    let mut terminal = 0;
    let mut moving_positive = 0;
    for case in cases {
        let anchor = case["anchor"].as_u64().unwrap();
        for h in case["horizons"].as_array().unwrap() {
            let h = h.as_u64().unwrap();
            endpoints += 1;
            terminal += usize::from(anchor + h == 10000);
            moving_positive += usize::from(anchor > 0 && h > 0);
        }
    }
    assert_eq!((endpoints, terminal, moving_positive), (1800, 48, 1368));
}
#[test]
fn recovered_source_plan_has_complete_epochs_and_exact_tail_batches() {
    let p = Plan::new(11, 30);
    for epoch in 0..3 {
        let mut seen = std::collections::BTreeSet::new();
        for j in 0..10 {
            let rows =
                probability_gram::batch_indices(&p.order, &p.boundaries, epoch * 10 + j).unwrap();
            assert_eq!(rows.len(), if j == 9 { 96 } else { 512 });
            for i in rows {
                assert!(seen.insert(*i));
            }
        }
        assert_eq!(
            seen.into_iter().collect::<Vec<_>>(),
            (0..4704).collect::<Vec<_>>()
        );
    }
}
#[test]
fn execution_conversion_changes_probability_at_zero_but_not_feature_radius() {
    let q = 1. / 97.;
    for h in [0, 1] {
        let source =
            json!({"horizon":h,"probability_ceilings":vec![q;4704],"feature_radii":vec![2.;4704]});
        let (ceilings, radii) = observe::converted(&source).unwrap();
        let expected = q * 0.002f64.exp() / (1. - q + q * 0.002f64.exp());
        assert!(
            ceilings
                .iter()
                .all(|v| (*v - expected).abs() < 1e-16 && *v > q)
        );
        assert!(radii.iter().all(|v| if h == 0 {
            *v == 0.
        } else {
            *v >= 2.002 && *v < 2.002 + 1e-14
        }));
    }
    let invalid =
        json!({"horizon":0,"probability_ceilings":vec![0.;4704],"feature_radii":vec![0.;4704]});
    assert!(observe::converted(&invalid).is_err());
}
