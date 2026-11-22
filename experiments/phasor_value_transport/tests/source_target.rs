#[path = "../src/stress/source_target.rs"]
mod source_target;

use serde_json::json;
use source_target::SourceTarget;

#[test]
fn declared_accuracy_and_retention_use_the_source_observations() {
    let rule = SourceTarget::from_case(&json!({
        "metric":"sequence_accuracy","target":0.95,"after_target":500
    }))
    .unwrap();
    assert!(rule.met(19, 20, None).unwrap());
    assert!(!rule.met(18, 20, None).unwrap());
    assert!(!rule.met(0, 0, None).unwrap());
    assert_eq!(rule.stop_after_crossing(0, 64, 30_000), 512);
    assert_eq!(rule.stop_after_crossing(320, 64, 30_000), 832);
    assert_eq!(rule.stop_after_crossing(29_952, 64, 30_000), 30_000);
    let immediate = SourceTarget::from_case(&json!({
        "metric":"sequence_accuracy","target":0.95,"after_target":0
    }))
    .unwrap();
    assert_eq!(immediate.stop_after_crossing(320, 64, 30_000), 320);
    assert_eq!(immediate.stop_after_crossing(30_000, 64, 30_000), 30_000);
}

#[test]
fn numeric_crossings_measure_error_on_generated_values() {
    let rule = SourceTarget::from_case(&json!({
        "metric":"negative_mse","target":-0.001,"after_target":500
    }))
    .unwrap();
    assert!(rule.met(0, 100, Some(0.001)).unwrap());
    assert!(!rule.met(100, 100, Some(0.0011)).unwrap());
    assert!(rule.met(100, 100, None).is_err());
    assert!(rule.met(100, 100, Some(-1.)).is_err());
}

#[test]
fn a_different_metric_or_missing_retention_is_rejected() {
    for case in [
        json!({"metric":"numeric_mse","target":0.001,"after_target":500}),
        json!({"metric":"sequence_accuracy","target":1.1,"after_target":500}),
        json!({"metric":"sequence_accuracy","target":0.99}),
    ] {
        assert!(SourceTarget::from_case(&case).is_err());
    }
}
