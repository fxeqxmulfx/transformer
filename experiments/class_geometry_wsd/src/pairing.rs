//! Compare exact coverage counts, not their redundantly serialized ratios.
use serde_json::Value;
pub fn coverage_counts_equal(a: &Value, b: &Value) -> bool {
    ["train", "validation"].into_iter().all(|split| {
        match (a[split].as_array(), b[split].as_array()) {
            (Some(a), Some(b)) if a.len() == 8 && b.len() == 8 => a.iter().zip(b).all(|(x, y)| {
                ["family", "rank", "count", "correct", "certified"]
                    .into_iter()
                    .all(|key| x[key] == y[key] && !x[key].is_null())
            }),
            _ => false,
        }
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn ratio_roundtrip_does_not_change_counts_but_wrong_count_fails() {
        let rows:Vec<_>=(0..8).map(|rank|json!({"family":"readout","rank":rank,"count":10,"correct":1,"certified":1,"coverage":0.1})).collect();
        let a = json!({"train":rows,"validation":rows});
        let mut b = a.clone();
        b["validation"][0]["coverage"] = json!(f64::from_bits(0.1f64.to_bits() + 1));
        assert_ne!(a, b);
        assert!(coverage_counts_equal(&a, &b));
        b["validation"][0]["certified"] = json!(2);
        assert!(!coverage_counts_equal(&a, &b));
    }
}
