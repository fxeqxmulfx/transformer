//! Predeclared observed thresholds: accuracy alone and joint accuracy/CE.
use serde_json::{Value, json};
fn point(row: &Value) -> Value {
    json!({"step":row["step"],"active_seconds":row["active_seconds"],"validation":row["validation"]})
}
pub fn summarize(reports: &mut [Value]) {
    let baseline = reports
        .iter()
        .find(|r| {
            r["method"] == "ce-bp"
                && r["status"] == "complete"
                && r["history"]
                    .as_array()
                    .is_some_and(|h| h.iter().all(|r| r["source_replay_match"] == true))
        })
        .and_then(|r| r["history"].as_array()?.last())
        .cloned();
    for report in reports {
        let Some(history) = report["history"].as_array() else {
            continue;
        };
        let first = |threshold: f64| {
            history
                .iter()
                .find(|r| r["validation"]["accuracy"].as_f64().unwrap() >= threshold)
                .map(point)
        };
        let best = history
            .iter()
            .max_by_key(|r| r["validation"]["correct"].as_u64().unwrap())
            .map(point);
        let minimum = history
            .iter()
            .min_by(|a, b| {
                a["validation"]["loss"]
                    .as_f64()
                    .unwrap()
                    .total_cmp(&b["validation"]["loss"].as_f64().unwrap())
            })
            .map(point);
        let matched = baseline
            .as_ref()
            .and_then(|base| {
                history.iter().find(|r| {
                    r["validation"]["correct"].as_u64().unwrap()
                        >= base["validation"]["correct"].as_u64().unwrap()
                        && r["validation"]["loss"].as_f64().unwrap()
                            <= base["validation"]["loss"].as_f64().unwrap()
                })
            })
            .map(point);
        let summary = json!({"first99":first(0.99),"first995":first(0.995),"best_accuracy":best,
            "minimum_ce":minimum,"endpoint":history.last().map(point),
            "matched_ce_endpoint":matched,"paired_ce_endpoint":baseline.as_ref().map(point)});
        report["summary"] = summary;
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn joint_target_requires_both_metrics_and_a_replayed_complete_control() {
        let row = |step, accuracy, correct, loss| {
            json!({"step":step,"active_seconds":1.,"source_replay_match":true,
            "validation":{"accuracy":accuracy,"correct":correct,"loss":loss}})
        };
        let mut r = vec![
            json!({"method":"ce-bp","status":"complete","history":[row(4000,0.99,99,0.01)]}),
            json!({"method":"local-pair","history":[row(100,1.,100,0.1),row(200,0.98,98,0.001),row(300,0.99,99,0.009)]}),
        ];
        summarize(&mut r);
        assert_eq!(r[1]["summary"]["matched_ce_endpoint"]["step"], 300);
        r[0]["status"] = json!("capped");
        summarize(&mut r);
        assert!(r[1]["summary"]["matched_ce_endpoint"].is_null());
    }
}
