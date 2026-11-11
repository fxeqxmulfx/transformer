//! Validate all 18 fixed endpoints and report mean/min/max, including failures.
use convex_channels::{channel::Channel, provenance::source_hash, task};
use convex_recall_compare::{convex::ce_gradient, data};
use serde_json::{Value, json};
use std::{fs, path::Path};

fn read(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display())))
        .unwrap()
}

fn verify_saved_channel(saved: &Value, endpoint: &Value, expected: &task::Task) {
    assert_eq!(saved["arrays"].as_array().unwrap().len(), 1);
    let mut channel = Channel::new(task::KEYS, task::KEYS, task::VALUES, task::CLASSES);
    channel.weights =
        saved["arrays"][0].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
    assert_eq!(channel.weights.len(), task::KEYS * task::KEYS * task::VALUES * task::CLASSES);
    assert!(channel.weights.iter().all(|x| x.is_finite()));
    assert!(
        (channel.feasibility_error() - endpoint["channel"]["feasibility_error"].as_f64().unwrap())
            .abs()
            < 1e-12
    );
    assert!(
        (task::rule_error(&channel, expected)
            - endpoint["channel"]["rule_error"].as_f64().unwrap())
        .abs()
            < 1e-12
    );
    assert!(
        (channel.gap(&expected.data.train.iter().collect::<Vec<_>>())
            - endpoint["channel"]["fw_gap"].as_f64().unwrap())
        .abs()
            < 1e-12
    );
    assert_eq!(task::spectrum(expected, Some(&channel)), endpoint["spectrum"]);
    for (split, samples) in [("train", &expected.data.train), ("test", &expected.data.test)] {
        let (mut loss, mut correct, mut count) = (0.0, 0usize, 0usize);
        for sample in samples {
            let logits = channel.logits(sample);
            for (i, &target) in sample.targets.iter().enumerate() {
                let row = &logits[i * task::CLASSES..][..task::CLASSES];
                loss += ce_gradient(row, target).0;
                let mut prediction = 0;
                for c in 1..row.len() {
                    if row[c] > row[prediction] {
                        prediction = c;
                    }
                }
                correct += usize::from(prediction == target);
                count += 1;
            }
        }
        assert!((loss / count as f64 - endpoint[split]["ce"].as_f64().unwrap()).abs() < 1e-12);
        assert_eq!(correct as f64 / count as f64, endpoint[split]["accuracy"].as_f64().unwrap());
    }
}

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    assert_eq!(args.len(), 1, "usage: report RUN_ROOT");
    let root = Path::new(&args[0]);
    let source = source_hash();
    let mut rows = Vec::new();
    for task_name in ["mqar", "recoded"] {
        for method in ["channel", "copy", "transformer"] {
            let mut results = Vec::new();
            for seed in 0..3 {
                let dir = root.join(format!("{task_name}-{method}-{seed}"));
                let meta = read(&dir.join("run.json"));
                let endpoint = read(&dir.join("final.json"));
                let expected = task::make(seed, task_name == "recoded");
                assert_eq!(meta["source_sha256"], source, "source changed: {}", dir.display());
                assert_eq!(meta["task"], task_name);
                assert_eq!(meta["method"], method);
                assert_eq!(meta["seed"], seed);
                assert_eq!(meta["train_sha256"], data::fingerprint(&expected.data.train));
                assert_eq!(meta["test_sha256"], data::fingerprint(&expected.data.test));
                assert_eq!(meta["output_rule"], json!(expected.rule));
                assert_eq!(meta["flat_target"], task::spectrum(&expected, None));
                assert_eq!(meta["atom_separation"], true);
                let steps = if method == "transformer" { 6000 } else { 1000 };
                assert_eq!(meta["steps"], steps);
                assert_eq!(endpoint["step"], steps);
                let log: Vec<Value> = fs::read_to_string(dir.join("metrics.jsonl"))
                    .unwrap()
                    .lines()
                    .map(|l| serde_json::from_str(l).unwrap())
                    .collect();
                assert_eq!(log[0]["step"], 0);
                assert!(log.windows(2).all(|w| w[0]["step"].as_u64() < w[1]["step"].as_u64()));
                for row in &log {
                    for split in ["train", "test"] {
                        let ce = row[split]["ce"].as_f64().unwrap();
                        let acc = row[split]["accuracy"].as_f64().unwrap();
                        assert!(ce.is_finite() && ce >= 0.0 && (0.0..=1.0).contains(&acc));
                        assert_eq!(row[split]["queries"], task::KEYS * task::TRAIN);
                    }
                    if method == "channel" {
                        let err = row["channel"]["feasibility_error"].as_f64().unwrap();
                        let gap = row["channel"]["fw_gap"].as_f64().unwrap();
                        assert!((0.0..1e-10).contains(&err) && gap.is_finite() && gap > -1e-10);
                    }
                    if task_name == "recoded" && method == "copy" {
                        assert_eq!(row["test"]["accuracy"], 0.0);
                    }
                }
                for key in ["step", "parameters", "train", "test", "channel"] {
                    assert_eq!(endpoint[key], log.last().unwrap()[key]);
                }
                let saved = read(&dir.join("parameters.json"));
                assert_eq!(saved["step"], steps);
                assert_eq!(
                    saved["arrays"][0].as_array().unwrap().len(),
                    endpoint["parameters"].as_u64().unwrap() as usize
                );
                if method == "channel" {
                    verify_saved_channel(&saved, &endpoint, &expected);
                }
                let first100 =
                    log.iter().find(|r| r["test"]["accuracy"] == 1.0).map(|r| r["step"].clone());
                results.push(
                    json!({"seed":seed,"first_observed_test_100":first100,"endpoint":endpoint}),
                );
            }
            let values = |split: &str, field: &str| -> Value {
                let xs: Vec<f64> =
                    results.iter().map(|r| r["endpoint"][split][field].as_f64().unwrap()).collect();
                json!({"mean":xs.iter().sum::<f64>()/3.0,
                    "min":xs.iter().copied().fold(f64::INFINITY,f64::min),
                    "max":xs.iter().copied().fold(f64::NEG_INFINITY,f64::max)})
            };
            rows.push(json!({"task":task_name,"method":method,"train_accuracy":values("train","accuracy"),
                "test_accuracy":values("test","accuracy"),"test_ce":values("test","ce"),"runs":results}));
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"validated_runs":18,"source_sha256":source,"rows":rows})
        )
        .unwrap()
    );
}
