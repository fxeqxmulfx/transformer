//! Validate and summarize every frozen run; generated summaries stay in runs/.
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

fn number(v: &Value, split: &str, metric: &str) -> f64 {
    v[split][metric].as_f64().expect("numeric metric")
}

fn stats(xs: impl Iterator<Item = f64>) -> Value {
    let xs: Vec<_> = xs.collect();
    json!({"mean":xs.iter().sum::<f64>() / xs.len() as f64,
        "min":xs.iter().copied().fold(f64::INFINITY,f64::min),
        "max":xs.iter().copied().fold(f64::NEG_INFINITY,f64::max)})
}

fn main() {
    let arg = std::env::args().nth(1).expect("usage: report RUNS_DIRECTORY");
    let root = Path::new(&arg);
    let mut groups = BTreeMap::<String, Vec<Value>>::new();
    let mut raw = Vec::new();
    let mut fingerprints = BTreeMap::<String, Value>::new();
    let mut common_source = None;
    for task in ["mqar", "div17", "div97"] {
        let methods: &[&str] =
            if task == "mqar" { &["matching", "transformer"] } else { &["copy", "table", "transformer"] };
        for &method in methods {
            for seed in 0..3 {
                let dir = root.join(format!("{task}-{method}-{seed}"));
                let meta: Value =
                    serde_json::from_slice(&fs::read(dir.join("run.json")).expect("missing run")).unwrap();
                let lines = fs::read_to_string(dir.join("metrics.jsonl")).unwrap();
                let rows: Vec<Value> =
                    lines.lines().map(|line| serde_json::from_str(line).unwrap()).collect();
                let last = rows.last().expect("empty metrics");
                assert_eq!(
                    last["step"],
                    if task == "mqar" { 6000 } else { 20000 },
                    "incomplete run: {}",
                    dir.display()
                );
                assert_eq!(last["seed"], seed);
                assert_eq!(last["method"], method);
                assert_eq!(last["task"], task);
                let source = meta["source_sha256"].clone();
                if let Some(common) = &common_source {
                    assert_eq!(&source, common, "mixed trainer source");
                } else {
                    common_source = Some(source);
                }
                let key = format!("{task}-{seed}");
                let hashes = json!([meta["train_sha256"], meta["test_sha256"]]);
                if let Some(previous) = fingerprints.insert(key, hashes.clone()) {
                    assert_eq!(previous, hashes, "methods used different data");
                }
                for row in &rows {
                    for split in ["train", "test"] {
                        let accuracy = number(row, split, "accuracy");
                        let ce = number(row, split, "ce");
                        assert!((0.0..=1.0).contains(&accuracy) && ce.is_finite() && ce >= -1e-10);
                        if method == "copy" {
                            let cap = meta[format!("copy_support_{split}")].as_f64().unwrap();
                            assert!(accuracy <= cap + 1e-12, "copy result violates support ceiling");
                        }
                    }
                    if method == "table" {
                        let chance_ce = (meta["vocab"].as_f64().unwrap()).ln();
                        assert!(
                            (number(row, "test", "ce") - chance_ce).abs() < 1e-9,
                            "unseen table row changed"
                        );
                    }
                }
                if method == "matching" {
                    assert_eq!(meta["separates"], true);
                    assert_eq!(last["identity_error"], 0.0);
                    assert_eq!(number(last, "test", "accuracy"), 1.0);
                }
                let first = |threshold| {
                    rows.iter()
                        .find(|r| number(r, "test", "accuracy") >= threshold)
                        .map(|r| r["step"].clone())
                };
                raw.push(json!({"task":task,"method":method,"seed":seed,"endpoint":last,
                    "first_test_99":first(0.99),"first_test_100":first(1.0),
                    "copy_support_test":meta["copy_support_test"]}));
                groups.entry(format!("{task}/{method}")).or_default().push(last.clone());
            }
        }
    }
    let groups: BTreeMap<_, _> = groups
        .into_iter()
        .map(|(name, rows)| {
            let value = json!({"runs":rows.len(),"parameters":rows[0]["parameters"],
            "train_accuracy":stats(rows.iter().map(|r|number(r,"train","accuracy"))),
            "test_accuracy":stats(rows.iter().map(|r|number(r,"test","accuracy"))),
            "test_ce":stats(rows.iter().map(|r|number(r,"test","ce"))),
            "test_all_queries":stats(rows.iter().map(|r|number(r,"test","all_queries_correct"))),
            "elapsed_seconds":stats(rows.iter().map(|r|r["elapsed_seconds"].as_f64().unwrap()))});
            (name, value)
        })
        .collect();
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({"validated_runs":raw.len(),
        "source_sha256":common_source,"groups":groups,"runs":raw}))
        .unwrap()
    );
}
