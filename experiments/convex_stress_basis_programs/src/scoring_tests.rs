//! Scoring failures use synthetic fixtures or original training rows only.
//! No actual heldout measurement is performed by these pre-freeze checks.
use crate::*;
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::PathBuf};

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Result<Self> {
        let path = std::env::temp_dir().join(format!(
            "basis-score-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        ));
        std::fs::create_dir(&path)?;
        Ok(Self(path))
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn grouped_training_trace_and_individual_queries_recount_and_reject_changed_ce() -> Result<()> {
    let rows = vec![
        basis::wire::example(
            "blocks",
            vec![1, 9, 9, 10, 10, 9],
            vec![-100, 15, 15, 16, 16, 15],
        ),
        basis::wire::example("blocks", vec![1, 10, 10, 9, 10], vec![-100, 15, 15, 15, 15]),
        basis::wire::example("blocks", vec![1, 9, 10, 9, 10], vec![-100, 15, 16, 15, 15]),
    ];
    let stage = fit::construct(36, &rows)?;
    let model = stage.endpoints[0].model.as_ref().unwrap();
    let corpus = model::corpus(&stage.discovery, &rows, model.prior)?;
    let input = &stage.discovery.input;
    let splits = BTreeMap::from([("train".into(), rows.clone()), ("unit_probe".into(), rows)]);
    let dir = Scratch::new()?;
    let path = dir.0.join("raw.jsonl");
    let saved = evaluation::evaluate(model, input, &corpus, &splits, &path)?;
    let audit = audit::verify(model, input, &corpus, &splits, &path, &saved)?;
    assert_eq!(audit["original_training_references"], 13);
    assert_eq!(audit["original_query_positions"], 26);
    assert_eq!(audit["independent_product_queries"], 15);
    assert_eq!(saved["splits"]["train"]["metrics"]["correct"], 13);
    assert_eq!(saved["splits"]["unit_probe"]["metrics"]["exact"], 3);
    let mut records: Vec<Value> = std::fs::read_to_string(&path)?
        .lines()
        .map(serde_json::from_str)
        .collect::<std::result::Result<_, _>>()?;
    let record = records
        .iter_mut()
        .find(|r| r["phase"] == "teacher")
        .unwrap();
    record["ce"] = json!(record["ce"].as_f64().unwrap() + 1.);
    let tampered = dir.0.join("tampered.jsonl");
    let mut raw = records
        .iter()
        .map(serde_json::to_string)
        .collect::<std::result::Result<Vec<_>, _>>()?
        .join("\n");
    raw.push('\n');
    std::fs::write(&tampered, raw)?;
    let mut changed = saved;
    // Renewing the checksum must not hide a wrong measurement.
    changed["raw_sha256"] = json!(artifacts::digest(&tampered)?);
    let error = audit::verify(model, input, &corpus, &splits, &tampered, &changed).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("raw floating measurement differs")
    );
    Ok(())
}

#[test]
fn own_output_readback_preserves_early_eos_and_wrong_limit_failures() -> Result<()> {
    let (vocabulary, rows) = input::training_only("basis-parity-noise20")?;
    let stage = fit::construct(vocabulary, &rows)?;
    let mut model = stage.endpoints[0].model.clone().unwrap();
    let corpus = model::corpus(&stage.discovery, &rows, model.prior)?;
    let input = &stage.discovery.input;
    let mut probe = rows[..2].to_vec();
    for row in &mut probe {
        row.limit = 2;
    }
    let splits = BTreeMap::from([("unit_probe".into(), probe)]);
    let dir = Scratch::new()?;
    for class in [0, 17] {
        let coordinate = model.head.coordinates[class];
        // A valid complete head which deliberately emits the same wrong answer.
        // This is an evaluator fixture, not an endpoint of the fitted domain.
        model.head = joint::head::Head::new(
            model.head.coordinates.clone(),
            vec![[0., coordinate, coordinate, 0., 1., 1.]],
        )?;
        let path = dir.0.join(format!("wrong-{class}.jsonl"));
        let saved = evaluation::evaluate(&model, input, &corpus, &splits, &path)?;
        audit::verify(&model, input, &corpus, &splits, &path, &saved)?;
        let metrics = &saved["splits"]["unit_probe"]["metrics"];
        assert_eq!(metrics["targets"], 4);
        assert_eq!(metrics["correct"], 0);
        assert_eq!(metrics["exact"], 0);
        assert_eq!(metrics["generated_tokens"], if class == 17 { 2 } else { 4 });
        assert_eq!(metrics["eos_rate"], if class == 17 { 1. } else { 0. });
        assert!(metrics["teacher_forced"]["loss"].as_f64().unwrap() > 0.);
        let raw = std::fs::read_to_string(path)?;
        let records: Vec<Value> = raw
            .lines()
            .map(serde_json::from_str)
            .collect::<std::result::Result<_, _>>()?;
        let result = records.iter().find(|r| r["phase"] == "result").unwrap();
        assert_eq!(
            result["generated"],
            if class == 17 {
                json!([17])
            } else {
                json!([0, 0])
            }
        );
        if class == 0 {
            let second = records
                .iter()
                .find(|r| r["phase"] == "generation" && r["step"] == 1)
                .unwrap();
            assert_eq!(second["tokens"].as_array().unwrap().last(), Some(&json!(0)));
        }
    }
    Ok(())
}
