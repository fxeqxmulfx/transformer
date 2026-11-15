use super::*;
use std::collections::BTreeMap;

struct Scratch(std::path::PathBuf);
impl Scratch {
    fn new(name: &str) -> Result<Self> {
        let path = std::env::temp_dir().join(format!(
            "coordinate-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        ));
        fs::create_dir(&path)?;
        Ok(Self(path))
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn example(word: Vec<i64>, answer: Vec<i64>) -> data::Example {
    let mut prompt = word;
    prompt.push(18);
    let tokens = prompt
        .iter()
        .chain(&answer[..answer.len() - 1])
        .copied()
        .collect();
    let targets: Vec<_> = std::iter::repeat_n(-100, prompt.len() - 1)
        .chain(answer.iter().copied())
        .collect();
    let mut previous = None;
    let changes = targets
        .iter()
        .map(|&y| {
            let changed = y >= 0 && previous != Some(y);
            if y >= 0 {
                previous = Some(y);
            }
            changed
        })
        .collect();
    data::Example {
        task: "toy".into(),
        tokens,
        targets,
        changes,
        prompt,
        limit: answer.len(),
        answer,
        field: None,
        row_index: None,
    }
}
fn dataset() -> frequency::dataset::Dataset {
    let train = vec![
        example(vec![3, 4, 4], vec![21, 22, 22, 17]),
        example(vec![5, 3, 5, 3], vec![22, 22, 22, 22, 17]),
        example(vec![3, 4, 5, 5, 3], vec![22, 21, 22, 22, 22, 17]),
    ];
    let test = vec![example(vec![3, 3, 3, 4], vec![23, 23, 23, 21, 17])];
    frequency::dataset::Dataset {
        name: "toy".into(),
        task: data::Task {
            vocab: 32,
            selection: "test".into(),
            splits: BTreeMap::from([("train".into(), train), ("test".into(), test)]),
        },
        original_fingerprints: BTreeMap::new(),
        tensor_fingerprints: BTreeMap::new(),
        empty_novel_splits: vec![],
    }
}

#[test]
fn six_states_full_raw_replay_and_original_final_diagnostics_stay_separate() -> Result<()> {
    let data = dataset();
    let input = parent::dataset::input(32, &data.task.splits["train"])?;
    let bound = evaluation::forecast(32, &data.task.splits)?;
    let stage = construct(input, json!({"final_token_diagnostic":true}), &bound)?;
    assert_eq!(stage.endpoints.len(), 6);
    assert!(stage.endpoints.iter().all(|e| e.error.is_none()));
    let restored: Stage = serde_json::from_str(&serde_json::to_string(&stage)?)?;
    assert_eq!(
        restored,
        construct(stage.input.clone(), stage.metadata.clone(), &bound)?
    );
    let a = Scratch::new("raw-a")?;
    let b = Scratch::new("raw-b")?;
    let first = score::measured(&data, &stage, &a.0)?;
    let second = score::measured(&data, &restored, &b.0)?;
    assert!(artifacts::same_json(&first, &second)?);
    for endpoint in first["endpoints"].as_array().unwrap() {
        let measurement = &endpoint["measurement"];
        let split = &measurement["splits"]["test"];
        assert_eq!(split["metrics"]["final_answer_accuracy"], 1.);
        if endpoint["prior"] == "Coordinates" {
            assert_eq!(split["goals"]["both99"], true);
            assert_eq!(split["unseen_label_correct"], 3);
        } else {
            assert_eq!(split["goals"]["original_success"], false);
            assert_eq!(split["metrics"]["sequence_accuracy"], 0.);
            assert_eq!(split["generated_unidentified_queries"], 3);
        }
        let file = measurement["raw_file"].as_str().unwrap();
        assert_eq!(digest(&a.0.join(file))?, digest(&b.0.join(file))?);
    }
    let lines = fs::read_to_string(a.0.join("raw-3.jsonl"))?;
    let raw: Vec<Value> = lines
        .lines()
        .map(serde_json::from_str)
        .collect::<std::result::Result<_, _>>()?;
    let generated = raw
        .iter()
        .find(|r| r["split"] == "test" && r["phase"] == "generation" && r["step"] == 1)
        .unwrap();
    assert_eq!(
        generated["tokens"].as_array().unwrap().last(),
        Some(&json!(0))
    );
    Ok(())
}

#[test]
fn combined_memory_refusal_precedes_search_and_keeps_all_six_failures() -> Result<()> {
    let data = dataset();
    let input = parent::dataset::input(32, &data.task.splits["train"])?;
    let forecast = evaluation::Forecast {
        scalar_work: 1,
        calls: 1,
        raw_bytes: 56 * 1024u64.pow(3),
    };
    let out = construct(input, json!({}), &forecast)?;
    assert!(out.error.is_some() && out.discovery.is_none());
    assert_eq!(out.endpoints.len(), 6);
    assert!(
        out.endpoints
            .iter()
            .all(|e| e.trained.is_none() && e.error.is_some())
    );
    Ok(())
}

#[test]
fn missing_or_modified_seal_blocks_every_score() -> Result<()> {
    let dir = Scratch::new("seals")?;
    assert!(all_sealed(&dir.0).is_err());
    for case in cases() {
        let at = dir.0.join(&case);
        fs::create_dir(&at)?;
        write(&at.join("state.json"), &json!({"fixture":case}))?;
        write(
            &at.join("seal.json"),
            &json!({"case":case,"endpoint_count":6,
            "all_modes_and_priors_sealed_before_scoring":true,"state_sha256":digest(&at.join("state.json"))?}),
        )?;
    }
    all_sealed(&dir.0)?;
    fs::write(dir.0.join(&cases()[0]).join("state.json"), b"{}\n")?;
    assert!(all_sealed(&dir.0).is_err());
    Ok(())
}

#[test]
fn summary_denominators_keep_refusals_and_both_prior_arms() -> Result<()> {
    let endpoints:Vec<_>=["Coordinates","Categorical"].iter().flat_map(|prior|["Forward","Full","Mini"].iter()
        .map(move|mode|json!({"prior":prior,"mode":mode,"error":null,"primary":{"original_success":true,"raw99":true,"exact99":true,"both99":true},"test":{"both99":true},"all_heldout99":true}))).collect();
    let mut rows: Vec<Value> = (0..7)
        .map(|_| json!({"results":{"endpoints":endpoints}}))
        .collect();
    rows[0]["results"]["endpoints"][0]["error"] = json!("refused");
    rows[0]["results"]["endpoints"][0]["primary"] = Value::Null;
    rows[0]["results"]["endpoints"][0]["test"] = Value::Null;
    rows[0]["results"]["endpoints"][0]["all_heldout99"] = json!(false);
    let totals = score::totals(&rows)?;
    assert_eq!(totals[0]["new_cases"], 7);
    assert_eq!(totals[0]["refusals"], 1);
    assert_eq!(totals[0]["primary_both99"], 6);
    rows.pop();
    assert!(score::totals(&rows).is_err());
    Ok(())
}
