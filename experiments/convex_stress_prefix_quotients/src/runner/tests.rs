use super::*;
use std::collections::BTreeMap;

fn temporary(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("prefix344-{}-{name}", std::process::id()))
}
fn example(prompt: Vec<i64>, answer: Vec<i64>, limit: usize) -> dataset::data::Example {
    let tokens = prompt
        .iter()
        .chain(&answer[..answer.len() - 1])
        .copied()
        .collect::<Vec<_>>();
    let targets = std::iter::repeat_n(-100, prompt.len() - 1)
        .chain(answer.iter().copied())
        .collect::<Vec<_>>();
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
    dataset::data::Example {
        tokens,
        targets,
        changes,
        prompt,
        answer,
        limit,
        field: None,
        row_index: None,
        task: "hand_fixture".into(),
    }
}
fn forward() -> joint::training::Policy {
    joint::training::Policy {
        mode: joint::training::Mode::Forward,
        epochs: 0,
        batch: 2,
        learning_rate: 1e5,
        initial_total_mass: 131072.,
        projection_cycles: 20000,
        projection_tolerance: 1e-10,
    }
}

#[test]
fn original_rollout_scoring_and_unseen_quantity_control_are_audited() -> Result<()> {
    let train = vec![
        example(vec![1, 100, 102, 18], vec![100, 101, 102, 17], 6),
        example(vec![1, 102, 104, 18], vec![102, 103, 104, 17], 6),
    ];
    let input = adapter::input(229, &train)?;
    let bank = vec![program::Program::Range {
        reverse: false,
        inclusive: true,
    }];
    let d = discovery::run(&input.observations, observations::coordinates(229)?, bank)?;
    let unseen = example(vec![1, 103, 106, 18], vec![103, 104, 105, 106, 17], 7);
    let splits = BTreeMap::from([("hand_check".to_string(), vec![unseen])]);
    let mut results = vec![];
    for prior in [model::Prior::Coordinates, model::Prior::Categorical] {
        let t = model::train(&d, &input.observations, prior, forward(), 1e9, 1.)?;
        score::audit_training(&t)?;
        let path = temporary(&format!("rollout-{prior:?}.jsonl"));
        let result = evaluation::evaluate(&t.model, &input, &splits, &path)?;
        let raw = fs::read_to_string(&path)?;
        assert!(
            raw.lines()
                .all(|l| serde_json::from_str::<Value>(l).is_ok())
        );
        assert_eq!(artifacts::digest(&path)?, dataset::hash(raw.as_bytes()));
        let last: Value = serde_json::from_str(raw.lines().last().unwrap())?;
        assert_eq!(last["phase"], "result");
        assert_eq!(last["generated"].as_array().unwrap().last().unwrap(), 17);
        assert_eq!(result["splits"]["hand_check"]["unseen_label_targets"], 2);
        results.push(result);
        fs::remove_file(path)?;
    }
    assert_eq!(results[0]["splits"]["hand_check"]["goals"]["both99"], true);
    assert_eq!(
        results[0]["splits"]["hand_check"]["unseen_label_correct"],
        2
    );
    assert_eq!(results[1]["splits"]["hand_check"]["metrics"]["correct"], 3);
    assert_eq!(results[1]["splits"]["hand_check"]["goals"]["both99"], false);
    Ok(())
}

#[test]
fn prefix_adapter_preserves_bos_ignore_and_transition_denominators() -> Result<()> {
    let e = dataset::data::Example {
        tokens: vec![1, 9, 10, 10, 23, 9],
        targets: vec![-100, 15, 15, 16, 16, 15],
        changes: vec![false, true, false, true, false, true],
        prompt: vec![],
        answer: vec![],
        limit: 0,
        field: None,
        row_index: None,
        task: "hand_prefix".into(),
    };
    let input = adapter::input(36, std::slice::from_ref(&e))?;
    assert_eq!(input.observations.len(), 5);
    assert_eq!(input.observations[0].tokens, vec![1, 9]);
    let f = prefix::Formula::Less(
        Box::new(prefix::Term::Count(Box::new(prefix::Formula::Symbol(9)))),
        Box::new(prefix::Term::Count(Box::new(prefix::Formula::Symbol(10)))),
    );
    let d = discovery::run(
        &input.observations,
        observations::coordinates(36)?,
        vec![program::Program::Prefix(f)],
    )?;
    let t = model::train(
        &d,
        &input.observations,
        model::Prior::Coordinates,
        forward(),
        1e9,
        1.,
    )?;
    let path = temporary("prefix.jsonl");
    let result = evaluation::evaluate(
        &t.model,
        &input,
        &BTreeMap::from([("fixture".to_string(), vec![e.clone()])]),
        &path,
    )?;
    let m = &result["splits"]["fixture"]["metrics"];
    assert_eq!(m["targets"], 5);
    assert_eq!(m["correct"], 5);
    assert_eq!(m["exact"], 1);
    assert_eq!(m["transition_targets"], 3);
    assert_eq!(m["transition_correct"], 3);
    assert!(m["eos_rate"].is_null());
    assert_eq!(result["actual_audit_calls"], 5);
    let mut wrong = e;
    wrong.changes[2] = true;
    assert!(adapter::input(36, &[wrong]).is_err());
    fs::remove_file(path)?;
    Ok(())
}

#[test]
fn all_seals_refusals_and_streaming_artifact_integrity_are_enforced() -> Result<()> {
    let root = temporary("seals");
    fs::create_dir(&root)?;
    assert!(all_sealed(&root).is_err());
    for case in cases() {
        let dir = root.join(&case);
        fs::create_dir(&dir)?;
        write(&dir.join("state.json"), &json!({"case":case}))?;
        write(
            &dir.join("seal.json"),
            &json!({"case":case,"endpoint_count":6,"all_modes_and_priors_sealed_before_scoring":true,
            "state_sha256":digest(&dir.join("state.json"))?}),
        )?;
    }
    all_sealed(&root)?;
    fs::write(root.join(&cases()[0]).join("state.json"), b"{}\n")?;
    assert!(all_sealed(&root).is_err());
    let object = json!({"text":"a\nb","number":0.12345678901234567,"list":[1,2,3]});
    let file = root.join("artifact.json");
    write(&file, &object)?;
    assert_eq!(read::<Value>(&file)?, object);
    assert_eq!(artifacts::canonical_digest(&object)?, digest(&file)?);
    assert!(write(&file, &object).is_err());
    let endpoints: Vec<_> = ["Coordinates", "Categorical"]
        .iter()
        .flat_map(|p| {
            ["Forward", "Full", "Mini"].iter().map(move |m| {
                json!({"prior":p,"mode":m,"error":"refused","measurement":null,
            "primary":null,"test":null,"all_heldout99":false})
            })
        })
        .collect();
    let rows: Vec<_> = (0..13)
        .map(|_| json!({"results":{"endpoints":endpoints}}))
        .collect();
    for total in summary::totals(&rows)? {
        assert_eq!(total["refusals"], 13);
        assert_eq!(total["fully_measured"], 0);
        assert_eq!(total["all_heldout99"], 0);
    }
    assert!(summary::totals(&rows[..12]).is_err());
    fs::remove_dir_all(root)?;
    Ok(())
}
