use super::*;
use head::kernel;

fn fixture(mode: head::Mode) -> Result<(Value, data::Task, Value)> {
    let observations: Vec<_> = (0..49)
        .filter(|x| ![8, 24, 40].contains(x))
        .map(|x| {
            [
                x,
                if (x % 7 + x / 7 < 6) ^ [4, 16, 30].contains(&x) {
                    0
                } else {
                    2
                },
            ]
        })
        .collect();
    let report = kernel::discovery::run(kernel::discovery::Request {
        prime: 3,
        input: kernel::discovery::Input {
            n: 7,
            vocabulary: 7,
            observations: observations.clone(),
            policy: kernel::discovery::Policy::default(),
        },
    })?;
    let source = kernel::initialize(&report)?.model;
    let mut records: Vec<_> = observations
        .iter()
        .enumerate()
        .map(|(id, &[cell, label])| records::Record { id, cell, label })
        .collect();
    records.push(records::Record {
        id: records.len(),
        cell: records[0].cell,
        label: records[0].label,
    });
    let pin = head::digest(&source)?;
    let context = head::scale::context::Model::new(source, records.clone(), &pin)?;
    let pin = head::digest(&context)?;
    let scaled = head::scale::Model::new(context, &pin)?;
    let pin = head::digest(&scaled)?;
    let endpoint = head::Endpoint {
        mode,
        source_reference: "context-scale:context:fixture".into(),
        source_gradient_training_run: mode != head::Mode::Forward,
        source_optimizer_updates: Some(usize::from(mode != head::Mode::Forward)),
        mixture_batch: 8,
    };
    let mixture = head::parent::parent::Model::new(scaled, endpoint, &pin)?;
    let pin = head::digest(&mixture)?;
    let parent = head::parent::Model::new(mixture, &pin)?;
    let mode = match mode {
        head::Mode::Full => "full_batch",
        head::Mode::Mini => "minibatch",
        head::Mode::Forward => "forward_only",
    };
    let example = |cell: usize, label: usize| data::Example {
        task: "fixture".into(),
        tokens: vec![(cell / 7) as i64, (cell % 7) as i64],
        targets: vec![-100, label as i64],
        changes: vec![false, false],
        prompt: vec![],
        answer: vec![],
        limit: 0,
        field: None,
        row_index: None,
    };
    let task = data::Task {
        vocab: 7,
        selection: "validation".into(),
        splits: BTreeMap::from([
            (
                "train".into(),
                records.iter().map(|r| example(r.cell, r.label)).collect(),
            ),
            (
                "validation".into(),
                vec![example(8, 0), example(24, 2), example(40, 2)],
            ),
        ]),
    };
    Ok((
        json!({"prior":parent,"supplement":null}),
        task,
        json!({"reference":"gated-context:context-mixture:context-scale:context:fixture","training_mode":mode}),
    ))
}

#[test]
fn original_records_and_mode_are_bound_while_heldout_targets_do_not_select_mixtures() -> Result<()>
{
    for mode in ["full_batch", "minibatch", "forward_only"] {
        let (value, mut task, entry) = fixture(import::mode(&json!(mode))?)?;
        let m = model::Model::construct(value.clone(), &task, &entry, 8)?;
        assert!(m.forecast(&task)? > 0);
        let saved = serde_json::to_value(&m)?;
        assert_eq!(
            saved["prior"]["parent"]["parent"]["parent"]["context"]["records"]
                .as_array()
                .unwrap()
                .len(),
            47
        );
        let restored = model::Model::load(saved.clone(), &m)?;
        assert_eq!(restored, m);
        task.splits.get_mut("validation").unwrap()[0].targets[1] = 6;
        assert_eq!(model::Model::construct(value.clone(), &task, &entry, 8)?, m);
        for split in task.splits.values() {
            metrics::evaluate(&m, split)?;
            metrics::evaluate(&model::Parent(&m), split)?;
        }
        let mut bad = saved;
        bad["prior"]["selection"]["weights"][0][0] = json!(2.);
        assert!(model::Model::load(bad, &m).is_err());
        let info = m.info();
        if mode == "forward_only" {
            assert_eq!(info["new_gradient_batches"], 0);
            assert_eq!(info["new_optimizer_updates"], 0);
        } else {
            assert!(info["new_gradient_batches"].as_u64().unwrap() > 0);
        }
    }
    let (value, mut task, entry) = fixture(head::Mode::Forward)?;
    task.splits.get_mut("train").unwrap()[0].targets[1] = 6;
    assert!(model::Model::construct(value, &task, &entry, 8).is_err());
    Ok(())
}

#[test]
fn eligibility_uses_current_context_hashes_without_heldout_scores_or_stale_metadata() -> Result<()>
{
    let choice = json!({"task":"fixture","reference":"gated-context:context-mixture:context-scale:context:fixture","training_mode":"full_batch",
        "model_sha256":"a".repeat(64),"saved_endpoint":{"model_sha256":"stale"},"metrics":{"validation":{"loss":8.}}});
    let mut snapshot = json!({"arms":[{"choices":[choice]},{"choices":[choice]}]});
    let original = import::eligible(&snapshot, "fixture")?;
    assert_eq!(original.len(), 1);
    assert_eq!(original[0]["source_model_sha256"], "a".repeat(64));
    for arm in snapshot["arms"].as_array_mut().unwrap() {
        arm["choices"][0]["saved_endpoint"] = json!({"model_sha256":"another stale object"});
        arm["choices"][0]["metrics"] = json!({"validation":{"loss":0.,"token_accuracy":1.}});
    }
    assert_eq!(import::eligible(&snapshot, "fixture")?, original);
    snapshot["arms"][1]["choices"][0]["model_sha256"] = json!("b".repeat(64));
    assert!(import::eligible(&snapshot, "fixture").is_err());
    Ok(())
}

#[test]
fn source_manifest_and_prediction_layout_mutations_are_rejected() -> Result<()> {
    let (value, mut task, entry) = fixture(head::Mode::Forward)?;
    task.splits.get_mut("train").unwrap()[0].tokens.push(1);
    assert!(model::Model::construct(value.clone(), &task, &entry, 8).is_err());
    let (mut value, task, entry) = fixture(head::Mode::Forward)?;
    value["prior"]["parent"]["parent"]["context"]["source"]["coefficients"][0] = json!(999.);
    assert!(model::Model::construct(value, &task, &entry, 8).is_err());
    let dir = std::env::temp_dir().join(format!("margin-guard-pins-{}", std::process::id()));
    fs::create_dir(&dir)?;
    let path = dir.join("source.json");
    write(&path, &json!({"source":"original"}))?;
    let files = BTreeMap::from([(path.to_str().unwrap().to_owned(), digest(&path)?)]);
    parent::pinned(&path, &files)?;
    fs::write(&path, b"changed")?;
    assert!(parent::pinned(&path, &files).is_err());
    assert!(parent::pinned(&dir.join("missing"), &files).is_err());
    fs::remove_dir_all(dir)?;
    Ok(())
}

#[test]
fn inherited_mode_batch_and_reference_cannot_be_relabelled() -> Result<()> {
    let (value, task, mut entry) = fixture(head::Mode::Forward)?;
    assert!(model::Model::construct(value.clone(), &task, &entry, 7).is_err());
    entry["training_mode"] = json!("full_batch");
    assert!(model::Model::construct(value.clone(), &task, &entry, 8).is_err());
    entry["training_mode"] = json!("forward_only");
    entry["reference"] = json!("gated-context:context-mixture:context-scale:context:changed");
    assert!(model::Model::construct(value, &task, &entry, 8).is_err());
    Ok(())
}

#[test]
fn disk_seal_reconstruction_rejects_changed_sources_and_changed_coefficients() -> Result<()> {
    let dir = std::env::temp_dir().join(format!("margin-sealed-model-{}", std::process::id()));
    fs::create_dir(&dir)?;
    let (value, mut task, mut entry) = fixture(head::Mode::Forward)?;
    let model = model::Model::construct(value.clone(), &task, &entry, 8)?;
    let pin = head::digest(&model)?;
    write(&dir.join("source.json"), &value)?;
    entry["file"] = json!("source.json");
    entry["source_model_sha256"] = json!(digest(&dir.join("source.json"))?);
    let sealed = json!({"model_sha256":pin});
    drop(model);
    let rebuilt = run::sealed_model(&dir, &task, &entry, 8, &sealed)?;
    assert_eq!(head::digest(&rebuilt)?, pin);
    drop(rebuilt);
    task.splits.get_mut("validation").unwrap()[0].targets[1] = 6;
    assert_eq!(
        head::digest(&run::sealed_model(&dir, &task, &entry, 8, &sealed)?)?,
        pin
    );
    let bad = json!({"model_sha256":"b".repeat(64)});
    assert!(run::sealed_model(&dir, &task, &entry, 8, &bad).is_err());
    fs::write(dir.join("source.json"), b"{}")?;
    assert!(run::sealed_model(&dir, &task, &entry, 8, &sealed).is_err());
    fs::remove_dir_all(dir)?;
    Ok(())
}
