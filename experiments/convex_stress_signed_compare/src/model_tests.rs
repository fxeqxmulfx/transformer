use super::*;
use convex_stress_shared_roles::training;
use head::kernel;

fn fixture(mode: head::Mode) -> Result<(Value, data::Task, Value)> {
    // Reviewed316 nonlinear fixture: three diagonals with raw aliases and
    // overlapping class hulls. Only observed inputs/labels enter construction.
    let observations: Vec<_> = (0usize..49)
        .filter(|cell| (cell % 7).abs_diff(cell / 7) <= 1)
        .map(|cell| [cell, if cell % 7 == cell / 7 { 2 } else { 0 }])
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
    let initial = kernel::initialize(&report)?.model;
    let records: Vec<_> = (0..45)
        .map(|id| {
            let [cell, label] = observations[id % observations.len()];
            records::Record { id, cell, label }
        })
        .collect();
    let policy = training::policies(records.len(), 8)?
        .into_iter()
        .find(|p| {
            matches!(
                (mode, p.mode),
                (head::Mode::Full, training::Mode::Full)
                    | (head::Mode::Mini, training::Mode::Mini)
                    | (head::Mode::Forward, training::Mode::Forward)
            )
        })
        .unwrap();
    let outcome = training::run(&initial, &records, policy)?;
    training::replay(&initial, &records, &outcome)?;
    let updates = outcome.steps.len();
    assert_eq!(
        updates,
        match mode {
            head::Mode::Full => 1,
            head::Mode::Mini => 6,
            head::Mode::Forward => 0,
        }
    );
    if mode == head::Mode::Mini {
        assert_eq!(outcome.steps.last().unwrap().count, 5);
    }
    let source = outcome.model;
    let pin = head::digest(&source)?;
    let context = head::regions::parent::scale::context::Model::new(source, records.clone(), &pin)?;
    let pin = head::digest(&context)?;
    let scaled = head::regions::parent::scale::Model::new(context, &pin)?;
    let pin = head::digest(&scaled)?;
    let endpoint = head::Endpoint {
        mode,
        source_reference: "context-scale:context:fixture".into(),
        source_gradient_training_run: mode != head::Mode::Forward,
        source_optimizer_updates: Some(updates),
        mixture_batch: 8,
    };
    let mixture = head::regions::parent::parent::Model::new(scaled, endpoint, &pin)?;
    let pin = head::digest(&mixture)?;
    let parent = head::regions::parent::Model::new(mixture, &pin)?;
    let pin = head::digest(&parent)?;
    let parent = head::regions::Model::new(parent, &pin)?;
    let pin = head::digest(&parent)?;
    let parent = head::parent::parent::Model::new(parent, &pin)?;
    let pin = head::digest(&parent)?;
    let parent = head::parent::Model::new(parent, &pin)?;
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
                (0..49)
                    .map(|cell| example(cell, if cell % 7 == cell / 7 { 2 } else { 0 }))
                    .collect(),
            ),
        ]),
    };
    Ok((
        json!({"prior":parent,"supplement":null}),
        task,
        json!({"reference":"crossing-exclusion:raw-exclusion:raw-regions:gated-context:context-mixture:context-scale:context:fixture","training_mode":mode}),
    ))
}

#[test]
fn original_records_and_mode_are_bound_while_heldout_targets_do_not_select_regions() -> Result<()> {
    for mode in ["full_batch", "minibatch", "forward_only"] {
        let (value, mut task, entry) = fixture(import::mode(&json!(mode))?)?;
        let m = model::Model::construct(value.clone(), &task, &entry, 8)?;
        assert!(m.forecast(&task)? > 0);
        let saved = serde_json::to_value(&m)?;
        assert_eq!(m.info()["prior"]["observed_records"], 45);
        let restored = model::Model::load(saved.clone(), &m)?;
        assert_eq!(restored, m);
        task.splits.get_mut("validation").unwrap()[0].targets[1] = 6;
        assert_eq!(model::Model::construct(value.clone(), &task, &entry, 8)?, m);
        for split in task.splits.values() {
            let new = metrics::evaluate(&m, split)?;
            let old = metrics::evaluate(&model::Parent(&m), split)?;
            let audit = m.audit(split)?;
            assert_eq!(audit["correct"], new["correct"]);
            assert_eq!(audit["parent_correct"], old["correct"]);
            assert_eq!(audit["targets"], new["targets"]);
        }
        let mut bad = saved;
        let cell = bad["prior"]["rows"]
            .as_array()
            .unwrap()
            .iter()
            .position(|v| !v.is_null())
            .unwrap();
        bad["prior"]["rows"][cell]["weights"][0] = json!(999.);
        assert!(model::Model::load(bad, &m).is_err());
        let info = m.info();
        assert_eq!(
            info["new_gradient_batches"],
            match mode {
                "full_batch" => 64,
                "minibatch" => 48,
                _ => 0,
            }
        );
        if mode == "forward_only" {
            assert_eq!(info["new_optimizer_updates"], 0);
        }
        if mode == "minibatch" {
            assert_eq!(
                info["prior"]["selection"]["steps"]
                    .as_array()
                    .unwrap()
                    .last()
                    .unwrap()["count"],
                5
            );
        }
        assert_eq!(info["new_local_row_updates"], 0);
        assert!(info["new_context_rows"].as_u64().unwrap() > 0);
        assert!(
            info["prior"]["statistics"]["negative_weight_rows"]
                .as_u64()
                .unwrap()
                > 0
        );
        assert_eq!(info["prior"]["target_convexity_verified"], false);
        assert_eq!(info["prior"]["candidate_pool_closure_proved"], false);
        assert_eq!(
            info["prior"]["unconditional_raw_label_identification"],
            false
        );
        let audited = m.audit(&task.splits["validation"])?;
        assert_eq!(audited["signed_feature_queries"], info["new_context_rows"]);
        assert_eq!(
            audited["signed_feature_correct"].as_u64().unwrap()
                + audited["signed_feature_errors"].as_u64().unwrap(),
            audited["signed_feature_queries"].as_u64().unwrap()
        );
    }
    let (value, mut task, entry) = fixture(head::Mode::Forward)?;
    task.splits.get_mut("train").unwrap()[0].targets[1] = 6;
    assert!(model::Model::construct(value, &task, &entry, 8).is_err());
    Ok(())
}

#[test]
fn eligibility_uses_current_context_hashes_without_heldout_scores_or_stale_metadata() -> Result<()>
{
    let choice = json!({"task":"fixture","reference":"crossing-exclusion:raw-exclusion:raw-regions:gated-context:context-mixture:context-scale:context:fixture","training_mode":"full_batch",
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
    *value
        .pointer_mut("/prior/parent/parent/parent/parent/parent/context/source/coefficients/0")
        .unwrap() = json!(999.);
    assert!(model::Model::construct(value, &task, &entry, 8).is_err());
    let dir = std::env::temp_dir().join(format!("signed-compare-pins-{}", std::process::id()));
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
    entry["reference"] = json!(
        "crossing-exclusion:raw-exclusion:raw-regions:gated-context:context-mixture:context-scale:context:changed"
    );
    assert!(model::Model::construct(value, &task, &entry, 8).is_err());
    Ok(())
}

#[test]
fn disk_seal_reconstruction_rejects_changed_sources_and_changed_coefficients() -> Result<()> {
    let dir = std::env::temp_dir().join(format!(
        "signed-compare-sealed-model-{}",
        std::process::id()
    ));
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
