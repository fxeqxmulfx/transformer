//! Synthetic controls have no production fixture switch or original task credit.
use super::*;
use probes::{
    boundary::Predictor,
    dataset::{Dataset, Split},
    inventory::{Entry, Inventory},
    wire::Record,
};
use resources::Guard;
use schema::{Header, Stage};
use serde_json::json;
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
};
static IDS: AtomicUsize = AtomicUsize::new(0);
struct Temp(PathBuf);
impl Temp {
    fn new() -> Result<Self> {
        let p = std::env::temp_dir().join(format!(
            "convex-probe-compare-{}-{}",
            std::process::id(),
            IDS.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&p)?;
        Ok(Self(p))
    }
    fn guard(&self) -> Result<Guard> {
        Guard::new(&self.0, &self.0)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn generation() -> Dataset {
    let records: Vec<_> = (36..=37)
        .flat_map(|a| {
            (36..=37).map(move |b| Record::generation("synthetic", &[a, b], &[a + 2, b + 2]))
        })
        .collect();
    let teachers = records
        .iter()
        .map(|r| {
            r.answer[..2]
                .iter()
                .map(|&y| ((y - 36) as f64).to_bits())
                .collect()
        })
        .collect();
    let split = Split {
        records,
        teacher_words: Some(teachers),
    };
    Dataset {
        name: "synthetic".into(),
        vocabulary: 40,
        selection: "validation".into(),
        numeric_step: Some(1.),
        numeric_offset: Some(0),
        splits: [
            ("train".into(), split.clone()),
            ("validation".into(), split.clone()),
            ("test".into(), split),
        ]
        .into(),
    }
}
fn dense(contradiction: bool) -> Dataset {
    let pairs: Vec<_> = (1..=4)
        .flat_map(|a| (1..=4).filter(move |&b| a != b).map(move |b| (a, b)))
        .collect();
    let records = (0..130)
        .map(|i| {
            let (a, b) = if contradiction {
                (1, 2)
            } else {
                pairs[i % pairs.len()]
            };
            let label = if contradiction {
                6 + i as i64 % 2
            } else if a < b {
                6
            } else {
                7
            };
            Record {
                task: "synthetic".into(),
                tokens: vec![a, b],
                targets: vec![-100, label],
                changes: vec![false, true],
                prompt: vec![],
                answer: vec![],
                generation_limit: 0,
            }
        })
        .collect();
    let split = Split {
        records,
        teacher_words: None,
    };
    Dataset {
        name: "synthetic".into(),
        vocabulary: 8,
        selection: "validation/novel".into(),
        numeric_step: None,
        numeric_offset: None,
        splits: [
            ("train".into(), split.clone()),
            ("validation".into(), split.clone()),
            ("validation/novel".into(), split),
        ]
        .into(),
    }
}
struct Fixture {
    inputs: inputs::Inputs,
    data: Vec<Option<Dataset>>,
    stages: Vec<Stage>,
}
fn fixture(root: &Path, guard: &Guard) -> Result<Fixture> {
    fs::create_dir(root)?;
    fs::create_dir(root.join("data"))?;
    let data = vec![
        Some(generation()),
        Some(dense(false)),
        Some(generation()),
        Some(dense(false)),
        Some(dense(true)),
        None,
    ];
    let mut entries = vec![];
    for (i, data) in data.iter().enumerate() {
        let name = format!("control-{i}");
        let (artifact, summary, refusal) = if let Some(data) = data {
            let file = io::json(&root.join(format!("data/{name}.json")), data, guard)?;
            (
                Some(probes::storage::Artifact {
                    bytes: file.bytes as usize,
                    sha256: file.sha256,
                }),
                Some(probe(data.summary())?),
                None,
            )
        } else {
            (None, None, Some("synthetic unavailable source".into()))
        };
        entries.push(Entry {
            source: json!({"case":{"name":name},"history":{"final_test":{"synthetic":true}}}),
            artifact,
            summary,
            refusal,
        });
    }
    let inventory = Inventory {
        schema: 1,
        producer: "0".repeat(40),
        model_updates: 0,
        model_measurements: 0,
        complete: false,
        cases: entries,
    };
    let inventory_artifact = io::json(&root.join("data/inventory.json"), &inventory, guard)?;
    let header = Header {
        schema: 1,
        producer: "0".repeat(40),
        data_inventory: inventory_artifact,
        required_modes: MODES,
        process_bytes: resources::PROCESS,
        total_ram_bytes: resources::RAM,
        file_max: resources::FILE_MAX,
        run_max: resources::RUN_MAX,
        per_run_forecast: resources::RUN_MAX,
    };
    io::json(&root.join("run.json"), &header, guard)?;
    let inputs = inputs::Inputs { header, inventory };
    assert!(inputs::load(root).is_err()); // Public original loader never accepts these fixtures.
    assert!(seal::collect(root, &inputs).is_err());
    let mut stages = vec![];
    for (entry, data) in inputs.inventory.cases.iter().zip(&data) {
        let name = inputs::id(entry)?;
        let stage = stage::build(root, name, data.as_ref(), entry.refusal.as_deref(), guard)?;
        io::json(&root.join(name).join("stage.json"), &stage, guard)?;
        stages.push(stage);
    }
    Ok(Fixture {
        inputs,
        data,
        stages,
    })
}
fn finish(root: &Path, fixture: &Fixture, guard: &Guard) -> Result<BTreeMap<String, io::Artifact>> {
    assert!(seal::require(root, &fixture.inputs).is_err());
    let barrier = seal::collect(root, &fixture.inputs)?;
    assert_eq!(barrier.entries.len(), 6);
    io::json(&root.join("seal.json"), &barrier, guard)?;
    let seal = seal::require(root, &fixture.inputs)?;
    for (stage, data) in fixture.stages.iter().zip(&fixture.data) {
        scoring::score_case(root, stage, data.as_ref(), seal.clone(), guard)?;
        scoring::replay_case(root, stage, data.as_ref(), &seal, guard)?;
    }
    let (files, rows) = verification::complete(root, &fixture.inputs, guard)?;
    assert_eq!(rows.len(), 6);
    assert_eq!(rows[4]["model_refusal"], "empty complete training support");
    assert_eq!(rows[5]["source_refusal"], "synthetic unavailable source");
    for row in &rows[..4] {
        assert_eq!(row["modes"].as_array().unwrap().len(), 2);
        assert!(row["raw_replayed_calls"].as_u64().unwrap() > 0);
    }
    Ok(files)
}
#[test]
fn complete_two_mode_original_repeat_and_raw_replay_match() -> Result<()> {
    let temp = Temp::new()?;
    let guard = temp.guard()?;
    let a = temp.0.join("original");
    let b = temp.0.join("repeat");
    let fa = fixture(&a, &guard)?;
    let first = finish(&a, &fa, &guard)?;
    let fb = fixture(&b, &guard)?;
    let second = finish(&b, &fb, &guard)?;
    assert_eq!(first, second);
    let stage = &fa.stages[1];
    assert_eq!(stage.endpoints[0].steps.len(), 0);
    assert_eq!(stage.endpoints[1].steps.len(), 2);
    assert_eq!(stage.endpoints[1].steps[1].end_record_exclusive, 130);
    assert!(stage.endpoints[1].final_ce < stage.endpoints[0].final_ce);
    let scored: schema::Scored = io::read(&a.join("control-0/scores/scores.json"))?;
    for mode in scored.modes {
        for split in mode.splits {
            assert!(split.score.raw99 && split.score.exact99 && split.score.original_success);
            assert_eq!(split.score.mse, Some(0.));
            assert_eq!(split.score.generated, split.split != "train");
        }
    }
    Ok(())
}
#[test]
fn heldout_labels_teachers_and_task_metadata_cannot_change_training() -> Result<()> {
    let temp = Temp::new()?;
    let guard = temp.guard()?;
    let a = temp.0.join("a");
    let b = temp.0.join("b");
    fs::create_dir(&a)?;
    fs::create_dir(&b)?;
    let data = generation();
    let mut changed = data.clone();
    changed.name = "different task description".into();
    changed.selection = "test".into();
    for (name, split) in &mut changed.splits {
        if name == "train" {
            continue;
        }
        for row in &mut split.records {
            *row = Record::generation("heldout answer permutation", &[36, 37], &[39, 38]);
        }
        for row in split.teacher_words.as_mut().unwrap() {
            row.fill(123.5f64.to_bits());
        }
    }
    let first = stage::build(&a, "control", Some(&data), None, &guard)?;
    let second = stage::build(&b, "control", Some(&changed), None, &guard)?;
    assert_eq!(first, second);
    Ok(())
}
#[test]
fn missing_mode_changed_policy_or_refusal_cannot_seal() -> Result<()> {
    let temp = Temp::new()?;
    let guard = temp.guard()?;
    let root = temp.0.join("run");
    let fixture = fixture(&root, &guard)?;
    let mut broken = fixture.stages[0].clone();
    broken.endpoints.pop();
    fs::write(
        root.join("control-0/stage.json"),
        serde_json::to_vec(&broken)?,
    )?;
    assert!(seal::collect(&root, &fixture.inputs).is_err());
    broken = fixture.stages[0].clone();
    broken.endpoints[0].config.batch_size = 128;
    assert!(stage::validate(&broken, "control-0", None).is_err());
    broken = fixture.stages[5].clone();
    broken.refusal = None;
    assert!(stage::validate(&broken, "control-5", Some("synthetic unavailable source")).is_err());
    let mut truncated = fixture.stages[0].clone();
    truncated.family.as_mut().unwrap().partition.retained.pop();
    assert!(stage::validate(&truncated, "control-0", None).is_err());
    Ok(())
}
#[test]
fn altered_checkpoint_and_extra_artifact_are_rejected() -> Result<()> {
    let temp = Temp::new()?;
    let guard = temp.guard()?;
    let root = temp.0.join("run");
    let fixture = fixture(&root, &guard)?;
    finish(&root, &fixture, &guard)?;
    fs::write(root.join("extra.json"), b"{}")?;
    assert!(verification::complete(&root, &fixture.inputs, &guard).is_err());
    fs::remove_file(root.join("extra.json"))?;
    let path = root.join("control-0/forward.bin");
    let mut bytes = fs::read(&path)?;
    bytes[80] ^= 1;
    fs::write(path, bytes)?;
    assert!(seal::require(&root, &fixture.inputs).is_err());
    Ok(())
}
#[test]
fn missing_replay_receipt_and_changed_raw_output_are_rejected() -> Result<()> {
    let temp = Temp::new()?;
    let guard = temp.guard()?;
    let root = temp.0.join("run");
    let fixture = fixture(&root, &guard)?;
    finish(&root, &fixture, &guard)?;
    let path = root.join("control-0/replay.json");
    let saved = fs::read(&path)?;
    fs::remove_file(&path)?;
    assert!(verification::complete(&root, &fixture.inputs, &guard).is_err());
    fs::write(&path, saved)?;
    let path = root.join("control-0/scores/forward-0.raw");
    let mut bytes = fs::read(&path)?;
    bytes[48] ^= 1;
    fs::write(path, bytes)?;
    assert!(verification::complete(&root, &fixture.inputs, &guard).is_err());
    Ok(())
}
#[test]
fn verification_recomputes_scores_even_if_replay_receipt_was_rewritten() -> Result<()> {
    let temp = Temp::new()?;
    let guard = temp.guard()?;
    let root = temp.0.join("run");
    let fixture = fixture(&root, &guard)?;
    finish(&root, &fixture, &guard)?;
    let path = root.join("control-0/scores/scores.json");
    let mut score: schema::Scored = io::read(&path)?;
    score.modes[0].splits[0].score.raw.correct -= 1;
    fs::write(&path, serde_json::to_vec(&score)?)?;
    let replay_path = root.join("control-0/replay.json");
    let mut replay: schema::Replayed = io::read(&replay_path)?;
    replay.score = io::artifact(&path)?;
    fs::write(replay_path, serde_json::to_vec(&replay)?)?;
    assert!(verification::complete(&root, &fixture.inputs, &guard).is_err());
    Ok(())
}
#[test]
fn trace_checks_prefix_order_complete_words_call_count_and_finiteness() -> Result<()> {
    let temp = Temp::new()?;
    let guard = temp.guard()?;
    let path = temp.0.join("trace.raw");
    let predictor = |_: &[i64]| -> probes::Result<Vec<f64>> { Ok(vec![-0., 2., 3.]) };
    let mut writer = trace::Recorder::new(predictor, &path, 3, 2, &guard)?;
    probe(writer.logits(&[1, 2]))?;
    probe(writer.logits(&[1, 3]))?;
    assert!(writer.logits(&[1, 4]).is_err());
    let (artifact, calls) = writer.finish()?;
    assert_eq!(calls, 2);
    let mut reader = trace::Replay::new(&path, 3, 2, &artifact)?;
    assert!(reader.logits(&[1, 3]).is_err());
    let mut reader = trace::Replay::new(&path, 3, 2, &artifact)?;
    let first = probe(reader.logits(&[1, 2]))?;
    assert_eq!(first[0].to_bits(), (-0f64).to_bits());
    assert!(reader.finish().is_err());
    assert!(trace::Replay::new(&path, 3, 1, &artifact).is_err());
    let mut bytes = fs::read(&path)?;
    bytes[48..56].copy_from_slice(&f64::NAN.to_bits().to_le_bytes());
    fs::write(&path, &bytes)?;
    assert!(trace::Replay::new(&path, 3, 2, &artifact).is_err());
    let actual = io::artifact(&path)?;
    assert!(
        trace::Replay::new(&path, 3, 2, &actual)?
            .logits(&[1, 2])
            .is_err()
    );
    Ok(())
}
#[test]
fn both_disks_ram_and_bounded_writes_stop_without_overwrite() -> Result<()> {
    let large = 100 * resources::RUN_MAX;
    assert!(resources::admitted(large, large, false, 0));
    assert!(!resources::admitted(0, large, false, 0));
    assert!(!resources::admitted(large, 0, false, 0));
    assert!(!resources::admitted(large, large, false, resources::RAM));
    assert!(!resources::admitted(large, large, false, u64::MAX));
    let need = 2 * resources::RUN_MAX + resources::STORAGE_RESERVE;
    assert!(!resources::admitted(need, need, true, 0));
    let temp = Temp::new()?;
    let guard = temp.guard()?;
    let path = temp.0.join("limited.bin");
    use std::io::Write;
    let mut writer = io::Writer::new(&path, 3, &guard)?;
    writer.write_all(b"abc")?;
    assert!(writer.write_all(b"d").is_err());
    writer.finish()?;
    assert!(io::Writer::new(&path, 3, &guard).is_err());
    assert_eq!(fs::read(path)?, b"abc");
    assert!(forecast::case(&generation())? < resources::RUN_MAX);
    assert!(inputs::owned(&dense(false)) >= 16 * 260);
    Ok(())
}
