use crate::{
    Result,
    dataset::{Dataset, Split},
    scoring,
    wire::Record,
};
use std::collections::BTreeMap;

fn fixture() -> Dataset {
    let row = Record::generation("fixture-only", &[51, 52], &[52, 53]);
    let shard = Split {
        records: vec![row],
        teacher_words: Some(vec![vec![0.01f64.to_bits(), 0.24f64.to_bits()]]),
    };
    Dataset {
        name: "synthetic-scoring-control".into(),
        vocabulary: 69,
        selection: "validation".into(),
        numeric_step: Some(0.25),
        numeric_offset: Some(16),
        splits: BTreeMap::from([
            ("train".into(), shard.clone()),
            ("validation".into(), shard),
        ]),
    }
}
fn peak(token: usize, classes: usize) -> Vec<f64> {
    let mut logits = vec![0.; classes];
    logits[token] = 2.;
    logits
}
#[test]
fn source_extract_retains_six_original_protocols_and_teachers() -> Result<()> {
    let cases = crate::cases()?;
    assert_eq!(cases.len(), 6);
    for case in &cases[..4] {
        assert_eq!(case["case"]["metric"], "negative_mse");
        assert_eq!(case["case"]["target"], -0.001);
        assert_eq!(
            case["history"]["design"]["numeric_grid"]["unquantized_teacher_sha256"]
                .as_object()
                .unwrap()
                .len(),
            3
        );
    }
    for case in &cases[4..] {
        assert_eq!(case["case"]["target"], 0.99);
    }
    Ok(())
}
#[test]
fn ordinary_ce_and_unquantized_mse_are_separate() -> Result<()> {
    let data = fixture();
    let mut calls = 0;
    let mut model = |prefix: &[i64]| -> Result<Vec<f64>> {
        calls += 1;
        Ok(peak([52, 53, 17][prefix.len() - 4], 69))
    };
    let measured = scoring::score(&data, "validation", &mut model)?;
    assert_eq!(calls, 6);
    assert_eq!(measured.raw.correct, 3);
    assert_eq!(measured.raw.exact, 1);
    assert!(measured.raw99 && measured.exact99 && measured.original_success);
    let expected_ce = (1. + 68. * (-2f64).exp()).ln();
    assert!((measured.cross_entropy - expected_ce).abs() < 1e-14);
    assert!((measured.mse.unwrap() - 0.0001).abs() < 1e-15);
    assert!(measured.mse.unwrap() > 0.); // Exact codec labels are not the real teacher.
    assert_eq!(measured.valid_numeric_components, 2);
    Ok(())
}
#[test]
fn early_eos_and_invalid_components_keep_every_denominator() -> Result<()> {
    let data = fixture();
    for token in [17, 0, 35] {
        let mut model = move |_: &[i64]| -> Result<Vec<f64>> { Ok(peak(token, 69)) };
        let measured = scoring::score(&data, "validation", &mut model)?;
        assert_eq!(measured.raw.target_count, 3);
        assert_eq!(measured.numeric_components, 2);
        assert_eq!(measured.valid_numeric_components, 0);
        assert_eq!(measured.mse, Some(64.));
        assert!(!measured.raw99 && !measured.exact99 && !measured.original_success);
        assert_eq!(measured.generated_tokens, if token == 17 { 1 } else { 3 });
    }
    Ok(())
}
#[test]
fn generation_reads_its_own_error_and_training_keeps_the_tail() -> Result<()> {
    let mut data = fixture();
    let mut prefixes = vec![];
    let mut model = |prefix: &[i64]| -> Result<Vec<f64>> {
        prefixes.push(prefix.to_vec());
        Ok(peak(54, 69))
    };
    let measured = scoring::score(&data, "validation", &mut model)?;
    assert_eq!(prefixes[4], vec![1, 51, 52, 18, 54]);
    assert_eq!(prefixes[5], vec![1, 51, 52, 18, 54, 54]);
    assert_eq!(measured.raw.exact, 0);
    let row = data.splits["train"].records[0].clone();
    data.splits.get_mut("train").unwrap().records = vec![row; 130];
    let train = data.training()?;
    assert_eq!(
        train
            .rows
            .chunks(127)
            .map(|batch| batch.len())
            .collect::<Vec<_>>(),
        [127, 3]
    );
    assert_eq!(
        train
            .rows
            .iter()
            .map(|row| row.original_index)
            .collect::<Vec<_>>(),
        (0..130).collect::<Vec<_>>()
    );
    assert_eq!(train.rows[129].targets, [-100, -100, -100, 52, 53, 17]);
    Ok(())
}
#[test]
fn dense_scoring_keeps_only_original_supervised_positions_and_tie_order() -> Result<()> {
    let mut data = fixture();
    data.numeric_step = None;
    data.numeric_offset = None;
    data.vocabulary = 70;
    data.splits.insert(
        "validation".into(),
        Split {
            records: vec![Record {
                task: "order-fixture".into(),
                tokens: vec![36, 37],
                targets: vec![-100, 68],
                changes: vec![false, false],
                prompt: vec![],
                answer: vec![],
                generation_limit: 0,
            }],
            teacher_words: None,
        },
    );
    let mut calls = 0;
    let mut model = |prefix: &[i64]| -> Result<Vec<f64>> {
        assert_eq!(prefix, [36, 37]);
        calls += 1;
        let mut scores = peak(68, 70);
        scores[69] = 2.;
        Ok(scores)
    };
    let result = scoring::score(&data, "validation", &mut model)?;
    assert_eq!(calls, 1);
    assert_eq!(result.raw.target_count, 1);
    assert_eq!(result.raw.transition_count, 0);
    assert_eq!(result.raw.correct, 1);
    assert_eq!(result.mse, None);
    assert!(!result.generated && result.original_success);
    let mut broken = |_: &[i64]| -> Result<Vec<f64>> { Ok(vec![f64::NAN; 70]) };
    assert!(scoring::score(&data, "validation", &mut broken).is_err());
    Ok(())
}
#[test]
fn masks_teachers_multiplicity_and_eos_are_bound_by_independent_hashes() -> Result<()> {
    let data = fixture();
    let rows = &data.splits["train"].records;
    let original = crate::wire::examples_hash(rows)?;
    let mut bad = rows.clone();
    bad[0].targets[0] = 36;
    assert_ne!(original, crate::wire::examples_hash(&bad)?);
    assert!(bad[0].validate(69).is_err());
    let mut bad = rows.clone();
    bad[0].answer.pop();
    assert!(bad[0].validate(69).is_err());
    let mut bad = rows.clone();
    bad.push(bad[0].clone());
    assert_ne!(original, crate::wire::examples_hash(&bad)?);
    let teacher = &data.splits["train"].teacher_words.clone().unwrap();
    let mut bad = teacher.clone();
    bad[0][0] ^= 1;
    assert_ne!(
        crate::wire::teacher_hash(teacher)?,
        crate::wire::teacher_hash(&bad)?
    );
    Ok(())
}

#[test]
fn both_disks_and_complete_ram_reserve_are_required() {
    use crate::storage::{DISK_RESERVE, GROWTH_RESERVE, RAM_LIMIT, admitted};
    assert!(admitted(
        DISK_RESERVE,
        DISK_RESERVE,
        RAM_LIMIT - GROWTH_RESERVE
    ));
    assert!(!admitted(DISK_RESERVE - 1, u64::MAX, 0));
    assert!(!admitted(u64::MAX, DISK_RESERVE - 1, 0));
    assert!(!admitted(
        u64::MAX,
        u64::MAX,
        RAM_LIMIT - GROWTH_RESERVE + 1
    ));
    assert!(!admitted(u64::MAX, u64::MAX, u64::MAX));
}

#[test]
fn native_transport_refuses_truncated_extra_and_oversized_payloads() -> Result<()> {
    let mut bytes = b"GPTPRB01".to_vec();
    for word in [1u64, 1, 1, 1, 69, 52, 53, 0.24f64.to_bits()] {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    let value = crate::native::parse(&bytes)?;
    assert_eq!(value.teacher, Some(vec![vec![0.24f64.to_bits()]]));
    for n in [0, 7, 8, 47, bytes.len() - 1] {
        assert!(crate::native::parse(&bytes[..n]).is_err());
    }
    let mut bad = bytes.clone();
    bad.push(0);
    assert!(crate::native::parse(&bad).is_err());
    let mut bad = bytes.clone();
    bad[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(crate::native::parse(&bad).is_err());
    let mut bad = bytes.clone();
    bad[48..56].copy_from_slice(&69u64.to_le_bytes());
    assert!(crate::native::parse(&bad).is_err());
    Ok(())
}

#[test]
fn refused_inventory_cannot_drop_a_case_or_claim_complete() -> Result<()> {
    use crate::inventory::{Entry, Inventory};
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "probe-data-inventory-{}-{stamp}",
        std::process::id()
    ));
    std::fs::create_dir(&path)?;
    let mut inventory = Inventory {
        schema: 1,
        producer: "0".repeat(40),
        model_updates: 0,
        model_measurements: 0,
        complete: false,
        cases: crate::cases()?
            .into_iter()
            .map(|source| Entry {
                source,
                artifact: None,
                summary: None,
                refusal: Some("synthetic refusal control".into()),
            })
            .collect(),
    };
    let write = |value: &Inventory| -> Result<()> {
        std::fs::write(path.join("inventory.json"), serde_json::to_vec(value)?)?;
        Ok(())
    };
    write(&inventory)?;
    assert!(!crate::inventory::read(&path)?.complete);
    inventory.complete = true;
    write(&inventory)?;
    assert!(crate::inventory::read(&path).is_err());
    inventory.complete = false;
    inventory.cases[0].source["case"]["target"] = serde_json::json!(0);
    write(&inventory)?;
    assert!(crate::inventory::read(&path).is_err());
    inventory.cases.pop();
    write(&inventory)?;
    assert!(crate::inventory::read(&path).is_err());
    std::fs::remove_dir_all(path)?;
    Ok(())
}
