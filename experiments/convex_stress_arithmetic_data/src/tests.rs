use crate::*;
use serde_json::{Value, json};

#[test]
fn all54_source_cases_keep_exact_historical_fingerprints_or_explicit_refusals() -> Result<()> {
    let mut supported = 0;
    let mut refused = vec![];
    for row in inventory::cases()? {
        if let Some(d) = dataset::build(&row)? {
            supported += 1;
            d.validate()?;
            let restored: dataset::Dataset = serde_json::from_slice(&serde_json::to_vec(&d)?)?;
            assert_eq!(d, restored);
        } else {
            refused.push(
                row["case"]["data"]["operation"]
                    .as_str()
                    .unwrap()
                    .to_string(),
            );
        }
    }
    assert_eq!(supported, 48);
    refused.sort();
    let mut expected = oracle::UNAVAILABLE.map(str::to_string);
    expected.sort();
    assert_eq!(refused, expected);
    Ok(())
}
#[test]
fn recovered_registry_matches_both_original_tensors_on_all_six_cases() -> Result<()> {
    let mut recovered = 0;
    for row in inventory::cases()? {
        let full = dataset::build_catalog(&row)?;
        if let Some(frozen) = dataset::build(&row)? {
            assert_eq!(full, frozen);
        } else {
            recovered += 1;
            assert_eq!(full.training.rows.len(), 4704);
            assert_eq!(full.validation.len(), 4705);
            for split in ["train", "validation"] {
                let mut wrong = row.clone();
                wrong["history"]["data_fingerprints"][split] = json!("different fixed table");
                assert!(dataset::build_catalog(&wrong).is_err());
            }
        }
    }
    assert_eq!(recovered, 6);
    Ok(())
}

#[test]
fn all22_existing_cases_match_independent_old_oracle_and_masks() -> Result<()> {
    let previous: Value = serde_json::from_str(ORIGINAL35)?;
    let cases = inventory::cases()?;
    let mut count = 0;
    for old in previous["tasks"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|r| r["case"]["kind"] == "arithmetic")
    {
        let legacy = reference::arithmetic(old)?;
        let row = cases
            .iter()
            .find(|r| r["case"]["name"] == old["name"])
            .unwrap();
        let d = dataset::build(row)?.unwrap();
        assert_eq!(d.training.rows, legacy.train);
        assert_eq!(d.validation, legacy.validation);
        assert_eq!(d.training.vocabulary, legacy.vocab);
        count += 1;
    }
    assert_eq!(count, 22);
    let row = [[2u32, 7, 3]];
    let mut bytes = b"torch.int64:(1, 3, 2):".to_vec();
    for x in [2i64, 7, -100, 3, 0, 0] {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    assert_eq!(reference::fingerprint(&row), hash(&bytes));
    Ok(())
}
#[test]
fn reject_changed_sources_labels_ranges_overlap_and_unsupported_oracles() -> Result<()> {
    let row = inventory::cases()?.remove(0);
    for field in ["data_seed", "target"] {
        let mut changed = row.clone();
        changed["case"][field] = json!(2);
        assert!(dataset::build(&changed).is_err());
    }
    let mut changed = row.clone();
    changed["history"]["data_fingerprints"]["train"] = json!("changed");
    assert!(dataset::build(&changed).is_err());
    let mut d = dataset::build(&row)?.unwrap();
    d.validation[0] = d.training.rows[0];
    assert!(d.validate().is_err());
    let mut d = dataset::build(&row)?.unwrap();
    d.training.rows[0][2] = 193;
    assert!(d.validate().is_err());
    for spec in [
        json!({"prime":97,"operation":"not-a-source-task"}),
        json!({"prime":15,"operation":"add"}),
        json!({"prime":97,"operation":"power-division","power":7}),
    ] {
        assert!(oracle::table(&spec).is_err());
    }
    Ok(())
}
#[test]
fn original_constants_and_integer_not_modular_semantics_stay_explicit() -> Result<()> {
    let at = |op: &str, a, b| -> Result<u32> {
        let rows = oracle::table(&json!({"prime":97,"operation":op}))?.unwrap();
        Ok(rows.iter().find(|r| r[0] == a && r[1] == b).unwrap()[2])
    };
    assert_eq!(at("floor-product", 96, 96)?, 95);
    assert_eq!(at("max", 90, 30)?, 90);
    assert_eq!(at("abs-diff/blind", 0, 95)?, 47);
    assert_eq!(at("sum-interval", 47, 0)?, 1);
    assert_eq!(at("sum-interval", 48, 0)?, 0);
    assert_eq!(at("sum-squares", 0, 0)?, 0);
    assert_eq!(at("mixed", 50, 0)?, 50);
    for p in [17, 97, 193] {
        let rows =
            oracle::table(&json!({"prime":p,"operation":"power-division","power":0}))?.unwrap();
        assert_eq!(rows.len(), (p * (p - 1)) as usize);
        assert!(rows.iter().all(|r| r[1] != 0 && r[2] == 1));
    }
    Ok(())
}

#[test]
fn full_recount_rejects_saved_labels_even_after_their_hash_is_replaced() -> Result<()> {
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };
    let root = std::env::temp_dir().join(format!(
        "arithmetic-data-{}-{}",
        std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
    ));
    fs::create_dir(&root)?;
    let saved = inventory::generate(&root, "fixture", "fixture", true)?;
    assert_eq!(
        saved,
        inventory::generate(&root, "fixture", "fixture", false)?
    );
    assert_eq!(saved["supported_additional_cases"], 26);
    assert_eq!(saved["refused_cases"], 6);
    assert_eq!(saved["new_accuracy_claims"], 0);
    let file = root.join("data/arithmetic-div.json");
    let mut data: Value = serde_json::from_reader(fs::File::open(&file)?)?;
    data["training"]["rows"][0][2] =
        json!((data["training"]["rows"][0][2].as_u64().unwrap() + 1) % 97);
    let mut bytes = serde_json::to_vec(&data)?;
    bytes.push(b'\n');
    fs::write(file, &bytes)?;
    let mut forged = saved;
    forged["cases"][0]["data"]["sha256"] = json!(hash(&bytes));
    fs::write(root.join("inventory.json"), serde_json::to_vec(&forged)?)?;
    assert!(inventory::generate(&root, "fixture", "fixture", false).is_err());
    fs::remove_dir_all(root)?;
    Ok(())
}
