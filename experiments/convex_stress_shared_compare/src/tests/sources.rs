use super::*;
#[test]
fn exact_review_bindings_prevent_substitution_even_after_receipt_rehashing() -> Result<()> {
    let temp = Temp::new()?;
    let root = &temp.0;
    let all = tasks(&suite()?)?;
    let source = "7".repeat(64);
    let producer = "8".repeat(40);
    let reviewed = "9".repeat(40);
    fs::write(
        root.join("binary"),
        "synthetic archived binary, never executed",
    )?;
    fs::write(root.join("review.exit"), "0\n")?;
    fs::write(root.join("review.commit"), format!("{reviewed}\n"))?;
    let mut summary = parent(&all);
    summary["source_fingerprint"] = json!(source);
    summary["producer_commit"] = json!(producer);
    summary["tasks"] = json!(35);
    summary["protocol"] = json!({"synthetic":true});
    summary["records"] = json!(
        all.iter()
            .map(|t| json!({"task":t,"status":"complete","artifacts":[]}))
            .collect::<Vec<_>>()
    );
    let inv = json!({"source_fingerprint":source,"producer_commit":producer,"binary_sha256":digest(&root.join("binary"))?,
      "tasks":all,"request":{"suite_sha256":hash(SUITE.as_bytes())},"protocol":summary["protocol"]});
    write(&root.join("inventory.json"), &inv)?;
    write(&root.join("summary.json"), &summary)?;
    write(&root.join("review.json"), &summary)?;
    let mut pin = reference(root.clone());
    pin.summary_sha256 = digest(&root.join("summary.json"))?;
    pin.inventory_sha256 = digest(&root.join("inventory.json"))?;
    pin.review.exit_sha256 = digest(&pin.review.exit)?;
    pin.review.commit_sha256 = digest(&pin.review.commit)?;
    let expected = json!({"source_fingerprint":source,"producer_commit":producer,"binary_sha256":digest(&pin.binary)?,
      "summary_sha256":pin.summary_sha256,"inventory_sha256":pin.inventory_sha256,"reviewed_commit":reviewed});
    parents::load(&pin, "parent", &expected, &all, &[])?;
    fs::write(&pin.review.commit, format!("{}\n", "a".repeat(40)))?;
    pin.review.commit_sha256 = digest(&pin.review.commit)?;
    assert!(parents::load(&pin, "parent", &expected, &all, &[]).is_err());
    fs::write(&pin.review.commit, format!("{reviewed}\n"))?;
    pin.review.commit_sha256 = digest(&pin.review.commit)?;
    fs::write(&pin.review.exit, "1\n")?;
    pin.review.exit_sha256 = digest(&pin.review.exit)?;
    assert!(parents::load(&pin, "parent", &expected, &all, &[]).is_err());
    fs::write(&pin.review.exit, "0\n")?;
    pin.review.exit_sha256 = digest(&pin.review.exit)?;
    fs::write(&pin.binary, "changed archived executable")?;
    assert!(parents::load(&pin, "parent", &expected, &all, &[]).is_err());
    Ok(())
}
#[test]
fn artifact_and_checkpoint_paths_remain_bound_to_saved_bytes() -> Result<()> {
    let temp = Temp::new()?;
    let path = temp.0.join("model.json");
    fs::write(&path, "original")?;
    let records = json!([parents::evidence(&path)?]);
    parents::artifacts(&records)?;
    fs::write(&path, "changed")?;
    assert!(parents::artifacts(&records).is_err());
    assert!(parents::checkpoint_path(&temp.0, "../outside").is_err());
    assert!(parents::checkpoint_path(&temp.0, "/absolute").is_err());
    assert_eq!(parents::checkpoint_path(&temp.0, "model.json")?, path);
    Ok(())
}
