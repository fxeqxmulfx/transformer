//! Closed reviews and exact parent bytes are prerequisites for integration.
use crate::*;
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Review {
    pub summary: PathBuf,
    pub exit: PathBuf,
    pub commit: PathBuf,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Parent {
    pub root: PathBuf,
    pub summary_sha256: String,
    pub inventory_sha256: String,
    pub review: Review,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub producer_commit: String,
    pub suite_sha256: String,
    pub parents: BTreeMap<String, Parent>,
}
fn commit(s: &str) -> bool {
    s.len() == 40 && s.bytes().all(|b| b.is_ascii_hexdigit())
}
pub(crate) fn load(request: &Request, suite: &Value) -> Result<Loaded> {
    check(
        commit(&request.producer_commit),
        "comparison producer must be a commit",
    )?;
    let p = protocol()?;
    let expected = p["sources"].as_object().ok_or("missing frozen sources")?;
    check(
        request.parents.keys().eq(expected.keys()),
        "all four reviewed parents are required",
    )?;
    let mut loaded = Loaded::default();
    for (name, pin) in &request.parents {
        let a = loaded
            .evidence
            .bind(&pin.root.join("summary.json"), Some(&pin.summary_sha256))?;
        loaded.snapshots.insert(format!("{name}-summary.json"), a);
        let a = loaded.evidence.bind(
            &pin.root.join("inventory.json"),
            Some(&pin.inventory_sha256),
        )?;
        loaded.snapshots.insert(format!("{name}-inventory.json"), a);
        loaded
            .evidence
            .bind(&pin.review.summary, Some(&pin.summary_sha256))?;
        loaded.evidence.bind(&pin.review.exit, None)?;
        loaded.evidence.bind(&pin.review.commit, None)?;
        check(
            fs::read_to_string(&pin.review.exit)?.trim() == "0",
            "numerical review is unfinished or failed",
        )?;
        check(
            commit(fs::read_to_string(&pin.review.commit)?.trim()),
            "numerical review lacks a commit",
        )?;
        let inventory: Value = read(&pin.root.join("inventory.json"))?;
        let summary: Value = read(&pin.root.join("summary.json"))?;
        for key in ["source_fingerprint", "producer_commit"] {
            check(
                inventory[key] == expected[name][key],
                "parent source or producer differs from freeze",
            )?;
        }
        check(
            summary["source_fingerprint"] == inventory["source_fingerprint"],
            "summary source differs from inventory",
        )?;
        match name.as_str() {
            "consequences" => {
                consequences::load(&pin.root, &inventory, &summary, suite, &mut loaded)?
            }
            "depth_original" | "depth_retained" => depth::load(
                &pin.root,
                &inventory,
                &summary,
                suite,
                name == "depth_retained",
                &mut loaded,
            )?,
            "transformer" => {
                transformer::load(&pin.root, &inventory, &summary, suite, &mut loaded)?
            }
            _ => return Err("unknown parent".into()),
        }
    }
    for offer in &loaded.retained {
        let old = loaded
            .prior
            .iter()
            .find(|r| r.task == offer.task && r.mode == offer.mode)
            .ok_or("retained task lacks prior choice")?;
        check(
            old.primary == offer.primary,
            "primary split differs between compared models",
        )?;
    }
    for transformer in &loaded.transformers {
        let endpoint = &transformer["endpoint"];
        let old = loaded
            .prior
            .iter()
            .find(|r| r.task == endpoint["case"]["task"])
            .ok_or("transformer task lacks prior choice")?;
        check(
            endpoint["primary_split"] == old.primary,
            "transformer primary split differs from original convex task",
        )?;
    }
    Ok(loaded)
}
pub fn prepare(destination: &Path, request: &Request) -> Result<()> {
    check(
        request.suite_sha256 == hash(SUITE.as_bytes()),
        "original suite bytes changed",
    )?;
    prepare_with(destination, request, &serde_json::from_str(SUITE)?)
}
pub(crate) fn prepare_with(destination: &Path, request: &Request, suite: &Value) -> Result<()> {
    check(
        !destination.exists(),
        "integration destination already exists",
    )?;
    let loaded = load(request, suite)?;
    let selection = selection::select(suite, &loaded.prior, &loaded.retained)?;
    fs::create_dir(destination)?;
    for (name, item) in &loaded.snapshots {
        artifacts::snapshot(item, &destination.join(name))?;
    }
    write(&destination.join("request.json"), request)?;
    write(&destination.join("selection.json"), &selection)?;
    let report = report::build(suite, &loaded, &selection)?;
    write(&destination.join("summary.json"), &report)?;
    write(
        &destination.join("manifest.json"),
        &json!({
            "source_fingerprint":fingerprint(),"binary_sha256":digest(&std::env::current_exe()?)?,"producer_commit":request.producer_commit,"protocol":protocol()?,
            "request_sha256":digest(&destination.join("request.json"))?,
            "selection_sha256":digest(&destination.join("selection.json"))?,
            "summary_sha256":digest(&destination.join("summary.json"))?,
            "complete_parent_snapshots":loaded.snapshots,"artifacts":loaded.evidence.0.values().collect::<Vec<_>>(),
            "new_gradient_updates":0,"new_prediction_evaluations":0,"reviewed_metrics_only":true
        }),
    )?;
    Ok(())
}
pub fn replay(root: &Path) -> Result<Value> {
    let request: Request = read(&root.join("request.json"))?;
    check(
        request.suite_sha256 == hash(SUITE.as_bytes()),
        "original suite bytes changed",
    )?;
    replay_with(root, &request, &serde_json::from_str(SUITE)?)
}
pub(crate) fn replay_with(root: &Path, request: &Request, suite: &Value) -> Result<Value> {
    let manifest: Value = read(&root.join("manifest.json"))?;
    check(
        manifest["source_fingerprint"] == fingerprint()
            && manifest["binary_sha256"] == digest(&std::env::current_exe()?)?
            && manifest["protocol"] == protocol()?
            && manifest["producer_commit"] == request.producer_commit
            && manifest["request_sha256"] == digest(&root.join("request.json"))?
            && manifest["selection_sha256"] == digest(&root.join("selection.json"))?
            && manifest["summary_sha256"] == digest(&root.join("summary.json"))?,
        "comparison source, request or sealed outputs changed",
    )?;
    let artifacts: Vec<Artifact> = serde_json::from_value(manifest["artifacts"].clone())?;
    for item in &artifacts {
        artifacts::verify(item)?;
    }
    let loaded = load(request, suite)?;
    check(
        serde_json::to_value(loaded.evidence.0.values().collect::<Vec<_>>())?
            == manifest["artifacts"],
        "artifact inventory changed",
    )?;
    check(
        serde_json::to_value(&loaded.snapshots)? == manifest["complete_parent_snapshots"],
        "parent snapshot inventory changed",
    )?;
    for (name, item) in &loaded.snapshots {
        check(
            digest(&root.join(name))? == item.sha256,
            "complete parent snapshot changed",
        )?;
    }
    let selection = selection::select(suite, &loaded.prior, &loaded.retained)?;
    check(
        read::<Value>(&root.join("selection.json"))? == selection,
        "observed-only reference selection does not replay",
    )?;
    let summary = report::build(suite, &loaded, &selection)?;
    check(
        read::<Value>(&root.join("summary.json"))? == summary,
        "complete comparison does not replay",
    )?;
    Ok(
        json!({"replayed":true,"source_fingerprint":fingerprint(),"manifest_sha256":digest(&root.join("manifest.json"))?,
        "summary_sha256":manifest["summary_sha256"],"selection_sha256":manifest["selection_sha256"],
        "parent_summaries":4,"prior_choices":105,"depth_outcomes":336,"transformer_endpoints":35,
        "initializer_arms":2,"choices_per_arm":105,"artifact_count":artifacts.len(),
        "fresh_prediction_recomputation":false,"full_training_rerun":false}),
    )
}
