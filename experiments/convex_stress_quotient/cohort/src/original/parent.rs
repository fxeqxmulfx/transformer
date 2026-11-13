use super::*;
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Review {
    pub exit: PathBuf,
    pub commit: PathBuf,
    pub replay: PathBuf,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Parent {
    pub root: PathBuf,
    pub binary: PathBuf,
    pub summary_sha256: String,
    pub manifest_sha256: String,
    pub replay_sha256: String,
    pub review: Review,
}
const BASE_FILES: [&str; 5] = [
    "summary.json",
    "manifest.json",
    "request.json",
    "selection.json",
    "replay.json",
];
fn local_file(file: &str) -> bool {
    let mut parts = Path::new(file).components();
    matches!(parts.next(), Some(std::path::Component::Normal(_)))
        && parts.next().is_none()
        && !file.starts_with('.')
}
fn artifact(v: &Value, deep: bool) -> Result<()> {
    let path = Path::new(v["path"].as_str().ok_or("parent artifact has no path")?);
    let expected = v["sha256"].as_str().ok_or("parent artifact has no hash")?;
    let bytes = v["bytes"].as_u64().ok_or("parent artifact has no size")?;
    if !path.is_absolute()
        || expected.len() != 64
        || !expected.bytes().all(|b| b.is_ascii_hexdigit())
        || bytes > 16 * 1024 * 1024 * 1024
    {
        return Err("invalid bounded parent artifact".into());
    }
    if deep
        && (fs::metadata(path)?.len() != bytes
            || artifacts::digest_limit(path, 16 * 1024 * 1024 * 1024)? != expected)
    {
        return Err(format!("parent artifact changed: {}", path.display()).into());
    }
    Ok(())
}
pub(super) fn load(pin: &Parent, suite: &Value, deep: bool) -> Result<Value> {
    let p = protocol()?;
    for (file, expected) in [
        ("summary.json", &pin.summary_sha256),
        ("manifest.json", &pin.manifest_sha256),
        ("replay.json", &pin.replay_sha256),
    ] {
        if digest(&pin.root.join(file))? != *expected {
            return Err("complete parent bytes differ from freeze".into());
        }
    }
    if fs::read_to_string(&pin.review.exit)?.trim() != "0"
        || !commit(fs::read_to_string(&pin.review.commit)?.trim())
        || digest(&pin.review.replay)? != pin.replay_sha256
    {
        return Err("complete independent parent review required".into());
    }
    let manifest: Value = read(&pin.root.join("manifest.json"))?;
    let replay: Value = read(&pin.root.join("replay.json"))?;
    let parent: Value = read(&pin.root.join("summary.json"))?;
    if manifest["source_fingerprint"] != p["parent_source_fingerprint"]
        || manifest["producer_commit"] != p["parent_producer_commit"]
        || manifest["binary_sha256"] != digest(&pin.binary)?
        || manifest["summary_sha256"] != pin.summary_sha256
        || manifest["request_sha256"] != digest(&pin.root.join("request.json"))?
        || manifest["selection_sha256"] != digest(&pin.root.join("selection.json"))?
        || parent["source_fingerprint"] != manifest["source_fingerprint"]
        || replay["source_fingerprint"] != manifest["source_fingerprint"]
        || replay["manifest_sha256"] != pin.manifest_sha256
        || replay["summary_sha256"] != pin.summary_sha256
        || replay["replayed"] != true
        || replay["initializer_arms"] != 2
        || replay["choices_per_arm"] != 105
        || replay["depth_outcomes"] != 336
        || replay["transformer_endpoints"] != 35
    {
        return Err("reviewed parent source, binary, coverage or replay seal differs".into());
    }
    let snapshots = manifest["complete_parent_snapshots"]
        .as_object()
        .ok_or("missing complete parent snapshots")?;
    if snapshots.len() != 8
        || parent["complete_parent_snapshots"] != manifest["complete_parent_snapshots"]
    {
        return Err("parent snapshot scope differs".into());
    }
    for (file, item) in snapshots {
        if !local_file(file) || digest(&pin.root.join(file))? != item["sha256"] {
            return Err("complete ancestor snapshot changed".into());
        }
        artifact(item, deep)?;
    }
    let mut paths = BTreeSet::new();
    for item in manifest["artifacts"]
        .as_array()
        .ok_or("missing parent artifact inventory")?
    {
        artifact(item, deep)?;
        if !paths.insert(item["path"].as_str().ok_or("missing artifact path")?) {
            return Err("duplicate parent artifact".into());
        }
    }
    comparison::select(&parent, &[], &tasks(suite)?)?;
    Ok(parent)
}
pub(super) fn snapshot(pin: &Parent, destination: &Path) -> Result<Value> {
    fs::create_dir(destination)?;
    let manifest: Value = read(&pin.root.join("manifest.json"))?;
    let mut files: Vec<_> = BASE_FILES.iter().map(|s| s.to_string()).collect();
    files.extend(
        manifest["complete_parent_snapshots"]
            .as_object()
            .ok_or("missing snapshots")?
            .keys()
            .cloned(),
    );
    let mut records = vec![];
    for file in files {
        if !local_file(&file) {
            return Err("nonlocal parent snapshot".into());
        }
        let source = pin.root.join(&file);
        let expected = digest(&source)?;
        fs::copy(&source, destination.join(&file))?;
        if digest(&destination.join(&file))? != expected {
            return Err("parent changed while copying".into());
        }
        records.push(json!({"file":file,"sha256":expected,"bytes":fs::metadata(&source)?.len()}));
    }
    Ok(json!(records))
}
pub(super) fn check_snapshot(root: &Path, records: &Value) -> Result<()> {
    let mut files = BTreeSet::new();
    for item in records
        .as_array()
        .ok_or("missing parent snapshot records")?
    {
        let file = item["file"].as_str().ok_or("missing snapshot file")?;
        if !local_file(file)
            || !files.insert(file)
            || digest(&root.join(file))? != item["sha256"]
            || fs::metadata(root.join(file))?.len()
                != item["bytes"].as_u64().ok_or("missing size")?
        {
            return Err("retained complete parent snapshot changed".into());
        }
    }
    if files.len() != 13 || !BASE_FILES.iter().all(|f| files.contains(f)) {
        return Err("parent snapshot coverage changed".into());
    }
    Ok(())
}
