//! Construct the fixed CFt4/CFt5 candidates before a comparison parent exists.
//! Only observed labels enter the numerical routine. No selection or held-out
//! metric is produced here; the original comparison gates remain separate.
use super::*;
mod import;
mod run;
mod summary;
pub use import::Reference;
pub(crate) use import::{attach, check_attachment, imported, retained_import};
#[cfg(test)]
pub(crate) use run::{record, replay_with, run_with};
pub use run::{replay_case, run_case};
pub(crate) use run::{retained, seal};
pub use summary::summarize;
pub(crate) use summary::{case_record, summarize_with};
const STAGING_PROTOCOL: &str = include_str!("../staging_protocol.json");

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub producer_commit: String,
    pub suite_sha256: String,
}
fn stage_protocol() -> Result<Value> {
    Ok(serde_json::from_str(STAGING_PROTOCOL)?)
}
pub fn prepare(root: &Path, request: Request) -> Result<()> {
    if request.suite_sha256 != hash(SUITE.as_bytes()) {
        return Err("original staging suite bytes differ".into());
    }
    let suite = suite()?;
    let inputs = tasks(&suite)?
        .iter()
        .map(|n| Ok(inputs::build(&suite, n)?.record))
        .collect::<Result<_>>()?;
    prepare_with(root, request, &suite, inputs)
}
fn manifest(request: &Request, suite: &Value, inputs: &[Value]) -> Result<Value> {
    if !commit(&request.producer_commit) || request.suite_sha256 != hash(SUITE.as_bytes()) {
        return Err("staging needs a committed producer and original suite".into());
    }
    inventory::records(suite, inputs)?;
    Ok(
        json!({"phase":"training_only","source_fingerprint":fingerprint(),
        "binary_sha256":digest(&std::env::current_exe()?)?,"producer_commit":request.producer_commit,
        "request":request,"model_protocol":protocol()?,"staging_protocol":stage_protocol()?,
        "suite_value_sha256":hash(&serde_json::to_vec(suite)?),"tasks":tasks(suite)?,"inputs":inputs,
        "parent_comparison":false,"heldout_evaluation":false}),
    )
}
pub(crate) fn prepare_with(
    root: &Path,
    request: Request,
    suite: &Value,
    inputs: Vec<Value>,
) -> Result<()> {
    if root.exists() {
        return Err("fresh staging destination required".into());
    }
    let inv = manifest(&request, suite, &inputs)?;
    fs::create_dir(root)?;
    write(&root.join("request.json"), &request)?;
    write(&root.join("inventory.json"), &inv)
}
pub(crate) fn inventory_with(root: &Path, suite: &Value) -> Result<Value> {
    let request: Request = read(&root.join("request.json"))?;
    let inv: Value = read(&root.join("inventory.json"))?;
    let inputs = inv["inputs"].as_array().ok_or("missing staging inputs")?;
    if inv != manifest(&request, suite, inputs)? {
        return Err("staging inventory, binary, source or protocol differs".into());
    }
    Ok(inv)
}
pub(crate) fn inventory(root: &Path) -> Result<Value> {
    inventory_with(root, &suite()?)
}
