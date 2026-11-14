//! Stream immutable CM34 states. Original-basis and complex-head checks precede
//! offers; refusals remain explicit and no scalar source stage is required.
use super::*;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Attempt {
    pub discovery: Option<discovery::Report>,
    pub discovery_error: Option<String>,
    pub head: Option<binding::Report>,
    pub head_error: Option<String>,
    pub discovery_seconds: f64,
    pub head_seconds: f64,
}
pub(crate) struct Stage {
    pub attempt: Attempt,
    pub prime: u16,
    pub sha: String,
    pub eligible: bool,
}
pub(crate) fn case(
    pin: &Reference,
    stage: &Value,
    task: &str,
    x: &inputs::Inputs,
) -> Result<Value> {
    let planned: Vec<String> = serde_json::from_value(stage["planned"].clone())?;
    if !planned.iter().any(|t| t == task) {
        return Ok(json!({"task":task,"status":"not_planned"}));
    }
    let original = stage["inputs"]
        .as_array()
        .ok_or("missing original kernel inputs")?
        .iter()
        .filter(|v| v["task"] == task)
        .collect::<Vec<_>>();
    if original.len() != 1 {
        return Err("missing unique original kernel input".into());
    }
    for key in [
        "task",
        "vocabulary",
        "input_sha256",
        "data_fingerprints",
        "original_batch",
        "policies",
    ] {
        if original[0][key] != x.record[key] {
            return Err("original data differs before kernel import".into());
        }
    }
    let rows = stage["records"]
        .as_array()
        .ok_or("missing training records")?
        .iter()
        .filter(|r| r["task"] == task)
        .collect::<Vec<_>>();
    if rows.len() != 1 {
        return Err("missing unique kernel training outcome".into());
    }
    let record = rows[0];
    for artifact in record["artifacts"]
        .as_array()
        .ok_or("missing staged artifact inventory")?
    {
        let path = Path::new(artifact["path"].as_str().ok_or("missing staged path")?);
        if digest(path)? != artifact["sha256"] || fs::metadata(path)?.len() != artifact["bytes"] {
            return Err("staged artifact differs from reviewed bytes".into());
        }
    }
    if record["status"] == "complete" {
        let dir = pin.root.join(name(task)?);
        let result: Value = read(&dir.join("result.json"))?;
        let input = x
            .input
            .as_ref()
            .ok_or("kernel model lacks original pair input")?;
        let policies = x
            .policies
            .as_ref()
            .ok_or("kernel model lacks original policy")?;
        if result != record["result"]
            || read::<discovery::Input>(&dir.join("input.json"))? != *input
            || read::<[training::Policy; 3]>(&dir.join("policies.json"))? != *policies
            || result["input_sha256"] != digest(&dir.join("input.json"))?
            || result["policies_sha256"] != digest(&dir.join("policies.json"))?
            || result["metadata"]["input"] != *original[0]
            || result["heldout_evaluation"] != false
            || result["parent_comparison"] != false
            || result["raw_label_identification"] != false
            || result["target_readout_compatibility_proved"] != false
        {
            return Err("kernel training input, policies or scope differ".into());
        }
        let rows = result["stages"]
            .as_array()
            .ok_or("missing complete kernel catalogue")?;
        if rows.len() != 55 {
            return Err("kernel catalogue lost a prime stage".into());
        }
        for (i, (s, p)) in rows.iter().zip(catalogue()).enumerate() {
            if s["prime"] != p || s["file"] != format!("stage-{i:02}-p{p}.json") {
                return Err("kernel prime stage order changed".into());
            }
        }
    }
    Ok(record.clone())
}
pub(crate) fn load(
    pin: &Reference,
    task: &str,
    record: &Value,
    saved: &Value,
    x: &inputs::Inputs,
    deep: bool,
) -> Result<Stage> {
    let raw_prime = saved["prime"].as_u64().ok_or("missing kernel prime")?;
    let prime = u16::try_from(raw_prime)?;
    let index = catalogue()
        .iter()
        .position(|&p| p == prime)
        .ok_or("unknown prime")?;
    let file = format!("stage-{index:02}-p{prime}.json");
    if saved["file"] != file || record["result"]["stages"][index] != *saved {
        return Err("noncanonical or unbound kernel stage".into());
    }
    let path = pin.root.join(name(task)?).join(&file);
    let sha = digest(&path)?;
    if saved["sha256"] != sha || saved["bytes"] != fs::metadata(&path)?.len() {
        return Err("actual kernel stage hash differs".into());
    }
    let a: Attempt = read(&path)?;
    if [a.discovery_seconds, a.head_seconds]
        .iter()
        .any(|t| !t.is_finite() || *t < 0.)
        || saved["retained"]["discovery_seconds"] != a.discovery_seconds
        || saved["retained"]["head_seconds"] != a.head_seconds
        || saved["retained"]["discovery_error"] != json!(a.discovery_error)
        || saved["retained"]["head_error"] != json!(a.head_error)
    {
        return Err("kernel durations or errors differ".into());
    }
    let mut eligible = false;
    match (&a.discovery, &a.discovery_error) {
        (Some(d), None) => {
            let input = x.input.as_ref().ok_or("missing original pair input")?;
            let policies = x.policies.as_ref().ok_or("missing original policies")?;
            let readout = x
                .readout
                .as_ref()
                .ok_or("missing original declared readout")?;
            if d.request
                != (discovery::Request {
                    input: input.clone(),
                    prime,
                })
            {
                return Err("kernel discovery input or prime differs".into());
            }
            if deep {
                discovery::check(&d.request, d)?;
            }
            eligible = outcomes::compatible(d, readout);
            match (&a.head, &a.head_error) {
                (Some(h), None) => {
                    if h.policies != *policies
                        || h.input_sha256 != hash(&serde_json::to_vec(input)?)
                        || h.discovery_sha256 != hash(&serde_json::to_vec(d)?)
                    {
                        return Err("kernel model is not bound to its actual discovery".into());
                    }
                    if deep {
                        binding::check(d, policies, h)?;
                    }
                    outcomes::offers(task, prime, &sha, h, eligible)?;
                }
                (None, Some(reason)) if !reason.is_empty() => {}
                _ => return Err("kernel stage needs one head outcome".into()),
            }
        }
        (None, Some(reason))
            if !reason.is_empty()
                && a.head.is_none()
                && a.head_error.is_none()
                && a.head_seconds == 0. => {}
        _ => return Err("kernel stage needs one discovery outcome".into()),
    }
    Ok(Stage {
        attempt: a,
        prime,
        sha,
        eligible,
    })
}
