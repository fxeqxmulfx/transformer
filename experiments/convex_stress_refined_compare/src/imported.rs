//! Stream actual saved models; independent numerical checking precedes offers.
use super::*;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Attempt {
    pub source_sha256: String,
    pub report: Option<binding::Report>,
    pub refusal: Option<String>,
    pub seconds: f64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Original {
    request: quotient::binding::Request,
    report: Option<quotient::binding::Report>,
    returned_error: Option<String>,
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
    let rows: Vec<_> = stage["records"]
        .as_array()
        .ok_or("missing training records")?
        .iter()
        .filter(|r| r["task"] == task)
        .collect();
    if rows.len() != 1 {
        return Err("missing or repeated staged case".into());
    }
    let record = rows[0];
    for artifact in record["artifacts"]
        .as_array()
        .ok_or("missing staged file inventory")?
    {
        let path = Path::new(artifact["path"].as_str().ok_or("missing staged path")?);
        if digest(path)? != artifact["sha256"] || fs::metadata(path)?.len() != artifact["bytes"] {
            return Err("staged actual artifact differs from reviewed bytes".into());
        }
    }
    if record["status"] == "complete" {
        let dir = pin.root.join(name(task)?);
        let result: Value = read(&dir.join("result.json"))?;
        let input = x
            .input
            .as_ref()
            .ok_or("staged model lacks original pair input")?;
        let policy = x
            .policies
            .as_ref()
            .ok_or("staged model lacks original policy")?;
        if result != record["result"]
            || read::<quotient::Input>(&dir.join("input.json"))? != *input
            || read::<[training::Policy; 3]>(&dir.join("policies.json"))? != *policy
            || result["input_sha256"] != digest(&dir.join("input.json"))?
            || result["policies_sha256"] != digest(&dir.join("policies.json"))?
            || result["heldout_evaluation"] != false
            || result["parent_comparison"] != false
            || result["unseen_identification"] != false
        {
            return Err("staged training input, policies or scope differ".into());
        }
        let original = &result["metadata"]["input"];
        for key in [
            "task",
            "vocabulary",
            "input_sha256",
            "data_fingerprints",
            "original_batch",
            "policies",
        ] {
            if original[key] != x.record[key] {
                return Err("original model/data record differs before import".into());
            }
        }
        let rows = result["stages"]
            .as_array()
            .ok_or("missing complete staged catalogue")?;
        if result["source_refusal"].is_null() {
            if rows.len() != 55 {
                return Err("complete refined source lost a prime stage".into());
            }
            for (i, (s, p)) in rows.iter().zip(quotient::catalogue()).enumerate() {
                if s["source_stage"]["prime"] != p || s["file"] != format!("stage-{i:02}-p{p}.json")
                {
                    return Err("refined prime stage order changed".into());
                }
            }
        } else if !rows.is_empty() {
            return Err("refused source silently gained models".into());
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
    let prime = saved["source_stage"]["prime"]
        .as_u64()
        .ok_or("missing source prime")? as u16;
    let index = quotient::catalogue()
        .iter()
        .position(|&p| p == prime)
        .ok_or("unknown prime")?;
    let file = format!("stage-{index:02}-p{prime}.json");
    if saved["file"] != file {
        return Err("noncanonical staged file".into());
    }
    let path = pin.root.join(name(task)?).join(&file);
    let sha = digest(&path)?;
    if saved["sha256"] != sha || saved["bytes"] != fs::metadata(&path)?.len() {
        return Err("actual refined stage hash differs".into());
    }
    let a: Attempt = read(&path)?;
    if a.source_sha256 != saved["source_stage"]["sha256"]
        || !a.seconds.is_finite()
        || a.seconds < 0.
        || saved["retained"]["seconds"] != a.seconds
        || saved["retained"]["source_sha256"] != a.source_sha256
        || saved["retained"]["refusal"] != json!(a.refusal)
    {
        return Err("staged source, timing or refusal differs".into());
    }
    let mut eligible = false;
    match (&a.report, &a.refusal) {
        (Some(r), None) => {
            let source = record["result"]["metadata"]["source_case"]["artifacts"]
                .as_array()
                .ok_or("missing original source artifacts")?;
            // The canonical filename and pinned hash select the original report;
            // no task arithmetic or metric outcome chooses a source stage.
            let original = source
                .iter()
                .filter(|v| {
                    v["path"].as_str().is_some_and(|p| {
                        Path::new(p).file_name().is_some_and(|n| n == file.as_str())
                    })
                })
                .collect::<Vec<_>>();
            if original.len() != 1 || original[0]["sha256"] != a.source_sha256 {
                return Err("missing unique original discovery artifact".into());
            }
            let original_path = Path::new(original[0]["path"].as_str().unwrap());
            if digest(original_path)? != a.source_sha256 {
                return Err("actual original discovery changed".into());
            }
            let old: Original = read(original_path)?;
            let discovery = old.report.ok_or("refined model lacks original discovery")?;
            let input = x.input.as_ref().ok_or("missing original pair input")?;
            let policies = x.policies.as_ref().ok_or("missing original policies")?;
            let readout = x
                .readout
                .as_ref()
                .ok_or("missing original declared readout")?;
            if old.returned_error.is_some()
                || discovery.request != old.request
                || discovery.discovery.request != old.request.discovery
                || old.request.discovery.input != *input
                || old.request.discovery.prime != prime
                || old.request.readout.as_ref() != Some(readout)
                || r.policies != *policies
                || r.input_sha256 != hash(&serde_json::to_vec(input)?)
                || r.discovery_sha256 != hash(&serde_json::to_vec(&discovery.discovery)?)
            {
                return Err(
                    "refined observed data, actual discovery or mode policy differs".into(),
                );
            }
            eligible = outcomes::compatible(&discovery.discovery, readout);
            if discovery.candidate_readout_compatible != Some(eligible) {
                return Err("candidate declared-readout eligibility changed".into());
            }
            if deep {
                binding::check(&discovery.discovery, policies, r)?;
            }
            outcomes::offers(task, prime, &sha, r, eligible)?;
        }
        (None, Some(reason)) if !reason.is_empty() => {}
        _ => return Err("stage must retain one report or refusal".into()),
    }
    Ok(Stage {
        attempt: a,
        prime,
        sha,
        eligible,
    })
}
