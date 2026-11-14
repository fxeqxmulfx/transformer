//! Bind original CM40 records and replay the selected actual shared-role stage.
use super::*;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Attempt {
    discovery: Option<shared::discovery::Report>,
    discovery_error: Option<String>,
    head: Option<shared::binding::Report>,
    head_error: Option<String>,
    discovery_seconds: f64,
    head_seconds: f64,
}
pub(super) struct Stage {
    pub discovery: Option<shared::discovery::Report>,
    pub report: Option<shared::binding::Report>,
    pub records: Vec<shared::records::Record>,
    pub evidence: Value,
}
pub(super) fn load(
    pin: &Reference,
    summary: &Value,
    task: &str,
    prime: u16,
    x: &inputs::Inputs,
) -> Result<Stage> {
    let rows = summary["records"]
        .as_array()
        .ok_or("missing shared cases")?
        .iter()
        .filter(|r| r["task"] == task)
        .collect::<Vec<_>>();
    if rows.len() != 1 {
        return Err("shared training case missing or repeated".into());
    }
    let row = rows[0];
    parents::artifacts(&row["artifacts"])?;
    if row["status"] != "complete" {
        return Ok(Stage {
            discovery: None,
            report: None,
            records: vec![],
            evidence: json!({"status":"unavailable","reason":"shared training process failed","original_case":row}),
        });
    }
    let input = x
        .input
        .as_ref()
        .ok_or("shared head needs original paired input")?;
    let dir = pin.root.join(name(task)?);
    let records = x
        .task
        .splits
        .get("train")
        .ok_or("missing original records")?
        .iter()
        .enumerate()
        .map(|(id, e)| shared::records::Record {
            id,
            cell: e.tokens[1] as usize + input.n * e.tokens[0] as usize,
            label: e.targets[1] as usize,
        })
        .collect::<Vec<_>>();
    shared::records::validate(input, &records)?;
    let batch = x.record["original_batch"]
        .as_u64()
        .ok_or("original batch omitted")? as usize;
    let policies = shared::training::policies(records.len(), batch)?;
    let q = x
        .readout
        .as_ref()
        .ok_or("declared original readout omitted")?;
    let answers = q.iter().collect::<BTreeSet<_>>().len();
    let partition = json!({"classes":q,"answers":answers});
    if read::<discovery::Input>(&dir.join("input.json"))? != *input
        || read::<Vec<shared::records::Record>>(&dir.join("records.json"))? != records
        || read::<Value>(&dir.join("readout.json"))? != partition
        || read::<[shared::training::Policy; 3]>(&dir.join("policies.json"))? != policies
        || read::<Value>(&dir.join("result.json"))? != row["result"]
    {
        return Err("shared original data, records, partition or batch differs".into());
    }
    let index = catalogue()
        .iter()
        .position(|&p| p == prime)
        .ok_or("unknown shared prime")?;
    let ledger = row["result"]["stages"]
        .as_array()
        .ok_or("shared catalogue omitted")?;
    if ledger.len() != 55 {
        return Err("shared catalogue scope changed".into());
    }
    let file = format!("stage-{index:02}-p{prime}.json");
    let saved = &ledger[index];
    let path = dir.join(&file);
    if saved["prime"] != prime
        || saved["file"] != file
        || saved["sha256"] != digest(&path)?
        || saved["bytes"] != fs::metadata(&path)?.len()
    {
        return Err("shared selected stage binding differs".into());
    }
    let a: Attempt = read(&path)?;
    if [a.discovery_seconds, a.head_seconds]
        .iter()
        .any(|x| !x.is_finite() || *x < 0.)
        || saved["retained"]["discovery_seconds"] != a.discovery_seconds
        || saved["retained"]["head_seconds"] != a.head_seconds
        || saved["retained"]["discovery_error"] != json!(a.discovery_error)
        || saved["retained"]["head_error"] != json!(a.head_error)
    {
        return Err("shared retained outcome or time differs".into());
    }
    let expected = shared::discovery::Request {
        input: input.clone(),
        readout: q.clone(),
        answers,
        prime,
    };
    match (&a.discovery, &a.discovery_error, &a.head, &a.head_error) {
        (Some(d), None, Some(h), None) => {
            shared::discovery::replay(&expected, d)?;
            shared::binding::replay(d, &records, &policies, h)?;
        }
        (Some(d), None, None, Some(reason)) => {
            shared::discovery::replay(&expected, d)?;
            if shared::binding::run(d, &records, policies.clone())
                .err()
                .map(|e| e.to_string())
                .as_ref()
                != Some(reason)
            {
                return Err("shared head refusal does not replay".into());
            }
        }
        (None, Some(reason), None, None) => {
            if shared::discovery::run(expected)
                .err()
                .map(|e| e.to_string())
                .as_ref()
                != Some(reason)
            {
                return Err("shared discovery refusal does not replay".into());
            }
        }
        _ => return Err("contradictory shared stage outcome".into()),
    }
    Ok(Stage {
        discovery: a.discovery,
        report: a.head,
        records,
        evidence: json!({"status":"replayed","stage":parents::evidence(&path)?,"retained":saved["retained"],
            "original_records_replayed":true,"complete_augmented_basis_replayed":true,"all_modes_replayed":true}),
    })
}
