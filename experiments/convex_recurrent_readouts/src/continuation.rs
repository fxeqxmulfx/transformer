//! Exact training-prefix validation before extending the original refit budget.
//! No old score or held-out label enters the check; only training receipts/model bytes.
use crate::{Result, experiment};
use bilinear::experiment as original;
use rotary::model::Model;
use serde_json::{Value, json};
use std::{fs, path::Path};

pub struct Replay {
    pub identity: Value,
    records: Vec<Value>,
    endpoint: Vec<u8>,
}
impl Replay {
    pub fn load(
        storage: &Path,
        protocol: &Value,
        variant: &str,
        mode: &str,
        input: &Value,
    ) -> Result<Option<Self>> {
        let Some(reference) = protocol.get("reference_prefix") else {
            return Ok(None);
        };
        let root = storage.join(
            reference["relative_cohort"]
                .as_str()
                .ok_or("reference cohort")?,
        );
        let path = experiment::cell(&root, variant, mode);
        let meta = original::score::read(&path.join("metadata.json"))?;
        let receipt = original::score::read(&path.join("training_receipt.json"))?;
        let endpoint = fs::read(path.join("endpoint.json"))?;
        let old_protocol: Value = serde_json::from_str(experiment::PROTOCOL)?;
        let records: Vec<Value> = fs::read_to_string(path.join("training.jsonl"))?
            .lines()
            .map(serde_json::from_str)
            .collect::<std::result::Result<_, _>>()?;
        if reference["updates"] != old_protocol["updates"]
            || meta["protocol"] != old_protocol
            || meta["source_fingerprint"] != reference["source_fingerprint"]
            || receipt["source_fingerprint"] != reference["source_fingerprint"]
            || meta["input"] != *input
            || meta["variant"] != variant
            || meta["mode"] != mode
            || receipt["variant"] != variant
            || receipt["mode"] != mode
            || receipt["status"] != "sealed"
            || receipt["updates"] != reference["updates"]
            || receipt["endpoint_sha256"] != original::hash(&endpoint)
            || records.len() != reference["updates"].as_u64().ok_or("reference updates")? as usize
        {
            return Err("invalid immutable training-prefix reference".into());
        }
        Ok(Some(Self {
            identity: json!({"source_fingerprint":reference["source_fingerprint"],
            "updates":records.len(),"endpoint_sha256":original::hash(&endpoint)}),
            records,
            endpoint,
        }))
    }
    pub fn observe(&self, update: usize, record: &Value, model: &Model, path: &Path) -> Result<()> {
        let Some(expected) = self.records.get(update) else {
            return Ok(());
        };
        for key in [
            "update",
            "examples_seen",
            "targets_seen",
            "batch_indices_sha256",
            "receipt",
        ] {
            if record[key] != expected[key] {
                return Err(format!(
                    "earlier training prefix differs at update{}: {key}",
                    update + 1
                )
                .into());
            }
        }
        if update + 1 == self.records.len() {
            let mut bytes = serde_json::to_vec(model)?;
            bytes.push(b'\n');
            if bytes != self.endpoint {
                return Err("earlier training checkpoint is not byte-identical".into());
            }
            original::write_json(&path.join("prefix_endpoint.json"), model)?;
            original::write_json(
                &path.join("prefix_replay.json"),
                &json!({"status":"byte-identical",
                "reference":self.identity,"deterministic_receipts_matched":self.records.len(),
                "heldout_labels_used":false}),
            )?;
        }
        Ok(())
    }
}
pub fn verify_seal(path: &Path, protocol: &Value, meta: &Value) -> Result<()> {
    let Some(reference) = protocol.get("reference_prefix") else {
        return Ok(());
    };
    let proof = original::score::read(&path.join("prefix_replay.json"))?;
    if proof["status"] != "byte-identical"
        || proof["reference"] != meta["prefix_reference"]
        || proof["deterministic_receipts_matched"] != reference["updates"]
        || proof["reference"]["updates"] != reference["updates"]
        || proof["heldout_labels_used"] != false
        || proof["reference"]["source_fingerprint"] != reference["source_fingerprint"]
        || proof["reference"]["endpoint_sha256"]
            != original::hash(&fs::read(path.join("prefix_endpoint.json"))?)
    {
        return Err("earlier training prefix seal failed".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rotary::atom::Family;
    #[test]
    fn prefix_check_refuses_changed_receipts_and_checkpoint_bytes() {
        let model = Model::zero(3, 16., Family::Rotary).unwrap();
        let records = vec![
            json!({"update":1,"receipt":{"eta":0.2}}),
            json!({"update":2,"receipt":{"eta":0.3}}),
        ];
        let mut endpoint = serde_json::to_vec(&model).unwrap();
        endpoint.push(b'\n');
        let identity = json!({"source_fingerprint":"test","updates":2,"endpoint_sha256":original::hash(&endpoint)});
        let reference = Replay {
            identity: identity.clone(),
            records: records.clone(),
            endpoint,
        };
        let path = std::env::temp_dir().join(format!("refit-prefix-{}", std::process::id()));
        fs::create_dir(&path).unwrap();
        reference.observe(0, &records[0], &model, &path).unwrap();
        assert!(!path.join("prefix_replay.json").exists());
        let mut changed = records[0].clone();
        changed["receipt"]["eta"] = json!(0.4);
        assert!(reference.observe(0, &changed, &model, &path).is_err());
        let mut other = model.clone();
        other.budget = 8.;
        other.weights[0] = 8.;
        assert!(reference.observe(1, &records[1], &other, &path).is_err());
        reference.observe(1, &records[1], &model, &path).unwrap();
        let protocol = json!({"reference_prefix":{"source_fingerprint":"test","updates":2}});
        let meta = json!({"prefix_reference":identity});
        verify_seal(&path, &protocol, &meta).unwrap();
        fs::write(path.join("prefix_endpoint.json"), b"changed").unwrap();
        assert!(verify_seal(&path, &protocol, &meta).is_err());
        fs::remove_dir_all(path).unwrap();
    }
}
