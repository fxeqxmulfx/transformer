//! Compare the revised budget against an independently frozen, sealed endpoint.
//! Every serialized field must agree except the protocol digest naming the cap.
use convex_stress_margin_budget::{Model, Result, digest, parent};
use serde_json::{Value, json};
use std::{fs::File, io::BufReader};

fn main() -> Result<()> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: trajectory FROZEN_MODEL")?;
    let mut envelope: Value = serde_json::from_reader(BufReader::new(File::open(path)?))?;
    for role in ["prior", "supplement"] {
        let mut old = envelope
            .as_object_mut()
            .ok_or("missing envelope")?
            .remove(role)
            .ok_or("missing role")?;
        if old.is_null() {
            continue;
        }
        let binding = serde_json::from_value(old["parent"]["binding"].clone())?;
        let source = parent::Model::load(old["parent"].clone(), &binding)?;
        let pin = digest(&source)?;
        let new = Model::new(source, &pin)?;
        let actual = serde_json::to_value(&new)?;
        let old_protocol = old["binding"]["protocol_sha256"].clone();
        old["binding"]["protocol_sha256"] = actual["binding"]["protocol_sha256"].clone();
        if serde_json::to_vec(&old)? != serde_json::to_vec(&actual)? {
            return Err(format!("numerical trajectory or other state changed in {role}").into());
        }
        println!(
            "{}",
            json!({"role":role,"all_state_equal_except_protocol_digest":true,
            "old_protocol_sha256":old_protocol,"new_protocol_sha256":actual["binding"]["protocol_sha256"],
            "selection_sha256":digest(new.selection())?,"certificate":new.certificate()})
        );
    }
    Ok(())
}
