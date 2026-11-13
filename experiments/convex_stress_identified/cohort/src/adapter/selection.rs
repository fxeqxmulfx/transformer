use super::*;
use serde_json::json;

/// All primary/split scores stay outside this function's decision. Prefer a
/// verified fixed-view certificate only within a1e-12 training-CE tie; preserve
/// earlier order otherwise. This is a declared structural inductive assumption.
pub fn select(parent: &search::Report, certificates: &Report, choices: &[Value]) -> Result<Value> {
    if certificates.entries.len() != parent.candidates.len() {
        return Err("selection certificate bank extent differs".into());
    }
    let mut references = vec![];
    for old in choices {
        let mut candidates = vec![];
        if !old["selected"].is_null() {
            let verified = certificates.entries.iter().any(|e| {
                e.certificate.certified && old["selected"]["model_sha256"] == e.model_sha256
            });
            candidates.push(json!({"id":"parent","train_ce":old["selected"]["train_ce"],
                "model_sha256":old["selected"]["model_sha256"],"certified_within_fixed_view":verified}));
        }
        for (index, (c, e)) in parent
            .candidates
            .iter()
            .zip(&certificates.entries)
            .enumerate()
        {
            if e.candidate != index || e.model_sha256 != hash(&serde_json::to_vec(&c.model)?) {
                return Err("selection model/certificate binding differs".into());
            }
            candidates.push(json!({"id":format!("identified-{index}"),"train_ce":c.audit["train_ce"],
                "model_sha256":e.model_sha256,"certified_within_fixed_view":e.certificate.certified}));
        }
        let (mut selected, mut best, mut identified) = (Value::Null, f64::INFINITY, false);
        for c in &candidates {
            let ce = c["train_ce"]
                .as_f64()
                .filter(|x| x.is_finite())
                .ok_or("missing finite ordinary training CE")?;
            let certified = c["certified_within_fixed_view"] == true;
            if ce + 1e-12 < best || ((ce - best).abs() <= 1e-12 && certified && !identified) {
                selected = c["id"].clone();
                best = ce;
                identified = certified;
            }
        }
        references.push(
            json!({"mode":old["training_mode"],"selected":selected,"candidates":candidates,
            "selection":"observed_ce_then_fixed_view_certificate_then_earlier",
            "unconditional_ground_truth_identification":false,"gradient_training_run":false}),
        );
    }
    Ok(Value::Array(references))
}
