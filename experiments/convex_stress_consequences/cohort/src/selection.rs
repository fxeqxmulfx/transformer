use super::*;
fn parent_certified(old: &Value) -> bool {
    if old["selected"]["certified_within_fixed_view"] == true {
        return true;
    }
    let seal = &old["sealed_reference"];
    seal["candidates"].as_array().is_some_and(|rows| {
        rows.iter().any(|c| {
            c["id"] == seal["selected"]
                && c["model_sha256"] == old["selected"]["model_sha256"]
                && c["certified_within_fixed_view"] == true
        })
    })
}
/// The immutable parent supplies prior certificates. New certificates can
/// recognize an exactly matching model. No held-out score participates.
pub(super) fn select(bank: &Bank, parents: &[Value]) -> Result<Value> {
    let mut refs = vec![];
    for old in parents {
        let mut candidates = vec![];
        if !old["selected"].is_null() {
            let verified = parent_certified(old)
                || bank.candidates.iter().any(|c| {
                    c.value.scope.full_table_certified_within_family
                        && old["selected"]["model_sha256"] == c.model_sha256
                });
            candidates.push(json!({"id":"parent","train_ce":old["selected"]["train_ce"],
                "model_sha256":old["selected"]["model_sha256"],"certified_within_fixed_view":verified}));
        }
        for (index, c) in bank.candidates.iter().enumerate() {
            if hash(&serde_json::to_vec(&c.value.model)?) != c.model_sha256 {
                return Err("selection model hash differs".into());
            }
            candidates.push(json!({"id":format!("consequence-{index}"),"train_ce":c.value.raw_audit["train_ce"],
                "model_sha256":c.model_sha256,"certified_within_fixed_view":c.value.scope.full_table_certified_within_family,
                "view":c.view,"candidate":c.candidate,"duplicate_of":c.duplicate_of}));
        }
        let (mut selected, mut best, mut identified) = (Value::Null, f64::INFINITY, false);
        for c in &candidates {
            let ce = c["train_ce"]
                .as_f64()
                .filter(|x| x.is_finite())
                .ok_or("missing finite observed CE")?;
            let certified = c["certified_within_fixed_view"] == true;
            if ce < best - 1e-12 || ((ce - best).abs() <= 1e-12 && certified && !identified) {
                selected = c["id"].clone();
                best = ce;
                identified = certified;
            }
        }
        refs.push(
            json!({"mode":old["training_mode"],"selected":selected,"candidates":candidates,
            "selection":"observed_ce_then_fixed_view_certificate_then_earlier",
            "unconditional_ground_truth_identification":false,"gradient_training_run":false}),
        );
    }
    Ok(refs.into())
}
pub(super) fn seal(dir: &Path, bank: &Bank, refs: &Value) -> Result<Value> {
    Ok(
        json!({"references":refs,"input_sha256":digest(&dir.join("input.json"))?,
        "parents_sha256":digest(&dir.join("parents.json"))?,"bank_sha256":digest(&dir.join("bank.json"))?,
        "views":bank.views.iter().map(|r| &r.artifact).collect::<Vec<_>>()}),
    )
}
